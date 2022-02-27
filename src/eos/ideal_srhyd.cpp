//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_srhyd.cpp
//! \brief derived class that implements ideal gas EOS in special relativistic hydro
//! Conserved to primitive variable inversion using algorithm described in Appendix C
//! of Galeazzi et al., PhysRevD, 88, 064009 (2013). Equation refs are to this paper.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealSRHydro::IdealSRHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;

  // Read flags specifying which variable to use in primitives
  // if nothing set in input file, use e as default
  if (!(pin->DoesParameterExist("hydro","use_e")) &&
      !(pin->DoesParameterExist("hydro","use_t")) ) {
    eos_data.use_e = true;
    eos_data.use_t = false;
  } else {
    eos_data.use_e = pin->GetOrAddBoolean("hydro","use_e",false);
    eos_data.use_t = pin->GetOrAddBoolean("hydro","use_t",false);
  }
  if (!(eos_data.use_e) && !(eos_data.use_t)) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Both use_e and use_t set to false" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eos_data.use_e && eos_data.use_t) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Both use_e and use_t set to true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationC22()
//! \brief Inline function to compute function f(z) defined in eq. C22 of Galeazzi et al.
//! The ConsToPrim algorithms finds the root of this function f(z)=0

KOKKOS_INLINE_FUNCTION
Real EquationC22(Real z, Real &u_d, Real q, Real r, Real gm1, Real pfloor) {
  Real const w = sqrt(1.0 + z*z);         // (C15)
  Real const wd = u_d/w;                  // (C15)
  Real eps = w*q - z*r + (z*z)/(1.0 + w); // (C16)

  //NOTE: The following generalizes to ANY equation of state
  eps = fmax(pfloor/(wd*gm1), eps);                           // (C18)
  Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));   // (C1) & (C21)

  return (z - r/h); // (C22)
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables for an ideal gas in SR hydro.
//! Implementation follows Wolfgang Kastaun's algorithm described in Appendix C of
//! Galeazzi et al., PhysRevD, 88, 064009 (2013).  Roots of "master function" (eq. C22)
//! found by false position method.
//!
//! In SR hydrodynamics, the conserved variables are: (D, E - D, m^i), where
//!    D = \gamma \rho is the density in the lab frame,
//!    \gamma = (1 + u^2)^{1/2} = (1 - v^2)^{-1/2} is the Lorentz factor,
//!    u^i = \gamma v^i are the spatial components of the 4-velocity (v^i is 3-vel),
//!    \rho is the comoving/fluid frame mass density,
//!    E = \gamma^2 w - P_g is the total energy,
//!    w = \rho + [\Gamma / (\Gamma - 1)] P_g is the total enthalpy,
//!    \Gamma is the adiabatic index, P_g is the gas pressure
//!    m^i = \gamma w u^i are components of the momentum in the lab frame.
//! Note we evolve (E-D). This improves accuracy/stability in high-density regions.
//!
//! In SR hydrodynamics, the primitive variables are: (\rho, P_gas, u^i).
//! Note components of the 4-velocity (not 3-velocity) are stored in the primitive
//! variables because tests show it is better to reconstruct the 4-vel.
//!
//! This function operates over range of cells given in argument list.

void IdealSRHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real &pfloor_ = eos_data.pfloor;
  Real &dfloor_ = eos_data.dfloor;
  bool &use_e = eos_data.use_e;

  // Parameters
  int const max_iterations = 25;
  Real const tol = 1.0e-12;
  Real const v_sq_max = 1.0 - tol;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, maxit_=0;
  Kokkos::parallel_reduce("hyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sum_d, int &sum_e, int &max_iter) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    Real& u_d  = cons(m,IDN,k,j,i);
    Real& u_e  = cons(m,IEN,k,j,i);
    const Real& u_m1 = cons(m,IM1,k,j,i);
    const Real& u_m2 = cons(m,IM2,k,j,i);
    const Real& u_m3 = cons(m,IM3,k,j,i);

    Real& w_d  = prim(m,IDN,k,j,i);
    Real& w_ux = prim(m,IVX,k,j,i);
    Real& w_uy = prim(m,IVY,k,j,i);
    Real& w_uz = prim(m,IVZ,k,j,i);
    Real& w_e  = prim(m,IEN,k,j,i);

    // apply density floor, without changing momentum or energy
    if (u_d < dfloor_) {
      u_d = dfloor_;
      sum_d++;
    }

    // apply energy floor
    // Real ee_min = pfloor_/gm1;
    // u_e = (u_e > ee_min) ?  u_e : ee_min;

    // Recast all variables (eq C2)
    Real q = u_e/u_d + 1.0;  // We evolve u_e = E-D
    Real r = sqrt(SQR(u_m1) + SQR(u_m2) + SQR(u_m3))/u_d;
    Real kk = r/(1.+q);

    // Enforce lower velocity bound (eq. C13). This bound combined with a floor on
    // the value of p will guarantee "some" result of the inversion
    kk = fmin(2.* sqrt(v_sq_max)/(1.0 + v_sq_max), kk);

    // Compute bracket (C23)
    auto zm = 0.5*kk/sqrt(1.0 - 0.25*kk*kk);
    auto zp = kk/sqrt(1.0 - kk*kk);

    // Evaluate master function (eq C22) at bracket values
    Real fm = EquationC22(zm, u_d, q, r, gm1, pfloor_);
    Real fp = EquationC22(zp, u_d, q, r, gm1, pfloor_);

    // For simplicity on the GPU, find roots using the false position method
    int iterations = max_iterations;
    // If bracket within tolerances, don't bother doing any iterations
    if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
      iterations = -1;
    }
    Real z = 0.5*(zm + zp);

    {int iter;
    for (iter=0; iter < iterations; ++iter) {
      z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
      Real f = EquationC22(z, u_d, q, r, gm1, pfloor_);

      // Quit if convergence reached
      // NOTE: both z and f are of order unity
      if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )) {
        break;
      }

      // assign zm-->zp if root bracketed by [z,zp]
      if (f * fp < 0.0) {
        zm = zp;
        fm = fp;
        zp = z;
        fp = f;
      } else {  // assign zp-->z if root bracketed by [zm,z]
        fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
        zp = z;
        fp = f;
      }
    }
    max_iter = (iter > max_iter)? iter : max_iter;
    }

    // iterations ended, compute primitives from resulting value of z
    Real const w = sqrt(1.0 + z*z); // (C15)
    w_d = u_d/w;                    // (C15)

    // NOTE(@ermost): The following generalizes to ANY equation of state
    Real eps = w*q - z*r + (z*z)/(1.0 + w);           // (C16)
    Real epsmin = pfloor_/(w_d*gm1);
    if (eps < epsmin) {                               // C18
      eps = epsmin;
      sum_e++;
    }
    Real h = (1. + eps) * (1.0 + (gm1*eps)/(1.+eps)); // (C1) & (C21)
    if (use_e) {
      w_e = w_d*eps;
    } else {
      w_e = gm1*eps;  // TODO(@user):  is this the correct expression?
    }

    Real const conv = 1.0/(h*u_d); // (C26)
    w_ux = conv * u_m1;            // (C26)
    w_uy = conv * u_m2;            // (C26)
    w_uz = conv * u_m3;            // (C26)

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
    }

    // TODO(@user): error handling
    // if (false) {
    //   Real gamma_adi = gm1+1.;
    //   Real rho_eps = w_p / gm1;
    //   // TODO(@ermost): Only ideal fluid for now
    //   Real wgas = w_d + gamma_adi / gm1 *w_p;
    //   auto gamma = sqrt(1. +z*z);
    //   cons(m,IDN,k,j,i) = w_d * gamma;
    //   cons(m,IEN,k,j,i) = wgas*gamma*gamma - w_p - w_d * gamma;
    //   cons(m,IM1,k,j,i) = wgas * gamma * w_vx;
    //   cons(m,IM2,k,j,i) = wgas * gamma * w_vy;
    //   cons(m,IM3,k,j,i) = wgas * gamma * w_vz;
    // }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Max<int>(maxit_));

  // store counters
  pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
  pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
  pmy_pack->pmesh->ecounter.maxit_c2p = maxit_;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables for SR hydrodynamics. Operates
//! over range of cells given in argument list.
//! Recall in SR hydrodynamics the conserved variables are: (D, E-D, m^i),
//!                        and the primitive variables are: (\rho, P_gas, u^i).

void IdealSRHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0);
  bool &use_e = eos_data.use_e;

  par_for("srhyd_prim2cons", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real& u_d  = cons(m, IDN,k,j,i);
    Real& u_e  = cons(m, IEN,k,j,i);
    Real& u_m1 = cons(m, IM1,k,j,i);
    Real& u_m2 = cons(m, IM2,k,j,i);
    Real& u_m3 = cons(m, IM3,k,j,i);

    const Real& w_d  = prim(m, IDN,k,j,i);
    const Real& w_ux = prim(m, IVX,k,j,i);
    const Real& w_uy = prim(m, IVY,k,j,i);
    const Real& w_uz = prim(m, IVZ,k,j,i);

    Real w_p;
    if (use_e) {
      const Real& w_e  = prim(m,IEN,k,j,i);
      w_p = w_e*gm1;
    } else {
      const Real& w_t  = prim(m,ITM,k,j,i);
      w_p = w_t*w_d;
    }

    // Calculate Lorentz factor
    Real u0 = sqrt(1.0 + SQR(w_ux) + SQR(w_uy) + SQR(w_uz));
    Real wgas_u0 = (w_d + gamma_prime * w_p) * u0;

    // Set conserved quantities
    u_d  = w_d * u0;
    u_e  = wgas_u0 * u0 - w_p - u_d;  // In SR, evolve E - D
    u_m1 = wgas_u0 * w_ux;            // In SR, w_ux/y/z are 4-velocity
    u_m2 = wgas_u0 * w_uy;
    u_m3 = wgas_u0 * w_uz;

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      cons(m,n,k,j,i) = prim(m,n,k,j,i)*u_d;
    }
  });

  return;
}
