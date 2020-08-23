//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf.cpp
//  \brief Local Lax Friedrichs (LLF) Riemann solver for hydrodynamics
//
//  Computes 1D fluxes using the LLF Riemann solver, also known as Rusanov's method.
//  This flux is very diffusive, even more diffusive than HLLE, and so it is not
//  recommended for use in applications.  However, it is useful for testing, or for
//  problems where other Riemann solvers fail.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "hydro/eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "hydro/rsolver/rsolver.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// LLF constructor
  
LLF::LLF(Hydro *phyd, std::unique_ptr<ParameterInput> &pin) : RiemannSolver(phyd, pin) {

  void RSolver(const int il, const  int iu, const int dir,
    const AthenaArray<Real> &wl, const AthenaArray<Real> &wr, AthenaArray<Real> &flx);
  
}

//----------------------------------------------------------------------------------------
//! \fn void LLF::RSolver
//  \brief The LLF Riemann solver for hydrodynamics (both adiabatic and isothermal)

void LLF::RSolver(const int il, const int iu, const int ivx, const AthenaArray<Real> &wl,
                  const AthenaArray<Real> &wr, AthenaArray<Real> &flx) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5],wri[5],du[5];
  Real fl[5],fr[5],flxi[5];
  Real gm1, iso_cs;
  if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) {
    gm1 = pmy_hydro->peos->GetGamma() - 1.0;
  }
  if (pmy_hydro->hydro_eos == HydroEOS::isothermal) {
    iso_cs = pmy_hydro->peos->SoundSpeed(wli);  // wli is just "dummy argument"
  }

  for (int i=il; i<=iu; ++i) {
    //--- Step 1.  Load L/R states into local variables
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) { wli[IPR]=wl(IPR,i); }

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) { wri[IPR]=wr(IPR,i); }

    //--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)

    Real cl = pmy_hydro->peos->SoundSpeed(wli);
    Real cr = pmy_hydro->peos->SoundSpeed(wri);
    Real a  = 0.5*std::max( (std::abs(wli[IVX]) + cl), (std::abs(wri[IVX]) + cr) );

    //--- Step 3.  Compute L/R fluxes

    Real mxl = wli[IDN]*wli[IVX];
    Real mxr = wri[IDN]*wri[IVX];

    fl[IDN] = mxl;
    fr[IDN] = mxr;

    fl[IVX] = mxl*wli[IVX];
    fr[IVX] = mxr*wri[IVX];

    fl[IVY] = mxl*wli[IVY];
    fr[IVY] = mxr*wri[IVY];

    fl[IVZ] = mxl*wli[IVZ];
    fr[IVZ] = mxr*wri[IVZ];

    Real el,er;
    if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) {
      el = wli[IPR]/gm1 + 0.5*wli[IDN]*(SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ]));
      er = wri[IPR]/gm1 + 0.5*wri[IDN]*(SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ]));
      fl[IVX] += wli[IPR];
      fr[IVX] += wri[IPR];
      fl[IEN] = (el + wli[IPR])*wli[IVX];
      fr[IEN] = (er + wri[IPR])*wri[IVX];
    } else {
      fl[IVX] += (iso_cs*iso_cs)*wli[IDN];
      fr[IVX] += (iso_cs*iso_cs)*wri[IDN];
    }

    //--- Step 4.  Compute difference in L/R states dU

    du[IDN] = wri[IDN]          - wli[IDN];
    du[IVX] = wri[IDN]*wri[IVX] - wli[IDN]*wli[IVX];
    du[IVY] = wri[IDN]*wri[IVY] - wli[IDN]*wli[IVY];
    du[IVZ] = wri[IDN]*wri[IVZ] - wli[IDN]*wli[IVZ];
    if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) { du[IEN] = er - el; }

    //--- Step 5. Compute the LLF flux at interface (see Toro eq. 10.42).

    flxi[IDN] = 0.5*(fl[IDN] + fr[IDN]) - a*du[IDN];
    flxi[IVX] = 0.5*(fl[IVX] + fr[IVX]) - a*du[IVX];
    flxi[IVY] = 0.5*(fl[IVY] + fr[IVY]) - a*du[IVY];
    flxi[IVZ] = 0.5*(fl[IVZ] + fr[IVZ]) - a*du[IVZ];
    if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) {
      flxi[IEN] = 0.5*(fl[IEN] + fr[IEN]) - a*du[IEN];
    }

    //--- Step 6. Store results into 3D array of fluxes

    flx(IDN,i) = flxi[IDN];
    flx(ivx,i) = flxi[IVX];
    flx(ivy,i) = flxi[IVY];
    flx(ivz,i) = flxi[IVZ];
    if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) { flx(IEN,i) = flxi[IEN]; }
  }
  return;
}

} // namespace hydro