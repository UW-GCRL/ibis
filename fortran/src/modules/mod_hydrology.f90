! =============================================================================
! mod_hydrology.f90  --  Hydrological flux diagnostics
!
! Replaces COMMON block /comhyd1/ from comhyd.h.
!
! Holds instantaneous hydrological flux terms (evaporation, transpiration,
! runoff, drainage) diagnosed at each model timestep for each land point.
! All arrays are dimensioned (npoi).  Units: kg_H2O m-2 s-1 except wtot
! which is kg_H2O (total stored water).
!
! Replaces: comhyd.h (COMMON /comhyd1/).
! =============================================================================
module mod_hydrology
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /comhyd1/  --  per-timestep hydrological fluxes and total water storage
  ! ---------------------------------------------------------------------------
  public :: ginvap, gsuvap, gtrans, gtransu, gtransl
  public :: grunof, gdrain, gadjust, wtot

  !> Total evaporation rate from all intercepted water (kg_H2O m-2 s-1).
  real(wp) :: ginvap(npoi)  = 0.0_wp

  !> Total evaporation rate from surface (snow/soil) (kg_H2O m-2 s-1).
  real(wp) :: gsuvap(npoi)  = 0.0_wp

  !> Total transpiration rate from all vegetation canopies (kg_H2O m-2 s-1).
  real(wp) :: gtrans(npoi)  = 0.0_wp

  !> Transpiration from upper canopy (kg_H2O m-2 s-1).
  real(wp) :: gtransu(npoi) = 0.0_wp

  !> Transpiration from lower canopy (kg_H2O m-2 s-1).
  real(wp) :: gtransl(npoi) = 0.0_wp

  !> Surface runoff rate (kg_H2O m-2 s-1).
  real(wp) :: grunof(npoi)  = 0.0_wp

  !> Drainage rate out of bottom of lowest soil layer (kg_H2O m-2 s-1).
  real(wp) :: gdrain(npoi)  = 0.0_wp

  !> Water flux due to adjustments in subroutine wadjust (kg_H2O m-2 s-1).
  real(wp) :: gadjust(npoi) = 0.0_wp

  !> Total water stored in snow, soil, puddles, and on vegetation (kg_H2O).
  real(wp) :: wtot(npoi)    = 0.0_wp

end module mod_hydrology
