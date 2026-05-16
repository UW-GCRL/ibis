! =============================================================================
! mod_run_params.f90  --  Runtime grid and calendar parameters for IBIS
!
! Replaces the COMMON blocks /compar1/ (dtime only; the physical constants
! that also live in /compar1/ are already in mod_constants), /compar2/,
! /compar3/, and /compar4/ from compar.h.
!
! All physical constants from /compar1/ (stef, vonk, grav, tmelt, hfus,
! hvap, hsub, ch2o, cice, cair, cvap, rair, rvap, cappa, rhow, epsilon)
! are already provided by mod_constants.  Only the runtime variables that
! are NOT yet covered by any module are included here.
!
! Variables:
!   /compar1/ remainder:
!     dtime          -- model timestep (seconds)
!   /compar2/:
!     garea(npoi)    -- area of each grid cell (m^2)
!     vzero(npoi)    -- array of zeros of length npoi
!   /compar3/:
!     ndaypy         -- number of days per year (365 or 366)
!     nlonsub        -- number of longitude points in the active sub-domain
!     nlatsub        -- number of latitude points in the active sub-domain
!   /compar4/:
!     ndaypm(12)     -- number of days in each calendar month
!
! All variables initialised to 0; set at model startup by the initialisation
! routines (initial.f / readpars.f).
!
! Replaces: compar.h (/compar1/ dtime, /compar2/, /compar3/, /compar4/).
! =============================================================================
module mod_run_params
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /compar1/ -- model timestep (runtime, set by main/initial)
  ! ---------------------------------------------------------------------------
  public :: dtime

  !> Model timestep (seconds).
  real(wp) :: dtime = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /compar2/ -- per-grid-point area and a zero-filled scratch vector
  ! ---------------------------------------------------------------------------
  public :: garea, vzero

  !> Area of each land grid cell (m^2).
  real(wp) :: garea(npoi) = 0.0_wp

  !> Array of zeros of length npoi; used as a convenient zero RHS.
  real(wp) :: vzero(npoi) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /compar3/ -- calendar and sub-domain dimensions (runtime)
  ! ---------------------------------------------------------------------------
  public :: ndaypy, nlonsub, nlatsub

  !> Number of days per year (365 for non-leap; 366 for leap).
  integer :: ndaypy  = 0

  !> Number of longitude points in the active sub-domain.
  !! Equals nlon for global runs; may be smaller for regional sub-domains.
  integer :: nlonsub = 0

  !> Number of latitude points in the active sub-domain.
  !! Equals nlat for global runs; may be smaller for regional sub-domains.
  integer :: nlatsub = 0

  ! ---------------------------------------------------------------------------
  ! /compar4/ -- days per calendar month
  ! ---------------------------------------------------------------------------
  public :: ndaypm

  !> Number of days in each calendar month (Jan=1 ... Dec=12).
  integer :: ndaypm(12) = 0

end module mod_run_params
