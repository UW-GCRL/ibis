! =============================================================================
! mod_dimensions.f90  --  IBIS compile-time grid and state dimensions
!
! Replaces the PARAMETER statements in compar.h (the integer/real parameters
! nlon, nlat, npoi, xres, yres, nband, nsoilay, nsnolay, npft, pi).
! These are compile-time constants that size all state arrays.
!
! Active configuration is selected at compile time:
!   Default (no flag):  South Asia 0.5 x 0.5 degree  (nlon=720, nlat=360, npoi=1718)
!   -DGLOBAL:           Global    0.5 x 0.5 degree  (nlon=720, nlat=360, npoi=58920)
!
! Use the Makefile targets:
!   make ibis          -- South Asia domain binary
!   make ibis_global   -- Global domain binary (ibis_global)
!
! NOTE ON pi: the value 3.1415927_wp is preserved from the F77 original
! (compar.h: parameter(pi=3.1415927)).  It differs from the true value of
! pi at the 8th decimal place.  Do NOT "improve" this value without a full
! regression test; even a tiny change propagates into solar geometry
! calculations.
!
! Replaces: compar.h (PARAMETER section).
! =============================================================================
module mod_dimensions
  use mod_precision, only: wp
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! Grid dimensions — selected at compile time via preprocessor
  ! ---------------------------------------------------------------------------
#ifdef GLOBAL
  ! Global 0.5 x 0.5 deg:  720 x 360,  58920 land points
  integer,  parameter, public :: nlon =  720
  integer,  parameter, public :: nlat =  360
  integer,  parameter, public :: npoi = 58920
  real(wp), parameter, public :: xres = 0.50_wp
  real(wp), parameter, public :: yres = 0.50_wp
#else
  ! South Asia 0.5 x 0.5 deg (default):  720 x 360,  1718 land points
  integer,  parameter, public :: nlon =  720
  integer,  parameter, public :: nlat =  360
  integer,  parameter, public :: npoi = 1718
  real(wp), parameter, public :: xres = 0.50_wp
  real(wp), parameter, public :: yres = 0.50_wp
#endif

  ! ---------------------------------------------------------------------------
  ! Other configurations (for reference; edit #else block above to activate)
  ! ---------------------------------------------------------------------------
  ! 10 x 10 deg TEST grid:  36 x 18,   147 land points
  !   nlon=36,  nlat=18,  npoi=147,  xres=10.0, yres=10.0
  !
  ! Global 1 x 1 deg:  360 x 180,  14545 land points
  !   nlon=360, nlat=180, npoi=14545, xres=1.0, yres=1.0
  !
  ! Global 2 x 2 deg:  180 x 90,   3728  land points
  !   nlon=180, nlat=90,  npoi=3728,  xres=2.0, yres=2.0
  !
  ! Global 4 x 4 deg:  90 x 45,    912   land points
  !   nlon=90,  nlat=45,  npoi=912,   xres=4.0, yres=4.0
  ! ---------------------------------------------------------------------------

  ! ---------------------------------------------------------------------------
  ! State-space dimensions (same for all grid configurations)
  ! ---------------------------------------------------------------------------
  integer, parameter, public :: nband   = 2   !< solar radiation wavebands (VIS, NIR)
  integer, parameter, public :: nsoilay = 6   !< number of soil layers
  integer, parameter, public :: nsnolay = 3   !< number of snow layers
  integer, parameter, public :: npft    = 12  !< number of plant functional types

  ! ---------------------------------------------------------------------------
  ! Mathematical constant
  ! ---------------------------------------------------------------------------
  real(wp), parameter, public :: pi = 3.1415927_wp  !< see header note

end module mod_dimensions
