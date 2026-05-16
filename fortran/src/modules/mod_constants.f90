! =============================================================================
! mod_constants.f90  --  IBIS runtime physical constants and grid scalars
!
! Replaces COMMON blocks /compar1/ through /compar4/ from compar.h.
! Variables here are set once at model startup (by the modern equivalent of
! subroutine inicon in initial.f) and treated as read-only thereafter.
!
! Physical constant values are initialised to the exact values from the
! original F77 code (found in ibis_2.6b4/initial.f, subroutine inicon):
!   stef  = 5.67e-8       (W m-2 K-4)
!   vonk  = 0.4
!   grav  = 9.80616       (m s-2)
!   tmelt = 273.16        (K)
!   hvap  = 2.5104e6      (J kg-1)
!   hfus  = 0.3336e6      (J kg-1)
!   hsub  = hvap + hfus   (J kg-1, derived)
!   ch2o  = 4.218e3       (J kg-1 K-1)
!   cice  = 2.106e3       (J kg-1 K-1)
!   cair  = 1.00464e3     (J kg-1 K-1)
!   cvap  = 1.81e3        (J kg-1 K-1)
!   rair  = 287.04        (J kg-1 K-1)
!   rvap  = 461.0         (J kg-1 K-1)
!   cappa = rair/cair     (dimensionless, derived)
!   rhow  = 1.0e3         (kg m-3)
!   epsilon = 1.0e-7      (dimensionless)
!
! Runtime parameters (set by main / I/O):
!   dtime     -- model timestep (s); read from ibis.infile
!   ndaypy    -- days per year (365 or 366, set each year in main)
!   nlonsub   -- longitude subset count (set by I/O)
!   nlatsub   -- latitude  subset count (set by I/O)
!   ndaypm    -- days per month (set each year in main)
!   garea     -- grid-cell area (m2); computed from grid geometry in I/O
!   vzero     -- array of zeros; legacy placeholder (candidate for removal)
!
! Replaces: compar.h (COMMON /compar1/ through /compar4/).
! =============================================================================
module mod_constants
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi
  implicit none
  private

  public :: constants_init
  public :: pi
  public :: epsilon, dtime, stef, vonk, grav, tmelt
  public :: hfus, hvap, hsub, ch2o, cice, cair, cvap
  public :: rair, rvap, cappa, rhow
  public :: garea, vzero
  public :: ndaypy, nlonsub, nlatsub, ndaypm

  ! ---------------------------------------------------------------------------
  ! /compar1/: physical constants
  ! Initialised to F77 values by constants_init(); see header for sources.
  ! ---------------------------------------------------------------------------

  !> Ratio of circumference to diameter of a circle.
  !> Original: pi = 3.1415927  (compar.h:88)
  real(wp), parameter :: pi = 3.1415927_wp

  !> Small positive number to guard against zero-divides (~machine epsilon).
  !> Original: epsilon = 1.0e-7  (initial.f:101)
  real(wp) :: epsilon = 0.0_wp

  !> Model timestep (s).  Set at runtime from ibis.infile (main.f:116).
  real(wp) :: dtime = 0.0_wp

  !> Stefan-Boltzmann constant (W m-2 K-4).
  real(wp) :: stef = 0.0_wp

  !> von Karman constant (dimensionless).
  real(wp) :: vonk = 0.0_wp

  !> Gravitational acceleration (m s-2).
  real(wp) :: grav = 0.0_wp

  !> Freezing point of water (K).
  real(wp) :: tmelt = 0.0_wp

  !> Latent heat of fusion of water (J kg-1).
  real(wp) :: hfus = 0.0_wp

  !> Latent heat of vaporisation of water (J kg-1).
  real(wp) :: hvap = 0.0_wp

  !> Latent heat of sublimation of ice (J kg-1). Derived: hvap + hfus.
  real(wp) :: hsub = 0.0_wp

  !> Specific heat of liquid water (J kg-1 K-1).
  real(wp) :: ch2o = 0.0_wp

  !> Specific heat of ice (J kg-1 K-1).
  real(wp) :: cice = 0.0_wp

  !> Specific heat of dry air at constant pressure (J kg-1 K-1).
  real(wp) :: cair = 0.0_wp

  !> Specific heat of water vapour at constant pressure (J kg-1 K-1).
  real(wp) :: cvap = 0.0_wp

  !> Gas constant for dry air (J kg-1 K-1).
  real(wp) :: rair = 0.0_wp

  !> Gas constant for water vapour (J kg-1 K-1).
  real(wp) :: rvap = 0.0_wp

  !> rair/cair (dimensionless). Derived quantity; set by constants_init.
  real(wp) :: cappa = 0.0_wp

  !> Density of liquid water (kg m-3).
  real(wp) :: rhow = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /compar2/: grid scalars
  ! ---------------------------------------------------------------------------

  !> Area of each land grid cell (m2). Set by I/O at startup.
  real(wp) :: garea(npoi) = 0.0_wp

  !> Array of zeros, length npoi. Legacy placeholder; candidate for removal.
  real(wp) :: vzero(npoi) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /compar3/ and /compar4/: calendar
  ! ---------------------------------------------------------------------------

  !> Number of days per year (365 or 366). Updated each year in main.
  integer :: ndaypy = 365

  !> Number of longitude points for domain subsetting. Set by I/O.
  integer :: nlonsub = 1

  !> Number of latitude points for domain subsetting. Set by I/O.
  integer :: nlatsub = 1

  !> Days per month. Updated each year in main to handle leap years.
  integer :: ndaypm(12) = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

contains

  ! ---------------------------------------------------------------------------
  !> Initialise physical constants to their F77 original values.
  !!
  !! Call this once at model startup, before any physics routine runs.
  !! Corresponds to the constant-assignment block in ibis_2.6b4/initial.f
  !! (subroutine inicon, lines 81-101).
  !!
  !! Derived quantities (hsub, cappa) are computed from their primaries so
  !! that internal consistency is guaranteed regardless of precision.
  ! ---------------------------------------------------------------------------
  subroutine constants_init()
    implicit none

    ! Physical constants -- values from ibis_2.6b4/initial.f
    stef    = 5.67e-8_wp       ! Stefan-Boltzmann (W m-2 K-4)
    vonk    = 0.4_wp           ! von Karman (dimensionless)
    grav    = 9.80616_wp       ! gravitational acceleration (m s-2)
    tmelt   = 273.16_wp        ! melting point of water (K)
    hvap    = 2.5104e6_wp      ! latent heat of vaporisation (J kg-1)
    hfus    = 0.3336e6_wp      ! latent heat of fusion (J kg-1)
    hsub    = hvap + hfus      ! latent heat of sublimation (J kg-1) [derived]
    ch2o    = 4.218e3_wp       ! specific heat of liquid water (J kg-1 K-1)
    cice    = 2.106e3_wp       ! specific heat of ice (J kg-1 K-1)
    cair    = 1.00464e3_wp     ! specific heat of dry air (J kg-1 K-1)
    cvap    = 1.81e3_wp        ! specific heat of water vapour (J kg-1 K-1)
    rair    = 287.04_wp        ! gas constant, dry air (J kg-1 K-1)
    rvap    = 461.0_wp         ! gas constant, water vapour (J kg-1 K-1)
    cappa   = rair / cair      ! rair/cair (dimensionless) [derived]
    rhow    = 1.0e3_wp         ! density of liquid water (kg m-3)
    epsilon = 1.0e-7_wp        ! small number to avoid zero-divides

    ! vzero is always zero; garea, dtime, ndaypy, ndaypm set elsewhere.
    vzero = 0.0_wp

  end subroutine constants_init

end module mod_constants
