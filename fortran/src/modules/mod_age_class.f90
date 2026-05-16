! =============================================================================
! mod_age_class.f90  --  Age-Class Management Engine (ACME) state
!
! Replaces COMMON blocks /comage1/, /comage2/, /comage3/ and associated
! PARAMETERs from comage.h.
!
! The ACME engine tracks cohorts of upper-canopy vegetation by stand age.
! Arrays are indexed 0:ncohort (101 elements), where cohort 0 is bare ground
! and cohorts 1..ncohort span stands aged 1..maxage years.
!
! Original types: real*4, integer*4, integer*2 in comage.h.
! Modernised to: real(wp), integer (default kind).
!
! File I/O parameters (ac_unit, ac_file, etc.) are kept here for traceability;
! they may be moved to the I/O module in a later phase.
!
! Replaces: comage.h (COMMON /comage1/, /comage2/, /comage3/).
! =============================================================================
module mod_age_class
  use mod_precision, only: wp
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! Compile-time parameters
  ! ---------------------------------------------------------------------------
  public :: maxage, ageint, ncohort, ac_var, npftu
  public :: ac_size, ac_recl, sum_recl

  !> Maximum stand age simulated (years).
  integer, parameter :: maxage  = 100

  !> Age-class width (years).  maxage must be divisible by ageint.
  integer, parameter :: ageint  = 1

  !> Number of age cohorts = maxage / ageint.
  integer, parameter :: ncohort = maxage / ageint

  !> Number of age-class-dependent C pool variables tracked per cohort.
  !! (cbiol_ac, cbior_ac, cbiow_ac)
  integer, parameter :: ac_var  = 3

  !> Number of upper-canopy PFTs (tree PFTs only).
  !! NOTE: npftu=8 (trees) != npft=12 (all PFTs); 4 lower-canopy PFTs not tracked.
  integer, parameter :: npftu   = 8

  !> Size in REALs of a single age-class data array = npftu * (ncohort+1).
  integer, parameter :: ac_size = npftu * (ncohort + 1)

  !> Size in BYTES of a complete single grid-point age-class record.
  integer, parameter :: ac_recl   = ((ncohort + 1) + (ac_size * ac_var)) * 4

  !> Size in BYTES of a single age-class summary record.
  integer, parameter :: sum_recl  = ac_size * 4

  ! ---------------------------------------------------------------------------
  ! File I/O constants
  ! ---------------------------------------------------------------------------
  public :: ac_unit, pt_unit, ac_file, pt_file, acsum_file, lc_unit

  integer,          parameter :: ac_unit = 51  !< binary file unit for age-class data
  integer,          parameter :: pt_unit = 52  !< text file unit for age-class printouts
  integer,          parameter :: lc_unit = 44  !< unit for land-class summaries

  character(len=12), parameter :: ac_file    = 'ageclass.dat'
  character(len=12), parameter :: pt_file    = 'ageclass.txt'
  character(len=11), parameter :: acsum_file = 'ac_summ.dat'

  ! ---------------------------------------------------------------------------
  ! Land-class dimension
  ! ---------------------------------------------------------------------------
  public :: n_lcs
  integer, parameter :: n_lcs = 100  !< maximum number of land classes

  ! ---------------------------------------------------------------------------
  ! /comage1/  --  initialisation biomass and area fraction
  ! ---------------------------------------------------------------------------
  public :: fu_acmin, cbiow_init, cbiol_init, cbior_init

  !> Minimum cover fraction per cohort = 1.0 / ncohort.  Set at init.
  real(wp) :: fu_acmin = 0.0_wp

  !> Initial wood biomass per PFT (kg C m-2).  Set at model init.
  real(wp) :: cbiow_init(npftu) = 0.0_wp

  !> Initial leaf biomass per PFT (kg C m-2).  Set at model init.
  real(wp) :: cbiol_init(npftu) = 0.0_wp

  !> Initial root biomass per PFT (kg C m-2).  Set at model init.
  real(wp) :: cbior_init(npftu) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comage2/  --  per-cohort area fractions and C pools
  ! ---------------------------------------------------------------------------
  public :: f_Aveg, cbiow_ac, cbiol_ac, cbior_ac

  !> Land area fraction occupied by upper canopy in each cohort (0=bare).
  real(wp) :: f_Aveg(0:ncohort) = 0.0_wp

  !> Wood biomass C in each cohort and PFT (kg C m-2).
  real(wp) :: cbiow_ac(npftu, 0:ncohort) = 0.0_wp

  !> Leaf biomass C in each cohort and PFT (kg C m-2).
  real(wp) :: cbiol_ac(npftu, 0:ncohort) = 0.0_wp

  !> Root biomass C in each cohort and PFT (kg C m-2).
  real(wp) :: cbior_ac(npftu, 0:ncohort) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comage3/  --  land-class usage flags
  ! ---------------------------------------------------------------------------
  public :: used

  !> Flag indicating which land-class values are in use (1=active, 0=inactive).
  integer :: used(n_lcs) = 0

end module mod_age_class
