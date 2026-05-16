! =============================================================================
! mod_pft_params.f90  --  Plant Functional Type (PFT) physiological parameters
!
! Replaces COMMON blocks /compft1/, /compft2/, /compft3/, /compft4/,
! /pft_broad/, /pft_conif/, /pft_shrub/, /pft_c3grass/, /pft_c4grass/
! from compft.h.
!
! All variables are set at model startup by RD_PARAM (readpars.f) reading
! from parameter files params.can and params.veg.  Initialised to 0 here;
! physics routines must not be called before RD_PARAM runs.
!
! Array dimensions:
!   npft  = 12  (all plant functional types)
!   nband =  2  (VIS and NIR radiation bands)
!
! Note on plai_init: dimensioned (4, 15) in the original -- 4 LAI sub-values
! per vegetation type, 15 vegetation classes.  Size 15 is not a named
! parameter in compar.h but is the maximum number of vegetation classes
! supported by params.veg.
!
! Replaces: compft.h (COMMON /compft1/–/compft4/, /pft_broad/–/pft_c4grass/).
! =============================================================================
module mod_pft_params
  use mod_precision,  only: wp
  use mod_dimensions, only: npft
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /compft1/: universal photosynthesis constants
  ! ---------------------------------------------------------------------------
  public :: tau15, kc15, ko15, cimax
  public :: alpha3, theta3, beta3
  public :: alpha4, beta4,  theta4
  public :: vmax_pft

  !> CO2/O2 specificity ratio at 15 C (dimensionless).
  real(wp) :: tau15  = 0.0_wp

  !> CO2 Michaelis-Menten constant at 15 C (mol mol-1).
  real(wp) :: kc15   = 0.0_wp

  !> O2  Michaelis-Menten constant at 15 C (mol mol-1).
  real(wp) :: ko15   = 0.0_wp

  !> Maximum internal CO2 concentration for model stability (mol mol-1).
  real(wp) :: cimax  = 0.0_wp

  !> Intrinsic quantum efficiency for C3 photosynthesis (dimensionless).
  real(wp) :: alpha3 = 0.0_wp

  !> Photosynthesis coupling coefficient for C3 plants (dimensionless).
  real(wp) :: theta3 = 0.0_wp

  !> Photosynthesis coupling coefficient for C3 plants (dimensionless).
  real(wp) :: beta3  = 0.0_wp

  !> Intrinsic quantum efficiency for C4 photosynthesis (dimensionless).
  real(wp) :: alpha4 = 0.0_wp

  !> Photosynthesis coupling coefficient for C4 plants (dimensionless).
  real(wp) :: beta4  = 0.0_wp

  !> Photosynthesis coupling coefficient for C4 plants (dimensionless).
  real(wp) :: theta4 = 0.0_wp

  !> Nominal Vmax of top leaf at 15 C per PFT (mol CO2 m-2 s-1).
  !! NOTE: annotated [not used] in original compft.h; retained for completeness.
  real(wp) :: vmax_pft(npft) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! PFT-group physiological parameters (Ball-Berry stomatal conductance)
  ! /pft_broad/, /pft_conif/, /pft_shrub/, /pft_c3grass/, /pft_c4grass/
  ! ---------------------------------------------------------------------------
  public :: gammaub, coefmub, coefbub, gsubmin   ! broadleaf trees
  public :: gammauc, coefmuc, coefbuc, gsucmin   ! conifer trees
  public :: gammals, coefmls, coefbls, gslsmin   ! shrubs
  public :: gammal3, coefml3, coefbl3, gsl3min   ! C3 grasses
  public :: gammal4, coefml4, coefbl4, gsl4min   ! C4 grasses

  real(wp) :: gammaub = 0.0_wp  !< leaf respiration coefficient, broadleaf
  real(wp) :: coefmub = 0.0_wp  !< Ball-Berry 'm' coefficient, broadleaf
  real(wp) :: coefbub = 0.0_wp  !< Ball-Berry 'b' coefficient, broadleaf
  real(wp) :: gsubmin = 0.0_wp  !< minimum stomatal conductance, broadleaf

  real(wp) :: gammauc = 0.0_wp  !< leaf respiration coefficient, conifer
  real(wp) :: coefmuc = 0.0_wp  !< Ball-Berry 'm' coefficient, conifer
  real(wp) :: coefbuc = 0.0_wp  !< Ball-Berry 'b' coefficient, conifer
  real(wp) :: gsucmin = 0.0_wp  !< minimum stomatal conductance, conifer

  real(wp) :: gammals = 0.0_wp  !< leaf respiration coefficient, shrub
  real(wp) :: coefmls = 0.0_wp  !< Ball-Berry 'm' coefficient, shrub
  real(wp) :: coefbls = 0.0_wp  !< Ball-Berry 'b' coefficient, shrub
  real(wp) :: gslsmin = 0.0_wp  !< minimum stomatal conductance, shrub

  real(wp) :: gammal3 = 0.0_wp  !< leaf respiration coefficient, C3 grass
  real(wp) :: coefml3 = 0.0_wp  !< Ball-Berry 'm' coefficient, C3 grass
  real(wp) :: coefbl3 = 0.0_wp  !< Ball-Berry 'b' coefficient, C3 grass
  real(wp) :: gsl3min = 0.0_wp  !< minimum stomatal conductance, C3 grass

  real(wp) :: gammal4 = 0.0_wp  !< leaf respiration coefficient, C4 grass
  real(wp) :: coefml4 = 0.0_wp  !< Ball-Berry 'm' coefficient, C4 grass
  real(wp) :: coefbl4 = 0.0_wp  !< Ball-Berry 'b' coefficient, C4 grass
  real(wp) :: gsl4min = 0.0_wp  !< minimum stomatal conductance, C4 grass

  ! ---------------------------------------------------------------------------
  ! /compft2/: biomass turnover time constants
  ! ---------------------------------------------------------------------------
  public :: tauleaf, tauroot, tauwood, tauwood0

  !> Foliar biomass turnover time constant per PFT (years).
  real(wp) :: tauleaf(npft)  = 0.0_wp

  !> Fine-root biomass turnover time constant per PFT (years).
  real(wp) :: tauroot(npft)  = 0.0_wp

  !> Wood biomass turnover time constant per PFT (years).
  !! Computed at runtime from tauwood0 and stress factors; not read directly.
  real(wp) :: tauwood(npft)  = 0.0_wp

  !> Unstressed (normal) wood biomass turnover time constant per PFT (years).
  !! Read from params.can via RD_PARAM.
  real(wp) :: tauwood0(npft) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /compft3/: bioclimatic limits for PFT establishment
  ! ---------------------------------------------------------------------------
  public :: TminL, TminU, Twarm, GDD

  !> Absolute minimum temperature lower limit per PFT (degrees C or K as read).
  real(wp) :: TminL(npft) = 0.0_wp

  !> Absolute minimum temperature upper limit per PFT (degrees C or K as read).
  real(wp) :: TminU(npft) = 0.0_wp

  !> Temperature of warmest month per PFT (degrees C or K as read).
  real(wp) :: Twarm(npft) = 0.0_wp

  !> Minimum growing degree-days per PFT (base 5 C for trees, 0 C for grasses).
  real(wp) :: GDD(npft)   = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /compft4/: canopy structure and LAI parameters
  ! ---------------------------------------------------------------------------
  public :: plai_init, plaiupper, plailower, xminlai
  public :: sapfrac_init, chifuz, chiflz, beta1, beta2

  !> Initial total LAI per vegetation type (4 sub-values, up to 15 veg types).
  !! Indexed (sub-value, veg-type); read from params.veg.
  real(wp) :: plai_init(4, 15) = 0.0_wp

  !> Potential LAI for uniform-initialisation upper canopy.
  real(wp) :: plaiupper   = 0.0_wp

  !> Potential LAI for uniform-initialisation lower canopy.
  real(wp) :: plailower   = 0.0_wp

  !> Minimum LAI threshold for each existing PFT.
  real(wp) :: xminlai     = 0.0_wp

  !> Initial sapwood fraction applied to all woody PFTs.
  real(wp) :: sapfrac_init = 0.0_wp

  !> Leaf orientation factor for upper canopy (dimensionless).
  real(wp) :: chifuz      = 0.0_wp

  !> Leaf orientation factor for lower canopy (dimensionless).
  real(wp) :: chiflz      = 0.0_wp

  !> Jackson root distribution parameter, lower canopy (dimensionless).
  real(wp) :: beta1       = 0.0_wp

  !> Jackson root distribution parameter, upper canopy (dimensionless).
  real(wp) :: beta2       = 0.0_wp

end module mod_pft_params
