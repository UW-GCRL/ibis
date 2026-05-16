! =============================================================================
! mod_bgc_params.f90  --  Soil biogeochemistry rate constants and parameters
!
! Replaces COMMON blocks /combgc1/, /combgc2/, /combgc3/ from combgc.h.
!
! IMPORTANT NOTE ON ORIGINAL DESIGN (anomaly):
! combgc.h declared COMMON blocks but NO subroutine in the F77 codebase ever
! wrote to them.  Instead, biogeochem.f (subroutine soilbgc) declared the
! exact same variables as LOCAL variables and hardcoded their values inline.
! combgc.h was apparently intended for external parameter loading but was
! never connected to the biogeochemistry routines.
!
! Modernisation resolution:
!   - The COMMON blocks are dropped.
!   - All parameter values are hardcoded in bgc_params_init(), preserving
!     the original values from ibis_2.6b4/biogeochem.f (soilbgc, lines 228-351).
!   - biogeochem.f90 will call bgc_params_init() instead of setting locals.
!   - The values can later be made file-readable without changing any other
!     module (only bgc_params_init needs updating).
!
! Replaces: combgc.h (COMMON /combgc1/, /combgc2/, /combgc3/).
! =============================================================================
module mod_bgc_params
  use mod_precision, only: wp
  implicit none
  private

  public :: bgc_params_init
  ! /combgc1/: litter decomposition structure parameters
  public :: lig_frac, fbsom, effac, cnr, fmax, rconst, cnleaf, cnroot, cnwood
  ! /combgc2/: per-pool first-order decay rate constants (d-1)
  public :: klm, kls, kll, krm, krs, krl, kwm, kws, kwl
  public :: kbn, kbp, knb, kns, kpb, kps, ksb
  ! /combgc3/: microbial decomposition efficiencies (fraction)
  public :: ylm, yls, yll, yrm, yrs, yrl, ywm, yws, ywl
  public :: ybn, ybp, ynb, yns, ypb, yps, ysb

  ! ---------------------------------------------------------------------------
  ! /combgc1/  --  litter chemistry and SOM structure parameters
  ! ---------------------------------------------------------------------------

  !> Split of lignified litter between protected/non-protected slow OM pools.
  real(wp) :: lig_frac = 0.0_wp

  !> Protected biomass as fraction of total soil organic C (Verberne 1990).
  real(wp) :: fbsom = 0.0_wp

  !> Efficiency of microbial biomass reincorporation (Molina et al. 1983).
  real(wp) :: effac = 0.0_wp

  !> C:N ratios of substrate pools and biomass (10 values).
  !! Index: 1=microbial biomass, 2=passive SOM, 3=protected slow C,
  !!        4=non-protected slow C, 5=resistant lignin, 6=structural litter,
  !!        7=metabolic litter, 8=woody biomass, 9-10=unused.
  real(wp) :: cnr(10) = 0.0_wp

  !> Maximum fraction allowed in resistant litter fraction (Verberne 1997).
  real(wp) :: fmax = 0.0_wp

  !> Verberne model constant = 1200 (Verberne 1997).
  real(wp) :: rconst = 0.0_wp

  !> Average C:N ratio for leaf litterfall.
  real(wp) :: cnleaf = 0.0_wp

  !> Average C:N ratio for root turnover litter.
  real(wp) :: cnroot = 0.0_wp

  !> Average C:N ratio for woody debris.
  real(wp) :: cnwood = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /combgc2/  --  first-order decay rate constants (per day)
  ! Values from Parton et al. 1987, Verberne et al. 1990, Geoderma 1997.
  ! ---------------------------------------------------------------------------

  real(wp) :: klm = 0.0_wp   !< leaf metabolic litter --> microbial biomass
  real(wp) :: kls = 0.0_wp   !< leaf structural litter --> microbial biomass
  real(wp) :: kll = 0.0_wp   !< leaf lignin --> non/protected OM
  real(wp) :: krm = 0.0_wp   !< root metabolic litter --> microbial biomass
  real(wp) :: krs = 0.0_wp   !< root structural litter --> microbial biomass
  real(wp) :: krl = 0.0_wp   !< root lignin --> non/protected OM
  real(wp) :: kwm = 0.0_wp   !< woody metabolic litter --> microbial biomass
  real(wp) :: kws = 0.0_wp   !< woody structural litter --> microbial biomass
  real(wp) :: kwl = 0.0_wp   !< wood lignin --> non/protected OM
  real(wp) :: kbn = 0.0_wp   !< microbial biomass --> non-protected OM
  real(wp) :: kbp = 0.0_wp   !< microbial biomass --> protected OM
  real(wp) :: knb = 0.0_wp   !< non-protected OM --> microbial biomass
  real(wp) :: kns = 0.0_wp   !< non-protected OM --> passive C
  real(wp) :: kpb = 0.0_wp   !< protected OM --> microbial biomass
  real(wp) :: kps = 0.0_wp   !< protected OM --> passive C
  real(wp) :: ksb = 0.0_wp   !< passive C --> microbial biomass

  ! ---------------------------------------------------------------------------
  ! /combgc3/  --  microbial decomposition efficiencies (fraction)
  ! ---------------------------------------------------------------------------

  real(wp) :: ylm = 0.0_wp   !< leaf metabolic litter decomposition efficiency
  real(wp) :: yls = 0.0_wp   !< leaf structural litter
  real(wp) :: yll = 0.0_wp   !< leaf lignin
  real(wp) :: yrm = 0.0_wp   !< root metabolic litter
  real(wp) :: yrs = 0.0_wp   !< root structural litter
  real(wp) :: yrl = 0.0_wp   !< root lignin
  real(wp) :: ywm = 0.0_wp   !< woody metabolic litter
  real(wp) :: yws = 0.0_wp   !< woody structural litter
  real(wp) :: ywl = 0.0_wp   !< wood lignin
  real(wp) :: ybn = 0.0_wp   !< microbial biomass --> non-protected OM
  real(wp) :: ybp = 0.0_wp   !< microbial biomass --> protected OM
  real(wp) :: ynb = 0.0_wp   !< non-protected OM --> microbial biomass
  real(wp) :: yns = 0.0_wp   !< non-protected OM --> passive C
  real(wp) :: ypb = 0.0_wp   !< protected OM --> microbial biomass
  real(wp) :: yps = 0.0_wp   !< protected OM --> passive C
  real(wp) :: ysb = 0.0_wp   !< passive C --> microbial biomass

contains

  ! ---------------------------------------------------------------------------
  !> Initialise BGC parameters to their F77 original values.
  !!
  !! Values sourced from ibis_2.6b4/biogeochem.f, subroutine soilbgc,
  !! lines 228-351.  References: Parton et al. 1987 (CENTURY), Verberne et al.
  !! 1990 & 1997 (Geoderma special issue), Molina et al. 1983 (NCSOIL).
  !!
  !! NOTE: cnr(9) and cnr(10) are declared in the original combgc.h but never
  !! assigned; they default to 0 here to match the uninitialized F77 behavior.
  ! ---------------------------------------------------------------------------
  subroutine bgc_params_init()
    implicit none

    ! -- /combgc2/ decay rate constants (per day) ----------------------------
    ! Leaf litter
    klm = 0.15_wp       ! DPM leaf --> microbial biomass
    kls = 0.01_wp       ! SPM leaf --> microbial biomass
    kll = 0.01_wp       ! RPM leaf --> non- or protected OM

    ! Root litter
    krm = 0.10_wp       ! DPM root --> microbial biomass
    krs = 0.005_wp      ! SPM root --> microbial biomass
    krl = 0.005_wp      ! RPM root --> non- or protected OM

    ! Woody litter
    kwm = 0.001_wp      ! DPM wood --> microbial biomass
    kws = 0.001_wp      ! SPM wood --> microbial biomass
    kwl = 0.001_wp      ! RPM wood --> non- or protected OM

    ! Biomass pools
    kbn = 0.045_wp      ! biomass --> non-protected OM
    kbp = 0.005_wp      ! biomass --> protected OM

    ! Slow and passive C pools
    knb = 0.001_wp      ! non-protected OM --> biomass
    kns = 0.000001_wp   ! non-protected OM --> stabilised OM
    kpb = 0.0001_wp     ! protected OM --> biomass
    kps = 0.000001_wp   ! protected OM --> stabilised OM
    ksb = 8.0e-7_wp     ! stabilised OM --> biomass

    ! -- /combgc3/ efficiencies (Verberne / CENTURY) -------------------------
    ylm = 0.4_wp        ! metabolic material efficiencies
    yrm = 0.4_wp
    ywm = 0.4_wp
    yls = 0.3_wp        ! structural efficiencies
    yrs = 0.3_wp
    yws = 0.3_wp
    yll = 1.0_wp        ! resistant / lignin fractions
    yrl = 1.0_wp
    ywl = 1.0_wp
    ybn = 1.0_wp        ! biomass --> non-protected pool
    ybp = 1.0_wp        ! biomass --> protected pool
    yps = 1.0_wp        ! protected --> passive
    yns = 1.0_wp        ! non-protected --> passive
    ysb = 0.20_wp       ! passive pool --> biomass
    ypb = 0.20_wp       ! protected --> biomass
    ynb = 0.25_wp       ! non-protected --> biomass

    ! -- /combgc1/ litter chemistry and SOM structure ------------------------
    lig_frac = 0.50_wp      ! lignin split between protected/non-protected OM
    fbsom    = 0.017_wp     ! protected biomass fraction of total SOC
    effac    = 0.40_wp      ! microbial biomass reincorporation efficiency

    ! C:N ratios (Parton 1987, Whitmore & Parry 1988)
    cnr(1)  =   8.0_wp    ! microbial biomass
    cnr(2)  =  15.0_wp    ! passive soil C
    cnr(3)  =  10.0_wp    ! protected slow soil C
    cnr(4)  =  15.0_wp    ! non-protected slow soil C
    cnr(5)  = 100.0_wp    ! resistant litter (lignin)
    cnr(6)  = 150.0_wp    ! structural plant litter
    cnr(7)  =   6.0_wp    ! metabolic plant litter
    cnr(8)  = 250.0_wp    ! woody biomass
    cnr(9)  =   0.0_wp    ! unused (uninitialized in original F77)
    cnr(10) =   0.0_wp    ! unused (uninitialized in original F77)

    fmax   = 0.45_wp        ! maximum resistant fraction (Verberne 1997)
    rconst = 1200.0_wp      ! Verberne model constant
    cnleaf = 40.0_wp        ! average C:N of leaf litterfall
    cnroot = 60.0_wp        ! average C:N of root turnover
    cnwood = 200.0_wp       ! average C:N of woody debris

  end subroutine bgc_params_init

end module mod_bgc_params
