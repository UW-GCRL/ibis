! =============================================================================
! mod_veg_dynamics.f90  --  Vegetation dynamics state variables for IBIS
!
! Replaces COMMON blocks /comveg5/, /comveg8/, /comveg9/, /comveg12/,
! /comveg13/, /comveg14/ from comveg.h.
!
! Covers:
!   comveg5  -- per-point phenology, competition, diagnostics, fluxes (scalars
!               and npoi arrays)
!   comveg8  -- per-point per-PFT biomass fractions and instantaneous fluxes;
!               NOTE: nppdummy appears after plai in source but is NOT in the
!               COMMON statement – it is a local variable in the physics
!               routine and is intentionally omitted here.
!   comveg9  -- per-point per-canopy-level LAI/SAI, heights
!   comveg12 -- root fraction per soil layer per canopy level
!   comveg13 -- PFT existence probability per grid point
!   comveg14 -- PFT-level allocation and specific leaf area parameters
!
! All variables initialised to 0.0; set at run-time by phenology, allocation,
! and vegetation dynamics routines (and by RD_PARAM for comveg14 values).
!
! Replaces: comveg.h (/comveg5/, /comveg8/, /comveg9/, /comveg12/–/comveg14/).
! =============================================================================
module mod_veg_dynamics
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, npft, nsoilay
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /comveg5/: phenology, climate diagnostics, competition, ecosystem fluxes
  ! ---------------------------------------------------------------------------
  public :: agddl, agddu, fl, fu
  public :: gdd0, gdd0this, gdd5, gdd5this
  public :: sapfrac, tc, tcthis, tcmin
  public :: totlail, totlaiu, totbiol, totbiou
  public :: tw, twthis
  public :: disturbf, disturbo, firefac
  public :: tco2mic, tco2root, tneetot, tnmin, tnpptot, tgpptot
  public :: totalit, totanlit, totcmic, totcsoi, totfall, totlit
  public :: totnlit, totnmic, totnsoi, totrlit, totrnlit
  public :: tempu, templ, dropu, dropls, dropl4, dropl3
  public :: vegtype0

  !> Annual accumulated growing degree days for bud burst, lower canopy (day-degrees).
  real(wp) :: agddl(npoi)    = 0.0_wp

  !> Annual accumulated growing degree days for bud burst, upper canopy (day-degrees).
  real(wp) :: agddu(npoi)    = 0.0_wp

  !> Fraction of snow-free area covered by lower canopy (dimensionless).
  real(wp) :: fl(npoi)       = 0.0_wp

  !> Fraction of overall area covered by upper canopy (dimensionless).
  real(wp) :: fu(npoi)       = 0.0_wp

  !> Growing degree days above 0 C (day-degrees).
  real(wp) :: gdd0(npoi)     = 0.0_wp

  !> Annual total growing degree days above 0 C for current year (day-degrees).
  real(wp) :: gdd0this(npoi) = 0.0_wp

  !> Growing degree days above 5 C (day-degrees).
  real(wp) :: gdd5(npoi)     = 0.0_wp

  !> Annual total growing degree days above 5 C for current year (day-degrees).
  real(wp) :: gdd5this(npoi) = 0.0_wp

  !> Fraction of woody biomass in sapwood (dimensionless).
  real(wp) :: sapfrac(npoi)  = 0.0_wp

  !> Coldest monthly temperature (C).
  real(wp) :: tc(npoi)       = 0.0_wp

  !> Coldest monthly temperature of current year (C).
  real(wp) :: tcthis(npoi)   = 0.0_wp

  !> Coldest daily temperature of current year (C).
  real(wp) :: tcmin(npoi)    = 0.0_wp

  !> Total leaf area index for the lower canopy (m2 m-2).
  real(wp) :: totlail(npoi)  = 0.0_wp

  !> Total leaf area index for the upper canopy (m2 m-2).
  real(wp) :: totlaiu(npoi)  = 0.0_wp

  !> Total biomass in the lower canopy (kg_C m-2).
  real(wp) :: totbiol(npoi)  = 0.0_wp

  !> Total biomass in the upper canopy (kg_C m-2).
  real(wp) :: totbiou(npoi)  = 0.0_wp

  !> Warmest monthly temperature (C).
  real(wp) :: tw(npoi)       = 0.0_wp

  !> Warmest monthly temperature of current year (C).
  real(wp) :: twthis(npoi)   = 0.0_wp

  !> Annual fire disturbance fraction (m2 m-2 yr-1).
  real(wp) :: disturbf(npoi) = 0.0_wp

  !> Fraction of biomass pool lost annually to non-fire disturbances.
  real(wp) :: disturbo(npoi) = 0.0_wp

  !> Annual average fuel dryness factor (dimensionless).
  real(wp) :: firefac(npoi)  = 0.0_wp

  !> Instantaneous microbial CO2 flux from soil (mol CO2 m-2 s-1).
  real(wp) :: tco2mic(npoi)  = 0.0_wp

  !> Instantaneous fine-root CO2 flux from soil (mol CO2 m-2 s-1).
  real(wp) :: tco2root(npoi) = 0.0_wp

  !> Instantaneous net ecosystem exchange of CO2 per timestep (kg_C m-2 timestep-1).
  real(wp) :: tneetot(npoi)  = 0.0_wp

  !> Instantaneous nitrogen mineralisation (kg_N m-2 timestep-1).
  real(wp) :: tnmin(npoi)    = 0.0_wp

  !> Instantaneous NPP (mol CO2 m-2 s-1).
  real(wp) :: tnpptot(npoi)  = 0.0_wp

  !> Instantaneous GPP (mol CO2 m-2 s-1).
  real(wp) :: tgpptot(npoi)  = 0.0_wp

  !> Total standing aboveground litter (kg_C m-2).
  real(wp) :: totalit(npoi)  = 0.0_wp

  !> Total standing aboveground nitrogen in litter (kg_N m-2).
  real(wp) :: totanlit(npoi) = 0.0_wp

  !> Total carbon in microbial pools (kg_C m-2).
  real(wp) :: totcmic(npoi)  = 0.0_wp

  !> Total carbon in all soil pools (kg_C m-2).
  real(wp) :: totcsoi(npoi)  = 0.0_wp

  !> Total litterfall and root turnover (kg_C m-2 yr-1).
  real(wp) :: totfall(npoi)  = 0.0_wp

  !> Total carbon in all litter pools (kg_C m-2).
  real(wp) :: totlit(npoi)   = 0.0_wp

  !> Total nitrogen in all litter pools (kg_N m-2).
  real(wp) :: totnlit(npoi)  = 0.0_wp

  !> Total nitrogen in microbial pool (kg_N m-2).
  real(wp) :: totnmic(npoi)  = 0.0_wp

  !> Total nitrogen in soil (kg_N m-2).
  real(wp) :: totnsoi(npoi)  = 0.0_wp

  !> Total root litter carbon belowground (kg_C m-2).
  real(wp) :: totrlit(npoi)  = 0.0_wp

  !> Total root litter nitrogen belowground (kg_N m-2).
  real(wp) :: totrnlit(npoi) = 0.0_wp

  !> Cold-phenology trigger for trees (dimensionless).
  real(wp) :: tempu(npoi)    = 0.0_wp

  !> Cold-phenology trigger for grasses/shrubs (dimensionless).
  real(wp) :: templ(npoi)    = 0.0_wp

  !> Drought-phenology trigger for trees (dimensionless).
  real(wp) :: dropu(npoi)    = 0.0_wp

  !> Drought-phenology trigger for shrubs (dimensionless).
  real(wp) :: dropls(npoi)   = 0.0_wp

  !> Drought-phenology trigger for C4 grasses (dimensionless).
  real(wp) :: dropl4(npoi)   = 0.0_wp

  !> Drought-phenology trigger for C3 grasses (dimensionless).
  real(wp) :: dropl3(npoi)   = 0.0_wp

  !> Annual vegetation type (IBIS classification).
  real(wp) :: vegtype0(npoi) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg8/: per-PFT biomass fractions and instantaneous carbon fluxes.
  !
  ! NOTE: nppdummy (canopy NPP before stem and root respiration) appears after
  ! plai in the variable declarations of comveg.h but is explicitly absent from
  ! the COMMON /comveg8/ statement; it is a local variable in the physics
  ! routines and is NOT included here.
  ! ---------------------------------------------------------------------------
  public :: biomass, frac, plai, tnpp, tgpp

  !> Total biomass of each plant functional type (kg_C m-2).
  real(wp) :: biomass(npoi, npft) = 0.0_wp

  !> Fraction of canopy occupied by each plant functional type (dimensionless).
  real(wp) :: frac(npoi, npft)    = 0.0_wp

  !> Total leaf area index of each plant functional type (m2 m-2).
  real(wp) :: plai(npoi, npft)    = 0.0_wp

  !> Instantaneous NPP for each PFT (mol CO2 m-2 s-1).
  real(wp) :: tnpp(npoi, npft)    = 0.0_wp

  !> Instantaneous GPP for each PFT (mol CO2 m-2 s-1).
  real(wp) :: tgpp(npoi, npft)    = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg9/: canopy structure – LAI, SAI, heights
  !            (index 1 = lower canopy, 2 = upper canopy)
  ! ---------------------------------------------------------------------------
  public :: lai, sai, zbot, ztop, ztopmx

  !> Canopy single-sided leaf area index (area leaf / area veg).
  real(wp) :: lai(npoi, 2)    = 0.0_wp

  !> Current single-sided stem area index (m2 m-2).
  real(wp) :: sai(npoi, 2)    = 0.0_wp

  !> Height of lowest branches above ground (m).
  real(wp) :: zbot(npoi, 2)   = 0.0_wp

  !> Height of plant top above ground (m).
  real(wp) :: ztop(npoi, 2)   = 0.0_wp

  !> Maximum annual height of plant top above ground (m).
  real(wp) :: ztopmx(npoi, 2) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg12/: root fraction per soil layer per canopy level
  !             (index 1 = lower canopy, 2 = upper canopy)
  ! ---------------------------------------------------------------------------
  public :: froot

  !> Fraction of roots in each soil layer (dimensionless).
  real(wp) :: froot(nsoilay, 2) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg13/: PFT existence probability
  ! ---------------------------------------------------------------------------
  public :: exist

  !> Probability of existence of each PFT in a grid cell (dimensionless).
  real(wp) :: exist(npoi, npft) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg14/: PFT-level allocation and specific leaf area
  !             (read from params.veg by RD_PARAM)
  ! ---------------------------------------------------------------------------
  public :: specla, aleaf, aroot, awood

  !> Specific leaf area (m2 kg-1).
  real(wp) :: specla(npft) = 0.0_wp

  !> Carbon allocation fraction to leaves (dimensionless).
  real(wp) :: aleaf(npft)  = 0.0_wp

  !> Carbon allocation fraction to fine roots (dimensionless).
  real(wp) :: aroot(npft)  = 0.0_wp

  !> Carbon allocation fraction to wood (dimensionless).
  real(wp) :: awood(npft)  = 0.0_wp

end module mod_veg_dynamics
