! =============================================================================
! mod_summary.f90  --  Time-averaged diagnostic summary fields
!
! Replaces COMMON blocks /comsum1/, /comsum3/–/comsum9/ from comsum.h.
!
! Note: /comsum2/ is absent in the original file (gap from prior refactoring).
!
! Holds accumulators and time-averaged output fields at daily, monthly,
! annual, and 10-day timescales.  All fields are per land point; some include
! per-PFT breakdowns.
!
! Array dimensions:
!   npoi -- number of land points
!   npft -- number of plant functional types (= 12)
!
! Replaces: comsum.h (COMMON /comsum1/, /comsum3/–/comsum9/).
! =============================================================================
module mod_summary
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, npft
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /comsum1/  --  time-step counters
  ! ---------------------------------------------------------------------------
  public :: ndtimes, nmtimes, nytimes

  !> Counter for daily average calculations.
  integer :: ndtimes = 0

  !> Counter for monthly average calculations.
  integer :: nmtimes = 0

  !> Counter for yearly average calculations.
  integer :: nytimes  = 0

  ! ---------------------------------------------------------------------------
  ! /comsum3/  --  daily accumulated / averaged fields (all npoi)
  ! ---------------------------------------------------------------------------
  public :: adrain, adsnow, adaet, adtrunoff, adsrunoff, addrainage
  public :: adrh, adsnod, adsnof, adwsoi, adwisoi, adtsoi
  public :: adwsoic, adtsoic
  public :: adco2mic, adco2root, adco2soi, adco2ratio, adnmintot
  public :: adtlaysoi, adwlaysoi, adneetot

  real(wp) :: adrain(npoi)     = 0.0_wp  !< daily avg. rainfall (mm day-1)
  real(wp) :: adsnow(npoi)     = 0.0_wp  !< daily avg. snowfall (mm day-1)
  real(wp) :: adaet(npoi)      = 0.0_wp  !< daily avg. actual ET (mm day-1)
  real(wp) :: adtrunoff(npoi)  = 0.0_wp  !< daily avg. total runoff (mm day-1)
  real(wp) :: adsrunoff(npoi)  = 0.0_wp  !< daily avg. surface runoff (mm day-1)
  real(wp) :: addrainage(npoi) = 0.0_wp  !< daily avg. drainage (mm day-1)
  real(wp) :: adrh(npoi)       = 0.0_wp  !< daily avg. relative humidity (%)
  real(wp) :: adsnod(npoi)     = 0.0_wp  !< daily avg. snow depth (m)
  real(wp) :: adsnof(npoi)     = 0.0_wp  !< daily avg. snow fraction (dimensionless)
  real(wp) :: adwsoi(npoi)     = 0.0_wp  !< daily avg. soil moisture (pore fraction)
  real(wp) :: adwisoi(npoi)    = 0.0_wp  !< daily avg. soil ice (pore fraction)
  real(wp) :: adtsoi(npoi)     = 0.0_wp  !< daily avg. soil temperature (degrees C)
  real(wp) :: adwsoic(npoi)    = 0.0_wp  !< daily avg. root-weighted soil moisture
  real(wp) :: adtsoic(npoi)    = 0.0_wp  !< daily avg. root-weighted soil temperature (C)
  real(wp) :: adco2mic(npoi)   = 0.0_wp  !< daily accum. microbial CO2 respiration (kg_C m-2 day-1)
  real(wp) :: adco2root(npoi)  = 0.0_wp  !< daily accum. root CO2 respiration (kg_C m-2 day-1)
  real(wp) :: adco2soi(npoi)   = 0.0_wp  !< daily accum. total soil CO2 respiration (kg_C m-2 day-1)
  real(wp) :: adco2ratio(npoi) = 0.0_wp  !< daily ratio of root to total soil CO2
  real(wp) :: adnmintot(npoi)  = 0.0_wp  !< daily accum. N mineralisation (kg_N m-2 day-1)
  real(wp) :: adtlaysoi(npoi)  = 0.0_wp  !< daily avg. top-layer soil temperature (C)
  real(wp) :: adwlaysoi(npoi)  = 0.0_wp  !< daily avg. top-layer soil moisture (fraction)
  real(wp) :: adneetot(npoi)   = 0.0_wp  !< daily accum. net ecosystem CO2 exchange (kg_C m-2 day-1)

  ! ---------------------------------------------------------------------------
  ! /comsum4/  --  monthly averaged fields (all npoi)
  ! ---------------------------------------------------------------------------
  public :: amtemp, amrain, amsnow, amcloud, amrh, amqa, amaet
  public :: amtrunoff, amsrunoff, amdrainage
  public :: amwsoi, amwisoi, amvwc, amawc, amtsoi, amsnod, amsnof
  public :: amlaiu, amlail, amsolar, amalbedo, amirdown, amirup
  public :: amsens, amlatent
  public :: amnpptot, amneetot, amco2mic, amco2root, amco2soi, amco2ratio
  public :: amnmintot

  real(wp) :: amtemp(npoi)      = 0.0_wp  !< monthly avg. air temperature (C)
  real(wp) :: amrain(npoi)      = 0.0_wp  !< monthly avg. rainfall (mm day-1)
  real(wp) :: amsnow(npoi)      = 0.0_wp  !< monthly avg. snowfall (mm day-1)
  real(wp) :: amcloud(npoi)     = 0.0_wp  !< monthly avg. cloudiness (%)
  real(wp) :: amrh(npoi)        = 0.0_wp  !< monthly avg. relative humidity (%)
  real(wp) :: amqa(npoi)        = 0.0_wp  !< monthly avg. specific humidity (kg kg-1)
  real(wp) :: amaet(npoi)       = 0.0_wp  !< monthly avg. actual ET (mm day-1)
  real(wp) :: amtrunoff(npoi)   = 0.0_wp  !< monthly avg. total runoff (mm day-1)
  real(wp) :: amsrunoff(npoi)   = 0.0_wp  !< monthly avg. surface runoff (mm day-1)
  real(wp) :: amdrainage(npoi)  = 0.0_wp  !< monthly avg. drainage (mm day-1)
  real(wp) :: amwsoi(npoi)      = 0.0_wp  !< monthly avg. 1-m soil moisture (fraction)
  real(wp) :: amwisoi(npoi)     = 0.0_wp  !< monthly avg. 1-m soil ice (fraction)
  real(wp) :: amvwc(npoi)       = 0.0_wp  !< monthly avg. 1-m volumetric water content
  real(wp) :: amawc(npoi)       = 0.0_wp  !< monthly avg. 1-m plant-available water content
  real(wp) :: amtsoi(npoi)      = 0.0_wp  !< monthly avg. 1-m soil temperature (C)
  real(wp) :: amsnod(npoi)      = 0.0_wp  !< monthly avg. snow depth (m)
  real(wp) :: amsnof(npoi)      = 0.0_wp  !< monthly avg. snow fraction
  real(wp) :: amlaiu(npoi)      = 0.0_wp  !< monthly avg. upper-canopy LAI (m2 m-2)
  real(wp) :: amlail(npoi)      = 0.0_wp  !< monthly avg. lower-canopy LAI (m2 m-2)
  real(wp) :: amsolar(npoi)     = 0.0_wp  !< monthly avg. incident solar (W m-2)
  real(wp) :: amalbedo(npoi)    = 0.0_wp  !< monthly avg. solar albedo
  real(wp) :: amirdown(npoi)    = 0.0_wp  !< monthly avg. downward IR (W m-2)
  real(wp) :: amirup(npoi)      = 0.0_wp  !< monthly avg. upward IR (W m-2)
  real(wp) :: amsens(npoi)      = 0.0_wp  !< monthly avg. sensible heat flux (W m-2)
  real(wp) :: amlatent(npoi)    = 0.0_wp  !< monthly avg. latent heat flux (W m-2)
  real(wp) :: amnpptot(npoi)    = 0.0_wp  !< monthly total NPP (kg_C m-2 month-1)
  real(wp) :: amneetot(npoi)    = 0.0_wp  !< monthly total NEE (kg_C m-2 month-1)
  real(wp) :: amco2mic(npoi)    = 0.0_wp  !< monthly total microbial CO2 (kg_C m-2 month-1)
  real(wp) :: amco2root(npoi)   = 0.0_wp  !< monthly total root CO2 (kg_C m-2 month-1)
  real(wp) :: amco2soi(npoi)    = 0.0_wp  !< monthly total soil CO2 (kg_C m-2 month-1)
  real(wp) :: amco2ratio(npoi)  = 0.0_wp  !< monthly ratio root/total soil CO2
  real(wp) :: amnmintot(npoi)   = 0.0_wp  !< monthly total N mineralisation (kg_N m-2 month-1)

  ! ---------------------------------------------------------------------------
  ! /comsum5/  --  monthly NPP per PFT
  ! ---------------------------------------------------------------------------
  public :: amnpp

  !> Monthly total NPP for each PFT (kg_C m-2 month-1).  (land_point, pft).
  real(wp) :: amnpp(npoi, npft) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsum6/  --  annual averaged / accumulated fields (all npoi)
  ! ---------------------------------------------------------------------------
  public :: ayprcp, ayaet, aytrans, aytrunoff, aysrunoff, aydrainage, aydwtot
  public :: aywsoi, aywisoi, ayvwc, ayawc, aytsoi, ayrratio, aytratio
  public :: aysolar, ayalbedo, ayirdown, ayirup, aysens, aylatent
  public :: aystresstu, aystresstl
  public :: ayanpptot, aynpptot, aygpptot
  public :: ayalit, ayblit, aycsoi, aycmic
  public :: ayanlit, aybnlit, aynsoi, ynleach
  public :: ayneetot, ayco2mic, ayco2root, ayco2soi, aynmintot, ayrootbio

  real(wp) :: ayprcp(npoi)      = 0.0_wp  !< annual avg. precip. (mm yr-1)
  real(wp) :: ayaet(npoi)       = 0.0_wp  !< annual avg. AET (mm yr-1)
  real(wp) :: aytrans(npoi)     = 0.0_wp  !< annual avg. transpiration (mm yr-1)
  real(wp) :: aytrunoff(npoi)   = 0.0_wp  !< annual avg. total runoff (mm yr-1)
  real(wp) :: aysrunoff(npoi)   = 0.0_wp  !< annual avg. surface runoff (mm yr-1)
  real(wp) :: aydrainage(npoi)  = 0.0_wp  !< annual avg. drainage (mm yr-1)
  real(wp) :: aydwtot(npoi)     = 0.0_wp  !< annual avg. total water recharge (mm yr-1)
  real(wp) :: aywsoi(npoi)      = 0.0_wp  !< annual avg. 1-m soil moisture (fraction)
  real(wp) :: aywisoi(npoi)     = 0.0_wp  !< annual avg. 1-m soil ice (fraction)
  real(wp) :: ayvwc(npoi)       = 0.0_wp  !< annual avg. 1-m volumetric water content
  real(wp) :: ayawc(npoi)       = 0.0_wp  !< annual avg. 1-m plant-available water content
  real(wp) :: aytsoi(npoi)      = 0.0_wp  !< annual avg. 1-m soil temperature (C)
  real(wp) :: ayrratio(npoi)    = 0.0_wp  !< annual avg. runoff ratio
  real(wp) :: aytratio(npoi)    = 0.0_wp  !< annual avg. transpiration ratio
  real(wp) :: aysolar(npoi)     = 0.0_wp  !< annual avg. incident solar (W m-2)
  real(wp) :: ayalbedo(npoi)    = 0.0_wp  !< annual avg. solar albedo
  real(wp) :: ayirdown(npoi)    = 0.0_wp  !< annual avg. downward IR (W m-2)
  real(wp) :: ayirup(npoi)      = 0.0_wp  !< annual avg. upward IR (W m-2)
  real(wp) :: aysens(npoi)      = 0.0_wp  !< annual avg. sensible heat (W m-2)
  real(wp) :: aylatent(npoi)    = 0.0_wp  !< annual avg. latent heat (W m-2)
  real(wp) :: aystresstu(npoi)  = 0.0_wp  !< annual avg. upper-canopy soil moisture stress
  real(wp) :: aystresstl(npoi)  = 0.0_wp  !< annual avg. lower-canopy soil moisture stress
  real(wp) :: ayanpptot(npoi)   = 0.0_wp  !< annual above-ground NPP (kg_C m-2 yr-1)
  real(wp) :: aynpptot(npoi)    = 0.0_wp  !< annual total NPP (kg_C m-2 yr-1)
  real(wp) :: aygpptot(npoi)    = 0.0_wp  !< annual total GPP (kg_C m-2 yr-1)
  real(wp) :: ayalit(npoi)      = 0.0_wp  !< aboveground litter (kg_C m-2)
  real(wp) :: ayblit(npoi)      = 0.0_wp  !< belowground litter (kg_C m-2)
  real(wp) :: aycsoi(npoi)      = 0.0_wp  !< total soil carbon (kg_C m-2)
  real(wp) :: aycmic(npoi)      = 0.0_wp  !< microbial biomass carbon (kg_C m-2)
  real(wp) :: ayanlit(npoi)     = 0.0_wp  !< aboveground litter N (kg_N m-2)
  real(wp) :: aybnlit(npoi)     = 0.0_wp  !< belowground litter N (kg_N m-2)
  real(wp) :: aynsoi(npoi)      = 0.0_wp  !< total soil N (kg_N m-2)
  real(wp) :: ynleach(npoi)     = 0.0_wp  !< annual N leached from soil (kg_N m-2 yr-1)
  real(wp) :: ayneetot(npoi)    = 0.0_wp  !< annual total NEE (kg_C m-2 yr-1)
  real(wp) :: ayco2mic(npoi)    = 0.0_wp  !< annual microbial CO2 (kg_C m-2 yr-1)
  real(wp) :: ayco2root(npoi)   = 0.0_wp  !< annual root CO2 (kg_C m-2 yr-1)
  real(wp) :: ayco2soi(npoi)    = 0.0_wp  !< annual total soil CO2 (kg_C m-2 yr-1)
  real(wp) :: aynmintot(npoi)   = 0.0_wp  !< annual total N mineralisation (kg_N m-2 yr-1)
  real(wp) :: ayrootbio(npoi)   = 0.0_wp  !< annual avg. live root biomass (kg_C m-2)

  ! ---------------------------------------------------------------------------
  ! /comsum7/  --  annual NPP and GPP per PFT
  ! ---------------------------------------------------------------------------
  public :: ayanpp, aynpp, aygpp

  !> Annual above-ground NPP per PFT (kg_C m-2 yr-1).  (land_point, pft).
  real(wp) :: ayanpp(npoi, npft) = 0.0_wp

  !> Annual total NPP per PFT (kg_C m-2 yr-1).  (land_point, pft).
  real(wp) :: aynpp(npoi, npft)  = 0.0_wp

  !> Annual total GPP per PFT (kg_C m-2 yr-1).  (land_point, pft).
  real(wp) :: aygpp(npoi, npft)  = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsum8/  --  10-day running averages
  ! ---------------------------------------------------------------------------
  public :: a10td
  public :: a10ancub, a10ancuc, a10ancls, a10ancl3, a10ancl4
  public :: a10scalparamu, a10scalparaml
  public :: a10daylightu, a10daylightl

  !> 10-day average daily air temperature (K).
  real(wp) :: a10td(npoi)          = 0.0_wp

  !> 10-day average canopy photosynthesis, broadleaf (mol_CO2 m-2 s-1).
  real(wp) :: a10ancub(npoi)       = 0.0_wp

  !> 10-day average canopy photosynthesis, conifer (mol_CO2 m-2 s-1).
  real(wp) :: a10ancuc(npoi)       = 0.0_wp

  !> 10-day average canopy photosynthesis, shrubs (mol_CO2 m-2 s-1).
  real(wp) :: a10ancls(npoi)       = 0.0_wp

  !> 10-day average canopy photosynthesis, C3 grasses (mol_CO2 m-2 s-1).
  real(wp) :: a10ancl3(npoi)       = 0.0_wp

  !> 10-day average canopy photosynthesis, C4 grasses (mol_CO2 m-2 s-1).
  real(wp) :: a10ancl4(npoi)       = 0.0_wp

  !> 10-day average day-time scaling parameter, upper canopy.
  real(wp) :: a10scalparamu(npoi)  = 0.0_wp

  !> 10-day average day-time scaling parameter, lower canopy.
  real(wp) :: a10scalparaml(npoi)  = 0.0_wp

  !> 10-day average day-time PAR, upper canopy (micro-Ein m-2 s-1).
  real(wp) :: a10daylightu(npoi)   = 0.0_wp

  !> 10-day average day-time PAR, lower canopy (micro-Ein m-2 s-1).
  real(wp) :: a10daylightl(npoi)   = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsum9/  --  soil N storage and C leaching
  ! ---------------------------------------------------------------------------
  public :: storedn, yrleach

  !> Total N stored in soil profile (kg_N m-2).
  real(wp) :: storedn(npoi) = 0.0_wp

  !> Annual total C leached from soil profile (kg_C m-2 yr-1).
  real(wp) :: yrleach(npoi) = 0.0_wp

end module mod_summary
