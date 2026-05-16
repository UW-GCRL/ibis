! =============================================================================
! mod_atm_state.f90  --  Atmospheric forcing and state variables
!
! Replaces COMMON blocks /comatm1/ through /comatm7/ from comatm.h.
!
! Holds all atmospheric forcing fields received at each model timestep
! (radiation, wind, temperature, humidity, precipitation, pressure) plus
! weather-generator state and climatological averages.
!
! Array dimensions:
!   npoi  -- number of land points
!   nband -- number of radiation wavebands (= 2: VIS and NIR)
!
! Replaces: comatm.h (COMMON /comatm1/ through /comatm7/).
! =============================================================================
module mod_atm_state
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nband
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /comatm1/  --  solar zenith and downwelling IR
  ! ---------------------------------------------------------------------------
  public :: coszen, fira

  !> Cosine of solar zenith angle (dimensionless).
  real(wp) :: coszen(npoi) = 0.0_wp

  !> Incoming IR (longwave) flux (W m-2).
  real(wp) :: fira(npoi)   = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comatm2/  --  solar radiation and surface albedo per waveband
  ! ---------------------------------------------------------------------------
  public :: solad, solai, asurd, asuri

  !> Direct downward solar flux per waveband (W m-2).  (land_point, band).
  real(wp) :: solad(npoi, nband) = 0.0_wp

  !> Diffuse downward solar flux per waveband (W m-2).  (land_point, band).
  real(wp) :: solai(npoi, nband) = 0.0_wp

  !> Direct albedo of the surface system per waveband.  (land_point, band).
  real(wp) :: asurd(npoi, nband) = 0.0_wp

  !> Diffuse albedo of the surface system per waveband.  (land_point, band).
  real(wp) :: asuri(npoi, nband) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comatm3/  --  instantaneous meteorological forcing
  ! ---------------------------------------------------------------------------
  public :: ua, ta, qa, raina, rh, snowa

  !> Wind speed at atmospheric reference level (m s-1).
  real(wp) :: ua(npoi)    = 0.0_wp

  !> Air temperature at atmospheric reference level (K).
  real(wp) :: ta(npoi)    = 0.0_wp

  !> Specific humidity at atmospheric reference level (kg_H2O kg_air-1).
  real(wp) :: qa(npoi)    = 0.0_wp

  !> Rainfall rate (kg m-2 s-1 = mm s-1).
  real(wp) :: raina(npoi) = 0.0_wp

  !> Relative humidity (%).
  real(wp) :: rh(npoi)    = 0.0_wp

  !> Snowfall rate (kg_H2O m-2 s-1 = mm_water s-1).
  real(wp) :: snowa(npoi) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comatm4/  --  daily and multi-day statistics; precipitation array
  ! ---------------------------------------------------------------------------
  public :: psurf, cloud, td, tmax, tmin, qd, ud
  public :: precip, precipday, precipdaysum

  !> Surface pressure (Pa).
  real(wp) :: psurf(npoi)         = 0.0_wp

  !> Cloud fraction (dimensionless).
  real(wp) :: cloud(npoi)         = 0.0_wp

  !> Daily average air temperature (K).
  real(wp) :: td(npoi)            = 0.0_wp

  !> Maximum daily air temperature (K).
  real(wp) :: tmax(npoi)          = 0.0_wp

  !> Minimum daily air temperature (K).
  real(wp) :: tmin(npoi)          = 0.0_wp

  !> Daily average specific humidity (kg_H2O kg_air-1).
  real(wp) :: qd(npoi)            = 0.0_wp

  !> Daily average wind speed (m s-1).
  real(wp) :: ud(npoi)            = 0.0_wp

  !> Daily precipitation (mm day-1).
  real(wp) :: precip(npoi)        = 0.0_wp

  !> Daily precipitation values within a month (mm day-1).  (land_point, day_of_month).
  real(wp) :: precipday(npoi, 31) = 0.0_wp

  !> Monthly precipitation accumulator (mm).
  real(wp) :: precipdaysum(npoi)  = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comatm5/  --  weather-generator state matrix
  ! ---------------------------------------------------------------------------
  public :: xstore

  !> Weather-generator memory matrix (3 state variables per land point).
  real(wp) :: xstore(npoi, 3) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comatm6/  --  wet/dry day flags (integer)
  ! ---------------------------------------------------------------------------
  public :: iwet, iwetday, iwetdaysum

  !> Wet-day flag (1) / dry-day flag (0) for current day.
  integer :: iwet(npoi)          = 0

  !> Wet/dry flag for each day of the current month.  (land_point, day_of_month).
  integer :: iwetday(npoi, 31)   = 0

  !> Count of wet days so far in the current month.
  integer :: iwetdaysum(npoi)    = 0

  ! ---------------------------------------------------------------------------
  ! /comatm7/  --  atmospheric gas concentrations (scalars)
  ! ---------------------------------------------------------------------------
  public :: co2conc, o2conc

  !> Atmospheric CO2 concentration (mol mol-1).
  real(wp) :: co2conc = 0.0_wp

  !> Atmospheric O2 concentration (mol mol-1).
  real(wp) :: o2conc  = 0.0_wp

end module mod_atm_state
