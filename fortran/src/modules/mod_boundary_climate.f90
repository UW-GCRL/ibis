! =============================================================================
! mod_boundary_climate.f90  --  Boundary conditions and climatological data
!
! Replaces COMMON blocks /combcs1/ through /combcs5/ from combcs.h.
!
! Holds the static boundary conditions (topography, vegetation map, land mask)
! and the climatological monthly and daily forcing fields used by the weather
! generator.
!
! Array dimensions:
!   npoi      -- number of land points
!   nlon, nlat -- full longitude × latitude grid dimensions
!
! Replaces: combcs.h (COMMON /combcs1/ through /combcs5/).
! =============================================================================
module mod_boundary_climate
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nlon, nlat
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /combcs1/  --  static boundary conditions (per land point)
  ! ---------------------------------------------------------------------------
  public :: xintopo, xinveg, deltat

  !> Surface topographic elevation (m).
  real(wp) :: xintopo(npoi) = 0.0_wp

  !> Fixed vegetation map index.
  real(wp) :: xinveg(npoi)  = 0.0_wp

  !> Mean temperature of coldest month (degrees C).
  !! Used for vegetation bioclimatic limits.
  real(wp) :: deltat(npoi)  = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /combcs2/  --  full-grid land mask
  ! ---------------------------------------------------------------------------
  public :: lmask

  !> Land mask on the full lon×lat grid (0 = ocean/water, 1 = land).
  integer :: lmask(nlon, nlat) = 0

  ! ---------------------------------------------------------------------------
  ! /combcs3/  --  climatological monthly fields + anomalies (per land point)
  ! ---------------------------------------------------------------------------
  public :: xint, xinq, xinprec, xinwind, xincld, xinwet, xintrng

  !> Climatological temperature + anomaly, by month (degrees C).  (land_point, month).
  real(wp) :: xint(npoi, 12)    = 0.0_wp

  !> Climatological relative humidity + anomaly, by month (%).
  real(wp) :: xinq(npoi, 12)    = 0.0_wp

  !> Climatological precipitation + anomaly, by month (mm day-1).
  real(wp) :: xinprec(npoi, 12) = 0.0_wp

  !> Climatological wind speed + anomaly, by month (m s-1).
  real(wp) :: xinwind(npoi, 12) = 0.0_wp

  !> Climatological cloudiness + anomaly, by month (%).
  real(wp) :: xincld(npoi, 12)  = 0.0_wp

  !> Climatological wet days + anomaly, by month (days month-1).
  real(wp) :: xinwet(npoi, 12)  = 0.0_wp

  !> Climatological temperature range + anomaly, by month (degrees C).
  real(wp) :: xintrng(npoi, 12) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /combcs4/  --  long-term climatological monthly means
  ! ---------------------------------------------------------------------------
  public :: clmt, clmq, clmprec, clmw, clmwet, clmcld, clmtrng

  !> Long-term climatological mean temperature, by month (degrees C).
  real(wp) :: clmt(npoi, 12)    = 0.0_wp

  !> Long-term climatological mean relative humidity, by month (%).
  real(wp) :: clmq(npoi, 12)    = 0.0_wp

  !> Long-term climatological mean precipitation, by month (mm day-1).
  real(wp) :: clmprec(npoi, 12) = 0.0_wp

  !> Long-term climatological mean wind speed, by month (m s-1).
  real(wp) :: clmw(npoi, 12)    = 0.0_wp

  !> Long-term climatological mean wet days, by month (days month-1).
  real(wp) :: clmwet(npoi, 12)  = 0.0_wp

  !> Long-term climatological mean cloudiness, by month (%).
  real(wp) :: clmcld(npoi, 12)  = 0.0_wp

  !> Long-term climatological mean temperature range, by month (degrees C).
  real(wp) :: clmtrng(npoi, 12) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /combcs5/  --  daily combined forcing values (current day)
  ! ---------------------------------------------------------------------------
  public :: xintd, xinqd, xinprecd, xinwindd, xincldd, xintrngd

  !> Daily temperature forcing (climatology + anomaly, degrees C).
  real(wp) :: xintd(npoi)    = 0.0_wp

  !> Daily relative humidity forcing (climatology × anomaly, %).
  real(wp) :: xinqd(npoi)    = 0.0_wp

  !> Daily precipitation forcing (climatology × anomaly, mm day-1).
  real(wp) :: xinprecd(npoi) = 0.0_wp

  !> Daily wind speed forcing (climatology × anomaly, m s-1).
  real(wp) :: xinwindd(npoi) = 0.0_wp

  !> Daily cloud fraction forcing (climatology + anomaly, fraction).
  real(wp) :: xincldd(npoi)  = 0.0_wp

  !> Daily temperature range forcing (climatology × anomaly, degrees C).
  real(wp) :: xintrngd(npoi) = 0.0_wp

end module mod_boundary_climate
