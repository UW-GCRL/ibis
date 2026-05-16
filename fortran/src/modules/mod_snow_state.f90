! =============================================================================
! mod_snow_state.f90  --  Snow-layer state variables
!
! Replaces COMMON blocks /comsno1/, /comsno2/, /comsno3/ from comsno.h.
!
! /comsno1/ holds bulk snow physical parameters (scalars) set at model
! initialisation.  /comsno2/ and /comsno3/ hold the per-land-point and
! per-layer prognostic snow state.
!
! Array dimensions used:
!   npoi    -- number of land points in the active domain
!   nsnolay -- number of snow layers (= 3)
!
! Replaces: comsno.h (COMMON /comsno1/, /comsno2/, /comsno3/).
! =============================================================================
module mod_snow_state
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nsnolay
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /comsno1/  --  bulk snow physical parameters (scalars; set at init)
  ! ---------------------------------------------------------------------------
  public :: z0sno, rhos, consno, hsnotop, hsnomin, fimin, fimax

  !> Roughness length of snow surface (m).
  real(wp) :: z0sno   = 0.0_wp

  !> Density of snow (kg m-3).
  real(wp) :: rhos    = 0.0_wp

  !> Thermal conductivity of snow (W m-1 K-1).
  real(wp) :: consno  = 0.0_wp

  !> Thickness of top snow layer (m).
  real(wp) :: hsnotop = 0.0_wp

  !> Minimum total thickness of snow (m).
  real(wp) :: hsnomin = 0.0_wp

  !> Minimum fractional snow cover (dimensionless).
  real(wp) :: fimin   = 0.0_wp

  !> Maximum fractional snow cover (dimensionless).
  real(wp) :: fimax   = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsno2/  --  fractional snow cover per land point
  ! ---------------------------------------------------------------------------
  public :: fi

  !> Fractional snow cover (dimensionless).
  real(wp) :: fi(npoi) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsno3/  --  per-layer snow temperature and thickness
  ! ---------------------------------------------------------------------------
  public :: tsno, hsno

  !> Temperature of snow layers (K).  Index: (land_point, snow_layer).
  real(wp) :: tsno(npoi, nsnolay) = 0.0_wp

  !> Thickness of snow layers (m).  Index: (land_point, snow_layer).
  real(wp) :: hsno(npoi, nsnolay) = 0.0_wp

end module mod_snow_state
