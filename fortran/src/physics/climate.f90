! =============================================================================
! climate.f90  --  Climate-based phenology and PFT existence routines for IBIS
!
! Converted from climate.f (Fortran 77 fixed-form) to F2008 free-form.
!
! Subroutines provided:
!   climanl()    -- initialise GDD, tc, tw from climatological monthly means
!   climanl2()   -- update GDD, tc, tw with 30-year e-folding running average
!   existence()  -- determine which PFTs can exist in each grid cell
! =============================================================================

! -----------------------------------------------------------------------------
subroutine climanl()
! Initialises growing degree days, coldest monthly temperature, and warmest
! monthly temperature at the very start of the simulation using climatological
! monthly mean values.  Provides a climate 'history' for year 0.
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_dimensions,       only: npoi
  use mod_run_params,       only: ndaypm
  use mod_boundary_climate, only: xint, deltat
  use mod_veg_dynamics,     only: tc, tw, tcmin, gdd0, gdd5
  implicit none

  integer  :: it1w, it2w    ! indices of previous/following month for interpolation
  integer  :: i, k, lda     ! loop indices
  real(wp) :: rwork         ! 1 / ndaypm(k)
  real(wp) :: dt            ! interpolation fraction
  real(wp) :: dtemp         ! interpolated daily temperature

  ! Initialise growing degree day accumulators
  gdd0(:) = 0.0_wp
  gdd5(:) = 0.0_wp

  do i = 1, npoi
    ! Coldest and warmest monthly temperatures in year 0 (deg C)
    tc(i) = minval(xint(i, :))
    tw(i) = maxval(xint(i, :))
    tcmin(i) = tc(i) + deltat(i)
  end do

  ! Interpolate monthly climatological values to daily resolution and
  ! accumulate growing degree days
  do i = 1, npoi
    do k = 1, 12
      rwork = 1.0_wp / real(ndaypm(k), wp)
      do lda = 1, ndaypm(k)
        if (real(lda, wp) < real(ndaypm(k) + 1, wp) * 0.5_wp) then
          it1w = k - 1
          it2w = k
          dt   = (real(lda, wp) - 0.5_wp) * rwork + 0.5_wp
        else
          it1w = k
          it2w = k + 1
          dt   = (real(lda, wp) - 0.5_wp) * rwork - 0.5_wp
        end if
        if (it1w < 1)  it1w = 12
        if (it2w > 12) it2w = 1
        dtemp = xint(i, it1w) + dt * (xint(i, it2w) - xint(i, it1w))
        gdd0(i) = gdd0(i) + max(0.0_wp, dtemp)
        gdd5(i) = gdd5(i) + max(0.0_wp, dtemp - 5.0_wp)
      end do
    end do
  end do

  call existence()

end subroutine climanl


! -----------------------------------------------------------------------------
subroutine climanl2()
! Updates the 30-year e-folding running averages of growing degree days,
! coldest monthly temperature, and warmest monthly temperature.
! Called once per year after the annual summary.
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_dimensions,       only: npoi
  use mod_boundary_climate, only: deltat
  use mod_veg_dynamics,     only: tc, tw, tcmin, gdd0, gdd5, &
                                   gdd0this, gdd5this, tcthis, twthis
  implicit none

  integer  :: i
  real(wp) :: zweigc, zweigw   ! 30-year e-folding weights
  real(wp) :: rworkc, rworkw   ! complementary weights (1 - zweig)

  zweigc = exp(-1.0_wp / 30.0_wp)
  zweigw = exp(-1.0_wp / 30.0_wp)
  rworkc = 1.0_wp - zweigc
  rworkw = 1.0_wp - zweigw

  do i = 1, npoi
    tc(i) = zweigc * tc(i) + rworkc * tcthis(i)
    tw(i) = zweigw * tw(i) + rworkw * twthis(i)
    tcmin(i) = tc(i) + deltat(i)
    gdd0(i) = zweigc * gdd0(i) + rworkc * gdd0this(i)
    gdd5(i) = zweigc * gdd5(i) + rworkc * gdd5this(i)
  end do

  call existence()

end subroutine climanl2


! -----------------------------------------------------------------------------
subroutine existence()
! Determines which plant functional types (PFTs) are allowed to exist in each
! grid cell based on climate criteria read from params.veg.
!
! PFT indices:
!  1  tropical broadleaf evergreen trees
!  2  tropical broadleaf drought-deciduous trees
!  3  warm-temperate broadleaf evergreen trees
!  4  temperate conifer evergreen trees
!  5  temperate broadleaf cold-deciduous trees
!  6  boreal conifer evergreen trees
!  7  boreal broadleaf cold-deciduous trees
!  8  boreal conifer cold-deciduous trees
!  9  evergreen shrubs
! 10  deciduous shrubs
! 11  warm (C4) grasses
! 12  cool (C3) grasses
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi
  use mod_veg_dynamics, only: exist, tcmin, gdd5, gdd0, tw
  use mod_pft_params,   only: TminL, TminU, Twarm, GDD
  implicit none

  integer :: i

  do i = 1, npoi

    exist(i, 1:12) = 0.0_wp

    ! 1) tropical broadleaf evergreen trees  (tcmin > TminL(1))
    if (tcmin(i) > TminL(1)) exist(i, 1) = 1.0_wp

    ! 2) tropical broadleaf drought-deciduous trees
    if (tcmin(i) > TminL(2)) exist(i, 2) = 1.0_wp

    ! 3) warm-temperate broadleaf evergreen trees
    if (tcmin(i) < TminU(3) .and. tcmin(i) > TminL(3)) exist(i, 3) = 1.0_wp

    ! 4) temperate conifer evergreen trees
    if (tcmin(i) < TminU(4) .and. tcmin(i) > TminL(4) .and. &
        gdd5(i) > GDD(4))                                      exist(i, 4) = 1.0_wp

    ! 5) temperate broadleaf cold-deciduous trees
    if (tcmin(i) < TminU(5) .and. tcmin(i) > TminL(5) .and. &
        gdd5(i) > GDD(5))                                      exist(i, 5) = 1.0_wp

    ! 6) boreal conifer evergreen trees
    if ((tcmin(i) < TminU(6) .or. gdd5(i) < GDD(4)) .and. &
         tcmin(i) > TminL(6) .and. gdd5(i) > GDD(6))         exist(i, 6) = 1.0_wp

    ! 7) boreal broadleaf cold-deciduous trees
    if ((tcmin(i) < TminU(7) .or. gdd5(i) < GDD(5)) .and. &
         tcmin(i) > TminL(7) .and. gdd5(i) > GDD(7))         exist(i, 7) = 1.0_wp

    ! 8) boreal conifer cold-deciduous trees
    if ((tcmin(i) < TminU(8) .or. gdd5(i) < TminL(4)) .and. &
         gdd5(i) > GDD(8))                                     exist(i, 8) = 1.0_wp

    ! 9) evergreen shrubs
    if (gdd0(i) > GDD(9))  exist(i, 9)  = 1.0_wp

    ! 10) deciduous shrubs
    if (gdd0(i) > GDD(10)) exist(i, 10) = 1.0_wp

    ! 11) warm (C4) grasses
    if (tw(i) > Twarm(11) .and. gdd0(i) > GDD(11)) exist(i, 11) = 1.0_wp

    ! 12) cool (C3) grasses
    if (gdd0(i) > GDD(12)) exist(i, 12) = 1.0_wp

  end do

end subroutine existence
