! =============================================================================
! snow.f90  --  Snow-layer dynamics and heat conduction for IBIS
!
! Converted from snow.f (Fortran 77 fixed-form) to F2008 free-form.
!
! Subroutines provided:
!   snow()                                    -- advance snow one timestep
!   snowheat(tlsub, fhtop, sflo, xl, chl)    -- implicit snow heat conduction
!   vadapt(hcur, tcur, htop, indp, np, nlay) -- re-adapt snow layer profiles
! =============================================================================


! -----------------------------------------------------------------------------
subroutine snow()
! Steps the snow model through one timestep.
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi, nsnolay
  use mod_constants,     only: ch2o, hfus, cice, tmelt, epsilon
  use mod_run_params,    only: dtime, vzero
  use mod_snow_state,    only: fi, tsno, hsno, rhos, hsnotop, hsnomin, &
                                fimin, fimax
  use mod_soil_state,    only: heati, heatg
  use mod_canopy_params, only: chl, tlsub, tl, wliql, wsnol
  use mod_veg_dynamics,  only: fl, ztop, lai, sai
  use mod_canopy_fluxes, only: rainl, snowl, fvapi, trainl, tsnowl, &
                                snowg, tsnowg, solg, fvapg, raing, traing
  implicit none

  integer  :: i, k, npn
  real(wp) :: rwork, rwork2, finew, zhh, zdh
  integer  :: indsno(npoi)
  real(wp) :: hinit(nsnolay)
  real(wp) :: hsnoruf(npoi)
  real(wp) :: fiold(npoi)
  real(wp) :: fhtop(npoi)
  real(wp) :: sflo(npoi, nsnolay+2)
  real(wp) :: zmelt(npoi)
  real(wp) :: zheat(npoi)
  real(wp) :: dfi(npoi)
  real(wp) :: xl(npoi)
  real(wp) :: xh(npoi)
  real(wp) :: xm(npoi)
  real(wp) :: ht(npoi)
  real(wp) :: x1(npoi)
  real(wp) :: x2(npoi)
  real(wp) :: x3(npoi)

  do i = 1, npoi
    hsnoruf(i) = min(0.70_wp, max(hsnomin + 0.05_wp, fl(i)*ztop(i,1)))
    xl(i) = fl(i) * 2.0_wp * (lai(i,1) + sai(i,1))
  end do

  hinit(1) = hsnotop
  do k = 2, nsnolay
    hinit(k) = (hsnomin - hsnotop) / real(nsnolay - 1, wp)
  end do

  fiold = fi

  ! zero out arrays
  sflo  = 0.0_wp
  zmelt = 0.0_wp
  zheat = 0.0_wp

  ! build index of points with snow
  npn = 0
  do i = 1, npoi
    if (fi(i) > 0.0_wp) then
      npn = npn + 1
      indsno(npn) = i
    end if
  end do

  ! set surface heat flux fhtop and increment top layer thickness
  ! due to rainfall, snowfall, and sublimation on existing snow
  if (npn > 0) then
    rwork = dtime / rhos
    do i = 1, npoi
      fhtop(i) = heati(i)                                          + &
                 rainl(i) * (ch2o * (trainl(i) - tmelt)  + hfus   + &
                             cice * (tmelt     - tsno(i,1)))       + &
                 snowl(i) *  cice * (tsnowl(i) - tsno(i,1))
      if (fi(i) > 0.0_wp) &
        hsno(i,1) = hsno(i,1) + (rainl(i) + snowl(i) - fvapi(i)) * rwork
    end do
  end if

  ! step temperatures due to heat conduction, including buried
  ! lower-veg temperature tlsub
  if (npn > 0) call snowheat(tlsub, fhtop, sflo, xl, chl)

  ! put snowfall from 1-fi snow-free area onto existing snow,
  ! or create new snow if fi = 0
  ht = 0.0_wp
  do k = 1, nsnolay
    do i = 1, npoi
      ht(i) = ht(i) + hsno(i,k)
    end do
  end do

  do i = 1, npoi
    if (ht(i) == 0.0_wp) ht(i) = hsnomin
  end do

  rwork = dtime / rhos
  do i = 1, npoi
    dfi(i) = (1.0_wp - fi(i)) * rwork * snowg(i) / ht(i)
    dfi(i) = min(dfi(i), 1.0_wp - fi(i))
  end do

  do k = 1, nsnolay
    do i = 1, npoi
      if (fi(i) + dfi(i) > 0.0_wp) &
        tsno(i,k) = (tsno(i,k)*fi(i) + tsnowg(i)*dfi(i)) &
                  / (fi(i) + dfi(i))
      ! set initial thicknesses for newly created snow
      if (fi(i) == 0.0_wp .and. dfi(i) > 0.0_wp) hsno(i,k) = hinit(k)
    end do
  end do

  npn = 0
  do i = 1, npoi
    fi(i) = fi(i) + dfi(i)
    if (fi(i) > 0.0_wp) then
      npn = npn + 1
      indsno(npn) = i
    end if
  end do

  ! melt from any layer; reduce thicknesses and give extra heat to soil
  if (npn > 0) then
    rwork = 1.0_wp / rhos
    do k = 1, nsnolay
      do i = 1, npoi
        xh(i) = rhos*hsno(i,k)*cice * max(tsno(i,k) - tmelt, 0.0_wp)
        xm(i) = min(rhos*hsno(i,k), xh(i)/hfus)
        hsno(i,k) = hsno(i,k) - xm(i)*rwork
        tsno(i,k) = min(tsno(i,k), tmelt)
        zmelt(i)  = zmelt(i) + fi(i)*xm(i)
        zheat(i)  = zheat(i) + fi(i)*(xh(i) - hfus*xm(i))
      end do
    end do

    ! adjust fi and thicknesses for coverage-vs-volume relation
    ht = 0.0_wp
    do k = 1, nsnolay
      do i = 1, npoi
        ht(i) = ht(i) + hsno(i,k)
      end do
    end do

    ! linear variation for 0 < fi < 1
    zhh = 0.5_wp * hsnomin
    do i = 1, npoi
      zdh   = hsnoruf(i) - hsnomin
      finew = (-zhh + sqrt(zhh**2 + zdh*fi(i)*ht(i))) / zdh
      finew = max(0.0_wp, min(fimax, finew))
      x1(i) = fi(i) / max(finew, epsilon)
      fi(i) = finew
    end do

    do k = 1, nsnolay
      do i = 1, npoi
        hsno(i,k) = hsno(i,k) * x1(i)
      end do
    end do
  end if

  ! re-adapt snow thickness profile so top thickness = hsnotop and
  ! other thicknesses are equal; adjusts temperature to conserve heat
  call vadapt(hsno, tsno, hsnotop, indsno, npn, nsnolay)

  ! if fi is below fimin, melt all remaining snow and adjust fluxes
  if (npn > 0) then
    x1 = fi
    do k = 1, nsnolay
      do i = 1, npoi
        if (x1(i) < fimin) then
          xm(i)     = x1(i) * rhos * hsno(i,k)
          zmelt(i)  = zmelt(i) + xm(i)
          zheat(i)  = zheat(i) - xm(i)*(cice*(tmelt - tsno(i,k)) + hfus)
          hsno(i,k) = 0.0_wp
          tsno(i,k) = tmelt
          fi(i)     = 0.0_wp
        end if
      end do
    end do
  end if

  ! adjust buried lower veg for fi changes
  if (npn > 0) then
    do i = 1, npoi
      dfi(i) = fi(i) - fiold(i)

      if (dfi(i) > 0.0_wp) then
        tlsub(i) = (tlsub(i)*fiold(i) + tl(i)*dfi(i)) / fi(i)
        zheat(i) = zheat(i) + dfi(i)*xl(i)                        &
                 * ( wliql(i) * (ch2o*(tl(i) - tmelt) + hfus      &
                                 + cice*(tmelt - tsno(i,nsnolay))) &
                   + wsnol(i) *  cice*(tl(i) - tsno(i,nsnolay)) )
        hsno(i,nsnolay) = hsno(i,nsnolay)                         &
                        + dfi(i)*xl(i)*(wliql(i) + wsnol(i))      &
                          / (rhos*fi(i))
      end if

      if (dfi(i) < 0.0_wp) then
        zheat(i) = zheat(i) - dfi(i)*xl(i)*chl*(tlsub(i) - tl(i))
        rwork    = (1.0_wp - fiold(i)) / (1.0_wp - fi(i))
        wliql(i) = wliql(i) * rwork
        wsnol(i) = wsnol(i) * rwork
      end if
    end do
  end if

  ! areally average fluxes for use by soil model
  if (npn > 0) then
    rwork = 1.0_wp / dtime
    do i = 1, npoi
      rwork2   = 1.0_wp - fiold(i)
      heatg(i) = rwork2*heatg(i)            &
               + fiold(i)*sflo(i,nsnolay+2) &
               + zheat(i)*rwork
      solg(i)  = rwork2 * solg(i)
      fvapg(i) = rwork2 * fvapg(i)
      x1(i)    = rwork2 * raing(i)
      x2(i)    = zmelt(i) * rwork
      x3(i)    = tmelt
    end do
    call mix(raing, traing, x1, traing, x2, x3, vzero, vzero)
  end if

end subroutine snow


! -----------------------------------------------------------------------------
subroutine snowheat(tlsub, fhtop, sflo, xl, chl)
! Sets up and solves the implicit snow heat conduction equations using a
! tridiagonal system.  Appends an extra buried-lower-veg layer to the bottom
! of the snow column with conduction coefficient conbur/xl and heat capacity
! chl*xl.
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nsnolay
  use mod_constants,  only: cice
  use mod_run_params, only: dtime
  use mod_snow_state, only: fi, tsno, hsno, rhos, consno
  use mod_soil_state, only: hsoi, consoi, tsoi
  implicit none

  real(wp), intent(inout) :: tlsub(npoi)           !< buried lower-veg temperature (K)
  real(wp), intent(in)    :: fhtop(npoi)            !< heat flux into top snow layer (W m-2)
  real(wp), intent(out)   :: sflo(npoi, nsnolay+2)  !< downward heat flux at boundaries (W m-2)
  real(wp), intent(in)    :: xl(npoi)               !< lower-veg density (lai+sai)*fl
  real(wp), intent(in)    :: chl                    !< heat capacity of lower veg per unit l/s area

  integer  :: k, i, km1, kp1
  real(wp) :: rwork, dt, dti

  !> Implicit fraction (1.0 = fully implicit).
  real(wp), parameter :: rimp   = 1.0_wp
  !> Conduction coefficient for unit density buried veg layer (W m-2 K-1).
  !> Chosen equivalent to 10 cm of snow.
  real(wp), parameter :: conbur = 2.0_wp
  !> Small thickness to allow processing for zero-snow points (m).
  real(wp), parameter :: hfake  = 0.01_wp

  real(wp) :: con(npoi, nsnolay+2)
  real(wp) :: temp(npoi, nsnolay+1)
  real(wp) :: d1(npoi, nsnolay+1)
  real(wp) :: d2(npoi, nsnolay+1)
  real(wp) :: d3(npoi, nsnolay+1)
  real(wp) :: rhs(npoi, nsnolay+1)
  real(wp) :: w1(npoi, nsnolay+1)
  real(wp) :: w2(npoi, nsnolay+1)

  ! copy snow and buried-lower-veg temperatures into combined array temp
  temp(:, 1:nsnolay) = tsno
  temp(:, nsnolay+1) = tlsub

  ! set conduction coefficients between layers
  do k = 1, nsnolay+2
    if (k == 1) then
      con(:, k) = 0.0_wp

    else if (k <= nsnolay) then
      rwork = 0.5_wp / consno
      do i = 1, npoi
        con(i,k) = 1.0_wp / (  max(hsno(i,k-1), hfake)*rwork &
                              + max(hsno(i,k),   hfake)*rwork )
      end do

    else if (k == nsnolay+1) then
      rwork = 0.5_wp / consno
      do i = 1, npoi
        con(i,k) = 1.0_wp / (  max(hsno(i,k-1), hfake)*rwork &
                              + 0.5_wp*xl(i)/conbur )
      end do

    else   ! k == nsnolay+2
      rwork = 0.5_wp / conbur
      do i = 1, npoi
        con(i,k) = 1.0_wp / (  xl(i)*rwork &
                              + 0.5_wp*hsoi(1) / consoi(i,1) )
      end do
    end if
  end do

  ! set matrix diagonals and right-hand side
  do k = 1, nsnolay+1
    km1 = max(k-1, 1)
    kp1 = min(k+1, nsnolay+1)

    if (k <= nsnolay) then
      rwork = dtime / (rhos*cice)
      do i = 1, npoi
        dt      = rwork / max(hsno(i,k), hfake)
        d1(i,k) =           - dt*rimp* con(i,k)
        d2(i,k) = 1.0_wp + dt*rimp*(con(i,k) + con(i,k+1))
        d3(i,k) =           - dt*rimp* con(i,k+1)
        rhs(i,k) = temp(i,k) + dt                                      &
                 * ( (1.0_wp-rimp)*con(i,k)  *(temp(i,km1)-temp(i,k)) &
                   + (1.0_wp-rimp)*con(i,k+1)*(temp(i,kp1)-temp(i,k)) )
      end do

      ! add top-surface flux to the top layer right-hand side
      if (k == 1) then
        rwork = dtime / (rhos*cice)
        do i = 1, npoi
          dt = rwork / max(hsno(i,k), hfake)
          rhs(i,k) = rhs(i,k) + dt*fhtop(i)
        end do
      end if

    else   ! k == nsnolay+1: buried-lower-veg layer
      rwork = chl / dtime
      do i = 1, npoi
        dti      = xl(i)*rwork
        d1(i,k)  =       -  rimp* con(i,k)
        d2(i,k)  = dti  +   rimp*(con(i,k) + con(i,k+1))
        d3(i,k)  = 0.0_wp
        rhs(i,k) = dti*temp(i,k)                                             &
                 + ( (1.0_wp-rimp)*con(i,k)*(temp(i,km1) - temp(i,k))       &
                   + con(i,k+1)*(tsoi(i,1) - (1.0_wp-rimp)*temp(i,k)) )
      end do
    end if
  end do

  ! solve the tridiagonal systems
  call tridia(npoi, npoi, nsnolay+1, d1, d2, d3, rhs, temp, w1, w2)

  ! deduce downward heat fluxes between layers
  sflo(:, 1) = fhtop

  do k = 1, nsnolay+1
    if (k <= nsnolay) then
      rwork = rhos*cice / dtime
      do i = 1, npoi
        sflo(i,k+1) = sflo(i,k) - rwork*hsno(i,k) &
                                  *(temp(i,k) - tsno(i,k))
      end do
    else
      rwork = chl / dtime
      do i = 1, npoi
        sflo(i,k+1) = sflo(i,k) &
                    - xl(i)*rwork*(temp(i,nsnolay+1) - tlsub(i))
      end do
    end if
  end do

  ! copy temperature solution back to tsno and tlsub (snow points only)
  do k = 1, nsnolay
    do i = 1, npoi
      if (fi(i) > 0.0_wp) tsno(i,k) = temp(i,k)
    end do
  end do

  do i = 1, npoi
    if (fi(i) > 0.0_wp) tlsub(i) = temp(i,nsnolay+1)
  end do

end subroutine snowheat


! -----------------------------------------------------------------------------
subroutine vadapt(hcur, tcur, htop, indp, np, nlay)
! Re-adapts snow layer thicknesses so the top layer has thickness htop and
! the remaining layers are equal.  Adjusts tcur to conserve its vertical
! integral (e.g., temperature × thickness).
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nsnolay
  use mod_constants,  only: epsilon
  implicit none

  integer,  intent(in)    :: np              !< number of snow points in strip
  integer,  intent(in)    :: nlay            !< number of layers
  integer,  intent(in)    :: indp(npoi)      !< indices of snow points
  real(wp), intent(in)    :: htop            !< prescribed top layer thickness (m)
  real(wp), intent(inout) :: hcur(npoi,nlay) !< layer thicknesses (m)
  real(wp), intent(inout) :: tcur(npoi,nlay) !< tracer field (e.g., temperature)

  integer  :: i, j, k, ko
  real(wp) :: dz, rwork
  real(wp) :: ht(npoi)
  real(wp) :: h1(npoi)
  real(wp) :: za(npoi)
  real(wp) :: zb(npoi)
  real(wp) :: zheat(npoi)
  real(wp) :: hnew(npoi, nsnolay)
  real(wp) :: tnew(npoi, nsnolay)
  real(wp) :: zold(npoi, nsnolay+1)

  if (np == 0) return

  ! set distances zold from surface to old layer boundaries
  zold(:, 1) = 0.0_wp
  do k = 1, nlay
    do i = 1, npoi
      zold(i,k+1) = zold(i,k) + hcur(i,k)
    end do
  end do

  ! set new layer thicknesses hnew (total thickness is unchanged).
  ! if total thickness < nlay*htop, make all new layers equal including
  ! the top one; use epsilon to handle zero (no-snow) points.
  ht    = zold(:, nlay+1)
  rwork = real(nlay, wp) * htop

  do i = 1, npoi
    if (ht(i) >= rwork) then
      h1(i) = (ht(i) - htop) / real(nlay - 1, wp)
    else
      h1(i) = max(ht(i) / real(nlay, wp), epsilon)
    end if
  end do

  do k = 1, nlay
    hnew(:, k) = h1
  end do

  do i = 1, npoi
    if (ht(i) >= rwork) hnew(i,1) = htop
  end do

  ! integrate old temperature profile over each new layer to get tnew
  zb = 0.0_wp

  do k = 1, nlay
    za = zb
    zb = za + hnew(:, k)
    zheat = 0.0_wp

    do ko = 1, nlay
      do i = 1, npoi
        if (za(i) < zold(i,ko+1) .and. zb(i) > zold(i,ko)) then
          dz = min(zold(i,ko+1), zb(i)) - max(zold(i,ko), za(i))
          zheat(i) = zheat(i) + tcur(i,ko)*dz
        end if
      end do
    end do

    do i = 1, npoi
      tnew(i,k) = zheat(i) / hnew(i,k)
    end do
  end do

  ! copy new thicknesses and temperatures to output arrays (snow points only)
  do k = 1, nlay
    do j = 1, np
      i = indp(j)
      hcur(i,k) = hnew(i,k)
      tcur(i,k) = tnew(i,k)
    end do
  end do

end subroutine vadapt
