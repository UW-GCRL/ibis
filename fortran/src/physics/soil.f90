! =============================================================================
! soil.f90  --  Soil thermal and hydraulic model for IBIS
!
! Converted from soil.f (Fortran 77 fixed-form) to F2008 free-form.
! Replaces all include directives with use associations to Phase 1–3 modules.
!
! Subroutines:
!   setsoi    -- set diagnostic soil quantities (conductivity, qglif, latent heat)
!   soilctl   -- main soil timestep controller (puddles, transpiration, freeze/thaw)
!   soilh2o   -- implicit soil moisture solver (calls tridia)
!   soilheat  -- implicit soil heat conduction solver (calls tridia)
!   wadjust   -- bounds-enforcement for wsoi and wisoi
! =============================================================================


! -----------------------------------------------------------------------------
subroutine setsoi
! -----------------------------------------------------------------------------
! Sets diagnostic soil quantities:
!   - layer thermal conductivity (consoi)
!   - evaporation source fractions (qglif)
!   - latent heat values at ground and snow surfaces (hvasug, hvasui)
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi, nsoilay
  use mod_constants,    only: epsilon, tmelt
  use mod_soil_state,   only: sand, clay, poros, wsoi, wisoi, consoi, &
                               qglif, zwpmax, wpud, wipud, wpudmax, tsoi, &
                               hvasug, hvasui
  use mod_snow_state,   only: tsno
  use mod_atm_state,    only: ta
  use mod_thermo_funcs, only: hvapf, hsubf
  implicit none

  integer  :: i, k, msand, mclay
  real(wp) :: fsand, fsilt, fclay, powliq, powice, zcondry
  real(wp) :: zvap, zsub, zwpud, zwsoi, rwork1, rwork2

  ! ---- set thermal conductivity per soil layer ------------------------------
  do k = 1, nsoilay
    do i = 1, npoi
      msand = nint(sand(i,k))
      mclay = nint(clay(i,k))

      fsand = 0.01_wp * real(msand, wp)
      fclay = 0.01_wp * real(mclay, wp)
      fsilt = 0.01_wp * real(100 - msand - mclay, wp)

      powliq = poros(i,k) * wsoi(i,k) * (1.0_wp - wisoi(i,k))
      powice = poros(i,k) * wisoi(i,k)

      zcondry = fsand * 0.300_wp + fsilt * 0.265_wp + fclay * 0.250_wp

      consoi(i,k) = zcondry * ((0.56_wp * 100.0_wp)**powliq) &
                             * ((2.24_wp * 100.0_wp)**powice)
    end do
  end do

  ! ---- set qglif: fraction of soil evap from liquid/ice/puddle sources -----
  !
  ! zwpud  : fraction of surface covered by puddle (0 to zwpmax)
  ! zwsoi  : volumetric water content of top layer (liquid + ice fraction)
  ! qglif(*,1): fraction from soil liquid
  ! qglif(*,2): fraction from soil ice
  ! qglif(*,3): fraction from puddle liquid
  ! qglif(*,4): fraction from puddle ice
  !
  do i = 1, npoi
    zwpud = max(0.0_wp, min(zwpmax, zwpmax*(wpud(i)+wipud(i))/wpudmax))
    zwsoi = (1.0_wp - wisoi(i,1)) * wsoi(i,1) + wisoi(i,1)

    if (zwsoi >= epsilon) then
      rwork1 = 1.0_wp / zwsoi
      if (zwpud >= epsilon) then
        rwork2       = 1.0_wp / (wpud(i) + wipud(i))
        qglif(i,1) = (1.0_wp - zwpud) * (1.0_wp - wisoi(i,1)) * wsoi(i,1) * rwork1
        qglif(i,2) = (1.0_wp - zwpud) * wisoi(i,1) * rwork1
        qglif(i,3) = zwpud * wpud(i)  * rwork2
        qglif(i,4) = zwpud * wipud(i) * rwork2
      else
        qglif(i,1) = (1.0_wp - wisoi(i,1)) * wsoi(i,1) * rwork1
        qglif(i,2) = wisoi(i,1) * rwork1
        qglif(i,3) = 0.0_wp
        qglif(i,4) = 0.0_wp
      end if
    else
      ! 100% dry soil: assign all evap to puddles
      if (zwpud >= epsilon) then
        rwork2     = 1.0_wp / (wpud(i) + wipud(i))
        qglif(i,1) = 0.0_wp
        qglif(i,2) = 0.0_wp
        qglif(i,3) = zwpud * wpud(i)  * rwork2
        qglif(i,4) = zwpud * wipud(i) * rwork2
      else
        if (tsoi(i,1) >= tmelt) then
          qglif(i,1) = 0.0_wp
          qglif(i,2) = 0.0_wp
          qglif(i,3) = 1.0_wp
          qglif(i,4) = 0.0_wp
        else
          qglif(i,1) = 0.0_wp
          qglif(i,2) = 0.0_wp
          qglif(i,3) = 0.0_wp
          qglif(i,4) = 1.0_wp
        end if
      end if
    end if

    ! ---- latent heat values at soil and snow surfaces ----------------------
    zvap = hvapf(tsoi(i,1), ta(i))
    zsub = hsubf(tsoi(i,1), ta(i))

    hvasug(i) = (qglif(i,1) + qglif(i,3)) * zvap &
              + (qglif(i,2) + qglif(i,4)) * zsub

    hvasui(i) = hsubf(tsno(i,1), ta(i))
  end do

end subroutine setsoi


! -----------------------------------------------------------------------------
subroutine soilctl
! -----------------------------------------------------------------------------
! Steps the soil model through one timestep:
!   (0) Excess puddle liquid → runoff
!   (1) Partition rain between puddle and runoff
!   (2) Partition evap/condensation among four water stores
!   (3) Transfer puddle liquid to infiltration
!   (4) Compute upper heat flux (fhtop)
!   Then: transpiration uptake, soilh2o, soilheat, wadjust,
!         heat-conserving freeze/thaw adjustment, puddle freeze/thaw.
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi, nsoilay
  use mod_constants,     only: epsilon, tmelt, ch2o, hfus, cice, rhow
  use mod_run_params,    only: dtime
  use mod_soil_state,    only: wpud, wipud, wpudmax, poros, wisoi, &
                                wsoi, hydraul, heatg, tsoi, porosflo, &
                                upsoiu, upsoil, csoi, rhosoi, hsoi, qglif
  use mod_hydrology,     only: grunof, gdrain, gadjust
  use mod_canopy_fluxes, only: raing, traing, fvapg
  use mod_canopy_params, only: tl, tu
  implicit none

  integer  :: i, k
  real(wp) :: zfrez, zrunf, rwork, wipre, zdpud, cx, chav, zwsoi

  ! local work arrays
  real(wp) :: owsoi(npoi, nsoilay), otsoi(npoi, nsoilay)
  real(wp) :: c0pud(npoi, nsoilay), c1pud(npoi, nsoilay)
  real(wp) :: wflo(npoi, nsoilay+1)
  real(wp) :: fwtop(npoi), fhtop(npoi), fwpud(npoi), fsqueez(npoi)
  real(wp) :: dh(npoi), dw(npoi), zporos(npoi)

  c0pud = 0.0_wp
  c1pud = 0.0_wp

  ! ---- puddle dynamics and upper boundary conditions -----------------------
  do i = 1, npoi

    ! (0) excess puddle liquid → runoff immediately
    grunof(i) = min(wpud(i), max(0.0_wp, wpud(i)+wipud(i)-wpudmax)) / dtime
    wpud(i)   = wpud(i) - grunof(i) * dtime

    ! (1) partition rain: fraction zrunf runs off, rest increases puddle
    zfrez = max(0.0_wp, min(1.0_wp, (tsoi(i,1) - tmelt + 0.5_wp) * 2.0_wp))
    zrunf = zfrez * max(0.0_wp, min(1.0_wp, (wpud(i)+wipud(i)) / wpudmax))
    wpud(i)   = wpud(i)   + (1.0_wp - zrunf) * raing(i) * dtime
    grunof(i) = grunof(i) + zrunf * raing(i)

    ! (2) partition evap/condensation among water stores
    rwork = fvapg(i) * dtime
    if (fvapg(i) >= 0.0_wp) then
      ! evaporation: split according to qglif
      fwtop(i)  =             - qglif(i,1) * fvapg(i)
      wpud(i)   = wpud(i)   - qglif(i,3) * rwork
      wipud(i)  = wipud(i)  - qglif(i,4) * rwork

      wipre      = wisoi(i,1)
      wisoi(i,1) = max(0.0_wp, wipre - qglif(i,2)*rwork &
                               / (rhow * poros(i,1) * hsoi(1)))
      if (1.0_wp - wisoi(i,1) > epsilon) &
        wsoi(i,1) = wsoi(i,1) * (1.0_wp - wipre) / (1.0_wp - wisoi(i,1))
    else
      ! condensation: give all to puddles (prevents wsoi,wisoi > 1)
      fwtop(i) = 0.0_wp
      wpud(i)  = wpud(i)  - (qglif(i,1)+qglif(i,3)) * rwork
      wipud(i) = wipud(i) - (qglif(i,2)+qglif(i,4)) * rwork
    end if

    ! (3) transfer puddle liquid to top-layer infiltration
    zdpud      = rhow * dtime * max(0.0_wp, 1.0_wp-wisoi(i,1))**2 * hydraul(i,1)
    fwpud(i)   = max(0.0_wp, min(wpud(i), zdpud)) / dtime
    c0pud(i,1) = ch2o*wpud(i) + cice*wipud(i)

    ! (4) compute upper soil heat flux
    fhtop(i) = heatg(i) &
             + raing(i)  * ch2o * (traing(i) - tsoi(i,1)) &
             - grunof(i) * ch2o * max(tmelt - tsoi(i,1), 0.0_wp)

    gadjust(i) = 0.0_wp
  end do

  ! ---- transpiration uptake, porosity update, save old fields -------------
  do k = 1, nsoilay
    do i = 1, npoi
      porosflo(i,k) = poros(i,k) * max(epsilon, 1.0_wp - wisoi(i,k))
      porosflo(i,k) = max(porosflo(i,k), epsilon)

      wsoi(i,k) = wsoi(i,k) - dtime * (upsoiu(i,k) + upsoil(i,k)) &
                               / (rhow * porosflo(i,k) * hsoi(k))

      cx = c0pud(i,k) + &
           (  (1.0_wp - poros(i,k)) * csoi(i,k) * rhosoi(i,k)           &
            + poros(i,k) * (1.0_wp - wisoi(i,k)) * wsoi(i,k) * ch2o * rhow &
            + poros(i,k) * wisoi(i,k) * cice * rhow                     &
           ) * hsoi(k)

      tsoi(i,k) = tsoi(i,k) - dtime * ch2o * &
                  (  upsoiu(i,k) * (tu(i) - tsoi(i,k)) &
                   + upsoil(i,k) * (tl(i) - tsoi(i,k)) ) / cx

      owsoi(i,k) = wsoi(i,k)
      otsoi(i,k) = tsoi(i,k)
    end do
  end do

  ! ---- implicit soil moisture step -----------------------------------------
  call soilh2o(owsoi, fwtop, fwpud, fsqueez, wflo)

  ! ---- update drainage and puddle heat capacity ----------------------------
  do i = 1, npoi
    gdrain(i)  = wflo(i, nsoilay+1)
    c1pud(i,1) = ch2o*wpud(i) + cice*wipud(i)
  end do

  ! ---- implicit soil heat step ---------------------------------------------
  call soilheat(otsoi, owsoi, c0pud, fhtop, wflo, c1pud)

  ! ---- enforce bounds before phase adjustment ------------------------------
  call wadjust

  ! ---- heat-conserving liquid/ice adjustment (soil layers) -----------------
  do k = 1, nsoilay
    do i = 1, npoi
      zporos(i) = max(poros(i,k), epsilon)
      rwork = c1pud(i,k) / rhow / hsoi(k) &
            + (1.0_wp - zporos(i)) * csoi(i,k) * rhosoi(i,k) / rhow

      chav = rwork &
           + zporos(i) * (1.0_wp - wisoi(i,k)) * wsoi(i,k) * ch2o &
           + zporos(i) * wisoi(i,k) * cice

      ! freeze liquid below melting point
      zwsoi  = min(1.0_wp, wsoi(i,k))
      dh(i)  = chav * (tmelt - tsoi(i,k))
      dw(i)  = min(zporos(i) * (1.0_wp - wisoi(i,k)) * zwsoi, &
                   max(0.0_wp, dh(i) / hfus))
      wisoi(i,k) = wisoi(i,k) + dw(i) / zporos(i)
      wsoi(i,k)  = wsoi(i,k)  - (dw(i)/zporos(i)) * (1.0_wp - zwsoi) &
                               / max(epsilon, 1.0_wp - wisoi(i,k))

      chav = rwork &
           + zporos(i) * (1.0_wp - wisoi(i,k)) * wsoi(i,k) * ch2o &
           + zporos(i) * wisoi(i,k) * cice
      tsoi(i,k) = tmelt - (dh(i) - hfus*dw(i)) / chav

      ! melt ice above melting point
      dh(i) = chav * (tsoi(i,k) - tmelt)
      dw(i) = min(zporos(i) * wisoi(i,k), max(0.0_wp, dh(i) / hfus))
      wisoi(i,k) = wisoi(i,k) - dw(i) / zporos(i)
      wsoi(i,k)  = wsoi(i,k)  + (dw(i)/zporos(i)) * (1.0_wp - wsoi(i,k)) &
                               / max(epsilon, 1.0_wp - wisoi(i,k))

      chav = rwork &
           + zporos(i) * (1.0_wp - wisoi(i,k)) * wsoi(i,k) * ch2o &
           + zporos(i) * wisoi(i,k) * cice
      tsoi(i,k) = tmelt + (dh(i) - hfus*dw(i)) / chav

      porosflo(i,k) = zporos(i) * max(epsilon, 1.0_wp - wisoi(i,k))
    end do
  end do

  call wadjust

  ! ---- puddle liquid/ice adjustment (upper layer only) --------------------
  k = 1
  do i = 1, npoi
    rwork = (  (1.0_wp - poros(i,k)) * csoi(i,k) * rhosoi(i,k)              &
             + poros(i,k) * (1.0_wp - wisoi(i,k)) * wsoi(i,k) * ch2o * rhow &
             + poros(i,k) * wisoi(i,k) * cice * rhow                         &
            ) * hsoi(k)

    ! freeze puddle liquid below tmelt
    chav  = ch2o*wpud(i) + cice*wipud(i) + rwork
    dh(i) = chav * (tmelt - tsoi(i,k))
    dw(i) = min(wpud(i), max(0.0_wp, dh(i) / hfus))
    wipud(i)  = wipud(i) + dw(i)
    wpud(i)   = wpud(i)  - dw(i)
    chav = ch2o*wpud(i) + cice*wipud(i) + rwork
    tsoi(i,k) = tmelt - (dh(i) - hfus*dw(i)) / chav

    ! melt puddle ice above tmelt
    dh(i) = chav * (tsoi(i,k) - tmelt)
    dw(i) = min(wipud(i), max(0.0_wp, dh(i) / hfus))
    wipud(i)  = wipud(i) - dw(i)
    wpud(i)   = wpud(i)  + dw(i)
    chav = ch2o*wpud(i) + cice*wipud(i) + rwork
    tsoi(i,k) = tmelt + (dh(i) - hfus*dw(i)) / chav
  end do

end subroutine soilctl


! -----------------------------------------------------------------------------
subroutine soilh2o(owsoi, fwtop, fwpud, fsqueez, wflo)
! -----------------------------------------------------------------------------
! Sets up and solves the implicit soil moisture equation via tridia.
! Lower boundary: bperm=0 (impermeable) or bperm=1 (free drainage).
! Updates wsoi, wpud; returns wflo (drainage flux profile) and fsqueez.
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nsoilay
  use mod_constants,  only: epsilon, rhow
  use mod_run_params, only: dtime
  use mod_soil_state, only: hsoi, hydraul, suction, bex, ibex, porosflo, &
                             poros, wisoi, wsoi, wpud, bperm
  implicit none

  real(wp), intent(in)    :: owsoi(npoi, nsoilay)
  real(wp), intent(in)    :: fwtop(npoi), fwpud(npoi)
  real(wp), intent(out)   :: fsqueez(npoi)
  real(wp), intent(inout) :: wflo(npoi, nsoilay+1)

  integer :: k, i, km1, kka, kkb
  integer :: m(npoi), n(npoi)

  ! dmin and rimp are fixed parameters (never changed at runtime)
  real(wp), parameter :: dmin = 1.0e-9_wp  ! minimum diffusivity (m^2 s-1)
  real(wp), parameter :: rimp = 1.0_wp     ! implicit fraction

  real(wp) :: zbex, z, dt, zz
  real(wp) :: hsoim(nsoilay+1)
  real(wp) :: wsoim(npoi, nsoilay+1), wsoia(npoi, nsoilay+1), &
              wsoib(npoi, nsoilay+1)
  real(wp) :: weim(npoi, nsoilay+1), weip(npoi, nsoilay+1)
  real(wp) :: a(npoi), b(npoi), bwn(npoi), bwn1(npoi)
  real(wp) :: e(npoi, nsoilay+1), f(npoi, nsoilay+1), g(npoi, nsoilay+1)
  real(wp) :: d1(npoi, nsoilay), d2(npoi, nsoilay), d3(npoi, nsoilay)
  real(wp) :: rhs(npoi, nsoilay), w1(npoi, nsoilay), w2(npoi, nsoilay)

  ! ---- interpolation weights and boundary moistures: top layer -------------
  k = 1
  hsoim(k) = 0.5_wp * hsoi(k)
  do i = 1, npoi
    weip(i,k)  = 0.0_wp
    weim(i,k)  = 1.0_wp
    wsoim(i,k) = wsoi(i,k)
    wsoia(i,k) = min(wsoim(i,k), 1.0_wp)
    wsoib(i,k) = min(wsoim(i,k), 1.0_wp)
  end do

  ! middle layers
  do k = 2, nsoilay
    hsoim(k) = 0.5_wp * (hsoi(k-1) + hsoi(k))
    do i = 1, npoi
      weip(i,k)  = 0.5_wp * hsoi(k) / hsoim(k)
      weim(i,k)  = 1.0_wp - weip(i,k)
      wsoim(i,k) = weim(i,k)*wsoi(i,k-1) + weip(i,k)*wsoi(i,k)
      wsoia(i,k) = min(wsoim(i,k), 1.0_wp)
      wsoib(i,k) = min(wsoim(i,k), 1.0_wp)
    end do
  end do

  ! bottom (nsoilay+1) layer
  k = nsoilay + 1
  hsoim(k) = 0.5_wp * hsoi(k-1)
  do i = 1, npoi
    weip(i,k)  = 1.0_wp
    weim(i,k)  = 0.0_wp
    wsoim(i,k) = wsoi(i,k-1)
    wsoia(i,k) = min(wsoim(i,k), 1.0_wp)
    wsoib(i,k) = min(wsoim(i,k), 1.0_wp)
  end do

  ! ---- flux terms e, f, g --------------------------------------------------
  ! top layer interface: zero (no flux above surface)
  e(:, 1) = 0.0_wp
  f(:, 1) = 0.0_wp
  g(:, 1) = 0.0_wp

  ! middle layer interfaces
  do k = 2, nsoilay
    do i = 1, npoi
      a(i) = weim(i,k)*hydraul(i,k-1) + weip(i,k)*hydraul(i,k)
      b(i) = weim(i,k)*hydraul(i,k-1)*suction(i,k-1)*bex(i,k-1) &
           + weip(i,k)*hydraul(i,k  )*suction(i,k  )*bex(i,k  )
      zbex = weim(i,k)*bex(i,k-1) + weip(i,k)*bex(i,k)
      m(i) = 2*nint(zbex) + 3
      n(i) =   nint(zbex) + 2

      bwn1(i) = b(i) * (wsoib(i,k)**(n(i)-1))
      bwn(i)  = bwn1(i) * wsoib(i,k)
      if (bwn(i) < dmin) bwn1(i) = 0.0_wp
      bwn(i) = max(bwn(i), dmin)

      e(i,k) = (-1.0_wp + rimp*m(i)) * a(i) * (wsoia(i,k)**m(i))       &
             + ((1.0_wp-rimp)*bwn(i) - rimp*real(n(i),wp)*bwn1(i)*wsoib(i,k)) &
               * (wsoi(i,k) - wsoi(i,k-1)) / hsoim(k)
      f(i,k) = -rimp*real(m(i),wp)*a(i)*(wsoia(i,k)**(m(i)-1))         &
              + rimp*real(n(i),wp)*bwn1(i)*(wsoi(i,k)-wsoi(i,k-1)) / hsoim(k)
      g(i,k) = rimp * bwn(i)
    end do
  end do

  ! bottom interface: lower boundary condition
  k = nsoilay + 1
  do i = 1, npoi
    a(i)   = hydraul(i,nsoilay)
    b(i)   = hydraul(i,nsoilay) * suction(i,nsoilay) * real(ibex(i,nsoilay), wp)
    m(i)   = 2*ibex(i,nsoilay) + 3
    n(i)   = ibex(i,nsoilay) + 2
    e(i,k) = -a(i) * (wsoia(i,k)**m(i)) * bperm
    f(i,k) = 0.0_wp
    g(i,k) = 0.0_wp
  end do

  ! ---- scale e,f,g by ice-barrier factor -----------------------------------
  do k = 1, nsoilay+1
    kka = max(k-1, 1)
    kkb = min(k, nsoilay)
    do i = 1, npoi
      z      = max(0.0_wp, 1.0_wp - max(wisoi(i,kka), wisoi(i,kkb)))**2
      e(i,k) = z * e(i,k)
      f(i,k) = z * f(i,k)
      g(i,k) = z * g(i,k)
    end do
  end do

  ! ---- tridiagonal matrix diagonals and right-hand sides ------------------
  do k = 1, nsoilay
    do i = 1, npoi
      dt       = dtime / (porosflo(i,k) * hsoi(k))
      d1(i,k)  = dt * (f(i,k)*0.5_wp*hsoi(k)/hsoim(k) - g(i,k)/hsoim(k))
      rhs(i,k) = wsoi(i,k) + dt * (e(i,k+1) - e(i,k))
    end do

    if (k == 1) then
      do i = 1, npoi
        dt = dtime / (porosflo(i,k) * hsoi(k))
        rhs(i,k) = rhs(i,k) + dt * (fwtop(i) + fwpud(i)) / rhow
      end do
    end if

    km1 = max(k-1, 1)
    if (k < nsoilay) then
      do i = 1, npoi
        dt = dtime / (porosflo(i,k) * hsoi(k))
        d2(i,k) = 1.0_wp + dt * ( -f(i,k+1)*0.5_wp*hsoi(k+1)/hsoim(k+1)  &
                                   + f(i,k)  *0.5_wp*hsoi(km1)/hsoim(k)   &
                                   + g(i,k+1)/hsoim(k+1)                   &
                                   + g(i,k)  /hsoim(k) )
        d3(i,k) = dt * (-f(i,k+1)*0.5_wp*hsoi(k)/hsoim(k+1) &
                        - g(i,k+1)/hsoim(k+1))
      end do
    else   ! k == nsoilay
      do i = 1, npoi
        dt = dtime / (porosflo(i,k) * hsoi(k))
        d2(i,k) = 1.0_wp + dt * (-f(i,k+1)                          &
                                  + f(i,k)*0.5_wp*hsoi(km1)/hsoim(k) &
                                  + g(i,k)/hsoim(k))
        d3(i,k) = 0.0_wp
      end do
    end if
  end do

  ! ---- solve the tridiagonal system ----------------------------------------
  call tridia(npoi, npoi, nsoilay, d1, d2, d3, rhs, wsoi, w1, w2)

  ! ---- compute drainage flux and squeeze excess moisture ------------------
  do i = 1, npoi
    fsqueez(i)          = 0.0_wp
    wflo(i, nsoilay+1) = -rhow * e(i, nsoilay+1)
  end do

  do k = nsoilay, 1, -1
    do i = 1, npoi
      zz = rhow * poros(i,k) * max(epsilon, 1.0_wp - wisoi(i,k)) * hsoi(k)
      wsoi(i,k)  = wsoi(i,k) + dtime * fsqueez(i) / zz
      fsqueez(i) = max(wsoi(i,k) - 1.0_wp, 0.0_wp) * zz / dtime
      wsoi(i,k)  = min(wsoi(i,k), 1.0_wp)
      wflo(i,k)  = wflo(i,k+1) + (wsoi(i,k) - owsoi(i,k)) * zz / dtime
    end do
  end do

  ! ---- update puddle liquid due to squeeze and infiltration ---------------
  do i = 1, npoi
    wpud(i)   = wpud(i)   + (fsqueez(i) - fwpud(i)) * dtime
    wflo(i,1) = wflo(i,1) + (fsqueez(i) - fwpud(i))
  end do

end subroutine soilh2o


! -----------------------------------------------------------------------------
subroutine soilheat(otsoi, owsoi, c0pud, fhtop, wflo, c1pud)
! -----------------------------------------------------------------------------
! Sets up and solves the implicit soil heat conduction equation via tridia.
! Accounts for heat fluxes due to liquid water movement from soilh2o.
! Deduces layer heat fluxes hflo from the solution.
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nsoilay
  use mod_constants,  only: ch2o, rhow, cice
  use mod_run_params, only: dtime
  use mod_soil_state, only: hsoi, poros, wisoi, wsoi, consoi, tsoi, hflo, &
                             csoi, rhosoi
  implicit none

  real(wp), intent(in) :: otsoi(npoi, nsoilay), owsoi(npoi, nsoilay)
  real(wp), intent(in) :: c0pud(npoi, nsoilay), c1pud(npoi, nsoilay)
  real(wp), intent(in) :: fhtop(npoi)
  real(wp), intent(in) :: wflo(npoi, nsoilay+1)

  integer :: k, i, km1, kp1
  real(wp), parameter :: rimp = 1.0_wp   ! fully implicit

  real(wp) :: rwork, rwork1, rwork2, rwork3, t

  real(wp) :: whflo(npoi, nsoilay+1), con(npoi, nsoilay+1)
  real(wp) :: c0(npoi, nsoilay), c1(npoi, nsoilay)
  real(wp) :: d1(npoi, nsoilay), d2(npoi, nsoilay), d3(npoi, nsoilay)
  real(wp) :: rhs(npoi, nsoilay), w1(npoi, nsoilay), w2(npoi, nsoilay)

  ! ---- conduction coefficients and water-transport heat fluxes ------------
  ! top layer
  k = 1
  do i = 1, npoi
    con(i,k)   = 0.0_wp
    whflo(i,k) = wflo(i,k) * ch2o * tsoi(i,k)
  end do

  ! middle layers
  do k = 2, nsoilay
    do i = 1, npoi
      con(i,k) = 1.0_wp / (0.5_wp * (hsoi(k-1)/consoi(i,k-1) + &
                                      hsoi(k)  /consoi(i,k)))
      t = (hsoi(k)*tsoi(i,k-1) + hsoi(k-1)*tsoi(i,k)) / (hsoi(k-1) + hsoi(k))
      whflo(i,k) = wflo(i,k) * ch2o * t
    end do
  end do

  ! bottom layer
  k = nsoilay + 1
  do i = 1, npoi
    con(i,k)   = 0.0_wp
    whflo(i,k) = wflo(i,k) * ch2o * tsoi(i,k-1)
  end do

  ! ---- matrix diagonals and right-hand sides --------------------------------
  do k = 1, nsoilay
    km1 = max(k-1, 1)
    kp1 = min(k+1, nsoilay)
    do i = 1, npoi
      rwork1 = (1.0_wp - poros(i,k)) * csoi(i,k) * rhosoi(i,k)
      rwork2 = poros(i,k) * (1.0_wp - wisoi(i,k)) * ch2o * rhow
      rwork3 = poros(i,k) * wisoi(i,k) * cice * rhow

      c0(i,k) = c0pud(i,k) + (rwork1 + rwork2*owsoi(i,k) + rwork3) * hsoi(k)
      c1(i,k) = c1pud(i,k) + (rwork1 + rwork2*wsoi(i,k)  + rwork3) * hsoi(k)

      rwork = dtime / c1(i,k)

      d1(i,k) =          - rwork * rimp * con(i,k)
      d2(i,k) = 1.0_wp   + rwork * rimp * (con(i,k) + con(i,k+1))
      d3(i,k) =          - rwork * rimp * con(i,k+1)

      rhs(i,k) = (c0(i,k)/c1(i,k)) * tsoi(i,k) + rwork &
               * (  (1.0_wp - rimp)*con(i,k)  *(tsoi(i,km1) - tsoi(i,k)) &
                  + (1.0_wp - rimp)*con(i,k+1)*(tsoi(i,kp1) - tsoi(i,k)) &
                  + whflo(i,k) - whflo(i,k+1) )
    end do

    if (k == 1) then
      do i = 1, npoi
        rhs(i,k) = rhs(i,k) + (dtime / c1(i,k)) * fhtop(i)
      end do
    end if
  end do

  ! ---- solve the tridiagonal system ----------------------------------------
  call tridia(npoi, npoi, nsoilay, d1, d2, d3, rhs, tsoi, w1, w2)

  ! ---- deduce downward heat fluxes hflo between layers --------------------
  hflo(1:npoi, 1) = fhtop(1:npoi)
  do k = 1, nsoilay
    do i = 1, npoi
      hflo(i,k+1) = hflo(i,k) &
                  - (c1(i,k)*tsoi(i,k) - c0(i,k)*otsoi(i,k)) / dtime
    end do
  end do

end subroutine soilheat


! -----------------------------------------------------------------------------
subroutine wadjust
! -----------------------------------------------------------------------------
! Adjusts wsoi and wisoi to exactly 0 when they differ from 0 by less than
! epsilon, and clamps both to [0,1].  Diagnoses the total mass adjustment
! (gadjust) for budget tracking.
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nsoilay
  use mod_constants,  only: epsilon, rhow
  use mod_run_params, only: dtime
  use mod_soil_state, only: hsoi, poros, wsoi, wisoi
  use mod_hydrology,  only: gadjust
  implicit none

  integer  :: k, i
  real(wp) :: ztot0, ztot1

  do k = 1, nsoilay
    do i = 1, npoi
      ztot0 = hsoi(k) * poros(i,k) * rhow * &
              ((1.0_wp - wisoi(i,k)) * wsoi(i,k) + wisoi(i,k))

      if (wsoi(i,k)  < epsilon) wsoi(i,k)  = 0.0_wp
      if (wisoi(i,k) < epsilon) wisoi(i,k) = 0.0_wp

      wsoi(i,k)  = min(1.0_wp, wsoi(i,k))
      wisoi(i,k) = min(1.0_wp, wisoi(i,k))

      if (wisoi(i,k) >= 1.0_wp - epsilon) wsoi(i,k) = 0.0_wp

      ztot1 = hsoi(k) * poros(i,k) * rhow * &
              ((1.0_wp - wisoi(i,k)) * wsoi(i,k) + wisoi(i,k))

      gadjust(i) = gadjust(i) + (ztot1 - ztot0) / dtime
    end do
  end do

end subroutine wadjust
