! =============================================================================
! weather.f90  --  Daily and diurnal weather generation routines for IBIS
!
! Converted from weather.f (Fortran 77 fixed-form) to F2008 free-form.
!
! Subroutines provided:
!   daily(iyear,imonth,iday,seed,jdaily,iyranom,iyrlast,nrun)
!          -- generate daily weather from monthly climatology
!   diurnal(time,jday,plens,startp,endp,seed)
!          -- disaggregate daily weather to sub-daily timestep
! =============================================================================


! -----------------------------------------------------------------------------
subroutine daily(iyear, imonth, iday, seed, jdaily, iyranom, iyrlast, nrun)
! Generates daily weather conditions from monthly-mean climatological parameters
! using a Richardson-type stochastic weather generator (when jdaily=0) or
! reads pre-computed daily anomalies (when jdaily=1).
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_dimensions,       only: npoi
  use mod_constants,        only: grav, rair
  use mod_run_params,       only: ndaypm
  use mod_boundary_climate, only: xint, xinq, xinprec, xinwind, xincld, &
                                   xinwet, xintrng, xintopo,             &
                                   xinprecd, xintd, xintrngd, xincldd,  &
                                   xinqd, xinwindd
  use mod_atm_state,        only: td, tmax, tmin, cloud, precip, psurf, &
                                   qd, ud, iwet,                         &
                                   iwetday, iwetdaysum,                  &
                                   precipday, precipdaysum, xstore
  use mod_veg_dynamics,     only: tcthis, twthis, gdd0this, gdd5this
  use mod_summary,          only: a10td
  use mod_thermo_funcs,     only: esat, qsat
  implicit none

  integer, intent(in)    :: iyear, imonth, iday
  integer, intent(inout) :: seed
  integer, intent(in)    :: jdaily, iyranom, iyrlast, nrun

  !> Richardson autocorrelation matrices (constant, column-major fill).
  ! Declared real (float32) to match F77 weather.f where these are 'real'.
  real, parameter :: a(3,3) = reshape( &
    [0.600, 0.500,  0.005, &
     0.010, 0.250,  0.005, &
     0.020, 0.125,  0.250], [3,3])
  real, parameter :: b(3,3) = reshape( &
    [0.500, 0.250, -0.250, &
     0.000, 0.500,  0.250, &
     0.000, 0.000,  0.500], [3,3])

  ! ran2 returns default real (single precision) to match the F77 original.
  ! All stochastic-path variables that feed into rejection tests must also be
  ! default real so that the gamma and Box-Muller acceptance decisions are
  ! bit-identical to the F77 run (see math.f90 for the rationale).
  real, external :: ran2

  integer :: it1w, it2w, i, j, k
  ! All local variables are real (float32) to match F77 weather.f, which
  ! declares everything with implicit real (or explicit 'real').  Any
  ! module-level array (xint, xinwet, xstore, grav, rair, …) that is
  ! real(wp) must be cast with real() at the point of use so that all
  ! intermediate arithmetic stays in float32, matching the F77 results.
  real :: rwork, omcloud, omqd, omtmax, dt
  real :: pwet, pwd, pww, rndnum, rainpwd, alpha, beta, aa, ab
  real :: tr1, tr2, rn1, rn2, rn3, rn, s1, s2, s12, z
  real :: v
  real :: tdm, trngm, tmaxm, tminm, tmaxd, tmaxw, tmaxe, tmine
  real :: tmaxs, tmins, cloudm, cloudd, cloudw, cloude, clouds
  real :: tdum, qdm, qdd, qdw, qde, qdup, qdlow, y, b3, b2, b1, x1
  real :: amn, eud, precipfac, dif, humidfrac, sphumid
  real :: ee(3), r(3), rr(3), x(3)
  logical :: too_big

  ! Exponent for hypsometric pressure formula.
  ! Cast grav/rair to real so the division is float32, matching F77.
  rwork = real(grav) / real(rair) / 0.0065

  ! Wet/dry day omega parameters (EPIC weather generator style)
  omcloud = 0.90
  omqd    = 0.50
  omtmax  = 0.75

  ! Interpolation weights for monthly → daily climatological values
  if (jdaily == 0) then
    if (real(iday) < real(ndaypm(imonth) + 1) * 0.5) then
      it1w = imonth - 1
      it2w = imonth
      dt   = (real(iday) - 0.5) / real(ndaypm(imonth)) + 0.5
    else
      it1w = imonth
      it2w = imonth + 1
      dt   = (real(iday) - 0.5) / real(ndaypm(imonth)) - 0.5
    end if
    if (it1w < 1)  it1w = 12
    if (it2w > 12) it2w = 1
  end if

  ! Reset annual extremes on 1 January
  if (iday == 1 .and. imonth == 1) then
    tcthis   =  100.0_wp
    twthis   = -100.0_wp
    gdd0this =    0.0_wp
    gdd5this =    0.0_wp
  end if

  ! -------------------------------------------------------------------------
  ! Loop over all grid points
  ! -------------------------------------------------------------------------
  do i = 1, npoi

    if (jdaily == 0) then

      ! -----------------------------------------------------------------------
      ! Weather-generator branch
      ! -----------------------------------------------------------------------

      pwet = max(1.0, real(xinwet(i, imonth))) / real(ndaypm(imonth))
      pwd  = 0.75 * pwet
      pww  = 0.25 + pwd

      ! Generate precipitation sequence for entire month on the first day
      if (iday == 1) then

        xinprec(i, imonth) = max(0.0_wp, xinprec(i, imonth))
        xinwet(i, imonth)  = max(1.0_wp, xinwet(i, imonth))

        ! Retry loop: regenerate if any single-day amount exceeds 360 mm
        retry: do

          iwetdaysum(i)   = 0
          precipdaysum(i) = 0.0_wp
          iwetday(i, :)   = 0
          precipday(i, :) = 0.0_wp

          ! Decide wet/dry and precipitation amount for each day in the month
          do j = 1, ndaypm(imonth)

            rndnum = ran2(seed)

            if (j == 1) then
              if (rndnum <= pwd) then
                iwetday(i, j)   = 1
                iwetdaysum(i)   = iwetdaysum(i) + 1
              end if
            else
              if (iwetday(i, j-1) == 0) then
                if (rndnum <= pwd) then
                  iwetday(i, j) = 1
                  iwetdaysum(i) = iwetdaysum(i) + 1
                end if
              else
                ! BUG FIX 2026-05-14: original code was
                !   `if (rndnum > pww) iwetday(i, j) = 0`
                ! which never sets iwetday=1 in the wet->wet branch.
                ! Corrected to standard Markov: today wet with prob pww.
                if (rndnum <= pww) then
                  iwetday(i, j) = 1
                  iwetdaysum(i) = iwetdaysum(i) + 1
                end if
              end if
            end if

            if (iwetday(i, j) == 1) then
              rainpwd = real(xinprec(i, imonth)) * real(ndaypm(imonth)) / &
                        max(0.1, real(xinwet(i, imonth)))
              ! All float32 arithmetic to match F77 math.f (rainpwd, beta, etc. are real).
              beta  = max(1.0, -2.16 + 1.83 * rainpwd)
              alpha = rainpwd / beta
              aa    = 1.0 / alpha
              ab    = 1.0 / (1.0 - alpha)
              tr1   = exp(-18.42 / aa)
              tr2   = exp(-18.42 / ab)

              ! Rejection-sampling gamma deviate
              do
                rn1 = ran2(seed)
                rn2 = ran2(seed)
                if (rn1 - tr1 <= 0.0) then
                  s1 = 0.0
                else
                  s1 = rn1**aa
                end if
                if (rn2 - tr2 <= 0.0) then
                  s2 = 0.0
                else
                  s2 = rn2**ab
                end if
                s12 = s1 + s2
                if (s12 <= 1.0) exit
              end do

              z = s1 / s12
              rn3 = ran2(seed)
              precipday(i, j) = -z * log(rn3) * beta
              precipday(i, j) = max(precipday(i, j), 0.25_wp)
              precipday(i, j) = min(precipday(i, j), 150.0_wp)
            end if

            precipdaysum(i) = precipdaysum(i) + precipday(i, j)

          end do   ! j = 1, ndaypm

          ! Handle degenerate case (no rain despite non-zero target)
          if (precipdaysum(i) == 0.0_wp .and. xinprec(i, imonth) > 0.0_wp) then
            rndnum = 1.0 + real(ndaypm(imonth) - 1) * ran2(seed)
            iwetday(i, nint(rndnum))   = 1
            precipday(i, nint(rndnum)) = real(xinprec(i, imonth)) * real(ndaypm(imonth))
            precipdaysum(i) = precipday(i, nint(rndnum))
            iwetdaysum(i)   = 1
          end if

          precipfac = (real(xinprec(i, imonth)) * real(ndaypm(imonth))) / &
                       max(0.01, real(precipdaysum(i)))

          too_big = .false.
          do j = 1, ndaypm(imonth)
            precipday(i, j) = precipday(i, j) * precipfac
            if (precipday(i, j) > 360.0_wp) then
              if (real(xinwet(i, imonth)) < real(ndaypm(imonth))) then
                xinwet(i, imonth) = xinwet(i, imonth) + 1.0_wp
                pwet = real(xinwet(i, imonth)) / real(ndaypm(imonth))
                pwd  = 0.75 * pwet
                pww  = 0.25 + pwd
                print *, 'WARNING: retry (a)', i, int(xinwet(i,imonth)), &
                          iwetdaysum(i), int(precipday(i,j))
              else
                print *, 'WARNING: retry (b)', i, int(xinwet(i,imonth)), &
                          iwetdaysum(i), int(precipday(i,j))
              end if
              too_big = .true.
              exit
            end if
          end do

          if (.not. too_big) exit retry

        end do retry   ! regenerate month

        ! Verification
        iwetdaysum(i)   = 0
        precipdaysum(i) = 0.0_wp
        do j = 1, ndaypm(imonth)
          precipdaysum(i) = precipdaysum(i) + precipday(i, j)
          iwetdaysum(i)   = iwetdaysum(i)   + iwetday(i, j)
        end do
        dif = real(precipdaysum(i)) - real(xinprec(i, imonth)) * real(ndaypm(imonth))
        if (dif < -0.1 .or. dif > 0.1) then
          print *, 'ERROR in DAILY:', i, precipdaysum(i), &
                    xinprec(i,imonth) * real(ndaypm(imonth), wp), &
                    iwetdaysum(i), xinwet(i,imonth)
        end if

      end if   ! iday == 1

      iwet(i)   = iwetday(i, iday)
      precip(i) = precipday(i, iday)

      ! -------------------------------------------------------------------
      ! (3) Expected minimum and maximum temperatures
      ! -------------------------------------------------------------------
      ! Cast xint/xintrng to real so interpolation is float32, matching F77.
      tdm   = real(xint(i, it1w)) + dt * (real(xint(i, it2w))    - real(xint(i, it1w)))    + 273.16
      trngm = real(xintrng(i, it1w)) + dt * (real(xintrng(i, it2w)) - real(xintrng(i, it1w)))

      tmaxm = tdm + 0.56 * trngm
      tminm = tdm - 0.44 * trngm

      if (pwet /= 0.0) then
        tmaxd = tmaxm + pwet * omtmax * trngm
        tmaxw = tmaxd -        omtmax * trngm
      else
        tmaxd = tmaxm
        tmaxw = tmaxm
      end if

      if (iwet(i) == 0) tmaxe = tmaxd
      if (iwet(i) == 1) tmaxe = tmaxw
      tmine = tminm

      tmaxs = max(1.0, -0.0713 * (tdm - 273.16) + 4.89)
      tmins = max(1.0, -0.1280 * (tdm - 273.16) + 5.73)

      ! -------------------------------------------------------------------
      ! (4) Expected cloud cover
      ! -------------------------------------------------------------------
      if (iday == 1) then
        xincld(i, it1w) = max(0.0_wp, min(100.0_wp, xincld(i, it1w)))
        xincld(i, it2w) = max(0.0_wp, min(100.0_wp, xincld(i, it2w)))
      end if

      ! Cast xincld to real so interpolation is float32, matching F77.
      cloudm = (real(xincld(i, it1w)) + dt * (real(xincld(i, it2w)) - real(xincld(i, it1w)))) / 100.0

      if (pwet /= 0.0) then
        cloudd = (cloudm - pwet * omcloud) / (1.0 - pwet * omcloud)
        cloudd = min(1.0, max(0.0, cloudd))
        cloudw = (cloudm - (1.0 - pwet) * cloudd) / pwet
      else
        cloudd = cloudm
        cloudw = cloudm
      end if

      if (iwet(i) == 0) cloude = cloudd
      if (iwet(i) == 1) cloude = cloudw
      if (iwet(i) == 0) clouds = 0.24 * cloude
      if (iwet(i) == 1) clouds = 0.48 * cloude

      ! -------------------------------------------------------------------
      ! (5) Richardson multivariate weather generator
      ! -------------------------------------------------------------------
      ! Box-Muller: all float32 arithmetic (rn1, rn2, v are real) to match F77.
      do j = 1, 3
        do
          rn1 = ran2(seed)
          rn2 = ran2(seed)
          v   = sqrt(-2.0 * log(rn1)) * cos(6.283185 * rn2)
          if (abs(v) <= 2.5) exit
        end do
        ee(j) = v
      end do

      r  = 0.0
      rr = 0.0
      do j = 1, 3
        do k = 1, 3
          r(j)  = r(j)  + b(j, k) * ee(j)
          rr(j) = rr(j) + a(j, k) * real(xstore(i, k))
        end do
      end do

      do j = 1, 3
        x(j)       = r(j) + rr(j)
        xstore(i,j) = x(j)
      end do

      tmax(i) = tmaxe + tmaxs * x(1)
      tmin(i) = tmine + tmins * x(2)
      if (real(tmin(i)) > real(tmax(i))) then
        tdum    = real(tmax(i))
        tmax(i) = tmin(i)
        tmin(i) = tdum
      end if
      ! Cast tmax/tmin to real for float32 arithmetic, matching F77.
      td(i) = 0.44 * real(tmax(i)) + 0.56 * real(tmin(i))

      cloud(i) = max(0.0, min(1.0, cloude + clouds * x(3)))

      ! -------------------------------------------------------------------
      ! (6) Surface atmospheric pressure
      ! -------------------------------------------------------------------
      ! Cast td/xintopo to real so the hypsometric formula is float32, matching F77.
      psurf(i) = 101325.0 * (real(td(i)) / (real(td(i)) + 0.0065 * real(xintopo(i))))**rwork

      ! -------------------------------------------------------------------
      ! (7) Relative humidity → specific humidity
      ! -------------------------------------------------------------------
      if (iday == 1) then
        xinq(i, it1w) = max(0.0_wp, min(100.0_wp, xinq(i, it1w)))
        xinq(i, it2w) = max(0.0_wp, min(100.0_wp, xinq(i, it2w)))
      end if

      ! Cast xinq to real so interpolation is float32, matching F77.
      qdm = (real(xinq(i, it1w)) + dt * (real(xinq(i, it2w)) - real(xinq(i, it1w)))) / 100.0

      if (pwet /= 0.0) then
        qdd = (qdm - pwet * omqd) / (1.0 - pwet * omqd)
        if (qdd < 0.2) then
          qdd = 0.2
          if (qdd > qdm) qdm = qdd
        end if
        qdd = min(1.0, qdd)
        qdw = (qdm - (1.0 - pwet) * qdd) / pwet
      else
        qdd = qdm
        qdw = qdm
      end if

      if (iwet(i) == 0) qde = qdd
      if (iwet(i) == 1) qde = qdw

      qdup  = qde + (1.0 - qde) * exp(qde - 1.0)
      qdlow = qde * (1.0 - exp(-qde))

      rn = ran2(seed)
      y  = 2.0 / (qdup - qdlow)
      b3 = qde  - qdlow
      b2 = qdup - qde
      b1 = rn / y
      x1 = y * b3 / 2.0

      if (rn > x1) then
        qd(i) = qdup - sqrt(b2*b2 - 2.0*b2*(b1 - 0.5*b3))
      else
        qd(i) = qdlow + sqrt(2.0 * b1 * b3)
      end if

      amn   = (qdup + qde + qdlow) / 3.0
      qd(i) = max(0.30, min(0.99, real(qd(i)) * qde / amn))
      qd(i) = real(qd(i)) * real(qsat(esat(td(i)), psurf(i)))

      ! -------------------------------------------------------------------
      ! (8) Daily average wind speed
      ! -------------------------------------------------------------------
      ! Cast xinwind to real so interpolation is float32, matching F77.
      eud   = real(xinwind(i, it1w)) + dt * (real(xinwind(i, it2w)) - real(xinwind(i, it1w)))
      ud(i) = max(2.5, min(10.0, 1.13989 * eud * (-log(ran2(seed)))**0.30))

    else

      ! -----------------------------------------------------------------------
      ! Daily-data branch (jdaily == 1): apply pre-computed NCEP anomalies
      ! -----------------------------------------------------------------------
      ! jdaily==1: cast module arrays to real so arithmetic is float32, matching F77.
      precip(i) = real(xinprec(i, imonth)) * real(ndaypm(imonth)) * real(xinprecd(i))

      td(i)  = real(xint(i, imonth)) + 273.16 + real(xintd(i))
      trngm  = min(44.0, real(xintrng(i, imonth)) * real(xintrngd(i)))
      tmax(i) = real(td(i)) + 0.56 * trngm
      tmin(i) = real(td(i)) - 0.44 * trngm

      cloud(i) = max(0.0, min(1.0, &
                  (real(xincld(i, imonth)) + real(xincldd(i))) * 0.01))

      rwork   = real(grav) / real(rair) / 0.0065
      psurf(i) = 101325.0 * (real(td(i)) / (real(td(i)) + 0.0065 * real(xintopo(i))))**rwork

      humidfrac = real(xinq(i, imonth)) / 100.0
      sphumid   = humidfrac * qsat(esat(td(i)), psurf(i))
      qd(i)    = real(sphumid) * real(xinqd(i))

      ud(i) = real(xinwind(i, imonth)) * real(xinwindd(i))

    end if

    ! -------------------------------------------------------------------------
    ! Update annual climate extremes and growing degree days
    ! -------------------------------------------------------------------------
    tcthis(i)   = min(tcthis(i),   a10td(i) - 273.16_wp)
    twthis(i)   = max(twthis(i),   a10td(i) - 273.16_wp)
    gdd0this(i) = gdd0this(i) + max(0.0_wp, td(i) - 273.16_wp)
    gdd5this(i) = gdd5this(i) + max(0.0_wp, td(i) - 278.16_wp)

  end do   ! i = 1, npoi

end subroutine daily


! -----------------------------------------------------------------------------
subroutine diurnal(time, jday, plens, startp, endp, seed)
! Disaggregates daily weather statistics into sub-daily (hourly) values.
! Computes solar radiation, temperature, humidity, wind, IR, and precipitation
! at the current model timestep.
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_dimensions,       only: npoi, nband
  use mod_constants,        only: pi, stef
  use mod_run_params,       only: dtime
  use mod_work,             only: latindex, latscale
  use mod_atm_state,        only: coszen, fira, ua, ta, qa, raina, rh, snowa, &
                                   psurf, cloud, tmax, tmin, qd, ud, precip,  &
                                   solad, solai
  use mod_thermo_funcs,     only: esat, qsat
  implicit none

  real(wp), intent(in)    :: time     !< current model time (s since midnight)
  integer,  intent(in)    :: jday     !< current day of year
  real(wp), intent(inout) :: plens    !< precipitation event length (s)
  real(wp), intent(inout) :: startp   !< precipitation start time (s)
  real(wp), intent(inout) :: endp     !< precipitation end time (s)
  integer,  intent(inout) :: seed

  real, external :: ran2

  integer  :: i, jj, ib
  real(wp) :: rtime, orbit, angle, xdecl, sw, xlat, trans, fdiffuse
  real(wp) :: frac, gamma, qmin, qmax, qsa, emb, ea, ec, dtair, dtcloud
  integer  :: checkP, niter, plen, plenmin, plenmax

  rtime = time / 3600.0_wp

  ! Earth's orbital angle (radians)
  orbit = 2.0_wp * pi * real(jday, wp) / 365.2425_wp

  ! Solar hour angle (radians)
  angle = 2.0_wp * pi * (rtime - 12.0_wp) / 24.0_wp

  ! Solar declination (Hartmann, appendix A)
  xdecl =  0.006918_wp                       &
          - 0.399912_wp * cos(orbit)          &
          + 0.070257_wp * sin(orbit)          &
          - 0.006758_wp * cos(2.0_wp * orbit) &
          + 0.000907_wp * sin(2.0_wp * orbit) &
          - 0.002697_wp * cos(3.0_wp * orbit) &
          + 0.001480_wp * sin(3.0_wp * orbit)

  ! Effective solar constant (Hartmann, appendix A)
  sw = 1370.0_wp * (1.000110_wp                        &
                  + 0.034221_wp * cos(orbit)            &
                  + 0.001280_wp * sin(orbit)            &
                  + 0.000719_wp * cos(2.0_wp * orbit)   &
                  + 0.000077_wp * sin(2.0_wp * orbit))

  ! Outer retry loop: redo grid-cell loop if precipitation event is rescaled
  restart: do

    do i = 1, npoi

      ! Solar zenith angle
      jj   = latindex(i)
      xlat = latscale(jj) * pi / 180.0_wp
      coszen(i) = max(0.0_wp, sin(xlat)*sin(xdecl) + cos(xlat)*cos(xdecl)*cos(angle))

      ! Atmospheric transmissivity and diffuse fraction (Friend, A.)
      trans    = 0.251_wp + 0.509_wp * (1.0_wp - cloud(i))
      fdiffuse = 1.0045_wp + 0.0435_wp * trans &
               - 3.5227_wp * trans**2           &
               + 2.6313_wp * trans**3
      if (trans > 0.75_wp) fdiffuse = 0.166_wp

      ! Direct and diffuse solar radiation by waveband
      do ib = 1, nband
        frac         = 0.46_wp + 0.08_wp * real(ib - 1, wp)
        solad(i, ib) = sw * coszen(i) * frac * trans * (1.0_wp - fdiffuse)
        solai(i, ib) = sw * coszen(i) * frac * trans * fdiffuse
      end do

      ! Hourly air temperature (Campbell & Norman fourier series)
      gamma  = 0.44_wp - 0.46_wp * sin(      pi / 12.0_wp * rtime + 0.9_wp) &
                        + 0.11_wp * sin(2.0_wp * pi / 12.0_wp * rtime + 0.9_wp)
      ta(i) = tmax(i) * gamma + tmin(i) * (1.0_wp - gamma)

      ! Hourly specific humidity (preserve daily mean, cap RH at 99%)
      qmin = min(qd(i), 0.99_wp * qsat(esat(tmin(i)), psurf(i)))
      qmax = (qd(i) - 0.56_wp * qmin) / 0.44_wp
      qsa  = 0.99_wp * qsat(esat(ta(i)), psurf(i))
      qa(i) = min(qsa, qmax * gamma + qmin * (1.0_wp - gamma))

      ! Hourly relative humidity (%)
      rh(i) = 100.0_wp * qa(i) / qsat(esat(ta(i)), psurf(i))

      ! Hourly wind speed (EPIC weather generator)
      ua(i) = max(2.5_wp, min(10.0_wp, &
               1.13989_wp * ud(i) * (-log(ran2(seed)))**0.30_wp))

      ! Downward IR flux (Idso 1981)
      emb   = 0.01_wp * (psurf(i) * qa(i) / (0.622_wp + qa(i)))
      ea    = 0.70_wp + 5.95e-5_wp * emb * exp(1500.0_wp / ta(i))
      ec    = 0.950_wp
      dtair   = 0.0_wp
      dtcloud = 0.0_wp
      fira(i) = (1.0_wp - cloud(i)) * ea * stef * (ta(i) - dtair  )**4 &
              +            cloud(i)  * ec * stef * (ta(i) - dtcloud)**4

      ! Reset precipitation partitioning
      snowa(i) = 0.0_wp
      raina(i) = 0.0_wp

      ! Adjust precipitation event length if rate is too high (first timestep)
      if (time < dtime) then
        niter   = int(86400.0_wp / dtime)
        plen    = int(plens / dtime)
        plenmin = 1 + int((4.0_wp * 3600.0_wp - 1.0_wp) / dtime)
        plenmax = max(int(24.0_wp * 3600.0_wp / dtime), plenmin)
        checkP  = 0

        do while ((precip(i) / real(plen, wp) > 15.0_wp) .and. (plen < plenmax))
          plen   = plen + 1
          checkP = 1
        end do

        if (checkP == 1) then
          plens  = dtime * real(plen, wp)
          startp = dtime * real(min(niter - plen, &
                    int(ran2(seed) * real(niter - plen + 1, wp))), wp)
          endp   = startp + real(plen, wp) * dtime
          cycle restart
        end if
      end if

      ! Partition precipitation into rain and snow
      if (time >= startp .and. time < endp) then
        if (ta(i) - 273.15_wp > 2.5_wp) then
          raina(i) = precip(i) / plens
        else
          snowa(i) = precip(i) / plens
        end if
      end if

    end do   ! i = 1, npoi

    exit restart
  end do restart

end subroutine diurnal
