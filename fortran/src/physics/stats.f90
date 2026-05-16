! =============================================================================
! stats.f90  --  Instantaneous, daily, monthly, and annual statistics
!
! Converted from stats.f (F77 fixed-form) to F2008 free-form.
! All COMMON blocks replaced by USE associations to Phase 1–3 modules.
!
! Subroutines:
!   sumnow            -- instantaneous NPP, GPP, NEE; 10-day running means
!   sumday(istep, plens) -- daily accumulated statistics
!   summonth(istep, iday, imonth) -- monthly averaged statistics
!   sumyear(istep, iday, imonth) -- annual accumulated statistics
!   gdiag(iyear, iyear0) -- global domain diagnostic printout
!   vdiag(iyear, iyear0) -- per-vegetation-type diagnostic printout
! =============================================================================

! -----------------------------------------------------------------------------
subroutine sumnow
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi, npft, nsoilay
  use mod_run_params,   only: ndaypy, dtime
  use mod_atm_state,    only: td
  use mod_snow_state,   only: fi
  use mod_soil_state,   only: tsoi
  use mod_canopy_params, only: ts, &
                                ancub, ancuc, ancls, ancl3, ancl4, &
                                agcub, agcuc, agcls, agcl3, agcl4
  use mod_veg_dynamics, only: frac, lai, fu, fl, &
                                tgpp, tnpp, tgpptot, tnpptot, &
                                tco2mic, tco2root, tneetot, &
                                sapfrac, froot
  use mod_carbon_pools, only: cbiow, cbior
  use mod_summary,      only: a10td, a10ancub, a10ancuc, a10ancls, &
                                a10ancl3, a10ancl4
  implicit none

  ! Local variables
  integer :: i, k
  real(wp) :: rwood, rroot, rgrowth
  real(wp) :: stemtemp, roottemp, funca, funcb
  real(wp) :: zweight, smask
  real(wp) :: nppdummy(npoi, npft)

  ! ---------------------------------------------------------------------------
  ! Define working variables
  ! ---------------------------------------------------------------------------

  ! Maintenance respiration coefficients (per second), converted from per year
  rwood   = 0.0125_wp / (real(ndaypy, wp) * 86400.0_wp) * (1000.0_wp / 12.0_wp)
  rroot   = 1.2500_wp / (real(ndaypy, wp) * 86400.0_wp) * (1000.0_wp / 12.0_wp)

  ! Growth respiration coefficient (fraction)
  rgrowth = 0.30_wp

  ! 10-day time averaging factor
  zweight = exp(-1.0_wp / (10.0_wp * 86400.0_wp / dtime))

  ! Begin global grid
  do i = 1, npoi

    ! Snow masking for lower canopy
    smask = 1.0_wp - fi(i)

    ! -------------------------------------------------------------------
    ! Instantaneous GPP (canopy NPP before stem/root respiration)
    ! -------------------------------------------------------------------

    nppdummy(i,1)  = frac(i,1)  * ancub(i) * lai(i,2) * fu(i)
    nppdummy(i,2)  = frac(i,2)  * ancub(i) * lai(i,2) * fu(i)
    nppdummy(i,3)  = frac(i,3)  * ancub(i) * lai(i,2) * fu(i)
    nppdummy(i,4)  = frac(i,4)  * ancuc(i) * lai(i,2) * fu(i)
    nppdummy(i,5)  = frac(i,5)  * ancub(i) * lai(i,2) * fu(i)
    nppdummy(i,6)  = frac(i,6)  * ancuc(i) * lai(i,2) * fu(i)
    nppdummy(i,7)  = frac(i,7)  * ancub(i) * lai(i,2) * fu(i)
    nppdummy(i,8)  = frac(i,8)  * ancub(i) * lai(i,2) * fu(i)
    nppdummy(i,9)  = frac(i,9)  * ancls(i) * lai(i,1) * fl(i) * smask
    nppdummy(i,10) = frac(i,10) * ancls(i) * lai(i,1) * fl(i) * smask
    nppdummy(i,11) = frac(i,11) * ancl4(i) * lai(i,1) * fl(i) * smask
    nppdummy(i,12) = frac(i,12) * ancl3(i) * lai(i,1) * fl(i) * smask

    tgpp(i,1)  = frac(i,1)  * agcub(i) * lai(i,2) * fu(i)
    tgpp(i,2)  = frac(i,2)  * agcub(i) * lai(i,2) * fu(i)
    tgpp(i,3)  = frac(i,3)  * agcub(i) * lai(i,2) * fu(i)
    tgpp(i,4)  = frac(i,4)  * agcuc(i) * lai(i,2) * fu(i)
    tgpp(i,5)  = frac(i,5)  * agcub(i) * lai(i,2) * fu(i)
    tgpp(i,6)  = frac(i,6)  * agcuc(i) * lai(i,2) * fu(i)
    tgpp(i,7)  = frac(i,7)  * agcub(i) * lai(i,2) * fu(i)
    tgpp(i,8)  = frac(i,8)  * agcub(i) * lai(i,2) * fu(i)
    tgpp(i,9)  = frac(i,9)  * agcls(i) * lai(i,1) * fl(i) * smask
    tgpp(i,10) = frac(i,10) * agcls(i) * lai(i,1) * fl(i) * smask
    tgpp(i,11) = frac(i,11) * agcl4(i) * lai(i,1) * fl(i) * smask
    tgpp(i,12) = frac(i,12) * agcl3(i) * lai(i,1) * fl(i) * smask

    tgpptot(i) = 0.0_wp
    do k = 1, npft
      tgpptot(i) = tgpptot(i) + tgpp(i,k)
    end do

    ! -------------------------------------------------------------------
    ! Temperature functions for respiration
    ! -------------------------------------------------------------------

    stemtemp = ts(i)

    roottemp = 0.0_wp
    do k = 1, nsoilay
      roottemp = roottemp + tsoi(i,k) * 0.5_wp * (froot(k,1) + froot(k,2))
    end do

    ! Respiration on a 15-degree base (Lloyd and Taylor)
    funca = exp(3500.0_wp * (1.0_wp / 288.16_wp - 1.0_wp / stemtemp))
    funcb = exp(3500.0_wp * (1.0_wp / 288.16_wp - 1.0_wp / roottemp))

    ! -------------------------------------------------------------------
    ! Instantaneous NPP
    ! -------------------------------------------------------------------

    tnpp(i,1)  = nppdummy(i,1)  - rwood * cbiow(i,1)  * sapfrac(i) * funca &
                                 - rroot * cbior(i,1)               * funcb

    tnpp(i,2)  = nppdummy(i,2)  - rwood * cbiow(i,2)  * sapfrac(i) * funca &
                                 - rroot * cbior(i,2)               * funcb

    tnpp(i,3)  = nppdummy(i,3)  - rwood * cbiow(i,3)  * sapfrac(i) * funca &
                                 - rroot * cbior(i,3)               * funcb

    tnpp(i,4)  = nppdummy(i,4)  - rwood * cbiow(i,4)  * sapfrac(i) * funca &
                                 - rroot * cbior(i,4)               * funcb

    tnpp(i,5)  = nppdummy(i,5)  - rwood * cbiow(i,5)  * sapfrac(i) * funca &
                                 - rroot * cbior(i,5)               * funcb

    tnpp(i,6)  = nppdummy(i,6)  - rwood * cbiow(i,6)  * sapfrac(i) * funca &
                                 - rroot * cbior(i,6)               * funcb

    tnpp(i,7)  = nppdummy(i,7)  - rwood * cbiow(i,7)  * sapfrac(i) * funca &
                                 - rroot * cbior(i,7)               * funcb

    tnpp(i,8)  = nppdummy(i,8)  - rwood * cbiow(i,8)  * sapfrac(i) * funca &
                                 - rroot * cbior(i,8)               * funcb

    ! Shrubs: sapfrac implicit = 1.0 (no sapfrac term shown, rwood*cbiow*funca only)
    tnpp(i,9)  = nppdummy(i,9)  - rwood * cbiow(i,9)               * funca &
                                 - rroot * cbior(i,9)               * funcb

    tnpp(i,10) = nppdummy(i,10) - rwood * cbiow(i,10)              * funca &
                                 - rroot * cbior(i,10)              * funcb

    ! C4 and C3 grasses: no woody biomass respiration
    tnpp(i,11) = nppdummy(i,11) - rroot * cbior(i,11)              * funcb

    tnpp(i,12) = nppdummy(i,12) - rroot * cbior(i,12)              * funcb

    ! Apply growth respiration and compute total NPP
    tnpptot(i) = 0.0_wp
    do k = 1, npft
      if (tnpp(i,k) > 0.0_wp) tnpp(i,k) = tnpp(i,k) * (1.0_wp - rgrowth)
      tnpptot(i) = tnpptot(i) + tnpp(i,k)
    end do

    ! -------------------------------------------------------------------
    ! Total fine-root respiration
    ! -------------------------------------------------------------------

    tco2root(i) = 0.0_wp
    do k = 1, npft
      tco2root(i) = tco2root(i) + rroot * cbior(i,k) * funcb
    end do

    ! -------------------------------------------------------------------
    ! Instantaneous NEE
    ! -------------------------------------------------------------------

    tneetot(i) = tnpptot(i) - tco2mic(i)

    ! -------------------------------------------------------------------
    ! Update 10-day running-mean parameters
    ! -------------------------------------------------------------------

    a10td(i)    = zweight * a10td(i)    + (1.0_wp - zweight) * td(i)
    a10ancub(i) = zweight * a10ancub(i) + (1.0_wp - zweight) * ancub(i)
    a10ancuc(i) = zweight * a10ancuc(i) + (1.0_wp - zweight) * ancuc(i)
    a10ancls(i) = zweight * a10ancls(i) + (1.0_wp - zweight) * ancls(i)
    a10ancl3(i) = zweight * a10ancl3(i) + (1.0_wp - zweight) * ancl3(i)
    a10ancl4(i) = zweight * a10ancl4(i) + (1.0_wp - zweight) * ancl4(i)

  end do

  return
end subroutine sumnow


! -----------------------------------------------------------------------------
subroutine sumday (istep, plens)
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi, nsoilay
  use mod_run_params,   only: dtime
  use mod_atm_state,    only: raina, snowa, rh, precip
  use mod_canopy_fluxes, only: fvapa
  use mod_hydrology,    only: grunof, gdrain
  use mod_snow_state,   only: fi, hsno
  use mod_soil_state,   only: tsoi, wsoi, wisoi, hsoi
  use mod_veg_dynamics, only: froot, tco2mic, tco2root, tnmin
  use mod_carbon_pools, only: decompl, decomps
  use mod_summary,      only: ndtimes, &
                                adrain, adsnow, adaet, &
                                adtrunoff, adsrunoff, addrainage, &
                                adrh, adsnod, adsnof, &
                                adwsoi, adtsoi, adwisoi, &
                                adtlaysoi, adwlaysoi, &
                                adwsoic, adtsoic, &
                                adco2mic, adco2root, adco2soi, adco2ratio, &
                                adnmintot
  implicit none

  ! Arguments
  integer,   intent(in) :: istep
  real(wp),  intent(in) :: plens   ! precipitation length (unused, kept for interface)

  ! Local variables
  integer  :: i, k, niter
  real(wp) :: rwork, rwork2, rwork3, rwork4
  real(wp) :: tconst, bconst, btemp
  real(wp) :: rdepth, rdepth2, snodpth
  real(wp) :: soiltemp, soilmois, soilice
  real(wp) :: soitempc, soimoisc
  real(wp) :: factor, wfps, moist
  real(wp) :: precipfac

  ! ---------------------------------------------------------------------------
  ! Update counters
  ! ---------------------------------------------------------------------------

  if (istep == 1) ndtimes = 0
  ndtimes = ndtimes + 1

  rwork  = 1.0_wp / real(ndtimes, wp)
  rwork2 = 86400.0_wp
  rwork3 = 86400.0_wp * 12.0e-3_wp
  rwork4 = 86400.0_wp * 14.0e-3_wp

  ! Constants for Lloyd and Taylor (1994) temperature function
  tconst = 344.00_wp
  btemp  = 288.16_wp
  bconst = 10.0_wp

  ! Soil weighting factors
  rdepth  = 1.0_wp / (hsoi(1) + hsoi(2) + hsoi(3) + hsoi(4))
  rdepth2 = 1.0_wp / (hsoi(1) + hsoi(2))

  ! Begin global grid
  do i = 1, npoi

    ! -------------------------------------------------------------------
    ! Daily water budget terms
    ! -------------------------------------------------------------------

    adrain(i) = ((real(ndtimes,wp)-1.0_wp) * adrain(i) + raina(i) * 86400.0_wp) * rwork
    adsnow(i) = ((real(ndtimes,wp)-1.0_wp) * adsnow(i) + snowa(i) * 86400.0_wp) * rwork

    ! Verification of weather generator algorithm
    niter = int(86400.0_wp / dtime)
    if (istep == niter) then
      precipfac = precip(i) - adrain(i) - adsnow(i)
      if ((precipfac < -0.1_wp) .or. (precipfac > 0.1_wp)) &
        print *, 'ERROR in sumday:', i, adrain(i) + adsnow(i), precip(i)
    end if

    adaet(i)      = ((real(ndtimes,wp)-1.0_wp) * adaet(i)      - fvapa(i) * 86400.0_wp) * rwork

    adtrunoff(i)  = ((real(ndtimes,wp)-1.0_wp) * adtrunoff(i)  + &
                     (grunof(i) + gdrain(i)) * 86400.0_wp) * rwork
    adsrunoff(i)  = ((real(ndtimes,wp)-1.0_wp) * adsrunoff(i)  + &
                      grunof(i)              * 86400.0_wp) * rwork
    addrainage(i) = ((real(ndtimes,wp)-1.0_wp) * addrainage(i) + &
                      gdrain(i)              * 86400.0_wp) * rwork

    ! -------------------------------------------------------------------
    ! Daily atmospheric terms
    ! -------------------------------------------------------------------

    adrh(i) = ((real(ndtimes,wp)-1.0_wp) * adrh(i) + rh(i)) * rwork

    ! -------------------------------------------------------------------
    ! Daily snow parameters
    ! -------------------------------------------------------------------

    snodpth   = hsno(i,1) + hsno(i,2) + hsno(i,3)
    adsnod(i) = ((real(ndtimes,wp)-1.0_wp) * adsnod(i) + snodpth) * rwork
    adsnof(i) = ((real(ndtimes,wp)-1.0_wp) * adsnof(i) + fi(i))   * rwork

    ! -------------------------------------------------------------------
    ! Soil parameters
    ! -------------------------------------------------------------------

    soiltemp = 0.0_wp
    soilmois = 0.0_wp
    soilice  = 0.0_wp
    soitempc = 0.0_wp
    soimoisc = 0.0_wp

    ! Averages for first 2 layers of soil
    do k = 1, 2
      soiltemp = soiltemp + tsoi(i,k)  * hsoi(k)
      soilmois = soilmois + wsoi(i,k)  * hsoi(k)
      soilice  = soilice  + wisoi(i,k) * hsoi(k)
    end do

    soilmois = soilmois * rdepth2
    soilice  = soilice  * rdepth2
    soiltemp = soiltemp * rdepth2

    ! Root-profile weighted averages
    do k = 1, nsoilay
      soitempc = soitempc + tsoi(i,k) * 0.5_wp * (froot(k,1) + froot(k,2))
      soimoisc = soimoisc + wsoi(i,k) * 0.5_wp * (froot(k,1) + froot(k,2))
    end do

    adwsoi(i)  = ((real(ndtimes,wp)-1.0_wp) * adwsoi(i)  + soilmois) * rwork
    adtsoi(i)  = ((real(ndtimes,wp)-1.0_wp) * adtsoi(i)  + soiltemp) * rwork
    adwisoi(i) = ((real(ndtimes,wp)-1.0_wp) * adwisoi(i) + soilice)  * rwork

    adtlaysoi(i) = ((real(ndtimes,wp)-1.0_wp) * adtlaysoi(i) + tsoi(i,1)) * rwork
    adwlaysoi(i) = ((real(ndtimes,wp)-1.0_wp) * adwlaysoi(i) + wsoi(i,1)) * rwork

    adwsoic(i)  = ((real(ndtimes,wp)-1.0_wp) * adwsoic(i) + soimoisc) * rwork
    adtsoic(i)  = ((real(ndtimes,wp)-1.0_wp) * adtsoic(i) + soitempc) * rwork

    ! -------------------------------------------------------------------
    ! Daily soil CO2 fluxes
    ! -------------------------------------------------------------------

    adco2mic(i)  = ((real(ndtimes,wp)-1.0_wp) * adco2mic(i)  + tco2mic(i)  * rwork3) * rwork
    adco2root(i) = ((real(ndtimes,wp)-1.0_wp) * adco2root(i) + tco2root(i) * rwork3) * rwork
    adco2soi(i)  = adco2root(i) + adco2mic(i)

    if (adco2soi(i) > 0.0_wp) then
      adco2ratio(i) = adco2root(i) / adco2soi(i)
    else
      adco2ratio(i) = -999.99_wp
    end if

    ! -------------------------------------------------------------------
    ! Daily litter decomposition factor
    ! -------------------------------------------------------------------

    ! Temperature decomposition factor (Arrhenius, Lloyd and Taylor 1994)
    if (tsoi(i,1) > 237.13_wp) then
      factor = min(exp(tconst * ((1.0_wp / (btemp - 227.13_wp)) - &
                                  (1.0_wp / (tsoi(i,1) - 227.13_wp)))), bconst)
    else
      factor = exp(tconst * ((1.0_wp / (btemp - 227.13_wp)) - &
                               (1.0_wp / (237.13_wp - 227.13_wp))))
    end if

    ! Water-filled pore space (%)
    wfps = (1.0_wp - wisoi(i,1)) * wsoi(i,1) * 100.0_wp

    ! Moisture decomposition factor
    if (wfps >= 60.0_wp) then
      moist = 0.000371_wp * (wfps**2) - (0.0748_wp * wfps) + 4.13_wp
    else
      moist = exp((wfps - 60.0_wp)**2 / (-800.0_wp))
    end if

    factor    = max(0.001_wp, min(bconst, factor * moist))
    decompl(i) = ((real(ndtimes,wp)-1.0_wp) * decompl(i) + factor) * rwork

    ! -------------------------------------------------------------------
    ! Daily soil carbon decomposition factor (root-profile weighted)
    ! -------------------------------------------------------------------

    ! Re-compute soiltemp and soilmois for 4-layer depth
    soiltemp = 0.0_wp
    soilmois = 0.0_wp
    soilice  = 0.0_wp
    do k = 1, 4
      soiltemp = soiltemp + tsoi(i,k)  * hsoi(k)
      soilmois = soilmois + wsoi(i,k)  * hsoi(k)
      soilice  = soilice  + wisoi(i,k) * hsoi(k)
    end do
    soiltemp = soiltemp * rdepth
    soilmois = soilmois * rdepth
    soilice  = soilice  * rdepth

    if (soiltemp > 237.13_wp) then
      factor = min(exp(tconst * ((1.0_wp / (btemp - 227.13_wp)) - &
                                  (1.0_wp / (soiltemp - 227.13_wp)))), bconst)
    else
      factor = exp(tconst * ((1.0_wp / (btemp - 227.13_wp)) - &
                               (1.0_wp / (237.13_wp - 227.13_wp))))
    end if

    wfps = (1.0_wp - soilice) * soilmois * 100.0_wp

    if (wfps >= 60.0_wp) then
      moist = 0.000371_wp * (wfps**2) - (0.0748_wp * wfps) + 4.13_wp
    else
      moist = exp((wfps - 60.0_wp)**2 / (-800.0_wp))
    end if

    factor    = max(0.001_wp, min(bconst, factor * moist))
    decomps(i) = ((real(ndtimes,wp)-1.0_wp) * decomps(i) + factor) * rwork

    ! -------------------------------------------------------------------
    ! Daily nitrogen mineralisation
    ! -------------------------------------------------------------------

    adnmintot(i) = ((real(ndtimes,wp)-1.0_wp) * adnmintot(i) + &
                     tnmin(i) * rwork4) * rwork

  end do

  return
end subroutine sumday


! -----------------------------------------------------------------------------
subroutine summonth (istep, iday, imonth)
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi
  use mod_run_params,    only: ndaypm
  use mod_constants,     only: hvap
  use mod_atm_state,     only: raina, snowa, rh, ta, qa, cloud, &
                                 solad, solai, fira
  use mod_canopy_fluxes, only: fvapa, firb, fsena
  use mod_hydrology,     only: grunof, gdrain
  use mod_snow_state,    only: fi, hsno
  use mod_soil_state,    only: tsoi, wsoi, wisoi, poros, swilt, hsoi
  use mod_veg_dynamics,  only: tnpp, tco2mic, tco2root, tnmin, lai, fu, fl
  use mod_summary,       only: nmtimes, &
                                 amrain, amsnow, amaet, amtrunoff, amsrunoff, amdrainage, &
                                 amtemp, amcloud, amqa, amrh, &
                                 amsolar, amirup, amirdown, amsens, amlatent, &
                                 amlaiu, amlail, &
                                 amtsoi, amwsoi, amwisoi, amvwc, amawc, &
                                 amsnod, amsnof, &
                                 amnpp, amnpptot, &
                                 amco2mic, amco2root, amco2soi, amco2ratio, &
                                 amneetot, amnmintot
  implicit none

  ! Arguments
  integer, intent(in) :: istep, iday, imonth

  ! Local variables
  integer  :: i, k
  real(wp) :: rwork, rwork2, rwork3, rwork4
  real(wp) :: rdepth, solartot, snodpth
  real(wp) :: soiltemp, soilmois, soilice, vwc, awc

  ! ---------------------------------------------------------------------------
  ! Update counters
  ! ---------------------------------------------------------------------------

  if ((istep == 1) .and. (iday == 1)) nmtimes = 0
  nmtimes = nmtimes + 1

  rwork  = 1.0_wp / real(nmtimes, wp)
  rwork2 = real(ndaypm(imonth), wp) * 86400.0_wp
  rwork3 = real(ndaypm(imonth), wp) * 86400.0_wp * 12.0e-3_wp
  rwork4 = real(ndaypm(imonth), wp) * 86400.0_wp * 14.0e-3_wp

  rdepth = 1.0_wp / (hsoi(1) + hsoi(2) + hsoi(3) + hsoi(4))

  ! Begin global grid
  do i = 1, npoi

    ! -------------------------------------------------------------------
    ! Monthly water budget terms
    ! -------------------------------------------------------------------

    amrain(i)    = ((real(nmtimes,wp)-1.0_wp) * amrain(i)    + raina(i) * 86400.0_wp) * rwork
    amsnow(i)    = ((real(nmtimes,wp)-1.0_wp) * amsnow(i)    + snowa(i) * 86400.0_wp) * rwork
    amaet(i)     = ((real(nmtimes,wp)-1.0_wp) * amaet(i)     - fvapa(i) * 86400.0_wp) * rwork

    amtrunoff(i)  = ((real(nmtimes,wp)-1.0_wp) * amtrunoff(i)  + &
                     (grunof(i) + gdrain(i)) * 86400.0_wp) * rwork
    amsrunoff(i)  = ((real(nmtimes,wp)-1.0_wp) * amsrunoff(i)  + &
                      grunof(i)              * 86400.0_wp) * rwork
    amdrainage(i) = ((real(nmtimes,wp)-1.0_wp) * amdrainage(i) + &
                      gdrain(i)              * 86400.0_wp) * rwork

    ! -------------------------------------------------------------------
    ! Monthly atmospheric terms
    ! -------------------------------------------------------------------

    amtemp(i)  = ((real(nmtimes,wp)-1.0_wp) * amtemp(i)  + ta(i) - 273.16_wp) * rwork
    amcloud(i) = ((real(nmtimes,wp)-1.0_wp) * amcloud(i) + cloud(i) * 100.0_wp) * rwork
    amqa(i)    = ((real(nmtimes,wp)-1.0_wp) * amqa(i)    + qa(i)) * rwork
    amrh(i)    = ((real(nmtimes,wp)-1.0_wp) * amrh(i)    + rh(i)) * rwork

    ! -------------------------------------------------------------------
    ! Energy budget terms
    ! -------------------------------------------------------------------

    solartot = solad(i,1) + solad(i,2) + solai(i,1) + solai(i,2)

    amsolar(i)  = ((real(nmtimes,wp)-1.0_wp) * amsolar(i)  + solartot) * rwork
    amirup(i)   = ((real(nmtimes,wp)-1.0_wp) * amirup(i)   + firb(i))  * rwork
    amirdown(i) = ((real(nmtimes,wp)-1.0_wp) * amirdown(i) + fira(i))  * rwork
    amsens(i)   = ((real(nmtimes,wp)-1.0_wp) * amsens(i)   - fsena(i)) * rwork
    amlatent(i) = ((real(nmtimes,wp)-1.0_wp) * amlatent(i) - fvapa(i) * hvap) * rwork

    ! -------------------------------------------------------------------
    ! Monthly vegetation parameters
    ! -------------------------------------------------------------------

    amlaiu(i) = ((real(nmtimes,wp)-1.0_wp) * amlaiu(i) + fu(i) * lai(i,2)) * rwork
    amlail(i) = ((real(nmtimes,wp)-1.0_wp) * amlail(i) + fl(i) * lai(i,1)) * rwork

    ! -------------------------------------------------------------------
    ! Monthly soil parameters
    ! -------------------------------------------------------------------

    soiltemp = 0.0_wp
    soilmois = 0.0_wp
    soilice  = 0.0_wp
    vwc      = 0.0_wp
    awc      = 0.0_wp

    do k = 1, 4
      soiltemp = soiltemp + tsoi(i,k)  * hsoi(k)
      soilmois = soilmois + wsoi(i,k)  * hsoi(k)
      soilice  = soilice  + wisoi(i,k) * hsoi(k)

      vwc = vwc + (wisoi(i,k) + (1.0_wp - wisoi(i,k)) * wsoi(i,k)) * &
                   hsoi(k) * poros(i,k)

      awc = awc + max(0.0_wp, (wisoi(i,k) + &
                (1.0_wp - wisoi(i,k)) * wsoi(i,k)) - swilt(i,k)) * &
                 hsoi(k) * poros(i,k) * 100.0_wp
    end do

    soiltemp = soiltemp * rdepth - 273.16_wp
    soilmois = soilmois * rdepth
    soilice  = soilice  * rdepth
    vwc      = vwc      * rdepth
    awc      = awc      * rdepth

    amtsoi(i)  = ((real(nmtimes,wp)-1.0_wp) * amtsoi(i)  + soiltemp) * rwork
    amwsoi(i)  = ((real(nmtimes,wp)-1.0_wp) * amwsoi(i)  + soilmois) * rwork
    amwisoi(i) = ((real(nmtimes,wp)-1.0_wp) * amwisoi(i) + soilice)  * rwork
    amvwc(i)   = ((real(nmtimes,wp)-1.0_wp) * amvwc(i)   + vwc)      * rwork
    amawc(i)   = ((real(nmtimes,wp)-1.0_wp) * amawc(i)   + awc)      * rwork

    ! -------------------------------------------------------------------
    ! Snow parameters
    ! -------------------------------------------------------------------

    snodpth   = hsno(i,1) + hsno(i,2) + hsno(i,3)
    amsnod(i) = ((real(nmtimes,wp)-1.0_wp) * amsnod(i) + snodpth) * rwork
    amsnof(i) = ((real(nmtimes,wp)-1.0_wp) * amsnof(i) + fi(i))   * rwork

    ! -------------------------------------------------------------------
    ! Monthly NPP per PFT
    ! -------------------------------------------------------------------

    amnpp(i,1)  = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,1)  + tnpp(i,1)  * rwork3) * rwork
    amnpp(i,2)  = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,2)  + tnpp(i,2)  * rwork3) * rwork
    amnpp(i,3)  = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,3)  + tnpp(i,3)  * rwork3) * rwork
    amnpp(i,4)  = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,4)  + tnpp(i,4)  * rwork3) * rwork
    amnpp(i,5)  = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,5)  + tnpp(i,5)  * rwork3) * rwork
    amnpp(i,6)  = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,6)  + tnpp(i,6)  * rwork3) * rwork
    amnpp(i,7)  = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,7)  + tnpp(i,7)  * rwork3) * rwork
    amnpp(i,8)  = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,8)  + tnpp(i,8)  * rwork3) * rwork
    amnpp(i,9)  = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,9)  + tnpp(i,9)  * rwork3) * rwork
    amnpp(i,10) = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,10) + tnpp(i,10) * rwork3) * rwork
    amnpp(i,11) = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,11) + tnpp(i,11) * rwork3) * rwork
    amnpp(i,12) = ((real(nmtimes,wp)-1.0_wp) * amnpp(i,12) + tnpp(i,12) * rwork3) * rwork

    amnpptot(i) = amnpp(i,1)  + amnpp(i,2)  + amnpp(i,3)  + &
                  amnpp(i,4)  + amnpp(i,5)  + amnpp(i,6)  + &
                  amnpp(i,7)  + amnpp(i,8)  + amnpp(i,9)  + &
                  amnpp(i,10) + amnpp(i,11) + amnpp(i,12)

    ! -------------------------------------------------------------------
    ! Monthly biogeochemistry
    ! -------------------------------------------------------------------

    amco2mic(i)  = ((real(nmtimes,wp)-1.0_wp) * amco2mic(i)  + tco2mic(i)  * rwork3) * rwork
    amco2root(i) = ((real(nmtimes,wp)-1.0_wp) * amco2root(i) + tco2root(i) * rwork3) * rwork
    amco2soi(i)  = amco2root(i) + amco2mic(i)

    if (amco2soi(i) > 0.0_wp) then
      amco2ratio(i) = amco2root(i) / amco2soi(i)
    else
      amco2ratio(i) = -999.99_wp
    end if

    amneetot(i)  = amnpptot(i) - amco2mic(i)

    amnmintot(i) = ((real(nmtimes,wp)-1.0_wp) * amnmintot(i) + &
                     tnmin(i) * rwork4) * rwork

  end do

  return
end subroutine summonth


! -----------------------------------------------------------------------------
subroutine sumyear (istep, iday, imonth)
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi, nsoilay, nsnolay
  use mod_run_params,    only: ndaypy
  use mod_constants,     only: rhow, hvap
  use mod_atm_state,     only: raina, snowa, solad, solai, fira
  use mod_canopy_fluxes, only: fvapa, firb, fsena
  use mod_hydrology,     only: grunof, gdrain, wtot, gtrans
  use mod_snow_state,    only: fi, rhos, hsno
  use mod_soil_state,    only: tsoi, wsoi, wisoi, poros, swilt, hsoi, &
                                 wpud, wipud, stresstl, stresstu
  use mod_veg_dynamics,  only: tgpp, tnpp, lai, sai, fu, fl, &
                                 firefac, totalit, totrlit, totcsoi, totcmic, &
                                 totanlit, totrnlit, totnsoi, &
                                 tco2mic, tco2root, tnmin
  use mod_canopy_params, only: wliqu, wsnou, wliqs, wsnos, wliql, wsnol
  use mod_carbon_pools,  only: cbior
  use mod_summary,       only: nytimes, &
                                 ayprcp, ayaet, aytrans, aytrunoff, aysrunoff, &
                                 aydrainage, aydwtot, &
                                 aywsoi, aywisoi, aytsoi, ayvwc, ayawc, &
                                 aystresstu, aystresstl, &
                                 aysolar, ayirup, ayirdown, aysens, aylatent, &
                                 aynpp, aygpp, aynpptot, aygpptot, &
                                 ayco2mic, ayco2root, ayco2soi, ayneetot, &
                                 ayrootbio, aynmintot, &
                                 ayalit, ayblit, aycsoi, aycmic, &
                                 ayanlit, aybnlit, aynsoi
  implicit none

  ! Arguments
  integer, intent(in) :: istep, iday, imonth

  ! Local variables
  integer  :: i, k
  real(wp) :: rwork, rwork2, rwork3, rwork4
  real(wp) :: rdepth, solartot, wtotp
  real(wp) :: soiltemp, soilmois, soilice, vwc, awc
  real(wp) :: water, waterfrac, fueldry, allroots

  ! ---------------------------------------------------------------------------
  ! Update counters
  ! ---------------------------------------------------------------------------

  if ((istep == 1) .and. (iday == 1) .and. (imonth == 1)) nytimes = 0
  nytimes = nytimes + 1

  rwork  = 1.0_wp / real(nytimes, wp)
  rwork2 = real(ndaypy, wp) * 86400.0_wp
  rwork3 = real(ndaypy, wp) * 86400.0_wp * 12.0e-3_wp
  rwork4 = real(ndaypy, wp) * 86400.0_wp * 14.0e-3_wp

  rdepth = 1.0_wp / (hsoi(1) + hsoi(2) + hsoi(3) + hsoi(4))

  ! Begin global grid
  do i = 1, npoi

    ! -------------------------------------------------------------------
    ! Annual energy budget terms
    ! -------------------------------------------------------------------

    solartot = solad(i,1) + solad(i,2) + solai(i,1) + solai(i,2)

    aysolar(i)  = ((real(nytimes,wp)-1.0_wp) * aysolar(i)  + solartot) * rwork
    ayirup(i)   = ((real(nytimes,wp)-1.0_wp) * ayirup(i)   + firb(i))  * rwork
    ayirdown(i) = ((real(nytimes,wp)-1.0_wp) * ayirdown(i) + fira(i))  * rwork
    aysens(i)   = ((real(nytimes,wp)-1.0_wp) * aysens(i)   - fsena(i)) * rwork
    aylatent(i) = ((real(nytimes,wp)-1.0_wp) * aylatent(i) - fvapa(i) * hvap) * rwork

    ! -------------------------------------------------------------------
    ! Annual water budget terms
    ! -------------------------------------------------------------------

    ayprcp(i)     = ((real(nytimes,wp)-1.0_wp) * ayprcp(i)     + &
                     (raina(i) + snowa(i)) * rwork2) * rwork
    ayaet(i)      = ((real(nytimes,wp)-1.0_wp) * ayaet(i)      - &
                      fvapa(i)             * rwork2) * rwork
    aytrans(i)    = ((real(nytimes,wp)-1.0_wp) * aytrans(i)    + &
                      gtrans(i)            * rwork2) * rwork

    aytrunoff(i)  = ((real(nytimes,wp)-1.0_wp) * aytrunoff(i)  + &
                     (grunof(i) + gdrain(i)) * rwork2) * rwork
    aysrunoff(i)  = ((real(nytimes,wp)-1.0_wp) * aysrunoff(i)  + &
                      grunof(i)              * rwork2) * rwork
    aydrainage(i) = ((real(nytimes,wp)-1.0_wp) * aydrainage(i) + &
                      gdrain(i)  * rwork2)   * rwork

    ! Water storage change
    wtotp = wtot(i)

    wtot(i) = (wliqu(i) + wsnou(i)) * fu(i) * 2.0_wp * lai(i,2)  + &
               (wliqs(i) + wsnos(i)) * fu(i) * 2.0_wp * sai(i,2)  + &
               (wliql(i) + wsnol(i)) * fl(i) * 2.0_wp * &
               (lai(i,1) + sai(i,1)) * (1.0_wp - fi(i))

    wtot(i) = wtot(i) + wpud(i) + wipud(i)

    do k = 1, nsoilay
      wtot(i) = wtot(i) + poros(i,k) * wsoi(i,k) * (1.0_wp - wisoi(i,k)) * &
                           hsoi(k) * rhow + &
                           poros(i,k) * wisoi(i,k) * hsoi(k) * rhow
    end do

    do k = 1, nsnolay
      wtot(i) = wtot(i) + fi(i) * rhos * hsno(i,k)
    end do

    aydwtot(i) = ((real(nytimes,wp)-1.0_wp) * aydwtot(i) + &
                   wtot(i) - wtotp) * rwork

    ! -------------------------------------------------------------------
    ! Annual soil parameters
    ! -------------------------------------------------------------------

    soiltemp = 0.0_wp
    soilmois = 0.0_wp
    soilice  = 0.0_wp
    vwc      = 0.0_wp
    awc      = 0.0_wp

    do k = 1, 4
      soiltemp = soiltemp + tsoi(i,k)  * hsoi(k)
      soilmois = soilmois + wsoi(i,k)  * hsoi(k)
      soilice  = soilice  + wisoi(i,k) * hsoi(k)

      vwc = vwc + (wisoi(i,k) + (1.0_wp - wisoi(i,k)) * wsoi(i,k)) * &
                   hsoi(k) * poros(i,k)

      awc = awc + max(0.0_wp, (wisoi(i,k) + &
                (1.0_wp - wisoi(i,k)) * wsoi(i,k)) - swilt(i,k)) * &
                 hsoi(k) * poros(i,k) * 100.0_wp
    end do

    soiltemp = soiltemp * rdepth - 273.16_wp
    soilmois = soilmois * rdepth
    soilice  = soilice  * rdepth
    vwc      = vwc      * rdepth
    awc      = awc      * rdepth

    aywsoi(i)  = ((real(nytimes,wp)-1.0_wp) * aywsoi(i)  + soilmois) * rwork
    aywisoi(i) = ((real(nytimes,wp)-1.0_wp) * aywisoi(i) + soilice)  * rwork
    aytsoi(i)  = ((real(nytimes,wp)-1.0_wp) * aytsoi(i)  + soiltemp) * rwork
    ayvwc(i)   = ((real(nytimes,wp)-1.0_wp) * ayvwc(i)   + vwc)      * rwork
    ayawc(i)   = ((real(nytimes,wp)-1.0_wp) * ayawc(i)   + awc)      * rwork

    aystresstu(i) = rwork * ((real(nytimes,wp)-1.0_wp) * aystresstu(i) + stresstu(i))
    aystresstl(i) = rwork * ((real(nytimes,wp)-1.0_wp) * aystresstl(i) + stresstl(i))

    ! -------------------------------------------------------------------
    ! Annual GPP per PFT
    ! -------------------------------------------------------------------

    aygpp(i,1)  = ((real(nytimes,wp)-1.0_wp) * aygpp(i,1)  + tgpp(i,1)  * rwork3) * rwork
    aygpp(i,2)  = ((real(nytimes,wp)-1.0_wp) * aygpp(i,2)  + tgpp(i,2)  * rwork3) * rwork
    aygpp(i,3)  = ((real(nytimes,wp)-1.0_wp) * aygpp(i,3)  + tgpp(i,3)  * rwork3) * rwork
    aygpp(i,4)  = ((real(nytimes,wp)-1.0_wp) * aygpp(i,4)  + tgpp(i,4)  * rwork3) * rwork
    aygpp(i,5)  = ((real(nytimes,wp)-1.0_wp) * aygpp(i,5)  + tgpp(i,5)  * rwork3) * rwork
    aygpp(i,6)  = ((real(nytimes,wp)-1.0_wp) * aygpp(i,6)  + tgpp(i,6)  * rwork3) * rwork
    aygpp(i,7)  = ((real(nytimes,wp)-1.0_wp) * aygpp(i,7)  + tgpp(i,7)  * rwork3) * rwork
    aygpp(i,8)  = ((real(nytimes,wp)-1.0_wp) * aygpp(i,8)  + tgpp(i,8)  * rwork3) * rwork
    aygpp(i,9)  = ((real(nytimes,wp)-1.0_wp) * aygpp(i,9)  + tgpp(i,9)  * rwork3) * rwork
    aygpp(i,10) = ((real(nytimes,wp)-1.0_wp) * aygpp(i,10) + tgpp(i,10) * rwork3) * rwork
    aygpp(i,11) = ((real(nytimes,wp)-1.0_wp) * aygpp(i,11) + tgpp(i,11) * rwork3) * rwork
    aygpp(i,12) = ((real(nytimes,wp)-1.0_wp) * aygpp(i,12) + tgpp(i,12) * rwork3) * rwork

    aygpptot(i) = aygpp(i,1)  + aygpp(i,2)  + aygpp(i,3)  + &
                  aygpp(i,4)  + aygpp(i,5)  + aygpp(i,6)  + &
                  aygpp(i,7)  + aygpp(i,8)  + aygpp(i,9)  + &
                  aygpp(i,10) + aygpp(i,11) + aygpp(i,12)

    ! -------------------------------------------------------------------
    ! Annual NPP per PFT
    ! -------------------------------------------------------------------

    aynpp(i,1)  = ((real(nytimes,wp)-1.0_wp) * aynpp(i,1)  + tnpp(i,1)  * rwork3) * rwork
    aynpp(i,2)  = ((real(nytimes,wp)-1.0_wp) * aynpp(i,2)  + tnpp(i,2)  * rwork3) * rwork
    aynpp(i,3)  = ((real(nytimes,wp)-1.0_wp) * aynpp(i,3)  + tnpp(i,3)  * rwork3) * rwork
    aynpp(i,4)  = ((real(nytimes,wp)-1.0_wp) * aynpp(i,4)  + tnpp(i,4)  * rwork3) * rwork
    aynpp(i,5)  = ((real(nytimes,wp)-1.0_wp) * aynpp(i,5)  + tnpp(i,5)  * rwork3) * rwork
    aynpp(i,6)  = ((real(nytimes,wp)-1.0_wp) * aynpp(i,6)  + tnpp(i,6)  * rwork3) * rwork
    aynpp(i,7)  = ((real(nytimes,wp)-1.0_wp) * aynpp(i,7)  + tnpp(i,7)  * rwork3) * rwork
    aynpp(i,8)  = ((real(nytimes,wp)-1.0_wp) * aynpp(i,8)  + tnpp(i,8)  * rwork3) * rwork
    aynpp(i,9)  = ((real(nytimes,wp)-1.0_wp) * aynpp(i,9)  + tnpp(i,9)  * rwork3) * rwork
    aynpp(i,10) = ((real(nytimes,wp)-1.0_wp) * aynpp(i,10) + tnpp(i,10) * rwork3) * rwork
    aynpp(i,11) = ((real(nytimes,wp)-1.0_wp) * aynpp(i,11) + tnpp(i,11) * rwork3) * rwork
    aynpp(i,12) = ((real(nytimes,wp)-1.0_wp) * aynpp(i,12) + tnpp(i,12) * rwork3) * rwork

    aynpptot(i) = aynpp(i,1)  + aynpp(i,2)  + aynpp(i,3)  + &
                  aynpp(i,4)  + aynpp(i,5)  + aynpp(i,6)  + &
                  aynpp(i,7)  + aynpp(i,8)  + aynpp(i,9)  + &
                  aynpp(i,10) + aynpp(i,11) + aynpp(i,12)

    ! -------------------------------------------------------------------
    ! Annual carbon budget terms
    ! -------------------------------------------------------------------

    ! Fire factor (fuel dryness)
    water     = wisoi(i,1) + (1.0_wp - wisoi(i,1)) * wsoi(i,1)
    waterfrac = (water - swilt(i,1)) / (1.0_wp - swilt(i,1))
    fueldry   = max(0.0_wp, min(1.0_wp, -2.0_wp * (waterfrac - 0.5_wp)))

    firefac(i) = ((real(nytimes,wp)-1.0_wp) * firefac(i) + fueldry) * rwork

    ayco2mic(i)  = ((real(nytimes,wp)-1.0_wp) * ayco2mic(i)  + tco2mic(i)  * rwork3) * rwork
    ayco2root(i) = ((real(nytimes,wp)-1.0_wp) * ayco2root(i) + tco2root(i) * rwork3) * rwork
    ayco2soi(i)  = ayco2root(i) + ayco2mic(i)
    ayneetot(i)  = aynpptot(i) - ayco2mic(i)

    ! Annual average root biomass
    allroots = cbior(i,1)  + cbior(i,2)  + cbior(i,3)  + &
               cbior(i,4)  + cbior(i,5)  + cbior(i,6)  + &
               cbior(i,7)  + cbior(i,8)  + cbior(i,9)  + &
               cbior(i,10) + cbior(i,11) + cbior(i,12)

    ayrootbio(i) = ((real(nytimes,wp)-1.0_wp) * ayrootbio(i) + allroots) * rwork

    ! -------------------------------------------------------------------
    ! Annual biogeochemistry terms
    ! -------------------------------------------------------------------

    aynmintot(i) = ((real(nytimes,wp)-1.0_wp) * aynmintot(i) + &
                     tnmin(i) * rwork4) * rwork

    ayalit(i)  = ((real(nytimes,wp)-1.0_wp) * ayalit(i)  + totalit(i))  * rwork
    ayblit(i)  = ((real(nytimes,wp)-1.0_wp) * ayblit(i)  + totrlit(i))  * rwork
    aycsoi(i)  = ((real(nytimes,wp)-1.0_wp) * aycsoi(i)  + totcsoi(i))  * rwork
    aycmic(i)  = ((real(nytimes,wp)-1.0_wp) * aycmic(i)  + totcmic(i))  * rwork
    ayanlit(i) = ((real(nytimes,wp)-1.0_wp) * ayanlit(i) + totanlit(i)) * rwork
    aybnlit(i) = ((real(nytimes,wp)-1.0_wp) * aybnlit(i) + totrnlit(i)) * rwork
    aynsoi(i)  = ((real(nytimes,wp)-1.0_wp) * aynsoi(i)  + totnsoi(i))  * rwork

  end do

  return
end subroutine sumyear


! -----------------------------------------------------------------------------
subroutine gdiag (iyear, iyear0)
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi
  use mod_run_params,   only: garea
  use mod_veg_dynamics, only: totbiou, totbiol
  use mod_summary,      only: nytimes, &
                                ayneetot, aynpptot, aygpptot, &
                                aytrans, aytrunoff, aysrunoff, aydrainage, aydwtot, &
                                ayprcp, ayaet, ayco2soi, &
                                ayalit, ayblit, aycsoi, ayanlit, aybnlit, aynsoi, &
                                ayrratio, aytratio
  implicit none

  ! Arguments
  integer, intent(in) :: iyear, iyear0

  ! Local variables
  integer  :: i
  real(wp) :: gtarea
  real(wp) :: gnee, gnpp, ggpp, gbiomass
  real(wp) :: galitc, gblitc, gsoic, gco2soi
  real(wp) :: galitn, gblitn, gsoin
  real(wp) :: gprcp, gaet, gt, gtrunoff, gsrunoff, gdrainage, gdwtot
  real(wp) :: aratio, rratio, sratio, tratio

  ! Initialize
  gtarea    = 0.0_wp;  gnee      = 0.0_wp;  gnpp      = 0.0_wp
  ggpp      = 0.0_wp;  gbiomass  = 0.0_wp;  galitc    = 0.0_wp
  gblitc    = 0.0_wp;  gsoic     = 0.0_wp;  gco2soi   = 0.0_wp
  galitn    = 0.0_wp;  gblitn    = 0.0_wp;  gsoin     = 0.0_wp
  gprcp     = 0.0_wp;  gaet      = 0.0_wp;  gt        = 0.0_wp
  gtrunoff  = 0.0_wp;  gsrunoff  = 0.0_wp;  gdrainage = 0.0_wp
  gdwtot    = 0.0_wp

  do i = 1, npoi

    ayrratio(i) = min(1.0_wp, max(0.0_wp, aytrunoff(i)) / &
                   max(0.1_wp, ayprcp(i)))
    aytratio(i) = min(1.0_wp, max(0.0_wp, aytrans(i))   / &
                   max(0.1_wp, ayaet(i)))

    gtarea    = gtarea    + garea(i)
    gnee      = gnee      + garea(i) * ayneetot(i) * 1.0e-12_wp
    gnpp      = gnpp      + garea(i) * aynpptot(i) * 1.0e-12_wp
    ggpp      = ggpp      + garea(i) * aygpptot(i) * 1.0e-12_wp
    gbiomass  = gbiomass  + garea(i) * totbiou(i)  * 1.0e-12_wp &
                           + garea(i) * totbiol(i)  * 1.0e-12_wp
    galitc    = galitc    + garea(i) * ayalit(i)   * 1.0e-12_wp
    gblitc    = gblitc    + garea(i) * ayblit(i)   * 1.0e-12_wp
    gsoic     = gsoic     + garea(i) * aycsoi(i)   * 1.0e-12_wp
    gco2soi   = gco2soi   + garea(i) * ayco2soi(i) * 1.0e-12_wp
    galitn    = galitn    + garea(i) * ayanlit(i)  * 1.0e-12_wp
    gblitn    = gblitn    + garea(i) * aybnlit(i)  * 1.0e-12_wp
    gsoin     = gsoin     + garea(i) * aynsoi(i)   * 1.0e-12_wp

    gprcp     = gprcp     + garea(i) * ayprcp(i)
    gaet      = gaet      + garea(i) * ayaet(i)
    gt        = gt        + garea(i) * aytrans(i)
    gtrunoff  = gtrunoff  + garea(i) * aytrunoff(i)
    gsrunoff  = gsrunoff  + garea(i) * aysrunoff(i)
    gdrainage = gdrainage + garea(i) * aydrainage(i)
    gdwtot    = gdwtot    + garea(i) * aydwtot(i) * real(nytimes, wp)

  end do

  gprcp     = gprcp     / gtarea
  gaet      = gaet      / gtarea
  gt        = gt        / gtarea
  gtrunoff  = gtrunoff  / gtarea
  gsrunoff  = gsrunoff  / gtarea
  gdrainage = gdrainage / gtarea
  gdwtot    = gdwtot    / gtarea

  aratio = gaet     / gprcp
  rratio = gtrunoff / gprcp
  sratio = gsrunoff / gtrunoff
  tratio = gt       / gaet

  write (*,*)  ' '
  write (*,*)  '* * * annual diagnostic fields * * *'
  write (*,*)  ' '
  write (*,9001) gnee
  write (*,9000) gnpp
  write (*,9002) ggpp
  write (*,9010) gbiomass
  write (*,9020) galitc
  write (*,9021) gblitc
  write (*,9030) gsoic
  write (*,9032) gco2soi
  write (*,9034) galitn
  write (*,9036) gblitn
  write (*,9038) gsoin
  write (*,*)  ' '
  write (*,9040) gprcp
  write (*,9050) gaet
  write (*,9060) gt
  write (*,9070) gtrunoff
  write (*,9080) gsrunoff
  write (*,9090) gdrainage
  write (*,9095) gdwtot
  write (*,*)  ' '
  write (*,9100) aratio
  write (*,9110) rratio
  write (*,*)  ' '
  write (*,9120) tratio
  write (*,9130) sratio
  write (*,*)  ' '

  ! Write to history file
  if (iyear == iyear0) then
    open (20, file='ibis.out.global', status='unknown')
    write (20,*) ' '
    write (20,*) '* * * annual diagnostic fields * * *'
    write (20,*) ' '
    write (20,*) &
      'year       nee       npp       gpp   biomass   scarbon ' // &
      'snitrogen   alitter   blitter    co2soi    '              // &
      'aratio    rratio    tratio'
  end if

  write (20,9500) iyear, gnee, gnpp, ggpp, gbiomass, gsoic, &
                  gsoin, galitc, gblitc, gco2soi, &
                  100.0_wp * aratio, 100.0_wp * rratio, 100.0_wp * tratio

  flush(20)

9000 format (1x,'total npp             of the domain (gt-c/yr) : ',f12.3)
9001 format (1x,'total nee             of the domain (gt-c/yr) : ',f12.5)
9002 format (1x,'total gpp             of the domain (gt-c/yr) : ',f12.3)
9010 format (1x,'total biomass         of the domain (gt-c)    : ',f12.3)
9020 format (1x,'aboveground litter    of the domain (gt-c)    : ',f12.3)
9021 format (1x,'belowground litter    of the domain (gt-c)    : ',f12.3)
9030 format (1x,'total soil carbon     of the domain (gt-c)    : ',f12.3)
9032 format (1x,'total soil co2 flux   of the domain (gt-c)    : ',f12.3)
9034 format (1x,'aboveground litter n  of the domain (gt-c)    : ',f12.3)
9036 format (1x,'belowground litter n  of the domain (gt-c)    : ',f12.3)
9038 format (1x,'total soil nitrogen   of the domain (gt-c)    : ',f12.3)
9040 format (1x,'average precipitation of the domain (mm/yr)   : ',f12.3)
9050 format (1x,'average aet           of the domain (mm/yr)   : ',f12.3)
9060 format (1x,'average transpiration of the domain (mm/yr)   : ',f12.3)
9070 format (1x,'average runoff        of the domain (mm/yr)   : ',f12.3)
9080 format (1x,'average surf runoff   of the domain (mm/yr)   : ',f12.3)
9090 format (1x,'average drainage      of the domain (mm/yr)   : ',f12.3)
9095 format (1x,'average moisture recharge of the domain (mm/yr) : ',f12.3)
9100 format (1x,'total aet      / precipitation                : ',f12.3)
9110 format (1x,'total runoff   / precipitation                : ',f12.3)
9120 format (1x,'transpiration  / total aet                    : ',f12.3)
9130 format (1x,'surface runoff / total runoff                 : ',f12.3)
9500 format (1x,i4,12f10.2)

  return
end subroutine gdiag


! -----------------------------------------------------------------------------
subroutine vdiag (iyear, iyear0)
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi
  use mod_run_params,   only: garea
  use mod_veg_dynamics, only: vegtype0, totbiou, totbiol, totlaiu, totlail
  use mod_summary,      only: ayneetot, aynpptot, aygpptot, aytrunoff, aycsoi
  implicit none

  ! Arguments
  integer, intent(in) :: iyear, iyear0

  ! Local variables
  integer  :: i, k
  real(wp) :: vtarea(15), vnee(15), vnpp(15), vgpp(15)
  real(wp) :: vbiomass(15), vlai(15), vsoic(15), vrunoff(15)

  ! Initialize
  vtarea   = 0.0_wp
  vnee     = 0.0_wp
  vnpp     = 0.0_wp
  vgpp     = 0.0_wp
  vbiomass = 0.0_wp
  vlai     = 0.0_wp
  vsoic    = 0.0_wp
  vrunoff  = 0.0_wp

  ! Sum ecosystem properties over each vegetation type
  do i = 1, npoi
    k = int(max(1.0_wp, min(15.0_wp, vegtype0(i))))

    vtarea(k)   = vtarea(k)   + garea(i)
    vnee(k)     = vnee(k)     + garea(i) * ayneetot(i)
    vnpp(k)     = vnpp(k)     + garea(i) * aynpptot(i)
    vgpp(k)     = vgpp(k)     + garea(i) * aygpptot(i)
    vbiomass(k) = vbiomass(k) + garea(i) * totbiou(i) &
                               + garea(i) * totbiol(i)
    vlai(k)     = vlai(k)     + garea(i) * totlaiu(i) &
                               + garea(i) * totlail(i)
    vsoic(k)    = vsoic(k)    + garea(i) * aycsoi(i)
    vrunoff(k)  = vrunoff(k)  + garea(i) * aytrunoff(i)
  end do

  ! Calculate area averages
  do k = 1, 15
    vnee(k)     = vnee(k)     / max(1.0_wp, vtarea(k))
    vnpp(k)     = vnpp(k)     / max(1.0_wp, vtarea(k))
    vgpp(k)     = vgpp(k)     / max(1.0_wp, vtarea(k))
    vbiomass(k) = vbiomass(k) / max(1.0_wp, vtarea(k))
    vlai(k)     = vlai(k)     / max(1.0_wp, vtarea(k))
    vsoic(k)    = vsoic(k)    / max(1.0_wp, vtarea(k))
    vrunoff(k)  = vrunoff(k)  / max(1.0_wp, vtarea(k))
  end do

  if (iyear == iyear0) then
    open (30, file='ibis.out.vegtype', status='unknown')
  end if

  write (30,*) ' '
  write (30,*) '* annual diagnostic fields by vegetation type *'
  write (30,*) ' '
  write (30,*) &
    'year    veg           area       nee       npp       gpp   ' // &
    'biomass       lai   scarbon    runoff '

  do k = 1, 15
    write (30,9000) &
      iyear, k, vtarea(k) / 1.0e6_wp, vnee(k), vnpp(k), vgpp(k), &
      vbiomass(k), vlai(k), vsoic(k), vrunoff(k)
  end do

  flush(30)

9000 format (1x,i4,5x,i2,5x,1e10.3,7f10.3)

  return
end subroutine vdiag
