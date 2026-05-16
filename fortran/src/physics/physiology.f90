! =============================================================================
! physiology.f90  --  Leaf-level physiology for IBIS
!
! Converted from physiology.f (Fortran 77 fixed-form) to F2008 free-form.
! Replaces all include directives with use associations to Phase 1–3 modules.
!
! Subroutines:
!   stomata    -- photosynthesis and stomatal conductance (Farquhar/Ball-Berry)
!   drystress  -- soil moisture stress factor
!   co2        -- atmospheric CO2 concentration (trivial pass-through)
! =============================================================================


! -----------------------------------------------------------------------------
subroutine stomata
! -----------------------------------------------------------------------------
! Calculates canopy-averaged net and gross photosynthesis, stomatal conductance,
! and total conductance for water vapour, for five plant functional types:
!   - upper canopy: broadleaf (ub), conifer (uc)
!   - lower canopy: shrub (ls), C3 grass (l3), C4 grass (l4)
!
! Uses an implicit (iterative) scheme for ci, cs, gs convergence.
! Scales leaf-level values to canopy averages using a radiation-weighted
! 10-day running mean of the extinction-based scaling parameter.
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi
  use mod_constants,     only: epsilon
  use mod_run_params,    only: dtime
  use mod_atm_state,     only: psurf, co2conc, o2conc
  use mod_soil_state,    only: stresstu, stresstl
  use mod_veg_dynamics,  only: lai, sai, exist
  use mod_canopy_params, only: tl, tu, t12, t34, q12, q34, su, sl,        &
                                topparu, topparl,                           &
                                ciub, ciuc, cils, cil3, cil4,               &
                                csub, csuc, csls, csl3, csl4,               &
                                gsub, gsuc, gsls, gsl3, gsl4,               &
                                agcub, agcuc, agcls, agcl3, agcl4,          &
                                ancub, ancuc, ancls, ancl3, ancl4,          &
                                totcondub, totconduc,                        &
                                totcondls, totcondl3, totcondl4
  use mod_canopy_fluxes, only: fwetu, fwetl, termu, terml, scalcoefu, scalcoefl
  use mod_summary,       only: a10scalparamu, a10daylightu, &
                                a10scalparaml, a10daylightl
  use mod_pft_params,    only: tau15, kc15, ko15,                           &
                                alpha3, alpha4,                              &
                                theta3, beta3, theta4, beta4,               &
                                cimax,                                       &
                                gammaub, gammauc, gammals, gammal3, gammal4, &
                                coefmub, coefmuc, coefmls, coefml3, coefml4, &
                                coefbub, coefbuc, coefbls, coefbl3, coefbl4, &
                                gsubmin, gsucmin, gslsmin, gsl3min, gsl4min, &
                                vmax_pft
  use mod_thermo_funcs,  only: esat, qsat
  implicit none

  integer  :: i
  real(wp) :: rwork, tau, tleaf, tempvm, zweight
  real(wp) :: esat12, qsat12, rh12, esat34, qsat34, rh34
  real(wp) :: gbco2u, gbco2l
  real(wp) :: gscub, gscuc, gscls, gscl3, gscl4
  real(wp) :: vmax, vmaxub, vmaxuc, vmaxls, vmaxl3, vmaxl4
  real(wp) :: rdarkub, rdarkuc, rdarkls, rdarkl3, rdarkl4
  real(wp) :: agub, aguc, agls, agl3, agl4
  real(wp) :: anub, anuc, anls, anl3, anl4
  real(wp) :: duma, dumb, dumc, dume, dumq, dump
  real(wp) :: pxaiu, plaiu, pxail, plail
  real(wp) :: cscub, cscuc, cscls, cscl3, cscl4
  real(wp) :: extpar, scale
  real(wp) :: kc, ko, kco2, je, jc, ji, jp, js, gamstar

  ! ===========================================================================
  ! * * * Upper canopy physiology * * *
  ! ===========================================================================

  do i = 1, npoi

    ! ---- temperature-dependent parameters ------------------------------------
    rwork  = 3.47e-03_wp - 1.0_wp / tu(i)
    tau    = tau15 * exp(-4500.0_wp * rwork)
    kc     = kc15  * exp( 6000.0_wp * rwork)
    ko     = ko15  * exp( 1500.0_wp * rwork)
    tleaf  = tu(i) - 273.16_wp
    tempvm = exp(3500.0_wp * rwork) / &
             ((1.0_wp + exp(0.40_wp * (  5.0_wp - tleaf))) * &
              (1.0_wp + exp(0.40_wp * (tleaf - 50.0_wp))))

    ! CO2 compensation point (mol/mol)
    gamstar = o2conc / (2.0_wp * tau)

    ! constrain ci to stable range
    ciub(i) = max(1.05_wp*gamstar, min(cimax, ciub(i)))
    ciuc(i) = max(1.05_wp*gamstar, min(cimax, ciuc(i)))

    ! boundary layer conductance for CO2 (mol m-2 s-1)
    gbco2u = min(10.0_wp, max(0.1_wp, su(i) * 25.5_wp))

    ! relative humidity in upper canopy air space (minimum 0.30)
    esat12 = esat(t12(i))
    qsat12 = qsat(esat12, psurf(i))
    rh12   = max(0.30_wp, q12(i) / qsat12)

    ! ---- broadleaf tree physiology (ub) -------------------------------------
    if (exist(i,1) > 0.5_wp) then
      vmaxub = vmax_pft(1)    ! tropical broadleaf evergreen
    else if (exist(i,3) > 0.5_wp) then
      vmaxub = vmax_pft(3)    ! warm-temperate broadleaf evergreen
    else
      vmaxub = vmax_pft(5)    ! temperate/boreal broadleaf cold deciduous
    end if

    vmax    = vmaxub * tempvm * stresstu(i)
    rdarkub = gammaub * vmaxub * tempvm

    je = topparu(i) * 4.59e-06_wp * alpha3 * (ciub(i) - gamstar) / &
         (ciub(i) + 2.0_wp * gamstar)
    jc = vmax * (ciub(i) - gamstar) / &
         (ciub(i) + kc * (1.0_wp + o2conc / ko))

    duma  = theta3
    dumb  = je + jc
    dumc  = je * jc
    dume  = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq  = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    jp    = min(dumq/duma, dumc/dumq)

    js    = vmax / 2.2_wp
    duma  = beta3
    dumb  = jp + js
    dumc  = jp * js
    dume  = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq  = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    agub  = min(dumq/duma, dumc/dumq)
    anub  = agub - rdarkub

    csub(i) = 0.5_wp * (csub(i) + co2conc - anub / gbco2u)
    csub(i) = max(1.05_wp*gamstar, csub(i))
    gsub(i) = 0.5_wp * (gsub(i) + coefmub*anub*rh12/csub(i) + coefbub*stresstu(i))
    gsub(i) = max(gsubmin, coefbub*stresstu(i), gsub(i))
    ciub(i) = 0.5_wp * (ciub(i) + csub(i) - 1.6_wp*anub/gsub(i))
    ciub(i) = max(1.05_wp*gamstar, min(cimax, ciub(i)))

    ! ---- conifer tree physiology (uc) ----------------------------------------
    if (exist(i,4) > 0.5_wp) then
      vmaxuc = vmax_pft(4)    ! temperate conifer
    else
      vmaxuc = vmax_pft(6)    ! boreal conifer evergreen
    end if

    vmax    = vmaxuc * tempvm * stresstu(i)
    rdarkuc = gammauc * vmaxuc * tempvm

    je = topparu(i) * 4.59e-06_wp * alpha3 * (ciuc(i) - gamstar) / &
         (ciuc(i) + 2.0_wp * gamstar)
    jc = vmax * (ciuc(i) - gamstar) / &
         (ciuc(i) + kc * (1.0_wp + o2conc / ko))

    duma  = theta3
    dumb  = je + jc
    dumc  = je * jc
    dume  = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq  = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    jp    = min(dumq/duma, dumc/dumq)

    js    = vmax / 2.2_wp
    duma  = beta3
    dumb  = jp + js
    dumc  = jp * js
    dume  = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq  = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    aguc  = min(dumq/duma, dumc/dumq)
    anuc  = aguc - rdarkuc

    csuc(i) = 0.5_wp * (csuc(i) + co2conc - anuc / gbco2u)
    csuc(i) = max(1.05_wp*gamstar, csuc(i))
    gsuc(i) = 0.5_wp * (gsuc(i) + coefmuc*anuc*rh12/csuc(i) + coefbuc*stresstu(i))
    gsuc(i) = max(gsucmin, coefbuc*stresstu(i), gsuc(i))
    ciuc(i) = 0.5_wp * (ciuc(i) + csuc(i) - 1.6_wp*anuc/gsuc(i))
    ciuc(i) = max(1.05_wp*gamstar, min(cimax, ciuc(i)))

    ! ---- upper canopy scaling ------------------------------------------------
    ! Extinction coefficient for PAR through canopy:
    !   extpar = (k*A + h*B - h*C) / (A+B+C)
    extpar = (termu(i,6)*scalcoefu(i,1) + &
              termu(i,7)*scalcoefu(i,2) - &
              termu(i,7)*scalcoefu(i,3)) / &
              max(scalcoefu(i,4), epsilon)
    extpar = max(1.0e-1_wp, min(1.0e+1_wp, extpar))

    pxaiu = extpar * (lai(i,2) + sai(i,2))
    plaiu = extpar *  lai(i,2)

    ! 10-day running mean decay weight
    zweight = exp(-1.0_wp / (10.0_wp * 86400.0_wp / dtime))

    if (plaiu > 0.0_wp) then
      if (topparu(i) > 10.0_wp) then
        ! daytime: use current scaling
        scale = (1.0_wp - exp(-pxaiu)) / plaiu
        a10scalparamu(i) = zweight*a10scalparamu(i) + (1.0_wp-zweight)*scale*topparu(i)
        a10daylightu(i)  = zweight*a10daylightu(i)  + (1.0_wp-zweight)*topparu(i)
      else
        ! nighttime: use 10-day mean
        scale = a10scalparamu(i) / a10daylightu(i)
      end if
    else
      scale = 0.0_wp
    end if

    agcub(i) = agub * scale
    agcuc(i) = aguc * scale
    ancub(i) = anub * scale
    ancuc(i) = anuc * scale

    ! canopy average surface CO2 (big-leaf approach)
    cscub = max(1.05_wp*gamstar, co2conc - ancub(i)/gbco2u)
    cscuc = max(1.05_wp*gamstar, co2conc - ancuc(i)/gbco2u)

    ! canopy average stomatal conductance
    gscub = coefmub*ancub(i)*rh12/cscub + coefbub*stresstu(i)
    gscuc = coefmuc*ancuc(i)*rh12/cscuc + coefbuc*stresstu(i)
    gscub = max(gsubmin, coefbub*stresstu(i), gscub)
    gscuc = max(gsucmin, coefbuc*stresstu(i), gscuc)

    ! total conductance for water vapour
    rwork = 1.0_wp / su(i)
    dump  = 1.0_wp / 0.029_wp
    totcondub(i) = 1.0_wp / (rwork + dump/gscub)
    totconduc(i) = 1.0_wp / (rwork + dump/gscuc)

    ! scale by dry fraction (wet leaves don't do photosynthesis)
    rwork = 1.0_wp - fwetu(i)
    agcub(i) = rwork * agcub(i)
    agcuc(i) = rwork * agcuc(i)
    ancub(i) = rwork * ancub(i)
    ancuc(i) = rwork * ancuc(i)
  end do

  ! ===========================================================================
  ! * * * Lower canopy physiology * * *
  ! ===========================================================================

  do i = 1, npoi

    ! ---- temperature-dependent parameters ------------------------------------
    rwork  = 3.47e-03_wp - 1.0_wp / tl(i)
    tau    = tau15 * exp(-4500.0_wp * rwork)
    kc     = kc15  * exp( 6000.0_wp * rwork)
    ko     = ko15  * exp( 1500.0_wp * rwork)
    tleaf  = tl(i) - 273.16_wp
    tempvm = exp(3500.0_wp * rwork) / &
             ((1.0_wp + exp(0.40_wp * (  5.0_wp - tleaf))) * &
              (1.0_wp + exp(0.40_wp * (tleaf - 50.0_wp))))

    gamstar = o2conc / (2.0_wp * tau)

    cils(i) = max(1.05_wp*gamstar, min(cimax, cils(i)))
    cil3(i) = max(1.05_wp*gamstar, min(cimax, cil3(i)))
    cil4(i) = max(0.0_wp,          min(cimax, cil4(i)))

    ! boundary layer conductance for CO2 (mol m-2 s-1)
    gbco2l = min(10.0_wp, max(0.1_wp, sl(i) * 25.5_wp))

    ! relative humidity in lower canopy air space (minimum 0.30)
    esat34 = esat(t34(i))
    qsat34 = qsat(esat34, psurf(i))
    rh34   = max(0.30_wp, q34(i) / qsat34)

    ! ---- shrub physiology (ls) ----------------------------------------------
    vmaxls  = vmax_pft(9)    ! shrubs (evergreen or cold deciduous)
    vmax    = vmaxls * tempvm * stresstl(i)
    rdarkls = gammals * vmaxls * tempvm

    je = topparl(i) * 4.59e-06_wp * alpha3 * (cils(i) - gamstar) / &
         (cils(i) + 2.0_wp * gamstar)
    jc = vmax * (cils(i) - gamstar) / &
         (cils(i) + kc * (1.0_wp + o2conc / ko))

    duma  = theta3
    dumb  = je + jc
    dumc  = je * jc
    dume  = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq  = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    jp    = min(dumq/duma, dumc/dumq)

    js    = vmax / 2.2_wp
    duma  = beta3
    dumb  = jp + js
    dumc  = jp * js
    dume  = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq  = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    agls  = min(dumq/duma, dumc/dumq)
    anls  = agls - rdarkls

    csls(i) = 0.5_wp * (csls(i) + co2conc - anls/gbco2l)
    csls(i) = max(1.05_wp*gamstar, csls(i))
    gsls(i) = 0.5_wp * (gsls(i) + coefmls*anls*rh34/csls(i) + coefbls*stresstl(i))
    gsls(i) = max(gslsmin, coefbls*stresstl(i), gsls(i))
    cils(i) = 0.5_wp * (cils(i) + csls(i) - 1.6_wp*anls/gsls(i))
    cils(i) = max(1.05_wp*gamstar, min(cimax, cils(i)))

    ! ---- C3 grass physiology (l3) -------------------------------------------
    vmaxl3  = vmax_pft(12)   ! C3 grasses
    vmax    = vmaxl3 * tempvm * stresstl(i)
    rdarkl3 = gammal3 * vmaxl3 * tempvm

    je = topparl(i) * 4.59e-06_wp * alpha3 * (cil3(i) - gamstar) / &
         (cil3(i) + 2.0_wp * gamstar)
    jc = vmax * (cil3(i) - gamstar) / &
         (cil3(i) + kc * (1.0_wp + o2conc / ko))

    duma  = theta3
    dumb  = je + jc
    dumc  = je * jc
    dume  = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq  = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    jp    = min(dumq/duma, dumc/dumq)

    js    = vmax / 2.2_wp
    duma  = beta3
    dumb  = jp + js
    dumc  = jp * js
    dume  = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq  = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    agl3  = min(dumq/duma, dumc/dumq)
    anl3  = agl3 - rdarkl3

    csl3(i) = 0.5_wp * (csl3(i) + co2conc - anl3/gbco2l)
    csl3(i) = max(1.05_wp*gamstar, csl3(i))
    gsl3(i) = 0.5_wp * (gsl3(i) + coefml3*anl3*rh34/csl3(i) + coefbl3*stresstl(i))
    gsl3(i) = max(gsl3min, coefbl3*stresstl(i), gsl3(i))
    cil3(i) = 0.5_wp * (cil3(i) + csl3(i) - 1.6_wp*anl3/gsl3(i))
    cil3(i) = max(1.05_wp*gamstar, min(cimax, cil3(i)))

    ! ---- C4 grass physiology (l4) -------------------------------------------
    vmaxl4  = vmax_pft(11)   ! C4 grasses
    ! C4 uses different tempvm (different cold-clamp coefficients)
    tempvm  = exp(3500.0_wp * rwork) / &
              ((1.0_wp + exp(0.40_wp * ( 10.0_wp - tleaf))) * &
               (1.0_wp + exp(0.40_wp * (tleaf - 50.0_wp))))
    vmax    = vmaxl4 * tempvm * stresstl(i)
    rdarkl4 = gammal4 * vmaxl4 * tempvm

    kco2 = 18.0e+03_wp * vmax    ! initial C4 CO2 efficiency (mol m-2 s-1)

    je   = topparl(i) * 4.59e-06_wp * alpha4    ! light-limited
    jc   = vmax                                  ! Rubisco-limited

    duma = theta4
    dumb = je + jc
    dumc = je * jc
    dume = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    jp   = min(dumq/duma, dumc/dumq)

    ji   = kco2 * cil4(i)       ! CO2-limited

    duma = beta4
    dumb = jp + ji
    dumc = jp * ji
    dume = max(dumb**2 - 4.0_wp*duma*dumc, 0.0_wp)
    dumq = 0.5_wp * (dumb + sqrt(dume)) + 1.0e-15_wp
    agl4 = min(dumq/duma, dumc/dumq)
    anl4 = agl4 - rdarkl4

    ! csl4 limited to 1e-8 to avoid division by zero in gsl4
    csl4(i) = 0.5_wp * (csl4(i) + co2conc - anl4/gbco2l)
    csl4(i) = max(1.0e-8_wp, csl4(i))
    gsl4(i) = 0.5_wp * (gsl4(i) + coefml4*anl4*rh34/csl4(i) + coefbl4*stresstl(i))
    gsl4(i) = max(gsl4min, coefbl4*stresstl(i), gsl4(i))
    cil4(i) = 0.5_wp * (cil4(i) + csl4(i) - 1.6_wp*anl4/gsl4(i))
    cil4(i) = max(0.0_wp, min(cimax, cil4(i)))

    ! ---- lower canopy scaling -----------------------------------------------
    extpar = (terml(i,6)*scalcoefl(i,1) + &
              terml(i,7)*scalcoefl(i,2) - &
              terml(i,7)*scalcoefl(i,3)) / &
              max(scalcoefl(i,4), epsilon)
    extpar = max(1.0e-1_wp, min(1.0e+1_wp, extpar))

    pxail = extpar * (lai(i,1) + sai(i,1))
    plail = extpar *  lai(i,1)

    zweight = exp(-1.0_wp / (10.0_wp * 86400.0_wp / dtime))

    if (plail > 0.0_wp) then
      if (topparl(i) > 10.0_wp) then
        scale = (1.0_wp - exp(-pxail)) / plail
        a10scalparaml(i) = zweight*a10scalparaml(i) + (1.0_wp-zweight)*scale*topparl(i)
        a10daylightl(i)  = zweight*a10daylightl(i)  + (1.0_wp-zweight)*topparl(i)
      else
        scale = a10scalparaml(i) / a10daylightl(i)
      end if
    else
      scale = 0.0_wp
    end if

    agcls(i) = agls * scale
    agcl4(i) = agl4 * scale
    agcl3(i) = agl3 * scale
    ancls(i) = anls * scale
    ancl4(i) = anl4 * scale
    ancl3(i) = anl3 * scale

    ! canopy average surface CO2
    cscls = max(1.05_wp*gamstar, co2conc - ancls(i)/gbco2l)
    cscl3 = max(1.05_wp*gamstar, co2conc - ancl3(i)/gbco2l)
    cscl4 = max(1.0e-8_wp,       co2conc - ancl4(i)/gbco2l)

    ! canopy average stomatal conductance
    gscls = coefmls*ancls(i)*rh34/cscls + coefbls*stresstl(i)
    gscl3 = coefml3*ancl3(i)*rh34/cscl3 + coefbl3*stresstl(i)
    gscl4 = coefml4*ancl4(i)*rh34/cscl4 + coefbl4*stresstl(i)
    gscls = max(gslsmin, coefbls*stresstl(i), gscls)
    gscl3 = max(gsl3min, coefbl3*stresstl(i), gscl3)
    gscl4 = max(gsl4min, coefbl4*stresstl(i), gscl4)

    ! total conductance for water vapour
    rwork = 1.0_wp / sl(i)
    dump  = 1.0_wp / 0.029_wp
    totcondls(i) = 1.0_wp / (rwork + dump/gscls)
    totcondl3(i) = 1.0_wp / (rwork + dump/gscl3)
    totcondl4(i) = 1.0_wp / (rwork + dump/gscl4)

    ! scale by dry fraction
    rwork = 1.0_wp - fwetl(i)
    agcls(i) = rwork * agcls(i)
    agcl3(i) = rwork * agcl3(i)
    agcl4(i) = rwork * agcl4(i)
    ancls(i) = rwork * ancls(i)
    ancl3(i) = rwork * ancl3(i)
    ancl4(i) = rwork * ancl4(i)
  end do

end subroutine stomata


! -----------------------------------------------------------------------------
subroutine drystress
! -----------------------------------------------------------------------------
! Computes soil moisture stress factors (stressl, stressu, stresstl, stresstu)
! from the plant-available water content in each soil layer, weighted by root
! fraction.  stressfac=-5 gives a sigmoidal response to dryness.
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi, nsoilay
  use mod_soil_state,   only: wsoi, wisoi, swilt, sfield, stressl, stressu, &
                               stresstl, stresstu
  use mod_veg_dynamics, only: froot
  implicit none

  integer  :: i, k
  real(wp) :: awc, znorm, zwilt

  real(wp), parameter :: stressfac = -5.0_wp

  znorm = 1.0_wp - exp(stressfac)

  do i = 1, npoi
    stresstl(i) = 0.0_wp
    stresstu(i) = 0.0_wp

    do k = 1, nsoilay
      ! plant available water content (fraction of field-capacity range)
      awc = min(1.0_wp, max(0.0_wp, &
            (wsoi(i,k)*(1.0_wp - wisoi(i,k)) - swilt(i,k)) / &
            (sfield(i,k) - swilt(i,k))))

      zwilt = (1.0_wp - exp(stressfac * awc)) / znorm

      stressl(i,k) = froot(k,1) * max(0.0_wp, min(1.0_wp, zwilt))
      stressu(i,k) = froot(k,2) * max(0.0_wp, min(1.0_wp, zwilt))

      stresstl(i) = stresstl(i) + stressl(i,k)
      stresstu(i) = stresstu(i) + stressu(i,k)
    end do
  end do

end subroutine drystress


! -----------------------------------------------------------------------------
subroutine co2(co2init, co2conc_out, iyear)
! -----------------------------------------------------------------------------
! Returns the atmospheric CO2 concentration for the current year.
! Currently a trivial pass-through: co2conc_out = co2init.
! (IPCC transient scenarios are commented out in the original code.)
! -----------------------------------------------------------------------------
  use mod_precision, only: wp
  implicit none

  real(wp), intent(in)  :: co2init    !< prescribed CO2 concentration (mol/mol)
  real(wp), intent(out) :: co2conc_out !< CO2 concentration used this year
  integer,  intent(in)  :: iyear      !< current year (unused in active code)

  co2conc_out = co2init

end subroutine co2
