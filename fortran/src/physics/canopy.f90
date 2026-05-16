! =============================================================================
! canopy.f90  --  F77 -> F2008 conversion of canopy.f
!
! Contains subroutines (in call order):
!   canopy   -- controller: calls canini, drystress, turcof, stomata, turvap
!   canini   -- initialise aerodynamic geometry
!   turcof   -- wind speeds and transfer coefficients
!   turvap   -- implicit heat/moisture solver (matrix)
!   fstrat   -- stratification correction (Louis 1979)
!   impexp   -- implicit/explicit fraction for veg surfaces
!   impexp2  -- implicit/explicit fraction for snow skin
!   fwetcal  -- fractional wet areas
!   cascade  -- intercepted-water cascade through canopy levels
!   steph2o  -- step intercepted water for one canopy component
!   cascad2  -- end-of-timestep evaporation removal + phase fix
!   steph2o2 -- steph2o worker: evap removal + phase correction
!   mix      -- calorimetric mixing of masses
!   noveg    -- set prog vars when no veg exists
!   twet3    -- function: wet bulb temperature
! =============================================================================

! =============================================================================
      subroutine canopy
! =============================================================================
!
! calculates sensible heat and moisture flux coefficients,
! and steps canopy temperatures through one timestep
!
      implicit none
!
! local variables
!
      integer :: niter, iter
!
      call canini
!
! estimate soil moisture stress parameters
!
      call drystress
!
! iterate the whole canopy physics solution niter times
!
      niter = 3
!
      do iter = 1, niter
!
!       calculate wind speeds and aerodynamic transfer coefficients
!
        call turcof(iter)
!
!       calculate canopy photosynthesis rates and conductance
!
        call stomata
!
!       solve implicit system of heat and water balance equations
!
        call turvap(iter, niter)
!
      end do
!
      end subroutine canopy

! =============================================================================
      subroutine canini
! =============================================================================
!
! initializes aerodynamic quantities that remain constant
! through one timestep
!
      use mod_precision,      only: wp
      use mod_dimensions,     only: npoi
      use mod_constants,      only: grav, rair, rvap, cair, cvap, cappa
      use mod_atm_state,      only: ta, qa, psurf, ua
      use mod_snow_state,     only: fi, z0sno
      use mod_soil_state,     only: z0soi
      use mod_veg_dynamics,   only: fl, fu, lai, sai, ztop, zbot
      use mod_canopy_params,  only: alaiml, alaimu
      use mod_canopy_fluxes,  only: rhoa, cp, za, tfac, &
                                    z1, z2, z3, z4, z12, z34, &
                                    bdl, bdu, dil, diu, &
                                    exphl, expl, exphu, expu, &
                                    displ, dispu, &
                                    alogg, alogi, alogav, &
                                    alog1, alog2, alog3, alog4, aloga, &
                                    alogu, alogl, u2
      implicit none
!
! local variables
!
      integer :: i
      real(wp) :: siga, pa, x, x1, rwork
      real(wp) :: cvegl, dvegl, bvegl, cvegu, dvegu, bvegu
!
! define sigma level of atmospheric data
!
      siga = 0.999_wp
!
      tfac = 1.0_wp / (siga**cappa)
!
! atmospheric conditions at za
!
      do i = 1, npoi
!
        pa = psurf(i) * siga
!
        rhoa(i) = pa / (rair * ta(i) * &
                  (1.0_wp + (rvap / rair - 1.0_wp) * qa(i)))
!
        cp(i) = cair * (1.0_wp + (cvap / cair - 1.0_wp) * qa(i))
!
        za(i) = (psurf(i) - pa) / (rhoa(i) * grav)
!
        za(i) = max(za(i), ztop(i,2) + 1.0_wp)
!
      end do
!
! aerodynamic coefficients for the lower story
!
      do i = 1, npoi
!
        x = fl(i) * (1.0_wp - fi(i)) * 2.0_wp * (lai(i,1) + sai(i,1)) / alaiml
!
        x  = min(x, 3.0_wp)
        x1 = min(x, 1.0_wp)
!
        rwork = max(ztop(i,1) - zbot(i,1), 0.01_wp)
        cvegl = (0.4_wp / rwork) * max(1.0e-5_wp, x)
        dvegl = (0.1_wp * rwork) / max(1.0e-5_wp, x, x**2)
        bvegl = sqrt(2.0_wp * cvegl / dvegl)
!
        bdl(i) = 0.5_wp * bvegl * dvegl
        dil(i) = 1.0_wp / dvegl
!
        rwork = (1.0_wp - x1) * (max(z0soi(i), z0sno) + 0.01_wp)
!
        z3(i)  = x1 * ztop(i,1) + rwork
        z4(i)  = x1 * zbot(i,1) + rwork
        z34(i) = 0.5_wp * (z3(i) + z4(i))
!
        exphl(i) = exp(0.5_wp * bvegl * (z3(i) - z4(i)))
        expl(i)  = exphl(i)**2
!
        displ(i) = x1 * 0.7_wp * z3(i)
!
      end do
!
! aerodynamic coefficients for the upper story
!
      do i = 1, npoi
!
        x = fu(i) * 2.0_wp * (lai(i,2) + sai(i,2)) / alaimu
!
        x  = min(x, 3.0_wp)
        x1 = min(x, 1.0_wp)
!
        rwork  = max(ztop(i,2) - zbot(i,2), 0.01_wp)
        cvegu  = (0.4_wp / rwork) * max(1.0e-5_wp, x)
        dvegu  = (0.1_wp * rwork) / max(1.0e-5_wp, x, x**2)
        rwork  = 1.0_wp / dvegu
        bvegu  = sqrt(2.0_wp * cvegu * rwork)
        bdu(i) = 0.5_wp * bvegu * dvegu
        diu(i) = rwork
!
        rwork  = (1.0_wp - x1) * (z3(i) + 0.01_wp)
        z1(i)  = x1 * ztop(i,2) + rwork
        z2(i)  = x1 * zbot(i,2) + rwork
        z12(i) = 0.5_wp * (z1(i) + z2(i))
!
        exphu(i) = exp(0.5_wp * bvegu * (z1(i) - z2(i)))
        expu(i)  = exphu(i)**2
!
        dispu(i) = x1 * 0.7_wp * z1(i) + (1.0_wp - x1) * displ(i)
!
      end do
!
! mixing-length logarithms
!
      do i = 1, npoi
!
        alogg(i)  = log(z0soi(i))
        alogi(i)  = log(z0sno)
        alogav(i) = (1.0_wp - fi(i)) * alogg(i) + fi(i) * alogi(i)
!
        alog4(i) = log(max(z4(i), 1.1_wp*z0soi(i), 1.1_wp*z0sno))
        alog3(i) = log(z3(i) - displ(i))
        alog2(i) = log(z2(i) - displ(i))
        alog1(i) = log(z1(i) - dispu(i))
        aloga(i) = log(za(i)  - dispu(i))
!
! initialize u2, alogu, alogl for first iteration's fstrat
!
        u2(i)    = ua(i) / exphu(i)
        alogu(i) = log(max(0.01_wp, 0.1_wp*(z1(i) - z2(i))))
        alogl(i) = log(max(0.01_wp, 0.1_wp*(z3(i) - z4(i))))
!
      end do
!
      end subroutine canini

! =============================================================================
      subroutine turcof(iter)
! =============================================================================
!
! solves for wind speeds at various levels and computes
! aerodynamic transfer coefficients
!
      use mod_precision,      only: wp
      use mod_dimensions,     only: npoi
      use mod_constants,      only: vonk
      use mod_run_params,     only: dtime
      use mod_atm_state,      only: ua, ta, qa
      use mod_canopy_params,  only: cleaf, cstem, cgrass, dleaf, dstem, &
                                    su, ss, sl, t12, t34, q12, q34
      use mod_canopy_fluxes,  only: rhoa, za, tfac, &
                                    z1, z2, z3, z4, z12, z34, &
                                    bdl, bdu, dil, diu, &
                                    exphl, expl, exphu, expu, &
                                    alogav, &
                                    alog1, alog2, alog3, alog4, aloga, &
                                    alogu, alogl, u2, &
                                    u1, u3, u4, u12, u34, &
                                    cu, cl, sg, si, &
                                    richl, richu, straml, strahl, stramu, strahu
      implicit none
!
! arguments
!
      integer, intent(in) :: iter
!
! local variables
!
      integer  :: i
      real(wp) :: xfac, x, rwork, cdmax, tauu, a, b, c, d, taul
      real(wp) :: ca, cai, cbi, cci, cdi, cei, cfi, sg0, si0
      real(wp) :: yu(npoi), yl(npoi)
!
! set stratification factors for lower and upper regions
!
      xfac = 1.0_wp
!
      call fstrat(t34, t12, xfac, q34, q12, z3, z2, &
                  alogl, alogl, alog2, u2, richl, straml, strahl, iter)
!
      call fstrat(t12, ta,  tfac, q12, qa,  z1, za, &
                  alogu, alogu, aloga, ua, richu, stramu, strahu, iter)
!
! eliminate c/d from eq (28), tau_l/rho from (26),(27), to get
! lower-story roughness alogl
!
      do i = 1, npoi
!
        x = ((alog4(i) - alogav(i)) / vonk)**2 * bdl(i)
!
        rwork  = 1.0_wp / expl(i)
        yl(i)  = ((x+1)*expl(i) + (x-1)*rwork) &
               / ((x+1)*expl(i) - (x-1)*rwork)
!
        alogl(i) = alog3(i) - vonk * sqrt(yl(i) / bdl(i))
!
      end do
!
! eliminate tau_l/rho from (24),(25), tau_u/rho and a/b from
! (22),(23), to get upper-story roughness alogu
!
      do i = 1, npoi
!
        x = ((alog2(i) - alogl(i)) / vonk)**2 * bdu(i) / straml(i)
!
        rwork  = 1.0_wp / expu(i)
        yu(i)  = ((x+1)*expu(i) + (x-1)*rwork) &
               / ((x+1)*expu(i) - (x-1)*rwork)
!
        alogu(i) = alog1(i) - vonk * sqrt(yu(i) / bdu(i))
!
      end do
!
! define the maximum value of cd
!
      cdmax = 300.0_wp / (2.0_wp * dtime)
!
! get wind speeds at various levels
!
      do i = 1, npoi
!
        tauu = (ua(i) * vonk / (aloga(i) - alogu(i)))**2 * stramu(i)
!
        a = 0.5_wp * tauu * (yu(i)+1) / bdu(i)
        b = 0.5_wp * tauu * (yu(i)-1) / bdu(i)
!
        taul = bdu(i) * (a/expu(i) - b*expu(i))
!
        c = 0.5_wp * taul * (yl(i)+1) / bdl(i)
        d = 0.5_wp * taul * (yl(i)-1) / bdl(i)
!
        u1(i)  = max(0.01_wp, sqrt(max(0.0_wp, (a+b))))
        u12(i) = max(0.01_wp, sqrt(max(0.0_wp, (a/exphu(i)+b*exphu(i)))))
        u2(i)  = max(0.01_wp, sqrt(max(0.0_wp, (a/expu(i) +b*expu(i)))))
        u3(i)  = max(0.01_wp, sqrt(max(0.0_wp, (c+d))))
        u34(i) = max(0.01_wp, sqrt(max(0.0_wp, (c/exphl(i)+d*exphl(i)))))
        u4(i)  = max(0.01_wp, sqrt(max(0.0_wp, (c/expl(i) +d*expl(i)))))
!
      end do
!
! compute inverse air-air transfer coefficients
!
      do i = 1, npoi
!
        ca = ua(i)*strahu(i)*vonk**2 / &
             ((aloga(i)-alogu(i)) * (aloga(i)-alog1(i)))
!
        ca  = min(cdmax, ca / (1.0_wp + ca * 1.0e-20_wp))
        cai = 1.0_wp / (rhoa(i)*ca)
!
        cbi = diu(i) * (z1(i)-z12(i))  / (rhoa(i) * 0.5_wp*(u1(i)+u12(i)))
        cci = diu(i) * (z12(i)-z2(i))  / (rhoa(i) * 0.5_wp*(u12(i)+u2(i)))
!
        cdi = (alog2(i)-alogl(i)) * (alog2(i)-alog3(i)) / &
              (rhoa(i)*u2(i)*strahl(i)*vonk**2)
!
        cei = dil(i) * (z3(i)-z34(i))  / (rhoa(i) * 0.5_wp*(u3(i)+u34(i)))
        cfi = dil(i) * (z34(i)-z4(i))  / (rhoa(i) * 0.5_wp*(u34(i)+u4(i)))
!
        cu(i) = 1.0_wp / (cai + cbi)
        cl(i) = 1.0_wp / (cci + cdi + cei)
!
! air-solid transfer coefficients for upper leaves, stems, lower story
!
        su(i) = rhoa(i) * cleaf  * sqrt(u12(i) / dleaf(2))
        ss(i) = rhoa(i) * cstem  * sqrt(u12(i) / dstem(2))
        sl(i) = rhoa(i) * cgrass * sqrt(u34(i) / dleaf(1))
!
! air-solid transfer coefficients for soil and snow
!
        sg0 = rhoa(i) * (0.004_wp + 0.012_wp * u4(i))
        si0 = rhoa(i) * (0.003_wp + 0.010_wp * u4(i))
!
        sg(i) = 1.0_wp / (cfi + 1.0_wp / sg0)
        si(i) = 1.0_wp / (cfi + 1.0_wp / si0)
!
      end do
!
      end subroutine turcof

! =============================================================================
      subroutine turvap(iter, niter)
! =============================================================================
!
! solves canopy system with linearized implicit sensible heat and
! moisture fluxes
!
      use mod_precision,      only: wp
      use mod_dimensions,     only: npoi, nsoilay
      use mod_constants,      only: epsilon, tmelt, hvap, ch2o, cice, &
                                    stef, rvap, grav, rhow
      use mod_run_params,     only: dtime
      use mod_atm_state,      only: ta, qa, psurf
      use mod_snow_state,     only: fi, consno, hsnotop, hsno, tsno
      use mod_soil_state,     only: wpud, wipud, wpudmax, wisoi, wsoi, poros, &
                                    swilt, suction, bex, hvasug, hvasui, &
                                    csoi, rhosoi, tsoi, consoi, hsoi, &
                                    upsoiu, upsoil, stressl, stressu, &
                                    stresstl, stresstu, tg, ti
      use mod_hydrology,      only: ginvap, gsuvap, gtrans, gtransu, gtransl
      use mod_veg_dynamics,   only: lai, sai, fu, fl, frac
      use mod_canopy_params,  only: tl, tu, ts, t12, t34, q12, q34, &
                                    su, sl, ss, &
                                    wliqu, wliql, wliqs, wsnou, wsnol, wsnos, &
                                    chu, chs, chl, &
                                    totcondub, totconduc, totcondls, &
                                    totcondl3, totcondl4
      use mod_canopy_fluxes,  only: cp, tfac, cu, cl, sg, si, &
                                    fwetu, fwetl, fwets, &
                                    fwetux, fwetsx, fwetlx, &
                                    rliqu, rliql, rliqs, &
                                    firb, firg, firi, firl, firs, firu, &
                                    solg, soli, solu, sols, soll, &
                                    pfluxu, pfluxs, pfluxl, &
                                    fsena, fvapa, fsenu, fsens, fsenl, fseng, fseni, &
                                    fvapuw, fvaput, fvaps, fvaplw, fvaplt, fvapg, fvapi
      use mod_thermo_funcs,   only: esat, qsat, dqsat, hvapf, hsubf
      implicit none
!
! arguments
!
      integer, intent(in) :: iter, niter
!
! local variables
!
      integer :: i, k
      real(wp) :: rwork, zwtot, rwork2, tgav, tiav, tuav, &
                  tsav, tlav, quav, qsav, qlav, qgav, qiav, zwpud, zwsoi, &
                  psig, hfac, hfac2, zwopt, zwdry, betaw, emisoil, e, qs1, &
                  dqs1, xnumer, xdenom, betafac, betas
!
      real(wp) :: xu(npoi), xs(npoi), xl(npoi)
      real(wp) :: chux(npoi), chsx(npoi), chlx(npoi)
      real(wp) :: chgx(npoi), wlgx(npoi), wigx(npoi)
      real(wp) :: fradu(npoi), frads(npoi), fradl(npoi)
      real(wp) :: wu(npoi), ws(npoi), wl(npoi), wg(npoi), wi(npoi)
      real(wp) :: qu(npoi), qs(npoi), ql(npoi), qg(npoi), qi(npoi)
      real(wp) :: dqu(npoi), dqs(npoi), dql(npoi), dqg(npoi), dqi(npoi)
      real(wp) :: tuold(npoi), tsold(npoi), tlold(npoi), tgold(npoi), tiold(npoi)
      real(wp) :: tupre(npoi), tspre(npoi), tlpre(npoi), tgpre(npoi), tipre(npoi)
      real(wp) :: suw(npoi), ssw(npoi), slw(npoi)
      real(wp) :: sut(npoi), slt(npoi), slt0(npoi)
      real(wp) :: suh(npoi), ssh(npoi), slh(npoi)
      real(wp) :: cog(npoi), coi(npoi), zirg(npoi), ziri(npoi)
      real(wp) :: qgfac(npoi), qgfac0(npoi)
!
      save xu, xs, xl, chux, chsx, chlx, chgx, wlgx, wigx
      save cog, coi, zirg, ziri
      save wu, ws, wl, wg, wi
      save tuold, tsold, tlold, tgold, tiold
!
      integer,  parameter :: nqn = 9
      integer,  parameter :: mplate(nqn,nqn) = reshape( &
        [1, 0, 0, 1, 0, 1, 0, 0, 0,  &  ! col 1: tu
         0, 1, 0, 1, 0, 1, 0, 0, 0,  &  ! col 2: ts
         0, 0, 1, 0, 1, 0, 1, 0, 0,  &  ! col 3: tl
         1, 1, 0, 1, 1, 0, 0, 0, 0,  &  ! col 4: t12
         0, 0, 1, 1, 1, 0, 0, 1, 1,  &  ! col 5: t34
         1, 1, 0, 0, 0, 1, 1, 0, 0,  &  ! col 6: q12
         0, 0, 1, 0, 0, 1, 1, 1, 1,  &  ! col 7: q34
         0, 0, 0, 0, 1, 0, 1, 1, 0,  &  ! col 8: tg
         0, 0, 0, 0, 1, 0, 1, 0, 1], &  ! col 9: ti
        [nqn, nqn])
!
      real(wp) :: arr(npoi,nqn,nqn), rhs(npoi,nqn), vec(npoi,nqn)
!
! if first iteration, save original canopy temps and initialize
! soil/snow skin temps
!
      if (iter == 1) then
!
!       weights for canopy coverages
!
        do i = 1, npoi
          xu(i) = 2.0_wp * lai(i,2) * fu(i)
          xs(i) = 2.0_wp * sai(i,2) * fu(i)
          xl(i) = 2.0_wp * (lai(i,1) + sai(i,1)) * fl(i) * (1.0_wp - fi(i))
        end do
!
!       specific heats per leaf/stem area
!
        do i = 1, npoi
          chux(i) = chu + ch2o * wliqu(i) + cice * wsnou(i)
          chsx(i) = chs + ch2o * wliqs(i) + cice * wsnos(i)
          chlx(i) = chl + ch2o * wliql(i) + cice * wsnol(i)
        end do
!
        do i = 1, npoi
          rwork = poros(i,1) * rhow
          chgx(i) = ch2o * wpud(i) + cice * wipud(i) &
                  + ((1.0_wp - poros(i,1))*csoi(i,1)*rhosoi(i,1) &
                  +  rwork*(1.0_wp - wisoi(i,1))*wsoi(i,1)*ch2o &
                  +  rwork*wisoi(i,1)*cice &
                    ) * hsoi(1)
!
          wlgx(i) = wpud(i) + &
                    rwork * (1.0_wp - wisoi(i,1)) * wsoi(i,1) * hsoi(1)
!
          wigx(i) = wipud(i) + rwork * wisoi(i,1) * hsoi(1)
        end do
!
!       conductivity coefficients between ground skin and first layer
!
        do i = 1, npoi
          cog(i) = consoi(i,1) / (0.5_wp * hsoi(1))
          coi(i) = consno      / (0.5_wp * max(hsno(i,1), hsnotop))
        end do
!
!       d(ir emitted)/dt for soil
!
        rwork = 4.0_wp * 0.95_wp * stef
!
        do i = 1, npoi
          zirg(i) = rwork * (tg(i)**3)
          ziri(i) = rwork * (ti(i)**3)
        end do
!
!       save temperature memory
!
        do i = 1, npoi
          tuold(i) = tu(i)
          tsold(i) = ts(i)
          tlold(i) = tl(i)
          tgold(i) = tg(i)
          tiold(i) = ti(i)
        end do
!
      end if
!
! set implicit/explicit factors
!
      call impexp(wu, tu, chux, wliqu, wsnou, iter)
      call impexp(ws, ts, chsx, wliqs, wsnos, iter)
      call impexp(wl, tl, chlx, wliql, wsnol, iter)
      call impexp(wg, tg, chgx, wlgx,  wigx,  iter)
      call impexp2(wi, ti, tiold, iter)
!
! adjust t* for this iteration
!
      do i = 1, npoi
        tu(i) = wu(i) * tu(i) + (1.0_wp - wu(i)) * tuold(i)
        ts(i) = ws(i) * ts(i) + (1.0_wp - ws(i)) * tsold(i)
        tl(i) = wl(i) * tl(i) + (1.0_wp - wl(i)) * tlold(i)
        tg(i) = wg(i) * tg(i) + (1.0_wp - wg(i)) * tgold(i)
        ti(i) = wi(i) * ti(i) + (1.0_wp - wi(i)) * tiold(i)
      end do
!
! save current "central" values for final flux calculations
!
      do i = 1, npoi
        tupre(i) = tu(i)
        tspre(i) = ts(i)
        tlpre(i) = tl(i)
        tgpre(i) = tg(i)
        tipre(i) = ti(i)
      end do
!
! calculate specific humidities and derivatives
!
      do i = 1, npoi
!
        e      = esat(tu(i))
        qu(i)  = qsat(e, psurf(i))
        dqu(i) = dqsat(tu(i), qu(i))
        dqu(i) = min(dqu(i), qu(i) * 0.1_wp)
!
        e      = esat(ts(i))
        qs(i)  = qsat(e, psurf(i))
        dqs(i) = dqsat(ts(i), qs(i))
        dqs(i) = min(dqs(i), qs(i) * 0.1_wp)
!
        e      = esat(tl(i))
        ql(i)  = qsat(e, psurf(i))
        dql(i) = dqsat(tl(i), ql(i))
        dql(i) = min(dql(i), ql(i) * 0.1_wp)
!
        e      = esat(tg(i))
        qg(i)  = qsat(e, psurf(i))
        dqg(i) = dqsat(tg(i), qg(i))
        dqg(i) = min(dqg(i), qg(i) * 0.1_wp)
!
        e      = esat(ti(i))
        qi(i)  = qsat(e, psurf(i))
        dqi(i) = dqsat(ti(i), qi(i))
        dqi(i) = min(dqi(i), qi(i) * 0.1_wp)
!
      end do
!
! set qgfac0 (soil surface humidity factor)
!
      do i = 1, npoi
!
        zwpud = max(0.0_wp, min(0.5_wp, 0.5_wp*(wpud(i)+wipud(i))/wpudmax))
        zwsoi = (1.0_wp - wisoi(i,1)) * wsoi(i,1) + wisoi(i,1)
        zwtot = zwpud + (1.0_wp - zwpud) * zwsoi
!
        psig  = -grav * suction(i,1) * (zwtot ** (-bex(i,1)))
        hfac  = exp(psig / (rvap * tg(i)))
        hfac2 = q34(i) / qg(i)
!
        zwopt   = 1.0_wp
        zwdry   = swilt(i,1)
        betaw   = max(0.0_wp, min(1.0_wp, (zwtot - zwdry)/(zwopt - zwdry)))
!
        emisoil  = 0.95_wp
        e        = esat(t34(i))
        qs1      = qsat(e, psurf(i))
        dqs1     = dqsat(t34(i), qs1)
        xnumer   = hvap * dqs1
        xdenom   = cp(i) + (4.0_wp * emisoil * stef * (t34(i))**3) / sg(i)
        betafac  = xnumer / xdenom
        betas    = betaw / (1.0_wp + betafac * (1.0_wp - betaw))
!
        qgfac0(i) = betas * hfac + (1.0_wp - betas) * hfac2
!
      end do
!
! set wet-fraction modifiers and absorbed radiative fluxes
!
      do i = 1, npoi
!
        fwetux(i) = fwetu(i)
        if (q12(i) > qu(i)) fwetux(i) = 1.0_wp
!
        fwetsx(i) = fwets(i)
        if (q12(i) > qs(i)) fwetsx(i) = 1.0_wp
!
        fwetlx(i) = fwetl(i)
        if (q34(i) > ql(i)) fwetlx(i) = 1.0_wp
!
        qgfac(i) = qgfac0(i)
        if (q34(i) > qg(i)) qgfac(i) = 1.0_wp
!
        fradu(i) = 0.0_wp
        if (lai(i,2) > epsilon) &
          fradu(i) = (solu(i) + firu(i)) / (2.0_wp * lai(i,2))
!
        frads(i) = 0.0_wp
        if (sai(i,2) > epsilon) &
          frads(i) = (sols(i) + firs(i)) / (2.0_wp * sai(i,2))
!
        fradl(i) = 0.0_wp
        if ((lai(i,1)+sai(i,1)) > epsilon) &
          fradl(i) = (soll(i) + firl(i)) / (2.0_wp * (lai(i,1) + sai(i,1)))
!
      end do
!
! canopy-air moisture transfer coefficients
!
      do i = 1, npoi
!
!       upper leaves (wet evaporation + transpiration + sensible heat)
!
        suw(i) = min(fwetux(i) * su(i), &
                     0.8_wp * (wliqu(i) + wsnou(i)) / &
                     max(dtime * (qu(i) - q12(i)), epsilon))
!
        sut(i) = (1.0_wp - fwetux(i)) * 0.5_wp * &
                 (totcondub(i) * frac(i,1) + &
                  totcondub(i) * frac(i,2) + &
                  totcondub(i) * frac(i,3) + &
                  totconduc(i) * frac(i,4) + &
                  totcondub(i) * frac(i,5) + &
                  totconduc(i) * frac(i,6) + &
                  totcondub(i) * frac(i,7) + &
                  totcondub(i) * frac(i,8))
        sut(i) = max(0.0_wp, sut(i))
!
        suh(i) = suw(i) * (     rliqu(i)  * hvapf(tu(i),ta(i)) + &
                           (1.0_wp-rliqu(i)) * hsubf(tu(i),ta(i))) + &
                 sut(i) *                     hvapf(tu(i),ta(i))
!
!       upper stems
!
        ssw(i) = min(fwetsx(i) * ss(i), &
                     0.8_wp * (wliqs(i) + wsnos(i)) / &
                     max(dtime * (qs(i) - q12(i)), epsilon))
!
        ssh(i) = ssw(i) * (     rliqs(i)  * hvapf(ts(i),ta(i)) + &
                           (1.0_wp-rliqs(i)) * hsubf(ts(i),ta(i)))
!
!       lower veg (wet evaporation + transpiration + sensible heat)
!
        slw(i) = min(fwetlx(i) * sl(i), &
                     0.8_wp * (wliql(i) + wsnol(i)) / &
                     max(dtime * (ql(i) - q34(i)), epsilon))
!
        slt0(i) = (1.0_wp - fwetlx(i)) * 0.5_wp * &
                  (totcondls(i) * frac(i,9)  + &
                   totcondls(i) * frac(i,10) + &
                   totcondl4(i) * frac(i,11) + &
                   totcondl3(i) * frac(i,12))
        slt0(i) = max(0.0_wp, slt0(i))
!
        slt(i) = slt0(i) * lai(i,1) / max(lai(i,1)+sai(i,1), epsilon)
!
        slh(i) = slw(i) * (     rliql(i)  * hvapf(tl(i),ta(i)) + &
                           (1.0_wp-rliql(i)) * hsubf(tl(i),ta(i))) + &
                 slt(i) *                     hvapf(tl(i),ta(i))
!
      end do
!
! set the matrix of coefficients and the right-hand sides
!
      arr = 0.0_wp
      rhs = 0.0_wp
!
      rwork = 1.0_wp / dtime
!
! upper leaf temperature tu
!
      do i = 1, npoi
        rwork2       = su(i)*cp(i)
        arr(i,1,1)   = chux(i)*rwork + wu(i)*rwork2 + wu(i)*suh(i)*dqu(i)
        arr(i,1,4)   = -rwork2
        arr(i,1,6)   = -suh(i)
        rhs(i,1)     = tuold(i)*chux(i)*rwork &
                     - (1.0_wp-wu(i))*rwork2*tu(i) &
                     - suh(i) * (qu(i)-wu(i)*dqu(i)*tu(i)) &
                     + fradu(i) - pfluxu(i)
      end do
!
! upper stem temperature ts
!
      do i = 1, npoi
        rwork2     = ss(i)*cp(i)
        arr(i,2,2) = chsx(i)*rwork + ws(i)*rwork2 + ws(i)*ssh(i)*dqs(i)
        arr(i,2,4) = -rwork2
        arr(i,2,6) = -ssh(i)
        rhs(i,2)   = tsold(i)*chsx(i)*rwork &
                   - (1.0_wp-ws(i))*rwork2*ts(i) &
                   - ssh(i) * (qs(i)-ws(i)*dqs(i)*ts(i)) &
                   + frads(i) - pfluxs(i)
      end do
!
! lower veg temperature tl
!
      do i = 1, npoi
        rwork2     = sl(i)*cp(i)
        arr(i,3,3) = chlx(i)*rwork + wl(i)*rwork2 + wl(i)*slh(i)*dql(i)
        arr(i,3,5) = -rwork2
        arr(i,3,7) = -slh(i)
        rhs(i,3)   = tlold(i)*chlx(i)*rwork &
                   - (1.0_wp-wl(i))*rwork2*tl(i) &
                   - slh(i) * (ql(i)-wl(i)*dql(i)*tl(i)) &
                   + fradl(i) - pfluxl(i)
      end do
!
! upper air temperature t12
!
      do i = 1, npoi
        rwork      = xu(i)*su(i)
        rwork2     = xs(i)*ss(i)
        arr(i,4,1) = -wu(i)*rwork
        arr(i,4,2) = -ws(i)*rwork2
        arr(i,4,4) = cu(i) + cl(i) + rwork + rwork2
        arr(i,4,5) = -cl(i)
        rhs(i,4)   = cu(i)*ta(i)*tfac &
                   + (1.0_wp-wu(i))*rwork*tu(i) &
                   + (1.0_wp-ws(i))*rwork2*ts(i)
      end do
!
! lower air temperature t34
!
      do i = 1, npoi
        rwork      = xl(i)*sl(i)
        rwork2     = fi(i)*si(i)
        arr(i,5,3) = -wl(i)*rwork
        arr(i,5,4) = -cl(i)
        arr(i,5,5) = cl(i) + rwork + (1.0_wp-fi(i))*sg(i) + rwork2
        arr(i,5,8) = -wg(i)*(1.0_wp-fi(i))*sg(i)
        arr(i,5,9) = -wi(i)*rwork2
        rhs(i,5)   = (1.0_wp-wl(i))*rwork             *tl(i) &
                   + (1.0_wp-wg(i))*(1.0_wp-fi(i))*sg(i)*tg(i) &
                   + (1.0_wp-wi(i))*rwork2              *ti(i)
      end do
!
! upper air specific humidity q12
!
      do i = 1, npoi
        rwork      = xu(i)*(suw(i)+sut(i))
        rwork2     = xs(i)*ssw(i)
        arr(i,6,1) = -wu(i)*rwork *dqu(i)
        arr(i,6,2) = -ws(i)*rwork2*dqs(i)
        arr(i,6,6) = cu(i) + cl(i) + rwork + rwork2
        arr(i,6,7) = -cl(i)
        rhs(i,6)   = cu(i)*qa(i) &
                   + rwork  * (qu(i)-wu(i)*dqu(i)*tu(i)) &
                   + rwork2 * (qs(i)-ws(i)*dqs(i)*ts(i))
      end do
!
! lower air specific humidity q34
!
      do i = 1, npoi
        rwork      = xl(i)*(slw(i)+slt(i))
        rwork2     = (1.0_wp-fi(i))*sg(i)
        arr(i,7,3) = -wl(i)*rwork*dql(i)
        arr(i,7,6) = -cl(i)
        arr(i,7,7) = cl(i) + rwork + rwork2 + fi(i)*si(i)
        arr(i,7,8) = -wg(i)*rwork2*qgfac(i)*dqg(i)
        arr(i,7,9) = -wi(i)*fi(i)*si(i)*dqi(i)
        rhs(i,7)   = rwork           * (ql(i)-wl(i)*dql(i)*tl(i)) &
                   + rwork2*qgfac(i) * (qg(i)-wg(i)*dqg(i)*tg(i)) &
                   + fi(i)*si(i)     * (qi(i)-wi(i)*dqi(i)*ti(i))
      end do
!
! soil skin temperature tg
!
      do i = 1, npoi
        rwork      = sg(i)*cp(i)
        rwork2     = sg(i)*hvasug(i)
        arr(i,8,5) = -rwork
        arr(i,8,7) = -rwork2
        arr(i,8,8) = rwork + rwork2*qgfac(i)*dqg(i) + cog(i) + zirg(i)
        rhs(i,8)   = -rwork2*qgfac(i)*(qg(i)-dqg(i)*tg(i)) &
                   + cog(i)*tsoi(i,1) &
                   + solg(i) + firg(i) + zirg(i) * tgold(i)
      end do
!
! snow skin temperature ti
!
      do i = 1, npoi
        rwork      = si(i)*cp(i)
        rwork2     = si(i)*hvasui(i)
        arr(i,9,5) = -rwork
        arr(i,9,7) = -rwork2
        arr(i,9,9) = rwork + rwork2*dqi(i) + coi(i) + ziri(i)
        rhs(i,9)   = -rwork2*(qi(i)-dqi(i)*ti(i)) &
                   + coi(i)*tsno(i,1) &
                   + soli(i) + firi(i) + ziri(i) * tiold(i)
      end do
!
! solve the systems of equations
!
      call linsolve(arr, rhs, vec, mplate, nqn)
!
! copy this iteration's solution to t*, q12, q34
!
      do i = 1, npoi
        tu(i)  = vec(i,1)
        ts(i)  = vec(i,2)
        tl(i)  = vec(i,3)
        t12(i) = vec(i,4)
        t34(i) = vec(i,5)
        tg(i)  = vec(i,8)
        ti(i)  = vec(i,9)
        q12(i) = vec(i,6)
        q34(i) = vec(i,7)
      end do
!
! all done except for final flux calculations,
! so loop back for the next iteration (except the last)
!
      if (iter < niter) return
!
! evaluate sensible heat fluxes
!
      do i = 1, npoi
!
        fsena(i) = cp(i) * cu(i) * (ta(i)*tfac - t12(i))
!
        tgav     = wg(i)*tg(i) + (1.0_wp-wg(i))*tgpre(i)
        fseng(i) = cp(i) * sg(i) * (tgav - t34(i))
!
        tiav     = wi(i)*ti(i) + (1.0_wp-wi(i))*tipre(i)
        fseni(i) = cp(i) * si(i) * (tiav - t34(i))
!
        tuav     = wu(i)*tu(i) + (1.0_wp - wu(i))*tupre(i)
        fsenu(i) = cp(i) * su(i) * (tuav - t12(i))
!
        tsav     = ws(i)*ts(i) + (1.0_wp - ws(i))*tspre(i)
        fsens(i) = cp(i) * ss(i) * (tsav - t12(i))
!
        tlav     = wl(i)*tl(i) + (1.0_wp - wl(i))*tlpre(i)
        fsenl(i) = cp(i) * sl(i) * (tlav - t12(i))
!
      end do
!
! calculate moisture fluxes
!
      do i = 1, npoi
!
        fvapa(i)  = cu(i) * (qa(i) - q12(i))
!
        quav      = qu(i) + wu(i)*dqu(i)*(tu(i) - tupre(i))
        fvapuw(i) = suw(i) * (quav - q12(i))
        fvaput(i) = max(0.0_wp, sut(i) * (quav - q12(i)))
!
        qsav      = qs(i) + ws(i)*dqs(i)*(ts(i) - tspre(i))
        fvaps(i)  = ssw(i) * (qsav - q12(i))
!
        qlav      = ql(i) + wl(i)*dql(i)*(tl(i) - tlpre(i))
        fvaplw(i) = slw(i)  * (qlav - q34(i))
        fvaplt(i) = max(0.0_wp, slt0(i) * (qlav - q34(i)))
!
        qgav      = qg(i) + wg(i)*dqg(i)*(tg(i) - tgpre(i))
        fvapg(i)  = sg(i) * (qgfac(i)*qgav - q34(i))
!
        qiav      = qi(i) + wi(i)*dqi(i)*(ti(i) - tipre(i))
        fvapi(i)  = si(i) * (qiav - q34(i))
!
      end do
!
! adjust IR fluxes and constrain snow skin temperature
!
      do i = 1, npoi
        firg(i) = firg(i) - wg(i)*zirg(i)*(tg(i) - tgold(i))
        firi(i) = firi(i) - wi(i)*ziri(i)*(ti(i) - tiold(i))
        firb(i) = firb(i) + (1.0_wp-fi(i))*wg(i)*zirg(i)*(tg(i)-tgold(i)) &
                           +          fi(i) *wi(i)*ziri(i)*(ti(i)-tiold(i))
        ti(i) = min(ti(i), tmelt)
      end do
!
! set actual soil water uptake rates
!
      do k = 1, nsoilay
        do i = 1, npoi
          upsoiu(i,k) = fvaput(i) * 2.0_wp * lai(i,2) * fu(i) * &
                        stressu(i,k) / max(stresstu(i), epsilon)
!
          upsoil(i,k) = fvaplt(i) * 2.0_wp * lai(i,1) * fl(i) * &
                        (1.0_wp - fi(i)) * &
                        stressl(i,k) / max(stresstl(i), epsilon)
        end do
      end do
!
! set net evaporation rates
!
      do i = 1, npoi
!
        ginvap(i) = fvapuw(i) * 2.0_wp * lai(i,2) * fu(i) + &
                    fvaps(i)  * 2.0_wp * sai(i,2) * fu(i) + &
                    fvaplw(i) * 2.0_wp * (lai(i,1) + sai(i,1)) * &
                                         fl(i) * (1.0_wp - fi(i))
!
        gsuvap(i)  = fvapg(i) * (1.0_wp - fi(i)) + fvapi(i) * fi(i)
!
        gtrans(i)  = fvaput(i) * 2.0_wp * lai(i,2) * fu(i) + &
                     fvaplt(i) * 2.0_wp * lai(i,1) * fl(i) * (1.0_wp-fi(i))
!
        gtransu(i) = fvaput(i) * 2.0_wp * lai(i,2) * fu(i)
        gtransl(i) = fvaplt(i) * 2.0_wp * lai(i,1) * fl(i) * (1.0_wp-fi(i))
!
      end do
!
      end subroutine turvap

! =============================================================================
      subroutine fstrat(tb, tt, ttfac, qb, qt, zb, zt, &
                        albm, albh, alt, u, rich, stram, strah, iter)
! =============================================================================
!
! computes mixing-length stratification correction factors
! for momentum and heat/vapor (Louis 1979)
!
      use mod_precision,   only: wp
      use mod_dimensions,  only: npoi
      use mod_constants,   only: grav, vonk
      implicit none
!
! arguments
!
      integer, intent(in) :: iter
      real(wp), intent(in)    :: ttfac
      real(wp), intent(in)    :: tb(npoi), tt(npoi), qb(npoi), qt(npoi)
      real(wp), intent(in)    :: zb(npoi), zt(npoi)
      real(wp), intent(in)    :: albm(npoi), albh(npoi), alt(npoi)
      real(wp), intent(in)    :: u(npoi)
      real(wp), intent(inout) :: rich(npoi), stram(npoi), strah(npoi)
!
! local variables
!
      integer :: i, j, np, nq
      integer :: indp(npoi), indq(npoi)
      real(wp) :: zht, zhb, xm, xh, rwork, ym, yh, z, w
      real(wp) :: stramx(npoi), strahx(npoi)
!
      np = 0
      nq = 0
!
! calculate Richardson numbers
!
      do i = 1, npoi
        zht = tt(i)*ttfac*(1.0_wp + 0.622_wp*qt(i))
        zhb = tb(i)*        (1.0_wp + 0.622_wp*qb(i))
        rich(i) = grav * max(zt(i)-zb(i), 0.0_wp) &
                * (zht-zhb) / (0.5_wp*(zht+zhb) * u(i)**2)
        rich(i) = max(-2.0_wp, min(rich(i), 1.0_wp))
      end do
!
! set up indices for negative/positive ri
!
      do i = 1, npoi
        if (rich(i) <= 0.0_wp) then
          np = np + 1
          indp(np) = i
        else
          nq = nq + 1
          indq(nq) = i
        end if
      end do
!
! momentum and heat factors for negative ri (unstable)
!
      if (np > 0) then
        do j = 1, np
          i    = indp(j)
          xm   = max(alt(i)-albm(i), 0.5_wp)
          xh   = max(alt(i)-albh(i), 0.5_wp)
          rwork = sqrt(-rich(i))
          ym   = (vonk/xm)**2 * exp(0.5_wp*xm) * rwork
          yh   = (vonk/xh)**2 * exp(0.5_wp*xh) * rwork
          stramx(i) = 1.0_wp - 2*5*rich(i) / (1.0_wp + 75*ym)
          strahx(i) = 1.0_wp - 3*5*rich(i) / (1.0_wp + 75*yh)
        end do
      end if
!
! momentum and heat factors for positive ri (stable)
!
      if (nq > 0) then
        do j = 1, nq
          i    = indq(j)
          z    = sqrt(1.0_wp + 5 * rich(i))
          stramx(i) = 1.0_wp / (1.0_wp + 2*5*rich(i) / z)
          strahx(i) = 1.0_wp / (1.0_wp + 3*5*rich(i) * z)
        end do
      end if
!
! weight with previous iteration values (except first iteration)
!
      if (iter == 1) then
        do i = 1, npoi
          stram(i) = stramx(i)
          strah(i) = strahx(i)
        end do
      else
        w = 0.5_wp
        do i = 1, npoi
          stram(i) = w * stramx(i) + (1.0_wp - w) * stram(i)
          strah(i) = w * strahx(i) + (1.0_wp - w) * strah(i)
        end do
      end if
!
      end subroutine fstrat

! =============================================================================
      subroutine impexp(wimp, tveg, ch, wliq, wsno, iter)
! =============================================================================
!
! sets the implicit vs explicit fraction for veg surfaces
!
      use mod_precision,  only: wp
      use mod_dimensions, only: npoi
      use mod_constants,  only: epsilon, hfus, tmelt
      implicit none
!
! arguments
!
      integer, intent(in) :: iter
      real(wp), intent(inout) :: wimp(npoi)
      real(wp), intent(in)    :: tveg(npoi), ch(npoi)
      real(wp), intent(in)    :: wliq(npoi), wsno(npoi)
!
! local variables
!
      integer  :: i
      real(wp) :: h, z, winew
!
      if (iter == 1) then
        wimp = 1.0_wp
        return
      end if
!
      do i = 1, npoi
        h     = ch(i) * (tveg(i) - tmelt)
        z     = max(abs(h), epsilon)
        winew = 1.0_wp
        if (h >  epsilon) winew = 1.0_wp - min(1.0_wp, hfus * wsno(i) / z)
        if (h < -epsilon) winew = 1.0_wp - min(1.0_wp, hfus * wliq(i) / z)
        wimp(i) = 0.5_wp * (wimp(i) + winew)
      end do
!
      end subroutine impexp

! =============================================================================
      subroutine impexp2(wimp, t, told, iter)
! =============================================================================
!
! sets the implicit vs explicit fraction for snow skin temperature
!
      use mod_precision,  only: wp
      use mod_dimensions, only: npoi
      use mod_constants,  only: epsilon, tmelt
      implicit none
!
! arguments
!
      integer, intent(in) :: iter
      real(wp), intent(inout) :: wimp(npoi)
      real(wp), intent(in)    :: t(npoi), told(npoi)
!
! local variables
!
      integer :: i
!
      if (iter == 1) then
        wimp = 1.0_wp
        return
      end if
!
      do i = 1, npoi
        if ((t(i)-told(i)) > epsilon) &
          wimp(i) = (tmelt - told(i)) / (t(i) - told(i))
        wimp(i) = max(0.0_wp, min(1.0_wp, wimp(i)))
      end do
!
      end subroutine impexp2

! =============================================================================
      subroutine fwetcal
! =============================================================================
!
! calculates fwet[u,s,l], the fractional areas wetted by
! intercepted h2o, and rliq[u,s,l], the proportion due to liquid
!
      use mod_precision,     only: wp
      use mod_dimensions,    only: npoi
      use mod_constants,     only: epsilon, tmelt
      use mod_canopy_params, only: tl, tu, ts, &
                                   wliqu, wliql, wliqs, wsnou, wsnol, wsnos, &
                                   wliqumax, wsnoumax, wliqsmax, wsnosmax, &
                                   wliqlmax, wsnolmax
      use mod_canopy_fluxes, only: fwetu, fwetl, fwets, rliqu, rliql, rliqs
      implicit none
!
! local variables
!
      integer  :: i
      real(wp), parameter :: fmax = 0.25_wp
      real(wp) :: xliq, xtot
!
! upper leaves
!
      do i = 1, npoi
        xliq      = wliqu(i) / max(wliqumax, 0.01_wp)
        xtot      = xliq + wsnou(i) / max(wsnoumax, 0.01_wp)
        fwetu(i)  = min(fmax, xtot)
        rliqu(i)  = xliq / max(xtot, epsilon)
        if (fwetu(i) == 0.0_wp) then
          rliqu(i) = 1.0_wp
          if (tu(i) < tmelt) rliqu(i) = 0.0_wp
        end if
      end do
!
! upper stems
!
      do i = 1, npoi
        xliq      = wliqs(i) / max(wliqsmax, 0.01_wp)
        xtot      = xliq + wsnos(i) / max(wsnosmax, 0.01_wp)
        fwets(i)  = min(fmax, xtot)
        rliqs(i)  = xliq / max(xtot, epsilon)
        if (fwets(i) == 0.0_wp) then
          rliqs(i) = 1.0_wp
          if (ts(i) < tmelt) rliqs(i) = 0.0_wp
        end if
      end do
!
! lower veg
!
      do i = 1, npoi
        xliq      = wliql(i) / max(wliqlmax, 0.01_wp)
        xtot      = xliq + wsnol(i) / max(wsnolmax, 0.01_wp)
        fwetl(i)  = min(fmax, xtot)
        rliql(i)  = xliq / max(xtot, epsilon)
        if (fwetl(i) == 0.0_wp) then
          rliql(i) = 1.0_wp
          if (tl(i) < tmelt) rliql(i) = 0.0_wp
        end if
      end do
!
      end subroutine fwetcal

! =============================================================================
      subroutine cascade
! =============================================================================
!
! steps intercepted h2o due to drip, precip, and min/max limits
!
      use mod_precision,     only: wp
      use mod_dimensions,    only: npoi
      use mod_constants,     only: epsilon, tmelt
      use mod_run_params,    only: dtime, vzero
      use mod_atm_state,     only: ta, qa, psurf, raina, snowa
      use mod_veg_dynamics,  only: lai, sai, fu, fl
      use mod_canopy_params, only: t12, t34, tl, tu, ts, &
                                   wliqu, wliqs, wliql, wsnou, wsnos, wsnol, &
                                   wliqumax, wsnoumax, wliqsmax, wsnosmax, &
                                   wliqlmax, wsnolmax, wliqmin, wsnomin, &
                                   tdripu, tdrips, tdripl, &
                                   tblowu, tblows, tblowl
      use mod_canopy_fluxes, only: rainu, snowu, trainu, tsnowu, &
                                   rainl, snowl, trainl, tsnowl, &
                                   raing, snowg, traing, tsnowg, &
                                   pfluxu, pfluxs, pfluxl
      implicit none
!
! local variables
!
      integer  :: i
      real(wp) :: twetbulb
      real(wp), external :: twet3
      real(wp) :: xai(npoi), rain(npoi), train(npoi), snow(npoi), tsnow(npoi)
      real(wp) :: x1(npoi), x2(npoi), x3(npoi), x4(npoi)
!
! set wliqmin, wsnomin
!
      wliqmin = 0.0010_wp * (dtime/3600.0_wp) * (wliqumax / 0.2_wp)
      wsnomin = 0.0010_wp * (dtime/3600.0_wp) * (wsnoumax / 2.0_wp)
!
! adjust rainfall and snowfall at above-tree level
!
      do i = 1, npoi
        rainu(i) = raina(i)
        if (ta(i) > tmelt) then
          twetbulb = twet3(ta(i), qa(i), psurf(i))
        else
          twetbulb = tmelt
        end if
        trainu(i) = max(twetbulb, tmelt)
        x1(i) = 0.0_wp
        x2(i) = max(t12(i), tmelt)
      end do
!
      call mix(rainu, trainu, rainu, trainu, x1, x2, vzero, vzero)
!
      do i = 1, npoi
        snowu(i)  = snowa(i)
        tsnowu(i) = min(ta(i), tmelt)
        x1(i) = 0.0_wp
        x2(i) = min(t12(i), tmelt)
      end do
!
      call mix(snowu, tsnowu, snowu, tsnowu, x1, x2, vzero, vzero)
!
! set up for upper leaves
!
      do i = 1, npoi
        xai(i)   = 2.0_wp * lai(i,2)
        rain(i)  = rainu(i)
        train(i) = trainu(i)
        snow(i)  = snowu(i)
        tsnow(i) = tsnowu(i)
      end do
!
      call steph2o(tu, wliqu, wsnou, xai, pfluxu, rain, train, snow, tsnow, &
                   tdripu, tblowu, wliqumax, wsnoumax, wliqmin, wsnomin)
!
! set up for upper stems
!
      do i = 1, npoi
        xai(i) = 2.0_wp * sai(i,2)
      end do
!
      call steph2o(ts, wliqs, wsnos, xai, pfluxs, rain, train, snow, tsnow, &
                   tdrips, tblows, wliqsmax, wsnosmax, wliqmin, wsnomin)
!
! adjust rainfall/snowfall at below-tree level
!
      do i = 1, npoi
        x1(i) = fu(i)*rain(i)
        x2(i) = (1.0_wp - fu(i))*rainu(i)
        x3(i) = 0.0_wp
        x4(i) = max(t34(i), tmelt)
      end do
      call mix(rainl, trainl, x1, train, x2, trainu, x3, x4)
!
      do i = 1, npoi
        x1(i) = fu(i)*snow(i)
        x2(i) = (1.0_wp - fu(i))*snowu(i)
        x3(i) = 0.0_wp
        x4(i) = min(t34(i), tmelt)
      end do
      call mix(snowl, tsnowl, x1, tsnow, x2, tsnowu, x3, x4)
!
! set up for lower veg
!
      do i = 1, npoi
        xai(i)   = 2.0_wp * (lai(i,1) + sai(i,1))
        rain(i)  = rainl(i)
        train(i) = trainl(i)
        snow(i)  = snowl(i)
        tsnow(i) = tsnowl(i)
      end do
!
      call steph2o(tl, wliql, wsnol, xai, pfluxl, rain, train, snow, tsnow, &
                   tdripl, tblowl, wliqlmax, wsnolmax, wliqmin, wsnomin)
!
! adjust rainfall/snowfall at soil level
!
      do i = 1, npoi
        x1(i) = fl(i) * rain(i)
        x2(i) = (1.0_wp - fl(i)) * rainl(i)
      end do
      call mix(raing, traing, x1, train, x2, trainl, vzero, vzero)
!
      do i = 1, npoi
        x1(i) = fl(i) * snow(i)
        x2(i) = (1.0_wp - fl(i)) * snowl(i)
      end do
      call mix(snowg, tsnowg, x1, tsnow, x2, tsnowl, vzero, vzero)
!
      end subroutine cascade

! =============================================================================
      subroutine steph2o(tveg, wliq, wsno, xai, pflux, &
                         rain, train, snow, tsnow, &
                         tdrip, tblow, wliqmax, wsnomax, wliqmin, wsnomin)
! =============================================================================
!
! steps intercepted h2o for one canopy component through one timestep
!
      use mod_precision,  only: wp
      use mod_dimensions, only: npoi
      use mod_constants,  only: epsilon, ch2o, cice, tmelt, hfus
      use mod_run_params, only: dtime, vzero
      implicit none
!
! arguments
!
      real(wp), intent(in)    :: tdrip, tblow, wliqmax, wsnomax, wliqmin, wsnomin
      real(wp), intent(in)    :: tveg(npoi), xai(npoi)
      real(wp), intent(inout) :: wliq(npoi), wsno(npoi)
      real(wp), intent(inout) :: pflux(npoi)
      real(wp), intent(inout) :: rain(npoi), train(npoi)
      real(wp), intent(inout) :: snow(npoi), tsnow(npoi)
!
! local variables
!
      integer  :: i
      real(wp) :: rwork, x, rwork2, dw
      real(wp) :: fint(npoi), drip(npoi), blow(npoi)
!
! calculate intercepted precip fraction per unit leaf/stem area
!
      do i = 1, npoi
        if (xai(i) >= epsilon) then
          fint(i) = (1.0_wp - exp(-0.5_wp*xai(i))) / xai(i)
        else
          fint(i) = 0.5_wp
        end if
      end do
!
      rwork = 1.0_wp / dtime
!
      do i = 1, npoi
!
!       liquid
!
        drip(i) = xai(i)*wliq(i)/tdrip
        wliq(i) = wliq(i) * (1.0_wp - dtime/tdrip)
        wliq(i) = wliq(i) + dtime*rain(i)*fint(i)
        pflux(i) = rain(i)*fint(i) * (tveg(i)-train(i))*ch2o
        rain(i)  = rain(i)*(1.0_wp - xai(i)*fint(i))
        x        = wliq(i)
        wliq(i)  = min(wliq(i), wliqmax)
        if (wliq(i) < wliqmin) wliq(i) = 0.0_wp
        drip(i)  = drip(i) + xai(i)*(x - wliq(i))*rwork
!
!       snow
!
        blow(i)  = xai(i)*wsno(i)/tblow
        wsno(i)  = wsno(i) * (1.0_wp - dtime/tblow)
        wsno(i)  = wsno(i) + dtime*snow(i)*fint(i)
        pflux(i) = pflux(i) + snow(i)*fint(i) * (tveg(i)-tsnow(i))*cice
        snow(i)  = snow(i)*(1.0_wp - xai(i)*fint(i))
        x        = wsno(i)
        wsno(i)  = min(wsno(i), wsnomax)
        if (wsno(i) < wsnomin) wsno(i) = 0.0_wp
        blow(i)  = blow(i) + xai(i)*(x - wsno(i))*rwork
!
      end do
!
! change phase below/above melt point
!
      rwork2 = ch2o - cice
!
      do i = 1, npoi
!
!       liquid below freezing
!
        dw = 0.0_wp
        if (tveg(i) < tmelt) dw = wliq(i)
        pflux(i) = pflux(i) + dw * (rwork2*(tmelt-tveg(i)) - hfus) * rwork
        wliq(i)  = wliq(i) - dw
        wsno(i)  = wsno(i) + dw
!
!       snow above freezing
!
        dw = 0.0_wp
        if (tveg(i) > tmelt) dw = wsno(i)
        pflux(i) = pflux(i) + dw * (rwork2*(tveg(i)-tmelt) + hfus) * rwork
        wsno(i)  = wsno(i) - dw
        wliq(i)  = wliq(i) + dw
!
      end do
!
! adjust rainfall/snowfall for interception, drip, blowoff
!
      call mix(rain, train, rain, train, drip, tveg, vzero, vzero)
      call mix(snow, tsnow, snow, tsnow, blow, tveg, vzero, vzero)
!
      end subroutine steph2o

! =============================================================================
      subroutine cascad2
! =============================================================================
!
! removes evaporation from intercepted h2o and does final phase adjustment
!
      use mod_precision,     only: wp
      use mod_dimensions,    only: npoi
      use mod_snow_state,    only: fi
      use mod_veg_dynamics,  only: lai, sai, fu, fl
      use mod_canopy_params, only: tl, ts, tu, &
                                   wliqu, wliqs, wliql, wsnou, wsnos, wsnol, &
                                   chu, chs, chl
      use mod_canopy_fluxes, only: rliqu, rliqs, rliql, fvapuw, fvaps, fvaplw
      implicit none
!
! local variables
!
      integer  :: i
      real(wp) :: fveg(npoi), xai(npoi)
!
! upper leaves
!
      do i = 1, npoi
        fveg(i) = fu(i)
        xai(i)  = 2.0_wp * lai(i,2)
      end do
      call steph2o2(tu, wliqu, wsnou, fveg, xai, rliqu, fvapuw, chu)
!
! upper stems
!
      do i = 1, npoi
        fveg(i) = fu(i)
        xai(i)  = 2.0_wp * sai(i,2)
      end do
      call steph2o2(ts, wliqs, wsnos, fveg, xai, rliqs, fvaps, chs)
!
! lower veg
!
      do i = 1, npoi
        fveg(i) = (1.0_wp - fi(i)) * fl(i)
        xai(i)  = 2.0_wp * (lai(i,1) + sai(i,1))
      end do
      call steph2o2(tl, wliql, wsnol, fveg, xai, rliql, fvaplw, chl)
!
      end subroutine cascad2

! =============================================================================
      subroutine steph2o2(tveg, wliq, wsno, fveg, xai, rliq, fvapw, cveg)
! =============================================================================
!
! removes evaporation from intercepted h2o and does final
! heat-conserving phase adjustment for one veg component
!
      use mod_precision,     only: wp
      use mod_dimensions,    only: npoi
      use mod_constants,     only: ch2o, cice, hfus, tmelt
      use mod_run_params,    only: dtime
      use mod_atm_state,     only: ta
      use mod_canopy_fluxes, only: fvapa, fsena
      use mod_thermo_funcs,  only: hvapf, hsubf
      implicit none
!
! arguments
!
      real(wp), intent(in)    :: cveg
      real(wp), intent(inout) :: tveg(npoi), wliq(npoi), wsno(npoi)
      real(wp), intent(in)    :: fveg(npoi), xai(npoi), rliq(npoi)
      real(wp), intent(in)    :: fvapw(npoi)
!
! local variables
!
      integer  :: i
      real(wp) :: zm, rwork, chav
      real(wp) :: dh(npoi), dw(npoi)
!
! step intercepted h2o due to evaporation/sublimation
!
      do i = 1, npoi
!
        wliq(i) = wliq(i) - dtime *      rliq(i)  * fvapw(i)
        wsno(i) = wsno(i) - dtime * (1.0_wp-rliq(i)) * fvapw(i)
!
        if ((wliq(i) < 0.0_wp .or. wsno(i) < 0.0_wp) &
            .and. fveg(i)*xai(i) > 0.0_wp) then
!
          zm       = max(-wliq(i), 0.0_wp) * fveg(i) * xai(i) / dtime
          fvapa(i) = fvapa(i) + zm
          fsena(i) = fsena(i) - zm * hvapf(tveg(i), ta(i))
          wliq(i)  = max(wliq(i), 0.0_wp)
!
          zm       = max(-wsno(i), 0.0_wp) * fveg(i) * xai(i) / dtime
          fvapa(i) = fvapa(i) + zm
          fsena(i) = fsena(i) - zm * hsubf(tveg(i), ta(i))
          wsno(i)  = max(wsno(i), 0.0_wp)
!
        end if
!
      end do
!
! final heat-conserving correction for liquid/snow below/above melting point
!
      rwork = 1.0_wp / hfus
!
      do i = 1, npoi
!
        chav = cveg + ch2o*wliq(i) + cice*wsno(i)
!
!       correct for liquid below melt point
!
        if (tveg(i) < tmelt .and. wliq(i) > 0.0_wp) then
          dh(i)    = chav*(tmelt - tveg(i))
          dw(i)    = min(wliq(i), max(0.0_wp, dh(i)*rwork))
          wliq(i)  = wliq(i) - dw(i)
          wsno(i)  = wsno(i) + dw(i)
          chav     = cveg + ch2o*wliq(i) + cice*wsno(i)
          tveg(i)  = tmelt - (dh(i) - hfus*dw(i)) / chav
        end if
!
!       correct for snow above melt point
!
        if (tveg(i) > tmelt .and. wsno(i) > 0.0_wp) then
          dh(i)    = chav*(tveg(i) - tmelt)
          dw(i)    = min(wsno(i), max(0.0_wp, dh(i)*rwork))
          wsno(i)  = wsno(i) - dw(i)
          wliq(i)  = wliq(i) + dw(i)
          chav     = cveg + ch2o*wliq(i) + cice*wsno(i)
          tveg(i)  = tmelt + (dh(i) - hfus*dw(i)) / chav
        end if
!
      end do
!
      end subroutine steph2o2

! =============================================================================
      subroutine mix(xm, tm, x1, t1, x2, t2, x3, t3)
! =============================================================================
!
! calorimetrically mixes masses x1, x2, x3 with temperatures
! t1, t2, t3 into combined mass xm with temperature tm
!
      use mod_precision,  only: wp
      use mod_dimensions, only: npoi
      use mod_constants,  only: epsilon
      implicit none
!
! arguments
!
      real(wp), intent(out) :: xm(npoi), tm(npoi)
      real(wp), intent(in)  :: x1(npoi), t1(npoi)
      real(wp), intent(in)  :: x2(npoi), t2(npoi)
      real(wp), intent(in)  :: x3(npoi), t3(npoi)
!
! local variables
!
      integer  :: i
      real(wp) :: xtmp, ytmp, ttmp
!
      do i = 1, npoi
        xtmp = x1(i) + x2(i) + x3(i)
        ytmp = sign(max(abs(xtmp), epsilon), xtmp)
        if (abs(xtmp) >= epsilon) then
          ttmp = (t1(i)*x1(i) + t2(i)*x2(i) + t3(i)*x3(i)) / ytmp
        else
          ttmp = 0.0_wp
          xtmp = 0.0_wp
        end if
        xm(i) = xtmp
        tm(i) = ttmp
      end do
!
      end subroutine mix

! =============================================================================
      subroutine noveg
! =============================================================================
!
! if no veg surfaces exist, set prog vars to nominal values
!
      use mod_precision,     only: wp
      use mod_dimensions,    only: npoi
      use mod_snow_state,    only: fi
      use mod_soil_state,    only: tg, ti
      use mod_veg_dynamics,  only: lai, sai, fu, fl
      use mod_canopy_params, only: tu, ts, tl, &
                                   wliqu, wliqs, wliql, wsnou, wsnos, wsnol
      implicit none
!
! local variables
!
      integer  :: i
      real(wp) :: tav, x, y
!
      do i = 1, npoi
!
        tav = (1.0_wp - fi(i))*tg(i) + fi(i)*ti(i)
!
        if (lai(i,2) == 0.0_wp .or. fu(i) == 0.0_wp) then
          tu(i)    = tav
          wliqu(i) = 0.0_wp
          wsnou(i) = 0.0_wp
        end if
!
        if (sai(i,2) == 0.0_wp .or. fu(i) == 0.0_wp) then
          ts(i)    = tav
          wliqs(i) = 0.0_wp
          wsnos(i) = 0.0_wp
        end if
!
        x = 2.0_wp * (lai(i,1) + sai(i,1))
        y = fl(i) * (1.0_wp - fi(i))
!
        if (x == 0.0_wp .or. y == 0.0_wp) then
          tl(i)    = tav
          wliql(i) = 0.0_wp
          wsnol(i) = 0.0_wp
        end if
!
      end do
!
      end subroutine noveg

! =============================================================================
      real(wp) function twet3(tak, q, p)
! =============================================================================
!
! calculates the wet bulb temperature given air temp, specific
! humidity, and air pressure (approximation, Molling 2000)
!
! INPUT: tak - air temperature (K), q - specific humidity (kg/kg),
!        p   - air pressure (Pa)
! OUTPUT: twet3 - wet bulb temperature (K)
!
      use mod_precision,    only: wp
      use mod_constants,    only: cair, tmelt
      use mod_thermo_funcs, only: esat, qsat, hvapf
      implicit none
!
! arguments
!
      real(wp), intent(in) :: tak, q, p
!
! local variables
!
      integer  :: i
      real(wp) :: ta, twk, twold, diff, twb
!
! air temperature in C
!
      ta = tak - tmelt
!
! first guess for wet bulb temp (C), then K
!
      twet3 = ta * q / qsat(esat(tak), p)
      twk   = twet3 + tmelt
!
! iterate to converge
!
      do i = 1, 20
        twold = twk - tmelt
        twb   = ta - (hvapf(twk, tak) / cair) * (qsat(esat(twk), p) - q)
        diff  = twb - twold
        twk   = twold + 0.2_wp * diff + tmelt
        if (abs(twk - tmelt - twold) < 0.02_wp) exit
      end do
!
      if (i > 20) then
        print *, 'Warning, twet3 failed to converge after 20 iterations!'
        print *, 'twet3, twetold: ', twk, twold+tmelt
        print *, 'twetbulb is being set to the air temperature'
      end if
!
      twet3 = twk
!
      end function twet3
