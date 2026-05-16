c =====================================================================
c canopy_exchange_wrap.f
c
c F77 wrappers for canopy_exchange tractable building blocks:
c   - wrap_canini    -- aerodynamic initialization
c   - wrap_impexp    -- implicit/explicit fraction for veg w/ intercepted water
c   - wrap_impexp2   -- implicit/explicit fraction for snow surface
c   - wrap_fstrat    -- stratification correction (Richardson #)
c
c The big coupled solvers turcof + turvap need a separate wrapper that
c chains canini + drystress + 3*(turcof + stomata + turvap). That's
c documented in verify/canopy_exchange/README.md as a focused-effort item.
c =====================================================================


c ---------------------------------------------------------------------
c wrap_canini -- initialize aerodynamic quantities for one timestep.
c ---------------------------------------------------------------------
      subroutine wrap_canini(
     >    n_in,
     >    ta_in, qa_in, psurf_in, ua_in,
     >    fi_in, z0soi_in, z0sno_in,
     >    fl_in, fu_in,
     >    lai_in, sai_in, ztop_in, zbot_in,
     >    alaiml_in, alaimu_in,
     >    grav_in, rair_in, rvap_in, cair_in, cvap_in, cappa_in,
     >    cp_out, rhoa_out, za_out, tfac_out,
     >    z1_out, z12_out, z2_out, z3_out, z34_out, z4_out,
     >    bdl_out, bdu_out, dil_out, diu_out,
     >    exphl_out, expl_out, exphu_out, expu_out,
     >    displ_out, dispu_out,
     >    alogav_out, alog1_out, alog2_out, alog3_out, alog4_out,
     >    aloga_out, alogu_out, alogl_out, u2_out)
cf2py intent(in)  n_in
cf2py intent(in)  ta_in, qa_in, psurf_in, ua_in
cf2py intent(in)  fi_in, z0soi_in, z0sno_in
cf2py intent(in)  fl_in, fu_in
cf2py intent(in)  lai_in, sai_in, ztop_in, zbot_in
cf2py intent(in)  alaiml_in, alaimu_in
cf2py intent(in)  grav_in, rair_in, rvap_in, cair_in, cvap_in, cappa_in
cf2py intent(out) cp_out, rhoa_out, za_out, tfac_out
cf2py intent(out) z1_out, z12_out, z2_out, z3_out, z34_out, z4_out
cf2py intent(out) bdl_out, bdu_out, dil_out, diu_out
cf2py intent(out) exphl_out, expl_out, exphu_out, expu_out
cf2py intent(out) displ_out, dispu_out
cf2py intent(out) alogav_out, alog1_out, alog2_out, alog3_out, alog4_out
cf2py intent(out) aloga_out, alogu_out, alogl_out, u2_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'comsno.h'
      include 'comsoi.h'
      include 'comveg.h'
      include 'com1d.h'

      integer n_in
      real*8 ta_in(n_in)
      real*8 qa_in(n_in)
      real*8 psurf_in(n_in)
      real*8 ua_in(n_in)
      real*8 fi_in(n_in)
      real*8 z0soi_in(n_in)
      real*8 z0sno_in
      real*8 fl_in(n_in)
      real*8 fu_in(n_in)
      real*8 lai_in(n_in, 2)
      real*8 sai_in(n_in, 2)
      real*8 ztop_in(n_in, 2)
      real*8 zbot_in(n_in, 2)
      real*8 alaiml_in, alaimu_in
      real*8 grav_in, rair_in, rvap_in, cair_in, cvap_in, cappa_in
      real*8 cp_out(n_in), rhoa_out(n_in), za_out(n_in)
      real*8 tfac_out
      real*8 z1_out(n_in), z12_out(n_in), z2_out(n_in)
      real*8 z3_out(n_in), z34_out(n_in), z4_out(n_in)
      real*8 bdl_out(n_in), bdu_out(n_in), dil_out(n_in), diu_out(n_in)
      real*8 exphl_out(n_in), expl_out(n_in)
      real*8 exphu_out(n_in), expu_out(n_in)
      real*8 displ_out(n_in), dispu_out(n_in)
      real*8 alogav_out(n_in)
      real*8 alog1_out(n_in), alog2_out(n_in)
      real*8 alog3_out(n_in), alog4_out(n_in)
      real*8 aloga_out(n_in), alogu_out(n_in), alogl_out(n_in)
      real*8 u2_out(n_in)

      integer i, k

c --- physical constants ---
      grav  = grav_in
      rair  = rair_in
      rvap  = rvap_in
      cair  = cair_in
      cvap  = cvap_in
      cappa = cappa_in
      alaiml = alaiml_in
      alaimu = alaimu_in
      z0sno  = z0sno_in

c --- safe defaults for trailing npoi entries ---
      do i = 1, npoi
        ta(i)    = 288.0
        qa(i)    = 0.008
        psurf(i) = 101325.0
        ua(i)    = 3.0
        fi(i)    = 0.0
        z0soi(i) = 0.01
        fl(i)    = 0.5
        fu(i)    = 0.5
        do k = 1, 2
          lai(i, k)  = 1.0
          sai(i, k)  = 0.1
          ztop(i, k) = 1.0
          zbot(i, k) = 0.05
        end do
      end do
      do i = 1, n_in
        ta(i)     = ta_in(i)
        qa(i)     = qa_in(i)
        psurf(i)  = psurf_in(i)
        ua(i)     = ua_in(i)
        fi(i)     = fi_in(i)
        z0soi(i)  = z0soi_in(i)
        fl(i)     = fl_in(i)
        fu(i)     = fu_in(i)
        do k = 1, 2
          lai(i, k)  = lai_in(i, k)
          sai(i, k)  = sai_in(i, k)
          ztop(i, k) = ztop_in(i, k)
          zbot(i, k) = zbot_in(i, k)
        end do
      end do

      call canini

      tfac_out = tfac
      do i = 1, n_in
        cp_out(i)    = cp(i)
        rhoa_out(i)  = rhoa(i)
        za_out(i)    = za(i)
        z1_out(i)    = z1(i)
        z12_out(i)   = z12(i)
        z2_out(i)    = z2(i)
        z3_out(i)    = z3(i)
        z34_out(i)   = z34(i)
        z4_out(i)    = z4(i)
        bdl_out(i)   = bdl(i)
        bdu_out(i)   = bdu(i)
        dil_out(i)   = dil(i)
        diu_out(i)   = diu(i)
        exphl_out(i) = exphl(i)
        expl_out(i)  = expl(i)
        exphu_out(i) = exphu(i)
        expu_out(i)  = expu(i)
        displ_out(i) = displ(i)
        dispu_out(i) = dispu(i)
        alogav_out(i) = alogav(i)
        alog1_out(i)  = alog1(i)
        alog2_out(i)  = alog2(i)
        alog3_out(i)  = alog3(i)
        alog4_out(i)  = alog4(i)
        aloga_out(i)  = aloga(i)
        alogu_out(i)  = alogu(i)
        alogl_out(i)  = alogl(i)
        u2_out(i)     = u2(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_impexp -- impexp for one canopy layer (tveg/wliq/wsno/ch fields).
c ---------------------------------------------------------------------
      subroutine wrap_impexp(
     >    n_in, iter_in,
     >    wimp_in, tveg_in, ch_in, wliq_in, wsno_in,
     >    tmelt_in, hfus_in, epsilon_in,
     >    wimp_out)
cf2py intent(in)  n_in, iter_in
cf2py intent(in)  wimp_in, tveg_in, ch_in, wliq_in, wsno_in
cf2py intent(in)  tmelt_in, hfus_in, epsilon_in
cf2py intent(out) wimp_out

      include 'implicit.h'
      include 'compar.h'

      integer n_in, iter_in
      real*8  wimp_in(n_in)
      real*8  tveg_in(n_in)
      real*8  ch_in(n_in)
      real*8  wliq_in(n_in)
      real*8  wsno_in(n_in)
      real*8  tmelt_in, hfus_in, epsilon_in
      real*8  wimp_out(n_in)

      real wimp(npoi), tveg(npoi), ch(npoi), wliq(npoi), wsno(npoi)
      integer i

      tmelt   = tmelt_in
      hfus    = hfus_in
      epsilon = epsilon_in

      do i = 1, npoi
        wimp(i) = 1.0
        tveg(i) = 288.0
        ch(i)   = 2000.0
        wliq(i) = 0.0
        wsno(i) = 0.0
      end do
      do i = 1, n_in
        wimp(i) = wimp_in(i)
        tveg(i) = tveg_in(i)
        ch(i)   = ch_in(i)
        wliq(i) = wliq_in(i)
        wsno(i) = wsno_in(i)
      end do

      call impexp(wimp, tveg, ch, wliq, wsno, iter_in)

      do i = 1, n_in
        wimp_out(i) = wimp(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_impexp2 -- impexp for a thin surface layer (no intercepted water).
c ---------------------------------------------------------------------
      subroutine wrap_impexp2(
     >    n_in, iter_in,
     >    wimp_in, t_in, told_in,
     >    tmelt_in, epsilon_in,
     >    wimp_out)
cf2py intent(in)  n_in, iter_in
cf2py intent(in)  wimp_in, t_in, told_in
cf2py intent(in)  tmelt_in, epsilon_in
cf2py intent(out) wimp_out

      include 'implicit.h'
      include 'compar.h'

      integer n_in, iter_in
      real*8  wimp_in(n_in)
      real*8  t_in(n_in)
      real*8  told_in(n_in)
      real*8  tmelt_in, epsilon_in
      real*8  wimp_out(n_in)

      real wimp(npoi), t(npoi), told(npoi)
      integer i

      tmelt   = tmelt_in
      epsilon = epsilon_in

      do i = 1, npoi
        wimp(i) = 1.0
        t(i)    = 288.0
        told(i) = 288.0
      end do
      do i = 1, n_in
        wimp(i) = wimp_in(i)
        t(i)    = t_in(i)
        told(i) = told_in(i)
      end do

      call impexp2(wimp, t, told, iter_in)

      do i = 1, n_in
        wimp_out(i) = wimp(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_fstrat -- Richardson number / stratification correction.
c ---------------------------------------------------------------------
      subroutine wrap_fstrat(
     >    n_in,
     >    tb_in, tt_in, ttfac_in, qb_in, qt_in,
     >    zb_in, zt_in, albm_in, albh_in, alt_in, u_in,
     >    rich_in, stram_in, strah_in, iter_in,
     >    grav_in, vonk_in,
     >    rich_out, stram_out, strah_out)
cf2py intent(in)  n_in, iter_in
cf2py intent(in)  tb_in, tt_in, ttfac_in, qb_in, qt_in
cf2py intent(in)  zb_in, zt_in, albm_in, albh_in, alt_in, u_in
cf2py intent(in)  rich_in, stram_in, strah_in
cf2py intent(in)  grav_in, vonk_in
cf2py intent(out) rich_out, stram_out, strah_out

      include 'implicit.h'
      include 'compar.h'

      integer n_in, iter_in
      real*8  tb_in(n_in), tt_in(n_in), ttfac_in
      real*8  qb_in(n_in), qt_in(n_in)
      real*8  zb_in(n_in), zt_in(n_in)
      real*8  albm_in(n_in), albh_in(n_in), alt_in(n_in)
      real*8  u_in(n_in)
      real*8  rich_in(n_in), stram_in(n_in), strah_in(n_in)
      real*8  grav_in, vonk_in
      real*8  rich_out(n_in), stram_out(n_in), strah_out(n_in)

      real tb(npoi), tt(npoi), ttfac
      real qb(npoi), qt(npoi)
      real zb(npoi), zt(npoi)
      real albm(npoi), albh(npoi), alt(npoi), u(npoi)
      real rich(npoi), stram(npoi), strah(npoi)
      integer i

      grav = grav_in
      vonk = vonk_in
      ttfac = ttfac_in

c Padding designed so Ri > 0 (stable). Pure inversion: cold bottom,
c warm top -> ensures padding contributes to nq (not np). That way,
c an all-stable input scenario can drive np=0 and exercise the
c "if (np.gt.0)" else-branch (otherwise unreachable).
      do i = 1, npoi
        tb(i) = 280.0
        tt(i) = 290.0
        qb(i) = 0.008
        qt(i) = 0.008
        zb(i) = 0.5
        zt(i) = 5.0
        albm(i) = -2.0
        albh(i) = -2.0
        alt(i) = 1.0
        u(i) = 3.0
        rich(i) = 0.1
        stram(i) = 1.0
        strah(i) = 1.0
      end do
      do i = 1, n_in
        tb(i) = tb_in(i)
        tt(i) = tt_in(i)
        qb(i) = qb_in(i)
        qt(i) = qt_in(i)
        zb(i) = zb_in(i)
        zt(i) = zt_in(i)
        albm(i) = albm_in(i)
        albh(i) = albh_in(i)
        alt(i) = alt_in(i)
        u(i) = u_in(i)
        rich(i) = rich_in(i)
        stram(i) = stram_in(i)
        strah(i) = strah_in(i)
      end do

      call fstrat(tb, tt, ttfac, qb, qt, zb, zt,
     >            albm, albh, alt, u,
     >            rich, stram, strah, iter_in)

      do i = 1, n_in
        rich_out(i)  = rich(i)
        stram_out(i) = stram(i)
        strah_out(i) = strah(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_turcof -- wind speeds + transfer coefficients (cu, cl, su, ss, sl,
c sg, si) for one iteration. Internally calls fstrat twice.
c ---------------------------------------------------------------------
      subroutine wrap_turcof(
     >    n_in, iter_in,
c canini outputs needed:
     >    rhoa_in, tfac_in,
     >    z1_in, z12_in, z2_in, z3_in, z34_in, z4_in,
     >    bdl_in, bdu_in, dil_in, diu_in,
     >    exphl_in, expl_in, exphu_in, expu_in,
     >    alogav_in, alog1_in, alog2_in, alog3_in, alog4_in,
     >    aloga_in, alogu_in, alogl_in, u2_in,
c forcing + state:
     >    ta_in, qa_in, ua_in, fi_in,
     >    t12_in, t34_in, q12_in, q34_in,
     >    richl_in, straml_in, strahl_in,
     >    richu_in, stramu_in, strahu_in,
c canopy/snow params:
     >    cleaf_in, cstem_in, cgrass_in,
     >    dleaf_lo, dleaf_up, dstem_lo, dstem_up,
c constants:
     >    grav_in, vonk_in, dtime_in,
c outputs:
     >    cu_out, cl_out, su_out, ss_out, sl_out, sg_out, si_out,
     >    u1_out, u12_out, u2_out, u3_out, u34_out, u4_out,
     >    alogu_out, alogl_out,
     >    richl_out, straml_out, strahl_out,
     >    richu_out, stramu_out, strahu_out)
cf2py intent(in)  n_in, iter_in
cf2py intent(in)  rhoa_in, tfac_in
cf2py intent(in)  z1_in, z12_in, z2_in, z3_in, z34_in, z4_in
cf2py intent(in)  bdl_in, bdu_in, dil_in, diu_in
cf2py intent(in)  exphl_in, expl_in, exphu_in, expu_in
cf2py intent(in)  alogav_in, alog1_in, alog2_in, alog3_in, alog4_in
cf2py intent(in)  aloga_in, alogu_in, alogl_in, u2_in
cf2py intent(in)  ta_in, qa_in, ua_in, fi_in
cf2py intent(in)  t12_in, t34_in, q12_in, q34_in
cf2py intent(in)  richl_in, straml_in, strahl_in
cf2py intent(in)  richu_in, stramu_in, strahu_in
cf2py intent(in)  cleaf_in, cstem_in, cgrass_in
cf2py intent(in)  dleaf_lo, dleaf_up, dstem_lo, dstem_up
cf2py intent(in)  grav_in, vonk_in, dtime_in
cf2py intent(out) cu_out, cl_out, su_out, ss_out, sl_out, sg_out, si_out
cf2py intent(out) u1_out, u12_out, u2_out, u3_out, u34_out, u4_out
cf2py intent(out) alogu_out, alogl_out
cf2py intent(out) richl_out, straml_out, strahl_out
cf2py intent(out) richu_out, stramu_out, strahu_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'comsno.h'
      include 'comsoi.h'
      include 'comveg.h'
      include 'com1d.h'

      integer n_in, iter_in
      real*8 rhoa_in(n_in), tfac_in
      real*8 z1_in(n_in), z12_in(n_in), z2_in(n_in)
      real*8 z3_in(n_in), z34_in(n_in), z4_in(n_in)
      real*8 bdl_in(n_in), bdu_in(n_in), dil_in(n_in), diu_in(n_in)
      real*8 exphl_in(n_in), expl_in(n_in)
      real*8 exphu_in(n_in), expu_in(n_in)
      real*8 alogav_in(n_in)
      real*8 alog1_in(n_in), alog2_in(n_in)
      real*8 alog3_in(n_in), alog4_in(n_in)
      real*8 aloga_in(n_in), alogu_in(n_in), alogl_in(n_in)
      real*8 u2_in(n_in)
      real*8 ta_in(n_in), qa_in(n_in), ua_in(n_in), fi_in(n_in)
      real*8 t12_in(n_in), t34_in(n_in), q12_in(n_in), q34_in(n_in)
      real*8 richl_in(n_in), straml_in(n_in), strahl_in(n_in)
      real*8 richu_in(n_in), stramu_in(n_in), strahu_in(n_in)
      real*8 cleaf_in, cstem_in, cgrass_in
      real*8 dleaf_lo, dleaf_up, dstem_lo, dstem_up
      real*8 grav_in, vonk_in, dtime_in
      real*8 cu_out(n_in), cl_out(n_in)
      real*8 su_out(n_in), ss_out(n_in), sl_out(n_in)
      real*8 sg_out(n_in), si_out(n_in)
      real*8 u1_out(n_in), u12_out(n_in), u2_out(n_in)
      real*8 u3_out(n_in), u34_out(n_in), u4_out(n_in)
      real*8 alogu_out(n_in), alogl_out(n_in)
      real*8 richl_out(n_in), straml_out(n_in), strahl_out(n_in)
      real*8 richu_out(n_in), stramu_out(n_in), strahu_out(n_in)

      integer i

c --- physical constants ---
      grav   = grav_in
      vonk   = vonk_in
      dtime  = dtime_in
      tfac   = tfac_in
      cleaf  = cleaf_in
      cstem  = cstem_in
      cgrass = cgrass_in
      dleaf(1) = dleaf_lo
      dleaf(2) = dleaf_up
      dstem(1) = dstem_lo
      dstem(2) = dstem_up

c --- Safe defaults for padded npoi entries. canini-style geometry:
c     neutral atmosphere, modest veg, so turcof's intermediate
c     quantities (yu, yl, ca, etc.) stay well-conditioned.
      do i = 1, npoi
        rhoa(i)   = 1.2
        z1(i)     = 20.0
        z12(i)    = 15.0
        z2(i)     = 10.0
        z3(i)     = 1.0
        z34(i)    = 0.5
        z4(i)     = 0.05
        bdl(i)    = 0.01
        bdu(i)    = 0.01
        dil(i)    = 10.0
        diu(i)    = 10.0
        exphl(i)  = 1.0
        expl(i)   = 1.0
        exphu(i)  = 1.0
        expu(i)   = 1.0
        alogav(i) = -4.6
        alog1(i)  = 3.0
        alog2(i)  = 2.3
        alog3(i)  = 0.0
        alog4(i)  = -3.0
        aloga(i)  = 3.5
        alogu(i)  = -2.3
        alogl(i)  = -2.3
        u2(i)     = 3.0
        ta(i)     = 288.0
        qa(i)     = 0.008
        ua(i)     = 3.0
        fi(i)     = 0.0
        t12(i)    = 288.0
        t34(i)    = 288.0
        q12(i)    = 0.008
        q34(i)    = 0.008
        richl(i)  = 0.0
        straml(i) = 1.0
        strahl(i) = 1.0
        richu(i)  = 0.0
        stramu(i) = 1.0
        strahu(i) = 1.0
      end do

      do i = 1, n_in
        rhoa(i)   = rhoa_in(i)
        z1(i)     = z1_in(i)
        z12(i)    = z12_in(i)
        z2(i)     = z2_in(i)
        z3(i)     = z3_in(i)
        z34(i)    = z34_in(i)
        z4(i)     = z4_in(i)
        bdl(i)    = bdl_in(i)
        bdu(i)    = bdu_in(i)
        dil(i)    = dil_in(i)
        diu(i)    = diu_in(i)
        exphl(i)  = exphl_in(i)
        expl(i)   = expl_in(i)
        exphu(i)  = exphu_in(i)
        expu(i)   = expu_in(i)
        alogav(i) = alogav_in(i)
        alog1(i)  = alog1_in(i)
        alog2(i)  = alog2_in(i)
        alog3(i)  = alog3_in(i)
        alog4(i)  = alog4_in(i)
        aloga(i)  = aloga_in(i)
        alogu(i)  = alogu_in(i)
        alogl(i)  = alogl_in(i)
        u2(i)     = u2_in(i)
        ta(i)     = ta_in(i)
        qa(i)     = qa_in(i)
        ua(i)     = ua_in(i)
        fi(i)     = fi_in(i)
        t12(i)    = t12_in(i)
        t34(i)    = t34_in(i)
        q12(i)    = q12_in(i)
        q34(i)    = q34_in(i)
        richl(i)  = richl_in(i)
        straml(i) = straml_in(i)
        strahl(i) = strahl_in(i)
        richu(i)  = richu_in(i)
        stramu(i) = stramu_in(i)
        strahu(i) = strahu_in(i)
      end do

      call turcof(iter_in)

      do i = 1, n_in
        cu_out(i)     = cu(i)
        cl_out(i)     = cl(i)
        su_out(i)     = su(i)
        ss_out(i)     = ss(i)
        sl_out(i)     = sl(i)
        sg_out(i)     = sg(i)
        si_out(i)     = si(i)
        u1_out(i)     = u1(i)
        u12_out(i)    = u12(i)
        u2_out(i)     = u2(i)
        u3_out(i)     = u3(i)
        u34_out(i)    = u34(i)
        u4_out(i)     = u4(i)
        alogu_out(i)  = alogu(i)
        alogl_out(i)  = alogl(i)
        richl_out(i)  = richl(i)
        straml_out(i) = straml(i)
        strahl_out(i) = strahl(i)
        richu_out(i)  = richu(i)
        stramu_out(i) = stramu(i)
        strahu_out(i) = strahu(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_turvap -- 9x9 implicit solve for canopy temperatures/humidities
c plus diagnostic sensible/moisture fluxes. iter=1, niter=1 so the
c post-solve flux block runs.
c ---------------------------------------------------------------------
      subroutine wrap_turvap(
     >    n_in,
c canopy state (will also be tuold/tsold/.../tiold inside turvap):
     >    tu_in, ts_in, tl_in, t12_in, t34_in, q12_in, q34_in,
     >    tg_in, ti_in,
c canini outputs:
     >    cp_in, rhoa_in, tfac_in,
c turcof outputs:
     >    cu_in, cl_in, su_in, ss_in, sl_in, sg_in, si_in,
c atmospheric forcing:
     >    ta_in, qa_in, psurf_in, fi_in,
c vegetation geom + interception:
     >    fu_in, fl_in, lai_in, sai_in, frac_in,
     >    wliqu_in, wsnou_in, wliqs_in, wsnos_in,
     >    wliql_in, wsnol_in,
     >    fwetu_in, fwets_in, fwetl_in,
     >    rliqu_in, rliqs_in, rliql_in,
c stomatal outputs:
     >    totcondub_in, totconduc_in,
     >    totcondls_in, totcondl3_in, totcondl4_in,
c radiation forcing:
     >    solu_in, sols_in, soll_in, solg_in, soli_in,
     >    firu_in, firs_in, firl_in, firg_in, firi_in, firb_in,
     >    pfluxu_in, pfluxs_in, pfluxl_in,
c soil thermal at layer 1:
     >    poros1_in, csoi1_in, rhosoi1_in, hsoi1_in,
     >    wpud_in, wipud_in, suction1_in, bex1_in,
     >    swilt1_in, wsoi1_in, wisoi1_in, tsoi1_in, consoi1_in,
c snow thermal:
     >    consno_in, hsno1_in, hsnotop_in, tsno1_in,
c saturation latent heats:
     >    hvasug_in, hvasui_in,
c drystress outputs (for upsoiu/upsoil; turvap reads them):
     >    stresstu_in, stresstl_in, stressu_in, stressl_in,
c canopy params:
     >    chu_in, chs_in, chl_in,
c constants:
     >    wpudmax_in, ch2o_in, cice_in,
     >    hvap_in, hsub_in, cvap_in, stef_in,
     >    rvap_in, grav_in, tmelt_in, epsilon_in, rhow_in, dtime_in,
c outputs - updated state:
     >    tu_out, ts_out, tl_out, t12_out, t34_out,
     >    q12_out, q34_out, tg_out, ti_out,
c outputs - sensible heat fluxes:
     >    fsena_out, fsenu_out, fsens_out,
     >    fsenl_out, fseng_out, fseni_out,
c outputs - moisture fluxes:
     >    fvapa_out, fvapuw_out, fvaput_out, fvaps_out,
     >    fvaplw_out, fvaplt_out, fvapg_out, fvapi_out,
c outputs - IR adjustments:
     >    firg_out, firi_out, firb_out,
c outputs - aggregated:
     >    ginvap_out, gsuvap_out, gtrans_out)
cf2py intent(in)  n_in
cf2py intent(in)  tu_in, ts_in, tl_in, t12_in, t34_in, q12_in, q34_in
cf2py intent(in)  tg_in, ti_in
cf2py intent(in)  cp_in, rhoa_in, tfac_in
cf2py intent(in)  cu_in, cl_in, su_in, ss_in, sl_in, sg_in, si_in
cf2py intent(in)  ta_in, qa_in, psurf_in, fi_in
cf2py intent(in)  fu_in, fl_in, lai_in, sai_in, frac_in
cf2py intent(in)  wliqu_in, wsnou_in, wliqs_in, wsnos_in
cf2py intent(in)  wliql_in, wsnol_in
cf2py intent(in)  fwetu_in, fwets_in, fwetl_in
cf2py intent(in)  rliqu_in, rliqs_in, rliql_in
cf2py intent(in)  totcondub_in, totconduc_in
cf2py intent(in)  totcondls_in, totcondl3_in, totcondl4_in
cf2py intent(in)  solu_in, sols_in, soll_in, solg_in, soli_in
cf2py intent(in)  firu_in, firs_in, firl_in, firg_in, firi_in, firb_in
cf2py intent(in)  pfluxu_in, pfluxs_in, pfluxl_in
cf2py intent(in)  poros1_in, csoi1_in, rhosoi1_in, hsoi1_in
cf2py intent(in)  wpud_in, wipud_in, suction1_in, bex1_in
cf2py intent(in)  swilt1_in, wsoi1_in, wisoi1_in, tsoi1_in, consoi1_in
cf2py intent(in)  consno_in, hsno1_in, hsnotop_in, tsno1_in
cf2py intent(in)  hvasug_in, hvasui_in
cf2py intent(in)  stresstu_in, stresstl_in, stressu_in, stressl_in
cf2py intent(in)  chu_in, chs_in, chl_in
cf2py intent(in)  wpudmax_in, ch2o_in, cice_in
cf2py intent(in)  hvap_in, hsub_in, cvap_in, stef_in
cf2py intent(in)  rvap_in, grav_in, tmelt_in, epsilon_in, rhow_in, dtime_in
cf2py intent(out) tu_out, ts_out, tl_out, t12_out, t34_out
cf2py intent(out) q12_out, q34_out, tg_out, ti_out
cf2py intent(out) fsena_out, fsenu_out, fsens_out
cf2py intent(out) fsenl_out, fseng_out, fseni_out
cf2py intent(out) fvapa_out, fvapuw_out, fvaput_out, fvaps_out
cf2py intent(out) fvaplw_out, fvaplt_out, fvapg_out, fvapi_out
cf2py intent(out) firg_out, firi_out, firb_out
cf2py intent(out) ginvap_out, gsuvap_out, gtrans_out

      include 'implicit.h'
      include 'compar.h'
      include 'comhyd.h'
      include 'comatm.h'
      include 'comsno.h'
      include 'comsoi.h'
      include 'comveg.h'
      include 'com1d.h'

      integer n_in
      real*8 tu_in(n_in), ts_in(n_in), tl_in(n_in)
      real*8 t12_in(n_in), t34_in(n_in)
      real*8 q12_in(n_in), q34_in(n_in)
      real*8 tg_in(n_in), ti_in(n_in)
      real*8 cp_in(n_in), rhoa_in(n_in), tfac_in
      real*8 cu_in(n_in), cl_in(n_in)
      real*8 su_in(n_in), ss_in(n_in), sl_in(n_in)
      real*8 sg_in(n_in), si_in(n_in)
      real*8 ta_in(n_in), qa_in(n_in), psurf_in(n_in), fi_in(n_in)
      real*8 fu_in(n_in), fl_in(n_in)
      real*8 lai_in(n_in,2), sai_in(n_in,2)
      real*8 frac_in(n_in,12)
      real*8 wliqu_in(n_in), wsnou_in(n_in)
      real*8 wliqs_in(n_in), wsnos_in(n_in)
      real*8 wliql_in(n_in), wsnol_in(n_in)
      real*8 fwetu_in(n_in), fwets_in(n_in), fwetl_in(n_in)
      real*8 rliqu_in(n_in), rliqs_in(n_in), rliql_in(n_in)
      real*8 totcondub_in(n_in), totconduc_in(n_in)
      real*8 totcondls_in(n_in)
      real*8 totcondl3_in(n_in), totcondl4_in(n_in)
      real*8 solu_in(n_in), sols_in(n_in), soll_in(n_in)
      real*8 solg_in(n_in), soli_in(n_in)
      real*8 firu_in(n_in), firs_in(n_in), firl_in(n_in)
      real*8 firg_in(n_in), firi_in(n_in), firb_in(n_in)
      real*8 pfluxu_in(n_in), pfluxs_in(n_in), pfluxl_in(n_in)
      real*8 poros1_in(n_in), csoi1_in(n_in), rhosoi1_in(n_in)
      real*8 hsoi1_in
      real*8 wpud_in(n_in), wipud_in(n_in)
      real*8 suction1_in(n_in), bex1_in(n_in)
      real*8 swilt1_in(n_in), wsoi1_in(n_in), wisoi1_in(n_in)
      real*8 tsoi1_in(n_in), consoi1_in(n_in)
      real*8 consno_in, hsno1_in(n_in), hsnotop_in
      real*8 tsno1_in(n_in)
      real*8 hvasug_in(n_in), hvasui_in(n_in)
      real*8 stresstu_in(n_in), stresstl_in(n_in)
      real*8 stressu_in(n_in,6), stressl_in(n_in,6)
      real*8 chu_in, chs_in, chl_in
      real*8 wpudmax_in, ch2o_in, cice_in
      real*8 hvap_in, hsub_in, cvap_in, stef_in
      real*8 rvap_in, grav_in, tmelt_in, epsilon_in, rhow_in, dtime_in

      real*8 tu_out(n_in), ts_out(n_in), tl_out(n_in)
      real*8 t12_out(n_in), t34_out(n_in)
      real*8 q12_out(n_in), q34_out(n_in)
      real*8 tg_out(n_in), ti_out(n_in)
      real*8 fsena_out(n_in), fsenu_out(n_in), fsens_out(n_in)
      real*8 fsenl_out(n_in), fseng_out(n_in), fseni_out(n_in)
      real*8 fvapa_out(n_in), fvapuw_out(n_in), fvaput_out(n_in)
      real*8 fvaps_out(n_in), fvaplw_out(n_in), fvaplt_out(n_in)
      real*8 fvapg_out(n_in), fvapi_out(n_in)
      real*8 firg_out(n_in), firi_out(n_in), firb_out(n_in)
      real*8 ginvap_out(n_in), gsuvap_out(n_in), gtrans_out(n_in)

      integer i, k

c --- physical constants ---
      ch2o    = ch2o_in
      cice    = cice_in
      hvap    = hvap_in
      hsub    = hsub_in
      cvap    = cvap_in
      stef    = stef_in
      rvap    = rvap_in
      grav    = grav_in
      tmelt   = tmelt_in
      epsilon = epsilon_in
      rhow    = rhow_in
      dtime   = dtime_in
      tfac    = tfac_in
      chu     = chu_in
      chs     = chs_in
      chl     = chl_in
      wpudmax = wpudmax_in
      consno  = consno_in
      hsnotop = hsnotop_in

c --- safe defaults for padded entries ---
      do i = 1, npoi
        tu(i)     = 288.0
        ts(i)     = 288.0
        tl(i)     = 288.0
        t12(i)    = 288.0
        t34(i)    = 288.0
        q12(i)    = 0.008
        q34(i)    = 0.008
        tg(i)     = 288.0
        ti(i)     = 273.0
        cp(i)     = 1005.0
        rhoa(i)   = 1.2
        cu(i)     = 0.05
        cl(i)     = 0.05
        su(i)     = 0.05
        ss(i)     = 0.05
        sl(i)     = 0.05
        sg(i)     = 0.05
        si(i)     = 0.05
        ta(i)     = 288.0
        qa(i)     = 0.008
        psurf(i)  = 101325.0
        fi(i)     = 0.0
        fu(i)     = 0.5
        fl(i)     = 0.5
        wliqu(i)  = 0.0
        wsnou(i)  = 0.0
        wliqs(i)  = 0.0
        wsnos(i)  = 0.0
        wliql(i)  = 0.0
        wsnol(i)  = 0.0
        fwetu(i)  = 0.0
        fwets(i)  = 0.0
        fwetl(i)  = 0.0
        rliqu(i)  = 1.0
        rliqs(i)  = 1.0
        rliql(i)  = 1.0
        totcondub(i) = 0.01
        totconduc(i) = 0.01
        totcondls(i) = 0.01
        totcondl3(i) = 0.01
        totcondl4(i) = 0.01
        solu(i)   = 0.0
        sols(i)   = 0.0
        soll(i)   = 0.0
        solg(i)   = 0.0
        soli(i)   = 0.0
        firu(i)   = 0.0
        firs(i)   = 0.0
        firl(i)   = 0.0
        firg(i)   = 0.0
        firi(i)   = 0.0
        firb(i)   = 0.0
        pfluxu(i) = 0.0
        pfluxs(i) = 0.0
        pfluxl(i) = 0.0
        wpud(i)   = 0.0
        wipud(i)  = 0.0
        hvasug(i) = hvap_in
        hvasui(i) = hsub_in
        stresstu(i) = 1.0
        stresstl(i) = 1.0
        do k = 1, 2
          lai(i,k) = 1.0
          sai(i,k) = 0.1
        end do
        do k = 1, 12
          frac(i,k) = 1.0 / 12.0
        end do
        do k = 1, 6
          poros(i,k) = 0.45
          csoi(i,k)  = 800.0
          rhosoi(i,k) = 1500.0
          suction(i,k) = 0.2
          bex(i,k)    = 6.0
          swilt(i,k)  = 0.15
          wsoi(i,k)   = 0.3
          wisoi(i,k)  = 0.0
          tsoi(i,k)   = 288.0
          consoi(i,k) = 1.5
          stressu(i,k) = 0.5
          stressl(i,k) = 0.5
        end do
        do k = 1, 7
          hsoi(k) = 0.1
        end do
        do k = 1, nsnolay
          hsno(i,k) = 0.05
          tsno(i,k) = 270.0
        end do
      end do

c --- user inputs ---
      do i = 1, n_in
        tu(i)     = tu_in(i)
        ts(i)     = ts_in(i)
        tl(i)     = tl_in(i)
        t12(i)    = t12_in(i)
        t34(i)    = t34_in(i)
        q12(i)    = q12_in(i)
        q34(i)    = q34_in(i)
        tg(i)     = tg_in(i)
        ti(i)     = ti_in(i)
        cp(i)     = cp_in(i)
        rhoa(i)   = rhoa_in(i)
        cu(i)     = cu_in(i)
        cl(i)     = cl_in(i)
        su(i)     = su_in(i)
        ss(i)     = ss_in(i)
        sl(i)     = sl_in(i)
        sg(i)     = sg_in(i)
        si(i)     = si_in(i)
        ta(i)     = ta_in(i)
        qa(i)     = qa_in(i)
        psurf(i)  = psurf_in(i)
        fi(i)     = fi_in(i)
        fu(i)     = fu_in(i)
        fl(i)     = fl_in(i)
        wliqu(i)  = wliqu_in(i)
        wsnou(i)  = wsnou_in(i)
        wliqs(i)  = wliqs_in(i)
        wsnos(i)  = wsnos_in(i)
        wliql(i)  = wliql_in(i)
        wsnol(i)  = wsnol_in(i)
        fwetu(i)  = fwetu_in(i)
        fwets(i)  = fwets_in(i)
        fwetl(i)  = fwetl_in(i)
        rliqu(i)  = rliqu_in(i)
        rliqs(i)  = rliqs_in(i)
        rliql(i)  = rliql_in(i)
        totcondub(i) = totcondub_in(i)
        totconduc(i) = totconduc_in(i)
        totcondls(i) = totcondls_in(i)
        totcondl3(i) = totcondl3_in(i)
        totcondl4(i) = totcondl4_in(i)
        solu(i)   = solu_in(i)
        sols(i)   = sols_in(i)
        soll(i)   = soll_in(i)
        solg(i)   = solg_in(i)
        soli(i)   = soli_in(i)
        firu(i)   = firu_in(i)
        firs(i)   = firs_in(i)
        firl(i)   = firl_in(i)
        firg(i)   = firg_in(i)
        firi(i)   = firi_in(i)
        firb(i)   = firb_in(i)
        pfluxu(i) = pfluxu_in(i)
        pfluxs(i) = pfluxs_in(i)
        pfluxl(i) = pfluxl_in(i)
        wpud(i)   = wpud_in(i)
        wipud(i)  = wipud_in(i)
        hvasug(i) = hvasug_in(i)
        hvasui(i) = hvasui_in(i)
        stresstu(i) = stresstu_in(i)
        stresstl(i) = stresstl_in(i)
        do k = 1, 2
          lai(i,k) = lai_in(i,k)
          sai(i,k) = sai_in(i,k)
        end do
        do k = 1, 12
          frac(i,k) = frac_in(i,k)
        end do
c only layer 1 fields are read by turvap; replicate to deeper layers
c just so the array isn't garbage:
        poros(i,1)   = poros1_in(i)
        csoi(i,1)    = csoi1_in(i)
        rhosoi(i,1)  = rhosoi1_in(i)
        suction(i,1) = suction1_in(i)
        bex(i,1)     = bex1_in(i)
        swilt(i,1)   = swilt1_in(i)
        wsoi(i,1)    = wsoi1_in(i)
        wisoi(i,1)   = wisoi1_in(i)
        tsoi(i,1)    = tsoi1_in(i)
        consoi(i,1)  = consoi1_in(i)
        do k = 1, 6
          stressu(i,k) = stressu_in(i,k)
          stressl(i,k) = stressl_in(i,k)
        end do
        hsno(i,1)    = hsno1_in(i)
        tsno(i,1)    = tsno1_in(i)
      end do
      hsoi(1) = hsoi1_in

      call turvap(1, 1)

      do i = 1, n_in
        tu_out(i)     = tu(i)
        ts_out(i)     = ts(i)
        tl_out(i)     = tl(i)
        t12_out(i)    = t12(i)
        t34_out(i)    = t34(i)
        q12_out(i)    = q12(i)
        q34_out(i)    = q34(i)
        tg_out(i)     = tg(i)
        ti_out(i)     = ti(i)
        fsena_out(i)  = fsena(i)
        fsenu_out(i)  = fsenu(i)
        fsens_out(i)  = fsens(i)
        fsenl_out(i)  = fsenl(i)
        fseng_out(i)  = fseng(i)
        fseni_out(i)  = fseni(i)
        fvapa_out(i)  = fvapa(i)
        fvapuw_out(i) = fvapuw(i)
        fvaput_out(i) = fvaput(i)
        fvaps_out(i)  = fvaps(i)
        fvaplw_out(i) = fvaplw(i)
        fvaplt_out(i) = fvaplt(i)
        fvapg_out(i)  = fvapg(i)
        fvapi_out(i)  = fvapi(i)
        firg_out(i)   = firg(i)
        firi_out(i)   = firi(i)
        firb_out(i)   = firb(i)
        ginvap_out(i) = ginvap(i)
        gsuvap_out(i) = gsuvap(i)
        gtrans_out(i) = gtrans(i)
      end do

      return
      end
