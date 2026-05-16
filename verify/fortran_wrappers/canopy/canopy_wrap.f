c =====================================================================
c canopy_wrap.f
c
c F77 wrappers for canopy.f pure-function pieces: fwetcal, mix, noveg, twet3.
c =====================================================================


c ---------------------------------------------------------------------
c wrap_fwetcal -- fwetcal takes no args; reads veg state from comveg
c and writes fwetu/s/l and rliqu/s/l. We seed the relevant fields then
c call.
c ---------------------------------------------------------------------
      subroutine wrap_fwetcal(
     >    n_in,
     >    tu_in, ts_in, tl_in,
     >    wliqu_in, wliqs_in, wliql_in,
     >    wsnou_in, wsnos_in, wsnol_in,
     >    wliqumax_in, wliqsmax_in, wliqlmax_in,
     >    wsnoumax_in, wsnosmax_in, wsnolmax_in,
     >    tmelt_in, epsilon_in,
     >    fwetu_out, fwets_out, fwetl_out,
     >    rliqu_out, rliqs_out, rliql_out)
cf2py intent(hide), depend(tu_in) :: n_in=len(tu_in)
cf2py intent(in)  tu_in
cf2py intent(in)  ts_in
cf2py intent(in)  tl_in
cf2py intent(in)  wliqu_in
cf2py intent(in)  wliqs_in
cf2py intent(in)  wliql_in
cf2py intent(in)  wsnou_in
cf2py intent(in)  wsnos_in
cf2py intent(in)  wsnol_in
cf2py intent(in)  wliqumax_in
cf2py intent(in)  wliqsmax_in
cf2py intent(in)  wliqlmax_in
cf2py intent(in)  wsnoumax_in
cf2py intent(in)  wsnosmax_in
cf2py intent(in)  wsnolmax_in
cf2py intent(in)  tmelt_in
cf2py intent(in)  epsilon_in
cf2py intent(out) fwetu_out
cf2py intent(out) fwets_out
cf2py intent(out) fwetl_out
cf2py intent(out) rliqu_out
cf2py intent(out) rliqs_out
cf2py intent(out) rliql_out

      include 'implicit.h'
      include 'compar.h'
      include 'comveg.h'
      include 'com1d.h'

      integer n_in
      real*8 tu_in(n_in), ts_in(n_in), tl_in(n_in)
      real*8 wliqu_in(n_in), wliqs_in(n_in), wliql_in(n_in)
      real*8 wsnou_in(n_in), wsnos_in(n_in), wsnol_in(n_in)
      real*8 wliqumax_in, wliqsmax_in, wliqlmax_in
      real*8 wsnoumax_in, wsnosmax_in, wsnolmax_in
      real*8 tmelt_in, epsilon_in
      real*8 fwetu_out(n_in), fwets_out(n_in), fwetl_out(n_in)
      real*8 rliqu_out(n_in), rliqs_out(n_in), rliql_out(n_in)

      integer i

      tmelt   = tmelt_in
      epsilon = epsilon_in
      wliqumax = wliqumax_in
      wliqsmax = wliqsmax_in
      wliqlmax = wliqlmax_in
      wsnoumax = wsnoumax_in
      wsnosmax = wsnosmax_in
      wsnolmax = wsnolmax_in

      do i = 1, npoi
        tu(i)    = 285.0
        ts(i)    = 285.0
        tl(i)    = 285.0
        wliqu(i) = 0.0
        wliqs(i) = 0.0
        wliql(i) = 0.0
        wsnou(i) = 0.0
        wsnos(i) = 0.0
        wsnol(i) = 0.0
      end do
      do i = 1, n_in
        tu(i)    = tu_in(i)
        ts(i)    = ts_in(i)
        tl(i)    = tl_in(i)
        wliqu(i) = wliqu_in(i)
        wliqs(i) = wliqs_in(i)
        wliql(i) = wliql_in(i)
        wsnou(i) = wsnou_in(i)
        wsnos(i) = wsnos_in(i)
        wsnol(i) = wsnol_in(i)
      end do

      call fwetcal

      do i = 1, n_in
        fwetu_out(i) = fwetu(i)
        fwets_out(i) = fwets(i)
        fwetl_out(i) = fwetl(i)
        rliqu_out(i) = rliqu(i)
        rliqs_out(i) = rliqs(i)
        rliql_out(i) = rliql(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_mix -- elementwise mix(xm, tm, x1, t1, x2, t2, x3, t3) is a
c subroutine in F77 that loops over npoi internally. We call it once
c per element by setting only one valid index and reading back.
c Simpler: call once with arrays of size n_in.
c ---------------------------------------------------------------------
      subroutine wrap_mix(
     >    n_in,
     >    x1_in, t1_in, x2_in, t2_in, x3_in, t3_in,
     >    epsilon_in,
     >    xm_out, tm_out)
cf2py intent(hide), depend(x1_in) :: n_in=len(x1_in)
cf2py intent(in)  x1_in
cf2py intent(in)  t1_in
cf2py intent(in)  x2_in
cf2py intent(in)  t2_in
cf2py intent(in)  x3_in
cf2py intent(in)  t3_in
cf2py intent(in)  epsilon_in
cf2py intent(out) xm_out
cf2py intent(out) tm_out

      include 'implicit.h'
      include 'compar.h'

      integer n_in
      real*8 x1_in(n_in), t1_in(n_in)
      real*8 x2_in(n_in), t2_in(n_in)
      real*8 x3_in(n_in), t3_in(n_in)
      real*8 epsilon_in
      real*8 xm_out(n_in), tm_out(n_in)

      integer i
      real xm(npoi), tm(npoi)
      real x1f(npoi), t1f(npoi)
      real x2f(npoi), t2f(npoi)
      real x3f(npoi), t3f(npoi)

      epsilon = epsilon_in

      do i = 1, npoi
        x1f(i) = 0.0; t1f(i) = 0.0
        x2f(i) = 0.0; t2f(i) = 0.0
        x3f(i) = 0.0; t3f(i) = 0.0
      end do
      do i = 1, n_in
        x1f(i) = x1_in(i); t1f(i) = t1_in(i)
        x2f(i) = x2_in(i); t2f(i) = t2_in(i)
        x3f(i) = x3_in(i); t3f(i) = t3_in(i)
      end do

      call mix(xm, tm, x1f, t1f, x2f, t2f, x3f, t3f)

      do i = 1, n_in
        xm_out(i) = xm(i)
        tm_out(i) = tm(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_noveg -- noveg sets canopy state to soil/snow average where no
c veg exists. Reads lai, sai, fu, fl, fi, tg, ti from common blocks.
c Updates tu/ts/tl/wliqu/wliqs/wliql/wsnou/wsnos/wsnol.
c ---------------------------------------------------------------------
      subroutine wrap_noveg(
     >    n_in,
     >    tu_in, ts_in, tl_in,
     >    wliqu_in, wliqs_in, wliql_in,
     >    wsnou_in, wsnos_in, wsnol_in,
     >    lai_in, sai_in,
     >    fu_in, fl_in, fi_in,
     >    tg_in, ti_in,
     >    tu_out, ts_out, tl_out,
     >    wliqu_out, wliqs_out, wliql_out,
     >    wsnou_out, wsnos_out, wsnol_out)
cf2py intent(hide), depend(fu_in) :: n_in=len(fu_in)
cf2py intent(in)  tu_in
cf2py intent(in)  ts_in
cf2py intent(in)  tl_in
cf2py intent(in)  wliqu_in
cf2py intent(in)  wliqs_in
cf2py intent(in)  wliql_in
cf2py intent(in)  wsnou_in
cf2py intent(in)  wsnos_in
cf2py intent(in)  wsnol_in
cf2py intent(in)  lai_in
cf2py intent(in)  sai_in
cf2py intent(in)  fu_in
cf2py intent(in)  fl_in
cf2py intent(in)  fi_in
cf2py intent(in)  tg_in
cf2py intent(in)  ti_in
cf2py intent(out) tu_out
cf2py intent(out) ts_out
cf2py intent(out) tl_out
cf2py intent(out) wliqu_out
cf2py intent(out) wliqs_out
cf2py intent(out) wliql_out
cf2py intent(out) wsnou_out
cf2py intent(out) wsnos_out
cf2py intent(out) wsnol_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsno.h'
      include 'comsoi.h'
      include 'comveg.h'

      integer n_in
      real*8 tu_in(n_in), ts_in(n_in), tl_in(n_in)
      real*8 wliqu_in(n_in), wliqs_in(n_in), wliql_in(n_in)
      real*8 wsnou_in(n_in), wsnos_in(n_in), wsnol_in(n_in)
      real*8 lai_in(n_in, 2), sai_in(n_in, 2)
      real*8 fu_in(n_in), fl_in(n_in), fi_in(n_in)
      real*8 tg_in(n_in), ti_in(n_in)
      real*8 tu_out(n_in), ts_out(n_in), tl_out(n_in)
      real*8 wliqu_out(n_in), wliqs_out(n_in), wliql_out(n_in)
      real*8 wsnou_out(n_in), wsnos_out(n_in), wsnol_out(n_in)

      integer i, k

      do i = 1, npoi
        tu(i) = 285.0; ts(i) = 285.0; tl(i) = 285.0
        wliqu(i) = 0.0; wliqs(i) = 0.0; wliql(i) = 0.0
        wsnou(i) = 0.0; wsnos(i) = 0.0; wsnol(i) = 0.0
        fu(i) = 0.0; fl(i) = 0.0; fi(i) = 0.0
        tg(i) = 285.0; ti(i) = 273.16
        do k = 1, 2
          lai(i, k) = 0.0; sai(i, k) = 0.0
        end do
      end do
      do i = 1, n_in
        tu(i) = tu_in(i); ts(i) = ts_in(i); tl(i) = tl_in(i)
        wliqu(i) = wliqu_in(i); wliqs(i) = wliqs_in(i); wliql(i) = wliql_in(i)
        wsnou(i) = wsnou_in(i); wsnos(i) = wsnos_in(i); wsnol(i) = wsnol_in(i)
        fu(i) = fu_in(i); fl(i) = fl_in(i); fi(i) = fi_in(i)
        tg(i) = tg_in(i); ti(i) = ti_in(i)
        do k = 1, 2
          lai(i, k) = lai_in(i, k); sai(i, k) = sai_in(i, k)
        end do
      end do

      call noveg

      do i = 1, n_in
        tu_out(i) = tu(i)
        ts_out(i) = ts(i)
        tl_out(i) = tl(i)
        wliqu_out(i) = wliqu(i); wliqs_out(i) = wliqs(i); wliql_out(i) = wliql(i)
        wsnou_out(i) = wsnou(i); wsnos_out(i) = wsnos(i); wsnol_out(i) = wsnol(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_twet3 -- vectorized wrapper around the F77 twet3 FUNCTION.
c twet3 takes scalars; we call it in a loop.
c
c F77 twet3 uses esat/qsat/hvapf (statement functions from comsat.h)
c which need hvap, ch2o, cvap, cair from compar.h.
c ---------------------------------------------------------------------
      subroutine wrap_twet3(
     >    n_in,
     >    tak_in, q_in, p_in,
     >    tmelt_in, cair_in,
     >    hvap_in, ch2o_in, cvap_in,
     >    tw_out)
cf2py intent(hide), depend(tak_in) :: n_in=len(tak_in)
cf2py intent(in)  tak_in
cf2py intent(in)  q_in
cf2py intent(in)  p_in
cf2py intent(in)  tmelt_in
cf2py intent(in)  cair_in
cf2py intent(in)  hvap_in
cf2py intent(in)  ch2o_in
cf2py intent(in)  cvap_in
cf2py intent(out) tw_out

      include 'implicit.h'
      include 'compar.h'

      integer n_in
      real*8 tak_in(n_in), q_in(n_in), p_in(n_in)
      real*8 tmelt_in, cair_in, hvap_in, ch2o_in, cvap_in
      real*8 tw_out(n_in)

      integer i
      real twet3, t_loc

c     Set common-block constants so esat/qsat/hvapf inside twet3 work.
      tmelt = tmelt_in
      cair  = cair_in
      hvap  = hvap_in
      ch2o  = ch2o_in
      cvap  = cvap_in
c     hsub/cice not used by twet3 but declared in comsat.h
      hsub = hvap_in + 0.3336e6
      cice = 2106.0

      do i = 1, n_in
        t_loc = twet3(real(tak_in(i)), real(q_in(i)), real(p_in(i)))
        tw_out(i) = t_loc
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_steph2o -- expose steph2o(tveg, wliq, wsno, xai, pflux,
c                                rain, train, snow, tsnow,
c                                tdrip, tblow, wliqmax, wsnomax,
c                                wliqmin, wsnomin).
c
c F77 steph2o takes inputs as args. Common-block state used:
c dtime/ch2o/cice/hfus/tmelt/epsilon from compar.h, plus vzero
c (zero-vector helper) referenced inside mix calls.
c ---------------------------------------------------------------------
      subroutine wrap_steph2o(
     >    n_in,
     >    tveg_in, wliq_in, wsno_in, xai_in,
     >    rain_in, train_in, snow_in, tsnow_in,
     >    tdrip_in, tblow_in,
     >    wliqmax_in, wsnomax_in, wliqmin_in, wsnomin_in,
     >    dtime_in, ch2o_in, cice_in, hfus_in,
     >    tmelt_in, epsilon_in,
     >    tveg_out, wliq_out, wsno_out, xai_out, pflux_out,
     >    rain_out, train_out, snow_out, tsnow_out)
cf2py intent(hide), depend(tveg_in) :: n_in=len(tveg_in)
cf2py intent(in)  tveg_in
cf2py intent(in)  wliq_in
cf2py intent(in)  wsno_in
cf2py intent(in)  xai_in
cf2py intent(in)  rain_in
cf2py intent(in)  train_in
cf2py intent(in)  snow_in
cf2py intent(in)  tsnow_in
cf2py intent(in)  tdrip_in
cf2py intent(in)  tblow_in
cf2py intent(in)  wliqmax_in
cf2py intent(in)  wsnomax_in
cf2py intent(in)  wliqmin_in
cf2py intent(in)  wsnomin_in
cf2py intent(in)  dtime_in
cf2py intent(in)  ch2o_in
cf2py intent(in)  cice_in
cf2py intent(in)  hfus_in
cf2py intent(in)  tmelt_in
cf2py intent(in)  epsilon_in
cf2py intent(out) tveg_out
cf2py intent(out) wliq_out
cf2py intent(out) wsno_out
cf2py intent(out) xai_out
cf2py intent(out) pflux_out
cf2py intent(out) rain_out
cf2py intent(out) train_out
cf2py intent(out) snow_out
cf2py intent(out) tsnow_out

      include 'implicit.h'
      include 'compar.h'

      integer n_in
      real*8 tveg_in(n_in), wliq_in(n_in), wsno_in(n_in)
      real*8 xai_in(n_in)
      real*8 rain_in(n_in), train_in(n_in)
      real*8 snow_in(n_in), tsnow_in(n_in)
      real*8 tdrip_in, tblow_in
      real*8 wliqmax_in, wsnomax_in, wliqmin_in, wsnomin_in
      real*8 dtime_in, ch2o_in, cice_in, hfus_in
      real*8 tmelt_in, epsilon_in
      real*8 tveg_out(n_in), wliq_out(n_in), wsno_out(n_in)
      real*8 xai_out(n_in), pflux_out(n_in)
      real*8 rain_out(n_in), train_out(n_in)
      real*8 snow_out(n_in), tsnow_out(n_in)

      integer i
      real tveg(npoi), wliq(npoi), wsno(npoi)
      real xai(npoi), pflux(npoi)
      real rain(npoi), train(npoi)
      real snow(npoi), tsnow(npoi)

      dtime   = dtime_in
      ch2o    = ch2o_in
      cice    = cice_in
      hfus    = hfus_in
      tmelt   = tmelt_in
      epsilon = epsilon_in

c     vzero is a common-block array of zeros, size npoi -- needed by
c     the mix calls at the end of steph2o. Initialize it.
      do i = 1, npoi
        vzero(i) = 0.0
        tveg(i)  = 285.0
        wliq(i)  = 0.0
        wsno(i)  = 0.0
        xai(i)   = 1.0
        pflux(i) = 0.0
        rain(i)  = 0.0
        train(i) = 285.0
        snow(i)  = 0.0
        tsnow(i) = 273.16
      end do

      do i = 1, n_in
        tveg(i)  = tveg_in(i)
        wliq(i)  = wliq_in(i)
        wsno(i)  = wsno_in(i)
        xai(i)   = xai_in(i)
        rain(i)  = rain_in(i)
        train(i) = train_in(i)
        snow(i)  = snow_in(i)
        tsnow(i) = tsnow_in(i)
      end do

      call steph2o(tveg, wliq, wsno, xai, pflux,
     >             rain, train, snow, tsnow,
     >             tdrip_in, tblow_in,
     >             wliqmax_in, wsnomax_in,
     >             wliqmin_in, wsnomin_in)

      do i = 1, n_in
        tveg_out(i)  = tveg(i)
        wliq_out(i)  = wliq(i)
        wsno_out(i)  = wsno(i)
        xai_out(i)   = xai(i)
        pflux_out(i) = pflux(i)
        rain_out(i)  = rain(i)
        train_out(i) = train(i)
        snow_out(i)  = snow(i)
        tsnow_out(i) = tsnow(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_steph2o2 -- expose steph2o2(tveg, wliq, wsno, fveg, xai, rliq,
c                                  fvapw, cveg).
c Reads ta (comatm), fvapa/fsena (com1d), uses hvapf/hsubf statement
c functions from comsat.h (needs hvap/cvap/ch2o/hsub/cice).
c ---------------------------------------------------------------------
      subroutine wrap_steph2o2(
     >    n_in,
     >    tveg_in, wliq_in, wsno_in,
     >    fveg_in, xai_in, rliq_in,
     >    fvapw_in, cveg_in,
     >    ta_in,
     >    fvapa_in, fsena_in,
     >    dtime_in, ch2o_in, cice_in, hfus_in,
     >    tmelt_in, hvap_in, hsub_in, cvap_in,
     >    tveg_out, wliq_out, wsno_out,
     >    fvapa_out, fsena_out)
cf2py intent(hide), depend(tveg_in) :: n_in=len(tveg_in)
cf2py intent(in)  tveg_in
cf2py intent(in)  wliq_in
cf2py intent(in)  wsno_in
cf2py intent(in)  fveg_in
cf2py intent(in)  xai_in
cf2py intent(in)  rliq_in
cf2py intent(in)  fvapw_in
cf2py intent(in)  cveg_in
cf2py intent(in)  ta_in
cf2py intent(in)  fvapa_in
cf2py intent(in)  fsena_in
cf2py intent(in)  dtime_in
cf2py intent(in)  ch2o_in
cf2py intent(in)  cice_in
cf2py intent(in)  hfus_in
cf2py intent(in)  tmelt_in
cf2py intent(in)  hvap_in
cf2py intent(in)  hsub_in
cf2py intent(in)  cvap_in
cf2py intent(out) tveg_out
cf2py intent(out) wliq_out
cf2py intent(out) wsno_out
cf2py intent(out) fvapa_out
cf2py intent(out) fsena_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'com1d.h'

      integer n_in
      real*8 tveg_in(n_in), wliq_in(n_in), wsno_in(n_in)
      real*8 fveg_in(n_in), xai_in(n_in), rliq_in(n_in)
      real*8 fvapw_in(n_in), cveg_in
      real*8 ta_in(n_in)
      real*8 fvapa_in(n_in), fsena_in(n_in)
      real*8 dtime_in, ch2o_in, cice_in, hfus_in
      real*8 tmelt_in, hvap_in, hsub_in, cvap_in
      real*8 tveg_out(n_in), wliq_out(n_in), wsno_out(n_in)
      real*8 fvapa_out(n_in), fsena_out(n_in)

      integer i
      real tveg(npoi), wliq(npoi), wsno(npoi)
      real fveg(npoi), xai(npoi), rliq(npoi), fvapw(npoi)

      dtime = dtime_in
      ch2o  = ch2o_in
      cice  = cice_in
      hfus  = hfus_in
      tmelt = tmelt_in
      hvap  = hvap_in
      hsub  = hsub_in
      cvap  = cvap_in

      do i = 1, npoi
        ta(i)    = 285.0
        fvapa(i) = 0.0
        fsena(i) = 0.0
        tveg(i) = 285.0; wliq(i) = 0.0; wsno(i) = 0.0
        fveg(i) = 1.0; xai(i) = 0.0; rliq(i) = 1.0; fvapw(i) = 0.0
      end do
      do i = 1, n_in
        ta(i)    = ta_in(i)
        fvapa(i) = fvapa_in(i)
        fsena(i) = fsena_in(i)
        tveg(i)  = tveg_in(i)
        wliq(i)  = wliq_in(i)
        wsno(i)  = wsno_in(i)
        fveg(i)  = fveg_in(i)
        xai(i)   = xai_in(i)
        rliq(i)  = rliq_in(i)
        fvapw(i) = fvapw_in(i)
      end do

      call steph2o2(tveg, wliq, wsno, fveg, xai, rliq, fvapw, cveg_in)

      do i = 1, n_in
        tveg_out(i) = tveg(i)
        wliq_out(i) = wliq(i)
        wsno_out(i) = wsno(i)
        fvapa_out(i) = fvapa(i)
        fsena_out(i) = fsena(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_cascade -- exposes F77 `cascade` subroutine (no-arg in F77; reads/
c writes ~30 common-block fields). We seed all inputs here and read all
c outputs after the call.
c ---------------------------------------------------------------------
      subroutine wrap_cascade(
     >    n_in,
     >    ta_in, qa_in, psurf_in, raina_in, snowa_in,
     >    t12_in, t34_in, fu_in, fl_in,
     >    tu_in, ts_in, tl_in,
     >    wliqu_in, wliqs_in, wliql_in,
     >    wsnou_in, wsnos_in, wsnol_in,
     >    lai_in, sai_in,
     >    tdripu_in, tblowu_in, tdrips_in, tblows_in,
     >    tdripl_in, tblowl_in,
     >    wliqumax_in, wsnoumax_in,
     >    wliqsmax_in, wsnosmax_in,
     >    wliqlmax_in, wsnolmax_in,
     >    dtime_in, tmelt_in, epsilon_in,
     >    ch2o_in, cice_in, hfus_in, hvap_in, cvap_in, cair_in,
     >    rainu_out, trainu_out, snowu_out, tsnowu_out,
     >    rainl_out, trainl_out, snowl_out, tsnowl_out,
     >    raing_out, traing_out, snowg_out, tsnowg_out,
     >    pfluxu_out, pfluxs_out, pfluxl_out,
     >    tu_out, ts_out, tl_out,
     >    wliqu_out, wliqs_out, wliql_out,
     >    wsnou_out, wsnos_out, wsnol_out)
cf2py intent(hide), depend(ta_in) :: n_in=len(ta_in)
cf2py intent(in)  ta_in, qa_in, psurf_in, raina_in, snowa_in
cf2py intent(in)  t12_in, t34_in, fu_in, fl_in
cf2py intent(in)  tu_in, ts_in, tl_in
cf2py intent(in)  wliqu_in, wliqs_in, wliql_in
cf2py intent(in)  wsnou_in, wsnos_in, wsnol_in
cf2py intent(in)  lai_in, sai_in
cf2py intent(in)  tdripu_in, tblowu_in, tdrips_in, tblows_in
cf2py intent(in)  tdripl_in, tblowl_in
cf2py intent(in)  wliqumax_in, wsnoumax_in
cf2py intent(in)  wliqsmax_in, wsnosmax_in
cf2py intent(in)  wliqlmax_in, wsnolmax_in
cf2py intent(in)  dtime_in, tmelt_in, epsilon_in
cf2py intent(in)  ch2o_in, cice_in, hfus_in, hvap_in, cvap_in, cair_in
cf2py intent(out) rainu_out, trainu_out, snowu_out, tsnowu_out
cf2py intent(out) rainl_out, trainl_out, snowl_out, tsnowl_out
cf2py intent(out) raing_out, traing_out, snowg_out, tsnowg_out
cf2py intent(out) pfluxu_out, pfluxs_out, pfluxl_out
cf2py intent(out) tu_out, ts_out, tl_out
cf2py intent(out) wliqu_out, wliqs_out, wliql_out
cf2py intent(out) wsnou_out, wsnos_out, wsnol_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'comveg.h'
      include 'com1d.h'

      integer n_in
      real*8 ta_in(n_in), qa_in(n_in), psurf_in(n_in)
      real*8 raina_in(n_in), snowa_in(n_in)
      real*8 t12_in(n_in), t34_in(n_in), fu_in(n_in), fl_in(n_in)
      real*8 tu_in(n_in), ts_in(n_in), tl_in(n_in)
      real*8 wliqu_in(n_in), wliqs_in(n_in), wliql_in(n_in)
      real*8 wsnou_in(n_in), wsnos_in(n_in), wsnol_in(n_in)
      real*8 lai_in(n_in, 2), sai_in(n_in, 2)
      real*8 tdripu_in, tblowu_in, tdrips_in, tblows_in
      real*8 tdripl_in, tblowl_in
      real*8 wliqumax_in, wsnoumax_in, wliqsmax_in, wsnosmax_in
      real*8 wliqlmax_in, wsnolmax_in
      real*8 dtime_in, tmelt_in, epsilon_in
      real*8 ch2o_in, cice_in, hfus_in, hvap_in, cvap_in, cair_in
      real*8 rainu_out(n_in), trainu_out(n_in)
      real*8 snowu_out(n_in), tsnowu_out(n_in)
      real*8 rainl_out(n_in), trainl_out(n_in)
      real*8 snowl_out(n_in), tsnowl_out(n_in)
      real*8 raing_out(n_in), traing_out(n_in)
      real*8 snowg_out(n_in), tsnowg_out(n_in)
      real*8 pfluxu_out(n_in), pfluxs_out(n_in), pfluxl_out(n_in)
      real*8 tu_out(n_in), ts_out(n_in), tl_out(n_in)
      real*8 wliqu_out(n_in), wliqs_out(n_in), wliql_out(n_in)
      real*8 wsnou_out(n_in), wsnos_out(n_in), wsnol_out(n_in)

      integer i, k

      dtime   = dtime_in
      tmelt   = tmelt_in
      epsilon = epsilon_in
      ch2o    = ch2o_in
      cice    = cice_in
      hfus    = hfus_in
      hvap    = hvap_in
      cvap    = cvap_in
      cair    = cair_in
      tdripu  = tdripu_in
      tblowu  = tblowu_in
      tdrips  = tdrips_in
      tblows  = tblows_in
      tdripl  = tdripl_in
      tblowl  = tblowl_in
      wliqumax = wliqumax_in
      wsnoumax = wsnoumax_in
      wliqsmax = wliqsmax_in
      wsnosmax = wsnosmax_in
      wliqlmax = wliqlmax_in
      wsnolmax = wsnolmax_in

c seed all npoi positions with safe defaults
      do i = 1, npoi
        vzero(i) = 0.0
        ta(i)    = 285.0
        qa(i)    = 0.005
        psurf(i) = 101325.0
        raina(i) = 0.0
        snowa(i) = 0.0
        t12(i)   = 285.0
        t34(i)   = 285.0
        fu(i)    = 0.5
        fl(i)    = 0.5
        tu(i)    = 285.0
        ts(i)    = 285.0
        tl(i)    = 285.0
        wliqu(i) = 0.0; wliqs(i) = 0.0; wliql(i) = 0.0
        wsnou(i) = 0.0; wsnos(i) = 0.0; wsnol(i) = 0.0
        do k = 1, 2
          lai(i, k) = 0.0
          sai(i, k) = 0.0
        end do
      end do

c overwrite first n_in with user inputs
      do i = 1, n_in
        ta(i)    = ta_in(i)
        qa(i)    = qa_in(i)
        psurf(i) = psurf_in(i)
        raina(i) = raina_in(i)
        snowa(i) = snowa_in(i)
        t12(i)   = t12_in(i)
        t34(i)   = t34_in(i)
        fu(i)    = fu_in(i)
        fl(i)    = fl_in(i)
        tu(i)    = tu_in(i)
        ts(i)    = ts_in(i)
        tl(i)    = tl_in(i)
        wliqu(i) = wliqu_in(i)
        wliqs(i) = wliqs_in(i)
        wliql(i) = wliql_in(i)
        wsnou(i) = wsnou_in(i)
        wsnos(i) = wsnos_in(i)
        wsnol(i) = wsnol_in(i)
        do k = 1, 2
          lai(i, k) = lai_in(i, k)
          sai(i, k) = sai_in(i, k)
        end do
      end do

      call cascade

      do i = 1, n_in
        rainu_out(i)  = rainu(i)
        trainu_out(i) = trainu(i)
        snowu_out(i)  = snowu(i)
        tsnowu_out(i) = tsnowu(i)
        rainl_out(i)  = rainl(i)
        trainl_out(i) = trainl(i)
        snowl_out(i)  = snowl(i)
        tsnowl_out(i) = tsnowl(i)
        raing_out(i)  = raing(i)
        traing_out(i) = traing(i)
        snowg_out(i)  = snowg(i)
        tsnowg_out(i) = tsnowg(i)
        pfluxu_out(i) = pfluxu(i)
        pfluxs_out(i) = pfluxs(i)
        pfluxl_out(i) = pfluxl(i)
        tu_out(i)    = tu(i)
        ts_out(i)    = ts(i)
        tl_out(i)    = tl(i)
        wliqu_out(i) = wliqu(i)
        wliqs_out(i) = wliqs(i)
        wliql_out(i) = wliql(i)
        wsnou_out(i) = wsnou(i)
        wsnos_out(i) = wsnos(i)
        wsnol_out(i) = wsnol(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_cascad2 -- exposes F77 `cascad2` subroutine.
c ---------------------------------------------------------------------
      subroutine wrap_cascad2(
     >    n_in,
     >    tu_in, ts_in, tl_in,
     >    wliqu_in, wliqs_in, wliql_in,
     >    wsnou_in, wsnos_in, wsnol_in,
     >    rliqu_in, rliqs_in, rliql_in,
     >    fvapuw_in, fvaps_in, fvaplw_in,
     >    fu_in, fl_in, fi_in,
     >    lai_in, sai_in,
     >    chu_in, chs_in, chl_in,
     >    ta_in, fvapa_in, fsena_in,
     >    dtime_in, tmelt_in,
     >    ch2o_in, cice_in, hfus_in, hvap_in, hsub_in, cvap_in,
     >    tu_out, ts_out, tl_out,
     >    wliqu_out, wliqs_out, wliql_out,
     >    wsnou_out, wsnos_out, wsnol_out,
     >    fvapa_out, fsena_out)
cf2py intent(hide), depend(ta_in) :: n_in=len(ta_in)
cf2py intent(in)  tu_in, ts_in, tl_in
cf2py intent(in)  wliqu_in, wliqs_in, wliql_in
cf2py intent(in)  wsnou_in, wsnos_in, wsnol_in
cf2py intent(in)  rliqu_in, rliqs_in, rliql_in
cf2py intent(in)  fvapuw_in, fvaps_in, fvaplw_in
cf2py intent(in)  fu_in, fl_in, fi_in
cf2py intent(in)  lai_in, sai_in
cf2py intent(in)  chu_in, chs_in, chl_in
cf2py intent(in)  ta_in, fvapa_in, fsena_in
cf2py intent(in)  dtime_in, tmelt_in
cf2py intent(in)  ch2o_in, cice_in, hfus_in, hvap_in, hsub_in, cvap_in
cf2py intent(out) tu_out, ts_out, tl_out
cf2py intent(out) wliqu_out, wliqs_out, wliql_out
cf2py intent(out) wsnou_out, wsnos_out, wsnol_out
cf2py intent(out) fvapa_out, fsena_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'comveg.h'
      include 'com1d.h'
      include 'comsno.h'

      integer n_in
      real*8 tu_in(n_in), ts_in(n_in), tl_in(n_in)
      real*8 wliqu_in(n_in), wliqs_in(n_in), wliql_in(n_in)
      real*8 wsnou_in(n_in), wsnos_in(n_in), wsnol_in(n_in)
      real*8 rliqu_in(n_in), rliqs_in(n_in), rliql_in(n_in)
      real*8 fvapuw_in(n_in), fvaps_in(n_in), fvaplw_in(n_in)
      real*8 fu_in(n_in), fl_in(n_in), fi_in(n_in)
      real*8 lai_in(n_in, 2), sai_in(n_in, 2)
      real*8 chu_in, chs_in, chl_in
      real*8 ta_in(n_in), fvapa_in(n_in), fsena_in(n_in)
      real*8 dtime_in, tmelt_in
      real*8 ch2o_in, cice_in, hfus_in, hvap_in, hsub_in, cvap_in
      real*8 tu_out(n_in), ts_out(n_in), tl_out(n_in)
      real*8 wliqu_out(n_in), wliqs_out(n_in), wliql_out(n_in)
      real*8 wsnou_out(n_in), wsnos_out(n_in), wsnol_out(n_in)
      real*8 fvapa_out(n_in), fsena_out(n_in)

      integer i, k

      dtime = dtime_in
      tmelt = tmelt_in
      ch2o  = ch2o_in
      cice  = cice_in
      hfus  = hfus_in
      hvap  = hvap_in
      hsub  = hsub_in
      cvap  = cvap_in
      chu   = chu_in
      chs   = chs_in
      chl   = chl_in

      do i = 1, npoi
        tu(i) = 285.0; ts(i) = 285.0; tl(i) = 285.0
        wliqu(i) = 0.0; wliqs(i) = 0.0; wliql(i) = 0.0
        wsnou(i) = 0.0; wsnos(i) = 0.0; wsnol(i) = 0.0
        rliqu(i) = 1.0; rliqs(i) = 1.0; rliql(i) = 1.0
        fvapuw(i) = 0.0; fvaps(i) = 0.0; fvaplw(i) = 0.0
        fu(i) = 0.5; fl(i) = 0.5; fi(i) = 0.0
        do k = 1, 2
          lai(i, k) = 0.0
          sai(i, k) = 0.0
        end do
        ta(i)    = 285.0
        fvapa(i) = 0.0
        fsena(i) = 0.0
      end do

      do i = 1, n_in
        tu(i)    = tu_in(i)
        ts(i)    = ts_in(i)
        tl(i)    = tl_in(i)
        wliqu(i) = wliqu_in(i); wliqs(i) = wliqs_in(i); wliql(i) = wliql_in(i)
        wsnou(i) = wsnou_in(i); wsnos(i) = wsnos_in(i); wsnol(i) = wsnol_in(i)
        rliqu(i) = rliqu_in(i); rliqs(i) = rliqs_in(i); rliql(i) = rliql_in(i)
        fvapuw(i) = fvapuw_in(i)
        fvaps(i)  = fvaps_in(i)
        fvaplw(i) = fvaplw_in(i)
        fu(i) = fu_in(i); fl(i) = fl_in(i); fi(i) = fi_in(i)
        do k = 1, 2
          lai(i, k) = lai_in(i, k)
          sai(i, k) = sai_in(i, k)
        end do
        ta(i)    = ta_in(i)
        fvapa(i) = fvapa_in(i)
        fsena(i) = fsena_in(i)
      end do

      call cascad2

      do i = 1, n_in
        tu_out(i) = tu(i); ts_out(i) = ts(i); tl_out(i) = tl(i)
        wliqu_out(i) = wliqu(i); wliqs_out(i) = wliqs(i); wliql_out(i) = wliql(i)
        wsnou_out(i) = wsnou(i); wsnos_out(i) = wsnos(i); wsnol_out(i) = wsnol(i)
        fvapa_out(i) = fvapa(i)
        fsena_out(i) = fsena(i)
      end do

      return
      end
