c =====================================================================
c stomata_wrap.f
c
c F77 wrapper for physiology.f stomata (Farquhar/Ball-Berry photosynthesis
c for 5 PFT classes: upper broadleaf+conifer, lower shrub+C3 grass+C4 grass).
c =====================================================================

      subroutine wrap_stomata(
     >    n_in,
     >    tu_in, tl_in, t12_in, t34_in, q12_in, q34_in,
     >    su_in, sl_in, psurf_in,
     >    topparu_in, topparl_in,
     >    termu_in, terml_in,
     >    scalcoefu_in, scalcoefl_in,
     >    ciub_in, ciuc_in, csub_in, csuc_in, gsub_in, gsuc_in,
     >    cils_in, cil3_in, cil4_in,
     >    csls_in, csl3_in, csl4_in,
     >    gsls_in, gsl3_in, gsl4_in,
     >    fwetu_in, fwetl_in,
     >    lai_in, sai_in, exist_in,
     >    stresstu_in, stresstl_in,
     >    a10scalparamu_in, a10daylightu_in,
     >    a10scalparaml_in, a10daylightl_in,
     >    co2conc_in, o2conc_in, dtime_in,
     >    tau15_in, kc15_in, ko15_in, cimax_in,
     >    alpha3_in, theta3_in, beta3_in,
     >    alpha4_in, theta4_in, beta4_in,
     >    vmax_pft_in,
     >    gammaub_in, coefmub_in, coefbub_in, gsubmin_in,
     >    gammauc_in, coefmuc_in, coefbuc_in, gsucmin_in,
     >    gammals_in, coefmls_in, coefbls_in, gslsmin_in,
     >    gammal3_in, coefml3_in, coefbl3_in, gsl3min_in,
     >    gammal4_in, coefml4_in, coefbl4_in, gsl4min_in,
     >    epsilon_in,
     >    ciub_out, ciuc_out, csub_out, csuc_out, gsub_out, gsuc_out,
     >    cils_out, cil3_out, cil4_out,
     >    csls_out, csl3_out, csl4_out,
     >    gsls_out, gsl3_out, gsl4_out,
     >    agcub_out, agcuc_out, agcls_out, agcl3_out, agcl4_out,
     >    ancub_out, ancuc_out, ancls_out, ancl3_out, ancl4_out,
     >    totcondub_out, totconduc_out,
     >    totcondls_out, totcondl3_out, totcondl4_out,
     >    a10scalparamu_out, a10daylightu_out,
     >    a10scalparaml_out, a10daylightl_out)
cf2py intent(in)  n_in
cf2py intent(in)  tu_in, tl_in, t12_in, t34_in, q12_in, q34_in
cf2py intent(in)  su_in, sl_in, psurf_in
cf2py intent(in)  topparu_in, topparl_in
cf2py intent(in)  termu_in, terml_in
cf2py intent(in)  scalcoefu_in, scalcoefl_in
cf2py intent(in)  ciub_in, ciuc_in, csub_in, csuc_in, gsub_in, gsuc_in
cf2py intent(in)  cils_in, cil3_in, cil4_in
cf2py intent(in)  csls_in, csl3_in, csl4_in
cf2py intent(in)  gsls_in, gsl3_in, gsl4_in
cf2py intent(in)  fwetu_in, fwetl_in
cf2py intent(in)  lai_in, sai_in, exist_in
cf2py intent(in)  stresstu_in, stresstl_in
cf2py intent(in)  a10scalparamu_in, a10daylightu_in
cf2py intent(in)  a10scalparaml_in, a10daylightl_in
cf2py intent(in)  co2conc_in, o2conc_in, dtime_in
cf2py intent(in)  tau15_in, kc15_in, ko15_in, cimax_in
cf2py intent(in)  alpha3_in, theta3_in, beta3_in
cf2py intent(in)  alpha4_in, theta4_in, beta4_in
cf2py intent(in)  vmax_pft_in
cf2py intent(in)  gammaub_in, coefmub_in, coefbub_in, gsubmin_in
cf2py intent(in)  gammauc_in, coefmuc_in, coefbuc_in, gsucmin_in
cf2py intent(in)  gammals_in, coefmls_in, coefbls_in, gslsmin_in
cf2py intent(in)  gammal3_in, coefml3_in, coefbl3_in, gsl3min_in
cf2py intent(in)  gammal4_in, coefml4_in, coefbl4_in, gsl4min_in
cf2py intent(in)  epsilon_in
cf2py intent(out) ciub_out, ciuc_out, csub_out, csuc_out, gsub_out, gsuc_out
cf2py intent(out) cils_out, cil3_out, cil4_out
cf2py intent(out) csls_out, csl3_out, csl4_out
cf2py intent(out) gsls_out, gsl3_out, gsl4_out
cf2py intent(out) agcub_out, agcuc_out, agcls_out, agcl3_out, agcl4_out
cf2py intent(out) ancub_out, ancuc_out, ancls_out, ancl3_out, ancl4_out
cf2py intent(out) totcondub_out, totconduc_out
cf2py intent(out) totcondls_out, totcondl3_out, totcondl4_out
cf2py intent(out) a10scalparamu_out, a10daylightu_out
cf2py intent(out) a10scalparaml_out, a10daylightl_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'comsno.h'
      include 'comsoi.h'
      include 'comveg.h'
      include 'com1d.h'
      include 'comsum.h'
      include 'compft.h'

      integer n_in
      real*8 tu_in(n_in)
      real*8 tl_in(n_in)
      real*8 t12_in(n_in)
      real*8 t34_in(n_in)
      real*8 q12_in(n_in)
      real*8 q34_in(n_in)
      real*8 su_in(n_in)
      real*8 sl_in(n_in)
      real*8 psurf_in(n_in)
      real*8 topparu_in(n_in)
      real*8 topparl_in(n_in)
      real*8 termu_in(n_in, 7)
      real*8 terml_in(n_in, 7)
      real*8 scalcoefu_in(n_in, 4)
      real*8 scalcoefl_in(n_in, 4)
      real*8 ciub_in(n_in)
      real*8 ciuc_in(n_in)
      real*8 csub_in(n_in)
      real*8 csuc_in(n_in)
      real*8 gsub_in(n_in)
      real*8 gsuc_in(n_in)
      real*8 cils_in(n_in)
      real*8 cil3_in(n_in)
      real*8 cil4_in(n_in)
      real*8 csls_in(n_in)
      real*8 csl3_in(n_in)
      real*8 csl4_in(n_in)
      real*8 gsls_in(n_in)
      real*8 gsl3_in(n_in)
      real*8 gsl4_in(n_in)
      real*8 fwetu_in(n_in)
      real*8 fwetl_in(n_in)
      real*8 lai_in(n_in, 2)
      real*8 sai_in(n_in, 2)
      real*8 exist_in(n_in, 12)
      real*8 stresstu_in(n_in)
      real*8 stresstl_in(n_in)
      real*8 a10scalparamu_in(n_in)
      real*8 a10daylightu_in(n_in)
      real*8 a10scalparaml_in(n_in)
      real*8 a10daylightl_in(n_in)
      real*8 co2conc_in, o2conc_in, dtime_in
      real*8 tau15_in, kc15_in, ko15_in, cimax_in
      real*8 alpha3_in, theta3_in, beta3_in
      real*8 alpha4_in, theta4_in, beta4_in
      real*8 vmax_pft_in(12)
      real*8 gammaub_in, coefmub_in, coefbub_in, gsubmin_in
      real*8 gammauc_in, coefmuc_in, coefbuc_in, gsucmin_in
      real*8 gammals_in, coefmls_in, coefbls_in, gslsmin_in
      real*8 gammal3_in, coefml3_in, coefbl3_in, gsl3min_in
      real*8 gammal4_in, coefml4_in, coefbl4_in, gsl4min_in
      real*8 epsilon_in

      real*8 ciub_out(n_in)
      real*8 ciuc_out(n_in)
      real*8 csub_out(n_in)
      real*8 csuc_out(n_in)
      real*8 gsub_out(n_in)
      real*8 gsuc_out(n_in)
      real*8 cils_out(n_in)
      real*8 cil3_out(n_in)
      real*8 cil4_out(n_in)
      real*8 csls_out(n_in)
      real*8 csl3_out(n_in)
      real*8 csl4_out(n_in)
      real*8 gsls_out(n_in)
      real*8 gsl3_out(n_in)
      real*8 gsl4_out(n_in)
      real*8 agcub_out(n_in)
      real*8 agcuc_out(n_in)
      real*8 agcls_out(n_in)
      real*8 agcl3_out(n_in)
      real*8 agcl4_out(n_in)
      real*8 ancub_out(n_in)
      real*8 ancuc_out(n_in)
      real*8 ancls_out(n_in)
      real*8 ancl3_out(n_in)
      real*8 ancl4_out(n_in)
      real*8 totcondub_out(n_in)
      real*8 totconduc_out(n_in)
      real*8 totcondls_out(n_in)
      real*8 totcondl3_out(n_in)
      real*8 totcondl4_out(n_in)
      real*8 a10scalparamu_out(n_in)
      real*8 a10daylightu_out(n_in)
      real*8 a10scalparaml_out(n_in)
      real*8 a10daylightl_out(n_in)

      integer i, k

c --- scalar params ---
      epsilon = epsilon_in
      dtime   = dtime_in
      co2conc = co2conc_in
      o2conc  = o2conc_in
      tau15   = tau15_in
      kc15    = kc15_in
      ko15    = ko15_in
      cimax   = cimax_in
      alpha3  = alpha3_in
      theta3  = theta3_in
      beta3   = beta3_in
      alpha4  = alpha4_in
      theta4  = theta4_in
      beta4   = beta4_in
      gammaub = gammaub_in
      coefmub = coefmub_in
      coefbub = coefbub_in
      gsubmin = gsubmin_in
      gammauc = gammauc_in
      coefmuc = coefmuc_in
      coefbuc = coefbuc_in
      gsucmin = gsucmin_in
      gammals = gammals_in
      coefmls = coefmls_in
      coefbls = coefbls_in
      gslsmin = gslsmin_in
      gammal3 = gammal3_in
      coefml3 = coefml3_in
      coefbl3 = coefbl3_in
      gsl3min = gsl3min_in
      gammal4 = gammal4_in
      coefml4 = coefml4_in
      coefbl4 = coefbl4_in
      gsl4min = gsl4min_in

      do k = 1, 12
        vmax_pft(k) = vmax_pft_in(k)
      end do

c --- seed all npoi positions ---
      do i = 1, npoi
        tu(i)  = 288.0; tl(i)  = 288.0
        t12(i) = 288.0; t34(i) = 288.0
        q12(i) = 0.008; q34(i) = 0.008
        su(i)  = 0.1;   sl(i)  = 0.1
        psurf(i) = 101325.0
        topparu(i) = 0.0; topparl(i) = 0.0
        do k = 1, 7
          termu(i, k) = 0.0
          terml(i, k) = 0.0
        end do
        do k = 1, 4
          scalcoefu(i, k) = 0.0
          scalcoefl(i, k) = 0.0
        end do
        ciub(i) = 270.0e-6; ciuc(i) = 270.0e-6
        csub(i) = 350.0e-6; csuc(i) = 350.0e-6
        gsub(i) = 0.1; gsuc(i) = 0.1
        cils(i) = 270.0e-6; cil3(i) = 270.0e-6; cil4(i) = 270.0e-6
        csls(i) = 350.0e-6; csl3(i) = 350.0e-6; csl4(i) = 350.0e-6
        gsls(i) = 0.1; gsl3(i) = 0.1; gsl4(i) = 0.1
        fwetu(i) = 0.0; fwetl(i) = 0.0
        do k = 1, 2
          lai(i, k) = 0.0
          sai(i, k) = 0.0
        end do
        do k = 1, 12
          exist(i, k) = 0.0
        end do
        stresstu(i) = 1.0; stresstl(i) = 1.0
        a10scalparamu(i) = 1.0; a10daylightu(i) = 100.0
        a10scalparaml(i) = 1.0; a10daylightl(i) = 100.0
      end do

c --- overwrite first n_in ---
      do i = 1, n_in
        tu(i)  = tu_in(i);  tl(i)  = tl_in(i)
        t12(i) = t12_in(i); t34(i) = t34_in(i)
        q12(i) = q12_in(i); q34(i) = q34_in(i)
        su(i)  = su_in(i);  sl(i)  = sl_in(i)
        psurf(i) = psurf_in(i)
        topparu(i) = topparu_in(i)
        topparl(i) = topparl_in(i)
        do k = 1, 7
          termu(i, k) = termu_in(i, k)
          terml(i, k) = terml_in(i, k)
        end do
        do k = 1, 4
          scalcoefu(i, k) = scalcoefu_in(i, k)
          scalcoefl(i, k) = scalcoefl_in(i, k)
        end do
        ciub(i) = ciub_in(i); ciuc(i) = ciuc_in(i)
        csub(i) = csub_in(i); csuc(i) = csuc_in(i)
        gsub(i) = gsub_in(i); gsuc(i) = gsuc_in(i)
        cils(i) = cils_in(i); cil3(i) = cil3_in(i); cil4(i) = cil4_in(i)
        csls(i) = csls_in(i); csl3(i) = csl3_in(i); csl4(i) = csl4_in(i)
        gsls(i) = gsls_in(i); gsl3(i) = gsl3_in(i); gsl4(i) = gsl4_in(i)
        fwetu(i) = fwetu_in(i); fwetl(i) = fwetl_in(i)
        do k = 1, 2
          lai(i, k) = lai_in(i, k)
          sai(i, k) = sai_in(i, k)
        end do
        do k = 1, 12
          exist(i, k) = exist_in(i, k)
        end do
        stresstu(i) = stresstu_in(i); stresstl(i) = stresstl_in(i)
        a10scalparamu(i) = a10scalparamu_in(i)
        a10daylightu(i)  = a10daylightu_in(i)
        a10scalparaml(i) = a10scalparaml_in(i)
        a10daylightl(i)  = a10daylightl_in(i)
      end do

      call stomata

      do i = 1, n_in
        ciub_out(i) = ciub(i); ciuc_out(i) = ciuc(i)
        csub_out(i) = csub(i); csuc_out(i) = csuc(i)
        gsub_out(i) = gsub(i); gsuc_out(i) = gsuc(i)
        cils_out(i) = cils(i); cil3_out(i) = cil3(i); cil4_out(i) = cil4(i)
        csls_out(i) = csls(i); csl3_out(i) = csl3(i); csl4_out(i) = csl4(i)
        gsls_out(i) = gsls(i); gsl3_out(i) = gsl3(i); gsl4_out(i) = gsl4(i)
        agcub_out(i) = agcub(i); agcuc_out(i) = agcuc(i)
        agcls_out(i) = agcls(i); agcl3_out(i) = agcl3(i); agcl4_out(i) = agcl4(i)
        ancub_out(i) = ancub(i); ancuc_out(i) = ancuc(i)
        ancls_out(i) = ancls(i); ancl3_out(i) = ancl3(i); ancl4_out(i) = ancl4(i)
        totcondub_out(i) = totcondub(i); totconduc_out(i) = totconduc(i)
        totcondls_out(i) = totcondls(i); totcondl3_out(i) = totcondl3(i)
        totcondl4_out(i) = totcondl4(i)
        a10scalparamu_out(i) = a10scalparamu(i)
        a10daylightu_out(i)  = a10daylightu(i)
        a10scalparaml_out(i) = a10scalparaml(i)
        a10daylightl_out(i)  = a10daylightl(i)
      end do

      return
      end
