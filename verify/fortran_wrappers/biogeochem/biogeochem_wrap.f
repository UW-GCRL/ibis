c =====================================================================
c biogeochem_wrap.f
c
c F77 wrapper for biogeochem.f soilbgc (CENTURY-like SOM decomposition).
c =====================================================================

      subroutine wrap_soilbgc(
     >    n_in,
     >    iyear_in, iyear0_in, imonth_in, iday_in,
     >    nspinsoil_in, spin_in, spinmax_in,
     >    yparams_in,
     >    clitlm_in, clitls_in, clitll_in,
     >    clitrm_in, clitrs_in, clitrl_in,
     >    clitwm_in, clitws_in, clitwl_in,
     >    csoipas_in, csoislon_in, csoislop_in,
     >    decompl_in, decomps_in,
     >    falll_in, fallr_in, fallw_in,
     >    cdisturb_in,
     >    sand_in, clay_in,
     >    hsoi_in,
     >    storedn_in,
     >    ndaypy_in,
     >    clitlm_out, clitls_out, clitll_out,
     >    clitrm_out, clitrs_out, clitrl_out,
     >    clitwm_out, clitws_out, clitwl_out,
     >    csoipas_out, csoislon_out, csoislop_out,
     >    totcmic_out, totlit_out, totalit_out, totrlit_out,
     >    totcsoi_out, totfall_out,
     >    tco2mic_out, tnmin_out,
     >    storedn_out, yrleach_out, ynleach_out)
cf2py intent(in)  n_in
cf2py intent(in)  iyear_in, iyear0_in, imonth_in, iday_in
cf2py intent(in)  nspinsoil_in, spin_in, spinmax_in
cf2py intent(in)  yparams_in
cf2py intent(in)  clitlm_in, clitls_in, clitll_in
cf2py intent(in)  clitrm_in, clitrs_in, clitrl_in
cf2py intent(in)  clitwm_in, clitws_in, clitwl_in
cf2py intent(in)  csoipas_in, csoislon_in, csoislop_in
cf2py intent(in)  decompl_in, decomps_in
cf2py intent(in)  falll_in, fallr_in, fallw_in
cf2py intent(in)  cdisturb_in
cf2py intent(in)  sand_in, clay_in
cf2py intent(in)  hsoi_in
cf2py intent(in)  storedn_in
cf2py intent(in)  ndaypy_in
cf2py intent(out) clitlm_out, clitls_out, clitll_out
cf2py intent(out) clitrm_out, clitrs_out, clitrl_out
cf2py intent(out) clitwm_out, clitws_out, clitwl_out
cf2py intent(out) csoipas_out, csoislon_out, csoislop_out
cf2py intent(out) totcmic_out, totlit_out, totalit_out, totrlit_out
cf2py intent(out) totcsoi_out, totfall_out
cf2py intent(out) tco2mic_out, tnmin_out
cf2py intent(out) storedn_out, yrleach_out, ynleach_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsoi.h'
      include 'comsum.h'
      include 'comveg.h'
      include 'comatm.h'

      integer n_in, ndaypy_in
      integer iyear_in, iyear0_in, imonth_in, iday_in
      integer nspinsoil_in, spin_in, spinmax_in
      real*8 yparams_in(16)
c y* common block (matches biogeochem_subset.f modification)
      real comyparams_v(16)
      common /comyparams/ comyparams_v
      integer kk
      real*8 clitlm_in(n_in)
      real*8 clitls_in(n_in)
      real*8 clitll_in(n_in)
      real*8 clitrm_in(n_in)
      real*8 clitrs_in(n_in)
      real*8 clitrl_in(n_in)
      real*8 clitwm_in(n_in)
      real*8 clitws_in(n_in)
      real*8 clitwl_in(n_in)
      real*8 csoipas_in(n_in)
      real*8 csoislon_in(n_in)
      real*8 csoislop_in(n_in)
      real*8 decompl_in(n_in)
      real*8 decomps_in(n_in)
      real*8 falll_in(n_in)
      real*8 fallr_in(n_in)
      real*8 fallw_in(n_in)
      real*8 cdisturb_in(n_in)
      real*8 sand_in(n_in, 6)
      real*8 clay_in(n_in, 6)
      real*8 hsoi_in(7)
      real*8 storedn_in(n_in)
      real*8 clitlm_out(n_in)
      real*8 clitls_out(n_in)
      real*8 clitll_out(n_in)
      real*8 clitrm_out(n_in)
      real*8 clitrs_out(n_in)
      real*8 clitrl_out(n_in)
      real*8 clitwm_out(n_in)
      real*8 clitws_out(n_in)
      real*8 clitwl_out(n_in)
      real*8 csoipas_out(n_in)
      real*8 csoislon_out(n_in)
      real*8 csoislop_out(n_in)
      real*8 totcmic_out(n_in)
      real*8 totlit_out(n_in)
      real*8 totalit_out(n_in)
      real*8 totrlit_out(n_in)
      real*8 totcsoi_out(n_in)
      real*8 totfall_out(n_in)
      real*8 tco2mic_out(n_in)
      real*8 tnmin_out(n_in)
      real*8 storedn_out(n_in)
      real*8 yrleach_out(n_in)
      real*8 ynleach_out(n_in)

      integer i, k

c --- compar params ---
      ndaypy = ndaypy_in
      do k = 1, nsoilay + 1
        hsoi(k) = hsoi_in(k)
      end do

c --- y* parameters (per-call override) ---
      do kk = 1, 16
        comyparams_v(kk) = yparams_in(kk)
      end do
      ndaypm(1)  = 31
      ndaypm(2)  = 28
      ndaypm(3)  = 31
      ndaypm(4)  = 30
      ndaypm(5)  = 31
      ndaypm(6)  = 30
      ndaypm(7)  = 31
      ndaypm(8)  = 31
      ndaypm(9)  = 30
      ndaypm(10) = 31
      ndaypm(11) = 30
      ndaypm(12) = 31

c --- safe defaults for all npoi ---
      do i = 1, npoi
        clitlm(i) = 0.0; clitls(i) = 0.0; clitll(i) = 0.0
        clitrm(i) = 0.0; clitrs(i) = 0.0; clitrl(i) = 0.0
        clitwm(i) = 0.0; clitws(i) = 0.0; clitwl(i) = 0.0
        csoipas(i) = 0.0; csoislon(i) = 0.0; csoislop(i) = 0.0
        decompl(i) = 0.5; decomps(i) = 0.5
        falll(i) = 0.0; fallr(i) = 0.0; fallw(i) = 0.0
        cdisturb(i) = 0.0
        do k = 1, nsoilay
          sand(i, k) = 30.0
          clay(i, k) = 20.0
        end do
        storedn(i) = 0.1
        yrleach(i) = 0.0
        ynleach(i) = 0.0
        tco2mic(i) = 0.0; tnmin(i) = 0.0
        totcmic(i) = 0.0
        totlit(i) = 0.0; totalit(i) = 0.0; totrlit(i) = 0.0
        totcsoi(i) = 0.0; totfall(i) = 0.0
      end do

c --- user inputs ---
      do i = 1, n_in
        clitlm(i)   = clitlm_in(i)
        clitls(i)   = clitls_in(i)
        clitll(i)   = clitll_in(i)
        clitrm(i)   = clitrm_in(i)
        clitrs(i)   = clitrs_in(i)
        clitrl(i)   = clitrl_in(i)
        clitwm(i)   = clitwm_in(i)
        clitws(i)   = clitws_in(i)
        clitwl(i)   = clitwl_in(i)
        csoipas(i)  = csoipas_in(i)
        csoislon(i) = csoislon_in(i)
        csoislop(i) = csoislop_in(i)
        decompl(i)  = decompl_in(i)
        decomps(i)  = decomps_in(i)
        falll(i)    = falll_in(i)
        fallr(i)    = fallr_in(i)
        fallw(i)    = fallw_in(i)
        cdisturb(i) = cdisturb_in(i)
        do k = 1, nsoilay
          sand(i, k) = sand_in(i, k)
          clay(i, k) = clay_in(i, k)
        end do
        storedn(i) = storedn_in(i)
      end do

      call soilbgc(iyear_in, iyear0_in, imonth_in, iday_in,
     >             nspinsoil_in, spin_in, spinmax_in)

      do i = 1, n_in
        clitlm_out(i)   = clitlm(i)
        clitls_out(i)   = clitls(i)
        clitll_out(i)   = clitll(i)
        clitrm_out(i)   = clitrm(i)
        clitrs_out(i)   = clitrs(i)
        clitrl_out(i)   = clitrl(i)
        clitwm_out(i)   = clitwm(i)
        clitws_out(i)   = clitws(i)
        clitwl_out(i)   = clitwl(i)
        csoipas_out(i)  = csoipas(i)
        csoislon_out(i) = csoislon(i)
        csoislop_out(i) = csoislop(i)
        totcmic_out(i)  = totcmic(i)
        totlit_out(i)   = totlit(i)
        totalit_out(i)  = totalit(i)
        totrlit_out(i)  = totrlit(i)
        totcsoi_out(i)  = totcsoi(i)
        totfall_out(i)  = totfall(i)
        tco2mic_out(i)  = tco2mic(i)
        tnmin_out(i)    = tnmin(i)
        storedn_out(i)  = storedn(i)
        yrleach_out(i)  = yrleach(i)
        ynleach_out(i)  = ynleach(i)
      end do

      return
      end
