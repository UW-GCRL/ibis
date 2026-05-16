c =====================================================================
c vegetation_wrap.f
c
c F77 wrapper for vegetation.f dynaveg (annual dynamic vegetation update).
c Internally dynaveg calls fire() and vegmap(); both are in vegetation_subset.f.
c =====================================================================


      subroutine wrap_dynaveg(
     >    n_in, isimfire_in,
     >    cbiol_in, cbiow_in, cbior_in,
     >    exist_in, aynpp_in,
     >    firefac_in, totlit_in,
     >    aleaf_in, aroot_in, awood_in,
     >    tauleaf_in, tauroot_in, tauwood0_in,
     >    specla_in, xminlai_in, woodnorm_in,
     >    gdd0_in, gdd5_in,
     >    ayneetot_in,
     >    cbiol_out, cbiow_out, cbior_out,
     >    plai_out, biomass_out,
     >    falll_out, fallr_out, fallw_out, cdisturb_out,
     >    disturbf_out, disturbo_out,
     >    ayanpp_out, aynpp_out, ayneetot_out, ayanpptot_out,
     >    totlaiu_out, totlail_out, totbiou_out, totbiol_out,
     >    fu_out, fl_out,
     >    ztop_out, zbot_out, sai_out,
     >    sapfrac_out, vegtype0_out, tauwood_out)
cf2py intent(in)  n_in, isimfire_in
cf2py intent(in)  cbiol_in, cbiow_in, cbior_in
cf2py intent(in)  exist_in, aynpp_in
cf2py intent(in)  firefac_in, totlit_in
cf2py intent(in)  aleaf_in, aroot_in, awood_in
cf2py intent(in)  tauleaf_in, tauroot_in, tauwood0_in
cf2py intent(in)  specla_in, xminlai_in, woodnorm_in
cf2py intent(in)  gdd0_in, gdd5_in
cf2py intent(in)  ayneetot_in
cf2py intent(out) cbiol_out, cbiow_out, cbior_out
cf2py intent(out) plai_out, biomass_out
cf2py intent(out) falll_out, fallr_out, fallw_out, cdisturb_out
cf2py intent(out) disturbf_out, disturbo_out
cf2py intent(out) ayanpp_out, aynpp_out, ayneetot_out, ayanpptot_out
cf2py intent(out) totlaiu_out, totlail_out, totbiou_out, totbiol_out
cf2py intent(out) fu_out, fl_out
cf2py intent(out) ztop_out, zbot_out, sai_out
cf2py intent(out) sapfrac_out, vegtype0_out, tauwood_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsoi.h'
      include 'comsum.h'
      include 'comveg.h'
      include 'compft.h'

      integer n_in, isimfire_in
      real*8 cbiol_in(n_in, 12)
      real*8 cbiow_in(n_in, 12)
      real*8 cbior_in(n_in, 12)
      real*8 exist_in(n_in, 12)
      real*8 aynpp_in(n_in, 12)
      real*8 firefac_in(n_in)
      real*8 totlit_in(n_in)
      real*8 aleaf_in(12)
      real*8 aroot_in(12)
      real*8 awood_in(12)
      real*8 tauleaf_in(12)
      real*8 tauroot_in(12)
      real*8 tauwood0_in(12)
      real*8 specla_in(12)
      real*8 xminlai_in
      real*8 woodnorm_in
      real*8 gdd0_in(n_in)
      real*8 gdd5_in(n_in)
      real*8 ayneetot_in(n_in)

      real*8 cbiol_out(n_in, 12)
      real*8 cbiow_out(n_in, 12)
      real*8 cbior_out(n_in, 12)
      real*8 plai_out(n_in, 12)
      real*8 biomass_out(n_in, 12)
      real*8 falll_out(n_in)
      real*8 fallr_out(n_in)
      real*8 fallw_out(n_in)
      real*8 cdisturb_out(n_in)
      real*8 disturbf_out(n_in)
      real*8 disturbo_out(n_in)
      real*8 ayanpp_out(n_in, 12)
      real*8 aynpp_out(n_in, 12)
      real*8 ayneetot_out(n_in)
      real*8 ayanpptot_out(n_in)
      real*8 totlaiu_out(n_in)
      real*8 totlail_out(n_in)
      real*8 totbiou_out(n_in)
      real*8 totbiol_out(n_in)
      real*8 fu_out(n_in)
      real*8 fl_out(n_in)
      real*8 ztop_out(n_in, 2)
      real*8 zbot_out(n_in, 2)
      real*8 sai_out(n_in, 2)
      real*8 sapfrac_out(n_in)
      real*8 vegtype0_out(n_in)
      real*8 tauwood_out(12)

      integer i, k

c --- PFT-scalar params (compft / comveg) ---
      do k = 1, 12
        tauleaf(k)  = tauleaf_in(k)
        tauroot(k)  = tauroot_in(k)
        tauwood0(k) = tauwood0_in(k)
        tauwood(k)  = tauwood0_in(k)
        aleaf(k)    = aleaf_in(k)
        aroot(k)    = aroot_in(k)
        awood(k)    = awood_in(k)
        specla(k)   = specla_in(k)
      end do
      xminlai  = xminlai_in
      woodnorm = woodnorm_in

c --- seed all npoi positions with safe defaults ---
      do i = 1, npoi
        firefac(i)  = 0.0
        totlit(i)   = 0.0
        gdd0(i)     = 0.0
        gdd5(i)     = 0.0
        ayneetot(i) = 0.0
        ayanpptot(i) = 0.0
        totlaiu(i)  = 0.0
        totlail(i)  = 0.0
        totbiou(i)  = 0.0
        totbiol(i)  = 0.0
        fu(i)       = 0.5
        fl(i)       = 0.5
        sapfrac(i)  = 0.1
        vegtype0(i) = 0.0
        falll(i)    = 0.0
        fallr(i)    = 0.0
        fallw(i)    = 0.0
        cdisturb(i) = 0.0
        disturbf(i) = 0.0
        disturbo(i) = 0.0
        do k = 1, 12
          cbiol(i, k) = 0.0
          cbiow(i, k) = 0.0
          cbior(i, k) = 0.0
          exist(i, k) = 0.0
          aynpp(i, k) = 0.0
          ayanpp(i, k) = 0.0
          plai(i, k)  = 0.0
          biomass(i, k) = 0.0
        end do
        do k = 1, 2
          ztop(i, k) = 0.0
          zbot(i, k) = 0.0
          sai(i, k)  = 0.0
        end do
      end do

c --- overwrite first n_in with user inputs ---
      do i = 1, n_in
        firefac(i)  = firefac_in(i)
        totlit(i)   = totlit_in(i)
        gdd0(i)     = gdd0_in(i)
        gdd5(i)     = gdd5_in(i)
        ayneetot(i) = ayneetot_in(i)
        do k = 1, 12
          cbiol(i, k) = cbiol_in(i, k)
          cbiow(i, k) = cbiow_in(i, k)
          cbior(i, k) = cbior_in(i, k)
          exist(i, k) = exist_in(i, k)
          aynpp(i, k) = aynpp_in(i, k)
        end do
      end do

      call dynaveg(isimfire_in)

      do i = 1, n_in
        do k = 1, 12
          cbiol_out(i, k)   = cbiol(i, k)
          cbiow_out(i, k)   = cbiow(i, k)
          cbior_out(i, k)   = cbior(i, k)
          plai_out(i, k)    = plai(i, k)
          biomass_out(i, k) = biomass(i, k)
          ayanpp_out(i, k)  = ayanpp(i, k)
          aynpp_out(i, k)   = aynpp(i, k)
        end do
        falll_out(i)    = falll(i)
        fallr_out(i)    = fallr(i)
        fallw_out(i)    = fallw(i)
        cdisturb_out(i) = cdisturb(i)
        disturbf_out(i) = disturbf(i)
        disturbo_out(i) = disturbo(i)
        ayneetot_out(i) = ayneetot(i)
        ayanpptot_out(i) = ayanpptot(i)
        totlaiu_out(i)  = totlaiu(i)
        totlail_out(i)  = totlail(i)
        totbiou_out(i)  = totbiou(i)
        totbiol_out(i)  = totbiol(i)
        fu_out(i)       = fu(i)
        fl_out(i)       = fl(i)
        sapfrac_out(i)  = sapfrac(i)
        vegtype0_out(i) = vegtype0(i)
        do k = 1, 2
          ztop_out(i, k) = ztop(i, k)
          zbot_out(i, k) = zbot(i, k)
          sai_out(i, k)  = sai(i, k)
        end do
      end do
      do k = 1, 12
        tauwood_out(k) = tauwood(k)
      end do

      return
      end
