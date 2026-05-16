c =====================================================================
c phenology_wrap.f
c
c F77 wrapper for vegetation.f pheno subroutine.
c =====================================================================


c ---------------------------------------------------------------------
c wrap_pheno -- exposes F77 `pheno` (no-arg; reads/writes via commons).
c
c Inputs (per grid point):
c   td_in, a10td_in   daily and 10-day temperatures (K)
c   a10ancub_in, a10ancls_in, a10ancl4_in, a10ancl3_in  10-day photosynthesis
c   tc_in              coldest-month temperature (C)
c   plai_in [n, 12]    prognostic LAI per PFT
c   fu_in, fl_in       canopy cover fractions
c   tempu_in, templ_in cold-phen state at entry
c   agddu_in, agddl_in GDD accumulators at entry
c   dropu_in, dropls_in, dropl4_in, dropl3_in  drought-phen state
c   zbot_in [n, 2]    canopy zbot (upper-zbot[:, 1] used)
c   ztop_in [n, 2]    canopy ztop (output; lower-ztop[:, 0] modified)
c   lai_in  [n, 2]    canopy LAI (output)
c   frac_in [n, 12]   PFT fractions (output)
c   epsilon_in         small-number floor
c ---------------------------------------------------------------------
      subroutine wrap_pheno(
     >    n_in,
     >    td_in, a10td_in,
     >    a10ancub_in, a10ancls_in, a10ancl4_in, a10ancl3_in,
     >    tc_in, plai_in, fu_in, fl_in,
     >    tempu_in, templ_in, agddu_in, agddl_in,
     >    dropu_in, dropls_in, dropl4_in, dropl3_in,
     >    zbot_in, ztop_in, lai_in, frac_in,
     >    epsilon_in,
     >    tempu_out, templ_out, agddu_out, agddl_out,
     >    dropu_out, dropls_out, dropl4_out, dropl3_out,
     >    frac_out, lai_out, ztop_out, zbot_out)
cf2py intent(in)  n_in
cf2py intent(in)  td_in, a10td_in
cf2py intent(in)  a10ancub_in, a10ancls_in, a10ancl4_in, a10ancl3_in
cf2py intent(in)  tc_in, plai_in, fu_in, fl_in
cf2py intent(in)  tempu_in, templ_in, agddu_in, agddl_in
cf2py intent(in)  dropu_in, dropls_in, dropl4_in, dropl3_in
cf2py intent(in)  zbot_in, ztop_in, lai_in, frac_in
cf2py intent(in)  epsilon_in
cf2py intent(out) tempu_out, templ_out, agddu_out, agddl_out
cf2py intent(out) dropu_out, dropls_out, dropl4_out, dropl3_out
cf2py intent(out) frac_out, lai_out, ztop_out, zbot_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'comsoi.h'
      include 'comsum.h'
      include 'comveg.h'

      integer n_in
      real*8 td_in(n_in), a10td_in(n_in)
      real*8 a10ancub_in(n_in), a10ancls_in(n_in)
      real*8 a10ancl4_in(n_in), a10ancl3_in(n_in)
      real*8 tc_in(n_in)
      real*8 plai_in(n_in, 12)
      real*8 fu_in(n_in), fl_in(n_in)
      real*8 tempu_in(n_in), templ_in(n_in)
      real*8 agddu_in(n_in), agddl_in(n_in)
      real*8 dropu_in(n_in), dropls_in(n_in)
      real*8 dropl4_in(n_in), dropl3_in(n_in)
      real*8 zbot_in(n_in, 2), ztop_in(n_in, 2)
      real*8 lai_in(n_in, 2), frac_in(n_in, 12)
      real*8 epsilon_in
      real*8 tempu_out(n_in), templ_out(n_in)
      real*8 agddu_out(n_in), agddl_out(n_in)
      real*8 dropu_out(n_in), dropls_out(n_in)
      real*8 dropl4_out(n_in), dropl3_out(n_in)
      real*8 frac_out(n_in, 12), lai_out(n_in, 2)
      real*8 ztop_out(n_in, 2), zbot_out(n_in, 2)

      integer i, k

      epsilon = epsilon_in

c seed all npoi entries with safe defaults
      do i = 1, npoi
        td(i) = 285.0
        a10td(i) = 285.0
        a10ancub(i) = 0.0
        a10ancls(i) = 0.0
        a10ancl4(i) = 0.0
        a10ancl3(i) = 0.0
        tc(i) = 10.0
        do k = 1, 12
          plai(i, k) = 0.0
          frac(i, k) = 0.0
        end do
        fu(i) = 0.5
        fl(i) = 0.5
        tempu(i) = 0.0; templ(i) = 0.0
        agddu(i) = 0.0; agddl(i) = 0.0
        dropu(i) = 1.0; dropls(i) = 1.0
        dropl4(i) = 1.0; dropl3(i) = 1.0
        do k = 1, 2
          lai(i, k) = 0.0
          zbot(i, k) = 0.0
          ztop(i, k) = 1.0
        end do
      end do

      do i = 1, n_in
        td(i)      = td_in(i)
        a10td(i)   = a10td_in(i)
        a10ancub(i) = a10ancub_in(i)
        a10ancls(i) = a10ancls_in(i)
        a10ancl4(i) = a10ancl4_in(i)
        a10ancl3(i) = a10ancl3_in(i)
        tc(i)      = tc_in(i)
        do k = 1, 12
          plai(i, k) = plai_in(i, k)
          frac(i, k) = frac_in(i, k)
        end do
        fu(i)      = fu_in(i)
        fl(i)      = fl_in(i)
        tempu(i)   = tempu_in(i)
        templ(i)   = templ_in(i)
        agddu(i)   = agddu_in(i)
        agddl(i)   = agddl_in(i)
        dropu(i)   = dropu_in(i)
        dropls(i)  = dropls_in(i)
        dropl4(i)  = dropl4_in(i)
        dropl3(i)  = dropl3_in(i)
        do k = 1, 2
          lai(i, k)  = lai_in(i, k)
          zbot(i, k) = zbot_in(i, k)
          ztop(i, k) = ztop_in(i, k)
        end do
      end do

      call pheno

      do i = 1, n_in
        tempu_out(i) = tempu(i)
        templ_out(i) = templ(i)
        agddu_out(i) = agddu(i)
        agddl_out(i) = agddl(i)
        dropu_out(i)  = dropu(i)
        dropls_out(i) = dropls(i)
        dropl4_out(i) = dropl4(i)
        dropl3_out(i) = dropl3(i)
        do k = 1, 12
          frac_out(i, k) = frac(i, k)
        end do
        do k = 1, 2
          lai_out(i, k)  = lai(i, k)
          ztop_out(i, k) = ztop(i, k)
          zbot_out(i, k) = zbot(i, k)
        end do
      end do

      return
      end
