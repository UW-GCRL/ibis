c =====================================================================
c radiation_wrap.f
c
c F77 wrappers exposing twoset and twostr from radiation.f as callable
c subroutines for f2py.  Both routines read inputs from common blocks
c (compar.h, comveg.h, com1d.h); the wrappers accept them as explicit
c arguments and copy them into the common-block storage before calling
c the original F77 routine.
c
c We include only the same headers that twoset/twostr themselves
c include -- not comatm.h -- so that the wrapper's positional-arg names
c (coszen, asurd, asuri, ...) don't collide with common-block names.
c
c npoi is a compile-time parameter in compar.h (= 1718). We populate
c only the first n_in elements of every array.
c =====================================================================


c ---------------------------------------------------------------------
c wrap_twoset
c ---------------------------------------------------------------------
      subroutine wrap_twoset(
     >    n_in,
     >    iv_in, ib_in,
     >    coszen_in,
     >    rhoveg_in, tauveg_in, oriev_in, orieh_in,
     >    fwetl_in, rliql_in,
     >    fwetu_in, rliqu_in,
     >    fwets_in, rliqs_in,
     >    tl_in, tu_in, ts_in,
     >    lai_in, sai_in,
     >    tmelt_in,
     >    omega_out, betad_out, betai_out, avmu_out, gdir_out)
cf2py intent(hide), depend(coszen_in) :: n_in=len(coszen_in)
cf2py intent(in)  iv_in
cf2py intent(in)  ib_in
cf2py intent(in)  coszen_in
cf2py intent(in)  rhoveg_in
cf2py intent(in)  tauveg_in
cf2py intent(in)  oriev_in
cf2py intent(in)  orieh_in
cf2py intent(in)  fwetl_in
cf2py intent(in)  rliql_in
cf2py intent(in)  fwetu_in
cf2py intent(in)  rliqu_in
cf2py intent(in)  fwets_in
cf2py intent(in)  rliqs_in
cf2py intent(in)  tl_in
cf2py intent(in)  tu_in
cf2py intent(in)  ts_in
cf2py intent(in)  lai_in
cf2py intent(in)  sai_in
cf2py intent(in)  tmelt_in
cf2py intent(out) omega_out
cf2py intent(out) betad_out
cf2py intent(out) betai_out
cf2py intent(out) avmu_out
cf2py intent(out) gdir_out

      include 'implicit.h'
      include 'compar.h'
      include 'comveg.h'
      include 'com1d.h'

      integer n_in, iv_in, ib_in
      real*8 coszen_in(n_in)
      real*8 rhoveg_in, tauveg_in, oriev_in, orieh_in
      real*8 fwetl_in(n_in), rliql_in(n_in)
      real*8 fwetu_in(n_in), rliqu_in(n_in)
      real*8 fwets_in(n_in), rliqs_in(n_in)
      real*8 tl_in(n_in), tu_in(n_in), ts_in(n_in)
      real*8 lai_in(n_in), sai_in(n_in)
      real*8 tmelt_in
      real*8 omega_out(n_in), betad_out(n_in), betai_out(n_in)
      real*8 avmu_out(n_in), gdir_out(n_in)

      integer i
      real cosz(npoi)
      real omega(npoi), betad(npoi), betai(npoi)
      real avmu(npoi), gdir(npoi)

      nsol = n_in
      do i = 1, n_in
        indsol(i) = i
      end do

      rhoveg(ib_in, iv_in) = rhoveg_in
      tauveg(ib_in, iv_in) = tauveg_in
      oriev(iv_in)         = oriev_in
      orieh(iv_in)         = orieh_in
      tmelt                = tmelt_in
      epsilon              = 1.0d-7

      do i = 1, n_in
        cosz(i)       = coszen_in(i)
        fwetl(i)      = fwetl_in(i)
        rliql(i)      = rliql_in(i)
        fwetu(i)      = fwetu_in(i)
        rliqu(i)      = rliqu_in(i)
        fwets(i)      = fwets_in(i)
        rliqs(i)      = rliqs_in(i)
        tl(i)         = tl_in(i)
        tu(i)         = tu_in(i)
        ts(i)         = ts_in(i)
        lai(i, iv_in) = lai_in(i)
        sai(i, iv_in) = sai_in(i)
      end do

      call twoset(omega, betad, betai, avmu, gdir, cosz, iv_in, ib_in)

      do i = 1, n_in
        omega_out(i) = omega(i)
        betad_out(i) = betad(i)
        betai_out(i) = betai(i)
        avmu_out(i)  = avmu(i)
        gdir_out(i)  = gdir(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_twostr
c ---------------------------------------------------------------------
      subroutine wrap_twostr(
     >    n_in,
     >    iv_in, ib_in,
     >    coszen_in,
     >    rhoveg_in, tauveg_in, oriev_in, orieh_in,
     >    fwetl_in, rliql_in,
     >    fwetu_in, rliqu_in,
     >    fwets_in, rliqs_in,
     >    tl_in, tu_in, ts_in,
     >    lai_in, sai_in,
     >    tmelt_in,
     >    asurd_in, asuri_in,
     >    abvegd_out, abvegi_out,
     >    refld_out, refli_out,
     >    fbeldd_out, fbeldi_out,
     >    fbelid_out, fbelii_out)
cf2py intent(hide), depend(coszen_in) :: n_in=len(coszen_in)
cf2py intent(in)  iv_in
cf2py intent(in)  ib_in
cf2py intent(in)  coszen_in
cf2py intent(in)  rhoveg_in
cf2py intent(in)  tauveg_in
cf2py intent(in)  oriev_in
cf2py intent(in)  orieh_in
cf2py intent(in)  fwetl_in
cf2py intent(in)  rliql_in
cf2py intent(in)  fwetu_in
cf2py intent(in)  rliqu_in
cf2py intent(in)  fwets_in
cf2py intent(in)  rliqs_in
cf2py intent(in)  tl_in
cf2py intent(in)  tu_in
cf2py intent(in)  ts_in
cf2py intent(in)  lai_in
cf2py intent(in)  sai_in
cf2py intent(in)  tmelt_in
cf2py intent(in)  asurd_in
cf2py intent(in)  asuri_in
cf2py intent(out) abvegd_out
cf2py intent(out) abvegi_out
cf2py intent(out) refld_out
cf2py intent(out) refli_out
cf2py intent(out) fbeldd_out
cf2py intent(out) fbeldi_out
cf2py intent(out) fbelid_out
cf2py intent(out) fbelii_out

      include 'implicit.h'
      include 'compar.h'
      include 'comveg.h'
      include 'com1d.h'

      integer n_in, iv_in, ib_in
      real*8 coszen_in(n_in)
      real*8 rhoveg_in, tauveg_in, oriev_in, orieh_in
      real*8 fwetl_in(n_in), rliql_in(n_in)
      real*8 fwetu_in(n_in), rliqu_in(n_in)
      real*8 fwets_in(n_in), rliqs_in(n_in)
      real*8 tl_in(n_in), tu_in(n_in), ts_in(n_in)
      real*8 lai_in(n_in), sai_in(n_in)
      real*8 tmelt_in
      real*8 asurd_in(n_in), asuri_in(n_in)
      real*8 abvegd_out(n_in), abvegi_out(n_in)
      real*8 refld_out(n_in), refli_out(n_in)
      real*8 fbeldd_out(n_in), fbeldi_out(n_in)
      real*8 fbelid_out(n_in), fbelii_out(n_in)

      integer i
      real cosz(npoi)
      real asd(npoi, nband), asi(npoi, nband)
      real abvegd(npoi), abvegi(npoi)
      real refld(npoi), refli(npoi)
      real fbeldd(npoi), fbeldi(npoi)
      real fbelid(npoi), fbelii(npoi)

      nsol = n_in
      do i = 1, n_in
        indsol(i) = i
      end do

      rhoveg(ib_in, iv_in) = rhoveg_in
      tauveg(ib_in, iv_in) = tauveg_in
      oriev(iv_in)         = oriev_in
      orieh(iv_in)         = orieh_in
      tmelt                = tmelt_in
c epsilon read by twostr (xai = max(lai+sai, epsilon)); also used by
c twoset's snow adjustment. Setting it explicitly so we don't depend on
c uninitialized common-block memory.
      epsilon              = 1.0d-7

      do i = 1, n_in
        cosz(i)           = coszen_in(i)
        fwetl(i)          = fwetl_in(i)
        rliql(i)          = rliql_in(i)
        fwetu(i)          = fwetu_in(i)
        rliqu(i)          = rliqu_in(i)
        fwets(i)          = fwets_in(i)
        rliqs(i)          = rliqs_in(i)
        tl(i)             = tl_in(i)
        tu(i)             = tu_in(i)
        ts(i)             = ts_in(i)
        lai(i, iv_in)     = lai_in(i)
        sai(i, iv_in)     = sai_in(i)
        asd(i, ib_in)     = asurd_in(i)
        asi(i, ib_in)     = asuri_in(i)
      end do

      call twostr(abvegd, abvegi, refld, refli,
     >            fbeldd, fbeldi, fbelid, fbelii,
     >            asd, asi, iv_in, cosz, ib_in)

      do i = 1, n_in
        abvegd_out(i) = abvegd(i)
        abvegi_out(i) = abvegi(i)
        refld_out(i)  = refld(i)
        refli_out(i)  = refli(i)
        fbeldd_out(i) = fbeldd(i)
        fbeldi_out(i) = fbeldi(i)
        fbelid_out(i) = fbelid(i)
        fbelii_out(i) = fbelii(i)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_radiation_full -- full radiation chain: solset + solsur + solalb +
c solarf (for ib=1,2) + irrad. Mirrors JAX compute_radiation().
c ---------------------------------------------------------------------
      subroutine wrap_radiation_full(
     >    n_in,
     >    coszen_in, solad_in, solai_in, fira_in,
     >    lai_in, sai_in, fu_in, fl_in, fi_in,
     >    tu_in, ts_in, tl_in, tg_in, ti_in,
     >    tsno1_in,
     >    wsoi1_in, wisoi1_in, albsav_in, albsan_in,
     >    fwetu_in, fwets_in, fwetl_in,
     >    rliqu_in, rliqs_in, rliql_in,
     >    rhoveg_in, tauveg_in, oriev_in, orieh_in,
     >    stef_in, tmelt_in, epsilon_in,
     >    solu_out, sols_out, soll_out, solg_out, soli_out,
     >    firu_out, firs_out, firl_out, firg_out, firi_out, firb_out,
     >    asurd_out, asuri_out,
     >    topparu_out, topparl_out,
     >    scalcoefu_out, scalcoefl_out)
cf2py intent(hide), depend(coszen_in) :: n_in=len(coszen_in)
cf2py intent(in)  coszen_in, solad_in, solai_in, fira_in
cf2py intent(in)  lai_in, sai_in, fu_in, fl_in, fi_in
cf2py intent(in)  tu_in, ts_in, tl_in, tg_in, ti_in
cf2py intent(in)  tsno1_in
cf2py intent(in)  wsoi1_in, wisoi1_in, albsav_in, albsan_in
cf2py intent(in)  fwetu_in, fwets_in, fwetl_in
cf2py intent(in)  rliqu_in, rliqs_in, rliql_in
cf2py intent(in)  rhoveg_in, tauveg_in, oriev_in, orieh_in
cf2py intent(in)  stef_in, tmelt_in, epsilon_in
cf2py intent(out) solu_out, sols_out, soll_out, solg_out, soli_out
cf2py intent(out) firu_out, firs_out, firl_out, firg_out, firi_out, firb_out
cf2py intent(out) asurd_out, asuri_out
cf2py intent(out) topparu_out, topparl_out
cf2py intent(out) scalcoefu_out, scalcoefl_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'comsno.h'
      include 'comsoi.h'
      include 'comveg.h'
      include 'com1d.h'

      integer n_in
      real*8 coszen_in(n_in), solad_in(n_in, 2), solai_in(n_in, 2)
      real*8 fira_in(n_in)
      real*8 lai_in(n_in, 2), sai_in(n_in, 2)
      real*8 fu_in(n_in), fl_in(n_in), fi_in(n_in)
      real*8 tu_in(n_in), ts_in(n_in), tl_in(n_in)
      real*8 tg_in(n_in), ti_in(n_in)
      real*8 tsno1_in(n_in)
      real*8 wsoi1_in(n_in), wisoi1_in(n_in)
      real*8 albsav_in(n_in), albsan_in(n_in)
      real*8 fwetu_in(n_in), fwets_in(n_in), fwetl_in(n_in)
      real*8 rliqu_in(n_in), rliqs_in(n_in), rliql_in(n_in)
      real*8 rhoveg_in(2, 2), tauveg_in(2, 2)
      real*8 oriev_in(2), orieh_in(2)
      real*8 stef_in, tmelt_in, epsilon_in
      real*8 solu_out(n_in), sols_out(n_in), soll_out(n_in)
      real*8 solg_out(n_in), soli_out(n_in)
      real*8 firu_out(n_in), firs_out(n_in), firl_out(n_in)
      real*8 firg_out(n_in), firi_out(n_in), firb_out(n_in)
      real*8 asurd_out(n_in, 2), asuri_out(n_in, 2)
      real*8 topparu_out(n_in), topparl_out(n_in)
      real*8 scalcoefu_out(n_in, 4), scalcoefl_out(n_in, 4)

      integer i, k, ib

      stef    = stef_in
      tmelt   = tmelt_in
      epsilon = epsilon_in

c veg radiation params (compile-time nband=2, 2 layers)
      do k = 1, 2
        oriev(k) = oriev_in(k)
        orieh(k) = orieh_in(k)
        do ib = 1, 2
          rhoveg(ib, k) = rhoveg_in(ib, k)
          tauveg(ib, k) = tauveg_in(ib, k)
        end do
      end do

c seed all npoi positions with safe defaults
c (coszen default = 0 so trailing entries don't inflate nsol; sunlit count
c then reflects only the n_in user points)
      do i = 1, npoi
        coszen(i) = 0.0
        fira(i)   = 300.0
        do ib = 1, 2
          solad(i, ib) = 0.0
          solai(i, ib) = 0.0
        end do
        do k = 1, 2
          lai(i, k) = 0.0
          sai(i, k) = 0.0
        end do
        fu(i) = 0.5; fl(i) = 0.5; fi(i) = 0.0
        tu(i) = 285.0; ts(i) = 285.0; tl(i) = 285.0
        tg(i) = 285.0; ti(i) = 270.0
        tsno(i, 1) = 270.0
        wsoi(i, 1) = 0.3; wisoi(i, 1) = 0.0
        albsav(i) = 0.15; albsan(i) = 0.25
        fwetu(i) = 0.0; fwets(i) = 0.0; fwetl(i) = 0.0
        rliqu(i) = 1.0; rliqs(i) = 1.0; rliql(i) = 1.0
      end do

c overwrite first n_in
      do i = 1, n_in
        coszen(i) = coszen_in(i)
        fira(i)   = fira_in(i)
        do ib = 1, 2
          solad(i, ib) = solad_in(i, ib)
          solai(i, ib) = solai_in(i, ib)
        end do
        do k = 1, 2
          lai(i, k) = lai_in(i, k)
          sai(i, k) = sai_in(i, k)
        end do
        fu(i) = fu_in(i); fl(i) = fl_in(i); fi(i) = fi_in(i)
        tu(i) = tu_in(i); ts(i) = ts_in(i); tl(i) = tl_in(i)
        tg(i) = tg_in(i); ti(i) = ti_in(i)
        tsno(i, 1) = tsno1_in(i)
        wsoi(i, 1) = wsoi1_in(i); wisoi(i, 1) = wisoi1_in(i)
        albsav(i) = albsav_in(i); albsan(i) = albsan_in(i)
        fwetu(i) = fwetu_in(i); fwets(i) = fwets_in(i); fwetl(i) = fwetl_in(i)
        rliqu(i) = rliqu_in(i); rliqs(i) = rliqs_in(i); rliql(i) = rliql_in(i)
      end do

      call solset
      do ib = 1, 2
        call solsur(ib)
        call solalb(ib)
        call solarf(ib)
      end do
      call irrad

      do i = 1, n_in
        solu_out(i) = solu(i); sols_out(i) = sols(i); soll_out(i) = soll(i)
        solg_out(i) = solg(i); soli_out(i) = soli(i)
        firu_out(i) = firu(i); firs_out(i) = firs(i); firl_out(i) = firl(i)
        firg_out(i) = firg(i); firi_out(i) = firi(i); firb_out(i) = firb(i)
        do ib = 1, 2
          asurd_out(i, ib) = asurd(i, ib)
          asuri_out(i, ib) = asuri(i, ib)
        end do
        topparu_out(i) = topparu(i)
        topparl_out(i) = topparl(i)
        do k = 1, 4
          scalcoefu_out(i, k) = scalcoefu(i, k)
          scalcoefl_out(i, k) = scalcoefl(i, k)
        end do
      end do

      return
      end
