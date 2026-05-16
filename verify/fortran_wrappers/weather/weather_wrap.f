c =====================================================================
c weather_wrap.f
c
c F77 wrappers for weather verification.
c   - wrap_ran2_seq(iseed, n) -> array of n successive uniform deviates
c   - wrap_daily(...)         -> one-day call of subroutine daily
c =====================================================================


c ---------------------------------------------------------------------
c wrap_ran2_seq -- call ran2 with idum=-iseed once (to trigger init),
c then n-1 more times. Returns the resulting length-n stream.
c ---------------------------------------------------------------------
      subroutine wrap_ran2_seq(iseed, n, out)
cf2py intent(in)  iseed
cf2py intent(in)  n
cf2py intent(out) out
cf2py depend(n)   out

      include 'implicit.h'

      integer iseed, n
      real*8  out(n)

      integer idum, i
      real    ran2

      idum = -iseed
      out(1) = ran2(idum)
      do i = 2, n
        out(i) = ran2(idum)
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_daily -- one call of `daily` (jdaily=0, weather-generator path).
c
c Inputs:
c   n_in            number of grid points (must be <= npoi=1718)
c   iseed           positive int; we pass seed=-iseed at entry so ran2
c                   re-initializes deterministically for this call
c   imonth_in       1..12
c   iday_in         1..ndaypm(imonth)
c   xint_in, xinq_in, xinprec_in, xinwind_in, xincld_in, xinwet_in,
c     xintrng_in    each [n_in, 12] climatological monthly inputs
c   xintopo_in      [n_in] elevation (m)
c   xstore_in       [n_in, 3] AR(1) memory state at entry
c   iwetday_in      [n_in, 31] integer wet-day flags at entry
c   precipday_in    [n_in, 31] daily precip storage at entry
c   iwetdaysum_in   [n_in] monthly wet-day count at entry
c   precipdaysum_in [n_in] monthly precip total at entry
c   tcthis_in, twthis_in, gdd0this_in, gdd5this_in [n_in] year-running fields
c   a10td_in        [n_in] 10-day running mean T (K) -- read-only by daily
c   grav_in, rair_in  scalar physical constants
c
c Outputs:
c   td_out, tmax_out, tmin_out, cloud_out, precip_out, qd_out, ud_out,
c     psurf_out, iwet_out   [n_in] daily outputs
c   xstore_out      [n_in, 3]
c   iwetday_out     [n_in, 31]
c   precipday_out   [n_in, 31]
c   iwetdaysum_out, precipdaysum_out [n_in]
c   tcthis_out, twthis_out, gdd0this_out, gdd5this_out [n_in]
c   xinwet_out      [n_in, 12]  -- xinwet may be mutated by the retry path
c ---------------------------------------------------------------------
      subroutine wrap_daily(
     >    n_in, iseed, imonth_in, iday_in,
     >    xint_in, xinq_in, xinprec_in, xinwind_in,
     >    xincld_in, xinwet_in, xintrng_in, xintopo_in,
     >    xstore_in, iwetday_in, precipday_in,
     >    iwetdaysum_in, precipdaysum_in,
     >    tcthis_in, twthis_in, gdd0this_in, gdd5this_in,
     >    a10td_in, grav_in, rair_in,
     >    td_out, tmax_out, tmin_out, cloud_out, precip_out,
     >    qd_out, ud_out, psurf_out, iwet_out,
     >    xstore_out, iwetday_out, precipday_out,
     >    iwetdaysum_out, precipdaysum_out,
     >    tcthis_out, twthis_out, gdd0this_out, gdd5this_out,
     >    xinwet_out)

cf2py intent(in)  n_in
cf2py intent(in)  iseed
cf2py intent(in)  imonth_in
cf2py intent(in)  iday_in
cf2py intent(in)  xint_in
cf2py intent(in)  xinq_in
cf2py intent(in)  xinprec_in
cf2py intent(in)  xinwind_in
cf2py intent(in)  xincld_in
cf2py intent(in)  xinwet_in
cf2py intent(in)  xintrng_in
cf2py intent(in)  xintopo_in
cf2py intent(in)  xstore_in
cf2py intent(in)  iwetday_in
cf2py intent(in)  precipday_in
cf2py intent(in)  iwetdaysum_in
cf2py intent(in)  precipdaysum_in
cf2py intent(in)  tcthis_in
cf2py intent(in)  twthis_in
cf2py intent(in)  gdd0this_in
cf2py intent(in)  gdd5this_in
cf2py intent(in)  a10td_in
cf2py intent(in)  grav_in
cf2py intent(in)  rair_in
cf2py intent(out) td_out
cf2py intent(out) tmax_out
cf2py intent(out) tmin_out
cf2py intent(out) cloud_out
cf2py intent(out) precip_out
cf2py intent(out) qd_out
cf2py intent(out) ud_out
cf2py intent(out) psurf_out
cf2py intent(out) iwet_out
cf2py intent(out) xstore_out
cf2py intent(out) iwetday_out
cf2py intent(out) precipday_out
cf2py intent(out) iwetdaysum_out
cf2py intent(out) precipdaysum_out
cf2py intent(out) tcthis_out
cf2py intent(out) twthis_out
cf2py intent(out) gdd0this_out
cf2py intent(out) gdd5this_out
cf2py intent(out) xinwet_out

      include 'implicit.h'
      include 'compar.h'
      include 'combcs.h'
      include 'comatm.h'
      include 'comsum.h'
      include 'comveg.h'

      integer n_in, iseed, imonth_in, iday_in
      real*8  xint_in(n_in, 12)
      real*8  xinq_in(n_in, 12)
      real*8  xinprec_in(n_in, 12)
      real*8  xinwind_in(n_in, 12)
      real*8  xincld_in(n_in, 12)
      real*8  xinwet_in(n_in, 12)
      real*8  xintrng_in(n_in, 12)
      real*8  xintopo_in(n_in)
      real*8  xstore_in(n_in, 3)
      integer iwetday_in(n_in, 31)
      real*8  precipday_in(n_in, 31)
      integer iwetdaysum_in(n_in)
      real*8  precipdaysum_in(n_in)
      real*8  tcthis_in(n_in)
      real*8  twthis_in(n_in)
      real*8  gdd0this_in(n_in)
      real*8  gdd5this_in(n_in)
      real*8  a10td_in(n_in)
      real*8  grav_in, rair_in

      real*8  td_out(n_in), tmax_out(n_in), tmin_out(n_in)
      real*8  cloud_out(n_in), precip_out(n_in)
      real*8  qd_out(n_in), ud_out(n_in), psurf_out(n_in)
      integer iwet_out(n_in)
      real*8  xstore_out(n_in, 3)
      integer iwetday_out(n_in, 31)
      real*8  precipday_out(n_in, 31)
      integer iwetdaysum_out(n_in)
      real*8  precipdaysum_out(n_in)
      real*8  tcthis_out(n_in), twthis_out(n_in)
      real*8  gdd0this_out(n_in), gdd5this_out(n_in)
      real*8  xinwet_out(n_in, 12)

      integer i, m
      integer seed
      integer iyranom_loc, iyrlast_loc, nrun_loc
      real    ran2_init_dummy

c --- 1. set physical constants in common ---
      grav    = grav_in
      rair    = rair_in
c we also touch a few that daily reads transitively, set to safe values
      epsilon = 1.0d-12
      tmelt   = 273.16
      hfus    = 0.3336d6
      hvap    = 2.5104d6
      hsub    = hvap + hfus
      ch2o    = 4.218d3
      cice    = 2.106d3
      cair    = 1.005d3
      cvap    = 1.81d3
      stef    = 5.67e-8

c --- 2. set ndaypm (used by daily and indexed by imonth) ---
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

c --- 3. seed all npoi positions with safe defaults, then overwrite first n_in
c daily loops i=1,npoi (compile-time, =1718), so trailing slots must be
c safe values that don't NaN or divide-by-zero.
      do i = 1, npoi
        xintopo(i) = 0.0
        do m = 1, 12
          xint(i, m)    = 20.0     ! 20C
          xinq(i, m)    = 60.0     ! 60% RH
          xinprec(i, m) = 1.0      ! 1 mm/day
          xinwind(i, m) = 3.0
          xincld(i, m)  = 50.0
          xinwet(i, m)  = 5.0
          xintrng(i, m) = 10.0
        end do
        xstore(i, 1) = 0.0
        xstore(i, 2) = 0.0
        xstore(i, 3) = 0.0
        iwet(i)     = 0
        iwetdaysum(i) = 0
        precipdaysum(i) = 0.0
        do m = 1, 31
          iwetday(i, m)   = 0
          precipday(i, m) = 0.0
        end do
        tcthis(i)   = 50.0
        twthis(i)   = -50.0
        gdd0this(i) = 0.0
        gdd5this(i) = 0.0
        a10td(i)    = 288.0
      end do

c --- 4. now overwrite the first n_in positions from user inputs ---
      do i = 1, n_in
        xintopo(i) = xintopo_in(i)
        do m = 1, 12
          xint(i, m)    = xint_in(i, m)
          xinq(i, m)    = xinq_in(i, m)
          xinprec(i, m) = xinprec_in(i, m)
          xinwind(i, m) = xinwind_in(i, m)
          xincld(i, m)  = xincld_in(i, m)
          xinwet(i, m)  = xinwet_in(i, m)
          xintrng(i, m) = xintrng_in(i, m)
        end do
        xstore(i, 1) = xstore_in(i, 1)
        xstore(i, 2) = xstore_in(i, 2)
        xstore(i, 3) = xstore_in(i, 3)
        iwetdaysum(i)   = iwetdaysum_in(i)
        precipdaysum(i) = precipdaysum_in(i)
        do m = 1, 31
          iwetday(i, m)   = iwetday_in(i, m)
          precipday(i, m) = precipday_in(i, m)
        end do
        tcthis(i)   = tcthis_in(i)
        twthis(i)   = twthis_in(i)
        gdd0this(i) = gdd0this_in(i)
        gdd5this(i) = gdd5this_in(i)
        a10td(i)    = a10td_in(i)
      end do

c --- 5. force ran2 re-init by calling once with negative seed.
c Once ran2 returns, internal SAVE state (iv, iy, idum2) is fresh.
c We pay one ran2 draw, so daily's first ran2 sees the *second*-from-init
c output. Reference Python's Ran2(iseed) also burns one call equivalently.
c Trick: just have daily's caller responsible for using the same trick.
c
c Simpler: pass seed=-iseed to daily. daily's first ran2(seed) will see
c idum<=0 and initialize. Then subsequent calls step normally.
      seed = -iseed

c --- 6. call daily ---
      iyranom_loc = 9999    ! not used in jdaily=0 path
      iyrlast_loc = 9999
      nrun_loc    = 1

      call daily(2000, imonth_in, iday_in, seed,
     >           0, iyranom_loc, iyrlast_loc, nrun_loc)

c --- 7. copy outputs ---
      do i = 1, n_in
        td_out(i)     = td(i)
        tmax_out(i)   = tmax(i)
        tmin_out(i)   = tmin(i)
        cloud_out(i)  = cloud(i)
        precip_out(i) = precip(i)
        qd_out(i)     = qd(i)
        ud_out(i)     = ud(i)
        psurf_out(i)  = psurf(i)
        iwet_out(i)   = iwet(i)
        xstore_out(i, 1) = xstore(i, 1)
        xstore_out(i, 2) = xstore(i, 2)
        xstore_out(i, 3) = xstore(i, 3)
        iwetdaysum_out(i)   = iwetdaysum(i)
        precipdaysum_out(i) = precipdaysum(i)
        do m = 1, 31
          iwetday_out(i, m)   = iwetday(i, m)
          precipday_out(i, m) = precipday(i, m)
        end do
        tcthis_out(i)   = tcthis(i)
        twthis_out(i)   = twthis(i)
        gdd0this_out(i) = gdd0this(i)
        gdd5this_out(i) = gdd5this(i)
        do m = 1, 12
          xinwet_out(i, m) = xinwet(i, m)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_diurnal -- one call of `diurnal` (sub-daily disaggregation).
c
c Inputs:
c   n_in            number of grid points (must be <= npoi)
c   iseed           positive int; passes seed=-iseed at entry to force
c                   ran2 re-init for this call
c   time_in         seconds into the day (0 .. 86400 - dtime)
c   jday_in         day of year (1..365)
c   plens_in, startp_in, endp_in   precip event timing (s) -- INOUT
c   dtime_in        timestep length (s) -- e.g., 3600 for hourly
c   cloud_in, tmax_in, tmin_in, qd_in, ud_in, psurf_in, precip_in [n_in]
c                   per-grid-point daily-state inputs (from daily output)
c   lat_in          [n_in] latitude in degrees (we map latindex(i)=i and
c                   latscale(i)=lat_in(i))
c   grav_in, rair_in  scalar constants (touched only because daily wrapper
c                   already touches these; diurnal does not use them).
c
c Outputs:
c   coszen_out, ta_out, qa_out, ua_out, rh_out [n_in]
c   fira_out, raina_out, snowa_out [n_in]
c   solad_out, solai_out [n_in, 2]   (nband=2)
c   plens_out, startp_out, endp_out  (may be updated by event-timing retry)
c ---------------------------------------------------------------------
      subroutine wrap_diurnal(
     >    n_in, iseed,
     >    time_in, jday_in,
     >    plens_in, startp_in, endp_in, dtime_in,
     >    cloud_in, tmax_in, tmin_in, qd_in, ud_in, psurf_in,
     >    precip_in, lat_in,
     >    coszen_out, ta_out, qa_out, ua_out, rh_out,
     >    fira_out, raina_out, snowa_out,
     >    solad_out, solai_out,
     >    plens_out, startp_out, endp_out)

cf2py intent(in)  n_in
cf2py intent(in)  iseed
cf2py intent(in)  time_in
cf2py intent(in)  jday_in
cf2py intent(in)  plens_in
cf2py intent(in)  startp_in
cf2py intent(in)  endp_in
cf2py intent(in)  dtime_in
cf2py intent(in)  cloud_in
cf2py intent(in)  tmax_in
cf2py intent(in)  tmin_in
cf2py intent(in)  qd_in
cf2py intent(in)  ud_in
cf2py intent(in)  psurf_in
cf2py intent(in)  precip_in
cf2py intent(in)  lat_in
cf2py intent(out) coszen_out
cf2py intent(out) ta_out
cf2py intent(out) qa_out
cf2py intent(out) ua_out
cf2py intent(out) rh_out
cf2py intent(out) fira_out
cf2py intent(out) raina_out
cf2py intent(out) snowa_out
cf2py intent(out) solad_out
cf2py intent(out) solai_out
cf2py intent(out) plens_out
cf2py intent(out) startp_out
cf2py intent(out) endp_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'comwork.h'

      integer n_in, iseed, jday_in
      real*8  time_in
      real*8  plens_in, startp_in, endp_in, dtime_in
      real*8  cloud_in(n_in), tmax_in(n_in), tmin_in(n_in)
      real*8  qd_in(n_in), ud_in(n_in), psurf_in(n_in)
      real*8  precip_in(n_in), lat_in(n_in)
      real*8  coszen_out(n_in), ta_out(n_in), qa_out(n_in)
      real*8  ua_out(n_in), rh_out(n_in)
      real*8  fira_out(n_in), raina_out(n_in), snowa_out(n_in)
      real*8  solad_out(n_in, 2), solai_out(n_in, 2)
      real*8  plens_out, startp_out, endp_out

      integer i, ib
      integer seed
      real    time_loc, plens_loc, startp_loc, endp_loc

c --- 1. physical constants + dtime ---
      dtime   = dtime_in
      stef    = 5.67e-8
      epsilon = 1.0d-12
      tmelt   = 273.16
      hvap    = 2.5104d6
      hfus    = 0.3336d6
      hsub    = hvap + hfus
      ch2o    = 4.218d3
      cice    = 2.106d3
      cair    = 1.005d3
      cvap    = 1.81d3

c --- 2. seed npoi-sized commons with safe defaults ---
      do i = 1, npoi
        cloud(i)  = 0.5
        tmax(i)   = 290.0
        tmin(i)   = 280.0
        qd(i)     = 0.005
        ud(i)     = 3.0
        psurf(i)  = 101325.0
        precip(i) = 0.0
        latindex(i) = 1
      end do

c diurnal references latscale(latindex(i)), so set latscale large enough.
      do i = 1, nlat
        latscale(i) = 0.0
      end do

c --- 3. overwrite first n_in with user inputs ---
c Map each grid point to its own latitude entry: latindex(i) = i, latscale(i) = lat_in(i).
c (Requires n_in <= nlat; nlat is from compar, typically 180 or so.)
      do i = 1, n_in
        cloud(i)    = cloud_in(i)
        tmax(i)     = tmax_in(i)
        tmin(i)     = tmin_in(i)
        qd(i)       = qd_in(i)
        ud(i)       = ud_in(i)
        psurf(i)    = psurf_in(i)
        precip(i)   = precip_in(i)
        latindex(i) = i
        latscale(i) = lat_in(i)
      end do

c --- 4. force ran2 re-init ---
      seed = -iseed
c "burn" one ran2 call so internal SAVE state matches the Python Ran2
c (the dummy call below mirrors what daily's first ran2 line does).
c Actually: just pass seed=-iseed to diurnal; its first ran2 call will
c trigger init. So no need to burn.

c --- 5. set up INOUT args ---
      time_loc   = time_in
      plens_loc  = plens_in
      startp_loc = startp_in
      endp_loc   = endp_in

c --- 6. call diurnal ---
      call diurnal(time_loc, jday_in,
     >             plens_loc, startp_loc, endp_loc, seed)

c --- 7. copy outputs ---
      do i = 1, n_in
        coszen_out(i) = coszen(i)
        ta_out(i)     = ta(i)
        qa_out(i)     = qa(i)
        ua_out(i)     = ua(i)
        rh_out(i)     = rh(i)
        fira_out(i)   = fira(i)
        raina_out(i)  = raina(i)
        snowa_out(i)  = snowa(i)
        do ib = 1, 2
          solad_out(i, ib) = solad(i, ib)
          solai_out(i, ib) = solai(i, ib)
        end do
      end do
      plens_out  = plens_loc
      startp_out = startp_loc
      endp_out   = endp_loc

      return
      end


c ---------------------------------------------------------------------
c wrap_daily_jdaily1 -- one call of `daily` with jdaily=1 (observed daily
c climatology branch, weather.f lines 824-891). Pure arithmetic; no RNG.
c ---------------------------------------------------------------------
      subroutine wrap_daily_jdaily1(
     >    n_in, imonth_in, iday_in,
     >    xint_in, xinq_in, xinprec_in, xinwind_in,
     >    xincld_in, xinwet_in, xintrng_in, xintopo_in,
     >    xintd_in, xinqd_in, xinprecd_in, xinwindd_in,
     >    xincldd_in, xintrngd_in,
     >    grav_in, rair_in,
     >    td_out, tmax_out, tmin_out, cloud_out, precip_out,
     >    qd_out, ud_out, psurf_out)
cf2py intent(in)  n_in, imonth_in, iday_in
cf2py intent(in)  xint_in, xinq_in, xinprec_in, xinwind_in
cf2py intent(in)  xincld_in, xinwet_in, xintrng_in, xintopo_in
cf2py intent(in)  xintd_in, xinqd_in, xinprecd_in, xinwindd_in
cf2py intent(in)  xincldd_in, xintrngd_in
cf2py intent(in)  grav_in, rair_in
cf2py intent(out) td_out, tmax_out, tmin_out, cloud_out, precip_out
cf2py intent(out) qd_out, ud_out, psurf_out

      include 'implicit.h'
      include 'compar.h'
      include 'combcs.h'
      include 'comatm.h'
      include 'comsum.h'
      include 'comveg.h'

      integer n_in, imonth_in, iday_in
      real*8 xint_in(n_in, 12)
      real*8 xinq_in(n_in, 12)
      real*8 xinprec_in(n_in, 12)
      real*8 xinwind_in(n_in, 12)
      real*8 xincld_in(n_in, 12)
      real*8 xinwet_in(n_in, 12)
      real*8 xintrng_in(n_in, 12)
      real*8 xintopo_in(n_in)
      real*8 xintd_in(n_in)
      real*8 xinqd_in(n_in)
      real*8 xinprecd_in(n_in)
      real*8 xinwindd_in(n_in)
      real*8 xincldd_in(n_in)
      real*8 xintrngd_in(n_in)
      real*8 grav_in, rair_in
      real*8 td_out(n_in), tmax_out(n_in), tmin_out(n_in)
      real*8 cloud_out(n_in), precip_out(n_in)
      real*8 qd_out(n_in), ud_out(n_in), psurf_out(n_in)

      integer i, m
      integer seed_unused
      integer iyranom_loc, iyrlast_loc, nrun_loc

      grav    = grav_in
      rair    = rair_in
      epsilon = 1.0d-12
      tmelt   = 273.16
      hfus    = 0.3336d6
      hvap    = 2.5104d6
      hsub    = hvap + hfus
      ch2o    = 4.218d3
      cice    = 2.106d3
      cair    = 1.005d3
      cvap    = 1.81d3
      stef    = 5.67e-8

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

c safe defaults
      do i = 1, npoi
        xintopo(i) = 0.0
        do m = 1, 12
          xint(i, m)    = 20.0
          xinq(i, m)    = 60.0
          xinprec(i, m) = 1.0
          xinwind(i, m) = 3.0
          xincld(i, m)  = 50.0
          xinwet(i, m)  = 5.0
          xintrng(i, m) = 10.0
        end do
        xintd(i)    = 0.0
        xinqd(i)    = 1.0
        xinprecd(i) = 1.0
        xinwindd(i) = 1.0
        xincldd(i)  = 0.0
        xintrngd(i) = 1.0
        tcthis(i)   = 50.0
        twthis(i)   = -50.0
        gdd0this(i) = 0.0
        gdd5this(i) = 0.0
        a10td(i)    = 288.0
      end do

      do i = 1, n_in
        xintopo(i) = xintopo_in(i)
        do m = 1, 12
          xint(i, m)    = xint_in(i, m)
          xinq(i, m)    = xinq_in(i, m)
          xinprec(i, m) = xinprec_in(i, m)
          xinwind(i, m) = xinwind_in(i, m)
          xincld(i, m)  = xincld_in(i, m)
          xinwet(i, m)  = xinwet_in(i, m)
          xintrng(i, m) = xintrng_in(i, m)
        end do
        xintd(i)    = xintd_in(i)
        xinqd(i)    = xinqd_in(i)
        xinprecd(i) = xinprecd_in(i)
        xinwindd(i) = xinwindd_in(i)
        xincldd(i)  = xincldd_in(i)
        xintrngd(i) = xintrngd_in(i)
      end do

      seed_unused = 1
      iyranom_loc = 9999
      iyrlast_loc = 9999
      nrun_loc    = 1

c jdaily=1 path
      call daily(2000, imonth_in, iday_in, seed_unused,
     >           1, iyranom_loc, iyrlast_loc, nrun_loc)

      do i = 1, n_in
        td_out(i)     = td(i)
        tmax_out(i)   = tmax(i)
        tmin_out(i)   = tmin(i)
        cloud_out(i)  = cloud(i)
        precip_out(i) = precip(i)
        qd_out(i)     = qd(i)
        ud_out(i)     = ud(i)
        psurf_out(i)  = psurf(i)
      end do

      return
      end
