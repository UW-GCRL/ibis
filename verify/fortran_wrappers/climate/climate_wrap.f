c =====================================================================
c climate_wrap.f
c
c F77 wrappers for climate.f: climanl, climanl2, existence.
c =====================================================================


c ---------------------------------------------------------------------
c wrap_climanl -- initial climate analysis (year 0).
c
c Inputs:
c   n_in            number of grid points (<= npoi)
c   xint_in         [n_in, 12] monthly mean temperature (C)
c   deltat_in       [n_in] temperature correction for tcmin
c   TminL_in, TminU_in, Twarm_in, GDD_in  [12] PFT bioclimatic limits
c
c Outputs:
c   tc_out, tw_out, tcmin_out, gdd0_out, gdd5_out [n_in]
c   exist_out       [n_in, 12] PFT existence flags
c ---------------------------------------------------------------------
      subroutine wrap_climanl(
     >    n_in,
     >    xint_in, deltat_in,
     >    TminL_in, TminU_in, Twarm_in, GDD_in,
     >    tc_out, tw_out, tcmin_out, gdd0_out, gdd5_out, exist_out)
cf2py intent(in)  n_in
cf2py intent(in)  xint_in, deltat_in
cf2py intent(in)  TminL_in, TminU_in, Twarm_in, GDD_in
cf2py intent(out) tc_out, tw_out, tcmin_out, gdd0_out, gdd5_out, exist_out

      include 'implicit.h'
      include 'compar.h'
      include 'combcs.h'
      include 'comveg.h'
      include 'compft.h'

      integer n_in
      real*8 xint_in(n_in, 12), deltat_in(n_in)
      real*8 TminL_in(12), TminU_in(12), Twarm_in(12), GDD_in(12)
      real*8 tc_out(n_in), tw_out(n_in), tcmin_out(n_in)
      real*8 gdd0_out(n_in), gdd5_out(n_in)
      real*8 exist_out(n_in, 12)

      integer i, k

c --- set ndaypm (climanl reads this) ---
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

c --- set PFT params (compft.h) ---
      do k = 1, 12
        TminL(k) = TminL_in(k)
        TminU(k) = TminU_in(k)
        Twarm(k) = Twarm_in(k)
        GDD(k)   = GDD_in(k)
      end do

c --- seed all npoi positions with safe defaults ---
      do i = 1, npoi
        deltat(i) = 0.0
        do k = 1, 12
          xint(i, k) = 20.0
        end do
        tc(i) = 20.0
        tw(i) = 20.0
        tcmin(i) = 20.0
        gdd0(i) = 0.0
        gdd5(i) = 0.0
        do k = 1, 12
          exist(i, k) = 0.0
        end do
      end do

c --- overwrite first n_in with user inputs ---
      do i = 1, n_in
        deltat(i) = deltat_in(i)
        do k = 1, 12
          xint(i, k) = xint_in(i, k)
        end do
      end do

      call climanl

      do i = 1, n_in
        tc_out(i) = tc(i)
        tw_out(i) = tw(i)
        tcmin_out(i) = tcmin(i)
        gdd0_out(i) = gdd0(i)
        gdd5_out(i) = gdd5(i)
        do k = 1, 12
          exist_out(i, k) = exist(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_climanl2 -- 30-year e-folding climate update.
c ---------------------------------------------------------------------
      subroutine wrap_climanl2(
     >    n_in,
     >    tc_in, tw_in, tcmin_in, gdd0_in, gdd5_in,
     >    tcthis_in, twthis_in, gdd0this_in, gdd5this_in,
     >    deltat_in,
     >    TminL_in, TminU_in, Twarm_in, GDD_in,
     >    tc_out, tw_out, tcmin_out, gdd0_out, gdd5_out, exist_out)
cf2py intent(in)  n_in
cf2py intent(in)  tc_in, tw_in, tcmin_in, gdd0_in, gdd5_in
cf2py intent(in)  tcthis_in, twthis_in, gdd0this_in, gdd5this_in
cf2py intent(in)  deltat_in
cf2py intent(in)  TminL_in, TminU_in, Twarm_in, GDD_in
cf2py intent(out) tc_out, tw_out, tcmin_out, gdd0_out, gdd5_out, exist_out

      include 'implicit.h'
      include 'compar.h'
      include 'combcs.h'
      include 'comveg.h'
      include 'compft.h'

      integer n_in
      real*8 tc_in(n_in), tw_in(n_in), tcmin_in(n_in)
      real*8 gdd0_in(n_in), gdd5_in(n_in)
      real*8 tcthis_in(n_in), twthis_in(n_in)
      real*8 gdd0this_in(n_in), gdd5this_in(n_in)
      real*8 deltat_in(n_in)
      real*8 TminL_in(12), TminU_in(12), Twarm_in(12), GDD_in(12)
      real*8 tc_out(n_in), tw_out(n_in), tcmin_out(n_in)
      real*8 gdd0_out(n_in), gdd5_out(n_in)
      real*8 exist_out(n_in, 12)

      integer i, k

      do k = 1, 12
        TminL(k) = TminL_in(k)
        TminU(k) = TminU_in(k)
        Twarm(k) = Twarm_in(k)
        GDD(k)   = GDD_in(k)
      end do

      do i = 1, npoi
        tc(i) = 0.0; tw(i) = 0.0; tcmin(i) = 0.0
        gdd0(i) = 0.0; gdd5(i) = 0.0
        tcthis(i) = 0.0; twthis(i) = 0.0
        gdd0this(i) = 0.0; gdd5this(i) = 0.0
        deltat(i) = 0.0
        do k = 1, 12
          exist(i, k) = 0.0
        end do
      end do

      do i = 1, n_in
        tc(i) = tc_in(i); tw(i) = tw_in(i); tcmin(i) = tcmin_in(i)
        gdd0(i) = gdd0_in(i); gdd5(i) = gdd5_in(i)
        tcthis(i) = tcthis_in(i); twthis(i) = twthis_in(i)
        gdd0this(i) = gdd0this_in(i); gdd5this(i) = gdd5this_in(i)
        deltat(i) = deltat_in(i)
      end do

      call climanl2

      do i = 1, n_in
        tc_out(i) = tc(i)
        tw_out(i) = tw(i)
        tcmin_out(i) = tcmin(i)
        gdd0_out(i) = gdd0(i)
        gdd5_out(i) = gdd5(i)
        do k = 1, 12
          exist_out(i, k) = exist(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_existence -- PFT existence based on bioclimatic thresholds.
c ---------------------------------------------------------------------
      subroutine wrap_existence(
     >    n_in,
     >    tcmin_in, tw_in, gdd0_in, gdd5_in,
     >    TminL_in, TminU_in, Twarm_in, GDD_in,
     >    exist_out)
cf2py intent(in)  n_in
cf2py intent(in)  tcmin_in, tw_in, gdd0_in, gdd5_in
cf2py intent(in)  TminL_in, TminU_in, Twarm_in, GDD_in
cf2py intent(out) exist_out

      include 'implicit.h'
      include 'compar.h'
      include 'comveg.h'
      include 'compft.h'

      integer n_in
      real*8 tcmin_in(n_in), tw_in(n_in)
      real*8 gdd0_in(n_in), gdd5_in(n_in)
      real*8 TminL_in(12), TminU_in(12), Twarm_in(12), GDD_in(12)
      real*8 exist_out(n_in, 12)

      integer i, k

      do k = 1, 12
        TminL(k) = TminL_in(k)
        TminU(k) = TminU_in(k)
        Twarm(k) = Twarm_in(k)
        GDD(k)   = GDD_in(k)
      end do

      do i = 1, npoi
        tcmin(i) = 0.0; tw(i) = 0.0
        gdd0(i) = 0.0;  gdd5(i) = 0.0
        do k = 1, 12
          exist(i, k) = 0.0
        end do
      end do

      do i = 1, n_in
        tcmin(i) = tcmin_in(i)
        tw(i)    = tw_in(i)
        gdd0(i)  = gdd0_in(i)
        gdd5(i)  = gdd5_in(i)
      end do

      call existence

      do i = 1, n_in
        do k = 1, 12
          exist_out(i, k) = exist(i, k)
        end do
      end do

      return
      end
