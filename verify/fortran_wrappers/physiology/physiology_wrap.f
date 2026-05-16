c =====================================================================
c physiology_wrap.f
c
c F77 wrappers exposing drystress (and later stomata) from physiology.f.
c =====================================================================


c ---------------------------------------------------------------------
c wrap_drystress
c
c drystress reads wsoi/wisoi/swilt/sfield/froot from common blocks
c (comsoi.h, comveg.h) and writes stressl/stressu/stresstl/stresstu.
c ---------------------------------------------------------------------
      subroutine wrap_drystress(
     >    n_in,
     >    wsoi_in, wisoi_in,
     >    swilt_in, sfield_in,
     >    froot_in,
     >    stresstl_out, stresstu_out,
     >    stressl_out, stressu_out)
cf2py intent(hide), depend(wsoi_in) :: n_in=size(wsoi_in,0)
cf2py intent(in)  wsoi_in
cf2py intent(in)  wisoi_in
cf2py intent(in)  swilt_in
cf2py intent(in)  sfield_in
cf2py intent(in)  froot_in
cf2py intent(out) stresstl_out
cf2py intent(out) stresstu_out
cf2py intent(out) stressl_out
cf2py intent(out) stressu_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsoi.h'
      include 'comveg.h'

      integer n_in
      real*8 wsoi_in(n_in, 6), wisoi_in(n_in, 6)
      real*8 swilt_in(n_in, 6), sfield_in(n_in, 6)
      real*8 froot_in(6, 2)
      real*8 stresstl_out(n_in), stresstu_out(n_in)
      real*8 stressl_out(n_in, 6), stressu_out(n_in, 6)

      integer i, k

c     Initialize all npoi entries to safe baseline.
      do i = 1, npoi
        do k = 1, nsoilay
          wsoi(i, k)   = 0.3
          wisoi(i, k)  = 0.0
          swilt(i, k)  = 0.15
          sfield(i, k) = 0.35
          stressl(i, k) = 0.0
          stressu(i, k) = 0.0
        end do
        stresstl(i) = 0.0
        stresstu(i) = 0.0
      end do
      do k = 1, nsoilay
        froot(k, 1) = 1.0 / nsoilay
        froot(k, 2) = 1.0 / nsoilay
      end do

c     Overwrite first n_in
      do i = 1, n_in
        do k = 1, 6
          wsoi(i, k)   = wsoi_in(i, k)
          wisoi(i, k)  = wisoi_in(i, k)
          swilt(i, k)  = swilt_in(i, k)
          sfield(i, k) = sfield_in(i, k)
        end do
      end do
      do k = 1, 6
        froot(k, 1) = froot_in(k, 1)
        froot(k, 2) = froot_in(k, 2)
      end do

      call drystress

      do i = 1, n_in
        stresstl_out(i) = stresstl(i)
        stresstu_out(i) = stresstu(i)
        do k = 1, 6
          stressl_out(i, k) = stressl(i, k)
          stressu_out(i, k) = stressu(i, k)
        end do
      end do

      return
      end
