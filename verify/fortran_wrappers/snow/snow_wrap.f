c =====================================================================
c snow_wrap.f
c
c F77 wrappers exposing snowheat, vadapt, snow_melt_block,
c snow_coverage_block from snow.f for f2py.
c
c Common-block deps (compar.h, comsoi.h, comsno.h, com1d.h).  We
c initialize ALL npoi entries to safe baseline before overwriting the
c first n_in to keep solvers and reductions from touching garbage.
c =====================================================================


c ---------------------------------------------------------------------
c wrap_snowheat -- expose snowheat(tlsub, fhtop, sflo, xl, chl).
c ---------------------------------------------------------------------
      subroutine wrap_snowheat(
     >    n_in,
     >    tsno_in, tlsub_in, fi_in, hsno_in,
     >    xl_in,
     >    hsoi_1_in, consoi_1_in, tsoi_1_in,
     >    fhtop_in,
     >    chl_in, rhos_in, consno_in,
     >    cice_in, dtime_in,
     >    tsno_out, tlsub_out, sflo_out)
cf2py intent(hide), depend(fi_in) :: n_in=len(fi_in)
cf2py intent(in)  tsno_in
cf2py intent(in)  tlsub_in
cf2py intent(in)  fi_in
cf2py intent(in)  hsno_in
cf2py intent(in)  xl_in
cf2py intent(in)  hsoi_1_in
cf2py intent(in)  consoi_1_in
cf2py intent(in)  tsoi_1_in
cf2py intent(in)  fhtop_in
cf2py intent(in)  chl_in
cf2py intent(in)  rhos_in
cf2py intent(in)  consno_in
cf2py intent(in)  cice_in
cf2py intent(in)  dtime_in
cf2py intent(out) tsno_out
cf2py intent(out) tlsub_out
cf2py intent(out) sflo_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsno.h'
      include 'comsoi.h'

      integer n_in
      real*8 tsno_in(n_in, 3), tlsub_in(n_in), fi_in(n_in)
      real*8 hsno_in(n_in, 3)
      real*8 xl_in(n_in)
      real*8 hsoi_1_in, consoi_1_in(n_in), tsoi_1_in(n_in)
      real*8 fhtop_in(n_in)
      real*8 chl_in, rhos_in, consno_in
      real*8 cice_in, dtime_in
      real*8 tsno_out(n_in, 3), tlsub_out(n_in), sflo_out(n_in, 5)

      integer i, k
      real tlsub(npoi)
      real fhtop(npoi)
      real sflo(npoi, nsnolay+2)
      real xl(npoi)

c     Set scalars (compar.h compar1, comsno1)
      cice  = cice_in
      dtime = dtime_in
      rhos   = rhos_in
      consno = consno_in

c     hsoi(1) used by snowheat for the snow->soil interface
      hsoi(1) = hsoi_1_in
      do k = 2, 7
        hsoi(k) = 0.0
      end do

c     Initialize ALL npoi entries to safe baseline
      do i = 1, npoi
        fi(i)    = 0.0
        tlsub(i) = 273.16
        fhtop(i) = 0.0
        xl(i)    = 0.0
        consoi(i, 1) = 1.0
        tsoi(i, 1)   = 280.0
        do k = 1, nsnolay
          tsno(i, k) = 268.0
          hsno(i, k) = 0.0
        end do
        do k = 1, nsnolay+2
          sflo(i, k) = 0.0
        end do
      end do

c     Overwrite first n_in
      do i = 1, n_in
        fi(i)    = fi_in(i)
        tlsub(i) = tlsub_in(i)
        fhtop(i) = fhtop_in(i)
        xl(i)    = xl_in(i)
        consoi(i, 1) = consoi_1_in(i)
        tsoi(i, 1)   = tsoi_1_in(i)
        do k = 1, 3
          tsno(i, k) = tsno_in(i, k)
          hsno(i, k) = hsno_in(i, k)
        end do
      end do

      call snowheat(tlsub, fhtop, sflo, xl, chl_in)

      do i = 1, n_in
        tlsub_out(i) = tlsub(i)
        do k = 1, 3
          tsno_out(i, k) = tsno(i, k)
        end do
        do k = 1, 5
          sflo_out(i, k) = sflo(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_vadapt -- expose vadapt(hcur, tcur, htop, indp, np, nlay).
c Takes a has_snow mask and builds indp/np from it.
c ---------------------------------------------------------------------
      subroutine wrap_vadapt(
     >    n_in,
     >    hcur_in, tcur_in,
     >    htop_in, has_snow_in,
     >    epsilon_in,
     >    hcur_out, tcur_out)
cf2py intent(hide), depend(has_snow_in) :: n_in=len(has_snow_in)
cf2py intent(in)  hcur_in
cf2py intent(in)  tcur_in
cf2py intent(in)  htop_in
cf2py intent(in)  has_snow_in
cf2py intent(in)  epsilon_in
cf2py intent(out) hcur_out
cf2py intent(out) tcur_out

      include 'implicit.h'
      include 'compar.h'

      integer n_in
      real*8 hcur_in(n_in, 3), tcur_in(n_in, 3)
      real*8 htop_in
      real*8 has_snow_in(n_in)
      real*8 epsilon_in
      real*8 hcur_out(n_in, 3), tcur_out(n_in, 3)

      integer i, k, np
      integer indp(npoi)
      real hcur(npoi, nsnolay), tcur(npoi, nsnolay)

      epsilon = epsilon_in

c     Init all to safe baseline (matches JAX behaviour for has_snow=False)
      do i = 1, npoi
        do k = 1, nsnolay
          hcur(i, k) = 0.0
          tcur(i, k) = 273.16
        end do
        indp(i) = 0
      end do

c     Overwrite first n_in with user data
      do i = 1, n_in
        do k = 1, 3
          hcur(i, k) = hcur_in(i, k)
          tcur(i, k) = tcur_in(i, k)
        end do
      end do

c     Build indp / np from has_snow mask
      np = 0
      do i = 1, n_in
        if (has_snow_in(i) .gt. 0.5) then
          np = np + 1
          indp(np) = i
        end if
      end do

      call vadapt(hcur, tcur, htop_in, indp, np, nsnolay)

      do i = 1, n_in
        do k = 1, 3
          hcur_out(i, k) = hcur(i, k)
          tcur_out(i, k) = tcur(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_snow_melt -- expose snow_melt_block.
c ---------------------------------------------------------------------
      subroutine wrap_snow_melt(
     >    n_in,
     >    hsno_in, tsno_in, fi_in,
     >    rhos_in, cice_in, hfus_in, tmelt_in,
     >    hsno_out, tsno_out, zmelt_out, zheat_out)
cf2py intent(hide), depend(fi_in) :: n_in=len(fi_in)
cf2py intent(in)  hsno_in
cf2py intent(in)  tsno_in
cf2py intent(in)  fi_in
cf2py intent(in)  rhos_in
cf2py intent(in)  cice_in
cf2py intent(in)  hfus_in
cf2py intent(in)  tmelt_in
cf2py intent(out) hsno_out
cf2py intent(out) tsno_out
cf2py intent(out) zmelt_out
cf2py intent(out) zheat_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsno.h'

      integer n_in
      real*8 hsno_in(n_in, 3), tsno_in(n_in, 3), fi_in(n_in)
      real*8 rhos_in, cice_in, hfus_in, tmelt_in
      real*8 hsno_out(n_in, 3), tsno_out(n_in, 3)
      real*8 zmelt_out(n_in), zheat_out(n_in)

      integer i, k
      real zmelt(npoi), zheat(npoi)

      cice  = cice_in
      hfus  = hfus_in
      tmelt = tmelt_in
      rhos  = rhos_in

      do i = 1, npoi
        fi(i)    = 0.0
        zmelt(i) = 0.0
        zheat(i) = 0.0
        do k = 1, nsnolay
          hsno(i, k) = 0.0
          tsno(i, k) = 273.16
        end do
      end do
      do i = 1, n_in
        fi(i) = fi_in(i)
        do k = 1, 3
          hsno(i, k) = hsno_in(i, k)
          tsno(i, k) = tsno_in(i, k)
        end do
      end do

      call snow_melt_block(zmelt, zheat)

      do i = 1, n_in
        zmelt_out(i) = zmelt(i)
        zheat_out(i) = zheat(i)
        do k = 1, 3
          hsno_out(i, k) = hsno(i, k)
          tsno_out(i, k) = tsno(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_snow_coverage -- expose snow_coverage_block.
c ---------------------------------------------------------------------
      subroutine wrap_snow_coverage(
     >    n_in,
     >    hsno_in, fi_in,
     >    hsnomin_in, hsnoruf_in, fimax_in,
     >    epsilon_in,
     >    fi_out, hsno_out)
cf2py intent(hide), depend(fi_in) :: n_in=len(fi_in)
cf2py intent(in)  hsno_in
cf2py intent(in)  fi_in
cf2py intent(in)  hsnomin_in
cf2py intent(in)  hsnoruf_in
cf2py intent(in)  fimax_in
cf2py intent(in)  epsilon_in
cf2py intent(out) fi_out
cf2py intent(out) hsno_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsno.h'

      integer n_in
      real*8 hsno_in(n_in, 3), fi_in(n_in)
      real*8 hsnomin_in, hsnoruf_in(n_in), fimax_in
      real*8 epsilon_in
      real*8 fi_out(n_in), hsno_out(n_in, 3)

      integer i, k
      real hsnoruf_local(npoi)

      epsilon = epsilon_in
      hsnomin = hsnomin_in
      fimax   = fimax_in

c     Safe baseline -- snowless cells get ht=0; algorithm yields
c     finew=0, then x1=fi/eps, but fi=0 too so the multiply gives 0.
      do i = 1, npoi
        fi(i)            = 0.0
        hsnoruf_local(i) = hsnomin + 0.1
        do k = 1, nsnolay
          hsno(i, k) = 0.0
        end do
      end do
      do i = 1, n_in
        fi(i)            = fi_in(i)
        hsnoruf_local(i) = hsnoruf_in(i)
        do k = 1, 3
          hsno(i, k) = hsno_in(i, k)
        end do
      end do

      call snow_coverage_block(hsnoruf_local)

      do i = 1, n_in
        fi_out(i) = fi(i)
        do k = 1, 3
          hsno_out(i, k) = hsno(i, k)
        end do
      end do

      return
      end
