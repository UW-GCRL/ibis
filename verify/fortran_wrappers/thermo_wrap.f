c =====================================================================
c thermo_wrap.f
c
c F77 wrappers exposing the statement functions from comsat.h
c (esat, desat, qsat, dqsat, hvapf, hsubf) as callable subroutines
c so they can be reached from Python via f2py.
c
c cvmgt is defined inline so utilities.f does not need to be linked in.
c
c Argument arrays at the Python boundary are declared `real*8` because
c f2py parses the source and doesn't honor `-fdefault-real-8`. The plain
c `real` declarations inside comsat.h still become double precision under
c that compile flag, so the internal computation is float64 throughout.
c
c Naming: array args are *_in / *_out. We avoid the names t/tair/p1/e1/q1
c because comsat.h declares those as scalars used by the statement
c function definitions.
c =====================================================================

c ---------------------------------------------------------------------
c cvmgt -- conditional merge (CRAY-style). Used by esat/desat in comsat.h.
c Defined as plain `real` so its return type tracks -fdefault-real-8
c (must match the type of the comsat.h statement-function callers).
c ---------------------------------------------------------------------
      real function cvmgt (x, y, l)
      real x, y
      logical l
      if (l) then
        cvmgt = x
      else
        cvmgt = y
      endif
      return
      end


c ---------------------------------------------------------------------
c wrap_esat -- saturation vapor pressure (Pa) given temperature (K).
c ---------------------------------------------------------------------
      subroutine wrap_esat(n, t_in, e_out)
cf2py intent(hide), depend(t_in) :: n=len(t_in)
cf2py intent(in)  t_in
cf2py intent(out) e_out
      integer n
      real*8 t_in(n), e_out(n)
      integer i
c     comsat.h refers to hvap/cvap/ch2o/hsub/cice for hvapf/hsubf;
c     declare them locally so the include compiles. esat does not use them.
      real hvap, cvap, ch2o, hsub, cice
      include 'comsat.h'
      do i = 1, n
        e_out(i) = esat(t_in(i))
      end do
      return
      end


c ---------------------------------------------------------------------
c wrap_desat -- d(esat)/dT (Pa/K) given temperature (K).
c ---------------------------------------------------------------------
      subroutine wrap_desat(n, t_in, d_out)
cf2py intent(hide), depend(t_in) :: n=len(t_in)
cf2py intent(in)  t_in
cf2py intent(out) d_out
      integer n
      real*8 t_in(n), d_out(n)
      integer i
      real hvap, cvap, ch2o, hsub, cice
      include 'comsat.h'
      do i = 1, n
        d_out(i) = desat(t_in(i))
      end do
      return
      end


c ---------------------------------------------------------------------
c wrap_qsat -- saturation specific humidity (kg/kg) given esat (Pa) and p (Pa).
c ---------------------------------------------------------------------
      subroutine wrap_qsat(n, e_in, p_in, q_out)
cf2py intent(hide), depend(e_in) :: n=len(e_in)
cf2py intent(in)  e_in
cf2py intent(in)  p_in
cf2py intent(out) q_out
      integer n
      real*8 e_in(n), p_in(n), q_out(n)
      integer i
      real hvap, cvap, ch2o, hsub, cice
      include 'comsat.h'
      do i = 1, n
        q_out(i) = qsat(e_in(i), p_in(i))
      end do
      return
      end


c ---------------------------------------------------------------------
c wrap_dqsat -- d(qsat)/dT (K-1) given temperature (K) and q1 (kg/kg).
c ---------------------------------------------------------------------
      subroutine wrap_dqsat(n, t_in, q1_in, d_out)
cf2py intent(hide), depend(t_in) :: n=len(t_in)
cf2py intent(in)  t_in
cf2py intent(in)  q1_in
cf2py intent(out) d_out
      integer n
      real*8 t_in(n), q1_in(n), d_out(n)
      integer i
      real hvap, cvap, ch2o, hsub, cice
      include 'comsat.h'
      do i = 1, n
        d_out(i) = dqsat(t_in(i), q1_in(i))
      end do
      return
      end


c ---------------------------------------------------------------------
c wrap_hvapf -- T-adjusted latent heat of vaporization (J/kg).
c   hvapf(t, tair) = hvap + cvap*(tair-273.16) - ch2o*(t-273.16)
c Physical constants passed in as args (originally from commac.h).
c ---------------------------------------------------------------------
      subroutine wrap_hvapf(n, t_in, ta_in, hvap_in, cvap_in, ch2o_in,
     >                     h_out)
cf2py intent(hide), depend(t_in) :: n=len(t_in)
cf2py intent(in)  t_in
cf2py intent(in)  ta_in
cf2py intent(in)  hvap_in
cf2py intent(in)  cvap_in
cf2py intent(in)  ch2o_in
cf2py intent(out) h_out
      integer n
      real*8 t_in(n), ta_in(n), h_out(n)
      real*8 hvap_in, cvap_in, ch2o_in
      integer i
      real hvap, cvap, ch2o, hsub, cice
      include 'comsat.h'
      hvap = hvap_in
      cvap = cvap_in
      ch2o = ch2o_in
      do i = 1, n
        h_out(i) = hvapf(t_in(i), ta_in(i))
      end do
      return
      end


c ---------------------------------------------------------------------
c wrap_hsubf -- T-adjusted latent heat of sublimation (J/kg).
c   hsubf(t, tair) = hsub + cvap*(tair-273.16) - cice*(t-273.16)
c ---------------------------------------------------------------------
      subroutine wrap_hsubf(n, t_in, ta_in, hsub_in, cvap_in, cice_in,
     >                     h_out)
cf2py intent(hide), depend(t_in) :: n=len(t_in)
cf2py intent(in)  t_in
cf2py intent(in)  ta_in
cf2py intent(in)  hsub_in
cf2py intent(in)  cvap_in
cf2py intent(in)  cice_in
cf2py intent(out) h_out
      integer n
      real*8 t_in(n), ta_in(n), h_out(n)
      real*8 hsub_in, cvap_in, cice_in
      integer i
      real hvap, cvap, ch2o, hsub, cice
      include 'comsat.h'
      hsub = hsub_in
      cvap = cvap_in
      cice = cice_in
      do i = 1, n
        h_out(i) = hsubf(t_in(i), ta_in(i))
      end do
      return
      end
