! =============================================================================
! mod_thermo_funcs.f90  --  Thermodynamic functions for IBIS
!
! Replaces the statement functions and associated PARAMETER declarations in
! comsat.h.  Statement functions are converted to elemental or regular
! Fortran functions.
!
! Lowe (1977) polynomial coefficients (a-, b-, c-, d-series) are private
! module PARAMETERs, preserving the exact single-precision decimal values
! from the original code for bit-for-bit regression traceability.
!
! Functions provided:
!   esat (t)        -- saturation vapour pressure (N m-2); t in K
!   desat(t)        -- d(esat)/dt (N m-2 K-1)
!   qsat (e1, p1)   -- saturation specific humidity (kg kg-1)
!   dqsat(t,  q1)   -- d(qsat)/dt (K-1); q1 = qsat (kg kg-1)
!   hvapf(t, tair)  -- corrected latent heat of vaporisation (J kg-1)
!   hsubf(t, tair)  -- corrected latent heat of sublimation  (J kg-1)
!
! Reference: Lowe P.R., 1977, J. Appl. Meteor., 16, 101-103.
!
! Replaces: comsat.h (statement functions and associated PARAMETER blocks).
! =============================================================================
module mod_thermo_funcs
  use mod_precision, only: wp
  use mod_constants, only: hvap, hsub, cvap, ch2o, cice
  implicit none
  private

  public :: esat, desat, qsat, dqsat, hvapf, hsubf

  ! ---------------------------------------------------------------------------
  ! Lowe (1977) SVP polynomial over liquid water (asat-series, 0..100 C)
  ! and over ice (bsat-series, -60..0 C).
  ! Original values: real (single precision) in comsat.h.
  ! ---------------------------------------------------------------------------
  real(wp), parameter :: asat0 =  6.1078000_wp
  real(wp), parameter :: asat1 =  4.4365185e-1_wp
  real(wp), parameter :: asat2 =  1.4289458e-2_wp
  real(wp), parameter :: asat3 =  2.6506485e-4_wp
  real(wp), parameter :: asat4 =  3.0312404e-6_wp
  real(wp), parameter :: asat5 =  2.0340809e-8_wp
  real(wp), parameter :: asat6 =  6.1368209e-11_wp

  real(wp), parameter :: bsat0 =  6.1091780_wp
  real(wp), parameter :: bsat1 =  5.0346990e-1_wp
  real(wp), parameter :: bsat2 =  1.8860134e-2_wp
  real(wp), parameter :: bsat3 =  4.1762237e-4_wp
  real(wp), parameter :: bsat4 =  5.8247203e-6_wp
  real(wp), parameter :: bsat5 =  4.8388032e-8_wp
  real(wp), parameter :: bsat6 =  1.8388269e-10_wp

  ! d(SVP)/dT polynomial over liquid (csat-series) and ice (dsat-series).
  real(wp), parameter :: csat0 =  4.4381000e-1_wp
  real(wp), parameter :: csat1 =  2.8570026e-2_wp
  real(wp), parameter :: csat2 =  7.9380540e-4_wp
  real(wp), parameter :: csat3 =  1.2152151e-5_wp
  real(wp), parameter :: csat4 =  1.0365614e-7_wp
  real(wp), parameter :: csat5 =  3.5324218e-10_wp
  real(wp), parameter :: csat6 = -7.0902448e-13_wp

  real(wp), parameter :: dsat0 =  5.0303052e-1_wp
  real(wp), parameter :: dsat1 =  3.7732550e-2_wp
  real(wp), parameter :: dsat2 =  1.2679954e-3_wp
  real(wp), parameter :: dsat3 =  2.4775631e-5_wp
  real(wp), parameter :: dsat4 =  3.0056931e-7_wp
  real(wp), parameter :: dsat5 =  2.1585425e-9_wp
  real(wp), parameter :: dsat6 =  7.1310977e-12_wp

contains

  ! ---------------------------------------------------------------------------
  ! Private helper: temperature offset clamped to the liquid-phase range.
  ! Returns t - 273.16, clamped to [0.0, 100.0].
  ! ---------------------------------------------------------------------------
  elemental pure function tsatl(t) result(r)
    real(wp), intent(in) :: t
    real(wp) :: r
    r = min(100.0_wp, max(t - 273.16_wp, 0.0_wp))
  end function tsatl

  ! ---------------------------------------------------------------------------
  ! Private helper: temperature offset clamped to the ice-phase range.
  ! Returns t - 273.16, clamped to [-60.0, 0.0].
  ! ---------------------------------------------------------------------------
  elemental pure function tsati(t) result(r)
    real(wp), intent(in) :: t
    real(wp) :: r
    r = max(-60.0_wp, min(t - 273.16_wp, 0.0_wp))
  end function tsati

  ! ---------------------------------------------------------------------------
  !> Saturation vapour pressure (N m-2) as a function of temperature (K).
  !!
  !! Uses Lowe (1977) polynomials.  For t >= 273.16 K, the liquid-water
  !! branch is active (asat coefficients); for t < 273.16 K the ice branch
  !! is active (bsat coefficients).  Both polynomial ranges are clamped by
  !! tsatl/tsati so only the active branch contributes non-zero terms.
  !!
  !! Multiply-by-100 converts from millibars (Lowe's output unit) to N m-2.
  ! ---------------------------------------------------------------------------
  elemental pure function esat(t) result(es)
    real(wp), intent(in) :: t
    real(wp) :: es, tl, ti, a0
    tl = tsatl(t)
    ti = tsati(t)
    a0 = merge(asat0, bsat0, t >= 273.16_wp)
    es = 100.0_wp * ( &
         a0 &
         + tl*(asat1 + tl*(asat2 + tl*(asat3 &
         + tl*(asat4 + tl*(asat5 + tl*asat6))))) &
         + ti*(bsat1 + ti*(bsat2 + ti*(bsat3 &
         + ti*(bsat4 + ti*(bsat5 + ti*bsat6))))) )
  end function esat

  ! ---------------------------------------------------------------------------
  !> Derivative of saturation vapour pressure w.r.t. temperature (N m-2 K-1).
  !!
  !! Uses Lowe (1977) d(SVP)/dT polynomials (c- and d-series).
  ! ---------------------------------------------------------------------------
  elemental pure function desat(t) result(des)
    real(wp), intent(in) :: t
    real(wp) :: des, tl, ti, c0
    tl = tsatl(t)
    ti = tsati(t)
    c0 = merge(csat0, dsat0, t >= 273.16_wp)
    des = 100.0_wp * ( &
          c0 &
          + tl*(csat1 + tl*(csat2 + tl*(csat3 &
          + tl*(csat4 + tl*(csat5 + tl*csat6))))) &
          + ti*(dsat1 + ti*(dsat2 + ti*(dsat3 &
          + ti*(dsat4 + ti*(dsat5 + ti*dsat6))))) )
  end function desat

  ! ---------------------------------------------------------------------------
  !> Saturation specific humidity (kg kg-1).
  !!
  !! e1 = saturation vapour pressure (N m-2).
  !! p1 = ambient pressure (N m-2).
  !! Clamped to avoid spurious values when SVP approaches ambient pressure.
  ! ---------------------------------------------------------------------------
  elemental pure function qsat(e1, p1) result(qs)
    real(wp), intent(in) :: e1, p1
    real(wp) :: qs
    qs = 0.622_wp * e1 / max(p1 - (1.0_wp - 0.622_wp)*e1, 0.622_wp*e1)
  end function qsat

  ! ---------------------------------------------------------------------------
  !> Derivative of saturation specific humidity w.r.t. temperature (K-1).
  !!
  !! t  = temperature (K).
  !! q1 = saturation specific humidity (kg kg-1).
  ! ---------------------------------------------------------------------------
  elemental pure function dqsat(t, q1) result(dqs)
    real(wp), intent(in) :: t, q1
    real(wp) :: dqs
    dqs = desat(t) * q1 * (1.0_wp + q1*(1.0_wp/0.622_wp - 1.0_wp)) / esat(t)
  end function dqsat

  ! ---------------------------------------------------------------------------
  !> Temperature-corrected latent heat of vaporisation (J kg-1).
  !!
  !! Corrects for the concept that the liquid-vapour phase change occurs at
  !! 273.16 K, with heat required to bring water and vapour to/from that
  !! reference temperature.  Matches the F77 statement function hvapf in
  !! comsat.h.
  !!
  !! t    = liquid water temperature (K).
  !! tair = temperature of the delivered/received vapour (K).
  !!
  !! Requires constants_init() to have been called before use.
  ! ---------------------------------------------------------------------------
  function hvapf(t, tair) result(h)
    real(wp), intent(in) :: t, tair
    real(wp) :: h
    h = hvap + cvap*(tair - 273.16_wp) - ch2o*(t - 273.16_wp)
  end function hvapf

  ! ---------------------------------------------------------------------------
  !> Temperature-corrected latent heat of sublimation (J kg-1).
  !!
  !! As hvapf but for the ice-vapour phase change.
  !!
  !! t    = ice temperature (K).
  !! tair = temperature of the delivered/received vapour (K).
  !!
  !! Requires constants_init() to have been called before use.
  ! ---------------------------------------------------------------------------
  function hsubf(t, tair) result(h)
    real(wp), intent(in) :: t, tair
    real(wp) :: h
    h = hsub + cvap*(tair - 273.16_wp) - cice*(t - 273.16_wp)
  end function hsubf

end module mod_thermo_funcs
