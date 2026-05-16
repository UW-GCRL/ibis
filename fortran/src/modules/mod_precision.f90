! =============================================================================
! mod_precision.f90  --  IBIS working-precision selector
!
! Defines the Fortran kind parameters used throughout IBIS.  Every real
! declaration in physics and I/O routines must use real(wp).
!
! Default build: double precision (wp = dp).
! Single-precision regression build: compile with -DIBIS_SP.
!   gfortran -DIBIS_SP ...
! This lets you run the modernised code against the original F77 (single-
! precision) reference output to verify bit-for-bit numerical identity
! before switching to the production double-precision build.
!
! Replaces: implicit.h (which only contained 'implicit none').
! Source traceability: compar.h originally used default real (= single).
! =============================================================================
module mod_precision
  implicit none
  private

  !> 32-bit real kind (matches original F77 default real).
  integer, parameter, public :: sp = selected_real_kind(6, 37)

  !> 64-bit real kind (production double-precision build).
  integer, parameter, public :: dp = selected_real_kind(15, 307)

#ifdef IBIS_SP
  !> Working precision: single -- for bit-for-bit regression vs F77 baseline.
  integer, parameter, public :: wp = sp
#else
  !> Working precision: double -- default production build.
  integer, parameter, public :: wp = dp
#endif

end module mod_precision
