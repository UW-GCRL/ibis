! =============================================================================
! test_phase1.f90  --  Unit tests for Phase 1 foundation modules
!
! Tests: mod_precision, mod_dimensions, mod_constants
!
! Run:  ./tests/test_phase1
! Exit: 0 = all PASS, non-zero = at least one FAIL
!
! Checks:
!   (1)  wp, sp, dp are valid Fortran kinds (> 0)
!   (2)  Dimension parameters have their expected values
!   (3)  constants_init() assigns correct F77 values
!   (4)  Derived quantities (hsub = hvap+hfus, cappa = rair/cair) are
!        consistent to round-off
!   (5)  vzero is all zero after constants_init()
! =============================================================================
program test_phase1
  use mod_precision,  only: wp, sp, dp
  use mod_dimensions, only: nlon, nlat, npoi, nband, nsoilay, nsnolay, npft, &
                             xres, yres, pi
  ! Rename 'epsilon' on import to avoid shadowing the Fortran intrinsic.
  use mod_constants,  only: ibis_eps => epsilon, &
                             stef, vonk, grav, tmelt, &
                             hfus, hvap, hsub, ch2o, cice, cair, cvap, &
                             rair, rvap, cappa, rhow, vzero, &
                             constants_init
  implicit none

  integer :: nfail = 0
  ! Relative tolerance for constant comparisons.
  ! tight enough to catch wrong values, but above binary round-off for
  ! decimal literals like 9.80616.
  real(wp), parameter :: tol = 1.0e-12_wp

  write(*,'(a)') '=== Phase 1 unit tests ==='
  write(*,*)

  ! -------------------------------------------------------------------------
  ! 1. Kind validity
  ! -------------------------------------------------------------------------
  call section('mod_precision: kind parameters')

  call check_int('sp > 0',   sp,      gt=0)
  call check_int('dp > 0',   dp,      gt=0)
  call check_int('wp > 0',   wp,      gt=0)
  call check_int('dp >= sp', dp - sp, ge=0)

#ifdef IBIS_SP
  call check_int('wp == sp (IBIS_SP build)', wp, eq=sp)
  write(*,'(a)') '  [single-precision regression build]'
#else
  call check_int('wp == dp (default build)', wp, eq=dp)
  write(*,'(a)') '  [double-precision production build]'
#endif

  ! -------------------------------------------------------------------------
  ! 2. Dimension parameters
  ! -------------------------------------------------------------------------
  call section('mod_dimensions: compile-time parameters')

  call check_int('nlon',    nlon,    eq=720)
  call check_int('nlat',    nlat,    eq=360)
  call check_int('npoi',    npoi,    eq=1718)
  call check_int('nband',   nband,   eq=2)
  call check_int('nsoilay', nsoilay, eq=6)
  call check_int('nsnolay', nsnolay, eq=3)
  call check_int('npft',    npft,    eq=12)

  call check_real('xres', xres, 0.50_wp,     tol)
  call check_real('yres', yres, 0.50_wp,     tol)
  call check_real('pi',   pi,   3.1415927_wp, tol)

  ! -------------------------------------------------------------------------
  ! 3. constants_init: assigned values
  ! -------------------------------------------------------------------------
  call section('mod_constants: constants_init()')
  call constants_init()

  call check_real('stef',    stef,    5.67e-8_wp,   tol)
  call check_real('vonk',    vonk,    0.4_wp,       tol)
  call check_real('grav',    grav,    9.80616_wp,   tol)
  call check_real('tmelt',   tmelt,   273.16_wp,    tol)
  call check_real('hvap',    hvap,    2.5104e6_wp,  tol)
  call check_real('hfus',    hfus,    0.3336e6_wp,  tol)
  call check_real('ch2o',    ch2o,    4.218e3_wp,   tol)
  call check_real('cice',    cice,    2.106e3_wp,   tol)
  call check_real('cair',    cair,    1.00464e3_wp, tol)
  call check_real('cvap',    cvap,    1.81e3_wp,    tol)
  call check_real('rair',    rair,    287.04_wp,    tol)
  call check_real('rvap',    rvap,    461.0_wp,     tol)
  call check_real('rhow',    rhow,    1.0e3_wp,     tol)
  call check_real('ibis_eps', ibis_eps, 1.0e-7_wp, tol)

  ! -------------------------------------------------------------------------
  ! 4. Derived quantities
  ! -------------------------------------------------------------------------
  call section('mod_constants: derived quantities')

  call check_real('hsub = hvap + hfus',     hsub,  hvap + hfus, tol)
  call check_real('cappa = rair/cair',      cappa, rair / cair, tol)
  call check_real('hsub approx 2.8440e6',   hsub,  2.8440e6_wp, 1.0e3_wp)

  ! -------------------------------------------------------------------------
  ! 5. vzero is all zero (use sum rather than element-wise /= to avoid
  !    real-equality warnings from -Wcompare-reals)
  ! -------------------------------------------------------------------------
  call section('mod_constants: vzero')
  if (sum(abs(vzero)) < tiny(1.0_wp)) then
    call pass('vzero(1:npoi) is all zero')
  else
    call fail_test('vzero has non-zero element(s)')
  end if

  ! -------------------------------------------------------------------------
  ! Summary
  ! -------------------------------------------------------------------------
  write(*,*)
  if (nfail == 0) then
    write(*,'(a)') 'Result: ALL TESTS PASSED'
    stop 0
  else
    write(*,'(i0,a)') nfail, ' test(s) FAILED'
    error stop 'Phase 1 tests failed'
  end if

contains

  subroutine section(name)
    character(len=*), intent(in) :: name
    write(*,*)
    write(*,'(a,a)') '--- ', name
  end subroutine section

  subroutine pass(msg)
    character(len=*), intent(in) :: msg
    write(*,'(a,a)') '  PASS: ', msg
  end subroutine pass

  subroutine fail_test(msg)
    character(len=*), intent(in) :: msg
    write(*,'(a,a)') '  FAIL: ', msg
    nfail = nfail + 1
  end subroutine fail_test

  subroutine check_int(label, got, eq, gt, ge)
    character(len=*),  intent(in)           :: label
    integer,           intent(in)           :: got
    integer, optional, intent(in)           :: eq, gt, ge
    if (present(eq)) then
      if (got == eq) then
        call pass(label // ' == expected')
      else
        write(*,'(4a,i0,a,i0)') '  FAIL: ', label, &
          ' got=', got, ' expected=', eq
        nfail = nfail + 1
      end if
    else if (present(gt)) then
      if (got > gt) then
        call pass(label)
      else
        write(*,'(3a,i0)') '  FAIL: ', label, ' got=', got
        nfail = nfail + 1
      end if
    else if (present(ge)) then
      if (got >= ge) then
        call pass(label)
      else
        write(*,'(3a,i0)') '  FAIL: ', label, ' got=', got
        nfail = nfail + 1
      end if
    end if
  end subroutine check_int

  ! Relative-or-absolute tolerance check.
  ! Passes if |got - expected| <= tol_in  OR
  !           |got - expected| / |expected| <= tol_in  (when expected /= 0)
  subroutine check_real(label, got, expected, tol_in)
    character(len=*), intent(in) :: label
    real(wp),         intent(in) :: got, expected, tol_in
    real(wp) :: absdiff, reldiff
    absdiff = abs(got - expected)
    if (absdiff <= tol_in) then
      call pass(label)
      return
    end if
    if (abs(expected) > tiny(expected)) then
      reldiff = absdiff / abs(expected)
      if (reldiff <= tol_in) then
        call pass(label)
        return
      end if
    end if
    write(*,'(2a,2(a,es22.15),a,es10.3)') &
      '  FAIL: ', label, &
      '  got=', got, '  expected=', expected, '  absdiff=', absdiff
    nfail = nfail + 1
  end subroutine check_real

end program test_phase1
