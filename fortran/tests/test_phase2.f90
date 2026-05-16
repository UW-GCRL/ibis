! =============================================================================
! test_phase2.f90  --  Unit tests for Phase 2 state and function modules
!
! Tests: mod_bgc_params, mod_soil_texture, mod_age_class,
!        mod_diagnostics, mod_pft_params, mod_thermo_funcs
!
! Run:  ./tests/test_phase2
! Exit: 0 = all PASS, non-zero = at least one FAIL
!
! Checks:
!   (1)  mod_bgc_params: bgc_params_init() sets expected hardcoded values
!   (2)  mod_soil_texture: ndat==11, lookup arrays zero at startup
!   (3)  mod_age_class: compile-time parameters, zero initialisation
!   (4)  mod_diagnostics: compile-time parameters, zero initialisation
!   (5)  mod_pft_params: array sizes, zero initialisation
!   (6)  mod_thermo_funcs: esat/desat/qsat/dqsat/hvapf/hsubf correctness
! =============================================================================
program test_phase2
  use mod_precision,    only: wp
  use mod_bgc_params,   only: bgc_params_init, &
                               klm, ylm, lig_frac, cnleaf, fmax, rconst, &
                               cnr, effac, fbsom
  use mod_soil_texture, only: ndat, texdat, porosdat
  use mod_age_class,    only: maxage, ncohort, npftu, ac_size, &
                               f_Aveg, cbiow_ac, n_lcs, used
  use mod_diagnostics,  only: nvars, nfiles, ldiag, nfreq
  use mod_pft_params,   only: vmax_pft, tauleaf, TminL, plai_init
  use mod_dimensions,   only: npft
  use mod_thermo_funcs, only: esat, desat, qsat, dqsat, hvapf, hsubf
  use mod_constants,    only: constants_init, &
                               ibis_hvap => hvap, ibis_hsub => hsub
  implicit none

  integer :: nfail = 0

  write(*,'(a)') '=== Phase 2 unit tests ==='
  write(*,*)

  ! -------------------------------------------------------------------------
  ! 1. mod_bgc_params
  ! -------------------------------------------------------------------------
  call section('mod_bgc_params: bgc_params_init()')
  call bgc_params_init()

  ! Relative tolerance tight enough to catch wrong values
  call check_real('klm',      klm,      0.15_wp,   1.0e-12_wp)
  call check_real('ylm',      ylm,      0.4_wp,    1.0e-12_wp)
  call check_real('lig_frac', lig_frac, 0.50_wp,   1.0e-12_wp)
  call check_real('fbsom',    fbsom,    0.017_wp,  1.0e-12_wp)
  call check_real('effac',    effac,    0.40_wp,   1.0e-12_wp)
  call check_real('fmax',     fmax,     0.45_wp,   1.0e-12_wp)
  call check_real('rconst',   rconst,   1200.0_wp, 1.0e-12_wp)
  call check_real('cnleaf',   cnleaf,   40.0_wp,   1.0e-12_wp)
  call check_real('cnr(1)',   cnr(1),    8.0_wp,   1.0e-12_wp)  ! microbial biomass C:N

  ! -------------------------------------------------------------------------
  ! 2. mod_soil_texture
  ! -------------------------------------------------------------------------
  call section('mod_soil_texture: compile-time parameters and initialisation')

  call check_int('ndat', ndat, eq=11)

  if (abs(porosdat(1)) < tiny(1.0_wp)) then
    call pass('porosdat(1) is zero at startup')
  else
    call fail_test('porosdat(1) non-zero at startup')
  end if
  if (sum(abs(texdat)) < tiny(1.0_wp)) then
    call pass('texdat is all zero at startup')
  else
    call fail_test('texdat has non-zero element at startup')
  end if

  ! -------------------------------------------------------------------------
  ! 3. mod_age_class
  ! -------------------------------------------------------------------------
  call section('mod_age_class: compile-time parameters and initialisation')

  call check_int('maxage',  maxage,  eq=100)
  call check_int('ncohort', ncohort, eq=100)
  call check_int('npftu',   npftu,   eq=8)
  call check_int('ac_size', ac_size, eq=npftu*(ncohort+1))
  call check_int('n_lcs',   n_lcs,   eq=100)

  if (sum(abs(f_Aveg))  < tiny(1.0_wp)) then
    call pass('f_Aveg(0:ncohort) all zero')
  else
    call fail_test('f_Aveg has non-zero element at startup')
  end if
  if (sum(abs(cbiow_ac)) < tiny(1.0_wp)) then
    call pass('cbiow_ac all zero')
  else
    call fail_test('cbiow_ac has non-zero element at startup')
  end if
  if (sum(abs(used)) == 0) then
    call pass('used(n_lcs) all zero')
  else
    call fail_test('used has non-zero element at startup')
  end if

  ! -------------------------------------------------------------------------
  ! 4. mod_diagnostics
  ! -------------------------------------------------------------------------
  call section('mod_diagnostics: compile-time parameters and initialisation')

  call check_int('nvars',  nvars,  eq=119)
  call check_int('nfiles', nfiles, eq=10)

  if (sum(abs(ldiag)) == 0) then
    call pass('ldiag(nvars,nfiles) all zero')
  else
    call fail_test('ldiag has non-zero element at startup')
  end if
  if (sum(abs(nfreq)) == 0) then
    call pass('nfreq(nfiles) all zero')
  else
    call fail_test('nfreq has non-zero element at startup')
  end if

  ! -------------------------------------------------------------------------
  ! 5. mod_pft_params
  ! -------------------------------------------------------------------------
  call section('mod_pft_params: array sizes and initialisation')

  call check_int('size(vmax_pft)',    size(vmax_pft),    eq=npft)
  call check_int('size(tauleaf)',     size(tauleaf),     eq=npft)
  call check_int('size(TminL)',       size(TminL),       eq=npft)
  call check_int('size(plai_init,2)', size(plai_init,2), eq=15)

  if (sum(abs(vmax_pft)) < tiny(1.0_wp)) then
    call pass('vmax_pft all zero before readpars')
  else
    call fail_test('vmax_pft non-zero before readpars')
  end if

  ! -------------------------------------------------------------------------
  ! 6. mod_thermo_funcs: esat / desat / qsat / dqsat
  ! -------------------------------------------------------------------------
  call section('mod_thermo_funcs: esat / desat / qsat / dqsat')

  ! At t = 273.16 K: tl=0, ti=0 => esat = 100 * asat0 = 610.78 Pa exactly.
  call check_real('esat(273.16 K) == 610.78 Pa', &
                   esat(273.16_wp), 610.78_wp, 1.0e-10_wp)

  ! Ice branch (t < 273.16): positive and less than liquid-branch value.
  if (esat(250.0_wp) > 0.0_wp .and. esat(250.0_wp) < esat(273.16_wp)) then
    call pass('esat(250 K) ice branch: 0 < esat < esat(273.16)')
  else
    call fail_test('esat(250 K) outside expected ice-branch range')
  end if

  ! desat positive over both liquid and ice ranges.
  if (desat(250.0_wp) > 0.0_wp) then
    call pass('desat(250 K) > 0')
  else
    call fail_test('desat(250 K) <= 0')
  end if
  if (desat(300.0_wp) > 0.0_wp) then
    call pass('desat(300 K) > 0')
  else
    call fail_test('desat(300 K) <= 0')
  end if

  ! qsat at 273.16 K and standard pressure ≈ 3.8e-3 kg/kg.
  call check_real('qsat(esat(273.16), 101325) ~ 3.8e-3', &
                   qsat(esat(273.16_wp), 101325.0_wp), 3.8e-3_wp, 1.0e-4_wp)

  ! qsat must lie in (0, 1) for any reasonable inputs.
  if (qsat(esat(300.0_wp), 101325.0_wp) > 0.0_wp .and. &
      qsat(esat(300.0_wp), 101325.0_wp) < 1.0_wp) then
    call pass('qsat(300 K, 101325 Pa) in (0, 1)')
  else
    call fail_test('qsat(300 K, 101325 Pa) outside (0, 1)')
  end if

  ! dqsat > 0: specific humidity increases with temperature.
  if (dqsat(300.0_wp, qsat(esat(300.0_wp), 101325.0_wp)) > 0.0_wp) then
    call pass('dqsat(300 K) > 0')
  else
    call fail_test('dqsat(300 K) <= 0')
  end if

  ! -------------------------------------------------------------------------
  ! 7. mod_thermo_funcs: hvapf / hsubf (requires constants_init first)
  ! -------------------------------------------------------------------------
  call section('mod_thermo_funcs: hvapf / hsubf (after constants_init)')
  call constants_init()

  ! At the reference temperature both correction terms are exactly zero.
  call check_real('hvapf(273.16, 273.16) == hvap', &
                   hvapf(273.16_wp, 273.16_wp), ibis_hvap, 1.0e-12_wp)
  call check_real('hsubf(273.16, 273.16) == hsub', &
                   hsubf(273.16_wp, 273.16_wp), ibis_hsub, 1.0e-12_wp)

  ! Warmer vapour delivery requires more latent heat.
  if (hvapf(273.16_wp, 280.0_wp) > hvapf(273.16_wp, 273.16_wp)) then
    call pass('hvapf increases with tair (warmer vapour)')
  else
    call fail_test('hvapf does not increase with tair')
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
    error stop 'Phase 2 tests failed'
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

  subroutine check_int(label, got, eq)
    character(len=*), intent(in) :: label
    integer,          intent(in) :: got, eq
    if (got == eq) then
      call pass(label // ' == expected')
    else
      write(*,'(2a,i0,a,i0)') '  FAIL: ', label // ' got=', got, &
                                ' expected=', eq
      nfail = nfail + 1
    end if
  end subroutine check_int

  ! Relative-or-absolute tolerance check.
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

end program test_phase2
