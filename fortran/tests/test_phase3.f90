! =============================================================================
! test_phase3.f90  --  Unit tests for Phase 3 state modules
!
! Covers all 11 Phase 3 modules:
!   mod_snow_state, mod_hydrology, mod_carbon_pools,
!   mod_atm_state, mod_boundary_climate, mod_soil_state,
!   mod_canopy_fluxes, mod_work, mod_summary,
!   mod_canopy_params, mod_veg_dynamics
!
! Each test verifies:
!   - USE association compiles without error
!   - Default-initialised scalars are zero (or the expected constant value)
!   - Array shapes match the expected dimension parameters
! =============================================================================
program test_phase3
  use mod_precision,        only: wp
  use mod_dimensions,       only: npoi, npft, nsoilay, nsnolay, nband, nlon, nlat

  ! Phase 3 state modules
  use mod_snow_state,       only: fi, tsno, hsno, z0sno, rhos, consno, &
                                   hsnotop, hsnomin, fimin, fimax
  use mod_hydrology,        only: ginvap, gsuvap, gtrans, gtransu, gtransl, &
                                   grunof, gdrain, gadjust, wtot
  use mod_carbon_pools,     only: cbiol, cbior, cbiow, &
                                   clitll, clitlm, clitls, clitrl, csoipas, &
                                   csoislo, decompl, decomps, &
                                   falll, fallr, fallw, cdisturb
  use mod_atm_state,        only: coszen, fira, solad, solai, asurd, asuri, &
                                   ua, ta, qa, raina, rh, snowa, &
                                   psurf, cloud, td, tmax, tmin, qd, ud, &
                                   precip, precipday, precipdaysum, &
                                   xstore, iwet, iwetday, iwetdaysum, &
                                   co2conc, o2conc
  use mod_boundary_climate, only: xintopo, xinveg, deltat, lmask, &
                                   xint, xinq, xinprec, xinwind
  use mod_soil_state,       only: wpudmax, zwpmax, bperm, &
                                   wpud, wipud, z0soi, albsav, albsan, &
                                   stresstl, stresstu, heati, heatg, tg, ti
  use mod_canopy_fluxes,    only: termu, terml, scalcoefu, scalcoefl, &
                                   nsol, indsol, tfac
  use mod_work,             only: ocean_fill, ndim2, ndim3, ndim4, &
                                   lonindex, latindex, lonscale, latscale, work
  use mod_summary,          only: ndtimes, nmtimes, nytimes, &
                                   adrain, adsnow, adaet, amnpp
  use mod_canopy_params,    only: alaiml, alaimu, cgrass, woodnorm, &
                                   q12, q34, tl, ts, tu, &
                                   agcub, agcuc, ancub, totcondub, &
                                   ciub, gsub, &
                                   dleaf, dstem, orieh, oriev, &
                                   rhoveg, tauveg
  use mod_veg_dynamics,     only: agddl, agddu, fl, fu, sapfrac, &
                                   tc, totlail, totlaiu, totbiol, totbiou, &
                                   firefac, tneetot, tnpptot, &
                                   biomass, frac, plai, tnpp, tgpp, &
                                   lai, sai, ztop, ztopmx, &
                                   froot, exist, specla, aleaf, aroot, awood

  implicit none

  integer :: npass, nfail

  npass = 0
  nfail = 0

  write(*,'(a)') '=== Phase 3 module tests ==='
  write(*,*)

  ! ===========================================================================
  ! mod_snow_state
  ! ===========================================================================
  write(*,'(a)') '--- mod_snow_state ---'

  call check_real('z0sno init',    z0sno,            0.0_wp, npass, nfail)
  call check_real('rhos init',     rhos,             0.0_wp, npass, nfail)
  call check_real('consno init',   consno,           0.0_wp, npass, nfail)
  call check_real('hsnotop init',  hsnotop,          0.0_wp, npass, nfail)
  call check_real('fimin init',    fimin,            0.0_wp, npass, nfail)
  call check_real('fimax init',    fimax,            0.0_wp, npass, nfail)
  call check_real('fi(1) init',    fi(1),            0.0_wp, npass, nfail)
  call check_int ('fi size',       size(fi),         npoi,   npass, nfail)
  call check_int ('tsno dim1',     size(tsno, 1),    npoi,   npass, nfail)
  call check_int ('tsno dim2',     size(tsno, 2),    nsnolay,npass, nfail)
  call check_int ('hsno dim1',     size(hsno, 1),    npoi,   npass, nfail)
  call check_int ('hsno dim2',     size(hsno, 2),    nsnolay,npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_hydrology
  ! ===========================================================================
  write(*,'(a)') '--- mod_hydrology ---'

  call check_real('ginvap(1) init', ginvap(1), 0.0_wp, npass, nfail)
  call check_real('gdrain(1) init', gdrain(1), 0.0_wp, npass, nfail)
  call check_real('wtot(1) init',   wtot(1),   0.0_wp, npass, nfail)
  call check_int ('ginvap size',    size(ginvap), npoi, npass, nfail)
  call check_int ('gsuvap size',    size(gsuvap), npoi, npass, nfail)
  call check_int ('gtrans size',    size(gtrans), npoi, npass, nfail)
  call check_int ('grunof size',    size(grunof), npoi, npass, nfail)
  call check_int ('gadjust size',   size(gadjust),npoi, npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_carbon_pools
  ! ===========================================================================
  write(*,'(a)') '--- mod_carbon_pools ---'

  call check_real('cbiol(1,1) init', cbiol(1,1), 0.0_wp, npass, nfail)
  call check_real('clitll(1) init',  clitll(1),  0.0_wp, npass, nfail)
  call check_real('csoipas(1) init', csoipas(1), 0.0_wp, npass, nfail)
  call check_real('decompl(1) init', decompl(1), 0.0_wp, npass, nfail)
  call check_real('cdisturb(1) init',cdisturb(1),0.0_wp, npass, nfail)
  call check_int ('cbiol dim1',      size(cbiol,1), npoi, npass, nfail)
  call check_int ('cbiol dim2',      size(cbiol,2), npft, npass, nfail)
  call check_int ('cbior dim2',      size(cbior,2), npft, npass, nfail)
  call check_int ('cbiow dim2',      size(cbiow,2), npft, npass, nfail)
  call check_int ('clitll size',     size(clitll),  npoi, npass, nfail)
  call check_int ('falll size',      size(falll),   npoi, npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_atm_state
  ! ===========================================================================
  write(*,'(a)') '--- mod_atm_state ---'

  call check_real('coszen(1) init',  coszen(1), 0.0_wp, npass, nfail)
  call check_real('co2conc init',    co2conc,   0.0_wp, npass, nfail)
  call check_real('o2conc init',     o2conc,    0.0_wp, npass, nfail)
  call check_int ('coszen size',     size(coszen), npoi, npass, nfail)
  call check_int ('solad dim1',      size(solad,1), npoi, npass, nfail)
  call check_int ('solad dim2',      size(solad,2), nband,npass, nfail)
  call check_int ('solai dim2',      size(solai,2), nband,npass, nfail)
  call check_int ('asurd dim2',      size(asurd,2), nband,npass, nfail)
  call check_int ('ua size',         size(ua),   npoi, npass, nfail)
  call check_int ('precipday dim1',  size(precipday,1), npoi, npass, nfail)
  call check_int ('precipday dim2',  size(precipday,2), 31,   npass, nfail)
  call check_int ('xstore dim2',     size(xstore, 2), 3,    npass, nfail)
  call check_int ('iwet(1) init',    iwet(1), 0,    npass, nfail)
  call check_int ('iwetday dim2',    size(iwetday,2), 31,   npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_boundary_climate
  ! ===========================================================================
  write(*,'(a)') '--- mod_boundary_climate ---'

  call check_real('xintopo(1) init', xintopo(1), 0.0_wp, npass, nfail)
  call check_real('xinveg(1) init',  xinveg(1),  0.0_wp, npass, nfail)
  call check_real('xint(1,1) init',  xint(1,1),  0.0_wp, npass, nfail)
  call check_int ('lmask(1,1) init', lmask(1,1), 0,     npass, nfail)
  call check_int ('xintopo size',    size(xintopo), npoi, npass, nfail)
  call check_int ('lmask dim1',      size(lmask,1), nlon, npass, nfail)
  call check_int ('lmask dim2',      size(lmask,2), nlat, npass, nfail)
  call check_int ('xint dim2',       size(xint,2),  12,   npass, nfail)
  call check_int ('xinq dim2',       size(xinq,2),  12,   npass, nfail)
  call check_int ('xinprec dim2',    size(xinprec,2),12,  npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_soil_state
  ! ===========================================================================
  write(*,'(a)') '--- mod_soil_state ---'

  call check_real('wpudmax init',   wpudmax, 0.0_wp, npass, nfail)
  call check_real('zwpmax init',    zwpmax,  0.0_wp, npass, nfail)
  call check_real('bperm init',     bperm,   0.0_wp, npass, nfail)
  call check_real('wpud(1) init',   wpud(1), 0.0_wp, npass, nfail)
  call check_real('tg(1) init',     tg(1),   0.0_wp, npass, nfail)
  call check_real('ti(1) init',     ti(1),   0.0_wp, npass, nfail)
  call check_int ('wpud size',      size(wpud),  npoi, npass, nfail)
  call check_int ('z0soi size',     size(z0soi), npoi, npass, nfail)
  call check_int ('albsav size',    size(albsav),npoi, npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_canopy_fluxes
  ! ===========================================================================
  write(*,'(a)') '--- mod_canopy_fluxes ---'

  call check_int ('nsol init',        nsol,              0,    npass, nfail)
  call check_real('tfac init',        tfac,              0.0_wp, npass, nfail)
  call check_int ('termu dim1',       size(termu,1),     npoi, npass, nfail)
  call check_int ('termu dim2',       size(termu,2),     7,    npass, nfail)
  call check_int ('terml dim2',       size(terml,2),     7,    npass, nfail)
  call check_int ('scalcoefu dim1',   size(scalcoefu,1), npoi, npass, nfail)
  call check_int ('scalcoefu dim2',   size(scalcoefu,2), 4,    npass, nfail)
  call check_int ('scalcoefl dim2',   size(scalcoefl,2), 4,    npass, nfail)
  call check_int ('indsol size',      size(indsol),      npoi, npass, nfail)
  call check_int ('indsol(1) init',   indsol(1),         0,    npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_work
  ! ===========================================================================
  write(*,'(a)') '--- mod_work ---'

  call check_real('ocean_fill',     ocean_fill, 9.0e20_wp, npass, nfail)
  call check_int ('ndim2',          ndim2,      nlon*nlat,            npass, nfail)
  call check_int ('ndim4',          ndim4,      nlon,                 npass, nfail)
  call check_int ('ndim3',          ndim3,      nlon*nlat*nlon,       npass, nfail)
  call check_real('lonscale(1) init',lonscale(1), 0.0_wp, npass, nfail)
  call check_real('latscale(1) init',latscale(1), 0.0_wp, npass, nfail)
  call check_int ('lonindex size',  size(lonindex), npoi, npass, nfail)
  call check_int ('latindex size',  size(latindex), npoi, npass, nfail)
  call check_int ('lonscale size',  size(lonscale), nlon, npass, nfail)
  call check_int ('latscale size',  size(latscale), nlat, npass, nfail)
  call check_int ('work size',      size(work),     nlon*nlat, npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_summary
  ! ===========================================================================
  write(*,'(a)') '--- mod_summary ---'

  call check_int ('ndtimes init',   ndtimes, 0,    npass, nfail)
  call check_int ('nmtimes init',   nmtimes, 0,    npass, nfail)
  call check_int ('nytimes init',   nytimes, 0,    npass, nfail)
  call check_real('adrain(1) init', adrain(1), 0.0_wp, npass, nfail)
  call check_real('adsnow(1) init', adsnow(1), 0.0_wp, npass, nfail)
  call check_int ('adrain size',    size(adrain), npoi, npass, nfail)
  call check_int ('amnpp dim1',     size(amnpp,1), npoi, npass, nfail)
  call check_int ('amnpp dim2',     size(amnpp,2), npft, npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_canopy_params
  ! ===========================================================================
  write(*,'(a)') '--- mod_canopy_params ---'

  call check_real('alaiml init',    alaiml,    0.0_wp, npass, nfail)
  call check_real('alaimu init',    alaimu,    0.0_wp, npass, nfail)
  call check_real('woodnorm init',  woodnorm,  0.0_wp, npass, nfail)
  call check_real('q12(1) init',    q12(1),    0.0_wp, npass, nfail)
  call check_real('tl(1) init',     tl(1),     0.0_wp, npass, nfail)
  call check_real('agcub(1) init',  agcub(1),  0.0_wp, npass, nfail)
  call check_real('ciub(1) init',   ciub(1),   0.0_wp, npass, nfail)
  call check_real('gsub(1) init',   gsub(1),   0.0_wp, npass, nfail)
  call check_int ('q12 size',       size(q12),    npoi, npass, nfail)
  call check_int ('agcub size',     size(agcub),  npoi, npass, nfail)
  call check_int ('dleaf size',     size(dleaf),  2,    npass, nfail)
  call check_int ('rhoveg dim1',    size(rhoveg,1),nband,npass, nfail)
  call check_int ('rhoveg dim2',    size(rhoveg,2),2,    npass, nfail)
  call check_int ('tauveg dim1',    size(tauveg,1),nband,npass, nfail)
  call check_int ('totcondub size', size(totcondub),npoi, npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! mod_veg_dynamics
  ! ===========================================================================
  write(*,'(a)') '--- mod_veg_dynamics ---'

  call check_real('agddl(1) init',  agddl(1),  0.0_wp, npass, nfail)
  call check_real('fl(1) init',     fl(1),     0.0_wp, npass, nfail)
  call check_real('fu(1) init',     fu(1),     0.0_wp, npass, nfail)
  call check_real('firefac(1) init',firefac(1),0.0_wp, npass, nfail)
  call check_real('tneetot(1) init',tneetot(1),0.0_wp, npass, nfail)
  call check_real('biomass(1,1) init',biomass(1,1),0.0_wp,npass,nfail)
  call check_real('plai(1,1) init', plai(1,1), 0.0_wp, npass, nfail)
  call check_real('lai(1,1) init',  lai(1,1),  0.0_wp, npass, nfail)
  call check_real('froot(1,1) init',froot(1,1),0.0_wp, npass, nfail)
  call check_real('exist(1,1) init',exist(1,1),0.0_wp, npass, nfail)
  call check_real('specla(1) init', specla(1), 0.0_wp, npass, nfail)
  call check_real('aleaf(1) init',  aleaf(1),  0.0_wp, npass, nfail)
  call check_int ('agddl size',     size(agddl),  npoi, npass, nfail)
  call check_int ('biomass dim1',   size(biomass,1),npoi,npass, nfail)
  call check_int ('biomass dim2',   size(biomass,2),npft,npass, nfail)
  call check_int ('plai dim2',      size(plai,2),  npft, npass, nfail)
  call check_int ('lai dim2',       size(lai,2),   2,    npass, nfail)
  call check_int ('ztop dim2',      size(ztop,2),  2,    npass, nfail)
  call check_int ('froot dim1',     size(froot,1), nsoilay,npass,nfail)
  call check_int ('froot dim2',     size(froot,2), 2,    npass, nfail)
  call check_int ('exist dim2',     size(exist,2), npft, npass, nfail)
  call check_int ('specla size',    size(specla),  npft, npass, nfail)
  call check_int ('aroot size',     size(aroot),   npft, npass, nfail)
  write(*,*)

  ! ===========================================================================
  ! Summary
  ! ===========================================================================
  write(*,'(a,i0,a,i0,a)') 'Results: ', npass, ' passed, ', nfail, ' failed.'
  if (nfail > 0) error stop 'One or more Phase 3 tests FAILED.'
  write(*,'(a)') 'All Phase 3 tests PASSED.'

contains

  subroutine check_real(name, got, expected, np, nf)
    character(len=*), intent(in)    :: name
    real(wp),         intent(in)    :: got, expected
    integer,          intent(inout) :: np, nf
    real(wp), parameter :: tol = 1.0e-6_wp
    real(wp) :: diff
    ! For very large values (like ocean_fill), use relative tolerance;
    ! for values near zero, use absolute tolerance.
    if (abs(expected) > 1.0_wp) then
      diff = abs(got - expected) / abs(expected)
    else
      diff = abs(got - expected)
    end if
    if (diff <= tol) then
      write(*,'(a,a,a)') '  PASS: ', trim(name)
      np = np + 1
    else
      write(*,'(a,a,a,g0,a,g0)') '  FAIL: ', trim(name), &
            '  got=', got, '  expected=', expected
      nf = nf + 1
    end if
  end subroutine check_real

  subroutine check_int(name, got, expected, np, nf)
    character(len=*), intent(in)    :: name
    integer,          intent(in)    :: got, expected
    integer,          intent(inout) :: np, nf
    if (got == expected) then
      write(*,'(a,a,a)') '  PASS: ', trim(name)
      np = np + 1
    else
      write(*,'(a,a,a,i0,a,i0)') '  FAIL: ', trim(name), &
            '  got=', got, '  expected=', expected
      nf = nf + 1
    end if
  end subroutine check_int

end program test_phase3
