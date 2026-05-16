! =============================================================================
! diagnostics.f90  --  Diagnostic point output routines for IBIS
!
! Converted from diagnostics.f (Fortran 77 fixed-form) to F2008 free-form.
! Original author comments: last update 12/18/00 CCM.
!
! Subroutines provided:
!   inidiag(idiag)                           -- read diag.infile and init
!   diaginit(idiag, diaglat, diaglon)        -- open files, find grid cells
!   wdiag(i, iyear, imonth, iday, istep)    -- write one row of diagnostics
! =============================================================================

! -----------------------------------------------------------------------------
subroutine inidiag(idiag)
! Reads diagnostics specification from 'diag.infile' and initialises the
! diagnostic output system by calling diaginit.
! -----------------------------------------------------------------------------
  use mod_diagnostics, only: nfiles, nvars, ldiag, diagstart, diagend, nfreq
  implicit none

  integer, intent(out) :: idiag      ! number of diagnostic files requested

  real    :: diaglat(nfiles)         ! latitude  of each diagnostic point
  real    :: diaglon(nfiles)         ! longitude of each diagnostic point

  integer :: lun, i, ivar, ifile

  lun = 12
  open(lun, status='old', file='diag.infile')

  ! Skip 26 header comment lines
  do i = 1, 26
    read(lun,*)
  end do

  ! Read per-file specifications
  do i = 1, nfiles
    read(lun,*) diaglat(i)
    read(lun,*) diaglon(i)
    read(lun,*) diagstart(i)
    read(lun,*) diagend(i)
    read(lun,*) nfreq(i)
    read(lun,*)
  end do

  ! Skip 8 separator lines
  do i = 1, 8
    read(lun,*)
  end do

  ! Read variable selection table: ldiag(ivar, ifile)
  do ivar = 1, nvars
    read(lun,*) (ldiag(ivar, ifile), ifile = 1, nfiles)
  end do

  close(lun)

  call diaginit(idiag, diaglat, diaglon)

end subroutine inidiag


! -----------------------------------------------------------------------------
subroutine diaginit(idiag, diaglat, diaglon)
! Opens diagnostic output files and determines which grid cells correspond
! to the requested diagnostic coordinates.
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi, xres, yres
  use mod_work,         only: latscale, lonscale, latindex, lonindex
  use mod_run_params,   only: nlonsub, nlatsub
  use mod_diagnostics,  only: nfiles, nvars, ldiag, ndiagpt, diagstart, diagend
  implicit none

  integer, intent(out) :: idiag                    ! number of active diag files
  real,    intent(in)  :: diaglat(nfiles)          ! requested latitudes
  real,    intent(in)  :: diaglon(nfiles)          ! requested longitudes

  character(len=10) :: filen
  character(len=10) :: varname(nvars)
  character(len=10) :: header(16, nfiles)

  integer :: ifile, ivar, kolumn, countvar, i, n
  integer :: lat(nfiles), lon(nfiles)

  real(wp) :: alat0, alon0

  data varname / 'adco2mic', 'adco2root', 'adco2soi', 'adneetot',  &
                 'adtsoic', 'adwsoic', 'amco2root', 'amco2soi',    &
                 'ancl3', 'ancl4', 'ancub', 'ancuc', 'asurd1',     &
                 'asurd2', 'asuri1', 'asuri2', 'ayco2soi',         &
                 'biomass1', 'biomass2', 'biomass3', 'biomass4',   &
                 'biomass5', 'biomass6', 'biomass7', 'biomass8',   &
                 'biomass9', 'biomass10', 'biomass11', 'biomass12', &
                 'cloud', 'coszen', 'fi', 'fira', 'firb', 'frac1', 'frac2', &
                 'frac3', 'frac4', 'frac5', 'frac6', 'frac7', 'frac8',      &
                 'frac9', 'frac10', 'frac11', 'frac12', 'gadjust', &
                 'gdrain', 'ginvap', 'grunof', 'gsuvap', 'gtrans', &
                 'gtransl', 'gtransu', 'hsno1', 'hsno2', 'hsno3',  &
                 'hsnotop', 'plai1', 'plai2', 'plai3', 'plai4', 'plai5',    &
                 'plai6', 'plai7', 'plai8', 'plai9', 'plai10', 'plai11',    &
                 'plai12', 'precip', 'psurf', 'qa', 'raina', 'snowa',        &
                 'solad1', 'solad2', 'solai1', 'solai2', 'ta', 'td', 'tl',  &
                 'totalit', 'totcmic', 'totcsoi', 'totrlit', 'trng',         &
                 'ts', 'tsno1', 'tsno2', 'tsno3', 'tsoi1', 'tsoi2',         &
                 'tsoi3', 'tsoi4', 'tsoi5', 'tsoi6', 'tu',                  &
                 'wet', 'wipud', 'wisoi1', 'wisoi2', 'wisoi3', 'wisoi4',     &
                 'wisoi5', 'wisoi6', 'wliql',                                &
                 'wliqs', 'wliqu',                                           &
                 'wpud', 'wsnol', 'wsnos', 'wsnou', 'wsoi1', 'wsoi2',       &
                 'wsoi3', 'wsoi4', 'wsoi5', 'wsoi6' /

  ! Build column header arrays for each diagnostic file
  do ifile = 1, nfiles
    kolumn   = 4
    countvar = 0
    do ivar = 1, nvars
      if (ldiag(ivar, ifile) == 1) then
        kolumn = kolumn + 1
        header(kolumn, ifile) = varname(ivar)
        countvar = countvar + 1
      end if
    end do
    if (countvar < 12) then
      do i = countvar + 5, 16
        header(i, ifile) = 'empty'
      end do
    end if
  end do

  ! Open each diagnostic output file and write its header
  do i = 1, idiag
    filen = 'ibis.diag '
    write(filen(10:10), '(i1)') i - 1
    open(i + 20, status='unknown', file=filen, position='append')
    write(i + 20, 3000) diaglat(i), diaglon(i)
    write(i + 20, 3100) diagstart(i), diagend(i)
    write(i + 20, 3200) (header(kolumn, i), kolumn = 5, 16)
3000 format('%Cell_latitude ', f8.2, 1x, ' Cell_longitude ', f8.2)
3100 format('%Begin_year ', i8, ' End_year ', i8)
3200 format('%Year', 2x, 'Month', 4x, 'Day', 4x, 'Step', 3x, 12(a12, 1x))
  end do

  ! Determine which grid cells correspond to the requested coordinates
  alat0 = latscale(1)
  alon0 = lonscale(1)

  do i = 1, idiag
    lat(i) = 1 + int((alat0 - diaglat(i)) / real(yres, kind=wp) + 0.5_wp)
    lat(i) = min(lat(i), nlatsub)
    lat(i) = max(lat(i), 1)
    lon(i) = 1 + int((diaglon(i) - alon0) / real(xres, kind=wp) + 0.5_wp)
    lon(i) = min(lon(i), nlonsub)
    lon(i) = max(lon(i), 1)
    ndiagpt(i) = -999
    do n = 1, npoi
      if (latindex(n) == lat(i) .and. lonindex(n) == lon(i)) then
        ndiagpt(i) = n
      end if
    end do
    if (ndiagpt(i) < 1) then
      print *, 'Warning: diag point ', i, ':', diaglon(i), diaglat(i), &
               ' is not a land point.  Ignoring'
    end if
  end do

end subroutine diaginit


! -----------------------------------------------------------------------------
subroutine wdiag(i, iyear, imonth, iday, istep)
! Writes one row of diagnostic data for diagnostic point i to its output file.
! Called once per timestep for each active diagnostic point.
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_diagnostics,      only: nvars, ndiagpt, ldiag
  use mod_canopy_params,    only: ancub, ancuc, ancl3, ancl4,         &
                                   tl, ts, tu,                         &
                                   wliql, wliqs, wliqu,                &
                                   wsnol, wsnos, wsnou
  use mod_canopy_fluxes,    only: firb
  use mod_atm_state,        only: coszen, fira, asurd, asuri,         &
                                   cloud, psurf, qa, raina, snowa,     &
                                   ta, td, tmax, precip,               &
                                   solad, solai, iwet
  use mod_snow_state,       only: fi, hsno, hsnotop, tsno
  use mod_hydrology,        only: ginvap, gsuvap, gtrans, gtransu,    &
                                   gtransl, grunof, gdrain, gadjust
  use mod_veg_dynamics,     only: biomass, frac, plai,                 &
                                   totalit, totcmic, totcsoi, totrlit
  use mod_soil_state,       only: tsoi, wsoi, wisoi, wipud, wpud
  use mod_summary,          only: adco2mic, adco2root, adco2soi,      &
                                   adneetot, adtsoic, adwsoic,         &
                                   amco2root, amco2soi,                &
                                   ayco2soi
  implicit none

  integer, intent(in) :: i       ! diagnostic file index (1-based)
  integer, intent(in) :: iyear, imonth, iday, istep

  real(wp) :: diag(nvars)
  real(wp) :: rowdata(16)
  integer  :: ivar, kolumn, countvar, pt

  pt = ndiagpt(i)    ! land-point index for this diagnostic point

  diag(1)   = adco2mic(pt)
  diag(2)   = adco2root(pt)
  diag(3)   = adco2soi(pt)
  diag(4)   = adneetot(pt)
  diag(5)   = adtsoic(pt)
  diag(6)   = adwsoic(pt)
  diag(7)   = amco2root(pt)
  diag(8)   = amco2soi(pt)
  diag(9)   = ancl3(pt)
  diag(10)  = ancl4(pt)
  diag(11)  = ancub(pt)
  diag(12)  = ancuc(pt)
  diag(13)  = asurd(pt, 1)
  diag(14)  = asurd(pt, 2)
  diag(15)  = asuri(pt, 1)
  diag(16)  = asuri(pt, 2)
  diag(17)  = ayco2soi(pt)
  diag(18)  = biomass(pt, 1)
  diag(19)  = biomass(pt, 2)
  diag(20)  = biomass(pt, 3)
  diag(21)  = biomass(pt, 4)
  diag(22)  = biomass(pt, 5)
  diag(23)  = biomass(pt, 6)
  diag(24)  = biomass(pt, 7)
  diag(25)  = biomass(pt, 8)
  diag(26)  = biomass(pt, 9)
  diag(27)  = biomass(pt, 10)
  diag(28)  = biomass(pt, 11)
  diag(29)  = biomass(pt, 12)
  diag(30)  = cloud(pt)
  diag(31)  = coszen(pt)
  diag(32)  = fi(pt)
  diag(33)  = fira(pt)
  diag(34)  = firb(pt)
  diag(35)  = frac(pt, 1)
  diag(36)  = frac(pt, 2)
  diag(37)  = frac(pt, 3)
  diag(38)  = frac(pt, 4)
  diag(39)  = frac(pt, 5)
  diag(40)  = frac(pt, 6)
  diag(41)  = frac(pt, 7)
  diag(42)  = frac(pt, 8)
  diag(43)  = frac(pt, 9)
  diag(44)  = frac(pt, 10)
  diag(45)  = frac(pt, 11)
  diag(46)  = frac(pt, 12)
  diag(47)  = gadjust(pt)
  diag(48)  = gdrain(pt)
  diag(49)  = ginvap(pt)
  diag(50)  = grunof(pt)
  diag(51)  = gsuvap(pt)
  diag(52)  = gtrans(pt)
  diag(53)  = gtransl(pt)
  diag(54)  = gtransu(pt)
  diag(55)  = hsno(pt, 1)
  diag(56)  = hsno(pt, 2)
  diag(57)  = hsno(pt, 3)
  diag(58)  = hsnotop
  diag(59)  = plai(pt, 1)
  diag(60)  = plai(pt, 2)
  diag(61)  = plai(pt, 3)
  diag(62)  = plai(pt, 4)
  diag(63)  = plai(pt, 5)
  diag(64)  = plai(pt, 6)
  diag(65)  = plai(pt, 7)
  diag(66)  = plai(pt, 8)
  diag(67)  = plai(pt, 9)
  diag(68)  = plai(pt, 10)
  diag(69)  = plai(pt, 11)
  diag(70)  = plai(pt, 12)
  diag(71)  = precip(pt)
  diag(72)  = psurf(pt)
  diag(73)  = qa(pt)
  diag(74)  = raina(pt)
  diag(75)  = snowa(pt)
  diag(76)  = solad(pt, 1)
  diag(77)  = solad(pt, 2)
  diag(78)  = solai(pt, 1)
  diag(79)  = solai(pt, 2)
  diag(80)  = ta(pt)
  diag(81)  = td(pt)
  diag(82)  = tl(pt)
  diag(83)  = totalit(pt)
  diag(84)  = totcmic(pt)
  diag(85)  = totcsoi(pt)
  diag(86)  = totrlit(pt)
  diag(87)  = tmax(pt)
  diag(88)  = ts(pt)
  diag(89)  = tsno(pt, 1)
  diag(90)  = tsno(pt, 2)
  diag(91)  = tsno(pt, 3)
  diag(92)  = tsoi(pt, 1)
  diag(93)  = tsoi(pt, 2)
  diag(94)  = tsoi(pt, 3)
  diag(95)  = tsoi(pt, 4)
  diag(96)  = tsoi(pt, 5)
  diag(97)  = tsoi(pt, 6)
  diag(98)  = tu(pt)
  diag(99)  = real(iwet(pt), wp)      ! integer → real conversion
  diag(100) = wipud(pt)
  diag(101) = wisoi(pt, 1)
  diag(102) = wisoi(pt, 2)
  diag(103) = wisoi(pt, 3)
  diag(104) = wisoi(pt, 4)
  diag(105) = wisoi(pt, 5)
  diag(106) = wisoi(pt, 6)
  diag(107) = wliql(pt)
  diag(108) = wliqs(pt)
  diag(109) = wliqu(pt)
  diag(110) = wpud(pt)
  diag(111) = wsnol(pt)
  diag(112) = wsnos(pt)
  diag(113) = wsnou(pt)
  diag(114) = wsoi(pt, 1)
  diag(115) = wsoi(pt, 2)
  diag(116) = wsoi(pt, 3)
  diag(117) = wsoi(pt, 4)
  diag(118) = wsoi(pt, 5)
  diag(119) = wsoi(pt, 6)

  rowdata(1) = real(iyear,  wp)
  rowdata(2) = real(imonth, wp)
  rowdata(3) = real(iday,   wp)
  rowdata(4) = real(istep,  wp)

  kolumn   = 4
  countvar = 0
  do ivar = 1, nvars
    if (ldiag(ivar, i) == 1) then
      countvar         = countvar + 1
      kolumn           = kolumn + 1
      rowdata(kolumn)  = diag(ivar)
    end if
  end do

  if (countvar < 12) then
    rowdata(countvar + 5 : 16) = -999.0_wp
  end if

  write(i + 20, 3000) rowdata
3000 format(1x, f5.0, 2x, f3.0, 3x, f7.0, 1x, f5.0, 1x, 12(e12.3, 1x))

end subroutine wdiag
