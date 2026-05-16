! =============================================================================
! io.f90  --  NetCDF I/O routines for IBIS
!
! Converted from io.f (Fortran 77 fixed-form) to F2008 free-form.
!
! Subroutines provided:
!   wrestart(nday, iyear, iyear0)
!   wdaily(nday, iyear, iyear0)
!   wmonthly(nday, imonth, iyear, iyear0)
!   wyearly(nday, iyear, iyear0)
!   readit(isimveg, snorth, ssouth, swest, seast, iwest, jnorth)
!   restart(iyrlast)
!   coldstart()
!   rdanom(imonth, iyear, iyranom, nanom, iy2, istyr, iwest, jnorth)
!   inird(file, istyr)
!   rdday(jday, imonth, iyear, istyr, iwest, jnorth)
!   simdate(cdate)
!
! F77->F2008 conversions applied throughout:
!   - COMMON blocks replaced by module USE statements
!   - EQUIVALENCE eliminated; scratch variables declared locally
!   - Labeled DO loops replaced by unnamed do...end do
!   - 'OCEAN' replaced by ocean_fill from mod_work
!   - lenchr() replaced by len_trim() intrinsic
!   - float(n) replaced by real(n, wp)
!   - F77 operators (.eq., .ne., etc.) replaced by F90 forms (==, /=, etc.)
!   - cdummy(:) allocatable; readit calls allocate(cdummy(ndim3)) on entry
!   - ies-io.f90 uses bare 'real'; precision matches in IBIS_SP (-DIBIS_SP) build
! =============================================================================


! -----------------------------------------------------------------------------
subroutine wrestart(nday, iyear, iyear0)
! Writes restart NetCDF files (one per state variable, in restart/odd or
! restart/even depending on whether iyear is odd or even).
! Files are created only on the first two years of the run (nyears <= 2).
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: nlon, nlat, npoi, nsnolay, nsoilay, npft
  use mod_run_params,   only: nlonsub, nlatsub
  use mod_work,         only: ocean_fill, cdummy, lonscale, latscale
  use mod_soil_state,   only: hsoi, tsoi, wisoi, wsoi, wipud, wpud
  use mod_snow_state,   only: fi, tsno, hsno
  use mod_carbon_pools, only: cbiol, cbiow, cbior, &
                               clitlm, clitls, clitll, &
                               clitrm, clitrs, clitrl, &
                               clitwm, clitws, clitwl, &
                               falll, fallr, fallw,    &
                               csoislop, csoislon, csoipas
  use mod_veg_dynamics, only: totcmic, sapfrac, gdd0, gdd5, tc, tw, &
                               agddu, agddl, tempu, templ,           &
                               dropu, dropls, dropl4, dropl3
  use mod_summary,      only: a10td, a10ancub, a10ancls, a10ancl4, a10ancl3, &
                               a10scalparamu, a10scalparaml,                  &
                               a10daylightu, a10daylightl
  implicit none

  include 'netcdf.inc'

  integer, intent(in) :: nday, iyear, iyear0

  ! local variables
  integer :: lf, n, k, idies, istat, nyears
  integer :: istart(4), icount(4)
  character(len=21) :: tunits
  character(len=20) :: fdir
  character(len=11) :: cdate
  character(len=10) :: tdate
  character(len=80) :: dimnames(4)
  character(len=80) :: pftdef(npft)
  character(len=80) :: filen
  real(wp) :: slayers(nsnolay)
  real(wp) :: depthsoi(nsoilay)
  real(wp) :: pindex(npft)
  real(wp) :: ftime, tweight

  istart = [1, 1, 1, 1]
  icount = [nlon, nlat, 1, 1]
  icount(1) = nlonsub
  icount(2) = nlatsub

  tweight = 1.0_wp

  ! check to see if iyear is odd or even, construct file name root
  if (mod(iyear,2) == 0) then
    fdir = 'restart/even'
  else
    fdir = 'restart/odd'
  end if
  lf = len_trim(fdir)

  ! tdate is december of this year
  tdate = 'DEC000'//char(0)//char(0)//char(0)//char(0)
  nyears = iyear - iyear0 + 1
  if (nyears < 10) then
    write(tdate(6:6),'(i1)') nyears
  else if (nyears < 100) then
    write(tdate(5:6),'(i2)') nyears
  else
    write(tdate(4:6),'(i3)') nyears
  end if

  ! initialize slayers, depthsoi, pindex, pftdef on first two years only
  if (nyears <= 2) then
    call simdate(cdate)
    ftime = real(nday, wp)

    tunits = 'days since 0000-12-31'
    write(tunits(12:15),'(i4)') iyear0-1

    do n = 1, nsnolay
      slayers(n) = real(n, wp)
    end do

    depthsoi(1) = hsoi(1)
    do n = 2, nsoilay
      depthsoi(n) = depthsoi(n-1) + hsoi(n)
    end do

    do n = 1, npft
      pindex(n) = real(n, wp)
    end do

    pftdef(1)  = 'trbrevtr - tropical broadleaf evergreen trees'//char(0)
    pftdef(2)  = 'trbrdetr - tropical broadleaf drought-deciduous trees'//char(0)
    pftdef(3)  = 'wtbrevtr - warm-temperate broadleaf evergreen trees'//char(0)
    pftdef(4)  = 'tecoevtr - temperate conifer evergreen trees'//char(0)
    pftdef(5)  = 'tebrdetr - temperate broadleaf cold-deciduous trees'//char(0)
    pftdef(6)  = 'bocoevtr - boreal conifer evergreen trees'//char(0)
    pftdef(7)  = 'bocodetr - boreal conifer cold-deciduous trees'//char(0)
    pftdef(8)  = 'bobrdetr - boreal broadleaf cold-deciduous trees'//char(0)
    pftdef(9)  = 'evsh - evergreen shrubs'//char(0)
    pftdef(10) = 'desh - deciduous shrubs'//char(0)
    pftdef(11) = 'c4gr - warm (c4) grasses'//char(0)
    pftdef(12) = 'c3gr - cool (c3) grasses'//char(0)

    dimnames(1) = 'longitude'
    dimnames(2) = 'latitude'
    dimnames(4) = 'time'
  end if

  ! fractional snow cover
  filen = fdir(1:lf)//'fsnocov.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for fractional snow cover', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'fsnocov', &
      'instantaneous fractional snow cover','fraction',3,dimnames, &
      ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(fi, cdummy)
  icount(3) = 1
  call writevar(filen,'fsnocov',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, fsnocov'
    stop 1
  end if

  ! temperature of snow layers
  filen = fdir(1:lf)//'tsno.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for snow temperature', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale, &
      'snowlayer','snow layers top to bottom','',nsnolay,slayers, &
      'down',tunits,'gregorian',istat)
    dimnames(3) = 'snowlayer'
    call inivar(idies,'tsno', &
      'instantaneous snow cover temperature','degK',4,dimnames, &
      ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, nsnolay
    call vec2arr(tsno(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  icount(3) = nsnolay
  call writevar(filen,'tsno',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, tsno'
    stop 1
  end if

  ! thickness of snow layers
  filen = fdir(1:lf)//'hsno.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for snow layer thickness', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale, &
      'snowlayer','snow layers top to bottom','',nsnolay,slayers, &
      'down',tunits,'gregorian',istat)
    dimnames(3) = 'snowlayer'
    call inivar(idies,'hsno','instantaneous snow layer thickness', &
      'meters',4,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, nsnolay
    call vec2arr(hsno(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  icount(3) = nsnolay
  call writevar(filen,'hsno',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, hsno'
    stop 1
  end if

  ! temperature of soil layers
  filen = fdir(1:lf)//'tsoi.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for soil temperature', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale, &
      'soillayer','depth of soil layer bottom','meter',nsoilay, &
      depthsoi,'down',tunits,'gregorian',istat)
    dimnames(3) = 'soillayer'
    call inivar(idies,'tsoi','instantaneous soil temperature', &
      'degK',4,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, nsoilay
    call vec2arr(tsoi(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  icount(3) = nsoilay
  call writevar(filen,'tsoi',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, tsoi'
    stop 1
  end if

  ! ice content of soil
  filen = fdir(1:lf)//'wisoi.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for soil ice content', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale, &
      'soillayer','depth of soil layer bottom','meter',nsoilay, &
      depthsoi,'down',tunits,'gregorian',istat)
    dimnames(3) = 'soillayer'
    call inivar(idies,'wisoi', &
      'instantaneous fraction of soil pore space containing ice', &
      'fraction',4,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, nsoilay
    call vec2arr(wisoi(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  icount(3) = nsoilay
  call writevar(filen,'wisoi',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, wisoi'
    stop 1
  end if

  ! water content of soil
  filen = fdir(1:lf)//'wsoi.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for soil water content', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale, &
      'soillayer','depth of soil layer bottom','meter',nsoilay, &
      depthsoi,'down',tunits,'gregorian',istat)
    dimnames(3) = 'soillayer'
    call inivar(idies,'wsoi', &
      'instantaneous fraction of soil pore space containing water', &
      'fraction',4,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, nsoilay
    call vec2arr(wsoi(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  icount(3) = nsoilay
  call writevar(filen,'wsoi',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, wsoi'
    stop 1
  end if

  ! carbon in leaf biomass pool
  filen = fdir(1:lf)//'cbiol.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for carbon in leaf biomass pool', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale, &
      'pft','plant functional type','_',npft,pindex,'', &
      tunits,'gregorian',istat)
    istat = NF_PUT_ATT_TEXT(idies,NF_GLOBAL,'pft_definition',npft*80,pftdef)
    dimnames(3) = 'pft'
    call inivar(idies,'cbiol', &
      'instantaneous carbon in leaf biomass pool','kg/m^2', &
      4,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, npft
    call vec2arr(cbiol(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  icount(3) = npft
  call writevar(filen,'cbiol',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, cbiol'
    stop 1
  end if

  ! carbon in wood
  filen = fdir(1:lf)//'cbiow.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for carbon in wood biomass pool', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale, &
      'pft','plant functional type','_',npft,pindex,'', &
      tunits,'gregorian',istat)
    istat = NF_PUT_ATT_TEXT(idies,NF_GLOBAL,'pft_definition',npft*80,pftdef)
    dimnames(3) = 'pft'
    call inivar(idies,'cbiow', &
      'instantaneous carbon in wood biomass pool','kg/m^2', &
      4,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, npft
    call vec2arr(cbiow(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  icount(3) = npft
  call writevar(filen,'cbiow',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, cbiow'
    stop 1
  end if

  ! carbon in root
  filen = fdir(1:lf)//'cbior.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for carbon in root biomass pool', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale, &
      'pft','plant functional type','_',npft,pindex,'', &
      tunits,'gregorian',istat)
    istat = NF_PUT_ATT_TEXT(idies,NF_GLOBAL,'pft_definition',npft*80,pftdef)
    dimnames(3) = 'pft'
    call inivar(idies,'cbior', &
      'instantaneous carbon in root biomass pool','kg/m^2', &
      4,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, npft
    call vec2arr(cbior(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  icount(3) = npft
  call writevar(filen,'cbior',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, cbior'
    stop 1
  end if

  ! sapwood fraction
  filen = fdir(1:lf)//'sapfrac.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for sapwood fraction', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'sapfrac','instantaneous sapwood fraction', &
      'fraction',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(sapfrac, cdummy)
  icount(3) = 1
  call writevar(filen,'sapfrac',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, sapfrac'
    stop 1
  end if

  ! leaf metabolic litter
  filen = fdir(1:lf)//'clitlm.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for leaf metabolic litter', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'clitlm', &
      'instantaneous leaf metabolic litter carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(clitlm, cdummy)
  icount(3) = 1
  call writevar(filen,'clitlm',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, clitlm'
    stop 1
  end if

  ! leaf structural litter
  filen = fdir(1:lf)//'clitls.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for leaf structural litter', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'clitls', &
      'instantaneous leaf structural litter carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(clitls, cdummy)
  icount(3) = 1
  call writevar(filen,'clitls',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, clitls'
    stop 1
  end if

  ! leaf lignin litter
  filen = fdir(1:lf)//'clitll.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for leaf lignin litter', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'clitll', &
      'instantaneous leaf lignin litter carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(clitll, cdummy)
  icount(3) = 1
  call writevar(filen,'clitll',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, clitll'
    stop 1
  end if

  ! root metabolic litter
  filen = fdir(1:lf)//'clitrm.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for root metabolic litter', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'clitrm', &
      'instantaneous root metabolic litter carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(clitrm, cdummy)
  icount(3) = 1
  call writevar(filen,'clitrm',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, clitrm'
    stop 1
  end if

  ! root structural litter
  filen = fdir(1:lf)//'clitrs.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for root structural litter', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'clitrs', &
      'instantaneous root structural litter carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(clitrs, cdummy)
  icount(3) = 1
  call writevar(filen,'clitrs',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, clitrs'
    stop 1
  end if

  ! root lignin litter
  filen = fdir(1:lf)//'clitrl.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for root lignin litter', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'clitrl', &
      'instantaneous root lignin litter carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(clitrl, cdummy)
  icount(3) = 1
  call writevar(filen,'clitrl',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, clitrl'
    stop 1
  end if

  ! woody metabolic litter
  filen = fdir(1:lf)//'clitwm.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for woody metabolic litter', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'clitwm', &
      'instantaneous woody metabolic litter carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(clitwm, cdummy)
  icount(3) = 1
  call writevar(filen,'clitwm',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, clitwm'
    stop 1
  end if

  ! woody structural litter
  filen = fdir(1:lf)//'clitws.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for woody structural litter', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'clitws', &
      'instantaneous woody structural litter carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(clitws, cdummy)
  icount(3) = 1
  call writevar(filen,'clitws',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, clitws'
    stop 1
  end if

  ! woody lignin litter
  filen = fdir(1:lf)//'clitwl.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for woody lignin litter', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'clitwl', &
      'instantaneous woody lignin litter carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(clitwl, cdummy)
  icount(3) = 1
  call writevar(filen,'clitwl',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, clitwl'
    stop 1
  end if

  ! annual leaf litterfall
  filen = fdir(1:lf)//'falll.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for annual leaf litterfall', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'falll','annual leaf litterfall carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(falll, cdummy)
  icount(3) = 1
  call writevar(filen,'falll',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, falll'
    stop 1
  end if

  ! annual fine root turnover
  filen = fdir(1:lf)//'fallr.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for annual fine root turnover', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'fallr','annual fine root turnover carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(fallr, cdummy)
  icount(3) = 1
  call writevar(filen,'fallr',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, fallr'
    stop 1
  end if

  ! annual wood turnover
  filen = fdir(1:lf)//'fallw.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for annual woody turnover', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'fallw','annual wood turnover carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(fallw, cdummy)
  icount(3) = 1
  call writevar(filen,'fallw',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, fallw'
    stop 1
  end if

  ! total microbial carbon
  filen = fdir(1:lf)//'totcmic.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for total microbial carbon', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'totcmic', &
      'instantaneous total microbial carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(totcmic, cdummy)
  icount(3) = 1
  call writevar(filen,'totcmic',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, totcmic'
    stop 1
  end if

  ! slow soil carbon, protected humus
  filen = fdir(1:lf)//'csoislop.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for slow soil carbon, protected humus', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'csoislop', &
      'instantaneous slow soil carbon protected humus', &
      'kg/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(csoislop, cdummy)
  icount(3) = 1
  call writevar(filen,'csoislop',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, csoislop'
    stop 1
  end if

  ! slow soil carbon, nonprotected humus
  filen = fdir(1:lf)//'csoislon.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for slow soil carbon, nonprotected humus', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'csoislon', &
      'instantaneous slow soil carbon nonprotected humus', &
      'kg/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(csoislon, cdummy)
  icount(3) = 1
  call writevar(filen,'csoislon',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, csoislon'
    stop 1
  end if

  ! passive soil carbon
  filen = fdir(1:lf)//'csoipas.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for passive soil carbon', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'csoipas', &
      'instantaneous passive soil carbon','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(csoipas, cdummy)
  icount(3) = 1
  call writevar(filen,'csoipas',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, csoipas'
    stop 1
  end if

  ! growing degree days above 0 deg_C
  filen = fdir(1:lf)//'gdd0.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for growing degree days above 0 deg_C', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'gdd0', &
      'instantaneous growing degree days above 0 deg_C', &
      'days degC',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(gdd0, cdummy)
  icount(3) = 1
  call writevar(filen,'gdd0',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, gdd0'
    stop 1
  end if

  ! growing degree days above 5 deg_C
  filen = fdir(1:lf)//'gdd5.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for growing degree days above 5 deg_C', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'gdd5', &
      'instantaneous growing degree days above 5 deg_C', &
      'days degC',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(gdd5, cdummy)
  icount(3) = 1
  call writevar(filen,'gdd5',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, gdd5'
    stop 1
  end if

  ! coldest monthly temperature
  filen = fdir(1:lf)//'tc.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for coldest monthly temperature', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'tc', &
      'instantaneous coldest monthly temperature','degC', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(tc, cdummy)
  icount(3) = 1
  call writevar(filen,'tc',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, tc'
    stop 1
  end if

  ! warmest monthly temperature
  filen = fdir(1:lf)//'tw.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for warmest monthly temperature', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'tw', &
      'instantaneous warmest monthly temperature','degC', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(tw, cdummy)
  icount(3) = 1
  call writevar(filen,'tw',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, tw'
    stop 1
  end if

  ! ice content of puddles
  filen = fdir(1:lf)//'wipud.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for ice content of puddles', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'wipud', &
      'instantaneous ice content of puddles','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(wipud, cdummy)
  icount(3) = 1
  call writevar(filen,'wipud',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, wipud'
    stop 1
  end if

  ! liquid content of puddles
  filen = fdir(1:lf)//'wpud.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for liquid content of puddles', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'wpud', &
      'instantaneous liquid water content of puddles','kg/m^2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(wpud, cdummy)
  icount(3) = 1
  call writevar(filen,'wpud',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, wpud'
    stop 1
  end if

  ! upper canopy growing degree days for bud burst
  filen = fdir(1:lf)//'agddu.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for upper canopy growing degree days', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'agddu', &
      'instantaneous growing degree days uc','days degC', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(agddu, cdummy)
  icount(3) = 1
  call writevar(filen,'agddu',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, agddu'
    stop 1
  end if

  ! lower canopy growing degree days for bud burst
  filen = fdir(1:lf)//'agddl.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for lower canopy growing degree days', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'agddl', &
      'instantaneous growing degree days lc','days degC', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(agddl, cdummy)
  icount(3) = 1
  call writevar(filen,'agddl',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, agddl'
    stop 1
  end if

  ! cold-phenology trigger for trees
  filen = fdir(1:lf)//'tempu.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for cold phenology trigger for trees', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'tempu', &
      'cold phenology trigger for trees','_', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(tempu, cdummy)
  icount(3) = 1
  call writevar(filen,'tempu',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, tempu'
    stop 1
  end if

  ! cold-phenology trigger for grasses/shrubs
  filen = fdir(1:lf)//'templ.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for cold phenology trigger for grasses/shrubs', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'templ', &
      'cold phenology trigger for grasses/shrubs','_', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(templ, cdummy)
  icount(3) = 1
  call writevar(filen,'templ',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, templ'
    stop 1
  end if

  ! 10-day average daily air temperature
  filen = fdir(1:lf)//'a10td.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for 10-day average daily air T', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'a10td','10-day average daily air T','K', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(a10td, cdummy)
  icount(3) = 1
  call writevar(filen,'a10td',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, a10td'
    stop 1
  end if

  ! 10-day average canopy photosynthesis rate - broadleaf
  filen = fdir(1:lf)//'a10ancub.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for 10-day average canopy photosynth. rate, broadleaf', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'a10ancub', &
      '10-day average canopy photosynth. rate, broadleaf', &
      'mol_co2 m-2 s-1',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(a10ancub, cdummy)
  icount(3) = 1
  call writevar(filen,'a10ancub',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, a10ancub'
    stop 1
  end if

  ! 10-day average canopy photosynthesis rate - shrubs
  filen = fdir(1:lf)//'a10ancls.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for 10-day average canopy photosynth. rate, shrubs', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'a10ancls', &
      '10-day average canopy photosynth. rate, shrubs', &
      'mol_co2 m-2 s-1',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(a10ancls, cdummy)
  icount(3) = 1
  call writevar(filen,'a10ancls',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, a10ancls'
    stop 1
  end if

  ! 10-day average canopy photosynthesis rate - c4 grasses
  filen = fdir(1:lf)//'a10ancl4.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for 10-day average canopy photosynth. rate, c4 grasses', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'a10ancl4', &
      '10-day average canopy photosynth. rate, c4 grasses', &
      'mol_co2 m-2 s-1',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(a10ancl4, cdummy)
  icount(3) = 1
  call writevar(filen,'a10ancl4',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, a10ancl4'
    stop 1
  end if

  ! 10-day average canopy photosynthesis rate - c3 grasses
  filen = fdir(1:lf)//'a10ancl3.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for 10-day average canopy photosynth. rate, c3 grasses', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'a10ancl3', &
      '10-day average canopy photosynth. rate, c3 grasses', &
      'mol_co2 m-2 s-1',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(a10ancl3, cdummy)
  icount(3) = 1
  call writevar(filen,'a10ancl3',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, a10ancl3'
    stop 1
  end if

  ! 10-day average canopy scaling parameter - upper canopy
  filen = fdir(1:lf)//'a10scalparamu.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for 10-day average canopy scaling parameter, upper canopy', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'a10scalparamu', &
      '10-day average canopy scaling parameter, upper canopy','_', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(a10scalparamu, cdummy)
  icount(3) = 1
  call writevar(filen,'a10scalparamu',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, a10scalparamu'
    stop 1
  end if

  ! 10-day average canopy scaling parameter - lower canopy
  filen = fdir(1:lf)//'a10scalparaml.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for 10-day average canopy scaling parameter, lower canopy', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'a10scalparaml', &
      '10-day average canopy scaling parameter, lower canopy','_', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(a10scalparaml, cdummy)
  icount(3) = 1
  call writevar(filen,'a10scalparaml',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, a10scalparaml'
    stop 1
  end if

  ! 10-day average daylight - upper canopy
  filen = fdir(1:lf)//'a10daylightu.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for 10-day average daylight, upper canopy', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'a10daylightu', &
      '10-day average daylight, upper canopy','W m-2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(a10daylightu, cdummy)
  icount(3) = 1
  call writevar(filen,'a10daylightu',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, a10daylightu'
    stop 1
  end if

  ! 10-day average daylight - lower canopy
  filen = fdir(1:lf)//'a10daylightl.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for 10-day average daylight, lower canopy', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'a10daylightl', &
      '10-day average daylight, lower canopy','W m-2', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(a10daylightl, cdummy)
  icount(3) = 1
  call writevar(filen,'a10daylightl',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, a10daylightl'
    stop 1
  end if

  ! drought-phenology trigger for trees
  filen = fdir(1:lf)//'dropu.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for drought-pheno. trigger for trees', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'dropu', &
      'drought-pheno. trigger for trees','_', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(dropu, cdummy)
  icount(3) = 1
  call writevar(filen,'dropu',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, dropu'
    stop 1
  end if

  ! drought-phenology trigger for shrubs
  filen = fdir(1:lf)//'dropls.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for drought-pheno. trigger for shrubs', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'dropls', &
      'drought-pheno. trigger for shrubs','_', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(dropls, cdummy)
  icount(3) = 1
  call writevar(filen,'dropls',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, dropls'
    stop 1
  end if

  ! drought-phenology trigger for c4 grasses
  filen = fdir(1:lf)//'dropl4.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for drought-pheno. trigger for c4 grasses', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'dropl4', &
      'drought-pheno. trigger for c4 grasses','_', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(dropl4, cdummy)
  icount(3) = 1
  call writevar(filen,'dropl4',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, dropl4'
    stop 1
  end if

  ! drought-phenology trigger for c3 grasses
  filen = fdir(1:lf)//'dropl3.nc'
  if (nyears <= 2) then
    call inifile(idies,filen, &
      'restart file for drought-pheno. trigger for c3 grasses', &
      'ibis wrestart',cdate,nlonsub,lonscale,nlatsub,latscale,'','none', &
      'none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'dropl3', &
      'drought-pheno. trigger for c3 grasses','_', &
      3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(dropl3, cdummy)
  icount(3) = 1
  call writevar(filen,'dropl3',istart,icount,cdummy,ftime, &
    tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wrestart, dropl3'
    stop 1
  end if

end subroutine wrestart


! -----------------------------------------------------------------------------
subroutine wdaily(nday, iyear, iyear0)
! Writes out daily output files.
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: nlon, nlat, npoi, npft
  use mod_run_params,   only: nlonsub, nlatsub
  use mod_work,         only: ocean_fill, cdummy, lonscale, latscale
  use mod_atm_state,    only: cloud
  use mod_summary,      only: adrain, adrh, adsnow, adaet, adtrunoff, &
                               adsrunoff, addrainage, adwsoi, adwisoi, adsnod
  use mod_veg_dynamics, only: lai
  implicit none

  integer, intent(in) :: nday, iyear, iyear0

  integer  :: mstep, idies, istat
  integer  :: istart(4), icount(4)
  character(len=11) :: cdate
  character(len=10) :: tdate
  character(len=21) :: tunits
  character(len=80) :: dimnames(4), filen
  real(wp) :: ftime, tweight

  istart = [1, 1, 1, 1]
  icount = [nlon, nlat, 1, 1]
  icount(1) = nlonsub
  icount(2) = nlatsub

  ftime   = real(nday, wp)
  mstep   = nday
  tweight = 1.0_wp

  tdate = '0000000'//char(0)//char(0)//char(0)
  if (nday < 10) then
    write(tdate(3:3),'(i1)') nday
  else if (nday < 100) then
    write(tdate(2:3),'(i2)') nday
  else
    write(tdate(1:3),'(i3)') nday
  end if
  if (iyear >= 1000) then
    write(tdate(4:7),'(i4)') iyear
  else if (iyear < 10) then
    write(tdate(7:7),'(i1)') iyear
  else if (iyear < 100) then
    write(tdate(6:7),'(i2)') iyear
  else
    write(tdate(5:7),'(i3)') iyear
  end if

  if (mstep == 1) then
    call simdate(cdate)
    tunits = 'days since 0000-12-31'
    write(tunits(12:15),'(i4)') iyear0-1
    dimnames(1) = 'longitude'
    dimnames(2) = 'latitude'
    dimnames(4) = 'time'
  end if

  ! rainfall
  filen = 'output/daily/rain.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average rainfall','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'rain','average rainfall','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(adrain, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'rain',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, rain'
    stop 1
  end if

  ! cloudiness
  filen = 'output/daily/cloud.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average cloudiness','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'cloud','average cloudiness','%',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(cloud, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'cloud',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, cloud'
    stop 1
  end if

  ! relative humidity
  filen = 'output/daily/rh.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average relative humidity','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'rh','average rh','%',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(adrh, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'rh',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, rh'
    stop 1
  end if

  ! snowfall
  filen = 'output/daily/snow.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average snowfall','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'snow','average snowfall','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(adsnow, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'snow',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, snow'
    stop 1
  end if

  ! actual evapotranspiration
  filen = 'output/daily/aet.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average aet','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'aet','average aet','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(adaet, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'aet',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, aet'
    stop 1
  end if

  ! total runoff
  filen = 'output/daily/trunoff.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average total runoff','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'trunoff','average total runoff','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(adtrunoff, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'trunoff',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, trunoff'
    stop 1
  end if

  ! surface runoff
  filen = 'output/daily/srunoff.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average surface runoff','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'srunoff','average surface runoff','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(adsrunoff, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'srunoff',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, srunoff'
    stop 1
  end if

  ! drainage
  filen = 'output/daily/drainage.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average drainage','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'drainage','average drainage','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(addrainage, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'drainage',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, drainage'
    stop 1
  end if

  ! soil moisture
  filen = 'output/daily/wsoi.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average soil moisture','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'wsoi','average soil moisture','fraction',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(adwsoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'wsoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, wsoi'
    stop 1
  end if

  ! soil ice
  filen = 'output/daily/wisoi.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average soil ice','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'wisoi','average soil ice','fraction',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(adwisoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'wisoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, wisoi'
    stop 1
  end if

  ! snow depth
  filen = 'output/daily/snod.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily average snow depth','ibis wdaily',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'snod','average snow depth','meters',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(adsnod, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'snod',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, snod'
    stop 1
  end if

  ! upper and lower canopy leaf area index
  filen = 'output/daily/laicanopy.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'daily lai-leaf area index of upper and lower vegetation canopies', &
      'ibis wdaily',cdate,nlonsub,lonscale,nlatsub,latscale, &
      '','','',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'laiu','daily lai of upper canopy','m2/m2',3,dimnames,ocean_fill,istat)
    call inivar(idies,'lail','daily lai of lower canopy','m2/m2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(lai(:,1), cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'laiu',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, laiu'
    stop 1
  end if
  call vec2arr(lai(:,2), cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'lail',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wdaily, lail'
    stop 1
  end if

end subroutine wdaily


! -----------------------------------------------------------------------------
subroutine wmonthly(nday, imonth, iyear, iyear0)
! Writes out monthly output files.
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: nlon, nlat, npoi
  use mod_run_params, only: nlonsub, nlatsub, ndaypm
  use mod_work,       only: ocean_fill, cdummy, lonscale, latscale
  use mod_summary,    only: amtemp, amrain, amsnow, amcloud, amrh, amqa, &
                             amaet, amtrunoff, amsrunoff, amdrainage,    &
                             amwsoi, amwisoi, amvwc, amawc, amtsoi,      &
                             amsnod, amsnof, amsolar, amirdown, amirup,  &
                             amsens, amlatent, amlaiu, amlail,           &
                             amnpptot, amco2ratio
  implicit none

  integer, intent(in) :: nday, imonth, iyear, iyear0

  integer  :: mstep, idies, istat
  integer  :: istart(4), icount(4)
  character(len=11) :: cdate
  character(len=10) :: tdate
  character(len=21) :: tunits
  character(len=80) :: dimnames(4), filen
  character(len=3)  :: chmon(12)
  real(wp) :: ftime, tweight

  chmon = ['JAN','FEB','MAR','APR','MAY','JUN', &
           'JUL','AUG','SEP','OCT','NOV','DEC']

  istart = [1, 1, 1, 1]
  icount = [nlon, nlat, 1, 1]
  icount(1) = nlonsub
  icount(2) = nlatsub

  ftime   = real(nday, wp)
  mstep   = 12*(iyear-iyear0) + imonth
  tweight = real(ndaypm(imonth), wp)

  tdate = chmon(imonth)//'0000'//char(0)//char(0)//char(0)
  if (iyear >= 1000) then
    write(tdate(4:7),'(i4)') iyear
  else if (iyear < 10) then
    write(tdate(7:7),'(i1)') iyear
  else if (iyear < 100) then
    write(tdate(6:7),'(i2)') iyear
  else
    write(tdate(5:7),'(i3)') iyear
  end if

  if (mstep == 1) then
    call simdate(cdate)
    tunits = 'days since 0000-12-31'
    write(tunits(12:15),'(i4)') iyear0-1
    dimnames(1) = 'longitude'
    dimnames(2) = 'latitude'
    dimnames(4) = 'time'
  end if

  ! temperature
  filen = 'output/monthly/temp.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average air temperature','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'temp','average air temperature','C',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amtemp, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'temp',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, temp'
    stop 1
  end if

  ! rainfall
  filen = 'output/monthly/rain.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average rainfall','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'rain','average rainfall','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amrain, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'rain',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, rain'
    stop 1
  end if

  ! cloudiness
  filen = 'output/monthly/cloud.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average cloudiness','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'cloud','average cloudiness','%',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amcloud, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'cloud',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, cloud'
    stop 1
  end if

  ! relative humidity
  filen = 'output/monthly/rh.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average relative humidity','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'rh','average rh','%',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amrh, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'rh',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, rh'
    stop 1
  end if

  ! snowfall
  filen = 'output/monthly/snow.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average snowfall','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'snow','average snowfall','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amsnow, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'snow',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, snow'
    stop 1
  end if

  ! specific humidity
  filen = 'output/monthly/qa.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average specific humidity','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'qa','average specific humidity','kg-h2o/kg-air',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amqa, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'qa',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, qa'
    stop 1
  end if

  ! evapotranspiration
  filen = 'output/monthly/aet.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average aet','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'aet','average evapotranspiration','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amaet, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'aet',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, aet'
    stop 1
  end if

  ! total runoff, surface runoff, drainage
  filen = 'output/monthly/runoff.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average total runoff','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'trunoff','average total runoff','mm/day',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'srunoff','average surface runoff','mm/day',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'drainage','average drainage','mm/day',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amtrunoff, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'trunoff',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, trunoff'
    stop 1
  end if
  call vec2arr(amsrunoff, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'srunoff',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, srunoff'
    stop 1
  end if
  call vec2arr(amdrainage, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'drainage',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, drainage'
    stop 1
  end if

  ! soil temperature
  filen = 'output/monthly/tsoi.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average soil temperature','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'tsoi','average soil temperature','degC',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amtsoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'tsoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, tsoi'
    stop 1
  end if

  ! soil moisture, ice, volumetric water content, plant available water
  filen = 'output/monthly/wsoi.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average soil moisture','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'wsoi','average soil moisture','fraction',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'wisoi','average soil ice','fraction',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'vwc','average volumetric water content','fraction',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'awc','average plant available water content','cm',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amwsoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'wsoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, wsoi'
    stop 1
  end if
  call vec2arr(amwisoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'wisoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, wisoi'
    stop 1
  end if
  call vec2arr(amvwc, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'vwc',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, vwc'
    stop 1
  end if
  call vec2arr(amawc, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'awc',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, awc'
    stop 1
  end if

  ! snow depth
  filen = 'output/monthly/snod.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average snow depth','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'snod','average snow depth','meters',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amsnod, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'snod',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, snod'
    stop 1
  end if

  ! snow fraction
  filen = 'output/monthly/snof.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average snow fraction','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'snof','average snow fraction','m^2/m^3',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amsnof, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'snof',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, snof'
    stop 1
  end if

  ! incident solar radiation
  filen = 'output/monthly/solar.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average incident solar radiation','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'solar','average incident solar radiation','W/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amsolar, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'solar',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, solar'
    stop 1
  end if

  ! downward and upward infrared radiation
  filen = 'output/monthly/ir.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average infrared radiation','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'irdown','average downward IR','W/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'irup','average upward IR','W/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amirdown, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'irdown',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, irdown'
    stop 1
  end if
  call vec2arr(amirup, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'irup',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, irup'
    stop 1
  end if

  ! sensible heat flux
  filen = 'output/monthly/sens.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average sensible heat flux','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'sens','average sensible heat flux','W/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amsens, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'sens',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, sens'
    stop 1
  end if

  ! latent heat flux
  filen = 'output/monthly/latent.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average latent heat flux','ibis wmonthly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'latent','average latent heat flux','W/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amlatent, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'latent',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, latent'
    stop 1
  end if

  ! leaf area index, upper and lower canopies
  filen = 'output/monthly/lai.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly average leaf area index for upper and lower canopies', &
      'ibis wmonthly',cdate,nlonsub,lonscale,nlatsub,latscale, &
      '','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'laiu','average lai upper canopy','fraction',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'lail','average lai lower canopy','fraction',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amlaiu, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'laiu',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, laiu'
    stop 1
  end if
  call vec2arr(amlail, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'lail',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, lail'
    stop 1
  end if

  ! total net primary productivity
  filen = 'output/monthly/npptot.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly total net primary productivity of carbon','ibis wmonthly', &
      cdate,nlonsub,lonscale,nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'npptot','total npp of carbon','kg m-2 month-1',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amnpptot, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'npptot',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, npptot'
    stop 1
  end if

  ! co2 ratio
  filen = 'output/monthly/co2ratio.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'monthly ratio of root respiration to total soil respiration', &
      'ibis wmonthly',cdate,nlonsub,lonscale,nlatsub,latscale, &
      '','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'co2ratio','ratio of root to soil respiration', &
      'fraction',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(amco2ratio, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'co2ratio',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wmonthly, co2ratio'
    stop 1
  end if

end subroutine wmonthly


! -----------------------------------------------------------------------------
subroutine wyearly(nday, iyear, iyear0)
! Writes out yearly output files.
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: nlon, nlat, npoi, npft
  use mod_run_params,   only: nlonsub, nlatsub, ndaypy
  use mod_work,         only: ocean_fill, cdummy, lonscale, latscale
  use mod_summary,      only: aynpp, aynpptot, ayanpptot, ayaet,          &
                               aytrunoff, aysrunoff, aydrainage,           &
                               ayrratio, aytratio,                         &
                               aywsoi, aywisoi, ayvwc, ayawc, aytsoi,     &
                               aysens, ayrootbio, ayalit, ayblit,          &
                               aycsoi, aycmic, ayanlit, aybnlit,           &
                               aynsoi, aynmintot, ayneetot,                &
                               ayco2mic, ayco2root, ayco2soi
  use mod_veg_dynamics, only: plai, totlaiu, totlail, biomass, totbiou,    &
                               totbiol, disturbf, vegtype0, fu, fl,        &
                               zbot, ztop, exist
  implicit none
  include 'netcdf.inc'

  integer, intent(in) :: nday, iyear, iyear0

  integer  :: mstep, i, k, idies, istat
  integer  :: istart(4), icount(4)
  real(wp) :: pindex(npft)
  character(len=11) :: cdate
  character(len=10) :: tdate
  character(len=13) :: canopies(2)
  character(len=21) :: tunits
  character(len=80) :: dimnames(4), pftdef(npft), filen
  real(wp) :: ftime, tweight

  istart = [1, 1, 1, 1]
  icount = [nlon, nlat, 1, 1]
  icount(1) = nlonsub
  icount(2) = nlatsub

  ftime   = real(nday - ndaypy + 1, wp)
  tweight = real(ndaypy, wp)
  mstep   = iyear - iyear0 + 1

  tdate = 'ANN0000'//char(0)//char(0)//char(0)
  if (iyear >= 1000) then
    write(tdate(4:7),'(i4)') iyear
  else if (iyear < 10) then
    write(tdate(7:7),'(i1)') iyear
  else if (iyear < 100) then
    write(tdate(6:7),'(i2)') iyear
  else
    write(tdate(5:7),'(i3)') iyear
  end if

  if (mstep == 1) then
    call simdate(cdate)
    tunits = 'days since 0000-12-31'
    write(tunits(12:15),'(i4)') iyear0-1
    dimnames(1) = 'longitude'
    dimnames(2) = 'latitude'
    dimnames(4) = 'time'
    do i = 1, npft
      pindex(i) = real(i, wp)
    end do
    pftdef(1)  = 'trbrevtr - tropical broadleaf evergreen trees'//char(0)
    pftdef(2)  = 'trbrdetr - tropical broadleaf drought-deciduous trees'//char(0)
    pftdef(3)  = 'wtbrevtr - warm-temperate broadleaf evergreen trees'//char(0)
    pftdef(4)  = 'tecoevtr - temperate conifer evergreen trees'//char(0)
    pftdef(5)  = 'tebrdetr - temperate broadleaf cold-deciduous trees'//char(0)
    pftdef(6)  = 'bocoevtr - boreal conifer evergreen trees'//char(0)
    pftdef(7)  = 'bocodetr - boreal conifer cold-deciduous trees'//char(0)
    pftdef(8)  = 'bobrdetr - boreal broadleaf cold-deciduous trees'//char(0)
    pftdef(9)  = 'evsh - evergreen shrubs'//char(0)
    pftdef(10) = 'desh - deciduous shrubs'//char(0)
    pftdef(11) = 'c4gr - warm (c4) grasses'//char(0)
    pftdef(12) = 'c3gr - cool (c3) grasses'//char(0)
    canopies(1) = 'lower canopy'//char(0)
    canopies(2) = 'upper canopy'//char(0)
  end if

  ! net primary productivity, by pft and total
  filen = 'output/yearly/npp.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual net primary productivity of carbon','ibis wyearly',cdate, &
      nlonsub,lonscale,nlatsub,latscale,'pft','plant fuctional type','_', &
      npft,pindex,'',tunits,'gregorian',istat)
    istat = NF_PUT_ATT_TEXT(idies,NF_GLOBAL,'pft_definition',npft*80,pftdef)
    dimnames(3) = 'pft'
    call inivar(idies,'npp','npp of carbon for each pft','kg m-2 year-1',4,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'npptot','total npp','kg m-2 year-1',3,dimnames,ocean_fill,istat)
    call inivar(idies,'anpptot','total above-ground npp','kg m-2 year-1',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, npft
    call vec2arr(aynpp(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  istart(3) = 1
  istart(4) = mstep
  icount(3) = npft
  call writevar(filen,'npp',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, npp'
    stop 1
  end if
  call vec2arr(aynpptot, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'npptot',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, npptot'
    stop 1
  end if
  call vec2arr(ayanpptot, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'anpptot',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, anpptot'
    stop 1
  end if

  ! evapotranspiration
  filen = 'output/yearly/aet.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual average evapotranspiration','ibis wyearly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'aet','average evapotranspiration','mm/year',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(ayaet, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'aet',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, aet'
    stop 1
  end if

  ! trunoff, srunoff, drainage, rratio, tratio
  filen = 'output/yearly/runoff.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual runoff','ibis wyearly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'trunoff','total runoff','mm/year',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'srunoff','surface runoff','mm/year',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'drainage','drainage','mm/year',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'rratio','average runoff ratio','fraction',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'tratio','average transpiration ratio','fraction',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(aytrunoff, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'trunoff',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, trunoff'
    stop 1
  end if
  call vec2arr(aysrunoff, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'srunoff',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, srunoff'
    stop 1
  end if
  call vec2arr(aydrainage, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'drainage',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, drainage'
    stop 1
  end if
  call vec2arr(ayrratio, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'rratio',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, rratio'
    stop 1
  end if
  call vec2arr(aytratio, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'tratio',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, tratio'
    stop 1
  end if

  ! soil moisture, soil ice, volumetric water content, plant available water
  filen = 'output/yearly/wsoi.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual average soil moisture','ibis wyearly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'wsoi','average soil moisture','fraction',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'wisoi','average soil ice','fraction',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'vwc','average volumetric water content','fraction',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'awc','average volumetric water content','cm',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(aywsoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'wsoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, wsoi'
    stop 1
  end if
  call vec2arr(aywisoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'wisoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, wisoi'
    stop 1
  end if
  call vec2arr(ayvwc, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'vwc',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, vwc'
    stop 1
  end if
  call vec2arr(ayawc, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'awc',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, awc'
    stop 1
  end if

  ! soil temperature
  filen = 'output/yearly/tsoi.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual average soil temperature','ibis wyearly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'tsoi','average soil temperature','degrees C',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(aytsoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'tsoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, tsoi'
    stop 1
  end if

  ! sensible heat flux
  filen = 'output/yearly/sens.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual average sensible heat flux','ibis wyearly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'sens','average sensible heat flux','W/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(aysens, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'sens',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, sens'
    stop 1
  end if

  ! soil carbon: rootbio, totalit, totrlit, totcsoi, totcmic
  filen = 'output/yearly/csoi.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual total soil carbon','ibis wyearly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'rootbio','total live root biomass carbon','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totalit','total above ground litter carbon','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totrlit','total below ground litter carbon','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totcsoi','total soil carbon w/o litter','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totcmic','total microbial carbon','kg/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(ayrootbio, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'rootbio',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, rootbio'
    stop 1
  end if
  call vec2arr(ayalit, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totalit',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totalit'
    stop 1
  end if
  call vec2arr(ayblit, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totrlit',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totrlit'
    stop 1
  end if
  call vec2arr(aycsoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totcsoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totcsoi'
    stop 1
  end if
  call vec2arr(aycmic, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totcmic',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totcmic'
    stop 1
  end if

  ! soil nitrogen: totanlit, totrnlit, totnsoi, nmintot
  filen = 'output/yearly/nsoi.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual total soil nitrogen','ibis wyearly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'totanlit','total above ground litter nitrogen','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totrnlit','total below ground litter nitrogen','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totnsoi','total soil nitrogen w/o litter','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'nmintot','total nitrogen mineralization','kg/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(ayanlit, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totanlit',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totanlit'
    stop 1
  end if
  call vec2arr(aybnlit, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totrnlit',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totrnlit'
    stop 1
  end if
  call vec2arr(aynsoi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totnsoi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totnsoi'
    stop 1
  end if
  call vec2arr(aynmintot, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'nmintot',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, nmintot'
    stop 1
  end if

  ! co2 carbon exchange: neetot, co2mic, co2root, co2soi
  filen = 'output/yearly/co2fluxes.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual total carbon from exchange of co2','ibis wyearly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'neetot','total net ecosystem echange carbon','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'co2mic','total microbe respiration carbon','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'co2root','total root respiration carbon','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'co2soi','total soil respiration carbon','kg/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(ayneetot, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'neetot',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, neetot'
    stop 1
  end if
  call vec2arr(ayco2mic, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'co2mic',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, co2mic'
    stop 1
  end if
  call vec2arr(ayco2root, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'co2root',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, co2root'
    stop 1
  end if
  call vec2arr(ayco2soi, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'co2soi',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, co2soi'
    stop 1
  end if

  ! fire disturbance regime
  filen = 'output/yearly/disturbf.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual fire disturbance regime','ibis wyearly',cdate,nlonsub,lonscale, &
      nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'disturbf','fire disturbance regime','fraction/year',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(disturbf, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'disturbf',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, disturbf'
    stop 1
  end if

  ! vegetation type
  filen = 'output/yearly/vegtype0.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual vegetation type - ibis classification','ibis wyearly',cdate, &
      nlonsub,lonscale,nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'vegtype0','vegetation type','_',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(vegtype0, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'vegtype0',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, vegtype0'
    stop 1
  end if

  ! fractional cover of upper and lower canopies
  filen = 'output/yearly/fcover.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual fractional cover of canopies','ibis wyearly',cdate, &
      nlonsub,lonscale,nlatsub,latscale,'','none','none',1,[0.0_wp],'',tunits,'gregorian',istat)
    dimnames(3) = 'time'
    call inivar(idies,'fu','fractional cover of upper canopy','fraction',3,dimnames,ocean_fill,istat)
    call inivar(idies,'fl','fractional cover of lower canopy','fraction',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  call vec2arr(fu, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'fu',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, fu'
    stop 1
  end if
  call vec2arr(fl, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'fl',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, fl'
    stop 1
  end if

  ! lai, by pft, total upper canopy, total lower canopy
  filen = 'output/yearly/plai.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual leaf area index','ibis wyearly',cdate, &
      nlonsub,lonscale,nlatsub,latscale,'pft','plant fuctional type','_', &
      npft,pindex,'',tunits,'gregorian',istat)
    istat = NF_PUT_ATT_TEXT(idies,NF_GLOBAL,'pft_definition',npft*80,pftdef)
    dimnames(3) = 'pft'
    call inivar(idies,'plai','leaf area index for each pft','fraction',4,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totlaiu','total lai for upper canopy','fraction',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totlail','total lai for lower canopy','fraction',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, npft
    call vec2arr(plai(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  istart(3) = 1
  istart(4) = mstep
  icount(3) = npft
  call writevar(filen,'plai',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, plai'
    stop 1
  end if
  call vec2arr(totlaiu, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totlaiu',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totlaiu'
    stop 1
  end if
  call vec2arr(totlail, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totlail',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totlail'
    stop 1
  end if

  ! biomass, by pft, upper canopy, lower canopy
  filen = 'output/yearly/biomass.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual biomass of carbon','ibis wyearly',cdate, &
      nlonsub,lonscale,nlatsub,latscale,'pft','plant fuctional type','_', &
      npft,pindex,'',tunits,'gregorian',istat)
    istat = NF_PUT_ATT_TEXT(idies,NF_GLOBAL,'pft_definition',npft*80,pftdef)
    dimnames(3) = 'pft'
    call inivar(idies,'biomass','biomass for each pft','kg/m^2',4,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totbiou','total biomass for upper canopy','kg/m^2',3,dimnames,ocean_fill,istat)
    dimnames(3) = 'time'
    call inivar(idies,'totbiol','total biomass for lower canopy','kg/m^2',3,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, npft
    call vec2arr(biomass(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  istart(3) = 1
  istart(4) = mstep
  icount(3) = npft
  call writevar(filen,'biomass',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, biomass'
    stop 1
  end if
  call vec2arr(totbiou, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totbiou',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totbiou'
    stop 1
  end if
  call vec2arr(totbiol, cdummy)
  istart(3) = mstep
  icount(3) = 1
  call writevar(filen,'totbiol',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, totbiol'
    stop 1
  end if

  ! bottom and top of vegetation canopies
  filen = 'output/yearly/zcanopy.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual height of vegetation canopies','ibis wyearly',cdate, &
      nlonsub,lonscale,nlatsub,latscale,'canopy','canopy','_',2,pindex,'up', &
      tunits,'gregorian',istat)
    istat = NF_PUT_ATT_TEXT(idies,NF_GLOBAL,'canopy_def', &
      26+5,'1='//canopies(1)//' 2='//canopies(2))
    dimnames(3) = 'canopy'
    call inivar(idies,'zbot','bottom heights of lower and upper canopies', &
      'meters',4,dimnames,ocean_fill,istat)
    call inivar(idies,'ztop','top heights of lower and upper canopies', &
      'meters',4,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, 2
    call vec2arr(zbot(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  istart(3) = 1
  istart(4) = mstep
  icount(3) = 2
  call writevar(filen,'zbot',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, zbot'
    stop 1
  end if
  do k = 1, 2
    call vec2arr(ztop(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  istart(3) = 1
  istart(4) = mstep
  icount(3) = 2
  call writevar(filen,'ztop',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, ztop'
    stop 1
  end if

  ! existence of pfts
  filen = 'output/yearly/exist.nc'
  if (mstep == 1) then
    call inifile(idies,filen, &
      'annual existence of each plant functional type','ibis wyearly',cdate, &
      nlonsub,lonscale,nlatsub,latscale,'pft','plant fuctional type','_', &
      npft,pindex,'',tunits,'gregorian',istat)
    istat = NF_PUT_ATT_TEXT(idies,NF_GLOBAL,'pft_definition',npft*80,pftdef)
    dimnames(3) = 'pft'
    call inivar(idies,'exist','existence for each pft','_',4,dimnames,ocean_fill,istat)
    call endini(idies,istat)
  end if
  do k = 1, npft
    call vec2arr(exist(:,k), cdummy((k-1)*nlonsub*nlatsub + 1))
  end do
  istart(3) = 1
  istart(4) = mstep
  icount(3) = npft
  call writevar(filen,'exist',istart,icount,cdummy,ftime,tweight,tdate,istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in wyearly, exist'
    stop 1
  end if

end subroutine wyearly


! -----------------------------------------------------------------------------
subroutine readit(isimveg, snorth, ssouth, swest, seast, iwest, jnorth)
! Reads in initialization files and initializes some fields.
! Allocates cdummy to full I/O scratch size (ndim3) on first call.
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_dimensions,       only: nlon, nlat, npoi, nsoilay, xres, yres
  use mod_constants,        only: pi
  use mod_run_params,       only: nlonsub, nlatsub, garea
  use mod_work,             only: work, cdummy, ndim3, ndim4, lonscale, latscale, &
                                   lonindex, latindex
  use mod_boundary_climate, only: lmask, xintopo, xinveg, deltat, &
                                   xint, xinprec, xintrng, xincld, xinq, xinwet, xinwind, &
                                   clmt, clmtrng, clmprec, clmcld, clmq, clmwet
  use mod_soil_state,       only: sand, clay
  implicit none

  integer,  intent(in)    :: isimveg
  real(wp), intent(in)    :: snorth, ssouth, swest, seast
  integer,  intent(inout) :: iwest, jnorth

  integer  :: istat, i, j, ntime, jj, ii, ndim, nlpoints
  integer  :: jsouth, ieast
  integer  :: istart(4), icount(4)
  real(wp) :: xlat
  real(wp) :: xmask(nlon, nlat)   ! replaces equivalence(xmask(1,1),work(1))
  character(len=80) :: filen, aname

  ! Allocate I/O scratch buffer on first use
  if (.not. allocated(cdummy)) allocate(cdummy(ndim3))

  istart = [1, 1, 1, 1]
  icount = [nlon, nlat, 1, 1]
  icount(3) = 1

  ! land mask, latitudes, longitudes
  filen = 'input/surta.nc'
  aname = 'surta'
  call readvar(filen, aname, 'level', istart, icount, &
    xmask, lonscale, latscale, cdummy(1), cdummy(ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading surta'
    stop 1
  end if

  if (abs(abs(lonscale(1)-lonscale(2))-xres) > 0.001_wp .or. &
      abs(abs(latscale(1)-latscale(2))-yres) > 0.001_wp) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    write(*,*) 'resolution mismatch!'
    write(*,*) 'xres, yres in mod_dimensions = ', xres, yres
    write(*,*) 'xres, yres from input/surta.nc = ', &
      abs(lonscale(1)-lonscale(2)), abs(latscale(1)-latscale(2))
    stop 1
  end if

  ! subset the grid if not default whole grid
  if (snorth < latscale(1) .or. ssouth > latscale(nlat) .or. &
      swest > lonscale(1)  .or. seast  < lonscale(nlon)) then
    jnorth = 0
    if (snorth < (latscale(nlat)+latscale(nlat-1))/2.0_wp) then
      jnorth = nlat
    else if (snorth >= (latscale(1)+latscale(2))/2.0_wp) then
      jnorth = 1
    else
      do j = nlat-1, 1, -1
        if (snorth >= (latscale(j)+latscale(j+1))/2.0_wp) jnorth = j
      end do
    end if
    jsouth = 0
    if (ssouth < (latscale(nlat)+latscale(nlat-1))/2.0_wp) then
      jsouth = nlat
    else if (ssouth >= (latscale(1)+latscale(2))/2.0_wp) then
      jsouth = 1
    else
      do j = nlat-1, 1, -1
        if (ssouth >= (latscale(j)+latscale(j+1))/2.0_wp) jsouth = j
      end do
    end if
    iwest = 0
    if (swest < (lonscale(1)+lonscale(2))/2.0_wp) then
      iwest = 1
    else if (swest >= (lonscale(nlon)+lonscale(nlon-1))/2.0_wp) then
      iwest = nlon
    else
      do i = 2, nlon
        if (swest >= (lonscale(i)+lonscale(i-1))/2.0_wp) iwest = i
      end do
    end if
    ieast = 0
    if (seast < (lonscale(1)+lonscale(2))/2.0_wp) then
      ieast = 1
    else if (seast >= (lonscale(nlon)+lonscale(nlon-1))/2.0_wp) then
      ieast = nlon
    else
      do i = 2, nlon
        if (seast >= (lonscale(i)+lonscale(i-1))/2.0_wp) ieast = i
      end do
    end if
    nlonsub = ieast - iwest + 1
    nlatsub = jsouth - jnorth + 1
    istart(1) = iwest
    icount(1) = nlonsub
    istart(2) = jnorth
    icount(2) = nlatsub
  else
    iwest  = 1
    ieast  = nlon
    jnorth = 1
    jsouth = nlat
    nlonsub = nlon
    nlatsub = nlat
  end if

  ! initialize lonindex, latindex; compute approximate grid-cell areas
  nlpoints = 0
  do j = jnorth, jsouth
    jj = j - jnorth + 1
    do i = iwest, ieast
      ii = i - iwest + 1
      lmask(ii,jj) = nint(xmask(i,j))
      if (lmask(ii,jj) == 1) then
        nlpoints = nlpoints + 1
        lonindex(nlpoints) = ii
        latindex(nlpoints) = jj
        xlat = latscale(j) * pi / 180.0_wp
        garea(nlpoints) = yres * 111400.0_wp * xres * 111400.0_wp * cos(xlat)
      end if
    end do
  end do

  ! compress lat/lon scales to sub-domain
  do j = jnorth, jsouth
    jj = j - jnorth + 1
    latscale(jj) = latscale(j)
  end do
  do i = iwest, ieast
    ii = i - iwest + 1
    lonscale(ii) = lonscale(i)
  end do

  if (nlpoints /= npoi) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    write(*,*) 'number of land points in input/surta.nc'
    write(*,*) 'does not match number of land points in mod_dimensions'
    write(*,*) 'in surta =', nlpoints, '  in mod_dimensions =', npoi
    stop 1
  else
    write(*,'(1x,a)')  ' '
    write(*,'(1x,a,i10)') 'number of land points: ', nlpoints
    write(*,'(1x,a)')  ' '
  end if

  ! topography
  filen = 'input/topo.nc'
  aname = 'topo'
  call readvar(filen, aname, 'level', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading topo'
    stop 1
  end if
  call arr2vec(cdummy, xintopo)

  ! fixed vegetation map
  if (isimveg <= 1) then
    filen = 'input/vegtype.nc'
    aname = 'vegtype'
    call readvar(filen, aname, 'level', istart, icount, &
      cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
    if (istat < 0) then
      write(*,'(1x,a)') 'ERROR in subroutine readit'
      print *, 'while reading vegtype'
      stop 1
    end if
    call arr2vec(cdummy, xinveg)
  end if

  ! delta t
  filen = 'input/deltat.nc'
  aname = 'deltat'
  call readvar(filen, aname, 'level', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading deltat'
    stop 1
  end if
  call arr2vec(cdummy, deltat)

  ! 3-d soil texture (6 layers)
  icount(3) = 6
  icount(4) = 1
  filen = 'input/soita.sand.nc'
  aname = 'sandpct'
  call readvar(filen, aname, 'layer', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading soita.sand'
    stop 1
  end if
  do j = 1, nsoilay
    call arr2vec(cdummy((j-1)*nlonsub*nlatsub + 1), sand(:,j))
  end do

  icount(3) = 6
  icount(4) = 1
  filen = 'input/soita.clay.nc'
  aname = 'claypct'
  call readvar(filen, aname, 'layer', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading soita.clay'
    stop 1
  end if
  do j = 1, nsoilay
    call arr2vec(cdummy((j-1)*nlonsub*nlatsub + 1), clay(:,j))
  end do

  ! 3-d climate arrays (12 months)
  icount(3) = 1
  icount(4) = 12

  filen = 'input/wetd.mon.nc'
  aname = 'wetd'
  call readvar(filen, aname, 'level', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading wetd'
    stop 1
  end if
  do ntime = 1, 12
    call arr2vec(cdummy((ntime-1)*nlonsub*nlatsub + 1), clmwet(:,ntime))
  end do

  filen = 'input/temp.mon.nc'
  aname = 'temp'
  call readvar(filen, aname, 'level', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading temp'
    stop 1
  end if
  do ntime = 1, 12
    call arr2vec(cdummy((ntime-1)*nlonsub*nlatsub + 1), clmt(:,ntime))
  end do

  filen = 'input/trange.mon.nc'
  aname = 'trange'
  call readvar(filen, aname, 'level', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading trange'
    stop 1
  end if
  do ntime = 1, 12
    call arr2vec(cdummy((ntime-1)*nlonsub*nlatsub + 1), clmtrng(:,ntime))
  end do

  filen = 'input/prec.mon.nc'
  aname = 'prec'
  call readvar(filen, aname, 'level', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading prec'
    stop 1
  end if
  do ntime = 1, 12
    call arr2vec(cdummy((ntime-1)*nlonsub*nlatsub + 1), clmprec(:,ntime))
  end do

  filen = 'input/wspd.mon.nc'
  aname = 'wspd'
  call readvar(filen, aname, 'level', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading wspd'
    stop 1
  end if
  do ntime = 1, 12
    call arr2vec(cdummy((ntime-1)*nlonsub*nlatsub + 1), xinwind(:,ntime))
  end do

  filen = 'input/cld.mon.nc'
  aname = 'cld'
  call readvar(filen, aname, 'level', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading cld'
    stop 1
  end if
  do ntime = 1, 12
    call arr2vec(cdummy((ntime-1)*nlonsub*nlatsub + 1), clmcld(:,ntime))
  end do

  filen = 'input/rh.mon.nc'
  aname = 'rh'
  call readvar(filen, aname, 'level', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat < 0) then
    write(*,'(1x,a)') 'ERROR in subroutine readit'
    print *, 'while reading rh'
    stop 1
  end if
  do ntime = 1, 12
    call arr2vec(cdummy((ntime-1)*nlonsub*nlatsub + 1), clmq(:,ntime))
  end do

  ! copy all climatology fields to clm+anom fields for spin up
  ndim = npoi * 12
  call scopy(ndim, clmt,    xint)
  call scopy(ndim, clmtrng, xintrng)
  call scopy(ndim, clmprec, xinprec)
  call scopy(ndim, clmcld,  xincld)
  call scopy(ndim, clmq,    xinq)
  call scopy(ndim, clmwet,  xinwet)

end subroutine readit


! -----------------------------------------------------------------------------
subroutine restart(iyrlast)
! Reads restart files and re-initialises model state.
! iyrlast: last completed year (odd -> restart/odd, even -> restart/even).
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_dimensions,       only: npoi, nsnolay, nsoilay, npft
  use mod_run_params,       only: nlonsub, nlatsub
  use mod_work,             only: cdummy, work, ndim4
  use mod_snow_state,       only: fi, tsno, hsno
  use mod_soil_state,       only: tsoi, wsoi, wisoi, wipud, wpud
  use mod_carbon_pools,     only: cbiol, cbiow, cbior, &
                                   clitlm, clitls, clitll, &
                                   clitrm, clitrs, clitrl, &
                                   clitwm, clitws, clitwl, &
                                   csoislop, csoislon, csoipas, &
                                   falll, fallr, fallw
  use mod_veg_dynamics,     only: sapfrac, tc, tw, tcmin, gdd0, gdd5, &
                                   agddu, agddl, tempu, templ, &
                                   dropu, dropls, dropl4, dropl3, totcmic
  use mod_summary,          only: a10td, a10ancub, a10ancls, a10ancl4, &
                                   a10ancl3, a10scalparamu, a10scalparaml, &
                                   a10daylightu, a10daylightl
  use mod_boundary_climate, only: deltat
  implicit none

  integer, intent(in) :: iyrlast

  integer            :: istart(4), icount(4)
  character(len=20)  :: fdir
  character(len=80)  :: filen
  integer            :: lf, i, istat, nlevel

  istart = [1, 1, 1, 1]
  icount = [nlonsub, nlatsub, 1, 1]

  if (mod(iyrlast, 2) == 0) then
    fdir = 'restart/even'
  else
    fdir = 'restart/odd'
  end if
  lf = len_trim(fdir)

  ! fsnocov
  icount(3) = 1
  filen = fdir(1:lf)//'fsnocov.nc'
  call readvar(filen, 'fsnocov', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, fsnocov'
    stop 1
  end if
  call arr2vec(cdummy, fi)

  ! nsnolay variables: tsno and hsno
  icount(3) = nsnolay

  filen = fdir(1:lf)//'tsno.nc'
  call readvar(filen, 'tsno', 'snowlayer', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, tsno'
    stop 1
  end if
  do nlevel = 1, nsnolay
    call arr2vec(cdummy((nlevel-1)*nlonsub*nlatsub + 1), tsno(:,nlevel))
  end do

  filen = fdir(1:lf)//'hsno.nc'
  call readvar(filen, 'hsno', 'snowlayer', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, hsno'
    stop 1
  end if
  do nlevel = 1, nsnolay
    call arr2vec(cdummy((nlevel-1)*nlonsub*nlatsub + 1), hsno(:,nlevel))
  end do

  ! nsoilay variables: tsoi, wisoi, wsoi
  icount(3) = nsoilay

  filen = fdir(1:lf)//'tsoi.nc'
  call readvar(filen, 'tsoi', 'soillayer', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, tsoi'
    stop 1
  end if
  do nlevel = 1, nsoilay
    call arr2vec(cdummy((nlevel-1)*nlonsub*nlatsub + 1), tsoi(:,nlevel))
  end do

  filen = fdir(1:lf)//'wisoi.nc'
  call readvar(filen, 'wisoi', 'soillayer', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, wisoi'
    stop 1
  end if
  do nlevel = 1, nsoilay
    call arr2vec(cdummy((nlevel-1)*nlonsub*nlatsub + 1), wisoi(:,nlevel))
  end do

  filen = fdir(1:lf)//'wsoi.nc'
  call readvar(filen, 'wsoi', 'soillayer', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, wsoi'
    stop 1
  end if
  do nlevel = 1, nsoilay
    call arr2vec(cdummy((nlevel-1)*nlonsub*nlatsub + 1), wsoi(:,nlevel))
  end do

  ! npft variables
  icount(3) = npft

  filen = fdir(1:lf)//'cbiol.nc'
  call readvar(filen, 'cbiol', 'pft', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, cbiol'
    stop 1
  end if
  do nlevel = 1, npft
    call arr2vec(cdummy((nlevel-1)*nlonsub*nlatsub + 1), cbiol(:,nlevel))
  end do

  filen = fdir(1:lf)//'cbiow.nc'
  call readvar(filen, 'cbiow', 'pft', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, cbiow'
    stop 1
  end if
  do nlevel = 1, npft
    call arr2vec(cdummy((nlevel-1)*nlonsub*nlatsub + 1), cbiow(:,nlevel))
  end do

  filen = fdir(1:lf)//'cbior.nc'
  call readvar(filen, 'cbior', 'pft', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, cbior'
    stop 1
  end if
  do nlevel = 1, npft
    call arr2vec(cdummy((nlevel-1)*nlonsub*nlatsub + 1), cbior(:,nlevel))
  end do

  ! single level variables
  icount(3) = 1

  filen = fdir(1:lf)//'sapfrac.nc'
  call readvar(filen, 'sapfrac', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, sapfrac'
    stop 1
  end if
  call arr2vec(cdummy, sapfrac)

  filen = fdir(1:lf)//'clitlm.nc'
  call readvar(filen, 'clitlm', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, clitlm'
    stop 1
  end if
  call arr2vec(cdummy, clitlm)

  filen = fdir(1:lf)//'clitls.nc'
  call readvar(filen, 'clitls', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, clitls'
    stop 1
  end if
  call arr2vec(cdummy, clitls)

  filen = fdir(1:lf)//'clitll.nc'
  call readvar(filen, 'clitll', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, clitll'
    stop 1
  end if
  call arr2vec(cdummy, clitll)

  filen = fdir(1:lf)//'clitrm.nc'
  call readvar(filen, 'clitrm', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, clitrm'
    stop 1
  end if
  call arr2vec(cdummy, clitrm)

  filen = fdir(1:lf)//'clitrs.nc'
  call readvar(filen, 'clitrs', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, clitrs'
    stop 1
  end if
  call arr2vec(cdummy, clitrs)

  filen = fdir(1:lf)//'clitrl.nc'
  call readvar(filen, 'clitrl', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, clitrl'
    stop 1
  end if
  call arr2vec(cdummy, clitrl)

  filen = fdir(1:lf)//'clitwm.nc'
  call readvar(filen, 'clitwm', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, clitwm'
    stop 1
  end if
  call arr2vec(cdummy, clitwm)

  filen = fdir(1:lf)//'clitws.nc'
  call readvar(filen, 'clitws', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, clitws'
    stop 1
  end if
  call arr2vec(cdummy, clitws)

  filen = fdir(1:lf)//'clitwl.nc'
  call readvar(filen, 'clitwl', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, clitwl'
    stop 1
  end if
  call arr2vec(cdummy, clitwl)

  filen = fdir(1:lf)//'falll.nc'
  call readvar(filen, 'falll', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, falll'
    stop 1
  end if
  call arr2vec(cdummy, falll)

  filen = fdir(1:lf)//'fallr.nc'
  call readvar(filen, 'fallr', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, fallr'
    stop 1
  end if
  call arr2vec(cdummy, fallr)

  filen = fdir(1:lf)//'fallw.nc'
  call readvar(filen, 'fallw', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, fallw'
    stop 1
  end if
  call arr2vec(cdummy, fallw)

  filen = fdir(1:lf)//'totcmic.nc'
  call readvar(filen, 'totcmic', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, totcmic'
    stop 1
  end if
  call arr2vec(cdummy, totcmic)

  filen = fdir(1:lf)//'csoislop.nc'
  call readvar(filen, 'csoislop', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, csoislop'
    stop 1
  end if
  call arr2vec(cdummy, csoislop)

  filen = fdir(1:lf)//'csoislon.nc'
  call readvar(filen, 'csoislon', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, csoislon'
    stop 1
  end if
  call arr2vec(cdummy, csoislon)

  filen = fdir(1:lf)//'csoipas.nc'
  call readvar(filen, 'csoipas', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, csoipas'
    stop 1
  end if
  call arr2vec(cdummy, csoipas)

  filen = fdir(1:lf)//'gdd0.nc'
  call readvar(filen, 'gdd0', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, gdd0'
    stop 1
  end if
  call arr2vec(cdummy, gdd0)

  filen = fdir(1:lf)//'gdd5.nc'
  call readvar(filen, 'gdd5', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, gdd5'
    stop 1
  end if
  call arr2vec(cdummy, gdd5)

  filen = fdir(1:lf)//'tc.nc'
  call readvar(filen, 'tc', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, tc'
    stop 1
  end if
  call arr2vec(cdummy, tc)

  filen = fdir(1:lf)//'tw.nc'
  call readvar(filen, 'tw', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, tw'
    stop 1
  end if
  call arr2vec(cdummy, tw)

  filen = fdir(1:lf)//'wipud.nc'
  call readvar(filen, 'wipud', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, wipud'
    stop 1
  end if
  call arr2vec(cdummy, wipud)

  filen = fdir(1:lf)//'wpud.nc'
  call readvar(filen, 'wpud', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, wpud'
    stop 1
  end if
  call arr2vec(cdummy, wpud)

  filen = fdir(1:lf)//'agddu.nc'
  call readvar(filen, 'agddu', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, agddu'
    stop 1
  end if
  call arr2vec(cdummy, agddu)

  filen = fdir(1:lf)//'agddl.nc'
  call readvar(filen, 'agddl', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, agddl'
    stop 1
  end if
  call arr2vec(cdummy, agddl)

  filen = fdir(1:lf)//'tempu.nc'
  call readvar(filen, 'tempu', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, tempu'
    stop 1
  end if
  call arr2vec(cdummy, tempu)

  filen = fdir(1:lf)//'templ.nc'
  call readvar(filen, 'templ', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, templ'
    stop 1
  end if
  call arr2vec(cdummy, templ)

  filen = fdir(1:lf)//'a10td.nc'
  call readvar(filen, 'a10td', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, a10td'
    stop 1
  end if
  call arr2vec(cdummy, a10td)

  filen = fdir(1:lf)//'a10ancub.nc'
  call readvar(filen, 'a10ancub', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, a10ancub'
    stop 1
  end if
  call arr2vec(cdummy, a10ancub)

! a10ancuc is not used; its restart entry is skipped (per original comments)

  filen = fdir(1:lf)//'a10ancls.nc'
  call readvar(filen, 'a10ancls', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, a10ancls'
    stop 1
  end if
  call arr2vec(cdummy, a10ancls)

  filen = fdir(1:lf)//'a10ancl4.nc'
  call readvar(filen, 'a10ancl4', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, a10ancl4'
    stop 1
  end if
  call arr2vec(cdummy, a10ancl4)

  filen = fdir(1:lf)//'a10ancl3.nc'
  call readvar(filen, 'a10ancl3', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, a10ancl3'
    stop 1
  end if
  call arr2vec(cdummy, a10ancl3)

  filen = fdir(1:lf)//'a10scalparamu.nc'
  call readvar(filen, 'a10scalparamu', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, a10scalparamu'
    stop 1
  end if
  call arr2vec(cdummy, a10scalparamu)

  filen = fdir(1:lf)//'a10scalparaml.nc'
  call readvar(filen, 'a10scalparaml', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, a10scalparaml'
    stop 1
  end if
  call arr2vec(cdummy, a10scalparaml)

  filen = fdir(1:lf)//'a10daylightu.nc'
  call readvar(filen, 'a10daylightu', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, a10daylightu'
    stop 1
  end if
  call arr2vec(cdummy, a10daylightu)

  filen = fdir(1:lf)//'a10daylightl.nc'
  call readvar(filen, 'a10daylightl', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, a10daylightl'
    stop 1
  end if
  call arr2vec(cdummy, a10daylightl)

  filen = fdir(1:lf)//'dropu.nc'
  call readvar(filen, 'dropu', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, dropu'
    stop 1
  end if
  call arr2vec(cdummy, dropu)

  filen = fdir(1:lf)//'dropls.nc'
  call readvar(filen, 'dropls', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, dropls'
    stop 1
  end if
  call arr2vec(cdummy, dropls)

  filen = fdir(1:lf)//'dropl4.nc'
  call readvar(filen, 'dropl4', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, dropl4'
    stop 1
  end if
  call arr2vec(cdummy, dropl4)

  filen = fdir(1:lf)//'dropl3.nc'
  call readvar(filen, 'dropl3', '', istart, icount, &
    cdummy, work(1), work(ndim4), work(2*ndim4), work(3*ndim4), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in restart, dropl3'
    stop 1
  end if
  call arr2vec(cdummy, dropl3)

  ! compute tcmin from tc and seasonal temperature offset
  do i = 1, npoi
    tcmin(i) = tc(i) + deltat(i)
  end do

  call existence

end subroutine restart


! -----------------------------------------------------------------------------
subroutine coldstart()
! Initialises snow and soil state variables to cold-start default values.
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nsnolay, nsoilay
  use mod_snow_state, only: fi, tsno, hsno
  use mod_soil_state, only: tsoi, wsoi, wisoi
  implicit none

  call const(fi,    npoi,           0.0_wp)
  call const(hsno,  npoi * nsnolay, 0.0_wp)
  call const(tsno,  npoi * nsnolay, 273.16_wp)
  call const(tsoi,  npoi * nsoilay, 278.16_wp)
  call const(wsoi,  npoi * nsoilay, 0.50_wp)
  call const(wisoi, npoi * nsoilay, 0.00_wp)

end subroutine coldstart


! -----------------------------------------------------------------------------
subroutine rdanom(imonth, iyear, iyranom, nanom, iy2, istyr, iwest, jnorth)
! Reads monthly anomalies for month imonth+1 and adds them to climatology.
! Reads: temp, prec, cld, rh anomalies (trange/wspd/wetd disabled).
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_dimensions,       only: npoi
  use mod_run_params,       only: nlonsub, nlatsub
  use mod_work,             only: cdummy, work
  use mod_boundary_climate, only: xint, xinprec, xincld, xinq, &
                                   clmt, clmprec, clmcld, clmq
  implicit none

  integer, intent(in) :: imonth, iyear, iyranom, nanom, iy2, istyr, &
                          iwest, jnorth

  real(wp)           :: anom(npoi)
  integer            :: imon, iyr, istat, jyear
  integer            :: istart(4), icount(4)
  character(len=80)  :: filen, aname

  istart = [1, 1, 1, 1]
  icount = [1, 1, 1, 1]
  istart(1) = iwest
  istart(2) = jnorth
  icount(1) = nlonsub
  icount(2) = nlatsub
  icount(3) = 1
  icount(4) = 1

  ! determine which (if any) month to read
  ! if prior to November of the year before an anomaly year, return
  if (iyear < iyranom - 1) then
    return
  else if (iyear == iyranom - 1 .and. imonth < 11) then
    return

  ! if December of the final anomaly year and also the final run year,
  ! set January anomalies to zero (use climatology) and return
  else if (iyear == istyr + nanom - 1 .and. imonth == 12) then
    if (iyear == iy2) then
      print *, 'WARNING: last month of run; no anomalies for January'
      print *, 'Using climatologies for month year ='
      print *, imonth + 1, iyear + 1
      xint(:,1)    = clmt(:,1)
      xinprec(:,1) = clmprec(:,1)
      xincld(:,1)  = clmcld(:,1)
      xinq(:,1)    = clmq(:,1)
      return
    end if
  end if

  iyr  = iyear - istyr
  imon = imonth + 1
  if (imon == 13) then
    imon = 1
    iyr  = iyr + 1
  end if
  jyear = iyr / nanom
  istart(4) = (iyr - nanom * jyear) * 12 + imon

  if (iyr > 0 .and. (iyr - nanom * jyear) == 0) then
    print *, 'WARNING: Attempted to read past last month in anomaly file'
    print *, 'Looping back to the beginning of the file'
  end if

  if (istart(4) > 0) then
    print *, 'rdanom reading month year step ='
    print *, imon, iyr + istyr - nanom * jyear, istart(4)
  else
    print *, 'WARNING, anomalies begin in year ', istyr
    print *, 'Not reading in anomalies for month year ='
    print *, imon, iyr + istyr
    return
  end if

  ! mean temperature
  aname = 'temp'
  filen = 'input/anom/temp.danom.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdanom, temp'
    stop 1
  end if
  call arr2vec(work, anom)
  xint(:,imon) = clmt(:,imon) + anom

  ! precipitation rate
  aname = 'prec'
  filen = 'input/anom/prec.danom.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdanom, prec'
    stop 1
  end if
  call arr2vec(work, anom)
  xinprec(:,imon) = clmprec(:,imon) + anom

  ! cloudiness
  aname = 'cld'
  filen = 'input/anom/cld.danom.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdanom, cld'
    stop 1
  end if
  call arr2vec(work, anom)
  xincld(:,imon) = clmcld(:,imon) + anom

  ! relative humidity
  aname = 'rh'
  filen = 'input/anom/rh.danom.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdanom, rh'
    stop 1
  end if
  call arr2vec(work, anom)
  xinq(:,imon) = clmq(:,imon) + anom

! trange, wind speed, and wet days anomalies disabled (retained for future use)

end subroutine rdanom


! -----------------------------------------------------------------------------
subroutine inird(file, istyr)
! Extracts the start year from the 'units' attribute of the 'time' variable
! in a NetCDF file and returns istyr = that year + 1.
! Uses low-level NetCDF-3 C interface via netcdf.inc.
! -----------------------------------------------------------------------------
  implicit none
  include 'netcdf.inc'

  character(len=*), intent(in)    :: file
  integer,          intent(inout) :: istyr

  integer           :: idies, istat, idtime, lf1
  character(len=80) :: units

  ! open file
  istat = NF_OPEN(file, NF_NOWRITE, idies)
  if (istat /= NF_NOERR) then
    print *, 'Error in inird while trying to open file'
    print *, file
    print *, NF_STRERROR(istat)
    istyr = -1
    return
  end if

  ! get variable id for time
  istat = NF_INQ_VARID(idies, 'time', idtime)
  if (istat /= NF_NOERR) then
    print *, 'Error in inird while trying to get time id'
    print *, NF_STRERROR(istat)
    istyr = -1
    return
  end if

  ! read units attribute
  units = ' '
  istat = NF_GET_ATT_TEXT(idies, idtime, 'units', units)
  if (istat /= NF_NOERR) then
    print *, 'Error in inird while trying to get time units'
    print *, NF_STRERROR(istat)
    istyr = -1
    return
  end if

  ! parse year from 'days since YYYY-...' and add 1
  lf1 = index(units, 'since') + 6
  read(units(lf1:lf1+3), '(i4)') istyr
  istyr = istyr + 1

end subroutine inird


! -----------------------------------------------------------------------------
subroutine rdday(jday, imonth, iyear, istyr, iwest, jnorth)
! Reads daily forcing fields for day jday of year iyear.
! Reads: prec, temp, trange, cld, wspd, sphum.
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_run_params,       only: nlonsub, nlatsub
  use mod_work,             only: cdummy, work
  use mod_boundary_climate, only: xinprecd, xintd, xintrngd, xincldd, &
                                   xinwindd, xinqd
  implicit none

  integer, intent(in) :: jday, imonth, iyear, istyr, iwest, jnorth

  integer           :: i, istat
  character(len=80) :: filen, aname
  integer           :: istart(4), icount(4)

  istart = [1, 1, 1, 1]
  icount = [1, 1, 1, 1]
  istart(1) = iwest
  istart(2) = jnorth
  icount(1) = nlonsub
  icount(2) = nlatsub
  icount(3) = 1
  icount(4) = 1

  if (iyear < istyr) then
    print *, 'daily data begins in year ', istyr
    print *, 'not reading in daily data'
    return
  end if

  ! count days since Jan 1 of istyr
  if (iyear == istyr) then
    istart(4) = jday
  else
    istart(4) = 0
    do i = istyr, iyear - 1
      istart(4) = istart(4) + 365
      if (mod(i, 4) == 0) then
        if (mod(i, 100) /= 0) then
          istart(4) = istart(4) + 1
        else if (mod(i / 100, 4) == 0) then
          istart(4) = istart(4) + 1
        end if
      end if
    end do
    istart(4) = istart(4) + jday
  end if

  if (istart(4) > 0) then
    print *, 'rdday reading day month year step ='
    print *, jday, imonth, iyear + istyr, istart(4)
  else
    print *, 'WARNING, anomalies begin in year ', istyr
    print *, 'Not reading in anomalies for day month year ='
    print *, jday, imonth, iyear + istyr
    return
  end if

  ! daily precip
  aname = 'prec'
  filen = 'input/anom/daily/prec.fanom.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdday, prec'
    if (iyear > 1997) print *, 'Attempted to read past last day in file?'
    return
  end if
  call arr2vec(work, xinprecd)

  ! daily temperature
  aname = 'temp'
  filen = 'input/anom/daily/temp.danom.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdday, temp'
    if (iyear > 1997) print *, 'Attempted to read past last day in file?'
    return
  end if
  call arr2vec(work, xintd)

  ! daily temperature range
  aname = 'trange'
  filen = 'input/anom/daily/trange.fanomc.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdday, trange'
    if (iyear > 1997) print *, 'Attempted to read past last day in file?'
    return
  end if
  call arr2vec(work, xintrngd)

  ! daily cloudiness
  aname = 'cld'
  filen = 'input/anom/daily/cld.danom.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdday, cld'
    if (iyear > 1997) print *, 'Attempted to read past last day in file?'
    return
  end if
  call arr2vec(work, xincldd)

  ! daily windspeed
  aname = 'wspd'
  filen = 'input/anom/daily/wspd.fanomc.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdday, wspd'
    if (iyear > 1997) print *, 'Attempted to read past last day in file?'
    return
  end if
  call arr2vec(work, xinwindd)

  ! daily specific humidity
  aname = 'sphum'
  filen = 'input/anom/daily/sphum.fanom.nc'
  call readvar(filen, aname, 'level', istart, icount, &
    work, cdummy(1), cdummy(nlonsub+1), cdummy(2*nlonsub+1), &
    cdummy(3*nlonsub+1), istat)
  if (istat /= 0) then
    write(*,*) 'ERROR in rdday, sphum'
    if (iyear > 1997) print *, 'Attempted to read past last day in file?'
    return
  end if
  call arr2vec(work, xinqd)

end subroutine rdday


! -----------------------------------------------------------------------------
subroutine simdate(cdate)
! Returns a Y2K-compliant date string in dd-Mon-yyyy format (11 characters).
! Uses standard DATE_AND_TIME intrinsic (F2003+).  Example: '20-Feb-2026'
! -----------------------------------------------------------------------------
  implicit none

  character(len=11), intent(out) :: cdate

  character(len=8) :: idate
  integer          :: im

  character(len=3), parameter :: month_name(12) = &
    ['Jan','Feb','Mar','Apr','May','Jun', &
     'Jul','Aug','Sep','Oct','Nov','Dec']

  call DATE_AND_TIME(date=idate)   ! idate = 'YYYYMMDD'
  read(idate(5:6), '(I2)') im
  cdate = idate(7:8)//'-'//month_name(im)//'-'//idate(1:4)

end subroutine simdate
