! =============================================================================
! main.f90  --  IBIS top-level program and land surface physics dispatcher
!
! Contains:
!   program main    -- simulation controller (reads ibis.infile, time loops)
!   subroutine lsxmain -- land surface exchange dispatcher
!
! Converted from ibis_2.6b4/main.f (F77 fixed-form) to F2008 free-form.
! COMMON blocks (compar.h, comatm.h, comdiag.h) replaced by USE associations.
! =============================================================================

! =============================================================================
program main
! =============================================================================

  use mod_precision,   only: wp
  use mod_run_params,  only: dtime, ndaypy, ndaypm
  use mod_dimensions,  only: nlon, nlat, xres, yres
  use mod_atm_state,   only: co2conc, o2conc
  use mod_diagnostics, only: diagstart, diagend, ndiagpt, nfreq
  implicit none

  ! ---------------------------------------------------------------------------
  ! Local variables
  ! ---------------------------------------------------------------------------

  integer :: istep        ! timestep counter (per day)
  integer :: iday         ! daily loop counter
  integer :: imonth       ! monthly loop counter
  integer :: istyrd       ! first year daily means exist (from precip file)
  integer :: istyrm       ! first year monthly anomalies exist (from file)
  integer :: iwest        ! 1st longitude index for domain subset
  integer :: iy1          ! first year of year loop
  integer :: iy2          ! last year of year loop
  integer :: iyear        ! yearly loop counter
  integer :: iyear0       ! first year of simulation
  integer :: iyranom      ! year to start reading monthly anomalies
  integer :: iyrdaily     ! year to start reading daily means
  integer :: iyrlast      ! last year completed in this (or previous) run
  integer :: idiag        ! number of diagnostic files requested
  integer :: idailyout    ! 0: no daily output;  1: write daily output
  integer :: imonthout    ! 0: no monthly output; 1: write monthly output
  integer :: iyearout     ! 0: no yearly output;  1: write yearly output
  integer :: irestart     ! 0: cold start;  1: restart mode
  integer :: isimveg      ! 0: static veg; 1: dyn veg w/ fixed init; 2: cold start
  integer :: isimfire     ! 0: fixed fire; 1: dynamic fire
  integer :: isimco2      ! 0: fixed CO2;  1: time-varying CO2
  integer :: jday         ! Julian day of the simulation year
  integer :: jnorth       ! 1st latitude index for domain subset
  integer :: nanom        ! number of years in the anomaly files
  integer :: niter        ! total number of time steps per day
  integer :: nday         ! number of days since start of simulation
  integer :: nrun         ! number of years in this run
  integer :: nspinsoil    ! year when soil C reaches equilibrium
  integer :: plen         ! length of precipitation event (timesteps)
  integer :: plenmax      ! upper bound on plen
  integer :: plenmin      ! lower bound on plen
  integer :: seed         ! random number generator seed
  integer :: spin         ! spinup iteration counter
  integer :: spinmax      ! maximum spinup iterations per day
  integer :: eqyears      ! years past spinup to bring soil to equilibrium
  integer :: soilcspin    ! 0: no soil C spinup; 1: use acceleration procedure
  integer :: lun          ! Fortran unit number for ibis.infile
  integer :: i            ! loop index
  integer :: io_stat      ! IOSTAT flag for I/O error handling

  real(kind(1.0d0)) :: co2init     ! initial CO2 concentration (mol mol-1)
  real(kind(1.0d0)) :: o2init      ! initial O2 concentration (mol mol-1)
  real(kind(1.0d0)) :: plens       ! length of precipitation event (s)
  real(kind(1.0d0)) :: startp      ! precipitation event start time (s since midnight)
  real(kind(1.0d0)) :: endp        ! precipitation event end time (s)
  real(kind(1.0d0)) :: spincons    ! soilbgc calls per day during spinup
  real(kind(1.0d0)) :: spinfrac    ! fraction of nspinsoil at max spinup rate
  real(kind(1.0d0)) :: slope       ! rate of decrease of spinup iterations
  real(kind(1.0d0)) :: test        ! test value for timestep divisibility check
  real(kind(1.0d0)) :: time        ! time of day since midnight (s)

  real :: dran            ! random number from ran2 (returns default real)
  real(wp) :: snorth      ! north latitude bound for domain subset
  real(wp) :: ssouth      ! south latitude bound for domain subset
  real(wp) :: swest       ! west longitude bound for domain subset
  real(wp) :: seast       ! east longitude bound for domain subset

  real, external :: ran2  ! Numerical Recipes random number generator

  ! Format statements
1000 format (1x,'starting - year/month/day: ',i4,'/',i2,'/',i2)
9000 format (1x,'length of this simulation (years)   : ',i8)
9005 format (1x,'year to begin using anomalies       : ',i8)
9006 format (1x,'number of iterations per day        : ',i8)
9010 format (1x,'model lon, lat resolution (degrees) : ',2f8.2)
9020 format (1x,'model domain (nlon x nlat)          : ',i3,' by ',i3)
9030 format (1x,'last year run in this sequence      : ',i8)

  ! ---------------------------------------------------------------------------
  ! Default domain subsetting bounds (global)
  ! ---------------------------------------------------------------------------
  snorth =  90.0_wp
  ssouth = -90.0_wp
  swest  = -180.0_wp
  seast  =  180.0_wp

  ! ---------------------------------------------------------------------------
  ! Job control: read ibis.infile
  ! ---------------------------------------------------------------------------
  lun = 12
  open (lun, status='old', file='ibis.infile')
  read (lun,*) irestart
  read (lun,*) iyear0
  read (lun,*) nrun
  read (lun,*) iyranom
  read (lun,*) nanom
  read (lun,*) iyrdaily
  read (lun,*) soilcspin
  read (lun,*) iyearout
  read (lun,*) imonthout
  read (lun,*) idailyout
  read (lun,*) isimveg
  read (lun,*) isimfire
  read (lun,*) isimco2
  read (lun,*) co2init
  read (lun,*) o2init
  read (lun,*) dtime
  read (lun,*) idiag
  read (lun,*,end=99) snorth
  read (lun,*,end=99) ssouth
  read (lun,*,end=99) swest
  read (lun,*,end=99) seast
99 close (lun)

  ! Tell user about the simulation
  write (*,*) ' '
  write (*,*) '****************************************'
  write (*,*) '* IBIS: Integrated BIosphere Simulator *'
  write (*,*) '* Version 2.6b3                        *'
  write (*,*) '* March 2002                           *'
  write (*,*) '****************************************'
  write (*,*) ' '

  if (irestart == 1) then
    write (*,*) 'running in restart mode'
  end if

  write (*,*) ' '
  write (*,9000) nrun
  write (*,9005) iyranom
  write (*,*) ' '
  write (*,9010) xres, yres
  write (*,9020) nlon, nlat
  write (*,*) ' '

  ! ---------------------------------------------------------------------------
  ! Time control
  ! ---------------------------------------------------------------------------

  ! Determine number of timesteps per day
  niter = int(86400.0d0 / dtime)

  ! Verify that dtime divides evenly into 86400 s
  test = mod(86400.0d0, dtime)
  if (test > 1.0d-20) then
    write (*,*) 'dtime ', dtime, ' should be divisible into 86400'
    stop
  else
    write (*,9006) niter
  end if

  ! ---------------------------------------------------------------------------
  ! Check for restart and set iyrlast
  ! ---------------------------------------------------------------------------
  if (irestart == 1) then
    open (13, status='old', file='yearsrun.dat', iostat=io_stat)
    if (io_stat /= 0) then
      write (*,*) 'warning: ibis.infile restart flag=1, indicating ' // &
                  'restart, but yearsrun.dat not found'
      write (*,*) 'beginning from ', iyear0
      iyrlast = iyear0 - 1
    else
      read (13,*) iyrlast
      close (13)
    end if
    open (20, file='ibis.out.global',  status='unknown', position='append')
    open (30, file='ibis.out.vegtype', status='unknown', position='append')
  else
    iyrlast = iyear0 - 1
  end if

  write (*,9030) nrun + iyrlast
  write (*,*) ' '

  ! ---------------------------------------------------------------------------
  ! Initialization
  ! ---------------------------------------------------------------------------

  ! Initialize calendar (ndaypm(2) adjusted each year for leap years)
  ndaypm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  ndaypy = 365

  ! Read all parameter files
  call rd_param

  ! Set initial atmospheric gas concentrations
  co2conc = co2init
  o2conc  = o2init

  ! Read global boundary condition datasets
  call readit(isimveg, snorth, ssouth, swest, seast, iwest, jnorth)

  ! Check for diagnostic output; read diag.infile if needed
  if (idiag /= 0) call inidiag(idiag)

  ! Preliminary climate analysis
  if (irestart == 0) then
    call climanl
  end if

  ! Initialize monthly anomalies and daily means if needed
  if (iyranom <= iyrlast + nrun) then
    call inird('input/anom/temp.danom.nc', istyrm)
    if (istyrm <= 0) istyrm = 9999
  end if

  if (iyrdaily <= iyrlast + nrun) then
    call inird('input/anom/daily/prec.fanom.nc', istyrd)
    if (istyrd <= 0) istyrd = 9999
  end if

  ! Pre-load anomalies for the transition into the first anomaly year
  iy2 = iyrlast + nrun
  if (iyrlast + 1 >= iyranom) then
    call rdanom(11, iyrlast, iyranom, nanom, iy2, istyrm, iwest, jnorth)
    call rdanom(12, iyrlast, iyranom, nanom, iy2, istyrm, iwest, jnorth)
  end if

  ! Initialize the model state
  call initial(isimveg, irestart, iyrlast)

  ! Seed the random number generator
  seed = -1
  dran = ran2(seed)

  ! ---------------------------------------------------------------------------
  ! Start of time loops
  ! ---------------------------------------------------------------------------
  write (*,*) ' '
  write (*,*) '********************************'
  write (*,*) '* start of the IBIS simulation *'
  write (*,*) '********************************'
  write (*,*) ' '

  ! Elapsed day counter (accumulate past days in restart mode)
  nday = 0
  if (irestart == 1) then
    do iyear = iyear0, iyrlast
      nday = nday + 365
      if (mod(iyear,4) == 0) then
        if (mod(iyear,100) /= 0) then
          nday = nday + 1
        else if (mod(iyear/100,4) == 0) then
          nday = nday + 1
        end if
      end if
    end do
  end if

  ! Yearly loop
  iy1 = iyrlast + 1
  iy2 = iyrlast + nrun

  do iyear = iy1, iy2

    ! Reset Julian date
    jday = 0

    ! Determine the calendar for this year (Gregorian leap year rules)
    ndaypm(2) = 28
    ndaypy    = 365
    if (mod(iyear,4) == 0) then
      if (mod(iyear,100) /= 0) then
        ndaypm(2) = 29
        ndaypy    = 366
      else if (mod(iyear/100,4) == 0) then
        ndaypm(2) = 29
        ndaypy    = 366
      end if
    end if

    ! Monthly loop
    do imonth = 1, 12

      call rdanom(imonth, iyear, iyranom, nanom, iy2, istyrm, iwest, jnorth)

      ! Daily loop
      do iday = 1, ndaypm(imonth)

        write (*,1000) iyear, imonth, iday

        ! Update elapsed days and Julian day
        nday = nday + 1
        jday = jday + 1

        ! Determine today's climatic conditions
        if (iyear >= iyrdaily) then
          call rdday(jday, imonth, iyear, istyrd, iwest, jnorth)
          call daily(iyear, imonth, iday, seed, 1, iyranom, iyrlast, nrun)
        else
          call daily(iyear, imonth, iday, seed, 0, iyranom, iyrlast, nrun)
        end if

        ! Determine daily vegetation phenology
        call pheno

        ! Soil biogeochemistry (with optional spin-up acceleration)
        spinfrac = 0.75d0   ! fraction of nspinsoil at max spinup rate
        spincons = 40.0d0   ! soilbgc calls per day during acceleration
        eqyears  = 50       ! years past spinup to reach equilibrium
        nspinsoil = iyear0 + 150  ! year soil C reaches equilibrium

        if (soilcspin == 1) then
          if ((iyear - iyear0) <= &
              (spinfrac * (nspinsoil - iyear0 - eqyears))) then
            spinmax = int(spincons)
          else if ((iyear - iyear0) < &
                   (nspinsoil - iyear0 - eqyears)) then
            slope   = spincons / ((nspinsoil - iyear0 - eqyears) - &
                                  (spinfrac * (nspinsoil - iyear0 - eqyears)))
            spinmax = int(spincons - (slope * ((iyear - iyear0) - &
                                     (spinfrac * (nspinsoil - iyear0 - eqyears)))))
            spinmax = max(spinmax, 1)
          else
            spinmax = 1
          end if
        else
          spinmax = 1
        end if

        do spin = 1, spinmax
          call soilbgc(iyear, iyear0, imonth, iday, nspinsoil, spin, spinmax)
        end do

        ! Determine precipitation event timing
        plenmin = 1 + int((4.0d0 * 3600.0d0 - 1.0d0) / dtime)
        plenmax = max(int(24.0d0 * 3600.0d0 / dtime), plenmin)
        plen    = min(plenmax, int(plenmin + ran2(seed) * (plenmax - plenmin + 1)))
        plens   = dtime * plen
        startp  = dtime * min(niter - plen, int(ran2(seed) * (niter - plen + 1)))
        endp    = startp + plens

        ! Timestep loop
        do istep = 1, niter

          ! Time of day since midnight (s)
          time = (istep - 1) * dtime

          ! Determine sub-daily climatic forcing
          call diurnal(time, jday, plens, startp, endp, seed)

          ! Call the land surface model
          call lsxmain

          ! Accumulate summary variables
          call sumnow
          call sumday(istep, plens)
          call summonth(istep, iday, imonth)
          call sumyear(istep, iday, imonth)

          ! Write diagnostic output
          if (idiag /= 0) then
            do i = 1, idiag
              if (iyear >= diagstart(i) .and. &
                  iyear <= diagend(i)   .and. &
                  ndiagpt(i) >= 1) then
                if (mod(istep, nfreq(i)) == 0) then
                  call wdiag(i, iyear, imonth, iday, istep)
                end if
              end if
            end do
          end if

        end do  ! timestep loop

        ! Write daily output
        if (idailyout == 1) then
          write (*,*) ' '
          write (*,*) 'writing daily output'
          write (*,*) ' '
          call wdaily(nday, iyear, iyear0)
        end if

      end do  ! daily loop

      ! Write monthly output (use 1st day of month for GrADS compatibility)
      if (imonthout == 1) then
        write (*,*) ' '
        write (*,*) 'writing monthly output'
        write (*,*) ' '
        call wmonthly(nday - ndaypm(imonth) + 1, imonth, iyear, iyear0)
      end if

    end do  ! monthly loop

    ! Recalculate bioclimatic parameters when using anomalies
    if (iyear >= iyranom - 1 .or. iyear >= iyrdaily) call climanl2

    ! Update atmospheric CO2 concentration
    if (isimco2 == 1) call co2(co2init, co2conc, iyear)

    ! Perform vegetation dynamics
    if (isimveg /= 0) call dynaveg(isimfire)

    ! Calculate global and per-vegetation-type annual diagnostics
    call gdiag(iyear, iyear0)
    call vdiag(iyear, iyear0)

    ! Write annual output
    if (iyearout == 1) then
      write (*,*) ' '
      write (*,*) 'writing annual output'
      write (*,*) ' '
      call wyearly(nday, iyear, iyear0)
    end if

    ! Write restart files
    call wrestart(nday, iyear, iyear0)

    ! Update iyrlast and yearsrun.dat
    iyrlast = iyrlast + 1
    open  (12, file='yearsrun.dat', status='unknown')
    write (12,*) iyrlast
    close (12)

  end do  ! yearly loop

  write (*,*) ' '
  write (*,*) '*** end of run ***'
  write (*,*) ' '

  stop

end program main

! =============================================================================
subroutine lsxmain
! =============================================================================
! Land surface exchange main dispatcher.  Called once per timestep from
! program main.  Calls all physics routines in the correct order.
! =============================================================================

  use mod_dimensions,   only: npoi, nband
  use mod_canopy_fluxes, only: solg, firg, fseng, fvapg, &
                                soli, firi, fseni, fvapi
  use mod_soil_state,   only: heatg, heati, hvasug, hvasui
  implicit none

  integer :: ib  ! waveband index (1 = visible, 2 = near-IR)
  integer :: i   ! land-point loop index

  ! Set physical soil quantities
  call setsoi

  ! Calculate areal fractions wetted by intercepted water
  call fwetcal

  ! Set up solar geometry
  call solset

  ! Solar calculations for each waveband
  do ib = 1, nband
    call solsur(ib)   ! surface albedos for soil and snow
    call solalb(ib)   ! canopy albedo calculations
    call solarf(ib)   ! absorbed solar fluxes and incident PAR
  end do

  ! Calculate longwave IR fluxes
  call irrad

  ! Step intercepted water (drip/blowoff)
  call cascade

  ! Re-calculate wetted fractions after cascade
  call fwetcal

  ! Step canopy temperatures implicitly; compute sensible/latent heat fluxes
  call canopy

  ! Step intercepted water due to evaporation
  call cascad2

  ! Set vegetation temperatures and intercepted water for no-veg locations
  call noveg

  ! Compute net surface heat fluxes for soil and snow models
  do i = 1, npoi
    heatg(i) = solg(i) + firg(i) - fseng(i) - hvasug(i) * fvapg(i)
    heati(i) = soli(i) + firi(i) - fseni(i) - hvasui(i) * fvapi(i)
  end do

  ! Step snow model
  call snow

  ! Step soil model
  call soilctl

end subroutine lsxmain
