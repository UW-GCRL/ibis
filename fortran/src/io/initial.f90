! =============================================================================
! initial.f90  --  IBIS model initialisation routines
!
! Converted from ibis_2.6b4/initial.f (F77 fixed-form) to F2008 free-form.
!
! Contains:
!   subroutine initial(isimveg, irestart, iyrlast)   -- top-level dispatcher
!   subroutine inisurf(irestart)                      -- physical constants + state arrays
!   subroutine inisum                                  -- diagnostic accumulator arrays
!   subroutine inisnow                                 -- snow model scalars
!   subroutine inisoil(irestart)                       -- soil texture + properties
!   integer function textcls(msand, mclay)             -- USDA texture classifier
!   logical function inpoly(poly, npoints, xt, yt)     -- point-in-polygon test
!   subroutine iniveg(isimveg, irestart)               -- vegetation initialisation
!
! Replaces: ibis_2.6b4/initial.f
! =============================================================================

! =============================================================================
subroutine initial(isimveg, irestart, iyrlast)
! =============================================================================
! Top-level dispatcher: calls cold-start or restart reader, then all
! individual initialisation subroutines.
  implicit none
  integer, intent(in) :: isimveg   ! 0=static veg, 1=dyn veg, 2=dyn veg cold-start
  integer, intent(in) :: irestart  ! 0=initial run, 1=restart
  integer, intent(in) :: iyrlast   ! last year of previous run (restart only)

  if (irestart == 0) then
    call coldstart
  else
    call restart(iyrlast)
  end if

  call inisurf(irestart)
  call inisnow
  call inisoil(irestart)
  call iniveg(isimveg, irestart)
  call inisum

end subroutine initial

! =============================================================================
subroutine inisurf(irestart)
! =============================================================================
! Initialises physical constants (via constants_init) and zeroes all
! prognostic / diagnostic state arrays not set by coldstart/restart.
  use mod_precision,     only: wp
  use mod_constants,     only: constants_init
  use mod_run_params,    only: vzero
  use mod_dimensions,    only: npoi
  use mod_atm_state,     only: asurd, asuri, xstore, iwet, iwetday, precipday
  use mod_canopy_params, only: totcondub, totconduc, totcondls, totcondl3, &
                                totcondl4, &
                                tu, ts, tl, tlsub, t12, t34, q12, q34, &
                                ciub, ciuc, cils, cil3, cil4, &
                                csub, csuc, csls, csl3, csl4, &
                                gsub, gsuc, gsls, gsl3, gsl4, &
                                wliqu, wliqs, wliql, wsnou, wsnos, wsnol, &
                                su, ss, sl
  use mod_hydrology,     only: ginvap, gsuvap, gtrans, grunof, gdrain
  use mod_veg_dynamics,  only: totlit, totnlit, totfall, totalit, totrlit, &
                                totanlit, totrnlit, totcsoi, totnmic, tco2mic, &
                                tnpptot, tneetot, tnmin, totcmic, &
                                tempu, templ, dropu, dropls, dropl4, dropl3
  use mod_carbon_pools,  only: clitlm, clitls, clitll, clitrm, clitrs, clitrl, &
                                clitwm, clitws, clitwl, csoipas, csoislop, &
                                csoislon, cdisturb
  implicit none

  integer, intent(in) :: irestart  ! 0=initial run, 1=restart
  integer :: i, j                  ! loop indices

  ! ---------------------------------------------------------------------------
  ! Set physical constants (MKS); sets epsilon and vzero(mod_constants)
  ! ---------------------------------------------------------------------------
  call constants_init()
  vzero = 0.0_wp   ! zero mod_run_params' vzero as well

  ! ---------------------------------------------------------------------------
  ! Initialise integer arrays explicitly
  ! ---------------------------------------------------------------------------
  do i = 1, npoi
    iwet(i) = 0
    do j = 1, 31
      iwetday(i,j)   = 0
      precipday(i,j) = 0.0_wp
    end do
  end do

  ! ---------------------------------------------------------------------------
  ! Zero flux and diagnostic arrays
  ! ---------------------------------------------------------------------------
  asurd     = 0.0_wp
  asuri     = 0.0_wp

  totcondub = 0.0_wp
  totconduc = 0.0_wp
  totcondls = 0.0_wp
  totcondl3 = 0.0_wp
  totcondl4 = 0.0_wp

  ginvap    = 0.0_wp
  gsuvap    = 0.0_wp
  gtrans    = 0.0_wp
  grunof    = 0.0_wp
  gdrain    = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! Initialise canopy temperature fields to 10 C (283.16 K)
  ! ---------------------------------------------------------------------------
  tu    = 283.16_wp
  ts    = 283.16_wp
  tl    = 283.16_wp

  ! ---------------------------------------------------------------------------
  ! Weather-generator memory
  ! ---------------------------------------------------------------------------
  xstore = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! Lower canopy buried by snow: 0 C (273.16 K)
  ! ---------------------------------------------------------------------------
  tlsub  = 273.16_wp

  ! ---------------------------------------------------------------------------
  ! Canopy air conditions (used in turvap)
  ! ---------------------------------------------------------------------------
  t12 = 283.16_wp
  t34 = 283.16_wp
  q12 = 0.0_wp
  q34 = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! CO2 concentrations (mol/mol)
  ! ---------------------------------------------------------------------------
  ciub = 350.0e-06_wp
  ciuc = 350.0e-06_wp
  cils = 350.0e-06_wp
  cil3 = 350.0e-06_wp
  cil4 = 350.0e-06_wp

  csub = 350.0e-06_wp
  csuc = 350.0e-06_wp
  csls = 350.0e-06_wp
  csl3 = 350.0e-06_wp
  csl4 = 350.0e-06_wp

  ! ---------------------------------------------------------------------------
  ! Stomatal conductance (mol-H2O m-2 s-1)
  ! ---------------------------------------------------------------------------
  gsub = 0.5_wp
  gsuc = 0.5_wp
  gsls = 0.5_wp
  gsl3 = 0.5_wp
  gsl4 = 0.5_wp

  ! ---------------------------------------------------------------------------
  ! Soil biogeochemistry (cold-start only)
  ! ---------------------------------------------------------------------------
  if (irestart == 0) then
    clitlm   = 0.0_wp
    clitls   = 0.0_wp
    clitll   = 0.0_wp
    clitrm   = 0.0_wp
    clitrs   = 0.0_wp
    clitrl   = 0.0_wp
    clitwm   = 0.0_wp
    clitws   = 0.0_wp
    clitwl   = 0.0_wp
    totcmic  = 0.0_wp
    csoislop = 0.0_wp
    csoislon = 0.0_wp
    csoipas  = 0.0_wp
  end if

  totlit   = 0.0_wp
  totnlit  = 0.0_wp
  totfall  = 0.0_wp
  totalit  = 0.0_wp
  totrlit  = 0.0_wp
  totanlit = 0.0_wp
  totrnlit = 0.0_wp
  totcsoi  = 0.0_wp
  totnmic  = 0.0_wp
  tco2mic  = 0.0_wp
  tnpptot  = 0.0_wp
  tneetot  = 0.0_wp
  tnmin    = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! Carbon lost to atmosphere due to biomass burning
  ! ---------------------------------------------------------------------------
  cdisturb = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! Phenology flags (cold-start only)
  ! ---------------------------------------------------------------------------
  if (irestart == 0) then
    tempu  = 1.0_wp
    templ  = 1.0_wp
    dropu  = 1.0_wp
    dropls = 1.0_wp
    dropl4 = 1.0_wp
    dropl3 = 1.0_wp
  end if

  ! ---------------------------------------------------------------------------
  ! Intercepted water/snow and aerodynamic transfer coefficients
  ! ---------------------------------------------------------------------------
  wliqu = 0.0_wp
  wliqs = 0.0_wp
  wliql = 0.0_wp
  wsnou = 0.0_wp
  wsnos = 0.0_wp
  wsnol = 0.0_wp

  su    = 0.0_wp
  ss    = 0.0_wp
  sl    = 0.0_wp

end subroutine inisurf

! =============================================================================
subroutine inisum
! =============================================================================
! Initialises the total water content variable (for mass conservation checks)
! and zeroes all daily, monthly, and annual time-averaging accumulators.
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi, nsoilay, nsnolay
  use mod_constants,     only: rhow
  use mod_snow_state,    only: fi, rhos, hsno
  use mod_soil_state,    only: wpud, wipud, poros, wsoi, wisoi, hsoi
  use mod_canopy_params, only: wliqu, wsnou, wliqs, wsnos, wliql, wsnol
  use mod_veg_dynamics,  only: fu, fl, lai, sai, firefac
  use mod_hydrology,     only: wtot
  use mod_carbon_pools,  only: decompl, decomps
  use mod_summary,       only: adrain, adsnow, adaet, adtrunoff, adsrunoff, &
                                addrainage, adrh, adsnod, adsnof, adwsoi, &
                                adwisoi, adtsoi, adtlaysoi, adwlaysoi, &
                                adwsoic, adtsoic, adco2mic, adco2root, &
                                adnmintot, &
                                amtemp, amrain, amsnow, amaet, amtrunoff, &
                                amsrunoff, amdrainage, amcloud, amqa, amrh, &
                                amsolar, amirup, amirdown, amsens, amlatent, &
                                amlaiu, amlail, amtsoi, amwsoi, amwisoi, &
                                amvwc, amawc, amsnod, amsnof, &
                                amco2mic, amco2root, amnmintot, amnpp, &
                                aysolar, ayirup, ayirdown, aysens, aylatent, &
                                ayprcp, ayaet, aytrans, aytrunoff, aysrunoff, &
                                aydrainage, aywsoi, aywisoi, aytsoi, &
                                ayvwc, ayawc, aystresstu, aystresstl, &
                                ayco2mic, ayco2root, ayrootbio, aynmintot, &
                                ayalit, ayblit, aycsoi, aycmic, &
                                ayanlit, aybnlit, aynsoi, aygpp
  implicit none

  integer :: i, k  ! loop indices

  ! ---------------------------------------------------------------------------
  ! Compute initial total water content (mass conservation reference value)
  ! ---------------------------------------------------------------------------
  do i = 1, npoi
    wtot(i) = (wliqu(i) + wsnou(i)) * fu(i) * 2.0_wp * lai(i,2)      + &
              (wliqs(i) + wsnos(i)) * fu(i) * 2.0_wp * sai(i,2)      + &
              (wliql(i) + wsnol(i)) * fl(i) * 2.0_wp                  * &
              (lai(i,1) + sai(i,1)) * (1.0_wp - fi(i))

    wtot(i) = wtot(i) + wpud(i) + wipud(i)

    do k = 1, nsoilay
      wtot(i) = wtot(i) + &
                poros(i,k) * wsoi(i,k)  * (1.0_wp - wisoi(i,k)) * hsoi(k) * rhow + &
                poros(i,k) * wisoi(i,k) *                          hsoi(k) * rhow
    end do

    do k = 1, nsnolay
      wtot(i) = wtot(i) + fi(i) * rhos * hsno(i,k)
    end do
  end do

  ! ---------------------------------------------------------------------------
  ! Zero daily accumulators
  ! ---------------------------------------------------------------------------
  adrain     = 0.0_wp
  adsnow     = 0.0_wp
  adaet      = 0.0_wp
  adtrunoff  = 0.0_wp
  adsrunoff  = 0.0_wp
  addrainage = 0.0_wp
  adrh       = 0.0_wp
  adsnod     = 0.0_wp
  adsnof     = 0.0_wp
  adwsoi     = 0.0_wp
  adtsoi     = 0.0_wp
  adwisoi    = 0.0_wp
  adtlaysoi  = 0.0_wp
  adwlaysoi  = 0.0_wp
  adwsoic    = 0.0_wp
  adtsoic    = 0.0_wp
  adco2mic   = 0.0_wp
  adco2root  = 0.0_wp
  decompl    = 0.0_wp
  decomps    = 0.0_wp
  adnmintot  = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! Zero monthly accumulators
  ! ---------------------------------------------------------------------------
  amtemp     = 0.0_wp
  amrain     = 0.0_wp
  amsnow     = 0.0_wp
  amaet      = 0.0_wp
  amtrunoff  = 0.0_wp
  amsrunoff  = 0.0_wp
  amdrainage = 0.0_wp
  amcloud    = 0.0_wp
  amqa       = 0.0_wp
  amrh       = 0.0_wp
  amsolar    = 0.0_wp
  amirup     = 0.0_wp
  amirdown   = 0.0_wp
  amsens     = 0.0_wp
  amlatent   = 0.0_wp
  amlaiu     = 0.0_wp
  amlail     = 0.0_wp
  amtsoi     = 0.0_wp
  amwsoi     = 0.0_wp
  amwisoi    = 0.0_wp
  amvwc      = 0.0_wp
  amawc      = 0.0_wp
  amsnod     = 0.0_wp
  amsnof     = 0.0_wp
  amco2mic   = 0.0_wp
  amco2root  = 0.0_wp
  amnmintot  = 0.0_wp
  amnpp      = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! Zero annual accumulators
  ! ---------------------------------------------------------------------------
  aysolar    = 0.0_wp
  ayirup     = 0.0_wp
  ayirdown   = 0.0_wp
  aysens     = 0.0_wp
  aylatent   = 0.0_wp
  ayprcp     = 0.0_wp
  ayaet      = 0.0_wp
  aytrans    = 0.0_wp
  aytrunoff  = 0.0_wp
  aysrunoff  = 0.0_wp
  aydrainage = 0.0_wp
  aywsoi     = 0.0_wp
  aywisoi    = 0.0_wp
  aytsoi     = 0.0_wp
  ayvwc      = 0.0_wp
  ayawc      = 0.0_wp
  aystresstu = 0.0_wp
  aystresstl = 0.0_wp
  firefac    = 0.0_wp
  ayco2mic   = 0.0_wp
  ayco2root  = 0.0_wp
  ayrootbio  = 0.0_wp
  aynmintot  = 0.0_wp
  ayalit     = 0.0_wp
  ayblit     = 0.0_wp
  aycsoi     = 0.0_wp
  aycmic     = 0.0_wp
  ayanlit    = 0.0_wp
  aybnlit    = 0.0_wp
  aynsoi     = 0.0_wp
  aygpp      = 0.0_wp

end subroutine inisum

! =============================================================================
subroutine inisnow
! =============================================================================
! Sets bulk snow physical parameters (scalars).
  use mod_precision,   only: wp
  use mod_dimensions,  only: nsnolay
  use mod_run_params,  only: dtime
  use mod_constants,   only: rhow
  use mod_snow_state,  only: rhos, consno, hsnotop, hsnomin, fimin, fimax, z0sno
  implicit none

  ! density of snow
  rhos    = 0.25_wp * rhow

  ! thermal conductivity of snow (W m-1 K-1)
  consno  = 0.20_wp

  ! adaptive-grid thickness of top snow layer (m)
  hsnotop = 0.05_wp

  ! minimum total snow thickness (m); must be >= nsnolay * hsnotop for vadapt
  hsnomin = max(0.15_wp, real(nsnolay, wp) * hsnotop)

  ! minimum and maximum snow-cover fractions
  fimin   = 0.00002_wp * (dtime / 1800.0_wp) * (0.15_wp / hsnomin)
  fimax   = 1.0_wp

  ! roughness length of snow surface (m)
  z0sno   = 0.0005_wp

end subroutine inisnow

! =============================================================================
subroutine inisoil(irestart)
! =============================================================================
! Sets soil physical and hydraulic properties from texture data.
! On cold start, initialises soil water and temperature fields.
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi, nsoilay
  use mod_soil_state,    only: wpud, wipud, wsoi, wisoi, tsoi, tg, ti, &
                                z0soi, albsav, albsan, rhosoi, csoi, &
                                poros, sfield, swilt, bex, ibex, suction, hydraul, &
                                sand, clay
  use mod_snow_state,    only: tsno
  use mod_soil_texture,  only: ndat, texdat, porosdat, sfielddat, swiltdat, &
                                bexdat, suctiondat, hydrauldat
  implicit none

  integer, intent(in) :: irestart  ! 0=initial run, 1=restart

  integer :: i, k, l               ! loop indices
  integer :: msand, mclay          ! % sand and clay at grid point
  integer :: lmin                  ! matched textural class index (1..11)
  real(wp) :: fsand, fsilt, fclay  ! fractional sand, silt, clay
  real(wp) :: forganic             ! fractional organic content (placeholder)
  real(wp) :: xdat(ndat)           ! sand % for each textural class
  real(wp) :: ydat(ndat)           ! silt % for each textural class
  real(wp) :: zdat(ndat)           ! clay % for each textural class
  integer, external :: textcls     ! USDA texture classifier (in this file)

  ! ---------------------------------------------------------------------------
  ! Build local sand/silt/clay vectors from the module's texdat array
  ! ---------------------------------------------------------------------------
  do l = 1, ndat
    xdat(l) = texdat(1,l)
    ydat(l) = texdat(2,l)
    zdat(l) = texdat(3,l)
  end do

  ! ---------------------------------------------------------------------------
  ! Cold-start: zero puddle storage
  ! ---------------------------------------------------------------------------
  if (irestart == 0) then
    wpud  = 0.0_wp
    wipud = 0.0_wp
  end if

  ! ---------------------------------------------------------------------------
  ! Set bare-soil roughness length
  ! ---------------------------------------------------------------------------
  z0soi = 0.005_wp

  ! ---------------------------------------------------------------------------
  ! Cold-start: initialise soil water and temperature fields
  ! ---------------------------------------------------------------------------
  if (irestart == 0) then
    wsoi  = 0.50_wp
    wisoi = 0.00_wp
    tsoi  = 278.13_wp
    tg    = 278.13_wp
    ti    = 273.13_wp
  else
    do i = 1, npoi
      tg(i) = tsoi(i,1)
      ti(i) = tsno(i,1)
    end do
  end if

  ! ---------------------------------------------------------------------------
  ! Surface albedo from sand/silt/clay fractions (BATS table 3.II)
  ! ---------------------------------------------------------------------------
  do i = 1, npoi
    msand = nint(sand(i,1))
    mclay = nint(clay(i,1))
    fsand = 0.01_wp * real(msand, wp)
    fclay = 0.01_wp * real(mclay, wp)
    fsilt = 0.01_wp * real(100 - msand - mclay, wp)

    albsav(i) = fsand * 0.120_wp + fsilt * 0.085_wp + fclay * 0.050_wp
    albsan(i) = 2.0_wp * albsav(i)
  end do

  ! ---------------------------------------------------------------------------
  ! Set soil physical properties at each layer for the global domain.
  ! Layers beyond 6 use the layer-6 texture (analysis limit of WISE-IGBP).
  ! ---------------------------------------------------------------------------
  do k = 1, nsoilay
    do i = 1, npoi

      if (k <= 6) then
        msand = nint(sand(i,k))
        mclay = nint(clay(i,k))
      else
        msand = nint(sand(i,6))
        mclay = nint(clay(i,6))
      end if

      fsand    = 0.01_wp * real(msand,         wp)
      fclay    = 0.01_wp * real(mclay,         wp)
      fsilt    = 0.01_wp * real(100 - msand - mclay, wp)
      forganic = 0.010_wp  ! 1% organic (placeholder until BGC coupling)

      ! density of soil material (without pores) (kg m-3): Campbell & Norman 1998
      rhosoi(i,k) = 2650.0_wp * (1.0_wp - forganic) + &
                    1300.0_wp *            forganic

      ! volumetric heat capacity of soil material (J m-3 K-1): Campbell & Norman 1998
      csoi(i,k)   =  870.0_wp * (1.0_wp - forganic) + &
                    1920.0_wp *            forganic

      ! match sand/clay % to USDA textural class
      lmin = textcls(msand, mclay)

      ! porosity (fraction)
      poros(i,k)   = porosdat(lmin)

      ! field capacity (relative to porosity)
      sfield(i,k)  = 1.0_wp / poros(i,k) * sfielddat(lmin)

      ! wilting point (relative to porosity)
      swilt(i,k)   = 1.0_wp / poros(i,k) * swiltdat(lmin)

      ! Campbell moisture-release exponent b
      bex(i,k)     = bexdat(lmin)

      ! integer approximation of b for speed
      ibex(i,k)    = nint(bex(i,k))

      ! saturated matric potential (m H2O)
      suction(i,k) = suctiondat(lmin)

      ! saturated hydraulic conductivity (m s-1)
      hydraul(i,k) = hydrauldat(lmin)

    end do
  end do

end subroutine inisoil

! =============================================================================
integer function textcls(msand, mclay)
! =============================================================================
! Classifies soil in the USDA textural triangle from sand and clay %.
! Returns index 1..11 corresponding to textural class.
! Adapted for IBIS by cjk 01/11/01 from code by Aris Gerakis (1998-1999).
  implicit none

  integer, intent(in) :: msand, mclay

  logical, external :: inpoly  ! point-in-polygon test (in this file)

  real :: silty_loam(7,2),     sandy(7,2),        loam(7,2), &
          silty_clay_loam(7,2), clay_loam(7,2),    sandy_loam(7,2), &
          silty_clay(7,2),     sandy_clay_loam(7,2), loamy_sand(7,2), &
          clayey(7,2),         sandy_clay(7,2)

  ! Polygon coordinates: sand % in column 1, clay % in column 2.
  ! Vertex order follows the USDA textural triangle boundaries.
  data sandy           /85., 90., 100., 0., 0., 0., 0.,  &
                          0., 10.,   0., 0., 0., 0., 0./
  data loamy_sand      /70., 85., 90., 85., 0., 0., 0.,  &
                          0., 15., 10.,  0., 0., 0., 0./
  data sandy_loam      /50., 43., 52., 52., 80., 85., 70.,  &
                          0.,  7.,  7., 20., 20., 15.,  0./
  data loam            /43., 23., 45., 52., 52., 0., 0.,  &
                          7., 27., 27., 20.,  7., 0., 0./
  data silty_loam      / 0.,  0., 23., 50., 0., 0., 0.,  &
                          0., 27., 27.,  0., 0., 0., 0./
  data sandy_clay_loam /52., 45., 45., 65., 80., 0., 0.,  &
                         20., 27., 35., 35., 20.,  0., 0./
  data clay_loam       /20., 20., 45., 45., 0., 0., 0.,  &
                         27., 40., 40., 27., 0., 0., 0./
  data silty_clay_loam / 0.,  0., 20., 20., 0., 0., 0.,  &
                         27., 40., 40., 27., 0., 0., 0./
  data sandy_clay      /45., 45., 65., 0., 0., 0., 0.,  &
                         35., 55., 35., 0., 0., 0., 0./
  data silty_clay      / 0.,  0., 20., 0., 0., 0., 0.,  &
                         40., 60., 40., 0., 0., 0., 0./
  data clayey          /20.,  0.,  0., 45., 45., 0., 0.,  &
                         40., 60., 100., 55., 40., 0., 0./

  textcls = 0

  if (msand > 0 .and. mclay > 0) then
    if (inpoly(sandy,           3, msand, mclay)) textcls =  1   ! sand
    if (inpoly(loamy_sand,      4, msand, mclay)) textcls =  2   ! loamy sand
    if (inpoly(sandy_loam,      7, msand, mclay)) textcls =  3   ! sandy loam
    if (inpoly(loam,            5, msand, mclay)) textcls =  4   ! loam
    if (inpoly(silty_loam,      4, msand, mclay)) textcls =  5   ! silt loam
    if (inpoly(sandy_clay_loam, 5, msand, mclay)) textcls =  6   ! sandy clay loam
    if (inpoly(clay_loam,       4, msand, mclay)) textcls =  7   ! clay loam
    if (inpoly(silty_clay_loam, 4, msand, mclay)) textcls =  8   ! silty clay loam
    if (inpoly(sandy_clay,      3, msand, mclay)) textcls =  9   ! sandy clay
    if (inpoly(silty_clay,      3, msand, mclay)) textcls = 10   ! silty clay
    if (inpoly(clayey,          5, msand, mclay)) textcls = 11   ! clay
  end if

  ! Default to silt loam if no class matched
  if (textcls == 0) textcls = 5

end function textcls

! =============================================================================
logical function inpoly(poly, npoints, xt, yt)
! =============================================================================
! Returns .true. if point (xt, yt) lies inside or on the boundary of the
! polygon defined by poly(1:npoints, 1:2).
! Adapted for IBIS by cjk 01/11/01 from Fortran translation by Aris Gerakis
! (apr. 1998, nov. 1998, july 1999) of original C code by Bob Stein &
! Craig Yap (Galacticomm, Inc., 1995-1996, freeware).
  implicit none

  integer, intent(in) :: npoints, xt, yt
  real,    intent(in) :: poly(7,2)

  real    :: xnew, ynew, xold, yold, x1, y1, x2, y2
  integer :: i
  logical :: inside, on_border

  inside    = .false.
  on_border = .false.

  if (npoints < 3) then
    inpoly = .false.
    return
  end if

  xold = poly(npoints, 1)
  yold = poly(npoints, 2)

  do i = 1, npoints
    xnew = poly(i, 1)
    ynew = poly(i, 2)

    if (xnew > xold) then
      x1 = xold;  x2 = xnew
      y1 = yold;  y2 = ynew
    else
      x1 = xnew;  x2 = xold
      y1 = ynew;  y2 = yold
    end if

    ! Straddle test and non-vertical border test
    if ((xnew < xt .and. xt <= xold) .or. (.not. xnew < xt .and. .not. xt <= xold)) then
      if ((yt - y1) * (x2 - x1) == (y2 - y1) * (xt - x1)) then
        on_border = .true.
      else if ((yt - y1) * (x2 - x1) < (y2 - y1) * (xt - x1) + 0.001) then
        inside = .not. inside
      end if
    else if ((xnew == xt .or. xold == xt) .and. &
             (yt - y1) * (x2 - x1) == (y2 - y1) * (xt - x1) .and. &
             ((ynew <= yt .and. yt <= yold) .or. &
              (.not. ynew < yt .and. .not. yt < yold))) then
      on_border = .true.
    end if

    xold = xnew
    yold = ynew
  end do

  if (.not. on_border) then
    inpoly = inside
  else
    inpoly = .true.
  end if

end function inpoly

! =============================================================================
subroutine iniveg(isimveg, irestart)
! =============================================================================
! Initialises vegetation state (LAI, biomass, structure, rooting profiles).
! On cold start, uses either vegetation-map-based or uniform initial conditions.
! On restart, derives LAI from biomass read from the restart file.
  use mod_precision,          only: wp
  use mod_dimensions,         only: npoi, npft, nsoilay
  use mod_boundary_climate,   only: xinveg
  use mod_soil_state,         only: stresstu, stresstl, hsoi
  use mod_pft_params,         only: plai_init, plaiupper, plailower, xminlai, &
                                    sapfrac_init, beta1, beta2, chiflz, chifuz
  use mod_veg_dynamics,       only: agddu, agddl, specla, sapfrac, &
                                    biomass, plai, exist, tw, &
                                    totlaiu, totlail, totbiou, totbiol, &
                                    fu, fl, lai, sai, zbot, ztop, vegtype0, &
                                    froot
  use mod_canopy_params,      only: woodnorm, orieh, oriev
  use mod_carbon_pools,       only: cbiol, cbior, cbiow, falll, fallr, fallw
  use mod_summary,            only: a10td, a10ancub, a10ancuc, a10ancls, &
                                    a10ancl3, a10ancl4, &
                                    a10scalparamu, a10scalparaml, &
                                    a10daylightu, a10daylightl
  implicit none

  integer, intent(in) :: isimveg  ! 0=static, 1=dyn-veg-map, 2=dyn-veg-uniform
  integer, intent(in) :: irestart ! 0=cold start, 1=restart

  integer  :: i, j, k             ! loop indices
  integer  :: ideci, ievgr        ! counts of deciduous / evergreen PFTs
  integer  :: ishrub, igrass      ! counts of shrub / grass PFTs
  integer  :: ilower, iupper      ! counts of lower / upper canopy PFTs
  integer  :: inveg               ! vegetation type index (1..15)
  real(wp) :: wood                ! total woody biomass for this grid cell (kg_C m-2)
  real(wp) :: totdepth            ! total soil depth (cm)
  real(wp) :: frootnorm1          ! normalisation factor for lower-canopy roots
  real(wp) :: frootnorm2          ! normalisation factor for upper-canopy roots
  real(wp) :: depth(nsoilay)      ! cumulative soil depth per layer (cm)

  do i = 1, npoi

    ! -------------------------------------------------------------------------
    ! Initialise a few climatic variables needed by vegetation (cold start only)
    ! -------------------------------------------------------------------------
    if (irestart == 0) then
      agddu(i) = 1000.0_wp
      agddl(i) = 1000.0_wp
    end if

    ! -------------------------------------------------------------------------
    ! Initialise moisture stress factors
    ! -------------------------------------------------------------------------
    stresstu(i) = 1.0_wp
    stresstl(i) = 1.0_wp

    ! -------------------------------------------------------------------------
    ! Cold-start: initialise 10-day running means
    ! -------------------------------------------------------------------------
    if (irestart == 0) then
      a10td(i)         = 273.16_wp
      a10ancub(i)      = 10.0e-06_wp
      a10ancuc(i)      = 10.0e-06_wp
      a10ancls(i)      = 10.0e-06_wp
      a10ancl4(i)      = 10.0e-06_wp
      a10ancl3(i)      = 10.0e-06_wp
      a10scalparamu(i) = 0.5_wp * 5.0_wp
      a10scalparaml(i) = 0.5_wp * 5.0_wp
      a10daylightu(i)  = 5.0_wp
      a10daylightl(i)  = 5.0_wp
      falll(i)         = 0.0_wp
      fallr(i)         = 0.0_wp
      fallw(i)         = 0.0_wp
    end if

    ! -------------------------------------------------------------------------
    ! Count PFTs of each functional group for normalisation denominators
    ! -------------------------------------------------------------------------
    ievgr  = 0;  ideci  = 0
    ishrub = 0;  igrass = 0
    ilower = 0;  iupper = 0

    if (nint(exist(i, 1)) == 1) ievgr  = ievgr  + 1
    if (nint(exist(i, 3)) == 1) ievgr  = ievgr  + 1
    if (nint(exist(i, 4)) == 1) ievgr  = ievgr  + 1
    if (nint(exist(i, 6)) == 1) ievgr  = ievgr  + 1
    if (nint(exist(i, 2)) == 1) ideci  = ideci  + 1
    if (nint(exist(i, 5)) == 1) ideci  = ideci  + 1
    if (nint(exist(i, 7)) == 1) ideci  = ideci  + 1
    if (nint(exist(i, 8)) == 1) ideci  = ideci  + 1
    if (nint(exist(i, 9)) == 1) ishrub = ishrub + 1
    if (nint(exist(i,10)) == 1) ishrub = ishrub + 1
    if (nint(exist(i,11)) == 1) igrass = igrass + 1
    if (nint(exist(i,12)) == 1) igrass = igrass + 1

    ievgr  = max(1, ievgr);   ideci  = max(1, ideci)
    ishrub = max(1, ishrub);  igrass = max(1, igrass)
    iupper = max(1, ievgr  + ideci)
    ilower = max(1, ishrub + igrass)

    ! -------------------------------------------------------------------------
    ! Set initial LAI and biomass
    ! -------------------------------------------------------------------------
    if (irestart == 0) then

      if (isimveg == 0 .or. isimveg == 1) then
        ! Initialise using vegetation-map-based plai_init values (read from
        ! params.veg by RD_PARAM).
        inveg      = nint(xinveg(i))
        vegtype0(i) = xinveg(i)

        plai(i, 1) = exist(i, 1) / real(ievgr,  wp) * plai_init(1, inveg)
        plai(i, 2) = exist(i, 2) / real(ideci,  wp) * plai_init(2, inveg)
        plai(i, 3) = exist(i, 3) / real(ievgr,  wp) * plai_init(1, inveg)
        plai(i, 4) = exist(i, 4) / real(ievgr,  wp) * plai_init(1, inveg)
        plai(i, 5) = exist(i, 5) / real(ideci,  wp) * plai_init(2, inveg)
        plai(i, 6) = exist(i, 6) / real(ievgr,  wp) * plai_init(1, inveg)
        plai(i, 7) = exist(i, 7) / real(ideci,  wp) * plai_init(2, inveg)
        plai(i, 8) = exist(i, 8) / real(ideci,  wp) * plai_init(2, inveg)
        plai(i, 9) = exist(i, 9) / real(ishrub, wp) * plai_init(3, inveg)
        plai(i,10) = exist(i,10) / real(ishrub, wp) * plai_init(3, inveg)

        ! C4/C3 grass split for savanna (9) and grassland (10)
        if (inveg == 9 .or. inveg == 10) then
          if (tw(i) > 22.0_wp) then
            plai(i,11) = exist(i,11) * 0.80_wp * plai_init(4, inveg)
            plai(i,12) = exist(i,12) * 0.20_wp * plai_init(4, inveg)
          else
            plai(i,11) = exist(i,11) * 0.00_wp * plai_init(4, inveg)
            plai(i,12) = exist(i,12) * 1.00_wp * plai_init(4, inveg)
          end if
        else
          plai(i,11) = exist(i,11) / real(igrass, wp) * plai_init(4, inveg)
          plai(i,12) = exist(i,12) / real(igrass, wp) * plai_init(4, inveg)
        end if

      else if (isimveg == 2) then
        ! Uniform initial conditions
        plai(i, 1) = exist(i, 1) / real(iupper, wp) * plaiupper
        plai(i, 2) = exist(i, 2) / real(iupper, wp) * plaiupper
        plai(i, 3) = exist(i, 3) / real(iupper, wp) * plaiupper
        plai(i, 4) = exist(i, 4) / real(iupper, wp) * plaiupper
        plai(i, 5) = exist(i, 5) / real(iupper, wp) * plaiupper
        plai(i, 6) = exist(i, 6) / real(iupper, wp) * plaiupper
        plai(i, 7) = exist(i, 7) / real(iupper, wp) * plaiupper
        plai(i, 8) = exist(i, 8) / real(iupper, wp) * plaiupper
        plai(i, 9) = exist(i, 9) / real(ilower, wp) * plailower
        plai(i,10) = exist(i,10) / real(ilower, wp) * plailower
        plai(i,11) = exist(i,11) / real(ilower, wp) * plailower
        plai(i,12) = exist(i,12) / real(ilower, wp) * plailower
      end if

      ! Apply minimum LAI for each existing plant type
      do j = 1, npft
        plai(i,j) = max(plai(i,j), exist(i,j) * xminlai)
      end do

      ! Set sapwood fraction and compute biomass
      sapfrac(i) = sapfrac_init
      wood       = 0.0_wp

      do j = 1, npft
        cbiol(i,j)    = plai(i,j) / specla(j)
        cbior(i,j)    = 0.5_wp * cbiol(i,j)
        cbiow(i,j)    = 0.0_wp
        if (j < 9) cbiow(i,j) = plai(i,j) * 10.0_wp / 6.0_wp
        biomass(i,j)  = cbiol(i,j) + cbiow(i,j) + cbior(i,j)
        wood          = wood + cbiow(i,j)
      end do

    else
      ! Restart: derive LAI from biomass read from restart file
      wood = 0.0_wp
      do j = 1, npft
        plai(i,j)    = cbiol(i,j) * specla(j)
        biomass(i,j) = cbiol(i,j) + cbiow(i,j) + cbior(i,j)
        wood         = wood + cbiow(i,j)
      end do
    end if

    ! -------------------------------------------------------------------------
    ! Canopy structural diagnostics
    ! -------------------------------------------------------------------------
    totlaiu(i) = plai(i,1) + plai(i,2) + plai(i,3) + plai(i,4) + &
                 plai(i,5) + plai(i,6) + plai(i,7) + plai(i,8)

    totlail(i) = plai(i,9) + plai(i,10) + plai(i,11) + plai(i,12)

    totbiou(i) = biomass(i,1) + biomass(i,2) + biomass(i,3) + biomass(i,4) + &
                 biomass(i,5) + biomass(i,6) + biomass(i,7) + biomass(i,8)

    totbiol(i) = biomass(i,9) + biomass(i,10) + biomass(i,11) + biomass(i,12)

    ! Single-sided stem area index
    sai(i,1) = 0.050_wp * totlail(i)
    sai(i,2) = 0.250_wp * totlaiu(i)

    ! Fractional cover
    fu(i) = (1.0_wp - exp(-wood)) / (1.0_wp - exp(-woodnorm))
    fl(i) = totlail(i) / 1.0_wp

    fu(i) = max(0.25_wp, min(0.975_wp, fu(i)))
    fl(i) = max(0.25_wp, min(0.975_wp, fl(i)))

    ! Canopy LAI for physics
    lai(i,1) = totlail(i) / fl(i)
    lai(i,2) = totlaiu(i) / fu(i)

    ! Canopy heights
    zbot(i,1) = 0.05_wp
    ztop(i,1) = max(0.25_wp, lai(i,1) * 0.25_wp)

    zbot(i,2) = ztop(i,1) + 1.0_wp
    ztop(i,2) = max(zbot(i,2) + 1.00_wp, &
                    2.50_wp * totbiou(i) / fu(i) * 0.75_wp)

  end do  ! end i = 1, npoi

  ! ---------------------------------------------------------------------------
  ! Leaf orientation factors (chiflz, chifuz from params.veg via RD_PARAM)
  ! ---------------------------------------------------------------------------
  oriev(1) = max(-chiflz, 0.0_wp)
  oriev(2) = max(-chifuz, 0.0_wp)
  orieh(1) = max( chiflz, 0.0_wp)
  orieh(2) = max( chifuz, 0.0_wp)

  ! ---------------------------------------------------------------------------
  ! Rooting profiles (Jackson et al. 1996, 1997)
  ! beta1 = lower layer (grasses/shrubs); beta2 = upper layer (trees)
  ! ---------------------------------------------------------------------------
  totdepth = 0.0_wp
  do k = 1, nsoilay
    totdepth = totdepth + hsoi(k) * 100.0_wp
  end do

  frootnorm1 = 1.0_wp - beta1 ** totdepth
  frootnorm2 = 1.0_wp - beta2 ** totdepth

  do k = 1, nsoilay
    if (k == 1) then
      depth(k)    = hsoi(k) * 100.0_wp
      froot(k,1)  = 1.0_wp - beta1 ** depth(k)
      froot(k,2)  = 1.0_wp - beta2 ** depth(k)
    else
      depth(k)    = depth(k-1) + hsoi(k) * 100.0_wp
      froot(k,1)  = (1.0_wp - beta1 ** depth(k)) - &
                    (1.0_wp - beta1 ** depth(k-1))
      froot(k,2)  = (1.0_wp - beta2 ** depth(k)) - &
                    (1.0_wp - beta2 ** depth(k-1))
    end if
    froot(k,1) = froot(k,1) / frootnorm1
    froot(k,2) = froot(k,2) / frootnorm2
  end do

end subroutine iniveg
