! =============================================================================
! mod_canopy_fluxes.f90  --  Intra-canopy radiation, flux, and aerodynamic state
!
! Replaces COMMON blocks /com1d1/–/com1d10/ (excluding /com1d7/ which is
! absent in the original) from com1d.h.
!
! Holds all intermediate canopy-scale variables computed by the radiation,
! aerodynamic, and canopy flux routines and passed between them within a
! single timestep.  Nothing here persists across timesteps (all arrays are
! diagnostic).
!
! COMMON block mapping:
!   /com1d1/ : canopy photosynthesis scaling coefficients
!   /com1d2/ : canopy scaling coefficients
!   /com1d3/ : internal radiation and flux terms
!   /com1d4/ : nsol (integer, count of sunlit points)
!   /com1d5/ : radiation redistribution factors
!   /com1d6/ : indsol (sunlit-point index array)
!   /com1d8/ : aerodynamic and transfer coefficients, heights
!   /com1d9/ : tfac pressure ratio scalar
!   /com1d10/: intercepted-water wetness fractions
!
! Replaces: com1d.h (COMMON /com1d1/–/com1d10/).
! =============================================================================
module mod_canopy_fluxes
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /com1d1/  --  canopy photosynthesis scaling terms
  ! ---------------------------------------------------------------------------
  public :: termu, terml

  !> Scaling terms for upper canopy photosynthesis (7 terms per point).
  real(wp) :: termu(npoi, 7) = 0.0_wp

  !> Scaling terms for lower canopy photosynthesis (7 terms per point).
  real(wp) :: terml(npoi, 7) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /com1d2/  --  canopy scaling coefficients
  ! ---------------------------------------------------------------------------
  public :: scalcoefu, scalcoefl

  !> Upper canopy scaling coefficients (4 per point).
  real(wp) :: scalcoefu(npoi, 4) = 0.0_wp

  !> Lower canopy scaling coefficients (4 per point).
  real(wp) :: scalcoefl(npoi, 4) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /com1d3/  --  internal radiation and sensible/latent flux terms
  ! ---------------------------------------------------------------------------
  public :: firb,  firg,  firi,  firl,  firs,  firu
  public :: fsena, fseng, fseni, fsenl, fsens, fsenu
  public :: fvapa, fvapg, fvapi, fvaplt, fvaplw, fvaps, fvaput, fvapuw
  public :: raing, rainl, rainu
  public :: snowg, snowl, snowu
  public :: traing, trainl, trainu
  public :: tsnowg, tsnowl, tsnowu
  public :: soli, solg, soll, sols, solu

  ! IR radiation components (W m-2):
  real(wp) :: firb(npoi)   = 0.0_wp  !< net upward IR at atmospheric reference level
  real(wp) :: firg(npoi)   = 0.0_wp  !< IR absorbed by soil/ice
  real(wp) :: firi(npoi)   = 0.0_wp  !< IR absorbed by snow
  real(wp) :: firl(npoi)   = 0.0_wp  !< IR absorbed by lower canopy
  real(wp) :: firs(npoi)   = 0.0_wp  !< IR absorbed by upper canopy stems
  real(wp) :: firu(npoi)   = 0.0_wp  !< IR absorbed by upper canopy leaves

  ! Sensible heat fluxes (W m-2):
  real(wp) :: fsena(npoi)  = 0.0_wp  !< downward sens. heat, za to z12
  real(wp) :: fseng(npoi)  = 0.0_wp  !< upward sens. heat, soil to z34
  real(wp) :: fseni(npoi)  = 0.0_wp  !< upward sens. heat, snow to z34
  real(wp) :: fsenl(npoi)  = 0.0_wp  !< sens. heat, lower canopy to air
  real(wp) :: fsens(npoi)  = 0.0_wp  !< sens. heat, upper canopy stems to air
  real(wp) :: fsenu(npoi)  = 0.0_wp  !< sens. heat, upper canopy leaves to air

  ! Vapour fluxes (kg m-2 s-1, scaled by fraction as noted in com1d.h):
  real(wp) :: fvapa(npoi)  = 0.0_wp  !< downward H2O flux, za to z12
  real(wp) :: fvapg(npoi)  = 0.0_wp  !< evaporation from soil/bare ground
  real(wp) :: fvapi(npoi)  = 0.0_wp  !< evaporation from snow surface
  real(wp) :: fvaplt(npoi) = 0.0_wp  !< transpiration from lower canopy
  real(wp) :: fvaplw(npoi) = 0.0_wp  !< evaporation from lower canopy wet surface
  real(wp) :: fvaps(npoi)  = 0.0_wp  !< evaporation from upper canopy stems
  real(wp) :: fvaput(npoi) = 0.0_wp  !< transpiration from upper canopy leaves (dry)
  real(wp) :: fvapuw(npoi) = 0.0_wp  !< evaporation from upper canopy leaves (wet)

  ! Precipitation partitioning (kg m-2 s-1):
  real(wp) :: raing(npoi)  = 0.0_wp  !< rainfall rate at soil level
  real(wp) :: rainl(npoi)  = 0.0_wp  !< rainfall rate below upper canopy
  real(wp) :: rainu(npoi)  = 0.0_wp  !< rainfall rate above upper canopy
  real(wp) :: snowg(npoi)  = 0.0_wp  !< snowfall rate at soil level
  real(wp) :: snowl(npoi)  = 0.0_wp  !< snowfall rate below upper canopy
  real(wp) :: snowu(npoi)  = 0.0_wp  !< snowfall rate above upper canopy

  ! Precipitation temperatures (K):
  real(wp) :: traing(npoi) = 0.0_wp  !< rainfall temperature at soil level
  real(wp) :: trainl(npoi) = 0.0_wp  !< rainfall temperature below upper canopy
  real(wp) :: trainu(npoi) = 0.0_wp  !< rainfall temperature above upper canopy
  real(wp) :: tsnowg(npoi) = 0.0_wp  !< snowfall temperature at soil level
  real(wp) :: tsnowl(npoi) = 0.0_wp  !< snowfall temperature below upper canopy
  real(wp) :: tsnowu(npoi) = 0.0_wp  !< snowfall temperature above upper canopy

  ! Solar flux absorbed by each surface element (W m-2):
  real(wp) :: soli(npoi)   = 0.0_wp  !< snow surface
  real(wp) :: solg(npoi)   = 0.0_wp  !< snow-free soil
  real(wp) :: soll(npoi)   = 0.0_wp  !< lower canopy
  real(wp) :: sols(npoi)   = 0.0_wp  !< upper canopy stems
  real(wp) :: solu(npoi)   = 0.0_wp  !< upper canopy leaves

  ! ---------------------------------------------------------------------------
  ! /com1d4/  --  count of sunlit land points in current strip
  ! ---------------------------------------------------------------------------
  public :: nsol

  !> Number of land points with positive cosine of solar zenith angle.
  integer :: nsol = 0

  ! ---------------------------------------------------------------------------
  ! /com1d5/  --  radiation redistribution fractions
  ! ---------------------------------------------------------------------------
  public :: abupd, abupi, ablod, abloi
  public :: albsnd, albsni, albsod, albsoi
  public :: flodd, flodi, floii
  public :: fupdd, fupdi, fupii
  public :: relod, reloi, reupd, reupi
  public :: sol2d, sol2i, sol3d, sol3i
  public :: dummy_rad

  real(wp) :: abupd(npoi) = 0.0_wp  !< fraction direct  rad. absorbed by upper canopy
  real(wp) :: abupi(npoi) = 0.0_wp  !< fraction diffuse rad. absorbed by upper canopy
  real(wp) :: ablod(npoi) = 0.0_wp  !< fraction direct  rad. absorbed by lower canopy
  real(wp) :: abloi(npoi) = 0.0_wp  !< fraction diffuse rad. absorbed by lower canopy
  real(wp) :: albsnd(npoi) = 0.0_wp !< direct  snow albedo
  real(wp) :: albsni(npoi) = 0.0_wp !< diffuse snow albedo
  real(wp) :: albsod(npoi) = 0.0_wp !< direct  soil albedo
  real(wp) :: albsoi(npoi) = 0.0_wp !< diffuse soil albedo
  !> Placeholder (always 0): no direct flux produced for diffuse incident.
  real(wp) :: dummy_rad(npoi) = 0.0_wp
  real(wp) :: flodd(npoi) = 0.0_wp  !< downward direct  rad. per incident direct, lower
  real(wp) :: flodi(npoi) = 0.0_wp  !< downward diffuse rad. per incident direct, lower
  real(wp) :: floii(npoi) = 0.0_wp  !< downward diffuse rad. per incident diffuse, lower
  real(wp) :: fupdd(npoi) = 0.0_wp  !< downward direct  rad. per incident direct, upper
  real(wp) :: fupdi(npoi) = 0.0_wp  !< downward diffuse rad. per incident direct, upper
  real(wp) :: fupii(npoi) = 0.0_wp  !< downward diffuse rad. per incident diffuse, upper
  real(wp) :: relod(npoi) = 0.0_wp  !< upward direct  rad. per incident direct, lower
  real(wp) :: reloi(npoi) = 0.0_wp  !< upward diffuse rad. per incident diffuse, lower
  real(wp) :: reupd(npoi) = 0.0_wp  !< upward direct  rad. per incident direct, upper
  real(wp) :: reupi(npoi) = 0.0_wp  !< upward diffuse rad. per incident diffuse, upper
  real(wp) :: sol2d(npoi) = 0.0_wp  !< direct  downward rad. below upper canopy per veg area
  real(wp) :: sol2i(npoi) = 0.0_wp  !< diffuse downward rad. below upper canopy per veg area
  real(wp) :: sol3d(npoi) = 0.0_wp  !< direct  downward rad. below upper canopy per grid area
  real(wp) :: sol3i(npoi) = 0.0_wp  !< diffuse downward rad. below upper canopy per grid area

  ! ---------------------------------------------------------------------------
  ! /com1d6/  --  index array for sunlit land points
  ! ---------------------------------------------------------------------------
  public :: indsol

  !> Index into the land-point array for points with positive coszen.
  integer :: indsol(npoi) = 0

  ! ---------------------------------------------------------------------------
  ! /com1d8/  --  aerodynamic coefficients, heights, and transfer coefficients
  ! ---------------------------------------------------------------------------
  public :: aloga, alogav, alogg, alogi, alogl, alogu
  public :: alog1, alog2, alog3, alog4
  public :: bdl, bdu, cl, cp, cu
  public :: dil, displ, dispu, diu
  public :: exphl, exphu, expl, expu
  public :: pfluxl, pfluxs, pfluxu
  public :: rhoa, richl, richu, sg, si
  public :: strahl, strahu, straml, stramu
  public :: u1, u12, u2, u3, u34, u4
  public :: za, z1, z12, z2, z3, z34, z4

  real(wp) :: aloga(npoi)  = 0.0_wp  !< log(za - dispu)
  real(wp) :: alogav(npoi) = 0.0_wp  !< average of alogi and alogg
  real(wp) :: alogg(npoi)  = 0.0_wp  !< log(soil roughness)
  real(wp) :: alogi(npoi)  = 0.0_wp  !< log(snow roughness)
  real(wp) :: alogl(npoi)  = 0.0_wp  !< log(roughness length, lower canopy)
  real(wp) :: alogu(npoi)  = 0.0_wp  !< log(roughness length, upper canopy)
  real(wp) :: alog1(npoi)  = 0.0_wp  !< log(z1 - dispu)
  real(wp) :: alog2(npoi)  = 0.0_wp  !< log(z2 - displ)
  real(wp) :: alog3(npoi)  = 0.0_wp  !< log(z3 - displ)
  real(wp) :: alog4(npoi)  = 0.0_wp  !< log(max(z4, 1.1*z0sno, 1.1*z0soi))
  real(wp) :: bdl(npoi)    = 0.0_wp  !< aerodynamic drag coefficient, lower canopy
  real(wp) :: bdu(npoi)    = 0.0_wp  !< aerodynamic drag coefficient, upper canopy
  real(wp) :: cl(npoi)     = 0.0_wp  !< air transfer coeff (*rhoa), z34 to z12
  real(wp) :: cp(npoi)     = 0.0_wp  !< specific heat of moist air (J kg-1 K-1)
  real(wp) :: cu(npoi)     = 0.0_wp  !< air transfer coeff (*rhoa), z12 to za
  real(wp) :: dil(npoi)    = 0.0_wp  !< inverse momentum diffusion coeff, lower canopy
  real(wp) :: displ(npoi)  = 0.0_wp  !< zero-plane displacement, lower canopy (m)
  real(wp) :: dispu(npoi)  = 0.0_wp  !< zero-plane displacement, upper canopy (m)
  real(wp) :: diu(npoi)    = 0.0_wp  !< inverse momentum diffusion coeff, upper canopy
  real(wp) :: exphl(npoi)  = 0.0_wp  !< exp(lambda/2*(z3-z4)), lower canopy
  real(wp) :: exphu(npoi)  = 0.0_wp  !< exp(lambda/2*(z1-z2)), upper canopy
  real(wp) :: expl(npoi)   = 0.0_wp  !< exphl**2
  real(wp) :: expu(npoi)   = 0.0_wp  !< exphu**2
  real(wp) :: pfluxl(npoi) = 0.0_wp  !< heat flux, lower canopy intercepted water (W m-2)
  real(wp) :: pfluxs(npoi) = 0.0_wp  !< heat flux, upper canopy stem intercepted water (W m-2)
  real(wp) :: pfluxu(npoi) = 0.0_wp  !< heat flux, upper canopy leaf intercepted water (W m-2)
  real(wp) :: rhoa(npoi)   = 0.0_wp  !< moist air density at za (kg m-3)
  real(wp) :: richl(npoi)  = 0.0_wp  !< Richardson number, z3 to z2
  real(wp) :: richu(npoi)  = 0.0_wp  !< Richardson number, z1 to za
  real(wp) :: sg(npoi)     = 0.0_wp  !< air-soil transfer coefficient
  real(wp) :: si(npoi)     = 0.0_wp  !< air-snow transfer coefficient
  real(wp) :: strahl(npoi) = 0.0_wp  !< heat/vap stability correction, lower intercanopy
  real(wp) :: strahu(npoi) = 0.0_wp  !< heat/vap stability correction, above upper canopy
  real(wp) :: straml(npoi) = 0.0_wp  !< momentum stability correction, lower intercanopy
  real(wp) :: stramu(npoi) = 0.0_wp  !< momentum stability correction, above upper canopy
  real(wp) :: u1(npoi)     = 0.0_wp  !< wind speed at z1 (m s-1)
  real(wp) :: u12(npoi)    = 0.0_wp  !< wind speed at z12 (m s-1)
  real(wp) :: u2(npoi)     = 0.0_wp  !< wind speed at z2 (m s-1)
  real(wp) :: u3(npoi)     = 0.0_wp  !< wind speed at z3 (m s-1)
  real(wp) :: u34(npoi)    = 0.0_wp  !< wind speed at z34 (m s-1)
  real(wp) :: u4(npoi)     = 0.0_wp  !< wind speed at z4 (m s-1)
  real(wp) :: za(npoi)     = 0.0_wp  !< height of atmospheric forcing level (m)
  real(wp) :: z1(npoi)     = 0.0_wp  !< effective top of upper canopy (m)
  real(wp) :: z12(npoi)    = 0.0_wp  !< effective middle of upper canopy (m)
  real(wp) :: z2(npoi)     = 0.0_wp  !< effective bottom of upper canopy (m)
  real(wp) :: z3(npoi)     = 0.0_wp  !< effective top of lower canopy (m)
  real(wp) :: z34(npoi)    = 0.0_wp  !< effective middle of lower canopy (m)
  real(wp) :: z4(npoi)     = 0.0_wp  !< effective bottom of lower canopy (m)

  ! ---------------------------------------------------------------------------
  ! /com1d9/  --  pressure-ratio scalar
  ! ---------------------------------------------------------------------------
  public :: tfac

  !> (p_surf / p_ref)**(rair/cair); constant for the atmospheric forcing level.
  real(wp) :: tfac = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /com1d10/  --  canopy wetness fractions
  ! ---------------------------------------------------------------------------
  public :: fwetl, fwets, fwetu, fwetlx, fwetsx, fwetux
  public :: rliql, rliqs, rliqu

  !> Fraction of lower canopy leaf+stem area wetted by intercepted water/snow.
  real(wp) :: fwetl(npoi)  = 0.0_wp

  !> Fraction of upper canopy stem area wetted.
  real(wp) :: fwets(npoi)  = 0.0_wp

  !> Fraction of upper canopy leaf area wetted.
  real(wp) :: fwetu(npoi)  = 0.0_wp

  !> Fraction of lower canopy area that would be wetted if dew forms.
  real(wp) :: fwetlx(npoi) = 0.0_wp

  !> Fraction of upper canopy stems that would be wetted if dew forms.
  real(wp) :: fwetsx(npoi) = 0.0_wp

  !> Fraction of upper canopy leaves that would be wetted if dew forms.
  real(wp) :: fwetux(npoi) = 0.0_wp

  !> Proportion of fwetl due to liquid water.
  real(wp) :: rliql(npoi)  = 0.0_wp

  !> Proportion of fwets due to liquid water.
  real(wp) :: rliqs(npoi)  = 0.0_wp

  !> Proportion of fwetu due to liquid water.
  real(wp) :: rliqu(npoi)  = 0.0_wp

end module mod_canopy_fluxes
