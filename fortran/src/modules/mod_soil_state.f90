! =============================================================================
! mod_soil_state.f90  --  Soil thermal, hydraulic, and surface state variables
!
! Replaces COMMON blocks /comsoi1/, /comsoi2/, /comsoi4/ through /comsoi8/
! from comsoi.h.
!
! Note: /comsoi3/ is absent from the original file (gap from prior refactoring).
!
! /comsoi1/: scalar hydraulic boundary conditions; set by RD_PARAM.
! /comsoi2/: per-point soil/snow surface energy variables.
! /comsoi4/: soil layer thickness vector; set by RD_PARAM reading params.soi.
! /comsoi5/: per-point × per-layer soil thermal and hydraulic state.
! /comsoi6/: downward heat transport through soil layers.
! /comsoi7/: integer nint(bex) for computational speed.
! /comsoi8/: soil evaporation source fractions.
!
! Array dimensions:
!   npoi    -- number of land points
!   nsoilay -- number of soil layers (= 6)
!
! Replaces: comsoi.h (COMMON /comsoi1/, /comsoi2/, /comsoi4/–/comsoi8/).
! =============================================================================
module mod_soil_state
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nsoilay
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /comsoi1/  --  scalar soil hydraulic boundary parameters (set by readpars)
  ! ---------------------------------------------------------------------------
  public :: wpudmax, zwpmax, bperm

  !> Normalisation constant for surface puddles (kg m-2).
  real(wp) :: wpudmax = 0.0_wp

  !> Maximum fraction of soil surface covered by puddles (dimensionless).
  real(wp) :: zwpmax  = 0.0_wp

  !> Lower boundary condition for soil drainage
  !! (0 = impermeable; 1 = fully permeable).
  real(wp) :: bperm   = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsoi2/  --  per-land-point surface energy state
  ! ---------------------------------------------------------------------------
  public :: wpud, wipud, z0soi, albsav, albsan
  public :: stresstl, stresstu
  public :: heati, heatg, hvasug, hvasui
  public :: tg, ti

  !> Liquid content of surface puddles (kg_H2O m-2).
  real(wp) :: wpud(npoi)     = 0.0_wp

  !> Ice content of surface puddles (kg_H2O m-2).
  real(wp) :: wipud(npoi)    = 0.0_wp

  !> Roughness length of bare soil surface (m).
  real(wp) :: z0soi(npoi)    = 0.0_wp

  !> Saturated soil surface albedo, visible waveband.
  real(wp) :: albsav(npoi)   = 0.0_wp

  !> Saturated soil surface albedo, near-IR waveband.
  real(wp) :: albsan(npoi)   = 0.0_wp

  !> Column sum of lower-canopy soil moisture stress (dimensionless).
  real(wp) :: stresstl(npoi) = 0.0_wp

  !> Column sum of upper-canopy soil moisture stress (dimensionless).
  real(wp) :: stresstu(npoi) = 0.0_wp

  !> Net heat flux into snow surface (W m-2).
  real(wp) :: heati(npoi)    = 0.0_wp

  !> Net heat flux into soil surface (W m-2).
  real(wp) :: heatg(npoi)    = 0.0_wp

  !> Latent heat of vap/subl at soil surface (J kg-1).
  real(wp) :: hvasug(npoi)   = 0.0_wp

  !> Latent heat of vap/subl at snow surface (J kg-1).
  real(wp) :: hvasui(npoi)   = 0.0_wp

  !> Soil skin temperature (K).
  real(wp) :: tg(npoi)       = 0.0_wp

  !> Snow skin temperature (K).
  real(wp) :: ti(npoi)       = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsoi4/  --  soil layer thickness (m); set by RD_PARAM reading params.soi
  ! The array has nsoilay+1 elements; the extra element is used as a boundary.
  ! ---------------------------------------------------------------------------
  public :: hsoi

  !> Thickness of each soil layer (m).  Length nsoilay+1.
  real(wp) :: hsoi(nsoilay + 1) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsoi5/  --  per-layer soil thermal and hydraulic state
  ! All arrays: (land_point, soil_layer)
  ! ---------------------------------------------------------------------------
  public :: tsoi, wsoi, wisoi, consoi, csoi
  public :: hydraul, suction, bex, sfield, swilt
  public :: rhosoi, poros, porosflo
  public :: sandpct, claypct, sand, clay
  public :: stressl, stressu, upsoiu, upsoil

  !> Soil temperature (K).
  real(wp) :: tsoi(npoi, nsoilay)     = 0.0_wp

  !> Fraction of pore space filled by liquid water.
  real(wp) :: wsoi(npoi, nsoilay)     = 0.0_wp

  !> Fraction of pore space filled by ice.
  real(wp) :: wisoi(npoi, nsoilay)    = 0.0_wp

  !> Thermal conductivity of soil layer (W m-1 K-1).
  real(wp) :: consoi(npoi, nsoilay)   = 0.0_wp

  !> Volumetric heat capacity of soil (no pores) (J m-3 K-1).
  real(wp) :: csoi(npoi, nsoilay)     = 0.0_wp

  !> Saturated hydraulic conductivity (m s-1).
  real(wp) :: hydraul(npoi, nsoilay)  = 0.0_wp

  !> Saturated matric potential (m H2O).
  real(wp) :: suction(npoi, nsoilay)  = 0.0_wp

  !> Campbell moisture-release exponent b.
  real(wp) :: bex(npoi, nsoilay)      = 0.0_wp

  !> Field capacity (fraction of pore space).
  real(wp) :: sfield(npoi, nsoilay)   = 0.0_wp

  !> Wilting point (fraction of pore space).
  real(wp) :: swilt(npoi, nsoilay)    = 0.0_wp

  !> Soil density (no pores) (kg m-3).
  real(wp) :: rhosoi(npoi, nsoilay)   = 0.0_wp

  !> Porosity (fraction; equals theta at saturation).
  real(wp) :: poros(npoi, nsoilay)    = 0.0_wp

  !> Effective porosity after accounting for ice content.
  real(wp) :: porosflo(npoi, nsoilay) = 0.0_wp

  !> Percent sand of soil from input file.
  real(wp) :: sandpct(npoi, nsoilay)  = 0.0_wp

  !> Percent clay of soil from input file.
  real(wp) :: claypct(npoi, nsoilay)  = 0.0_wp

  !> Percent sand after processing.
  real(wp) :: sand(npoi, nsoilay)     = 0.0_wp

  !> Percent clay after processing.
  real(wp) :: clay(npoi, nsoilay)     = 0.0_wp

  !> Soil moisture stress factor for lower canopy (dimensionless).
  real(wp) :: stressl(npoi, nsoilay)  = 0.0_wp

  !> Soil moisture stress factor for upper canopy (dimensionless).
  real(wp) :: stressu(npoi, nsoilay)  = 0.0_wp

  !> Soil water uptake by upper-canopy transpiration (kg_H2O m-2 s-1).
  real(wp) :: upsoiu(npoi, nsoilay)   = 0.0_wp

  !> Soil water uptake by lower-canopy transpiration (kg_H2O m-2 s-1).
  real(wp) :: upsoil(npoi, nsoilay)   = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsoi6/  --  downward heat transport through soil layer interfaces (W m-2)
  ! Dimensioned (npoi, nsoilay+1): one flux value per layer interface.
  ! ---------------------------------------------------------------------------
  public :: hflo

  !> Downward heat flux at each soil layer boundary (W m-2).
  real(wp) :: hflo(npoi, nsoilay + 1) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comsoi7/  --  integer bex for speed in soil water potential computation
  ! ---------------------------------------------------------------------------
  public :: ibex

  !> nint(bex): integer approximation of Campbell exponent.
  integer :: ibex(npoi, nsoilay) = 0

  ! ---------------------------------------------------------------------------
  ! /comsoi8/  --  soil evaporation source fractions
  ! qglif(i, 1): fraction of fvapg from soil liquid
  ! qglif(i, 2): fraction of fvapg from soil ice
  ! qglif(i, 3): fraction of fvapg from puddle liquid
  ! qglif(i, 4): fraction of fvapg from puddle ice
  ! ---------------------------------------------------------------------------
  public :: qglif

  !> Soil evaporation source fractions (dimensionless).  (land_point, source_type).
  real(wp) :: qglif(npoi, 4) = 0.0_wp

end module mod_soil_state
