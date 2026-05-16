! =============================================================================
! mod_canopy_params.f90  --  Canopy physical and optical parameters for IBIS
!
! Replaces COMMON blocks /comveg1/, /comveg2/, /comveg3/, /comveg4/,
! /comveg10/, /comveg11/ from comveg.h.
!
! Covers:
!   comveg1  -- canopy physical constants (heat capacities, timing, water limits)
!   comveg2  -- per-point canopy state (temperatures, humidity, transfer coeffs,
!               intercepted water)
!   comveg3  -- per-point canopy-average photosynthesis rates
!   comveg4  -- per-point intercellular CO2, boundary-layer CO2, stomatal
!               conductance
!   comveg10 -- leaf/stem geometry constants (per canopy level)
!   comveg11 -- leaf/stem optical constants (per band per canopy level)
!
! All variables initialised to 0.0; set at run-time by the canopy physics
! and radiation routines.
!
! Replaces: comveg.h (/comveg1/–/comveg4/, /comveg10/, /comveg11/).
! =============================================================================
module mod_canopy_params
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, npft, nband
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /comveg1/: canopy physical constants (scalars, read from params.can)
  ! ---------------------------------------------------------------------------
  public :: alaiml, alaimu, cgrass, chl, chs, chu, cleaf, cstem
  public :: tblowl, tblows, tblowu, tdripl, tdrips, tdripu
  public :: wliqmin, wliqlmax, wliqsmax, wliqumax
  public :: wsnomin, wsnolmax, wsnosmax, wsnoumax
  public :: woodnorm

  !> Lower canopy leaf+stem area (2-sided) for drag coefficient normalisation (m2 m-2).
  real(wp) :: alaiml   = 0.0_wp

  !> Upper canopy leaf+stem area (2-sided) for drag coefficient normalisation (m2 m-2).
  real(wp) :: alaimu   = 0.0_wp

  !> Empirical constant in lower canopy–air aerodynamic transfer coeff (m s-0.5).
  real(wp) :: cgrass   = 0.0_wp

  !> Heat capacity of lower canopy leaves+stems per unit leaf/stem area (J kg-1 m-2).
  real(wp) :: chl      = 0.0_wp

  !> Heat capacity of upper canopy stems per unit stem area (J kg-1 m-2).
  real(wp) :: chs      = 0.0_wp

  !> Heat capacity of upper canopy leaves per unit leaf area (J kg-1 m-2).
  real(wp) :: chu      = 0.0_wp

  !> Empirical constant in upper canopy leaf–air aerodynamic transfer coeff (m s-0.5).
  real(wp) :: cleaf    = 0.0_wp

  !> Empirical constant in upper canopy stem–air aerodynamic transfer coeff (m s-0.5).
  real(wp) :: cstem    = 0.0_wp

  !> Decay time for snow blowoff from lower canopy leaves+stems (s).
  real(wp) :: tblowl   = 0.0_wp

  !> Decay time for snow blowoff from upper canopy stems (s).
  real(wp) :: tblows   = 0.0_wp

  !> Decay time for snow blowoff from upper canopy leaves (s).
  real(wp) :: tblowu   = 0.0_wp

  !> Decay time for liquid dripoff from lower canopy leaves+stems (s).
  real(wp) :: tdripl   = 0.0_wp

  !> Decay time for liquid dripoff from upper canopy stems (s).
  real(wp) :: tdrips   = 0.0_wp

  !> Decay time for liquid dripoff from upper canopy leaves (s).
  real(wp) :: tdripu   = 0.0_wp

  !> Minimum intercepted water on unit vegetated area (kg m-2).
  real(wp) :: wliqmin  = 0.0_wp

  !> Maximum intercepted water per unit lower canopy leaf+stem area (kg m-2).
  real(wp) :: wliqlmax = 0.0_wp

  !> Maximum intercepted water per unit upper canopy stem area (kg m-2).
  real(wp) :: wliqsmax = 0.0_wp

  !> Maximum intercepted water per unit upper canopy leaf area (kg m-2).
  real(wp) :: wliqumax = 0.0_wp

  !> Minimum intercepted snow on unit vegetated area (kg m-2).
  real(wp) :: wsnomin  = 0.0_wp

  !> Intercepted snow capacity for lower canopy leaves+stems (kg m-2).
  real(wp) :: wsnolmax = 0.0_wp

  !> Intercepted snow capacity for upper canopy stems (kg m-2).
  real(wp) :: wsnosmax = 0.0_wp

  !> Intercepted snow capacity for upper canopy leaves (kg m-2).
  real(wp) :: wsnoumax = 0.0_wp

  !> Woody biomass value for upper canopy closure (fu = 1.0) (kg_C m-2).
  real(wp) :: woodnorm = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg2/: per-grid-point canopy state (temperatures, humidity, transfer
  !            coefficients, intercepted water)
  ! ---------------------------------------------------------------------------
  public :: q12, q34, sl, ss, su, topparl, topparu
  public :: tl, ts, tu, tlsub, t12, t34
  public :: wliql, wliqs, wliqu, wsnol, wsnos, wsnou

  !> Specific humidity of air at z12 (kg kg-1).
  real(wp) :: q12(npoi)     = 0.0_wp

  !> Specific humidity of air at z34 (kg kg-1).
  real(wp) :: q34(npoi)     = 0.0_wp

  !> Air–vegetation transfer coefficient (*rhoa) for lower canopy leaves+stems
  !! (m s-1 * kg m-3).
  real(wp) :: sl(npoi)      = 0.0_wp

  !> Air–vegetation transfer coefficient (*rhoa) for upper canopy stems
  !! (m s-1 * kg m-3).
  real(wp) :: ss(npoi)      = 0.0_wp

  !> Air–vegetation transfer coefficient (*rhoa) for upper canopy leaves
  !! (m s-1 * kg m-3).
  real(wp) :: su(npoi)      = 0.0_wp

  !> PAR absorbed by top leaves of lower canopy (W m-2).
  real(wp) :: topparl(npoi) = 0.0_wp

  !> PAR absorbed by top leaves of upper canopy (W m-2).
  real(wp) :: topparu(npoi) = 0.0_wp

  !> Temperature of lower canopy leaves+stems (K).
  real(wp) :: tl(npoi)      = 0.0_wp

  !> Temperature of upper canopy stems (K).
  real(wp) :: ts(npoi)      = 0.0_wp

  !> Temperature of upper canopy leaves (K).
  real(wp) :: tu(npoi)      = 0.0_wp

  !> Temperature of lower canopy vegetation buried by snow (K).
  real(wp) :: tlsub(npoi)   = 0.0_wp

  !> Air temperature at z12 (K).
  real(wp) :: t12(npoi)     = 0.0_wp

  !> Air temperature at z34 (K).
  real(wp) :: t34(npoi)     = 0.0_wp

  !> Intercepted liquid water on lower canopy leaf+stem area (kg m-2).
  real(wp) :: wliql(npoi)   = 0.0_wp

  !> Intercepted liquid water on upper canopy stem area (kg m-2).
  real(wp) :: wliqs(npoi)   = 0.0_wp

  !> Intercepted liquid water on upper canopy leaf area (kg m-2).
  real(wp) :: wliqu(npoi)   = 0.0_wp

  !> Intercepted frozen water (snow) on lower canopy leaf+stem area (kg m-2).
  real(wp) :: wsnol(npoi)   = 0.0_wp

  !> Intercepted frozen water (snow) on upper canopy stem area (kg m-2).
  real(wp) :: wsnos(npoi)   = 0.0_wp

  !> Intercepted frozen water (snow) on upper canopy leaf area (kg m-2).
  real(wp) :: wsnou(npoi)   = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg3/: canopy-average gross and net photosynthesis rates; total stomatal
  !            conductance (per unit leaf area, all per PFT group)
  ! ---------------------------------------------------------------------------
  public :: agcub, agcuc, agcls, agcl3, agcl4
  public :: ancub, ancuc, ancls, ancl3, ancl4
  public :: totcondub, totconduc, totcondls, totcondl3, totcondl4

  !> Canopy-average gross photosynthesis – broadleaf (mol CO2 m-2 s-1).
  real(wp) :: agcub(npoi)     = 0.0_wp

  !> Canopy-average gross photosynthesis – conifer (mol CO2 m-2 s-1).
  real(wp) :: agcuc(npoi)     = 0.0_wp

  !> Canopy-average gross photosynthesis – shrubs (mol CO2 m-2 s-1).
  real(wp) :: agcls(npoi)     = 0.0_wp

  !> Canopy-average gross photosynthesis – C3 grasses (mol CO2 m-2 s-1).
  real(wp) :: agcl3(npoi)     = 0.0_wp

  !> Canopy-average gross photosynthesis – C4 grasses (mol CO2 m-2 s-1).
  real(wp) :: agcl4(npoi)     = 0.0_wp

  !> Canopy-average net photosynthesis – broadleaf (mol CO2 m-2 s-1).
  real(wp) :: ancub(npoi)     = 0.0_wp

  !> Canopy-average net photosynthesis – conifer (mol CO2 m-2 s-1).
  real(wp) :: ancuc(npoi)     = 0.0_wp

  !> Canopy-average net photosynthesis – shrubs (mol CO2 m-2 s-1).
  real(wp) :: ancls(npoi)     = 0.0_wp

  !> Canopy-average net photosynthesis – C3 grasses (mol CO2 m-2 s-1).
  real(wp) :: ancl3(npoi)     = 0.0_wp

  !> Canopy-average net photosynthesis – C4 grasses (mol CO2 m-2 s-1).
  real(wp) :: ancl4(npoi)     = 0.0_wp

  !> Total stomatal conductance – broadleaf (mol CO2 m-2 s-1).
  real(wp) :: totcondub(npoi) = 0.0_wp

  !> Total stomatal conductance – conifer (mol CO2 m-2 s-1).
  real(wp) :: totconduc(npoi) = 0.0_wp

  !> Total stomatal conductance – shrubs (mol CO2 m-2 s-1).
  real(wp) :: totcondls(npoi) = 0.0_wp

  !> Total stomatal conductance – C3 grasses (mol CO2 m-2 s-1).
  real(wp) :: totcondl3(npoi) = 0.0_wp

  !> Total stomatal conductance – C4 grasses (mol CO2 m-2 s-1).
  real(wp) :: totcondl4(npoi) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg4/: intercellular CO2 concentration, leaf boundary-layer CO2, and
  !            stomatal conductance per PFT group
  ! ---------------------------------------------------------------------------
  public :: ciub, ciuc, cils, cil3, cil4
  public :: csub, csuc, csls, csl3, csl4
  public :: gsub, gsuc, gsls, gsl3, gsl4

  !> Intercellular CO2 – broadleaf (mol CO2 mol_air-1).
  real(wp) :: ciub(npoi) = 0.0_wp

  !> Intercellular CO2 – conifer (mol CO2 mol_air-1).
  real(wp) :: ciuc(npoi) = 0.0_wp

  !> Intercellular CO2 – shrubs (mol CO2 mol_air-1).
  real(wp) :: cils(npoi) = 0.0_wp

  !> Intercellular CO2 – C3 plants (mol CO2 mol_air-1).
  real(wp) :: cil3(npoi) = 0.0_wp

  !> Intercellular CO2 – C4 plants (mol CO2 mol_air-1).
  real(wp) :: cil4(npoi) = 0.0_wp

  !> Leaf boundary-layer CO2 – broadleaf (mol CO2 mol_air-1).
  real(wp) :: csub(npoi) = 0.0_wp

  !> Leaf boundary-layer CO2 – conifer (mol CO2 mol_air-1).
  real(wp) :: csuc(npoi) = 0.0_wp

  !> Leaf boundary-layer CO2 – shrubs (mol CO2 mol_air-1).
  real(wp) :: csls(npoi) = 0.0_wp

  !> Leaf boundary-layer CO2 – C3 plants (mol CO2 mol_air-1).
  real(wp) :: csl3(npoi) = 0.0_wp

  !> Leaf boundary-layer CO2 – C4 plants (mol CO2 mol_air-1).
  real(wp) :: csl4(npoi) = 0.0_wp

  !> Stomatal conductance – upper canopy broadleaf (mol CO2 m-2 s-1).
  real(wp) :: gsub(npoi) = 0.0_wp

  !> Stomatal conductance – upper canopy conifer (mol CO2 m-2 s-1).
  real(wp) :: gsuc(npoi) = 0.0_wp

  !> Stomatal conductance – lower canopy shrubs (mol CO2 m-2 s-1).
  real(wp) :: gsls(npoi) = 0.0_wp

  !> Stomatal conductance – lower canopy C3 grasses (mol CO2 m-2 s-1).
  real(wp) :: gsl3(npoi) = 0.0_wp

  !> Stomatal conductance – lower canopy C4 grasses (mol CO2 m-2 s-1).
  real(wp) :: gsl4(npoi) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg10/: leaf/stem geometry constants (index 1 = lower, 2 = upper canopy)
  ! ---------------------------------------------------------------------------
  public :: dleaf, dstem, orieh, oriev

  !> Typical linear leaf dimension in aerodynamic transfer coeff (m).
  real(wp) :: dleaf(2) = 0.0_wp

  !> Typical linear stem dimension in aerodynamic transfer coeff (m).
  real(wp) :: dstem(2) = 0.0_wp

  !> Fraction of leaves/stems with horizontal orientation (dimensionless).
  real(wp) :: orieh(2) = 0.0_wp

  !> Fraction of leaves/stems with vertical orientation (dimensionless).
  real(wp) :: oriev(2) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg11/: leaf/stem optical properties per band per canopy level
  !             (index 1 = VIS, 2 = NIR; canopy 1 = lower, 2 = upper)
  ! ---------------------------------------------------------------------------
  public :: rhoveg, tauveg

  !> Reflectance of an average leaf/stem (nband, 2).
  real(wp) :: rhoveg(nband, 2) = 0.0_wp

  !> Transmittance of an average leaf/stem (nband, 2).
  real(wp) :: tauveg(nband, 2) = 0.0_wp

end module mod_canopy_params
