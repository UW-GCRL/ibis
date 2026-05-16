! =============================================================================
! mod_soil_texture.f90  --  Soil texture lookup tables
!
! Replaces COMMON block /comtex1/ from comtex.h.
! Contains lookup tables for 11 soil texture classes (excludes organics).
! Values are read from params.soi by subroutine RD_PARAM (readpars.f90).
!
! Original type: real*4 and integer*2 in comtex.h.
! Modernised to: real(wp) and integer (default kind).
!
! Replaces: comtex.h (COMMON /comtex1/).
! =============================================================================
module mod_soil_texture
  use mod_precision, only: wp
  implicit none
  private

  public :: ndat
  public :: texdat, porosdat, sfielddat, swiltdat, bexdat, suctiondat, hydrauldat

  ! ---------------------------------------------------------------------------
  ! Dimension parameter
  ! ---------------------------------------------------------------------------

  !> Number of soil texture classes (excludes organics).
  !! Source: comtex.h parameter (ndat = 11).
  integer, parameter :: ndat = 11

  ! ---------------------------------------------------------------------------
  ! /comtex1/  --  soil hydraulic property lookup tables
  ! Dimensioned (ndat) = 11 soil types.
  ! Set by RD_PARAM reading params.soi.
  ! ---------------------------------------------------------------------------

  !> Sand/silt/clay fractions for each soil type (first index: 1=sand,2=silt,3=clay).
  real(wp) :: texdat(3, ndat) = 0.0_wp

  !> Porosity volume fraction for each soil type.
  real(wp) :: porosdat(ndat) = 0.0_wp

  !> Field capacity volume fraction for each soil type.
  real(wp) :: sfielddat(ndat) = 0.0_wp

  !> Wilting point volume fraction for each soil type.
  real(wp) :: swiltdat(ndat) = 0.0_wp

  !> Campbell moisture-release b exponent for each soil type.
  real(wp) :: bexdat(ndat) = 0.0_wp

  !> Air entry potential (m H2O) for each soil type.
  real(wp) :: suctiondat(ndat) = 0.0_wp

  !> Saturated hydraulic conductivity (m s-1) for each soil type.
  real(wp) :: hydrauldat(ndat) = 0.0_wp

end module mod_soil_texture
