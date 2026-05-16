! =============================================================================
! mod_carbon_pools.f90  --  Carbon pool state variables
!
! Replaces COMMON blocks /comveg6/ and /comveg7/ from comveg.h.
!
! /comveg6/ holds living plant biomass C per PFT (leaf, root, wood).
! /comveg7/ holds litter C pools (leaf/root/wood × metabolic/structural/lignin)
! and soil organic matter C pools (passive, slow protected, slow unprotected),
! plus decomposition factors and annual flux diagnostics.
!
! Array dimensions:
!   npoi -- number of land points
!   npft -- number of plant functional types (= 12)
!
! Replaces: comveg.h (COMMON /comveg6/, /comveg7/).
! =============================================================================
module mod_carbon_pools
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, npft
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! /comveg6/  --  living biomass carbon per PFT
  ! ---------------------------------------------------------------------------
  public :: cbiol, cbior, cbiow

  !> Carbon in leaf biomass pool per PFT (kg_C m-2).
  real(wp) :: cbiol(npoi, npft) = 0.0_wp

  !> Carbon in fine-root biomass pool per PFT (kg_C m-2).
  real(wp) :: cbior(npoi, npft) = 0.0_wp

  !> Carbon in woody biomass pool per PFT (kg_C m-2).
  real(wp) :: cbiow(npoi, npft) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /comveg7/  --  litter carbon pools and soil organic matter
  ! ---------------------------------------------------------------------------
  public :: clitll, clitlm, clitls
  public :: clitrl, clitrm, clitrs
  public :: clitwl, clitwm, clitws
  public :: csoipas, csoislo, csoislon, csoislop
  public :: decompl, decomps
  public :: falll, fallr, fallw
  public :: cdisturb

  ! -- Leaf litter pools (kg_C m-2) --
  !> Carbon in leaf litter — lignin fraction.
  real(wp) :: clitll(npoi) = 0.0_wp
  !> Carbon in leaf litter — metabolic fraction.
  real(wp) :: clitlm(npoi) = 0.0_wp
  !> Carbon in leaf litter — structural fraction.
  real(wp) :: clitls(npoi) = 0.0_wp

  ! -- Fine-root litter pools (kg_C m-2) --
  !> Carbon in fine-root litter — lignin fraction.
  real(wp) :: clitrl(npoi) = 0.0_wp
  !> Carbon in fine-root litter — metabolic fraction.
  real(wp) :: clitrm(npoi) = 0.0_wp
  !> Carbon in fine-root litter — structural fraction.
  real(wp) :: clitrs(npoi) = 0.0_wp

  ! -- Woody litter pools (kg_C m-2) --
  !> Carbon in woody litter — lignin fraction.
  real(wp) :: clitwl(npoi) = 0.0_wp
  !> Carbon in woody litter — metabolic fraction.
  real(wp) :: clitwm(npoi) = 0.0_wp
  !> Carbon in woody litter — structural fraction.
  real(wp) :: clitws(npoi) = 0.0_wp

  ! -- Soil organic matter pools (kg_C m-2) --
  !> Carbon in passive humus pool.
  real(wp) :: csoipas(npoi)  = 0.0_wp
  !> Carbon in slow humus pool (combined).
  real(wp) :: csoislo(npoi)  = 0.0_wp
  !> Carbon in slow non-protected humus pool.
  real(wp) :: csoislon(npoi) = 0.0_wp
  !> Carbon in slow protected humus pool.
  real(wp) :: csoislop(npoi) = 0.0_wp

  ! -- Decomposition modifiers (dimensionless) --
  !> Litter decomposition rate factor.
  real(wp) :: decompl(npoi)  = 0.0_wp
  !> Soil organic matter decomposition rate factor.
  real(wp) :: decomps(npoi)  = 0.0_wp

  ! -- Annual litter flux diagnostics (kg_C m-2 yr-1) --
  !> Annual leaf litter fall.
  real(wp) :: falll(npoi)    = 0.0_wp
  !> Annual root litter input.
  real(wp) :: fallr(npoi)    = 0.0_wp
  !> Annual wood litter fall.
  real(wp) :: fallw(npoi)    = 0.0_wp

  !> Annual vegetation C lost to atmosphere via fire (biomass burning) (kg_C m-2 yr-1).
  real(wp) :: cdisturb(npoi) = 0.0_wp

end module mod_carbon_pools
