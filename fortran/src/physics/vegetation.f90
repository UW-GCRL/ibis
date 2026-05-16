! =============================================================================
! vegetation.f90  --  Phenology and dynamic vegetation routines for IBIS
!
! Converted from vegetation.f (Fortran 77 fixed-form) to F2008 free-form.
!
! Subroutines provided:
!   pheno()            -- update leaf phenology and canopy LAI
!   dynaveg(isimfire)  -- annual dynamic vegetation update
!   fire()             -- fire disturbance
!   vegmap()           -- classify grid cells into vegetation types
! =============================================================================


! -----------------------------------------------------------------------------
subroutine pheno()
! Updates leaf phenology indices (tempu, templ, dropu, dropls, dropl4, dropl3)
! and derives lai, sai, canopy height, and PFT fractions for each grid cell.
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi
  use mod_constants,  only: epsilon
  use mod_atm_state,  only: td
  use mod_summary,    only: a10td, a10ancub, a10ancls, a10ancl4, a10ancl3
  use mod_veg_dynamics, only: tc, fl, fu, lai, zbot, ztop,          &
                               plai, frac, tempu, templ,             &
                               agddu, agddl, dropu, dropls, dropl4, dropl3
  implicit none

  integer  :: i
  real(wp) :: ddays, ddfac, tthreshold, gthreshold
  real(wp) :: avglaiu, avglail, fu_phys

  ddays = 15.0_wp
  ddfac = 1.0_wp / ddays

  do i = 1, npoi

    ! -----------------------------------------------------------------------
    ! Upper canopy winter phenology
    ! -----------------------------------------------------------------------
    tthreshold = max(0.0_wp + 273.16_wp, tc(i) + 5.0_wp + 273.16_wp)
    gthreshold = 0.0_wp + 273.16_wp

    if (a10td(i) < gthreshold) then
      agddu(i) = 0.0_wp
    else
      agddu(i) = agddu(i) + td(i) - gthreshold
    end if

    if (a10td(i) < tthreshold) then
      tempu(i) = max(0.0_wp, tempu(i) - ddfac)
    else
      tempu(i) = min(1.0_wp, max(0.0_wp, agddu(i) - 100.0_wp) / 50.0_wp)
    end if

    ! -----------------------------------------------------------------------
    ! Lower canopy winter phenology
    ! -----------------------------------------------------------------------
    tthreshold = 0.0_wp + 273.16_wp
    gthreshold = -5.0_wp + 273.16_wp

    if (a10td(i) < gthreshold) then
      agddl(i) = 0.0_wp
    else
      agddl(i) = agddl(i) + td(i) - gthreshold
    end if

    if (a10td(i) < tthreshold) then
      templ(i) = max(0.0_wp, templ(i) - ddfac)
    else
      templ(i) = min(1.0_wp, max(0.0_wp, agddl(i) - 150.0_wp) / 50.0_wp)
    end if

    ! -----------------------------------------------------------------------
    ! Drought phenology
    ! -----------------------------------------------------------------------
    if (a10ancub(i) < 0.0_wp) dropu(i)  = max(0.1_wp, dropu(i)  - ddfac)
    if (a10ancub(i) >= 0.0_wp) dropu(i) = min(1.0_wp, dropu(i)  + ddfac)

    if (a10ancls(i) < 0.0_wp) dropls(i)  = max(0.1_wp, dropls(i) - ddfac)
    if (a10ancls(i) >= 0.0_wp) dropls(i) = min(1.0_wp, dropls(i) + ddfac)

    if (a10ancl4(i) < 0.0_wp) dropl4(i)  = max(0.1_wp, dropl4(i) - ddfac)
    if (a10ancl4(i) >= 0.0_wp) dropl4(i) = min(1.0_wp, dropl4(i) + ddfac)

    if (a10ancl3(i) < 0.0_wp) dropl3(i)  = max(0.1_wp, dropl3(i) - ddfac)
    if (a10ancl3(i) >= 0.0_wp) dropl3(i) = min(1.0_wp, dropl3(i) + ddfac)

    ! -----------------------------------------------------------------------
    ! Update LAI and PFT fractions
    ! -----------------------------------------------------------------------
    avglaiu = plai(i,1)             &
            + plai(i,2) * dropu(i)  &
            + plai(i,3)             &
            + plai(i,4)             &
            + plai(i,5) * tempu(i)  &
            + plai(i,6)             &
            + plai(i,7) * tempu(i)  &
            + plai(i,8) * tempu(i)

    frac(i,1) = plai(i,1)            / max(avglaiu, epsilon)
    frac(i,2) = plai(i,2) * dropu(i) / max(avglaiu, epsilon)
    frac(i,3) = plai(i,3)            / max(avglaiu, epsilon)
    frac(i,4) = plai(i,4)            / max(avglaiu, epsilon)
    frac(i,5) = plai(i,5) * tempu(i) / max(avglaiu, epsilon)
    frac(i,6) = plai(i,6)            / max(avglaiu, epsilon)
    frac(i,7) = plai(i,7) * tempu(i) / max(avglaiu, epsilon)
    frac(i,8) = plai(i,8) * tempu(i) / max(avglaiu, epsilon)

    avglail = plai(i,9)                              &
            + plai(i,10) * min(templ(i), dropls(i))  &
            + plai(i,11) * min(templ(i), dropl4(i))  &
            + plai(i,12) * min(templ(i), dropl3(i))

    frac(i,9)  = plai(i,9)                              / max(avglail, epsilon)
    frac(i,10) = plai(i,10) * min(templ(i), dropls(i)) / max(avglail, epsilon)
    frac(i,11) = plai(i,11) * min(templ(i), dropl4(i)) / max(avglail, epsilon)
    frac(i,12) = plai(i,12) * min(templ(i), dropl3(i)) / max(avglail, epsilon)

    lai(i,1) = avglail / fl(i)

    ! Note: fu_phys constrains fu for stability but the unconstrained value
    ! is used for lai(i,2) (matching original code where line 189 overrides 188)
    fu_phys  = max(0.25_wp, min(0.975_wp, fu(i)))
    lai(i,2) = avglaiu / fu_phys
    lai(i,2) = avglaiu / fu(i)         ! override: original uses unconstrained fu

    lai(i,1) = min(lai(i,1), 12.0_wp)
    lai(i,2) = min(lai(i,2), 12.0_wp)

    ! -----------------------------------------------------------------------
    ! Canopy height parameters
    ! -----------------------------------------------------------------------
    zbot(i,1) = 0.05_wp
    ztop(i,1) = max(0.25_wp, lai(i,1) * 0.25_wp)
    ztop(i,1) = min(ztop(i,1), zbot(i,2) - 0.5_wp)

  end do

end subroutine pheno


! -----------------------------------------------------------------------------
subroutine dynaveg(isimfire)
! Annual dynamic vegetation update: updates biomass pools, applies disturbances,
! updates fractional cover, canopy height, stem area index, and calls vegmap.
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi, npft
  use mod_veg_dynamics, only: fl, fu, sai, zbot, ztop, plai, biomass,  &
                               sapfrac, totlail, totlaiu, totbiol, totbiou, &
                               disturbf, disturbo, specla, aleaf, aroot, awood, exist
  use mod_carbon_pools, only: cbiol, cbiow, cbior, falll, fallr, fallw, cdisturb
  use mod_pft_params,   only: tauleaf, tauroot, tauwood, tauwood0, xminlai
  use mod_canopy_params, only: woodnorm
  use mod_summary,      only: aynpp, ayanpp, ayneetot, ayanpptot
  implicit none

  integer, intent(in) :: isimfire   !< fire switch (1 = fire active)

  integer  :: i, j
  real(wp) :: sapspeed, trans, saparea, sapvolume, denswood, wood, taufin

  do i = 1, npoi

    ! -----------------------------------------------------------------------
    ! Initialise disturbance and litter fall accumulators
    ! -----------------------------------------------------------------------
    falll(i)     = 0.0_wp
    fallr(i)     = 0.0_wp
    fallw(i)     = 0.0_wp
    cdisturb(i)  = 0.0_wp
    wood         = 0.001_wp

    ! -----------------------------------------------------------------------
    ! Update NPP and biomass pools for each PFT
    ! -----------------------------------------------------------------------
    do j = 1, npft

      aynpp(i,j)  = exist(i,j) * aynpp(i,j)
      ayanpp(i,j) = (aleaf(j) + awood(j)) * aynpp(i,j)

      taufin     = tauwood0(j) / 2.0_wp
      tauwood(j) = tauwood0(j) - (tauwood0(j) - taufin) * (1.0_wp - exist(i,j))

      falll(i) = falll(i) + cbiol(i,j) / tauleaf(j)
      fallr(i) = fallr(i) + cbior(i,j) / tauroot(j)
      fallw(i) = fallw(i) + cbiow(i,j) / tauwood(j)

      cbiol(i,j) = cbiol(i,j) * exp(-1.0_wp/tauleaf(j))  &
                 + aleaf(j) * tauleaf(j) * max(0.0_wp, aynpp(i,j)) &
                 * (1.0_wp - exp(-1.0_wp/tauleaf(j)))

      cbiow(i,j) = cbiow(i,j) * exp(-1.0_wp/tauwood(j))  &
                 + awood(j) * tauwood(j) * max(0.0_wp, aynpp(i,j)) &
                 * (1.0_wp - exp(-1.0_wp/tauwood(j)))

      cbior(i,j) = cbior(i,j) * exp(-1.0_wp/tauroot(j))  &
                 + aroot(j) * tauroot(j) * max(0.0_wp, aynpp(i,j)) &
                 * (1.0_wp - exp(-1.0_wp/tauroot(j)))

      if (j <= 8) wood = wood + max(0.0_wp, cbiow(i,j))

    end do

    ! -----------------------------------------------------------------------
    ! Apply disturbances
    ! -----------------------------------------------------------------------
    disturbf(i) = 0.005_wp
    disturbo(i) = 0.005_wp

    if (isimfire == 1) call fire()

    do j = 1, npft
      cdisturb(i) = cdisturb(i)                                    &
                  + cbiol(i,j) * (disturbf(i) + disturbo(i))       &
                  + cbiow(i,j) * (disturbf(i) + disturbo(i))       &
                  + cbior(i,j) * (disturbf(i) + disturbo(i))

      cbiol(i,j) = max(0.0_wp, cbiol(i,j) * (1.0_wp - disturbf(i) - disturbo(i)))
      cbiow(i,j) = max(0.0_wp, cbiow(i,j) * (1.0_wp - disturbf(i) - disturbo(i)))
      cbior(i,j) = max(0.0_wp, cbior(i,j) * (1.0_wp - disturbf(i) - disturbo(i)))
    end do

    ! -----------------------------------------------------------------------
    ! Maintain minimum biomass and update plai / biomass
    ! -----------------------------------------------------------------------
    do j = 1, npft
      cbiol(i,j)    = max(exist(i,j) * xminlai / specla(j), cbiol(i,j))
      plai(i,j)     = cbiol(i,j) * specla(j)
      biomass(i,j)  = cbiol(i,j) + cbiow(i,j) + cbior(i,j)
    end do

    ! -----------------------------------------------------------------------
    ! Update annual summary fields
    ! -----------------------------------------------------------------------
    ayneetot(i)  = ayneetot(i) - cdisturb(i)
    ayanpptot(i) = sum(ayanpp(i, 1:npft))

    totlaiu(i) = sum(plai(i, 1:8))
    totlail(i) = sum(plai(i, 9:npft))
    totbiou(i) = sum(biomass(i, 1:8))
    totbiol(i) = sum(biomass(i, 9:npft))

    ! -----------------------------------------------------------------------
    ! Update fractional cover
    ! -----------------------------------------------------------------------
    fu(i) = (1.0_wp - exp(-wood)) / (1.0_wp - exp(-woodnorm))
    fu(i) = fu(i) * (1.0_wp - disturbf(i) - disturbo(i))
    fu(i) = max(0.25_wp, min(0.975_wp, fu(i)))

    fl(i) = totlail(i) / 1.0_wp
    fl(i) = fl(i) * (1.0_wp - disturbf(i) - disturbo(i))
    fl(i) = max(0.25_wp, min(0.975_wp, fl(i)))

    ! -----------------------------------------------------------------------
    ! Update upper canopy height
    ! -----------------------------------------------------------------------
    zbot(i,2) = 3.0_wp
    ztop(i,2) = max(zbot(i,2) + 1.0_wp, 2.50_wp * totbiou(i) / fu(i) * 0.75_wp)

    ! -----------------------------------------------------------------------
    ! Stem area index and sapwood fraction
    ! -----------------------------------------------------------------------
    sai(i,1) = 0.050_wp * totlail(i)
    sai(i,2) = 0.250_wp * totlaiu(i)

    sapspeed  = 25.0_wp        ! m/day
    trans     = 0.0025_wp      ! m/day transpiration rate
    saparea   = trans / sapspeed
    sapvolume = saparea * ztop(i,2) * 0.75_wp
    denswood  = 400.0_wp       ! kg/m^3

    sapfrac(i) = min(0.50_wp, max(0.05_wp, sapvolume * denswood / wood))

  end do

  call vegmap()

end subroutine dynaveg


! -----------------------------------------------------------------------------
subroutine fire()
! Calculates fire disturbance probability from litter loading.
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi
  use mod_veg_dynamics, only: firefac, totlit, disturbf
  implicit none

  integer  :: i
  real(wp) :: burn

  do i = 1, npoi
    burn        = firefac(i) * min(1.0_wp, totlit(i) / 0.200_wp)
    disturbf(i) = min(1.0_wp, max(0.0_wp, 1.0_wp - exp(-0.5_wp * burn)))
  end do

end subroutine fire


! -----------------------------------------------------------------------------
subroutine vegmap()
! Classifies each grid cell into one of 15 IBIS vegetation types based on
! canopy leaf area, tree composition, and climate criteria.
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi
  use mod_veg_dynamics, only: totlaiu, totlail, plai, gdd5, gdd0, vegtype0
  implicit none

  integer  :: i, j, domtree
  real(wp) :: maxlai, totlai, treelai, shrublai, grasslai, ratio

  do i = 1, npoi

    treelai  = totlaiu(i)
    shrublai = plai(i,9)  + plai(i,10)
    grasslai = plai(i,11) + plai(i,12)
    totlai   = max(0.01_wp, totlail(i) + totlaiu(i))

    ! Dominant tree type by LAI
    domtree = 0
    maxlai  = 0.0_wp
    do j = 1, 8
      if (plai(i,j) > maxlai) then
        domtree = j
        maxlai  = plai(i,j)
      end if
    end do

    vegtype0(i) = -999.99_wp

    ! Tropical broadleaf evergreen (PFT 1)
    if (domtree == 1) then
      if (treelai > 2.5_wp)          vegtype0(i) =  1.0_wp
      if (treelai <= 2.5_wp)         vegtype0(i) =  9.0_wp
      if (treelai <= 0.5_wp) then
        if (grasslai >= shrublai)    vegtype0(i) = 10.0_wp
        if (shrublai >= grasslai)    vegtype0(i) = 11.0_wp
      end if
    end if

    ! Tropical broadleaf drought-deciduous (PFT 2)
    if (domtree == 2) then
      if (treelai > 2.5_wp)          vegtype0(i) =  2.0_wp
      if (treelai <= 2.5_wp)         vegtype0(i) =  9.0_wp
      if (treelai <= 0.5_wp) then
        if (grasslai >= shrublai)    vegtype0(i) = 10.0_wp
        if (shrublai >= grasslai)    vegtype0(i) = 11.0_wp
      end if
    end if

    ! Warm-temperate broadleaf evergreen (PFT 3)
    if (domtree == 3) then
      if (treelai > 2.5_wp)          vegtype0(i) =  3.0_wp
      if (treelai <= 2.5_wp)         vegtype0(i) =  9.0_wp
      if (treelai <= 0.5_wp) then
        if (grasslai >= shrublai)    vegtype0(i) = 10.0_wp
        if (shrublai >= grasslai)    vegtype0(i) = 11.0_wp
      end if
    end if

    ! Temperate conifer evergreen (PFT 4)
    if (domtree == 4) then
      if (treelai > 1.5_wp)          vegtype0(i) =  4.0_wp
      if (treelai <= 1.5_wp)         vegtype0(i) =  9.0_wp
      if (treelai <= 0.5_wp) then
        if (grasslai >= shrublai)    vegtype0(i) = 10.0_wp
        if (shrublai >= grasslai)    vegtype0(i) = 11.0_wp
      end if
    end if

    ! Temperate broadleaf cold-deciduous (PFT 5)
    if (domtree == 5) then
      if (treelai > 1.5_wp)          vegtype0(i) =  5.0_wp
      if (treelai <= 1.5_wp)         vegtype0(i) =  9.0_wp
      if (treelai <= 0.5_wp) then
        if (grasslai >= shrublai)    vegtype0(i) = 10.0_wp
        if (shrublai >= grasslai)    vegtype0(i) = 11.0_wp
      end if
    end if

    ! Boreal conifer evergreen (PFT 6)
    if (domtree == 6) vegtype0(i) =  6.0_wp

    ! Boreal broadleaf cold-deciduous (PFT 7)
    if (domtree == 7) vegtype0(i) =  7.0_wp

    ! Boreal conifer cold-deciduous (PFT 8)
    if (domtree == 8) vegtype0(i) =  7.0_wp

    ! Temperate/boreal forest mixtures (PFTs 4-8)
    if (domtree >= 4 .and. domtree <= 8) then
      ratio = (plai(i,5) + plai(i,7) + plai(i,8)) / &
              (plai(i,4) + plai(i,5) + plai(i,6) +   &
               plai(i,7) + plai(i,8))
      if (treelai > 1.0_wp) then
        if (ratio > 0.45_wp .and. ratio < 0.55_wp) vegtype0(i) = 8.0_wp
      end if
      if (domtree <= 5 .and. treelai <= 1.0_wp) then
        if (grasslai >= shrublai) vegtype0(i) = 10.0_wp
        if (shrublai >= grasslai) vegtype0(i) = 11.0_wp
      end if
    end if

    ! No dominant tree
    if (domtree == 0) then
      if (treelai > 1.0_wp)          vegtype0(i) =  9.0_wp
      if (treelai <= 1.0_wp) then
        if (grasslai >= shrublai)    vegtype0(i) = 10.0_wp
        if (shrublai >= grasslai)    vegtype0(i) = 11.0_wp
      end if
    end if

    ! Override by total LAI
    if (totlai < 1.0_wp)             vegtype0(i) = 12.0_wp
    if (totlai <= 0.4_wp)            vegtype0(i) = 14.0_wp

    ! Override by climate
    if (gdd5(i) < 350.0_wp) then
      if (totlai >= 0.4_wp)          vegtype0(i) = 13.0_wp
      if (totlai < 0.4_wp)           vegtype0(i) = 15.0_wp
    end if
    if (gdd0(i) < 100.0_wp)          vegtype0(i) = 15.0_wp

  end do

end subroutine vegmap
