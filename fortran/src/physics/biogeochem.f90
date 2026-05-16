! =============================================================================
! biogeochem.f90  --  Soil biogeochemistry routines for IBIS
!
! Converted from biogeochem.f (Fortran 77 fixed-form) to F2008 free-form.
!
! Subroutines provided:
!   soilbgc(iyear,iyear0,imonth,iday,nspinsoil,spin,spinmax)
!          -- daily soil carbon and nitrogen cycling
! =============================================================================


! -----------------------------------------------------------------------------
subroutine soilbgc(iyear, iyear0, imonth, iday, nspinsoil, spin, spinmax)
! Advances soil carbon and nitrogen pools by one day using the modified
! Verberne/CENTURY decomposition scheme.  Handles litter inputs, microbial
! growth, turnover of slow and passive pools, leaching, and N mineralisation.
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npoi
  use mod_run_params,   only: ndaypy
  use mod_soil_state,   only: hsoi, sand, clay
  use mod_carbon_pools, only: clitlm, clitls, clitll, clitrm, clitrs, clitrl, &
                               clitwm, clitws, clitwl, csoipas, csoislon,      &
                               csoislop, decompl, decomps,                     &
                               falll, fallr, fallw
  use mod_veg_dynamics, only: tco2mic, tnmin, totalit, totanlit, totcmic,      &
                               totcsoi, totfall, totlit, totnlit, totnmic,      &
                               totnsoi, totrlit, totrnlit
  use mod_summary,      only: storedn, yrleach, ynleach
  use mod_atm_state,    only: precip
  use mod_bgc_params,   only: bgc_params_init,                                  &
                               lig_frac, fbsom, effac, cnr,                     &
                               fmax, rconst, cnleaf, cnroot, cnwood,            &
                               klm, kls, kll, krm, krs, krl,                   &
                               kwm, kws, kwl, kbn, kbp, knb, kns, kpb, kps, ksb, &
                               ylm, yls, yll, yrm, yrs, yrl,                   &
                               ywm, yws, ywl, ybn, ybp, ynb, yns, ypb, yps, ysb
  implicit none

  integer, intent(in) :: iyear, iyear0, imonth, iday
  integer, intent(in) :: nspinsoil, spin, spinmax

  logical, save :: first_call = .true.

  integer  :: i, kk, msand, mclay, gridpt
  real(wp) :: totts, fracll, fracls, fraclm, fracrl, fracrs, fracrm
  real(wp) :: fracwl, fracws, fracwm
  real(wp) :: fleach, h20, rdepth, cfrac, texfact, fsand, fclay, fbpom

  ! Local C pool arrays (working variables)
  real(wp) :: outclm(npoi), outcls(npoi), outcll(npoi)
  real(wp) :: outcrm(npoi), outcrs(npoi), outcrl(npoi)
  real(wp) :: outcwm(npoi), outcws(npoi), outcwl(npoi)
  real(wp) :: outcsb(npoi), outcps(npoi), outcns(npoi)
  real(wp) :: outcnb(npoi), outcpb(npoi), outcbp(npoi), outcbn(npoi)
  real(wp) :: totc(npoi), dbdt(npoi), dcndt(npoi), dcpdt(npoi), dcsdt(npoi)
  real(wp) :: totmin(npoi), totimm(npoi), netmin(npoi)
  real(wp) :: cmicn(npoi), cmicp(npoi), cmicmx(npoi)
  real(wp) :: cleach(npoi), totcbegin(npoi), totcend(npoi)
  real(wp) :: totcin(npoi), fixsoin(npoi), deposn(npoi)

  ! Local N pool arrays (derived from C pools by C:N ratio)
  real(wp) :: nlitlm(npoi), nlitls(npoi), nlitll(npoi)
  real(wp) :: nlitrm(npoi), nlitrs(npoi), nlitrl(npoi)
  real(wp) :: nlitwm(npoi), nlitws(npoi), nlitwl(npoi)
  real(wp) :: nsoislop(npoi), nsoipas(npoi), nsoislon(npoi)

  ! N mineralisation/immobilisation tracking
  real(wp) :: nbiors(npoi), nbiols(npoi), nbiows(npoi), nbiowm(npoi)
  real(wp) :: nbiolm(npoi), nbiorm(npoi), nbioslon(npoi), nbioslop(npoi)
  real(wp) :: nbiopas(npoi)
  real(wp) :: nminrs(npoi), nminls(npoi), nminws(npoi), nminwm(npoi)
  real(wp) :: nminlm(npoi), nminrm(npoi), nminslon(npoi), nminslop(npoi)
  real(wp) :: nminpas(npoi)
  real(wp) :: nrelps(npoi), nrelns(npoi), nrelbn(npoi), nrelbp(npoi)
  real(wp) :: nrelll(npoi), nrelrl(npoi), nrelwl(npoi), totnrel(npoi)
  real(wp) :: ymintot(npoi), yminmic(npoi)

  ! Initialise BGC parameters on first call
  if (first_call) then
    call bgc_params_init()
    first_call = .false.
  end if

  gridpt = npoi
  totts  = 1.0_wp / real(ndaypy, wp)

  ! -------------------------------------------------------------------------
  ! Litter split fractions (Verberne / Geoderma 1997 erratum)
  ! -------------------------------------------------------------------------
  fracll = fmax * cnleaf**2 / (rconst + cnleaf**2)
  fracls = (1.0_wp/cnleaf - fracll/cnr(5) - (1.0_wp-fracll)/cnr(7)) / &
           (1.0_wp/cnr(6) - 1.0_wp/cnr(7))
  fraclm = 1.0_wp - fracll - fracls

  fracrl = fmax * cnroot**2 / (rconst + cnroot**2)
  fracrs = (1.0_wp/cnroot - fracrl/cnr(5) - (1.0_wp-fracrl)/cnr(7)) / &
           (1.0_wp/cnr(6) - 1.0_wp/cnr(7))
  fracrm = 1.0_wp - fracrl - fracrs

  fracwl = fmax * cnwood**2 / (rconst + cnwood**2)
  fracws = (1.0_wp/cnwood - fracwl/cnr(5) - (1.0_wp-fracwl)/cnr(7)) / &
           (1.0_wp/cnr(8) - 1.0_wp/cnr(7))
  fracwm = 1.0_wp - fracwl - fracws

  ! -------------------------------------------------------------------------
  ! Grid-point loop
  ! -------------------------------------------------------------------------
  do i = 1, npoi

    ! -----------------------------------------------------------------------
    ! Texture factors (top 1 m of soil = 4 layers)
    ! -----------------------------------------------------------------------
    rdepth  = 1.0_wp / (hsoi(1) + hsoi(2) + hsoi(3) + hsoi(4))
    cfrac   = 0.0_wp
    texfact = 0.0_wp

    do kk = 1, 4
      msand   = nint(sand(i, kk))
      mclay   = nint(clay(i, kk))
      fclay   = 0.01_wp * real(mclay, wp)
      fsand   = 0.01_wp * real(msand, wp)
      cfrac   = cfrac   + fclay * hsoi(kk)
      texfact = texfact + fsand * hsoi(kk)
    end do

    cfrac   = cfrac   * rdepth
    texfact = texfact * rdepth

    fbpom = 0.50_wp   ! protected fraction of decomposing biomass

    ! -----------------------------------------------------------------------
    ! Initialisations
    ! -----------------------------------------------------------------------
    if (iday == 1 .and. imonth == 1 .and. iyear == iyear0) then
      totcbegin(i) = 0.0_wp
      storedn(i)   = 0.0_wp
    end if

    if (iday == 1 .and. imonth == 1) then
      yrleach(i) = 0.0_wp
      cleach(i)  = 0.0_wp
      ynleach(i) = 0.0_wp
      ymintot(i) = 0.0_wp
      yminmic(i) = 0.0_wp
    end if

    totcin(i) = (falll(i) + fallr(i) + fallw(i)) * totts

    totc(i) = clitlm(i) + clitls(i) + clitrm(i) + clitrs(i) + &
              clitwm(i) + clitws(i) + csoislop(i) + csoislon(i) + &
              csoipas(i) + totcmic(i) + clitll(i) + clitrl(i) + clitwl(i)

    totcbegin(i) = totc(i)

    ! -----------------------------------------------------------------------
    ! Protected / non-protected microbial biomass split
    ! -----------------------------------------------------------------------
    cmicmx(i) = fbsom * totc(i)

    if (totcmic(i) >= cmicmx(i)) then
      cmicp(i) = cmicmx(i)
      cmicn(i) = totcmic(i) - cmicmx(i)
    else
      cmicn(i) = 0.0_wp
      cmicp(i) = totcmic(i)
    end if

    ! -----------------------------------------------------------------------
    ! Add daily litterfall to litter pools
    ! -----------------------------------------------------------------------
    clitlm(i) = clitlm(i) + fraclm * falll(i) * totts
    clitls(i) = clitls(i) + fracls * falll(i) * totts
    clitll(i) = clitll(i) + fracll * falll(i) * totts
    clitrm(i) = clitrm(i) + fracrm * fallr(i) * totts
    clitrs(i) = clitrs(i) + fracrs * fallr(i) * totts
    clitrl(i) = clitrl(i) + fracrl * fallr(i) * totts
    clitwm(i) = clitwm(i) + fracwm * fallw(i) * totts
    clitws(i) = clitws(i) + fracws * fallw(i) * totts
    clitwl(i) = clitwl(i) + fracwl * fallw(i) * totts

    ! -----------------------------------------------------------------------
    ! Decomposition: outflows from each litter/SOM pool
    ! -----------------------------------------------------------------------
    outcrs(i) = min(decomps(i) * krs * clitrs(i), clitrs(i))
    outcws(i) = min(decompl(i) * kws * clitws(i), clitws(i))
    outcls(i) = min(decompl(i) * kls * clitls(i), clitls(i))
    outclm(i) = min(decompl(i) * klm * clitlm(i), clitlm(i))
    outcrm(i) = min(decomps(i) * krm * clitrm(i), clitrm(i))
    outcwm(i) = min(decompl(i) * kwm * clitwm(i), clitwm(i))
    outcnb(i) = min(decomps(i) * knb * csoislon(i), csoislon(i))
    outcpb(i) = min(decomps(i) * kpb * csoislop(i), csoislop(i))
    outcsb(i) = min(decomps(i) * ksb * csoipas(i),  csoipas(i))

    ! Microbial biomass turnover (with effac recycle fraction)
    outcbp(i) = min(kbp * cmicp(i), cmicp(i)) * effac
    outcbn(i) = min(kbn * cmicn(i), cmicn(i)) * effac

    ! Net change in microbial biomass
    dbdt(i) =  outcrs(i)*yrs + outcws(i)*yws + outcls(i)*yls + &
               outclm(i)*ylm + outcrm(i)*yrm + outcwm(i)*ywm + &
               outcnb(i)*ynb + outcpb(i)*ypb + outcsb(i)*ysb  &
             - outcbp(i) - outcbn(i)

    ! Lignin outflows
    outcll(i) = min(decompl(i) * kll * clitll(i), clitll(i))
    outcrl(i) = min(decomps(i) * krl * clitrl(i), clitrl(i))
    outcwl(i) = min(decompl(i) * kwl * clitwl(i), clitwl(i))
    outcns(i) = min(decomps(i) * kns * csoislon(i), csoislon(i))

    ! Change in non-protected slow OM
    dcndt(i) =   lig_frac * (outcll(i)*yll + outcrl(i)*yrl + outcwl(i)*ywl) &
               + (1.0_wp - fbpom) * (ybn*outcbn(i) + ybp*outcbp(i))        &
               - outcnb(i) - outcns(i)

    ! Change in protected slow OM
    outcps(i) = min(decomps(i) * kps * csoislop(i), csoislop(i))

    dcpdt(i) =   lig_frac * (outcll(i)*yll + outcrl(i)*yrl + outcwl(i)*ywl) &
               + fbpom * (ybn*outcbn(i) + ybp*outcbp(i))                    &
               - outcpb(i) - outcps(i)

    ! Leaching
    h20    = 0.30e-03_wp
    fleach = h20 / 18.0_wp * (0.01_wp + 0.04_wp * texfact)

    ! Change in passive (stabilised) OM
    dcsdt(i) = (yns*outcns(i) + yps*outcps(i)) - outcsb(i) - fleach*csoipas(i)

    cleach(i)  = fleach * (csoipas(i) + csoislop(i) + csoislon(i))

    ynleach(i) = ynleach(i) + fleach*csoipas(i)/cnr(2)  &
                             + fleach*csoislop(i)/cnr(3) &
                             + fleach*csoislon(i)/cnr(4)

    dcndt(i) = dcndt(i) - fleach * csoislon(i)
    dcpdt(i) = dcpdt(i) - fleach * csoislop(i)

    if (spin == spinmax) yrleach(i) = yrleach(i) + cleach(i)

    ! -----------------------------------------------------------------------
    ! Nitrogen mineralisation / immobilisation
    ! -----------------------------------------------------------------------
    ! Structural root decomposition
    if (yrs/cnr(1) > 1.0_wp/cnr(6)) then
      nbiors(i) = (1.0_wp/cnr(6) - yrs/cnr(1)) * outcrs(i)
      nminrs(i) = 0.0_wp
    else
      nminrs(i) = (1.0_wp/cnr(6) - yrs/cnr(1)) * outcrs(i)
      nbiors(i) = 0.0_wp
    end if

    ! Structural leaf decomposition
    if (yls/cnr(1) > 1.0_wp/cnr(6)) then
      nbiols(i) = (1.0_wp/cnr(6) - yls/cnr(1)) * outcls(i)
      nminls(i) = 0.0_wp
    else
      nminls(i) = (1.0_wp/cnr(6) - yls/cnr(1)) * outcls(i)
      nbiols(i) = 0.0_wp
    end if

    ! Structural wood decomposition
    if (yws/cnr(1) > 1.0_wp/cnr(8)) then
      nbiows(i) = (1.0_wp/cnr(8) - yws/cnr(1)) * outcws(i)
      nminws(i) = 0.0_wp
    else
      nminws(i) = (1.0_wp/cnr(8) - yws/cnr(1)) * outcws(i)
      nbiows(i) = 0.0_wp
    end if

    ! Metabolic wood decomposition
    if (ywm/cnr(1) > 1.0_wp/cnr(8)) then
      nbiowm(i) = (1.0_wp/cnr(8) - ywm/cnr(1)) * outcwm(i)
      nminwm(i) = 0.0_wp
    else
      nminwm(i) = (1.0_wp/cnr(8) - ywm/cnr(1)) * outcwm(i)
      nbiowm(i) = 0.0_wp
    end if

    ! Metabolic leaf decomposition
    if (ylm/cnr(1) > 1.0_wp/cnr(7)) then
      nbiolm(i) = (1.0_wp/cnr(7) - ylm/cnr(1)) * outclm(i)
      nminlm(i) = 0.0_wp
    else
      nminlm(i) = (1.0_wp/cnr(7) - ylm/cnr(1)) * outclm(i)
      nbiolm(i) = 0.0_wp
    end if

    ! Metabolic root decomposition
    if (yrm/cnr(1) > 1.0_wp/cnr(7)) then
      nbiorm(i) = (1.0_wp/cnr(7) - yrm/cnr(1)) * outcrm(i)
      nminrm(i) = 0.0_wp
    else
      nminrm(i) = (1.0_wp/cnr(7) - yrm/cnr(1)) * outcrm(i)
      nbiorm(i) = 0.0_wp
    end if

    ! Non-protected OM decomposition
    if (ynb/cnr(1) > 1.0_wp/cnr(4)) then
      nbioslon(i) = (1.0_wp/cnr(4) - ynb/cnr(1)) * outcnb(i)
      nminslon(i) = 0.0_wp
    else
      nminslon(i) = (1.0_wp/cnr(4) - ynb/cnr(1)) * outcnb(i)
      nbioslon(i) = 0.0_wp
    end if

    ! Protected OM decomposition
    if (ypb/cnr(1) > 1.0_wp/cnr(3)) then
      nbioslop(i) = (1.0_wp/cnr(3) - ypb/cnr(1)) * outcpb(i)
      nminslop(i) = 0.0_wp
    else
      nminslop(i) = (1.0_wp/cnr(3) - ypb/cnr(1)) * outcpb(i)
      nbioslop(i) = 0.0_wp
    end if

    ! Passive OM decomposition
    if (ysb/cnr(1) > 1.0_wp/cnr(2)) then
      nbiopas(i) = (1.0_wp/cnr(2) - ysb/cnr(1)) * outcsb(i)
      nminpas(i) = 0.0_wp
    else
      nminpas(i) = (1.0_wp/cnr(2) - ysb/cnr(1)) * outcsb(i)
      nbiopas(i) = 0.0_wp
    end if

    totimm(i) = nbiors(i) + nbiols(i) + nbiows(i) + nbiowm(i) &
              + nbiolm(i) + nbiorm(i) + nbioslon(i) + nbioslop(i) + nbiopas(i)

    totmin(i) = nminrs(i) + nminls(i) + nminws(i) + nminwm(i) &
              + nminlm(i) + nminrm(i) + nminslon(i) + nminslop(i) + nminpas(i)

    nrelps(i) = outcps(i) * (1.0_wp/cnr(3) - 1.0_wp/cnr(2))
    nrelns(i) = outcns(i) * (1.0_wp/cnr(4) - 1.0_wp/cnr(2))
    nrelbn(i) = (1.0_wp-fbpom)*outcbn(i)*(1.0_wp/cnr(1) - 1.0_wp/cnr(4)) &
              + (1.0_wp-fbpom)*outcbp(i)*(1.0_wp/cnr(1) - 1.0_wp/cnr(4))
    nrelbp(i) = fbpom*outcbp(i)*(1.0_wp/cnr(1) - 1.0_wp/cnr(3)) &
              + fbpom*outcbn(i)*(1.0_wp/cnr(1) - 1.0_wp/cnr(3))
    nrelll(i) = lig_frac*outcll(i)*(1.0_wp/cnr(5) - 1.0_wp/cnr(3)) &
              + lig_frac*outcll(i)*(1.0_wp/cnr(5) - 1.0_wp/cnr(4))
    nrelrl(i) = lig_frac*outcrl(i)*(1.0_wp/cnr(5) - 1.0_wp/cnr(3)) &
              + lig_frac*outcrl(i)*(1.0_wp/cnr(5) - 1.0_wp/cnr(4))
    nrelwl(i) = lig_frac*outcwl(i)*(1.0_wp/cnr(5) - 1.0_wp/cnr(3)) &
              + lig_frac*outcwl(i)*(1.0_wp/cnr(5) - 1.0_wp/cnr(4))

    totnrel(i) = nrelps(i) + nrelns(i) + nrelbn(i) + &
                 nrelbp(i) + nrelll(i) + nrelrl(i) + nrelwl(i)

    netmin(i) = totmin(i) + totimm(i) + totnrel(i)
    tnmin(i)  = max(0.0_wp, netmin(i)) / (86400.0_wp * 0.014_wp)

    ! -----------------------------------------------------------------------
    ! Update C pools
    ! -----------------------------------------------------------------------
    totcmic(i)  = max(0.0_wp, totcmic(i)  + dbdt(i))
    csoislon(i) = max(0.0_wp, csoislon(i) + dcndt(i))
    csoislop(i) = max(0.0_wp, csoislop(i) + dcpdt(i))
    csoipas(i)  = max(0.0_wp, csoipas(i)  + dcsdt(i))
    clitlm(i)   = max(0.0_wp, clitlm(i)  - outclm(i))
    clitls(i)   = max(0.0_wp, clitls(i)  - outcls(i))
    clitll(i)   = max(0.0_wp, clitll(i)  - outcll(i))
    clitrm(i)   = max(0.0_wp, clitrm(i)  - outcrm(i))
    clitrs(i)   = max(0.0_wp, clitrs(i)  - outcrs(i))
    clitrl(i)   = max(0.0_wp, clitrl(i)  - outcrl(i))
    clitwm(i)   = max(0.0_wp, clitwm(i)  - outcwm(i))
    clitws(i)   = max(0.0_wp, clitws(i)  - outcws(i))
    clitwl(i)   = max(0.0_wp, clitwl(i)  - outcwl(i))

    ! -----------------------------------------------------------------------
    ! Update N pools (derived from C:N ratios)
    ! -----------------------------------------------------------------------
    totnmic(i)   = totcmic(i)  / cnr(1)
    nsoislon(i)  = csoislon(i) / cnr(4)
    nsoislop(i)  = csoislop(i) / cnr(3)
    nsoipas(i)   = csoipas(i)  / cnr(2)
    nlitlm(i)    = clitlm(i)   / cnr(7)
    nlitls(i)    = clitls(i)   / cnr(6)
    nlitll(i)    = clitll(i)   / cnr(5)
    nlitrm(i)    = clitrm(i)   / cnr(7)
    nlitrs(i)    = clitrs(i)   / cnr(6)
    nlitrl(i)    = clitrl(i)   / cnr(5)
    nlitwm(i)    = clitwm(i)   / cnr(8)
    nlitws(i)    = clitws(i)   / cnr(8)
    nlitwl(i)    = clitwl(i)   / cnr(8)

    ! -----------------------------------------------------------------------
    ! Litter, SOM, and N summaries
    ! -----------------------------------------------------------------------
    totlit(i) = clitlm(i) + clitls(i) + clitll(i) &
              + clitrm(i) + clitrs(i) + clitrl(i)  &
              + clitwm(i) + clitws(i) + clitwl(i)

    totalit(i) = clitlm(i) + clitls(i) + clitwm(i) + clitll(i) + clitws(i) + clitwl(i)

    totrlit(i) = clitrm(i) + clitrs(i) + clitrl(i)

    totcsoi(i) = csoipas(i) + csoislop(i) + totcmic(i) + csoislon(i)

    totfall(i) = falll(i) + fallr(i) + fallw(i)

    totnlit(i) = nlitlm(i) + nlitls(i) + nlitrm(i) + nlitrs(i) &
               + nlitwm(i) + nlitws(i) + nlitll(i) + nlitrl(i) + nlitwl(i)

    totanlit(i) = nlitlm(i) + nlitls(i) + nlitwm(i) + nlitll(i) + nlitws(i) + nlitwl(i)

    totrnlit(i) = nlitrm(i) + nlitrs(i) + nlitrl(i)

    totnsoi(i) = nsoislop(i) + nsoislon(i) + nsoipas(i) + totnmic(i) + totnlit(i)

    ! -----------------------------------------------------------------------
    ! Accumulated N mineralisation and CO2 flux (last spin cycle only)
    ! -----------------------------------------------------------------------
    if (spin == spinmax) then
      storedn(i) = storedn(i) + tnmin(i)

      totcend(i) = totlit(i) + totcsoi(i)

      ! N deposition and N fixation (CENTURY equations, units: kg_N m-2 day-1)
      deposn(i)  = (0.0005753_wp + 0.0028_wp  * precip(i)*0.1_wp) * 1.0e-3_wp
      fixsoin(i) = (-0.0004932_wp + 0.14_wp   * precip(i)*0.1_wp) * 1.0e-3_wp

      ! Daily CO2 flux from microbial decomposition (mol_C s-1)
      tco2mic(i) = (totcbegin(i) + totcin(i) - totcend(i) - cleach(i)) &
                 / (86400.0_wp * 0.012_wp)
    end if

  end do   ! i = 1, npoi

end subroutine soilbgc
