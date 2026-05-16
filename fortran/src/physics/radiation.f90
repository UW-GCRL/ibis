! =============================================================================
! radiation.f90  --  Solar and infra-red radiation routines for IBIS
!
! Converted from radiation.f (Fortran 77 fixed-form) to F2008 free-form.
!
! Subroutines provided:
!   solset()                               -- zero arrays, build coszen>0 index
!   solsur(ib)                             -- set soil and snow surface albedos
!   solalb(ib)                             -- compute effective canopy albedos
!   solarf(ib)                             -- compute absorbed solar fluxes
!   twostr(abvegd,abvegi,refld,refli,      -- two-stream radiative transfer
!          fbeldd,fbeldi,fbelid,fbelii,
!          asurd,asuri,iv,coszen,ib)
!   twoset(omega,betad,betai,avmu,gdir,    -- set two-stream parameters
!          coszen,iv,ib)
!   irrad()                                -- compute absorbed infra-red fluxes
! =============================================================================


! -----------------------------------------------------------------------------
subroutine solset()
! Zeros albedos and internal absorbed solar fluxes; builds the index of land
! points with positive cosine of solar zenith angle (stored in indsol, nsol).
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi
  use mod_atm_state,     only: coszen, asurd, asuri
  use mod_canopy_params, only: topparu, topparl
  use mod_canopy_fluxes, only: solu, sols, soll, solg, soli, &
                                scalcoefl, scalcoefu,          &
                                nsol, indsol
  implicit none

  integer :: i

  ! zero albedos
  asurd = 0.0_wp
  asuri = 0.0_wp

  ! zero absorbed solar fluxes and PAR-related arrays
  solu = 0.0_wp
  sols = 0.0_wp
  soll = 0.0_wp
  solg = 0.0_wp
  soli = 0.0_wp

  topparu  = 0.0_wp
  topparl  = 0.0_wp

  ! zero canopy scaling coefficients for night-time conditions
  scalcoefl = 0.0_wp
  scalcoefu = 0.0_wp

  ! build index of points with positive coszen
  nsol = 0
  do i = 1, npoi
    if (coszen(i) > 0.0_wp) then
      nsol = nsol + 1
      indsol(nsol) = i
    end if
  end do

end subroutine solset


! -----------------------------------------------------------------------------
subroutine solsur(ib)
! Sets soil and snow surface albedos as a function of soil moisture and
! snow temperature, prior to other solar calculations.
! ib = waveband number (1 = visible, 2 = near-IR).
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi
  use mod_constants,     only: tmelt
  use mod_atm_state,     only: coszen
  use mod_snow_state,    only: tsno
  use mod_soil_state,    only: wsoi, wisoi, albsav, albsan
  use mod_canopy_fluxes, only: nsol, indsol, albsnd, albsni, albsod, albsoi
  implicit none

  integer, intent(in) :: ib   ! waveband number (1=VIS, 2=NIR)

  integer  :: j, i
  real(wp) :: a7svlo, a7svhi, a7snlo, a7snhi, t7shi, t7slo, dinc, zw
  real(wp) :: x(npoi), zfac(npoi)

  a7svlo = 0.90_wp;  a7svhi = 0.70_wp
  a7snlo = 0.60_wp;  a7snhi = 0.40_wp
  t7shi  = tmelt
  t7slo  = tmelt - 15.0_wp

  if (nsol == 0) return

  if (ib == 1) then

    ! soil albedos (visible waveband)
    do j = 1, nsol
      i    = indsol(j)
      zw   = wsoi(i,1) * (1.0_wp - wisoi(i,1))
      dinc = 1.0_wp + min(1.0_wp, max(0.0_wp, 1.0_wp - (zw/0.50_wp)))
      albsod(i) = min(albsav(i) * dinc, 0.80_wp)
      albsoi(i) = albsod(i)
    end do

    ! snow albedos (visible waveband)
    do j = 1, nsol
      i    = indsol(j)
      x(i) = (a7svhi*(tsno(i,1) - t7slo) + a7svlo*(t7shi - tsno(i,1))) &
             / (t7shi - t7slo)
      x(i)    = min(a7svlo, max(a7svhi, x(i)))
      zfac(i) = max(0.0_wp, 1.5_wp / (1.0_wp + 4.0_wp*coszen(i)) - 0.5_wp)
      albsnd(i) = min(0.99_wp, x(i) + (1.0_wp - x(i))*zfac(i))
      albsni(i) = min(1.0_wp, x(i))
    end do

  else

    ! soil albedos (near-IR waveband)
    do j = 1, nsol
      i    = indsol(j)
      zw   = wsoi(i,1) * (1.0_wp - wisoi(i,1))
      dinc = 1.0_wp + min(1.0_wp, max(0.0_wp, 1.0_wp - (zw/0.50_wp)))
      albsod(i) = min(albsan(i) * dinc, 0.80_wp)
      albsoi(i) = albsod(i)
    end do

    ! snow albedos (near-IR waveband)
    do j = 1, nsol
      i    = indsol(j)
      x(i) = (a7snhi*(tsno(i,1) - t7slo) + a7snlo*(t7shi - tsno(i,1))) &
             / (t7shi - t7slo)
      x(i)    = min(a7snlo, max(a7snhi, x(i)))
      zfac(i) = max(0.0_wp, 1.5_wp / (1.0_wp + 4.0_wp*coszen(i)) - 0.5_wp)
      albsnd(i) = min(0.99_wp, x(i) + (1.0_wp - x(i))*zfac(i))
      albsni(i) = min(1.0_wp, x(i))
    end do

  end if

end subroutine solsur


! -----------------------------------------------------------------------------
subroutine solalb(ib)
! Calculates effective albedos of the surface system separately for unit
! incoming direct and diffuse flux.  Results are stored in asurd, asuri.
! ib = waveband number (1=VIS, 2=NIR).
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_atm_state,     only: coszen, asurd, asuri
  use mod_snow_state,    only: fi
  use mod_veg_dynamics,  only: fl, fu
  use mod_canopy_fluxes, only: nsol, indsol,                               &
                                albsod, albsoi, albsnd, albsni,             &
                                relod, reloi, reupd, reupi,                 &
                                ablod, abloi, abupd, abupi,                 &
                                flodd, flodi, floii, fupdd, fupdi, fupii,   &
                                dummy_rad
  implicit none

  integer, intent(in) :: ib   ! waveband number

  integer :: j, i

  if (nsol == 0) return

  ! (a) set albedos for two-stream lower-veg + soil system
  do j = 1, nsol
    i = indsol(j)
    asurd(i,ib) = albsod(i)
    asuri(i,ib) = albsoi(i)
  end do

  call twostr(ablod, abloi, relod, reloi, flodd, dummy_rad, &
              flodi, floii, asurd, asuri, 1, coszen, ib)

  ! (b) areally average surface albedos (lower veg, soil, snow)
  do j = 1, nsol
    i = indsol(j)
    asurd(i,ib) =  fl(i)*(1.0_wp - fi(i))*relod(i)            &
                + (1.0_wp - fl(i))*(1.0_wp - fi(i))*albsod(i) &
                +  fi(i)*albsnd(i)
    asuri(i,ib) =  fl(i)*(1.0_wp - fi(i))*reloi(i)            &
                + (1.0_wp - fl(i))*(1.0_wp - fi(i))*albsoi(i) &
                +  fi(i)*albsni(i)
  end do

  ! (c) obtain albedos for two-stream upper-veg + surface system
  call twostr(abupd, abupi, reupd, reupi, fupdd, dummy_rad, &
              fupdi, fupii, asurd, asuri, 2, coszen, ib)

  ! (d) compute average overall albedos
  do j = 1, nsol
    i = indsol(j)
    asurd(i,ib) = fu(i)*reupd(i) + (1.0_wp - fu(i))*asurd(i,ib)
    asuri(i,ib) = fu(i)*reupi(i) + (1.0_wp - fu(i))*asuri(i,ib)
  end do

end subroutine solalb


! -----------------------------------------------------------------------------
subroutine solarf(ib)
! Calculates solar fluxes absorbed by upper/lower canopy, soil, and snow.
! Uses unit-flux results stored by solalb.
! ib = waveband number (1=VIS, 2=NIR).
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_constants,     only: epsilon
  use mod_atm_state,     only: solad, solai
  use mod_veg_dynamics,  only: fl, fu, lai, sai
  use mod_canopy_params, only: topparu, topparl
  use mod_canopy_fluxes, only: nsol, indsol,                     &
                                abupd, abupi, ablod, abloi,      &
                                solu, sols, soll, solg, soli,    &
                                sol2d, sol2i, sol3d, sol3i,      &
                                flodd, flodi, floii,             &
                                fupdd, fupdi, fupii,             &
                                albsod, albsoi, albsnd, albsni,  &
                                scalcoefu, scalcoefl,            &
                                termu, terml
  implicit none

  integer, intent(in) :: ib   ! waveband number

  integer  :: j, i
  real(wp) :: x, y, xd, xi, xaiu, xail

  if (nsol == 0) return

  ! (f) absorbed by upper leaves/stems; downward fluxes below upper veg
  do j = 1, nsol
    i    = indsol(j)
    x    = solad(i,ib)*abupd(i) + solai(i,ib)*abupi(i)
    y    = lai(i,2) / max(lai(i,2) + sai(i,2), epsilon)
    solu(i)  = solu(i) + x * y
    sols(i)  = sols(i) + x * (1.0_wp - y)
    sol2d(i) = solad(i,ib)*fupdd(i)
    sol2i(i) = solad(i,ib)*fupdi(i) + solai(i,ib)*fupii(i)
  end do

  ! (g) areally average fluxes to lower veg, soil, snow
  do j = 1, nsol
    i = indsol(j)
    sol3d(i) = fu(i)*sol2d(i) + (1.0_wp - fu(i))*solad(i,ib)
    sol3i(i) = fu(i)*sol2i(i) + (1.0_wp - fu(i))*solai(i,ib)
  end do

  ! (h,i) absorbed by lower veg, snow-free soil, and snow
  do j = 1, nsol
    i  = indsol(j)
    soll(i) = soll(i) + sol3d(i)*ablod(i) + sol3i(i)*abloi(i)
    xd = (fl(i)*flodd(i) + 1.0_wp - fl(i)) * sol3d(i)
    xi = fl(i)*(sol3d(i)*flodi(i) + sol3i(i)*floii(i)) &
       + (1.0_wp - fl(i)) * sol3i(i)
    solg(i) = solg(i) + (1.0_wp - albsod(i))*xd + (1.0_wp - albsoi(i))*xi
    soli(i) = soli(i) + (1.0_wp - albsnd(i))*sol3d(i) &
                      + (1.0_wp - albsni(i))*sol3i(i)
  end do

  ! estimate absorbed PAR and canopy scaling parameters (visible band only)
  if (ib == 1) then
    do j = 1, nsol
      i = indsol(j)

      ! upper canopy
      xaiu = max(lai(i,2) + sai(i,2), epsilon)
      scalcoefu(i,1) = termu(i,1) * solad(i,ib)
      scalcoefu(i,2) = termu(i,2) * solad(i,ib) + termu(i,3) * solai(i,ib)
      scalcoefu(i,3) = termu(i,4) * solad(i,ib) + termu(i,5) * solai(i,ib)
      scalcoefu(i,4) = scalcoefu(i,1) + scalcoefu(i,2) + scalcoefu(i,3)
      topparu(i) = scalcoefu(i,4) * lai(i,2) / xaiu

      ! lower canopy
      xail = max(lai(i,1) + sai(i,1), epsilon)
      scalcoefl(i,1) = terml(i,1) * sol3d(i)
      scalcoefl(i,2) = terml(i,2) * sol3d(i) + terml(i,3) * sol3i(i)
      scalcoefl(i,3) = terml(i,4) * sol3d(i) + terml(i,5) * sol3i(i)
      scalcoefl(i,4) = scalcoefl(i,1) + scalcoefl(i,2) + scalcoefl(i,3)
      topparl(i) = scalcoefl(i,4) * lai(i,1) / xail
    end do
  end if

end subroutine solarf


! -----------------------------------------------------------------------------
subroutine twostr(abvegd, abvegi, refld, refli, fbeldd, fbeldi, &
                  fbelid, fbelii, asurd, asuri, iv, coszen, ib)
! Solves the two-stream radiative transfer problem for a vegetation layer over
! a surface of known albedo, for unit incoming direct and diffuse flux.
! Returns absorbed fluxes, reflected fluxes, and downward fluxes below the
! layer.  Stores canopy-scaling terms in termu or terml.
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi, nband
  use mod_constants,     only: epsilon
  use mod_veg_dynamics,  only: lai, sai
  use mod_canopy_fluxes, only: nsol, indsol, termu, terml
  implicit none

  real(wp), intent(out) :: abvegd(npoi)      !< direct  absorbed flux
  real(wp), intent(out) :: abvegi(npoi)      !< diffuse absorbed flux
  real(wp), intent(out) :: refld(npoi)       !< reflected flux (unit direct)
  real(wp), intent(out) :: refli(npoi)       !< reflected flux (unit diffuse)
  real(wp), intent(out) :: fbeldd(npoi)      !< downward direct below layer (per unit direct)
  real(wp), intent(out) :: fbeldi(npoi)      !< always 0 (no direct→diffuse scattering)
  real(wp), intent(out) :: fbelid(npoi)      !< downward diffuse below layer (per unit direct)
  real(wp), intent(out) :: fbelii(npoi)      !< downward diffuse below layer (per unit diffuse)
  real(wp), intent(in)  :: asurd(npoi,nband) !< direct  surface albedo
  real(wp), intent(in)  :: asuri(npoi,nband) !< diffuse surface albedo
  integer,  intent(in)  :: iv                !< 1=lower, 2=upper story
  real(wp), intent(in)  :: coszen(npoi)      !< cosine of solar zenith angle
  integer,  intent(in)  :: ib                !< waveband (1=VIS, 2=NIR)

  integer  :: j, i
  real(wp) :: b, c, c0, d, f, h, k, q, p, sigma
  real(wp) :: ud1, ui1, ud2, ui2, ud3, xai, s1, s2, p1, p2, p3, p4
  real(wp) :: rwork, dd1, di1, dd2, di2
  real(wp) :: h1, h2, h3, h4, h5, h6, h7, h8, h9, h10
  real(wp) :: absurd, absuri, tmp0

  real(wp) :: omega(npoi), betad(npoi), betai(npoi)
  real(wp) :: avmu(npoi),  gdir(npoi)

  if (nsol == 0) return

  call twoset(omega, betad, betai, avmu, gdir, coszen, iv, ib)

  do j = 1, nsol
    i = indsol(j)

    b    = 1.0_wp - omega(i)*(1.0_wp - betai(i))
    c    = omega(i)*betai(i)
    tmp0 = b*b - c*c

    q = sqrt(max(0.0_wp, tmp0))
    k = gdir(i) / max(coszen(i), 0.01_wp)
    p = avmu(i) * k

    ! perturb p away from q to avoid singular denominator
    if (abs(p-q) < 0.001_wp*p) p = (1.0_wp + sign(0.001_wp, p-q)) * p

    c0    = omega(i) * p
    d     = c0 * betad(i)
    f     = c0 * (1.0_wp - betad(i))
    h     = q / avmu(i)
    sigma = p*p - tmp0

    ud1 = b - c/asurd(i,ib);  ui1 = b - c/asuri(i,ib)
    ud2 = b - c*asurd(i,ib);  ui2 = b - c*asuri(i,ib)
    ud3 = f + c*asurd(i,ib)

    xai   = max(lai(i,iv) + sai(i,iv), epsilon)
    s1    = exp(-h*xai)
    s2    = exp(-k*xai)
    p1    = b + q;  p2 = b - q
    p3    = b + p;  p4 = b - p
    rwork = 1.0_wp / s1

    dd1 = p1*(ud1-q)*rwork - p2*(ud1+q)*s1
    di1 = p1*(ui1-q)*rwork - p2*(ui1+q)*s1
    dd2 = (ud2+q)*rwork - (ud2-q)*s1
    di2 = (ui2+q)*rwork - (ui2-q)*s1

    h1    = -d*p4 - c*f
    rwork = s2*(d - c - h1*(ud1+p)/sigma)
    h2    =  (1.0_wp/dd1) * ( (d - h1*p3/sigma)*(ud1-q)/s1 - p2*rwork )
    h3    = -(1.0_wp/dd1) * ( (d - h1*p3/sigma)*(ud1+q)*s1 - p1*rwork )
    h4    = -f*p3 - c*d
    rwork = s2*(ud3 - h4*(ud2-p)/sigma)
    h5    = -(1.0_wp/dd2) * ( h4*(ud2+q)/(sigma*s1) + rwork )
    h6    =  (1.0_wp/dd2) * ( h4*s1*(ud2-q)/sigma   + rwork )
    h7    =  c*(ui1-q)/(di1*s1)
    h8    = -c*s1*(ui1+q)/di1
    h9    =  (ui2+q)/(di2*s1)
    h10   = -s1*(ui2-q)/di2

    fbeldd(i) = s2
    fbeldi(i) = 0.0_wp
    fbelid(i) = h4/sigma*s2 + h5*s1 + h6/s1
    fbelii(i) = h9*s1 + h10/s1

    refld(i) = h1/sigma + h2 + h3
    refli(i) = h7 + h8

    absurd  = (1.0_wp - asurd(i,ib))*fbeldd(i) + (1.0_wp - asuri(i,ib))*fbelid(i)
    absuri  = (1.0_wp - asuri(i,ib))*fbelii(i)
    abvegd(i) = max(0.0_wp, 1.0_wp - refld(i) - absurd)
    abvegi(i) = max(0.0_wp, 1.0_wp - refli(i) - absuri)

    ! zero out veg absorption where there is no vegetation
    if (xai < epsilon) then
      abvegd(i) = 0.0_wp
      abvegi(i) = 0.0_wp
    end if

    ! canopy-scaling terms (visible band only)
    if (ib == 1) then
      if (iv == 1) then
        terml(i,1) = k*(1.0_wp + (h4-h1)/sigma)
        terml(i,2) = h*(h5 - h2);  terml(i,3) = h*(h9 - h7)
        terml(i,4) = h*(h3 - h6);  terml(i,5) = h*(h8 - h10)
        terml(i,6) = k;            terml(i,7) = h
      else
        termu(i,1) = k*(1.0_wp + (h4-h1)/sigma)
        termu(i,2) = h*(h5 - h2);  termu(i,3) = h*(h9 - h7)
        termu(i,4) = h*(h3 - h6);  termu(i,5) = h*(h8 - h10)
        termu(i,6) = k;            termu(i,7) = h
      end if
    end if
  end do

end subroutine twostr


! -----------------------------------------------------------------------------
subroutine twoset(omega, betad, betai, avmu, gdir, coszen, iv, ib)
! Sets two-stream parameters omega, betad, betai, avmu, gdir as weighted
! combinations for vertical, horizontal, and random leaf orientations.
! Adjusts for intercepted snow on the canopy.
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi, nband
  use mod_constants,     only: pi, tmelt
  use mod_canopy_params, only: rhoveg, tauveg, oriev, orieh, tl, ts, tu
  use mod_canopy_fluxes, only: nsol, indsol, fwetl, rliql, fwetu, rliqu, &
                                fwets, rliqs
  use mod_veg_dynamics,  only: lai, sai
  implicit none

  real(wp), intent(out) :: omega(npoi)  !< fraction of intercepted radiation scattered
  real(wp), intent(out) :: betad(npoi)  !< upscatter fraction, direct
  real(wp), intent(out) :: betai(npoi)  !< upscatter fraction, diffuse
  real(wp), intent(out) :: avmu(npoi)   !< average diffuse optical depth
  real(wp), intent(out) :: gdir(npoi)   !< average projected leaf area (solar direction)
  real(wp), intent(in)  :: coszen(npoi) !< cosine of solar zenith angle
  integer,  intent(in)  :: iv           !< 1=lower, 2=upper story
  integer,  intent(in)  :: ib           !< waveband (1=VIS, 2=NIR)

  integer  :: j, i, itab
  real(wp) :: zrho, ztau, orand, ztab, rwork, y, o, x
  real(wp) :: otmp(npoi)

  integer, parameter :: ntmu = 100

  !> Lookup table: omega*betad for random orientation as a function of coszen.
  !> Indexed 1..ntmu+1, covers coszen = 0 to 1.
  real(wp), save :: tablemu(ntmu+1)
  real(wp), save :: omegasno(nband)
  real(wp), save :: betadsno, betaisno

  data tablemu / &
    0.5000, 0.4967, 0.4933, 0.4900, 0.4867, 0.4833, 0.4800, 0.4767, &
    0.4733, 0.4700, 0.4667, 0.4633, 0.4600, 0.4567, 0.4533, 0.4500, &
    0.4467, 0.4433, 0.4400, 0.4367, 0.4333, 0.4300, 0.4267, 0.4233, &
    0.4200, 0.4167, 0.4133, 0.4100, 0.4067, 0.4033, 0.4000, 0.3967, &
    0.3933, 0.3900, 0.3867, 0.3833, 0.3800, 0.3767, 0.3733, 0.3700, &
    0.3667, 0.3633, 0.3600, 0.3567, 0.3533, 0.3500, 0.3467, 0.3433, &
    0.3400, 0.3367, 0.3333, 0.3300, 0.3267, 0.3233, 0.3200, 0.3167, &
    0.3133, 0.3100, 0.3067, 0.3033, 0.3000, 0.2967, 0.2933, 0.2900, &
    0.2867, 0.2833, 0.2800, 0.2767, 0.2733, 0.2700, 0.2667, 0.2633, &
    0.2600, 0.2567, 0.2533, 0.2500, 0.2467, 0.2433, 0.2400, 0.2367, &
    0.2333, 0.2300, 0.2267, 0.2233, 0.2200, 0.2167, 0.2133, 0.2100, &
    0.2067, 0.2033, 0.2000, 0.1967, 0.1933, 0.1900, 0.1867, 0.1833, &
    0.1800, 0.1767, 0.1733, 0.1700, 0.1667 /

  data omegasno  / 0.9, 0.7 /
  data betadsno, betaisno / 0.5, 0.5 /

  ! set two-stream parameters as weighted sum over three orientations
  do j = 1, nsol
    i = indsol(j)

    zrho  = rhoveg(ib,iv)
    ztau  = tauveg(ib,iv)
    orand = 1.0_wp - oriev(iv) - orieh(iv)

    omega(i) = zrho + ztau

    itab  = nint(coszen(i)*real(ntmu, wp) + 1.0_wp)
    ztab  = tablemu(itab)
    rwork = 1.0_wp / omega(i)

    betad(i) = (  oriev(iv) * 0.5_wp*(zrho + ztau)                       &
                + orieh(iv) * zrho                                         &
                + orand     * ((1.0_wp - ztab)*zrho + ztab*ztau) ) * rwork

    betai(i) = (  oriev(iv) * 0.5_wp*(zrho + ztau)                       &
                + orieh(iv) * zrho                                         &
                + orand     * ((2.0_wp/3.0_wp)*zrho + &
                               (1.0_wp/3.0_wp)*ztau) ) * rwork

    gdir(i) = oriev(iv) * (2.0_wp/pi) * sqrt(max(0.0_wp, 1.0_wp - coszen(i)**2)) &
            + orieh(iv) * coszen(i)                                                &
            + orand     * 0.5_wp

    avmu(i) = 1.0_wp
  end do

  ! adjust omega, betad, betai for intercepted snow
  if (iv == 1) then

    ! lower story
    do j = 1, nsol
      i       = indsol(j)
      y       = fwetl(i) * (1.0_wp - rliql(i))
      o       = omegasno(ib) * (0.6_wp + 0.4_wp * &
                max(0.0_wp, min(1.0_wp, (tmelt - tl(i))/1.0_wp)))
      otmp(i) = omega(i)
      rwork   = y * o
      omega(i) = (1.0_wp-y)*otmp(i) + rwork
      betad(i) = ((1.0_wp-y)*otmp(i)*betad(i) + rwork*betadsno) / omega(i)
      betai(i) = ((1.0_wp-y)*otmp(i)*betai(i) + rwork*betaisno) / omega(i)
    end do

  else

    ! upper story
    do j = 1, nsol
      i = indsol(j)
      x = lai(i,iv) / max(lai(i,iv) + sai(i,iv), 1.0e-7_wp)
      y = x*(fwetu(i)*(1.0_wp - rliqu(i))) &
        + (1.0_wp - x)*(fwets(i)*(1.0_wp - rliqs(i)))
      o = (   x           * min(1.0_wp, max(0.6_wp, (tmelt-tu(i))/0.1_wp)) &
            + (1.0_wp-x)  * min(1.0_wp, max(0.6_wp, (tmelt-ts(i))/0.1_wp)) ) &
          * omegasno(ib)
      otmp(i) = omega(i)
      rwork   = y * o
      omega(i) = (1.0_wp-y)*otmp(i) + rwork
      betad(i) = ((1.0_wp-y)*otmp(i)*betad(i) + rwork*betadsno) / omega(i)
      betai(i) = ((1.0_wp-y)*otmp(i)*betai(i) + rwork*betaisno) / omega(i)
    end do

  end if

end subroutine twoset


! -----------------------------------------------------------------------------
subroutine irrad()
! Calculates overall emitted IR flux and net absorbed-minus-emitted IR fluxes
! for upper leaves, upper stems, lower story, soil, and snow.
! -----------------------------------------------------------------------------
  use mod_precision,     only: wp
  use mod_dimensions,    only: npoi
  use mod_constants,     only: stef
  use mod_atm_state,     only: fira
  use mod_snow_state,    only: fi
  use mod_soil_state,    only: tg, ti
  use mod_veg_dynamics,  only: fl, fu, lai, sai
  use mod_canopy_params, only: tl, ts, tu
  use mod_canopy_fluxes, only: firb, firu, firs, firl, firg, firi
  implicit none

  integer  :: i
  real(wp), parameter :: emisoil = 0.95_wp   !< soil emissivity
  real(wp), parameter :: emisnow = 0.95_wp   !< snow emissivity
  real(wp), parameter :: avmuir  = 1.0_wp    !< average diffuse optical depth for IR

  real(wp) :: emu(npoi), ems(npoi), eml(npoi)
  real(wp) :: emg(npoi), emi(npoi)
  real(wp) :: fdown(npoi), fdowng(npoi), fup(npoi)
  real(wp) :: fupg(npoi), fupgb(npoi), fupi(npoi)

  do i = 1, npoi
    emu(i) = 1.0_wp - exp(-lai(i,2) / avmuir)
    ems(i) = 1.0_wp - exp(-sai(i,2) / avmuir)
    eml(i) = 1.0_wp - exp(-(lai(i,1) + sai(i,1)) / avmuir)

    emg(i) = emisoil
    emi(i) = emisnow

    fdown(i)  = (1.0_wp - fu(i)) * fira(i)                                &
              + fu(i) * ( (1.0_wp - emu(i))*(1.0_wp - ems(i))*fira(i)   &
                        + emu(i)*(1.0_wp - ems(i))*stef*tu(i)**4         &
                        + ems(i)*stef*ts(i)**4 )

    fdowng(i) = (1.0_wp - eml(i))*fdown(i) + eml(i)*stef*tl(i)**4

    fupg(i)   = (1.0_wp - emg(i))*fdowng(i) + emg(i)*stef*tg(i)**4
    fupgb(i)  = (1.0_wp - emg(i))*fdown(i)  + emg(i)*stef*tg(i)**4
    fupi(i)   = (1.0_wp - emi(i))*fdown(i)  + emi(i)*stef*ti(i)**4

    fup(i) = (1.0_wp - fi(i)) * (      fl(i)  * (      eml(i)*stef*tl(i)**4  &
                                                  + (1.0_wp-eml(i))*fupg(i) ) &
                                 + (1.0_wp-fl(i))*fupgb(i) )                  &
           + fi(i) * fupi(i)

    firb(i) = (1.0_wp - fu(i)) * fup(i)                                   &
            + fu(i) * ( (1.0_wp - emu(i))*(1.0_wp - ems(i))*fup(i)      &
                      + emu(i)*stef*tu(i)**4                               &
                      + ems(i)*(1.0_wp - emu(i))*stef*ts(i)**4 )

    firu(i) =   emu(i)*ems(i)*stef*ts(i)**4           &
              + emu(i)*(1.0_wp - ems(i))*fup(i)       &
              + emu(i)*fira(i)                         &
              - 2.0_wp*emu(i)*stef*tu(i)**4

    firs(i) =   ems(i)*emu(i)*stef*tu(i)**4           &
              + ems(i)*fup(i)                          &
              + ems(i)*(1.0_wp - emu(i))*fira(i)      &
              - 2.0_wp*ems(i)*stef*ts(i)**4

    firl(i) = eml(i)*fdown(i) + eml(i)*fupg(i) - 2.0_wp*eml(i)*stef*tl(i)**4

    firg(i) =      fl(i)  * (fdowng(i) - fupg(i))  &
              + (1.0_wp-fl(i)) * (fdown(i) - fupgb(i))

    firi(i) = fdown(i) - fupi(i)
  end do

end subroutine irrad
