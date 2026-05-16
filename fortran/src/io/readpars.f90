! =============================================================================
! readpars.f90  --  Parameter file reader for IBIS
!
! Converted from readpars.f (F77 fixed-form) to F2008 free-form.
! All COMMON blocks replaced by USE associations to Phase 1–3 modules.
!
! Subroutines:
!   RD_PARAM           -- reads all parameter files (params.can, .veg, .soi)
!   ReadItem           -- free-format single-item reader with comment handler
!   ReadItems(N,...)   -- free-format multi-item reader (up to 10 items)
!   CommentHandler     -- skips comment lines in parameter files
! =============================================================================

! -----------------------------------------------------------------------------
subroutine RD_PARAM
! -----------------------------------------------------------------------------
  use mod_precision,    only: wp
  use mod_dimensions,   only: npft, nband, nsoilay
  use mod_age_class,    only: npftu
  use mod_pft_params,   only: tau15, kc15, ko15, cimax, &
                                alpha3, theta3, beta3, &
                                alpha4, theta4, beta4, &
                                gammaub, coefmub, coefbub, gsubmin, &
                                gammauc, coefmuc, coefbuc, gsucmin, &
                                gammals, coefmls, coefbls, gslsmin, &
                                gammal4, coefml4, coefbl4, gsl4min, &
                                gammal3, coefml3, coefbl3, gsl3min, &
                                vmax_pft, tauleaf, tauroot, tauwood0, &
                                TminL, TminU, Twarm, GDD, &
                                plai_init, plaiupper, plailower, xminlai, &
                                sapfrac_init, chifuz, chiflz, beta1, beta2
  use mod_veg_dynamics, only: specla, aleaf, aroot, awood
  use mod_canopy_params, only: woodnorm, &
                                rhoveg, tauveg, &
                                alaimu, alaiml, cleaf, cstem, cgrass, &
                                chs, chu, chl, &
                                wliqumax, wliqsmax, wliqlmax, &
                                wsnoumax, wsnosmax, wsnolmax, &
                                tdripu, tdrips, tdripl, &
                                tblowu, tblows, tblowl, &
                                dleaf, dstem
  use mod_soil_state,   only: wpudmax, bperm, zwpmax, hsoi
  use mod_soil_texture, only: ndat, texdat, porosdat, sfielddat, swiltdat, &
                                bexdat, suctiondat, hydrauldat
  use mod_bgc_params,   only: lig_frac, fbsom, effac, cnr, &
                                fmax, rconst, cnleaf, cnroot, cnwood, &
                                klm, kls, kll, krm, krs, krl, kwm, kws, kwl, &
                                kbn, kbp, knb, kns, kpb, kps, ksb, &
                                ylm, yrm, ywm, yls, yrs, yws, yll, yrl, ywl, &
                                ybn, ybp, yps, yns, ysb, ypb, ynb
  implicit none

  ! Local variables
  integer,          parameter :: parm_unit = 9
  integer                     :: j, npft2, npftu2, nsoil2, nveg
  real(wp)                    :: dummyvar
  character(len=20)           :: parm_file

  ! ---------------------------------------------------------------------------
  ! Read params.can: canopy parameters
  ! ---------------------------------------------------------------------------

  parm_file = 'params.can'
  open(unit=parm_unit, file=parm_file, status='OLD', err=9001)

  call readitem(parm_unit, parm_file, tau15)
  call readitem(parm_unit, parm_file, kc15)
  call readitem(parm_unit, parm_file, ko15)
  call readitem(parm_unit, parm_file, cimax)
  call readitem(parm_unit, parm_file, dummyvar)

  npft2 = nint(dummyvar)
  if (npft2 /= npft) then
    write (*, 9003) parm_file, npft2, npft
    goto 9006
  end if

  call readitem(parm_unit, parm_file, dummyvar)

  npftu2 = nint(dummyvar)
  if (npftu2 /= npftu) then
    write (*, 9004) parm_file, npftu2, npftu
    goto 9006
  end if

  ! Standard photosynthesis parameters for C3 and C4 pathways
  call readitem(parm_unit, parm_file, alpha3)
  call readitem(parm_unit, parm_file, theta3)
  call readitem(parm_unit, parm_file, beta3)
  call readitem(parm_unit, parm_file, alpha4)
  call readitem(parm_unit, parm_file, theta4)
  call readitem(parm_unit, parm_file, beta4)

  ! Physiological parameters for broadleaf trees
  call readitems(parm_unit, parm_file, 4, &
    gammaub, coefmub, coefbub, gsubmin, &
    dummyvar, dummyvar, dummyvar, dummyvar, dummyvar, dummyvar)

  ! Physiological parameters for conifer trees
  call readitems(parm_unit, parm_file, 4, &
    gammauc, coefmuc, coefbuc, gsucmin, &
    dummyvar, dummyvar, dummyvar, dummyvar, dummyvar, dummyvar)

  ! Physiological parameters for shrubs
  call readitems(parm_unit, parm_file, 4, &
    gammals, coefmls, coefbls, gslsmin, &
    dummyvar, dummyvar, dummyvar, dummyvar, dummyvar, dummyvar)

  ! Physiological parameters for C4 grasses
  call readitems(parm_unit, parm_file, 4, &
    gammal4, coefml4, coefbl4, gsl4min, &
    dummyvar, dummyvar, dummyvar, dummyvar, dummyvar, dummyvar)

  ! Physiological parameters for C3 grasses
  call readitems(parm_unit, parm_file, 4, &
    gammal3, coefml3, coefbl3, gsl3min, &
    dummyvar, dummyvar, dummyvar, dummyvar, dummyvar, dummyvar)

  ! Vmax and allocation parameters per PFT
  do j = 1, npft
    call readitems(parm_unit, parm_file, 8, &
      vmax_pft(j), specla(j), &
      tauleaf(j), tauroot(j), tauwood0(j), &
      aleaf(j), aroot(j), awood(j), &
      dummyvar, dummyvar)
  end do

  call readitem(parm_unit, parm_file, woodnorm)

  ! Vegetation reflectance per band
  do j = 1, nband
    call readitems(parm_unit, parm_file, 2, &
      rhoveg(j,1), rhoveg(j,2), &
      dummyvar, dummyvar, dummyvar, dummyvar, &
      dummyvar, dummyvar, dummyvar, dummyvar)
  end do

  ! Vegetation transmittance per band
  do j = 1, nband
    call readitems(parm_unit, parm_file, 2, &
      tauveg(j,1), tauveg(j,2), &
      dummyvar, dummyvar, dummyvar, dummyvar, &
      dummyvar, dummyvar, dummyvar, dummyvar)
  end do

  call readitem(parm_unit, parm_file, chifuz)
  call readitem(parm_unit, parm_file, chiflz)

  call readitems(parm_unit, parm_file, 2, dleaf(1), dleaf(2), &
    dummyvar, dummyvar, dummyvar, dummyvar, &
    dummyvar, dummyvar, dummyvar, dummyvar)

  call readitems(parm_unit, parm_file, 2, dstem(1), dstem(2), &
    dummyvar, dummyvar, dummyvar, dummyvar, &
    dummyvar, dummyvar, dummyvar, dummyvar)

  call readitem(parm_unit, parm_file, alaimu)
  call readitem(parm_unit, parm_file, alaiml)

  call readitem(parm_unit, parm_file, cleaf)
  call readitem(parm_unit, parm_file, cstem)
  call readitem(parm_unit, parm_file, cgrass)

  call readitem(parm_unit, parm_file, chs)
  call readitem(parm_unit, parm_file, chu)
  call readitem(parm_unit, parm_file, chl)

  call readitem(parm_unit, parm_file, wliqumax)
  call readitem(parm_unit, parm_file, wliqsmax)
  call readitem(parm_unit, parm_file, wliqlmax)

  call readitem(parm_unit, parm_file, wsnoumax)
  call readitem(parm_unit, parm_file, wsnosmax)
  call readitem(parm_unit, parm_file, wsnolmax)

  call readitem(parm_unit, parm_file, tdripu)
  call readitem(parm_unit, parm_file, tdrips)
  call readitem(parm_unit, parm_file, tdripl)

  call readitem(parm_unit, parm_file, tblowu)
  call readitem(parm_unit, parm_file, tblows)
  call readitem(parm_unit, parm_file, tblowl)

  close (parm_unit)

  ! ---------------------------------------------------------------------------
  ! Read params.veg: vegetation PFT parameters
  ! ---------------------------------------------------------------------------

  parm_file = 'params.veg'
  open(unit=parm_unit, file=parm_file, status='OLD', err=9001)

  do j = 1, npft
    call readitems(parm_unit, parm_file, 4, &
      TminL(j), TminU(j), Twarm(j), GDD(j), &
      dummyvar, dummyvar, dummyvar, &
      dummyvar, dummyvar, dummyvar)
  end do

  call readitem(parm_unit, parm_file, dummyvar)
  nveg = nint(dummyvar)

  do j = 1, nveg
    call readitems(parm_unit, parm_file, 4, &
      plai_init(1,j), plai_init(2,j), &
      plai_init(3,j), plai_init(4,j), &
      dummyvar, dummyvar, dummyvar, &
      dummyvar, dummyvar, dummyvar)
  end do

  call readitem(parm_unit, parm_file, plaiupper)
  call readitem(parm_unit, parm_file, plailower)
  call readitem(parm_unit, parm_file, xminlai)
  call readitem(parm_unit, parm_file, sapfrac_init)
  call readitem(parm_unit, parm_file, beta1)
  call readitem(parm_unit, parm_file, beta2)

  close (parm_unit)

  ! ---------------------------------------------------------------------------
  ! Read params.soi: soil parameters
  ! ---------------------------------------------------------------------------

  parm_file = 'params.soi'
  open(unit=parm_unit, file=parm_file, status='OLD', err=9001)

  do j = 1, nsoilay
    call readitem(parm_unit, parm_file, hsoi(j))
  end do

  call readitem(parm_unit, parm_file, bperm)
  call readitem(parm_unit, parm_file, wpudmax)
  call readitem(parm_unit, parm_file, zwpmax)

  call readitem(parm_unit, parm_file, dummyvar)
  nsoil2 = nint(dummyvar)

  if (nsoil2 /= ndat) then
    write (*, 9031) parm_file, nsoil2
    write (*, 9032) ndat
    goto 9006
  end if

  do j = 1, ndat
    call readitems(parm_unit, parm_file, 9, &
      texdat(1,j), texdat(2,j), texdat(3,j), &
      porosdat(j), sfielddat(j), swiltdat(j), &
      bexdat(j), suctiondat(j), hydrauldat(j), &
      dummyvar)
  end do

  call readitem(parm_unit, parm_file, lig_frac)
  call readitem(parm_unit, parm_file, fbsom)
  call readitem(parm_unit, parm_file, effac)

  call readitems(parm_unit, parm_file, 8, &
    cnr(1), cnr(2), cnr(3), cnr(4), &
    cnr(5), cnr(6), cnr(7), cnr(8), &
    dummyvar, dummyvar)

  call readitems(parm_unit, parm_file, 5, &
    fmax, rconst, cnleaf, cnroot, cnwood, &
    dummyvar, dummyvar, dummyvar, dummyvar, dummyvar)

  call readitems(parm_unit, parm_file, 9, &
    klm, kls, kll, krm, krs, krl, kwm, kws, kwl, &
    dummyvar)

  call readitems(parm_unit, parm_file, 7, &
    kbn, kbp, knb, kns, kpb, kps, ksb, &
    dummyvar, dummyvar, dummyvar)

  call readitems(parm_unit, parm_file, 9, &
    ylm, yrm, ywm, yls, yrs, yws, yll, yrl, ywl, &
    dummyvar)

  call readitems(parm_unit, parm_file, 7, &
    ybn, ybp, yps, yns, ysb, ypb, ynb, &
    dummyvar, dummyvar, dummyvar)

  close (parm_unit)

  write (*,*) 'RD_PARAM: All data read in from parameter files successfully.'

  return

  ! ---------------------------------------------------------------------------
  ! Error handlers
  ! ---------------------------------------------------------------------------

9001 write (*, 2001) parm_file, parm_unit
2001 format ('RD_PARAM: Error opening parameter file ', a20, ' on unit ', i2)
     stop

9003 format ('RD_PARAM Warning: Number of PFTs in ', a10, ' is: ', &
             i2, ' number in mod_dimensions is: ', i2)

9004 format ('RD_PARAM Warning: Number of upper canopy (tree) PFTs ', &
             'in ', a10, ' is: ', i2, ' number in mod_age_class is: ', i2)

9006 write (*, 2006) parm_file, parm_unit
2006 format ('RD_PARAM: Data inconsistency in parameter file ', a20, ' on unit ', i2)
     close (parm_unit)
     stop

9031 format (' RD_PARAM Warning: Number of soil types in ', a10, ' is: ', i2)
9032 format (' Number of soil types in mod_soil_texture is: ', i2)

end subroutine RD_PARAM


! -----------------------------------------------------------------------------
subroutine ReadItem (funit, fname, item)
! -----------------------------------------------------------------------------
! Simple routine to read in data in free format, with a built-in error handler
! designed to locate and skip over comment strings buried in the data stream.
! -----------------------------------------------------------------------------
  use mod_precision, only: wp
  implicit none

  integer,          intent(in)    :: funit
  character(len=*), intent(in)    :: fname
  real(wp),         intent(inout) :: item

  integer :: iocode

100 continue
  read (funit, *, end=999, err=911, iostat=iocode) item
  return

911 call CommentHandler(funit, fname)
    goto 100

999 write (*, 2) fname, funit
  2 format ('RD_PARAM: Unexpected EOF encountered in file ', a20, ' on unit ', i2)
    stop

end subroutine ReadItem


! -----------------------------------------------------------------------------
subroutine ReadItems (funit, fname, N, &
                      item1, item2, item3, item4, item5, &
                      item6, item7, item8, item9, item10)
! -----------------------------------------------------------------------------
! Read up to N data items in free format, with comment-handler error recovery.
! Items item1..item10 are dummy arguments; only the first N are read.
! N must be in the range 1..10.
! -----------------------------------------------------------------------------
  use mod_precision, only: wp
  implicit none

  integer,          intent(in)    :: funit, N
  character(len=*), intent(in)    :: fname
  real(wp),         intent(inout) :: item1, item2, item3, item4, item5
  real(wp),         intent(inout) :: item6, item7, item8, item9, item10

  integer :: iocode

  if ((N > 10) .or. (N < 0)) then
    write (*, 3)
  3 format ('READITEMS: Invalid number of items specified.')
    stop
  end if

100 continue
  select case (N)
  case (1)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1
    return
  case (2)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1, item2
    return
  case (3)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1, item2, item3
    return
  case (4)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1, item2, item3, item4
    return
  case (5)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1, item2, item3, item4, item5
    return
  case (6)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1, item2, item3, item4, item5, item6
    return
  case (7)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1, item2, item3, item4, item5, item6, item7
    return
  case (8)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1, item2, item3, item4, item5, item6, item7, item8
    return
  case (9)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1, item2, item3, item4, item5, item6, item7, item8, item9
    return
  case (10)
    read (funit, *, end=999, err=911, iostat=iocode) &
      item1, item2, item3, item4, item5, item6, item7, item8, item9, item10
    return
  end select
  return

911 call CommentHandler(funit, fname)
    goto 100

999 write (*, 2) fname, funit
  2 format ('RD_PARAM: Unexpected EOF encountered in file ', a20, ' on unit ', i2)
    stop

end subroutine ReadItems


! -----------------------------------------------------------------------------
subroutine CommentHandler (parm_unit, parm_file)
! -----------------------------------------------------------------------------
! Detects and skips comment lines in parameter files.
! Comment indicators: #, *, c, C, !, /
! If a non-comment error is detected, echoes the offending line and stops.
! -----------------------------------------------------------------------------
  implicit none

  integer,          intent(in) :: parm_unit
  character(len=*), intent(in) :: parm_file

  integer           :: iocode, is_comment
  character(len=255) :: inputline

  backspace(parm_unit)

  read (parm_unit, 1002, end=800, err=801, iostat=iocode) inputline
1002 format (a255)

  is_comment = 0

  if ((index(inputline, 'C') == 1) .or. &
      (index(inputline, 'c') == 1) .or. &
      (index(inputline, '*') == 1) .or. &
      (index(inputline, '#') == 1) .or. &
      (index(inputline, '!') == 1) .or. &
      (index(inputline, '/') == 1)) is_comment = 1

  if (is_comment /= 1) goto 801

  return

800 write (*, 1000) parm_file, inputline
1000 format ('CommentHandler: Unexpected EOF encountered in ', &
             a20, '\nLast input was:', a255)
    stop

801 write (*, 1001) parm_file, inputline
1001 format ('CommentHandler: Error encountered reading data ', &
             'from ', a20, '\nLast input was:', a255)
    stop

end subroutine CommentHandler
