! =============================================================================
! utilities.f90  --  General utility routines for IBIS
!
! Converted from utilities.f (Fortran 77 fixed-form) to F2008 free-form.
!
! Subroutines / functions provided:
!   scopy(nt, arr, brr)          -- copy first nt elements of arr into brr
!   const(arr, nar, value)       -- fill arr with value
!   endrun()                     -- graceful stop
!   cvmgt(x, y, l)               -- CRAY cvmgt compatibility shim (= merge)
!   arr2vec(array, vect)         -- extract land points from 2-D array
!   vec2arr(vect, array)         -- scatter land vector back to 2-D array
!   lenchr(ch)                   -- index of last non-blank/non-null character
!
! Note: cvmgt(x, y, l) is retained for call-site compatibility with canopy.f
! and other routines.  It is equivalent to the F90 intrinsic merge(x, y, l).
! =============================================================================

! -----------------------------------------------------------------------------
subroutine scopy(nt, arr, brr)
! Copies the first nt elements of arr into brr.
! -----------------------------------------------------------------------------
  use mod_precision, only: wp
  implicit none

  integer,  intent(in)  :: nt
  real(wp), intent(in)  :: arr(nt)
  real(wp), intent(out) :: brr(nt)

  integer :: ia

  do ia = 1, nt
    brr(ia) = arr(ia)
  end do

end subroutine scopy


! -----------------------------------------------------------------------------
subroutine const(arr, nar, value)
! Sets all nar elements of real vector arr to value.
! -----------------------------------------------------------------------------
  use mod_precision, only: wp
  implicit none

  integer,  intent(in)    :: nar
  real(wp), intent(in)    :: value
  real(wp), intent(inout) :: arr(nar)

  integer :: j

  do j = 1, nar
    arr(j) = value
  end do

end subroutine const


! -----------------------------------------------------------------------------
subroutine endrun()
! Stops the model gracefully.
! -----------------------------------------------------------------------------
  implicit none
  stop
end subroutine endrun


! -----------------------------------------------------------------------------
real(wp) function cvmgt(x, y, l)
! CRAY cvmgt compatibility: returns x if l is .true., y otherwise.
! Equivalent to the Fortran 90 intrinsic merge(x, y, l).
! -----------------------------------------------------------------------------
  use mod_precision, only: wp
  implicit none

  real(wp), intent(in) :: x, y
  logical,  intent(in) :: l

  cvmgt = merge(x, y, l)

end function cvmgt


! -----------------------------------------------------------------------------
subroutine arr2vec(array, vect)
! Extracts land points from a 2-D lon×lat array and packs them into the
! compact land-point vector vect(npoi).  Uses lmask to identify land cells.
! -----------------------------------------------------------------------------
  use mod_precision,        only: wp
  use mod_dimensions,       only: npoi
  use mod_boundary_climate, only: lmask
  use mod_run_params,       only: nlonsub, nlatsub
  implicit none

  real(wp), intent(in)  :: array(nlonsub, nlatsub)
  real(wp), intent(out) :: vect(npoi)

  integer :: j, i, npts

  npts = 0

  do j = 1, nlatsub
    do i = 1, nlonsub
      if (lmask(i,j) == 1) then
        npts = npts + 1
        vect(npts) = array(i,j)
      end if
    end do
  end do

  if (npts /= npoi) then
    write(*,*) 'ERROR in arr2vec'
    write(*,*) 'npts not equal to npoi'
    write(*,*) 'npts = ', npts, '  npoi = ', npoi
    stop 1
  end if

end subroutine arr2vec


! -----------------------------------------------------------------------------
subroutine vec2arr(vect, array)
! Scatters the compact land-point vector vect(npoi) back into the 2-D
! lon×lat array array, filling non-land points with ocean_fill.
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi
  use mod_work,       only: ocean_fill, lonindex, latindex
  use mod_run_params, only: nlonsub, nlatsub
  implicit none

  real(wp), intent(in)  :: vect(npoi)
  real(wp), intent(out) :: array(nlonsub, nlatsub)

  integer :: j, i, ii, jj

  do j = 1, nlatsub
    do i = 1, nlonsub
      array(i,j) = ocean_fill
    end do
  end do

  do i = 1, npoi
    ii = lonindex(i)
    jj = latindex(i)
    array(ii,jj) = vect(i)
  end do

end subroutine vec2arr


! -----------------------------------------------------------------------------
integer function lenchr(ch)
! Returns the position of the last non-blank, non-null character in ch,
! or 0 if ch is empty or consists entirely of blanks/nulls.
! (F77 callers depended on a minimum return of 1, but F2008 allows
! zero-length string arguments so we return 0 in that case.)
! -----------------------------------------------------------------------------
  implicit none

  character(len=*), intent(in) :: ch

  integer :: i

  do i = len(ch), 1, -1
    if (ch(i:i) /= ' ' .and. ch(i:i) /= char(0)) then
      lenchr = i
      return
    end if
  end do

  lenchr = 0

end function lenchr
