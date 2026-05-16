! =============================================================================
! math.f90  --  Mathematical utility routines for IBIS
!
! Converted from math.f (Fortran 77 fixed-form) to F2008 free-form.
!
! Subroutines / functions provided:
!   ran2(idum)                         -- long-period random number generator
!   linsolve(arr, rhs, vec, mplate, nd) -- vectorised Gaussian elimination
!   tridia(ns, nd, ne, a, b, c, y, x, alpha, gamma) -- tridiagonal solver
!
! Reference for ran2: Numerical Recipes in Fortran 77, §7.1 (Press et al.).
! Reference for tridia: based on NCAR ULIB trdi by Starley Thompson.
! =============================================================================

! -----------------------------------------------------------------------------
real function ran2(idum)
! Long-period (> 2×10^18) random number generator with shuffle.
! Returns a uniform deviate in (0, 1).  Call with idum < 0 to initialise.
! idum must not be altered between calls except to reinitialise.
!
! NOTE: This function intentionally returns default real (single precision)
! to match the F77 original (math.f), which declared it as 'real function'.
! Callers must declare 'real, external :: ran2'.
! -----------------------------------------------------------------------------
  implicit none

  integer, intent(inout) :: idum

  integer, parameter :: im1   = 2147483563
  integer, parameter :: im2   = 2147483399
  integer, parameter :: imm1  = im1 - 1
  integer, parameter :: ia1   = 40014
  integer, parameter :: ia2   = 40692
  integer, parameter :: iq1   = 53668
  integer, parameter :: iq2   = 52774
  integer, parameter :: ir1   = 12211
  integer, parameter :: ir2   = 3791
  integer, parameter :: ntab  = 32
  integer, parameter :: ndiv  = 1 + imm1 / ntab

  ! am, eps, rnmx are intentionally single-precision (real, not real(wp)) to
  ! match the F77 original (math.f), which declared them as plain 'real'.
  ! Using double precision here would produce subtly different ran2 return
  ! values (differing at the ~7th digit), causing the Box-Muller and gamma
  ! rejection-sampling logic in weather.f90 to advance the seed by different
  ! amounts and permanently diverge from the F77 output.
  real, parameter :: am   = 1.0 / im1
  real, parameter :: eps  = 1.0e-7
  real, parameter :: rnmx = 1.0 - eps

  integer :: idum2, j, k
  integer :: iv(ntab), iy

  save iv, iy, idum2
  data idum2 / 123456789 /, iv / ntab*0 /, iy / 0 /

  if (idum <= 0) then
    idum  = max(-idum, 1)
    idum2 = idum
    do j = ntab + 8, 1, -1
      k    = idum / iq1
      idum = ia1 * (idum - k*iq1) - k*ir1
      if (idum < 0) idum = idum + im1
      if (j <= ntab) iv(j) = idum
    end do
    iy = iv(1)
  end if

  k    = idum / iq1
  idum = ia1 * (idum - k*iq1) - k*ir1
  if (idum < 0) idum = idum + im1

  k     = idum2 / iq2
  idum2 = ia2 * (idum2 - k*iq2) - k*ir2
  if (idum2 < 0) idum2 = idum2 + im2

  j     = 1 + iy / ndiv
  iy    = iv(j) - idum2
  iv(j) = idum
  if (iy < 1) iy = iy + imm1

  ran2 = min(am * iy, rnmx)

end function ran2


! -----------------------------------------------------------------------------
subroutine linsolve(arr, rhs, vec, mplate, nd)
! Solves multiple linear systems of equations simultaneously, vectorising
! over npoi land points.  Uses basic Gaussian elimination without pivoting.
!
! A template array mplate identifies structurally zero elements, allowing
! work to be skipped for entries that remain zero throughout elimination.
! The template is copied internally so the caller's mplate is unchanged.
!
! arr(npoi, nd, nd) -- equation coefficient matrices (overwritten)
! rhs(npoi, nd)     -- right-hand sides (overwritten)
! vec(npoi, nd)     -- solution (returned)
! mplate(nd, nd)    -- zero-pattern template (non-zero means entry may be non-zero)
! nd               -- number of equations (must be <= ndx = 9)
! -----------------------------------------------------------------------------
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi
  implicit none

  integer,  intent(in)    :: nd
  integer,  intent(in)    :: mplate(nd, nd)
  real(wp), intent(inout) :: arr(npoi, nd, nd)
  real(wp), intent(inout) :: rhs(npoi, nd)
  real(wp), intent(out)   :: vec(npoi, nd)

  integer, parameter :: ndx = 9

  integer  :: mplatex(ndx, ndx)
  real(wp) :: f(npoi)
  integer  :: id, i, j, m

  if (nd > ndx) then
    write(*,900) nd, ndx
900 format(/' *** fatal error ***' / &
            ' number of linsolve eqns', i4, ' exceeds limit', i4)
    call endrun()
  end if

  ! Copy zero-pattern template (may be updated below)
  mplatex(1:nd, 1:nd) = mplate(1:nd, 1:nd)

  ! Forward elimination: zero all elements below the diagonal
  do id = 1, nd - 1
    do i = id + 1, nd
      if (mplatex(i, id) /= 0) then
        do m = 1, npoi
          f(m) = arr(m, i, id) / arr(m, id, id)
        end do
        do j = id, nd
          if (mplatex(id, j) /= 0) then
            do m = 1, npoi
              arr(m, i, j) = arr(m, i, j) - f(m) * arr(m, id, j)
            end do
            mplatex(i, j) = 1
          end if
        end do
        do m = 1, npoi
          rhs(m, i) = rhs(m, i) - f(m) * rhs(m, id)
        end do
      end if
    end do
  end do

  ! Back-substitution: solve in reverse row order
  do id = nd, 1, -1
    f(1:npoi) = 0.0_wp
    if (id < nd) then
      do j = id + 1, nd
        if (mplatex(id, j) /= 0) then
          do m = 1, npoi
            f(m) = f(m) + arr(m, id, j) * vec(m, j)
          end do
        end if
      end do
    end if
    do m = 1, npoi
      vec(m, id) = (rhs(m, id) - f(m)) / arr(m, id, id)
    end do
  end do

end subroutine linsolve


! -----------------------------------------------------------------------------
subroutine tridia(ns, nd, ne, a, b, c, y, x, alpha, gamma)
! Solves ns tridiagonal linear systems simultaneously.
!
! Arguments:
!   ns          -- number of systems to solve (first array dimension in use)
!   nd          -- declared first dimension of arrays (>= ns)
!   ne          -- number of unknowns per system (> 2; second dimension)
!   a(nd, ne)   -- subdiagonals; a(j,2)...a(j,ne) used
!   b(nd, ne)   -- main diagonals; b(j,1)...b(j,ne) used
!   c(nd, ne)   -- superdiagonals; c(j,1)...c(j,ne-1) used
!   y(nd, ne)   -- right-hand sides (overwritten)
!   x(nd, ne)   -- solutions (returned)
!   alpha(nd,ne)-- work array
!   gamma(nd,ne)-- work array
!
! Algorithm: LU decomposition followed by forward/back substitution.
! Adapted from the NCAR ULIB routine trdi (Schneider & Thompson, JGR 1981).
! -----------------------------------------------------------------------------
  use mod_precision, only: wp
  implicit none

  integer,  intent(in)    :: ns, nd, ne
  real(wp), intent(in)    :: a(nd, ne), b(nd, ne), c(nd, ne)
  real(wp), intent(inout) :: y(nd, ne)
  real(wp), intent(out)   :: x(nd, ne)
  real(wp), intent(out)   :: alpha(nd, ne), gamma(nd, ne)

  integer :: nm1, i, j, ib

  nm1 = ne - 1

  ! LU decomposition
  do j = 1, ns
    alpha(j, 1) = 1.0_wp / b(j, 1)
    gamma(j, 1) = c(j, 1) * alpha(j, 1)
  end do
  do i = 2, nm1
    do j = 1, ns
      alpha(j, i) = 1.0_wp / (b(j,i) - a(j,i) * gamma(j, i-1))
      gamma(j, i) = c(j, i) * alpha(j, i)
    end do
  end do

  ! Forward substitution
  do j = 1, ns
    x(j, 1) = y(j, 1) * alpha(j, 1)
  end do
  do i = 2, nm1
    do j = 1, ns
      x(j, i) = (y(j,i) - a(j,i) * x(j, i-1)) * alpha(j, i)
    end do
  end do
  do j = 1, ns
    x(j, ne) = (y(j,ne) - a(j,ne) * x(j, nm1)) / &
               (b(j,ne) - a(j,ne) * gamma(j, nm1))
  end do

  ! Back substitution
  do i = 1, nm1
    ib = ne - i
    do j = 1, ns
      x(j, ib) = x(j, ib) - gamma(j, ib) * x(j, ib+1)
    end do
  end do

end subroutine tridia
