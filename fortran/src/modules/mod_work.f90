! =============================================================================
! mod_work.f90  --  I/O workspace arrays and grid coordinate vectors
!
! Replaces COMMON blocks /work1/ through /work4/ from comwork.h.
!
! Holds grid-index and coordinate arrays used by the I/O routines to map
! between the compact land-point array (npoi) and the full lon×lat grid.
! Also holds the scratch workspace arrays used during binary file I/O.
!
! Dimension parameters:
!   ndim2 = nlon * nlat         -- total grid points (one full field)
!   ndim4 = nlon                -- largest single dimension
!   ndim3 = nlon * nlat * ndim4 -- maximum I/O scratch size (one field × ndim4)
!
! Note: cdummy is declared allocatable because ndim3 can exceed 180 million
! elements for global domains.  The I/O initialisation routine must call
! allocate(cdummy(ndim3)) before any I/O that requires it.
!
! ocean_fill is the fill value written to non-land grid points in output files.
!
! The character variable 'aname' from comwork.h (declared but not in any
! COMMON block) is a local variable of the I/O routines; it is not included
! here.
!
! Replaces: comwork.h (COMMON /work1/–/work4/).
! =============================================================================
module mod_work
  use mod_precision,  only: wp
  use mod_dimensions, only: npoi, nlon, nlat
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! Compile-time parameters
  ! ---------------------------------------------------------------------------
  public :: ocean_fill, ndim2, ndim3, ndim4

  !> Fill value for non-land (ocean) grid points in output files.
  real(wp), parameter :: ocean_fill = 9.0e20_wp

  !> Total number of grid points in one full lon×lat field.
  integer,  parameter :: ndim2 = nlon * nlat

  !> Largest individual grid dimension (used to size the I/O scratch buffer).
  integer,  parameter :: ndim4 = nlon

  !> Maximum scratch buffer size = one full field × ndim4.
  integer,  parameter :: ndim3 = nlon * nlat * ndim4

  ! ---------------------------------------------------------------------------
  ! /work1/  --  land-point index arrays
  ! ---------------------------------------------------------------------------
  public :: lonindex, latindex

  !> Longitude index (i) of each land point in the full lon×lat array.
  integer :: lonindex(npoi) = 0

  !> Latitude index (j) of each land point in the full lon×lat array.
  integer :: latindex(npoi) = 0

  ! ---------------------------------------------------------------------------
  ! /work2/  --  grid coordinate vectors
  ! ---------------------------------------------------------------------------
  public :: lonscale, latscale

  !> Longitude of each grid column (degrees East).
  real(wp) :: lonscale(nlon) = 0.0_wp

  !> Latitude of each grid row (degrees North).
  real(wp) :: latscale(nlat) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /work3/  --  single full-grid scratch array
  ! ---------------------------------------------------------------------------
  public :: work

  !> Scratch workspace for one full lon×lat field (ndim2 = nlon*nlat elements).
  real(wp) :: work(ndim2) = 0.0_wp

  ! ---------------------------------------------------------------------------
  ! /work4/  --  large I/O scratch array (allocatable to avoid huge static BSS)
  ! ---------------------------------------------------------------------------
  public :: cdummy

  !> Large I/O scratch buffer (ndim3 elements).
  !! Must be allocated by the I/O init routine before first use:
  !!   allocate(cdummy(ndim3))
  real(wp), allocatable :: cdummy(:)

end module mod_work
