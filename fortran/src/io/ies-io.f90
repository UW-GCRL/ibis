! =============================================================================
! ies-io.f90  --  IES NetCDF file interface library
!
! Original author: Christine C. Molling (cmolling@facstaff.wisc.edu)
! University of Wisconsin - Madison, Institute for Environmental Studies
! Last original update: 4/1/98
!
! These subroutines provide a primary interface to IES-format NetCDF files.
! They wrap the low-level NetCDF Fortran-3 (NF_*) API.
!
! Subroutines:
!   inifile  -- create a new file with lon/lat/optional-3rd-real/time dims
!   inifilec -- create a new file with lon/lat/3rd-character/time dims
!   inivar   -- add a new variable to an existing file
!   endini   -- close and sync file to disk
!   writevar -- write a 2-, 3-, or 4-dimensional hyperslab
!   readvar  -- read  a 2-, 3-, or 4-dimensional hyperslab
!
! Converted from ibis_2.6b4/ies-io.f (F77 fixed-form) to F2008 free-form.
! No IBIS COMMON blocks are used; all state is passed via arguments.
! =============================================================================

! -----------------------------------------------------------------------------
! inifile -- create a new file with longitude, latitude, an optional real 3rd
!   dimension, and time.  Time must have units of "days since a date".
! -----------------------------------------------------------------------------
subroutine inifile(idies, filen, title, source, history, nlon, alons, &
                   nlat, alats, name3rd, long3rd, units3rd, n3rd, vals3rd, &
                   pos3rd, tunits, calendar, ierror)

  use mod_precision, only: wp
  implicit none

  integer,          intent(out) :: idies
  character(len=*), intent(in)  :: filen, title, source, history, name3rd
  character(len=*), intent(in)  :: long3rd, units3rd, pos3rd, tunits, calendar
  integer,          intent(in)  :: nlon, nlat, n3rd
  real(wp),         intent(in)  :: alons(nlon), alats(nlat), vals3rd(n3rd)
  integer,          intent(out) :: ierror

  include 'netcdf.inc'

  integer :: ierr, iddlon, idlon, iddlat, idlat, idd3rd, id3rd
  integer :: iddtime, idtime, iddlend, iddate, idtw, ll
  integer :: idims(4)        ! dimension id array (all NF_DEF_VAR calls use this)
  integer, external :: lenchr
  character(len=180) :: junk
  character(len=1)   :: cnull

  cnull  = char(0)
  ierror = 0

  ! Open file
  ierr = NF_CREATE(filen, NF_CLOBBER, idies)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Define global attributes
  ll = lenchr(title)
  junk = title(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'title', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ll = lenchr(source)
  junk = source(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'source', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ll = lenchr(history)
  junk = history(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'history', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ll = lenchr(calendar)
  junk = calendar(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'calendar', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'conventions', 9, &
                          'NCAR-CSM'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Define dimensions
  ierr = NF_DEF_DIM(idies, 'longitude', nlon, iddlon)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_DEF_DIM(idies, 'latitude', nlat, iddlat)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  idd3rd = -999
  if (name3rd /= '') then
    ierr = NF_DEF_DIM(idies, name3rd, n3rd, idd3rd)
    if (ierr /= NF_NOERR) then
      print *, 'Error in inifile'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
  end if

  ierr = NF_DEF_DIM(idies, 'time', NF_UNLIMITED, iddtime)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_DEF_DIM(idies, 'lengthd', 10, iddlend)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Define dimension variables (always pass idims array to NF_DEF_VAR)
  idims(1) = iddlon
  ierr = NF_DEF_VAR(idies, 'longitude', NF_FLOAT, 1, idims, idlon)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  idims(1) = iddlat
  ierr = NF_DEF_VAR(idies, 'latitude', NF_FLOAT, 1, idims, idlat)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  if (idd3rd /= -999) then
    idims(1) = idd3rd
    ierr = NF_DEF_VAR(idies, name3rd, NF_FLOAT, 1, idims, id3rd)
    if (ierr /= NF_NOERR) then
      print *, 'Error in inifile'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
  end if

  idims(1) = iddtime
  ierr = NF_DEF_VAR(idies, 'time', NF_FLOAT, 1, idims, idtime)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Define time_weights and date variables
  idims(1) = iddtime
  ierr = NF_DEF_VAR(idies, 'time_weights', NF_FLOAT, 1, idims, idtw)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  idims(1) = iddlend
  idims(2) = iddtime
  ierr = NF_DEF_VAR(idies, 'date', NF_CHAR, 2, idims, iddate)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Attributes for dimension variables
  ierr = NF_PUT_ATT_TEXT(idies, idlon, 'long_name', 10, 'longitude'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idlon, 'units', 13, 'degrees_east'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idlat, 'long_name', 9, 'latitude'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idlat, 'units', 14, 'degrees_north'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  if (idd3rd /= -999) then
    ll = lenchr(long3rd)
    junk = long3rd(1:ll)
    ierr = NF_PUT_ATT_TEXT(idies, id3rd, 'long_name', ll+1, junk//cnull)
    if (ierr /= NF_NOERR) then
      print *, 'Error in inifile'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if

    ll = lenchr(units3rd)
    junk = units3rd(1:ll)
    ierr = NF_PUT_ATT_TEXT(idies, id3rd, 'units', ll+1, junk//cnull)
    if (ierr /= NF_NOERR) then
      print *, 'Error in inifile'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idtime, 'long_name', 5, 'time'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ll = lenchr(tunits)
  junk = tunits(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, idtime, 'units', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idtw, 'long_name', 29, &
                          'number of days per time step'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idtw, 'units', 5, 'days'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, iddate, 'long_name', 25, &
                          'label for each time step'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, iddate, 'units', 1, char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! End define mode, enter data mode
  ierr = NF_ENDDEF(idies)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Write dimension variables (time written later by writevar)
  ierr = NF_PUT_VAR_DOUBLE(idies, idlon, alons)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_VAR_DOUBLE(idies, idlat, alats)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifile'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  if (idd3rd /= -999) then
    ierr = NF_PUT_VAR_DOUBLE(idies, id3rd, vals3rd)
    if (ierr /= NF_NOERR) then
      print *, 'Error in inifile'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
  end if

end subroutine inifile

! -----------------------------------------------------------------------------
! inifilec -- create a new file with longitude, latitude, a character-valued
!   3rd dimension, and time.  Time must have units of "days since a date".
! -----------------------------------------------------------------------------
subroutine inifilec(idies, filen, title, source, history, nlon, alons, &
                    nlat, alats, name3rd, long3rd, units3rd, n3rd, len3rd, &
                    chars3rd, tunits, calendar, ierror)

  use mod_precision, only: wp
  implicit none

  integer,          intent(out) :: idies
  character(len=*), intent(in)  :: filen, title, source, history, name3rd
  character(len=*), intent(in)  :: long3rd, units3rd, tunits, calendar
  integer,          intent(in)  :: nlon, nlat, n3rd, len3rd
  real(wp),         intent(in)  :: alons(nlon), alats(nlat)
  character(len=1), intent(in)  :: chars3rd(len3rd, n3rd)
  integer,          intent(out) :: ierror

  include 'netcdf.inc'

  integer :: ierr, iddlon, idlon, iddlat, idlat, idd3rd, id3rd
  integer :: iddlen, iddtime, idtime, iddate, idtw, ll
  integer :: iddlend
  integer :: idims(4)        ! dimension id array (all NF_DEF_VAR calls use this)
  integer, external :: lenchr
  character(len=180) :: junk
  character(len=1)   :: cnull

  cnull  = char(0)
  ierror = 0

  ! Open file
  ierr = NF_CREATE(filen, NF_CLOBBER, idies)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Define global attributes
  ll = lenchr(title)
  junk = title(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'title', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ll = lenchr(source)
  junk = source(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'source', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ll = lenchr(history)
  junk = history(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'history', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ll = lenchr(calendar)
  junk = calendar(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'calendar', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, NF_GLOBAL, 'conventions', 9, &
                          'NCAR-CSM'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Define dimensions
  ierr = NF_DEF_DIM(idies, 'longitude', nlon, iddlon)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_DEF_DIM(idies, 'latitude', nlat, iddlat)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_DEF_DIM(idies, name3rd, n3rd, idd3rd)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_DEF_DIM(idies, 'length', len3rd, iddlen)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_DEF_DIM(idies, 'lengthd', 10, iddlend)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_DEF_DIM(idies, 'time', NF_UNLIMITED, iddtime)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Define dimension variables (always pass idims array to NF_DEF_VAR)
  idims(1) = iddlon
  ierr = NF_DEF_VAR(idies, 'longitude', NF_FLOAT, 1, idims, idlon)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  idims(1) = iddlat
  ierr = NF_DEF_VAR(idies, 'latitude', NF_FLOAT, 1, idims, idlat)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  idims(1) = iddlen
  idims(2) = idd3rd
  ierr = NF_DEF_VAR(idies, name3rd, NF_CHAR, 2, idims, id3rd)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  idims(1) = iddtime
  ierr = NF_DEF_VAR(idies, 'time', NF_FLOAT, 1, idims, idtime)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Define time_weights and date variables
  idims(1) = iddtime
  ierr = NF_DEF_VAR(idies, 'time_weights', NF_FLOAT, 1, idims, idtw)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  idims(1) = iddlend
  idims(2) = iddtime
  ierr = NF_DEF_VAR(idies, 'date', NF_CHAR, 2, idims, iddate)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Attributes for dimension variables
  ierr = NF_PUT_ATT_TEXT(idies, idlon, 'long_name', 10, 'longitude'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idlon, 'units', 13, 'degrees_east'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idlat, 'long_name', 9, 'latitude'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idlat, 'units', 14, 'degrees_north'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  if (idd3rd /= -999) then
    ll = lenchr(long3rd)
    junk = long3rd(1:ll)
    ierr = NF_PUT_ATT_TEXT(idies, id3rd, 'long_name', ll+1, junk//cnull)
    if (ierr /= NF_NOERR) then
      print *, 'Error in inifilec'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if

    ll = lenchr(units3rd)
    junk = units3rd(1:ll)
    ierr = NF_PUT_ATT_TEXT(idies, id3rd, 'units', ll+1, junk//cnull)
    if (ierr /= NF_NOERR) then
      print *, 'Error in inifilec'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idtime, 'long_name', 5, 'time'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ll = lenchr(tunits)
  junk = tunits(1:ll)
  ierr = NF_PUT_ATT_TEXT(idies, idtime, 'units', ll+1, junk//cnull)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idtw, 'long_name', 29, &
                          'number of days per time step'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idtw, 'units', 5, 'days'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, iddate, 'long_name', 25, &
                          'label for each time step'//char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, iddate, 'units', 1, char(0))
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! End define mode, enter data mode
  ierr = NF_ENDDEF(idies)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Write dimension variables (time written later by writevar)
  ierr = NF_PUT_VAR_DOUBLE(idies, idlon, alons)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_VAR_DOUBLE(idies, idlat, alats)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_VAR_TEXT(idies, id3rd, chars3rd)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inifilec'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

end subroutine inifilec

! -----------------------------------------------------------------------------
! inivar -- add a new variable to an existing IES NetCDF file.
! -----------------------------------------------------------------------------
subroutine inivar(idies, varname, longname, units, ndims, dimnames, &
                  valmissing, ierror)

  use mod_precision, only: wp
  implicit none

  integer,          intent(in)  :: idies, ndims
  character(len=*), intent(in)  :: varname, longname, units
  character(len=*), intent(in)  :: dimnames(ndims)
  real(wp),         intent(in)  :: valmissing
  integer,          intent(out) :: ierror

  include 'netcdf.inc'

  integer :: ierr, idvar, iddims(4), i
  integer, external :: lenchr

  ierror = 0

  ! Enter redefine mode (warn but do not fail if already in define mode)
  ierr = NF_REDEF(idies)
  if (ierr /= NF_NOERR) then
    print *, 'Warning in inivar'
    print *, NF_STRERROR(ierr)
  end if

  ! Find dimension ids
  do i = 1, ndims
    ierr = NF_INQ_DIMID(idies, dimnames(i), iddims(i))
    if (ierr /= NF_NOERR) then
      print *, 'Error in inivar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
  end do

  ! Define the variable (iddims is always a rank-1 array)
  ierr = NF_DEF_VAR(idies, varname, NF_FLOAT, ndims, iddims, idvar)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inivar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Define attributes
  ierr = NF_PUT_ATT_TEXT(idies, idvar, 'long_name', &
                          lenchr(longname), longname)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inivar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_PUT_ATT_TEXT(idies, idvar, 'units', lenchr(units), units)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inivar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  if (valmissing /= 0.0_wp) then
    ierr = NF_PUT_ATT_REAL(idies, idvar, 'missing_value', NF_FLOAT, &
                            1, real(valmissing))
    if (ierr /= NF_NOERR) then
      print *, 'Error in inivar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
  end if

  ! Exit define mode
  ierr = NF_ENDDEF(idies)
  if (ierr /= NF_NOERR) then
    print *, 'Error in inivar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

end subroutine inivar

! -----------------------------------------------------------------------------
! endini -- close file and sync to disk.
! -----------------------------------------------------------------------------
subroutine endini(idies, ierror)

  implicit none

  integer, intent(in)  :: idies
  integer, intent(out) :: ierror

  include 'netcdf.inc'

  integer :: ierr

  ierr = NF_CLOSE(idies)
  if (ierr /= NF_NOERR) then
    print *, 'Error in endini'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierror = 0

end subroutine endini

! -----------------------------------------------------------------------------
! writevar -- write a 2-, 3-, or 4-dimensional hyperslab of data.
! -----------------------------------------------------------------------------
subroutine writevar(filen, varname, istart, icount, values, &
                    times, tweights, dates, ierror)

  use mod_precision, only: wp
  implicit none

  character(len=*),  intent(in)  :: filen, varname
  character(len=10), intent(in)  :: dates(*)
  integer,           intent(in)  :: istart(*), icount(*)
  real(wp),          intent(in)  :: values(*), times(*), tweights(*)
  integer,           intent(out) :: ierror

  include 'netcdf.inc'

  integer :: idies, idvar, ndims, ierr, itype, idtime, idtw, iddate
  integer :: ist(2), ict(2)

  ierror = 0

  ! Open file (warn but do not fail if already open)
  ierr = NF_OPEN(filen, NF_WRITE, idies)
  if (ierr /= NF_NOERR) then
    print *, 'Warning in writevar'
    print *, NF_STRERROR(ierr)
  end if

  ! Get id of variable
  ierr = NF_INQ_VARID(idies, varname, idvar)
  if (ierr /= NF_NOERR) then
    print *, 'Error in writevar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Inquire number of dimensions
  ierr = NF_INQ_VARNDIMS(idies, idvar, ndims)
  if (ierr /= NF_NOERR) then
    print *, 'Error in writevar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Write variable
  ierr = NF_PUT_VARA_DOUBLE(idies, idvar, istart, icount, values)
  if (ierr /= NF_NOERR) then
    print *, 'Error in writevar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Write time, time_weights, and dates for variables with >2 dimensions
  if (ndims > 2) then

    ierr = NF_INQ_VARID(idies, 'time', idtime)
    if (ierr /= NF_NOERR) then
      print *, 'Error in writevar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if

    ierr = NF_PUT_VARA_DOUBLE(idies, idtime, istart(ndims), icount(ndims), times)
    if (ierr /= NF_NOERR) then
      print *, 'Error in writevar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if

    ierr = NF_INQ_VARID(idies, 'time_weights', idtw)
    if (ierr /= NF_NOERR) then
      print *, 'Error in writevar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if

    ierr = NF_PUT_VARA_DOUBLE(idies, idtw, istart(ndims), icount(ndims), tweights)
    if (ierr /= NF_NOERR) then
      print *, 'Error in writevar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if

    ierr = NF_INQ_VARID(idies, 'date', iddate)
    if (ierr /= NF_NOERR) then
      print *, 'Error in writevar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if

    ist(1) = 1
    ist(2) = istart(ndims)
    ict(1) = 10
    ict(2) = icount(ndims)
    ierr = NF_PUT_VARA_TEXT(idies, iddate, ist, ict, dates)
    if (ierr /= NF_NOERR) then
      print *, 'Error in writevar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if

  end if

  ! Close file
  ierr = NF_CLOSE(idies)
  if (ierr /= NF_NOERR) then
    print *, 'Error in writevar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

end subroutine writevar

! -----------------------------------------------------------------------------
! readvar -- read a 2-, 3-, or 4-dimensional hyperslab of data.
! -----------------------------------------------------------------------------
subroutine readvar(filen, varname, name3d, istart, icount, values, &
                   alons, alats, vals3d, times, ierror)

  use mod_precision, only: wp
  implicit none

  character(len=*), intent(in)  :: filen, varname, name3d
  integer,          intent(in)  :: istart(*), icount(*)
  real(wp),         intent(out) :: values(*), alons(*), alats(*), vals3d(*)
  real(wp),         intent(out) :: times(*)
  integer,          intent(out) :: ierror

  include 'netcdf.inc'

  integer :: idies, idvar, ndims, idlon, idlat, id3d, idtime, ierr, itype

  ierror = 0

  ! Open file (warn but do not fail if already open)
  ierr = NF_OPEN(filen, NF_NOWRITE, idies)
  if (ierr /= NF_NOERR) then
    print *, 'Warning in readvar'
    print *, NF_STRERROR(ierr)
  end if

  ! Get id of variable
  ierr = NF_INQ_VARID(idies, varname, idvar)
  if (ierr /= NF_NOERR) then
    print *, 'Error in readvar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Inquire number of dimensions
  ierr = NF_INQ_VARNDIMS(idies, idvar, ndims)
  if (ierr /= NF_NOERR) then
    print *, 'Error in readvar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Read variable
  ierr = NF_GET_VARA_DOUBLE(idies, idvar, istart, icount, values)
  if (ierr /= NF_NOERR) then
    print *, 'Error in readvar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ! Read dimension variables
  ierr = NF_INQ_VARID(idies, 'longitude', idlon)
  ierr = NF_GET_VARA_DOUBLE(idies, idlon, istart(1), icount(1), alons)
  if (ierr /= NF_NOERR) then
    print *, 'Error in readvar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  ierr = NF_INQ_VARID(idies, 'latitude', idlat)
  ierr = NF_GET_VARA_DOUBLE(idies, idlat, istart(2), icount(2), alats)
  if (ierr /= NF_NOERR) then
    print *, 'Error in readvar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

  if (ndims >= 3) then
    ierr = NF_INQ_VARID(idies, 'time', idtime)
    if (ierr /= NF_NOERR) then
      print *, 'Error in readvar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
    ierr = NF_GET_VARA_DOUBLE(idies, idtime, istart(ndims), icount(ndims), times)
    if (ierr /= NF_NOERR) then
      print *, 'Error in readvar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
  end if

  if (ndims == 4) then
    ierr = NF_INQ_VARID(idies, name3d, id3d)
    if (ierr /= NF_NOERR) then
      print *, 'Error in readvar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
    ierr = NF_INQ_VARTYPE(idies, id3d, itype)
    if (ierr /= NF_NOERR) then
      print *, 'Error in readvar'
      print *, NF_STRERROR(ierr)
      ierror = -1
      return
    end if
    if (itype /= NF_CHAR) then
      ierr = NF_GET_VARA_DOUBLE(idies, id3d, istart(3), icount(3), vals3d)
      if (ierr /= NF_NOERR) then
        print *, 'Error in readvar'
        print *, NF_STRERROR(ierr)
        ierror = -1
        return
      end if
    end if
  end if

  ! Close file
  ierr = NF_CLOSE(idies)
  if (ierr /= NF_NOERR) then
    print *, 'Error in readvar'
    print *, NF_STRERROR(ierr)
    ierror = -1
    return
  end if

end subroutine readvar
