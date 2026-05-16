! =============================================================================
! mod_diagnostics.f90  --  Diagnostic output control state
!
! Replaces COMMON blocks /diag1/ and /diag2/ from comdiag.h.
!
! The diagnostics engine selects which of the nvars tracked output variables
! are written to each of the nfiles binary output files, and over what time
! range and at what frequency.  All arrays are set at model startup by the
! I/O initialisation routine (readdiag or equivalent).
!
! Replaces: comdiag.h (COMMON /diag1/, /diag2/).
! =============================================================================
module mod_diagnostics
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! Compile-time parameters
  ! ---------------------------------------------------------------------------
  public :: nvars, nfiles

  !> Number of diagnostic variables available for output.
  integer, parameter :: nvars  = 119

  !> Maximum number of diagnostic output files supported simultaneously.
  integer, parameter :: nfiles = 10

  ! ---------------------------------------------------------------------------
  ! /diag1/  --  variable selection flags
  ! ---------------------------------------------------------------------------
  public :: ldiag

  !> Selection flag: ldiag(v,f) = 1 means variable v is written to file f.
  !! Set to 0 (off) until read from ibis.infile.
  integer :: ldiag(nvars, nfiles) = 0

  ! ---------------------------------------------------------------------------
  ! /diag2/  --  per-file timing and point selection
  ! ---------------------------------------------------------------------------
  public :: diagstart, diagend, ndiagpt, nfreq

  !> First year of diagnostic output for each file.
  integer :: diagstart(nfiles) = 0

  !> Last year of diagnostic output for each file.
  integer :: diagend(nfiles) = 0

  !> Land-point index (within the npoi array) for each diagnostic file.
  !! A value of 0 means output is aggregated rather than point-specific.
  integer :: ndiagpt(nfiles) = 0

  !> Output frequency (number of time steps between writes) for each file.
  integer :: nfreq(nfiles) = 0

end module mod_diagnostics
