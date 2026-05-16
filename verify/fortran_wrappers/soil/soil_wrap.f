c =====================================================================
c soil_wrap.f
c
c F77 wrapper exposing setsoi (from soil.f) as a callable subroutine.
c setsoi takes no arguments -- it reads/writes everything via common
c blocks in comsoi.h / comatm.h / comsno.h / comveg.h / compar.h.
c
c Inputs we set via common blocks (per call):
c   sand, clay, poros, wsoi, wisoi, tsoi  -- comsoi.h (per layer)
c   wpud, wipud                            -- comsoi.h (per point)
c   ta                                     -- comatm.h
c   tsno (first layer)                     -- comsno.h
c   tmelt, hvap, hsub, ch2o, cvap, cice    -- compar.h (compar1 common)
c   epsilon                                -- compar.h
c   zwpmax, wpudmax                        -- comveg.h
c
c Outputs (read from common blocks):
c   consoi(npoi, nsoilay)
c   qglif(npoi, 4)
c   hvasug(npoi), hvasui(npoi)
c =====================================================================


c ---------------------------------------------------------------------
c wrap_setsoi
c ---------------------------------------------------------------------
      subroutine wrap_setsoi(
     >    n_in,
     >    sand_in, clay_in, poros_in, wsoi_in, wisoi_in, tsoi_in,
     >    wpud_in, wipud_in,
     >    ta_in, tsno1_in,
     >    zwpmax_in, wpudmax_in,
     >    tmelt_in, hvap_in, hsub_in, ch2o_in, cvap_in, cice_in,
     >    epsilon_in,
     >    consoi_out, qglif_out, hvasug_out, hvasui_out)
cf2py intent(hide), depend(wpud_in) :: n_in=len(wpud_in)
cf2py intent(in)  sand_in
cf2py intent(in)  clay_in
cf2py intent(in)  poros_in
cf2py intent(in)  wsoi_in
cf2py intent(in)  wisoi_in
cf2py intent(in)  tsoi_in
cf2py intent(in)  wpud_in
cf2py intent(in)  wipud_in
cf2py intent(in)  ta_in
cf2py intent(in)  tsno1_in
cf2py intent(in)  zwpmax_in
cf2py intent(in)  wpudmax_in
cf2py intent(in)  tmelt_in
cf2py intent(in)  hvap_in
cf2py intent(in)  hsub_in
cf2py intent(in)  ch2o_in
cf2py intent(in)  cvap_in
cf2py intent(in)  cice_in
cf2py intent(in)  epsilon_in
cf2py intent(out) consoi_out
cf2py intent(out) qglif_out
cf2py intent(out) hvasug_out
cf2py intent(out) hvasui_out

      include 'implicit.h'
      include 'compar.h'
      include 'comatm.h'
      include 'comsno.h'
      include 'comsoi.h'
      include 'comveg.h'

c     f2py's parser doesn't know `nsoilay` (a parameter in compar.h),
c     so the boundary-facing arg dimensions use the literal value 6.
c     Internal code uses the parameter `nsoilay`, which gfortran resolves
c     from compar.h.  6 == nsoilay; if compar.h ever changes nsoilay,
c     update here.
      integer n_in
      real*8 sand_in(n_in, 6), clay_in(n_in, 6)
      real*8 poros_in(n_in, 6), wsoi_in(n_in, 6)
      real*8 wisoi_in(n_in, 6), tsoi_in(n_in, 6)
      real*8 wpud_in(n_in), wipud_in(n_in)
      real*8 ta_in(n_in), tsno1_in(n_in)
      real*8 zwpmax_in, wpudmax_in
      real*8 tmelt_in, hvap_in, hsub_in
      real*8 ch2o_in, cvap_in, cice_in, epsilon_in
      real*8 consoi_out(n_in, 6), qglif_out(n_in, 4)
      real*8 hvasug_out(n_in), hvasui_out(n_in)

      integer i, k

c     Set scalar common-block values
      tmelt   = tmelt_in
      hvap    = hvap_in
      hsub    = hsub_in
      ch2o    = ch2o_in
      cvap    = cvap_in
      cice    = cice_in
      epsilon = epsilon_in
      zwpmax  = zwpmax_in
      wpudmax = wpudmax_in

c     Populate per-point, per-layer common-block arrays.
      do i = 1, n_in
        wpud(i)    = wpud_in(i)
        wipud(i)   = wipud_in(i)
        ta(i)      = ta_in(i)
        tsno(i, 1) = tsno1_in(i)
        do k = 1, nsoilay
          sand(i, k)  = sand_in(i, k)
          clay(i, k)  = clay_in(i, k)
          poros(i, k) = poros_in(i, k)
          wsoi(i, k)  = wsoi_in(i, k)
          wisoi(i, k) = wisoi_in(i, k)
          tsoi(i, k)  = tsoi_in(i, k)
        end do
      end do

c     setsoi loops over k=1..nsoilay and i=1..npoi internally. To stop it
c     processing garbage points beyond n_in, we have two options:
c       (a) zero out indices n_in+1..npoi before the call, OR
c       (b) trust we only read back the first n_in indices.
c     We do (b): the wrapper output arrays are sized n_in so callers see
c     only valid entries. Indices > n_in still compute but are ignored.
c     (Note: for setsoi this is safe because each point is independent.
c     For soilh2o, the tridiag solve on unused indices could hit
c     division-by-zero if common-block arrays are uninitialized; the
c     soilh2o wrapper below initializes ALL npoi entries to safe values.)

      call setsoi

c     Read back outputs (first n_in points only)
      do i = 1, n_in
        hvasug_out(i) = hvasug(i)
        hvasui_out(i) = hvasui(i)
        do k = 1, 4
          qglif_out(i, k) = qglif(i, k)
        end do
        do k = 1, nsoilay
          consoi_out(i, k) = consoi(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_soilh2o -- expose soilh2o(owsoi, fwtop, fwpud, fsqueez, wflo).
c
c soilh2o reads/writes wsoi/wpud from common blocks and uses several
c other comsoi.h arrays (poros, porosflo, hydraul, suction, bex, hsoi,
c wisoi, bperm) plus compar.h scalars (dtime, rhow, epsilon).
c
c npoi=1718 in compar.h; our test batch n_in <= npoi.  We initialize
c ALL npoi entries to physically safe baseline values before overwriting
c the first n_in with user data -- otherwise the tridiag solver runs on
c uninitialized garbage at indices > n_in and can divide by zero.
c ---------------------------------------------------------------------
      subroutine wrap_soilh2o(
     >    n_in,
     >    wsoi_in, wisoi_in, wpud_in,
     >    poros_in, porosflo_in,
     >    hydraul_in, suction_in, bex_in,
     >    hsoi_in,
     >    fwtop_in, fwpud_in,
     >    bperm_in, dtime_in, rhow_in, epsilon_in,
     >    wsoi_out, wpud_out, fsqueez_out, wflo_out)
cf2py intent(hide), depend(wpud_in) :: n_in=len(wpud_in)
cf2py intent(in)  wsoi_in
cf2py intent(in)  wisoi_in
cf2py intent(in)  wpud_in
cf2py intent(in)  poros_in
cf2py intent(in)  porosflo_in
cf2py intent(in)  hydraul_in
cf2py intent(in)  suction_in
cf2py intent(in)  bex_in
cf2py intent(in)  hsoi_in
cf2py intent(in)  fwtop_in
cf2py intent(in)  fwpud_in
cf2py intent(in)  bperm_in
cf2py intent(in)  dtime_in
cf2py intent(in)  rhow_in
cf2py intent(in)  epsilon_in
cf2py intent(out) wsoi_out
cf2py intent(out) wpud_out
cf2py intent(out) fsqueez_out
cf2py intent(out) wflo_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsoi.h'

      integer n_in
      real*8 wsoi_in(n_in, 6), wisoi_in(n_in, 6), wpud_in(n_in)
      real*8 poros_in(n_in, 6), porosflo_in(n_in, 6)
      real*8 hydraul_in(n_in, 6), suction_in(n_in, 6), bex_in(n_in, 6)
      real*8 hsoi_in(6)
      real*8 fwtop_in(n_in), fwpud_in(n_in)
      real*8 bperm_in, dtime_in, rhow_in, epsilon_in
      real*8 wsoi_out(n_in, 6), wpud_out(n_in)
      real*8 fsqueez_out(n_in)
      real*8 wflo_out(n_in, 7)

      integer i, k
      real owsoi(npoi, nsoilay)
      real fwtop(npoi), fwpud(npoi)
      real fsqueez(npoi)
      real wflo(npoi, nsoilay+1)

c     Set scalars (compar.h common)
      bperm   = bperm_in
      dtime   = dtime_in
      rhow    = rhow_in
      epsilon = epsilon_in

c     Per-layer thicknesses (hsoi has nsoilay+1=7 elements; only 1..6 used)
      do k = 1, 6
        hsoi(k) = hsoi_in(k)
      end do
      hsoi(7) = 0.0

c     Initialize ALL npoi entries to safe baseline; then overwrite first n_in.
c     This prevents soilh2o's tridiag solve from dividing by zero on
c     unused indices.
      do i = 1, npoi
        wpud(i) = 0.0
        fwtop(i) = 0.0
        fwpud(i) = 0.0
        do k = 1, nsoilay
          wsoi(i, k)     = 0.3
          wisoi(i, k)    = 0.0
          poros(i, k)    = 0.45
          porosflo(i, k) = 0.45
          hydraul(i, k)  = 1.0e-5
          suction(i, k)  = 0.15
          bex(i, k)      = 5.0
          owsoi(i, k)    = 0.3
        end do
      end do

c     Overwrite the first n_in entries with the user-supplied values.
c     IMPORTANT: ibex (common comsoi7) is a pre-cached nint(bex) used by
c     soilh2o's bottom-boundary flux. If we set bex but not ibex, ibex
c     stays at whatever was in memory and the bottom-layer exponents
c     m = 2*ibex+3 are garbage, corrupting the tridiag through the lower
c     BC. Set ibex = nint(bex) explicitly. Also keep the npoi-wide
c     baseline ibex consistent.
      do i = 1, npoi
        do k = 1, nsoilay
          ibex(i, k) = nint(bex(i, k))
        end do
      end do
      do i = 1, n_in
        wpud(i)  = wpud_in(i)
        fwtop(i) = fwtop_in(i)
        fwpud(i) = fwpud_in(i)
        do k = 1, 6
          wsoi(i, k)     = wsoi_in(i, k)
          wisoi(i, k)    = wisoi_in(i, k)
          poros(i, k)    = poros_in(i, k)
          porosflo(i, k) = porosflo_in(i, k)
          hydraul(i, k)  = hydraul_in(i, k)
          suction(i, k)  = suction_in(i, k)
          bex(i, k)      = bex_in(i, k)
          owsoi(i, k)    = wsoi_in(i, k)
          ibex(i, k)     = nint(bex_in(i, k))
        end do
      end do

      call soilh2o(owsoi, fwtop, fwpud, fsqueez, wflo)

c     Read back state outputs from common blocks (first n_in points only).
      do i = 1, n_in
        wpud_out(i)    = wpud(i)
        fsqueez_out(i) = fsqueez(i)
        do k = 1, 6
          wsoi_out(i, k) = wsoi(i, k)
        end do
        do k = 1, 7
          wflo_out(i, k) = wflo(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_soilheat -- expose soilheat(otsoi, owsoi, c0pud, fhtop, wflo, c1pud).
c
c soilheat reads tsoi/wsoi/wisoi/poros/csoi/consoi/rhosoi from comsoi.h
c plus ch2o/cice/rhow/dtime from compar.h common. Returns updated tsoi
c (common) and hflo (common, written via local copy).
c ---------------------------------------------------------------------
      subroutine wrap_soilheat(
     >    n_in,
     >    tsoi_in, otsoi_in, owsoi_in, wsoi_in, wisoi_in,
     >    poros_in, consoi_in, csoi_in, rhosoi_in,
     >    hsoi_in,
     >    c0pud_in, c1pud_in,
     >    fhtop_in, wflo_in,
     >    ch2o_in, cice_in, rhow_in, dtime_in,
     >    tsoi_out, hflo_out)
cf2py intent(hide), depend(fhtop_in) :: n_in=len(fhtop_in)
cf2py intent(in)  tsoi_in
cf2py intent(in)  otsoi_in
cf2py intent(in)  owsoi_in
cf2py intent(in)  wsoi_in
cf2py intent(in)  wisoi_in
cf2py intent(in)  poros_in
cf2py intent(in)  consoi_in
cf2py intent(in)  csoi_in
cf2py intent(in)  rhosoi_in
cf2py intent(in)  hsoi_in
cf2py intent(in)  c0pud_in
cf2py intent(in)  c1pud_in
cf2py intent(in)  fhtop_in
cf2py intent(in)  wflo_in
cf2py intent(in)  ch2o_in
cf2py intent(in)  cice_in
cf2py intent(in)  rhow_in
cf2py intent(in)  dtime_in
cf2py intent(out) tsoi_out
cf2py intent(out) hflo_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsoi.h'

      integer n_in
      real*8 tsoi_in(n_in, 6), otsoi_in(n_in, 6)
      real*8 owsoi_in(n_in, 6), wsoi_in(n_in, 6), wisoi_in(n_in, 6)
      real*8 poros_in(n_in, 6), consoi_in(n_in, 6)
      real*8 csoi_in(n_in, 6), rhosoi_in(n_in, 6)
      real*8 hsoi_in(6)
      real*8 c0pud_in(n_in, 6), c1pud_in(n_in, 6)
      real*8 fhtop_in(n_in)
      real*8 wflo_in(n_in, 7)
      real*8 ch2o_in, cice_in, rhow_in, dtime_in
      real*8 tsoi_out(n_in, 6), hflo_out(n_in, 7)

      integer i, k
      real otsoi(npoi, nsoilay), owsoi(npoi, nsoilay)
      real c0pud(npoi, nsoilay), c1pud(npoi, nsoilay)
      real fhtop(npoi)
      real wflo(npoi, nsoilay+1)

c     Set scalars
      ch2o  = ch2o_in
      cice  = cice_in
      rhow  = rhow_in
      dtime = dtime_in

c     Per-layer thicknesses
      do k = 1, 6
        hsoi(k) = hsoi_in(k)
      end do
      hsoi(7) = 0.0

c     Initialize ALL npoi entries to safe baseline.
      do i = 1, npoi
        fhtop(i) = 0.0
        do k = 1, nsoilay
          tsoi(i, k)   = 285.0
          wsoi(i, k)   = 0.3
          wisoi(i, k)  = 0.0
          poros(i, k)  = 0.45
          consoi(i, k) = 0.5
          csoi(i, k)   = 870.0
          rhosoi(i, k) = 2650.0
          otsoi(i, k)  = 285.0
          owsoi(i, k)  = 0.3
          c0pud(i, k)  = 0.0
          c1pud(i, k)  = 0.0
        end do
        do k = 1, nsoilay+1
          wflo(i, k) = 0.0
        end do
      end do

c     Overwrite the first n_in with user data
      do i = 1, n_in
        fhtop(i) = fhtop_in(i)
        do k = 1, 6
          tsoi(i, k)   = tsoi_in(i, k)
          wsoi(i, k)   = wsoi_in(i, k)
          wisoi(i, k)  = wisoi_in(i, k)
          poros(i, k)  = poros_in(i, k)
          consoi(i, k) = consoi_in(i, k)
          csoi(i, k)   = csoi_in(i, k)
          rhosoi(i, k) = rhosoi_in(i, k)
          otsoi(i, k)  = otsoi_in(i, k)
          owsoi(i, k)  = owsoi_in(i, k)
          c0pud(i, k)  = c0pud_in(i, k)
          c1pud(i, k)  = c1pud_in(i, k)
        end do
        do k = 1, 7
          wflo(i, k) = wflo_in(i, k)
        end do
      end do

      call soilheat(otsoi, owsoi, c0pud, fhtop, wflo, c1pud)

c     Read back outputs (tsoi from common; hflo from common)
      do i = 1, n_in
        do k = 1, 6
          tsoi_out(i, k) = tsoi(i, k)
        end do
        do k = 1, 7
          hflo_out(i, k) = hflo(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_wadjust -- expose wadjust().
c
c wadjust clamps wsoi/wisoi to safe ranges and accumulates the
c adjustment delta into gadjust (comhyd.h). Returns the adjusted
c wsoi/wisoi plus the gadjust delta produced by this call.
c
c We zero gadjust on entry and read it back on exit so the returned
c gadjust_out is per-call (not accumulated forever).
c ---------------------------------------------------------------------
      subroutine wrap_wadjust(
     >    n_in,
     >    wsoi_in, wisoi_in,
     >    poros_in, hsoi_in,
     >    rhow_in, dtime_in, epsilon_in,
     >    wsoi_out, wisoi_out, gadjust_out)
cf2py intent(hide), depend(poros_in) :: n_in=size(poros_in,0)
cf2py intent(in)  wsoi_in
cf2py intent(in)  wisoi_in
cf2py intent(in)  poros_in
cf2py intent(in)  hsoi_in
cf2py intent(in)  rhow_in
cf2py intent(in)  dtime_in
cf2py intent(in)  epsilon_in
cf2py intent(out) wsoi_out
cf2py intent(out) wisoi_out
cf2py intent(out) gadjust_out

      include 'implicit.h'
      include 'compar.h'
      include 'comhyd.h'
      include 'comsoi.h'

      integer n_in
      real*8 wsoi_in(n_in, 6), wisoi_in(n_in, 6)
      real*8 poros_in(n_in, 6), hsoi_in(6)
      real*8 rhow_in, dtime_in, epsilon_in
      real*8 wsoi_out(n_in, 6), wisoi_out(n_in, 6)
      real*8 gadjust_out(n_in)

      integer i, k

      rhow    = rhow_in
      dtime   = dtime_in
      epsilon = epsilon_in

      do k = 1, 6
        hsoi(k) = hsoi_in(k)
      end do
      hsoi(7) = 0.0

c     Initialize all npoi entries with safe values; zero gadjust everywhere.
      do i = 1, npoi
        gadjust(i) = 0.0
        do k = 1, nsoilay
          wsoi(i, k)  = 0.3
          wisoi(i, k) = 0.0
          poros(i, k) = 0.45
        end do
      end do

      do i = 1, n_in
        do k = 1, 6
          wsoi(i, k)  = wsoi_in(i, k)
          wisoi(i, k) = wisoi_in(i, k)
          poros(i, k) = poros_in(i, k)
        end do
      end do

      call wadjust

      do i = 1, n_in
        gadjust_out(i) = gadjust(i)
        do k = 1, 6
          wsoi_out(i, k)  = wsoi(i, k)
          wisoi_out(i, k) = wisoi(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_freeze_thaw -- expose the soil-layer freeze/thaw block carved
c out from soilctl. Modifies tsoi, wsoi, wisoi (common comsoi5);
c also writes porosflo at end (matches F77 behaviour).
c ---------------------------------------------------------------------
      subroutine wrap_freeze_thaw(
     >    n_in,
     >    tsoi_in, wsoi_in, wisoi_in,
     >    poros_in, csoi_in, rhosoi_in,
     >    hsoi_in, c1pud_in,
     >    ch2o_in, cice_in, rhow_in,
     >    hfus_in, tmelt_in, epsilon_in,
     >    tsoi_out, wsoi_out, wisoi_out, porosflo_out)
cf2py intent(hide), depend(tsoi_in) :: n_in=size(tsoi_in,0)
cf2py intent(in)  tsoi_in
cf2py intent(in)  wsoi_in
cf2py intent(in)  wisoi_in
cf2py intent(in)  poros_in
cf2py intent(in)  csoi_in
cf2py intent(in)  rhosoi_in
cf2py intent(in)  hsoi_in
cf2py intent(in)  c1pud_in
cf2py intent(in)  ch2o_in
cf2py intent(in)  cice_in
cf2py intent(in)  rhow_in
cf2py intent(in)  hfus_in
cf2py intent(in)  tmelt_in
cf2py intent(in)  epsilon_in
cf2py intent(out) tsoi_out
cf2py intent(out) wsoi_out
cf2py intent(out) wisoi_out
cf2py intent(out) porosflo_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsoi.h'

      integer n_in
      real*8 tsoi_in(n_in, 6), wsoi_in(n_in, 6), wisoi_in(n_in, 6)
      real*8 poros_in(n_in, 6), csoi_in(n_in, 6), rhosoi_in(n_in, 6)
      real*8 hsoi_in(6), c1pud_in(n_in, 6)
      real*8 ch2o_in, cice_in, rhow_in
      real*8 hfus_in, tmelt_in, epsilon_in
      real*8 tsoi_out(n_in, 6), wsoi_out(n_in, 6)
      real*8 wisoi_out(n_in, 6), porosflo_out(n_in, 6)

      integer i, k
      real c1pud_local(npoi, nsoilay)

      ch2o    = ch2o_in
      cice    = cice_in
      rhow    = rhow_in
      hfus    = hfus_in
      tmelt   = tmelt_in
      epsilon = epsilon_in

      do k = 1, 6
        hsoi(k) = hsoi_in(k)
      end do
      hsoi(7) = 0.0

c     Safe-baseline every npoi entry then overwrite first n_in.
c     The block reads/writes the common-block tsoi/wsoi/wisoi/poros/
c     csoi/rhosoi/porosflo arrays; we need ALL npoi entries to be
c     physically sensible to avoid divide-by-zero garbage.
      do i = 1, npoi
        do k = 1, nsoilay
          tsoi(i, k)     = 285.0
          wsoi(i, k)     = 0.3
          wisoi(i, k)    = 0.0
          poros(i, k)    = 0.45
          csoi(i, k)     = 870.0
          rhosoi(i, k)   = 2650.0
          c1pud_local(i, k) = 0.0
        end do
      end do
      do i = 1, n_in
        do k = 1, 6
          tsoi(i, k)        = tsoi_in(i, k)
          wsoi(i, k)        = wsoi_in(i, k)
          wisoi(i, k)       = wisoi_in(i, k)
          poros(i, k)       = poros_in(i, k)
          csoi(i, k)        = csoi_in(i, k)
          rhosoi(i, k)      = rhosoi_in(i, k)
          c1pud_local(i, k) = c1pud_in(i, k)
        end do
      end do

      call soil_freeze_thaw_block(c1pud_local)

      do i = 1, n_in
        do k = 1, 6
          tsoi_out(i, k)     = tsoi(i, k)
          wsoi_out(i, k)     = wsoi(i, k)
          wisoi_out(i, k)    = wisoi(i, k)
          porosflo_out(i, k) = porosflo(i, k)
        end do
      end do

      return
      end


c ---------------------------------------------------------------------
c wrap_puddle_freeze_thaw -- expose the puddle freeze/thaw block.
c Inputs are first-layer state (tsoi_0_in, wpud_in, wipud_in,
c poros_0_in, wsoi_0_in, wisoi_0_in, csoi_0_in, rhosoi_0_in, hsoi_0_in).
c ---------------------------------------------------------------------
      subroutine wrap_puddle_freeze_thaw(
     >    n_in,
     >    tsoi_0_in, wpud_in, wipud_in,
     >    poros_0_in, wsoi_0_in, wisoi_0_in,
     >    csoi_0_in, rhosoi_0_in, hsoi_0_in,
     >    ch2o_in, cice_in, rhow_in,
     >    hfus_in, tmelt_in,
     >    tsoi_0_out, wpud_out, wipud_out)
cf2py intent(hide), depend(tsoi_0_in) :: n_in=len(tsoi_0_in)
cf2py intent(in)  tsoi_0_in
cf2py intent(in)  wpud_in
cf2py intent(in)  wipud_in
cf2py intent(in)  poros_0_in
cf2py intent(in)  wsoi_0_in
cf2py intent(in)  wisoi_0_in
cf2py intent(in)  csoi_0_in
cf2py intent(in)  rhosoi_0_in
cf2py intent(in)  hsoi_0_in
cf2py intent(in)  ch2o_in
cf2py intent(in)  cice_in
cf2py intent(in)  rhow_in
cf2py intent(in)  hfus_in
cf2py intent(in)  tmelt_in
cf2py intent(out) tsoi_0_out
cf2py intent(out) wpud_out
cf2py intent(out) wipud_out

      include 'implicit.h'
      include 'compar.h'
      include 'comsoi.h'

      integer n_in
      real*8 tsoi_0_in(n_in), wpud_in(n_in), wipud_in(n_in)
      real*8 poros_0_in(n_in), wsoi_0_in(n_in), wisoi_0_in(n_in)
      real*8 csoi_0_in(n_in), rhosoi_0_in(n_in)
      real*8 hsoi_0_in
      real*8 ch2o_in, cice_in, rhow_in
      real*8 hfus_in, tmelt_in
      real*8 tsoi_0_out(n_in), wpud_out(n_in), wipud_out(n_in)

      integer i, k

      ch2o  = ch2o_in
      cice  = cice_in
      rhow  = rhow_in
      hfus  = hfus_in
      tmelt = tmelt_in

c     Only the first layer is touched, but the block accesses
c     poros(i,1), csoi(i,1), rhosoi(i,1), wsoi(i,1), wisoi(i,1).
      hsoi(1) = hsoi_0_in
      do k = 2, 7
        hsoi(k) = 0.0
      end do

      do i = 1, npoi
        wpud(i)      = 0.0
        wipud(i)     = 0.0
        tsoi(i, 1)   = 285.0
        wsoi(i, 1)   = 0.3
        wisoi(i, 1)  = 0.0
        poros(i, 1)  = 0.45
        csoi(i, 1)   = 870.0
        rhosoi(i, 1) = 2650.0
      end do
      do i = 1, n_in
        tsoi(i, 1)   = tsoi_0_in(i)
        wpud(i)      = wpud_in(i)
        wipud(i)     = wipud_in(i)
        poros(i, 1)  = poros_0_in(i)
        wsoi(i, 1)   = wsoi_0_in(i)
        wisoi(i, 1)  = wisoi_0_in(i)
        csoi(i, 1)   = csoi_0_in(i)
        rhosoi(i, 1) = rhosoi_0_in(i)
      end do

      call puddle_freeze_thaw_block

      do i = 1, n_in
        tsoi_0_out(i) = tsoi(i, 1)
        wpud_out(i)   = wpud(i)
        wipud_out(i)  = wipud(i)
      end do

      return
      end
