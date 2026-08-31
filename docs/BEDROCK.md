# Thermal-only bedrock layers

## The problem

IBIS 2.6b4 solves soil heat on 6 layers totalling 4.0 m with a zero-flux base:

    hsoi = [0.10, 0.15, 0.25, 0.50, 1.00, 2.00]      # compar.h: nsoilay = 6

Our port reproduces this faithfully. It is nonetheless too shallow.

A zero-flux boundary **reflects** the annual temperature wave rather than
absorbing it. The half-space estimate `exp(-z/d)` therefore understates the
amplitude near the base by roughly a factor of two. The correct finite-domain
form (Carslaw & Jaeger) for a zero-flux base at depth `H` is

    A(z)/A(0) = sqrt( [cosh(2(H-z)/d) + cos(2(H-z)/d)]
                    / [cosh(2H/d)     + cos(2H/d)]     )

Damping depths measured from our own run (`d = sqrt(2*kappa/omega)`):

| band    | kappa (m2/s) | d (m) |
|---------|--------------|-------|
| 30S-30N | 3.40e-07     | 1.83  |
| 30-60N  | 5.24e-07     | 2.17  |
| 60-90N  | 9.16e-07     | 2.95  |

With `H = 4.0 m` the base sees 23-55% of the surface annual amplitude, not the
11-26% the half-space form suggests.

**The error is concentrated at the base, not spread through the column.**
Measured against a finely-resolved 32 m reference, at our node midpoints:

| node | mid (m) | global | Arctic |
|------|---------|--------|--------|
| 1-4  | 0.05-0.75 | 1.00x | 1.00x |
| 5    | 1.50    | 0.97x  | 1.06x  |
| 6    | 3.00    | 1.34x  | 1.52x  |

Our diagnosed thaw depth of 1.14 m sits where the bias is +2%. The active
layer -- and therefore methane -- is barely affected. What the shallow column
costs us is deep soil temperature and interannual thermal memory.

Burke et al. (2020, TC 14, 3155) put the threshold explicitly: *"Any models
with a maximum soil depth of less than 4.5 m will be unable to represent this
[active layer thickness] value."* We sit just under it.

## The fix

Add hydrologically-inactive layers below the soil and move the zero-flux
boundary down. No water, carbon or roots exist there; only heat conducts. This
is what essentially every land-surface model that faced this problem did:

| model | column | layers | hydro | thermal-only | bottom BC |
|-------|--------|--------|-------|--------------|-----------|
| JULES default | 3.0 m | 4 | 4 | 0 | zero flux |
| **IBIS (was)** | **4.0 m** | **6** | **6** | **0** | **zero flux** |
| **IBIS (now)** | **12.0 m** | **9** | **6** | **3** | **zero flux** |
| LPJmL4 | 13 m | 6 | 5 (3 m) | 1 x 10 m | zero flux |
| CLM5/CTSM | 49.6 m | 25 | 20 (8.6 m) | 5 | zero flux |
| JSBACH | 9.83 m | 5 | 5 | 0 | zero flux |
| LPJ-GUESS | 49.5 m | 20 | 15 (1.5 m) | 5 (48 m) | zero flux |
| CLASSIC offline | 61.4 m | 20 | to SDEP | below SDEP | zero flux |

Zero flux at the base is near-universal, even in 250 m and 1400 m columns.
We keep it. We deliberately do **not** prescribe a geothermal flux at 4 m:
Ross et al. (2024, unverified) report that injecting it at shallow depth adds
spurious heat and destroys more permafrost than a zero-flux base does. Both Chadburn (JULES)
and Gonzalez-Rouco (JSBACH) explicitly defer geothermal flux to future work.

## Why (2, 2, 4) m and not one thick layer

LPJmL4 uses a single 10 m buffer. Copying that literally would be a bad trade.
Driving the real `soilheat()` with an annual sinusoid and comparing amplitude
at the 6 soil nodes against a fine 32 m reference:

| config | layers | depth | tropics | global | Arctic | frozen (2x kappa) |
|--------|--------|-------|---------|--------|--------|--------|
| current | 6 | 4 m | 25.6% | 36.9% | 53.1% | 62.7% |
| LPJmL4 style (+1 x 10 m) | 7 | 14 m | 20.1% | 25.7% | 28.2% | 19.6% |
| Option A (+2: 2, 6) | 8 | 12 m | 3.0% | 2.5% | 3.9% | 3.7% |
| **adopted (+3: 2, 2, 4)** | **9** | **12 m** | **4.0%** | **2.8%** | **1.5%** | **0.3%** |
| (+4: 2, 2, 4, 8) | 10 | 20 m | 4.0% | 2.8% | 1.4% | 0.7% |

The LPJmL4 grid reaches **14 m yet is ~7x less accurate than ours at 12 m**,
because the 1 m -> 10 m jump distorts the flux at that interface and the error
propagates back up. **Thickness ratio governs the result, not total depth.**
Keeping every ratio at or below 2 (1 -> 2 -> 2 -> 4) is what buys the accuracy.
Extending past 12 m buys nothing measurable.

LPJmL4 gets away with its grid because its hydrology bottoms at 3 m and its
deepest reported node sits well above the buffer. Our deepest soil node is at
3.0 m, right against the seam, so we cannot absorb the same crudeness.

Bedrock properties follow CLM: `thk_bedrock = 3.0 W/m/K`,
`csol_bedrock = 2.0e6 J/m3/K`.

## Cost

+5.9% on a full timestep (measured, 4 points, CPU). The soil column appears in
one tridiagonal solve, which grows from 6 to 9 rows. There is no new hydrology
state and no extra spin-up: new layers are seeded by downward continuation
from the deepest soil node.

That continuation is **isothermal** (`BedrockParams.dtdz = 0.0`), because
isothermal is the equilibrium under a zero-flux base: `d2T/dz2 = 0` makes the
steady profile linear, and zero gradient at the base forces the slope to zero.
Seeding a geothermal gradient instead would create a transient that our own
boundary condition then takes decades to erase. The geothermal form of downward
continuation belongs to models that impose a basal heat flux (CryoGrid, GIPL,
UVic); set `dtdz` non-zero only alongside such a flux.

## Using it

Off by default -- `nbedrock = 0` reproduces the Fortran column exactly, and
`soilheat()` with its default `n_hydro` is bit-identical to before.

```ini
[grid]
nbedrock = 3
```

or in Python:

```python
grid = GridConfig(npoi=npoi, nsoilay=6, nbedrock=3)
```

`GridConfig.nbedrock` and `SimParams.bedrock` must agree on layer count;
a mismatch raises `ValueError` rather than broadcasting silently.

**Restart compatibility:** enabling bedrock adds `soil.tsoi_deep` to the state
pytree. Checkpoints written without it cannot be loaded into a bedrock-enabled
template (and vice versa) -- `load_state` will fail on the structure mismatch
rather than silently corrupting the column. Re-seed from a cold start, or
extend the checkpoint with `scripts/convert_restart.py`.

## Validated in the full model

Two identical 200-year GSWP3 spin-ups from the same `spinup_fire_v8` state,
differing only in the column (`verify/timestepping/pipeline/spinup_bedrock_v9.py
--nbedrock 0|3`). Both 0 NaN. Wall clock 201.9 vs 203.0 min -- **+0.5% at
scale**, well under the +5.9% micro-benchmark, because at 88 484 cells the
timestep is dominated by other physics.

**Seasonal amplitude of soil temperature** (`verify/bedrock/seasonal_amplitude.py`),
the quantity the change exists for:

| node | 30S-30N | 30-60N | 60-90N |
|------|---------|--------|--------|
| 0.05-0.75 m | +0.1 to +1.6% | +0.2 to +1.7% | +0.7 to +2.0% |
| 1.50 m | +4.0% | +3.8% | +0.1% |
| **3.00 m** | **-22.9%** | **-23.3%** | **-31.5%** |

This matches the idealised 1-D prediction quantitatively. The driver predicted
a +52% amplitude inflation at the Arctic 3.0 m node, so removing it should cut
the swing by 1 - 1/1.52 = **-34.6%**; the full model gives **-31.5%**. For
30-60N, predicted -25.4%, measured -23.3%. The remaining gap is the ~1.5%
residual error of the 12 m column plus the moisture and phase-change physics
the idealised driver does not carry.

**Everything else moved little, as predicted:**

| quantity | control | bedrock |
|---|---|---|
| NH perennially-frozen extent (Mkm2) | 22.87 | 23.90 |
| active layer depth (m) | 1.375 | 1.376 |
| global CH4 (Tg/yr) | 256.5 | 256.3 |
| CH4 latitudinal shape error | 15.8 | 15.5 |
| mean biomass (kgC/m2) | 3.292 | 3.292 |
| mean soil carbon (kgC/m2) | 14.347 | 14.330 |
| soil base T, NH mean (K) | 283.727 | 282.619 |

Methane moving by 0.08% was the falsification test: had it moved
substantially, the claim that the bias was confined to the 3.0 m node -- below
the active layer -- would have been wrong. It holds.

**One result went the wrong way.** NH perennially-frozen extent rose from 22.87
to 23.90 Mkm2 against Obu (2021)'s 21.8, so the deeper column sits slightly
*further* from observations, not closer. It is a 4.5% change on a diagnostic
whose one-year "frozen in all 12 months" definition is itself an upper bound,
so it is not strong evidence either way -- but it does mean this change should
not be advertised as improving permafrost extent. It improves the deep thermal
solution; extent was already close and stayed close.

**A note on measuring this.** Permafrost extent and active layer depth cannot
be read from a final state: in a December snapshot most of the boreal zone is
frozen, and "any frozen layer" reports ~41 Mkm2 against a true ~23. The first
version of `compare_ab.py` made exactly that error. Both quantities need the
seasonal cycle, which is why they live in `seasonal_amplitude.py`.

## Known limitation

Deepening the column does not by itself buy taliks. The DECP phase-change
scheme IBIS shares with CLM cannot represent a closed talik at any column
depth (Hotten et al. 2025, unverified). That is an independent second gate; do not read a
correct thermal boundary as a complete permafrost capability.

## References

Everything in the tables above was measured in this repo (see
`python-port/tests/test_bedrock.py` and the damping-depth calculation from our
own run output). The citations below are supporting context from a literature
search; those marked (unverified) were not checked against the primary source,
so confirm them before relying on them in a paper.

- Burke, E. J. et al. (2020), *The Cryosphere* 14, 3155. https://tc.copernicus.org/articles/14/3155/2020/
- Carslaw, H. S. & Jaeger, J. C. (1959), *Conduction of Heat in Solids*, 2nd ed.
- Chadburn, S. E. et al. (2015), *Geosci. Model Dev.* 8, 1493. https://gmd.copernicus.org/articles/8/1493/2015/
- Garcia-Pereira, F. et al. (2025) -- permafrost-extent bias vs column depth. **(unverified)**
- Lawrence, D. M. et al. (2008), *JGR* 113, F02011. https://doi.org/10.1029/2007JF000883
- Melton, J. R. et al. (2019), *Geosci. Model Dev.* 12, 4443. https://gmd.copernicus.org/articles/12/4443/2019/
- Schaphoff, S. et al. (2018), *Geosci. Model Dev.* 11, 1343 (LPJmL4). https://gmd.copernicus.org/articles/11/1343/2018/
