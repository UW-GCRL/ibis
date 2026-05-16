# canopy_exchange (turcof + turvap) — verification scaffold

**Status**: PENDING — wrapper not yet implemented.

## Scope

F77 `canopy.f` lines 75–1599 (canini, fstrat, turcof, turvap, impexp, impexp2)
is the coupled implicit energy-balance solver:

```
canini -> drystress -> for iter in 1..3: { turcof -> stomata -> turvap }
```

JAX has these as separate building blocks in `physics/canopy_exchange.py`
and orchestrates them inside `timestepping/loops.py`.

## Why this is heavy

1. **Many fields**: turvap reads ~50 common-block fields + writes ~30. Wrapper
   would need ~100 args total.
2. **Coupled solver**: turvap assembles a 9×9 linearized matrix (tu, ts, tl,
   t12, t34, q12, q34, tg, ti) and calls `linsolve`. The JAX port solves the
   same system but inlined; small precision differences propagate amplified.
3. **Iterative**: 3 outer iterations for non-linearity; intermediate state
   (richl, richu, straml, strahl, stramu, strahu, wu, ws, wl, wg, wi) carries
   between iterations.
4. **Dependencies**: turcof depends on canini outputs; turvap depends on
   turcof outputs AND stomata outputs. Cannot test in isolation.

## Recommended path

- Build a single F77 wrapper `wrap_canopy` that mirrors `canopy.f`'s outer
  subroutine (canini + drystress + 3-iter loop). Black-box test.
- JAX side: extract the canopy_exchange portion of `loops.py` into a
  standalone helper that takes the same state and returns the same outputs.
- LOOSE tolerance — the 9×9 solve amplifies FP noise; expect ~1e-6 or worse
  on temperatures.

## Estimated effort: 3–5 hours
