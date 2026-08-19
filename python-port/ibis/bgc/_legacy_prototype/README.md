# Quarantined prototype code — do not import

Scaffolding that shipped in the initial monorepo commit, was never called, never
verified, and never appeared in `verify/STATUS.md`. Kept only as a record of
intent; the parameterisations are wrong. See `docs/FIRE.md` §2.2 defect 7 and
`docs/METHANE.md` §2.2.

- `fire_emissions_UNUSED.py` — superseded by `ibis/bgc/fire_emissions.py`
  (stage F5). The original used Andreae & Merlet (2001) four-class factors and
  carried three conflicting unit conventions for the same number: the state
  field was documented gC m⁻² yr⁻¹, the code returned kgC m⁻² yr⁻¹, and
  `nee.py` consumed it as gC m⁻² d⁻¹.
