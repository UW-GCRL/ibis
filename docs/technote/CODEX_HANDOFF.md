# Technote continuation record

This early documentation edition follows the content review completed on 2 November 2026 and a separate language
edit of all 17 chapters. See [the edition note](EARLY_EDITION.md) for the publication scope and [the chapter index](README.md)
to read the note.

The language edit preserves the scientific content of the reviewed snapshot `46055f8`: equations, tables, numerical
values, code identifiers, conditions, citations and qualifications. It improves sentence structure and explanations.
The reviewed model implementation is `c6698fd`; model code and parameter changes are outside this documentation-only
publication.

## Recovery material for contributors

The full internal review and continuation records remain on the project server:

- `/shared/nas2/Hangkai/ibis/technote_review_2026-11-01/RESUME.md` records current branch state and work priority.
- `/shared/nas2/Hangkai/ibis/technote_review_2026-11-01/decisions_taken.md` records the scientific decisions.
- `/shared/nas2/Hangkai/ibis/technote_review_2026-11-01/codex_takeover_2026-11-02_UhmANV/` preserves the inherited
  technote, the original writer changes, source snapshots, figure archives and saved review evidence.

The last directory also contains `energy_summary-before.py`, the original energy-summary script, and
`complete_energy_partition.py`, the recovery script. The original summary summed the month axis of an array shaped
`(year, 12, 4)`; the corrected script sums the component axis. The missing T/ET section was recovered from the existing
624 outputs. Chapter 16 documents the method, validation, available comparisons and scientific limits.

The complete internal handoff is in the development checkout at
`/shared/nas2/Hangkai/ibis-port/docs/technote/CODEX_HANDOFF.md`. This public continuation page provides the recovery
locations needed by the note without reproducing the internal workflow journal or unpublished tower data.

## Next edition

The remaining calibration, final evaluation and numerical verification work is listed in Chapter 16. Update the
scientific results only when that work is complete. Retain the release and diagnostic labels until replacement
results are available, and report methane performance at band level. The saved scientific preparation and detailed
resume instructions remain on the project server.
