# IBIS 3.0 Technical Note — early edition

2 November 2026

This early edition provides a readable description of IBIS 3.0 while the next round of calibration and evaluation
is being completed. It covers the released model, the corrections reviewed since the release, and the remaining
scientific work. Historical results and exploratory comparisons retain their labels and qualifications.

The language edit makes explanations more direct, divides dense sentences and clarifies how successive calculations
fit together. It preserves the scientific content: equations, numerical values, parameter tables, conditions,
citations and known limitations. All 17 chapters received a prose review. All 61 referenced figures are included so
the note can be read directly on GitHub or from a downloaded copy.

Start with the [chapter index](README.md). Chapter 1 gives the model overview; Chapter 16 explains calibration and
the remaining work; Chapter 17 describes numerical implementation and the scope of verification.

The model code on GitHub main remains the 30 October release. This publication updates documentation only. Sections
that describe later corrections refer to the reviewed implementation, not to a newly validated model release. A later
edition will incorporate completed refits and final evaluation results when they are available.

## Notes for developers

The scientific text was frozen at the reviewed documentation snapshot `46055f8`, whose implementation reference is
`c6698fd`. This edition was prepared on a branch from GitHub main at `e54e342`; model code and parameter files are
excluded from the publication change. Equation and table numbering is retained for stable cross-references.
Protected-content checks compare display math, inline math, numerical tokens, code spans, tables, headings and link
targets before and after the language edit. Figures retain their reviewed content and provenance.

Scientific preparation started after the review is checkpointed separately. Its findings have not been inserted
into this language edition. The next edition should update results only after the corresponding work is complete,
and should keep methane performance statements at band level.
