# Ramachandran contours: DTFE, alpha shapes, and a stable density window

This repository contains the expository paper and its supporting implementation
for the unsmoothed periodic **Delaunay/DTFE → guarded alpha → epsilon cleanup**
procedure, with the fixed **1.25× stability requirement**.

- **[Read the paper (PDF)](output/pdf/dtfe_alpha_epsilon.pdf)**
- [LaTeX source](paper/dtfe_alpha_epsilon/dtfe_alpha_epsilon.tex)
- [Figures: PNG and PDF](paper/dtfe_alpha_epsilon/figures/)
- [Bibliography](paper/dtfe_alpha_epsilon/references.bib)
- [Numerical results](paper/dtfe_alpha_epsilon/numerical_results.json)
- [Algorithm and code guide](METHOD_CODE.md)

## Build the paper

With Python 3, TeX Live, `latexmk`, and Poppler installed:

```sh
cd paper/dtfe_alpha_epsilon
python build.py
```

All displayed figures and tables are included. Compiling the paper requires
neither the original datasets nor a new numerical experiment. The PDF is written
to `output/pdf/dtfe_alpha_epsilon.pdf` from the repository root.

## Scope

The repository includes the paper, generated figures, small numerical and
provenance records, mathematical notes, and relevant implementation code and
tests. The multi-gigabyte observation datasets, geometry/mask caches, and
archived experiments are not included. No Git LFS is required for this scope.
The data-dependent experiment runners and figure-regeneration script need those
external inputs; their requirements are identified in the code guide.

The two-dimensional results describe six Top8000 classes. The fixed number
1.25 is a stability-span convention; the cleanup ratio is computed from the
data. The paper distinguishes construction guarantees, observed KDE agreement,
and remaining validation questions.
