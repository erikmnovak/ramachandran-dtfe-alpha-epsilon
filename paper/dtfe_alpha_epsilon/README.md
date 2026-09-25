# DTFE → guarded alpha → epsilon: the current exposition

This self-contained paper explains the adopted unsmoothed three-stage
procedure, with the fixed **1.25× stability requirement**. Its title is
*From samples to Ramachandran contours: Delaunay density, guarded alpha shapes,
and a stable density window*.

- [Compiled PDF](../../output/pdf/dtfe_alpha_epsilon.pdf).
- [Editable LaTeX](dtfe_alpha_epsilon.tex) and [bibliography](references.bib).
- [Figures](figures/) as PDF and 320-dpi PNG.
- [Full-precision numerical results](numerical_results.json) and [adopted policy snapshot](adopted_policy.json).
- [Figure provenance and checks](figure_generation.json).
- [Build verification](build_validation.json).

The paper develops the method in one narrative: the purpose of Ramachandran
regions, the academic KDE reference, periodic Delaunay density and whole-triangle
calibration, guarded alpha selection, density-window cleanup and its stopping
rule, then six-class assessment. It defines the geometry and topology terms
before using them. Two short propositions distinguish actual construction
properties from empirical validation.

The fixed 1.25 is the span over which the output must remain unchanged. The
cleanup ratio `k` is computed separately for each class and shared by its two
levels. The paper treats 1.25 as a declared local robustness convention adopted
during Top8000 development, not a universal optimum or a guarantee inherited
from Bobrowski, Mukherjee and Taylor. KDE is not an input to the per-dataset
alpha or cleanup selector.

## Figures and evidence

| Asset | Purpose |
| --- | --- |
| Flowchart drawn directly in LaTeX | Introduce the complete procedure. |
| `dtfe_schematic` | Explain assigned area and minimum-corner selection; explicitly synthetic. |
| `cispro_alpha` | Show the real CisPro bridge and the removed triangles. |
| `general_cleanup` | Separate before/after views at each target, with component and loop counts. |
| `general_stability` | Show actual changes and the accepted `[k,1.25k]` interval, without KDE. |
| `general_feature_fates` | Locate the loss of the small inner island and retention/later loss of an outer island. |
| `kde_comparison_980`, `kde_comparison_995` | Final all-six comparisons to the published KDE boundaries. |

The data figures use existing adopted results, not newly fitted contours.
The renderer verifies the saved mask hashes, recomputes the twelve final
Jaccard values, checks fine-grid nesting, and verifies the selected interval
conditions recorded in the study. It does not retune alpha or epsilon.
The result table includes all residual Betti differences. Exact triangle
construction counts and containing-square raster counts are distinguished.

Primary input groups, relative to the project root:

- `documents/DTFE_ALPHA_EPSILON_POLICY.json`: adopted policy and result pins.
- `novak_work/validation_results/uniform_unsmoothed_alpha_guarded_v1/`:
  alpha decisions, exact areas, triangle sets and construction counts.
- `novak_work/validation_results/uniform_unsmoothed_alpha_guarded_assessment_v1/`:
  alpha/infinity masks for the bridge illustration.
- `novak_work/validation_results/uniform_alpha_epsilon_stability_v1/`:
  `span_1.25` selections, final masks and assessment.
- `novak_work/validation_results/uniform_alpha_epsilon_events_v1/`:
  complete grid change events and the feature close-ups.
- `novak_work/validation_results/internal_alpha_unfloored_assessment_v1/`:
  only its `published_reference` masks are used, not its alternative methods.
- The saved CisPro geometry in
  `remaining_full_smoothing_assessment_v1/CisPro/grid/geometry/` supplies
  coordinates of removed triangles. That directory name is historical;
  the adopted region and all densities in this paper are unsmoothed.

`figure_generation.json` records every input file and SHA-256. Source selection
and assessment artifacts are read without alteration. This exposition does
not migrate the older notebook or standalone contour API.

## Build

From this directory, with Python 3, TeX Live, `latexmk`, and Poppler available:

```sh
python build.py
```

The normal build uses the supplied figures and tables; it does not fit a
density, download data, or run an experiment. In the full project, the PDF
is written to `output/pdf/dtfe_alpha_epsilon.pdf` and intermediate files to
`tmp/pdfs/dtfe_alpha_epsilon/`. When this source folder is extracted on its
own, `build.py` writes the PDF beside the source and uses a local `build/`.

To regenerate the scientific figures, run inside the full project with its
saved result artifacts and NumPy, SciPy, and Matplotlib installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python build.py --figures
```

The source bundle contains all assets needed to compile the paper, but does
not duplicate the large raw observations or event-mask archives. Those are
needed only for regenerating the figures from data. The reference dataset
is attributed to the Richardson Laboratory; its local original license and
upstream provenance are in `data_points/top8000_reference_provenance/`; the original
reference-data license is also included here as `REFERENCE_DATA_LICENSE`.

This paper is the exposition of `dtfe-alpha-epsilon-span125-v1`.
The older `paper/manuscript.tex` remains a separate record of its earlier
smoothed method and should not be used to identify the current protocol.
