# Current procedure: Delaunay/DTFE → guarded alpha → epsilon cleanup

The adopted working procedure has three stages: build unsmoothed periodic DTFE
whole-triangle regions, apply the common guarded alpha rule, and apply the
density-window cleanup using the **first stable interval spanning 1.25×**.
This is the current rule for continued project work, following the user's
selection of the 1.25× result across the six Top8000 classes.

**The fixed value 1.25 specifies how long a result must remain stable. It is
not the cleanup ratio k.** Each class gets its own data-derived k, shared by
its 98% and 99.5% regions. The thresholds used for cleanup are kL and L/k.
There is no smoothing, area floor, or affine triangle-interior clipping.

The [machine-readable policy](DTFE_ALPHA_EPSILON_POLICY.json) records the fixed
rule, stopping conditions, and full-precision results. The flow is:

```text
Periodic Delaunay mesh and raw vertex densities
    → construction-calibrated whole triangles at 98% and 99.5%
    → shared guarded alpha cutoff, holding both density gates fixed
    → first joint cleanup plateau containing [k, 1.25k]
    → final cleaned regions and measured coverage
```

## 1. Build the DTFE starting regions

Treat phi and psi as periodic angles. Samples on opposite plotting edges can
be neighbors. The periodic Delaunay construction includes the required seam
incidences; translated copies provide geometry, not extra observations. Merge
equal periodic coordinates and retain their observation multiplicity w_i.

Assign each vertex one third of the area of every incident triangle, including
all periodic corner incidences. With positive assigned area A_i, use the raw
density rho_i = w_i / A_i. No floor correction or smoothing is applied. A common
division of all densities and thresholds by the total observation count only
changes their units, not the selected regions.

For each triangle T, define its gate as its smallest corner density:

    g_T = min(rho_i for corners i of T).

Start at infinite alpha: include the whole closed triangle when g_T >= L.
For each target q, choose the largest inclusive gate L whose finished region
contains at least ceil(qN) weighted construction observations. Include all
observations tied at the gate. The targets are 98% and 99.5%, with the inner
region contained in the outer one. These percentages describe construction
counts; they are not integrated density mass or guaranteed new-data coverage.

This uses whole triangles. It does not trace an affine density contour through
triangle interiors. The [core construction specification](ALPHA_PRIMARY_METHOD.md)
gives the observation-inclusion and tie-handling details.

## 2. Apply the common guarded alpha rule

Hold both density gates fixed. Remove triangles whose circumradius exceeds
alpha, using the same alpha at both target levels. Evaluate the active radius
events and infinity. A candidate must:

1. Preserve every triangle of the uncut 98% region.
2. Retain the required 99.5% construction count at its fixed density gate.
3. Introduce no new connected component of the open complement.
4. When an outer component splits, leave an unchanged inner triangle in every
   surviving piece. Existing coreless outer components may remain or shrink,
   but may not split into new coreless pieces.

Among passing candidates, minimize the exact outer triangle area. For identical
triangle sets, prefer the least restrictive evaluated event, with infinity
preferred. This is the same rule for every class; KDE and a requested number
of components do not enter the selection.

The current alpha radii are 1.39511469 degrees for General, 5.98135638 for Gly,
2.95268237 for TransPro, 14.55507444 for CisPro, 3.09104371 for PrePro, and
2.36932142 for IleVal. They are computed outputs for these inputs. All six
inner triangle regions survive the alpha step unchanged, and alpha removes
the CisPro outer bridge. The [sealed alpha construction](../novak_work/validation_results/uniform_unsmoothed_alpha_guarded_v1/terminal.json)
and [mathematical specification](ALPHA_PRIMARY_METHOD.md) remain unchanged.

## 3. Select and apply the epsilon cleanup

### The density window at a candidate ratio k

Assign a point its largest minimum-corner gate among the alpha-eligible
triangles containing it:

    F_alpha(x) = max(g_T for T containing x with R_T <= alpha).

Where no alpha-eligible triangle covers a point, set the field to zero and
mark that location as alpha-excluded. The region F_alpha >= L is the selected
triangle union. The maximum handles shared closed edges and vertices. The
cleanup evaluates this field on a periodic raster. Set the high and low thresholds to

    high = k L,       low = L / k,       k >= 1.

This multiplicative window is symmetric in log density. It is not an additive
epsilon of kL. For a candidate k:

1. Label the starting foreground components. Remove a whole component only
   when its peak is strictly below kL; a peak equal to kL is retained.
2. Relabel the background after the removals. This matters when removal joins
   background pieces that were previously separated.
3. Fill a background component only when its minimum score is at least L/k,
   unless it winds around a periodic direction, is tied for largest background
   area, or touches any alpha-excluded cell.

Foreground uses eight-neighbor connectivity and background uses four neighbors,
with both angular seams identified. Protecting alpha-excluded background means
cleanup cannot fill an alpha-excluded gap back in. The entire background
component containing such a gap remains protected.

This operation removes weak islands and fills eligible shallow holes. It
retains the boundaries of surviving components, and cannot sever a bridge
inside one of them. At k = 1 it is the identity. The [cleanup implementation](../novak_work/rama_alpha_window_cleanup.py)
fixes the peak, pit, connectivity, and protection conventions.

### The adopted 1.25× stopping requirement

Use the complete geometry-change events over k in [1,64], at 768-square and
1536-square resolution. Between actual events the output mask is constant.
Join the event streams for both contour levels and both grids, so a joint
plateau means all four individual masks remain unchanged as k varies.

Choose the **first** plateau beginning at k that meets all four conditions:

1. Each of the four masks stays unchanged throughout the closed interval
   [k,1.25k]. This requires full region identity, not just equal component counts.
2. The Betti numbers agree between the two resolutions at each target level.
3. Both regions are nonempty at both resolutions.
4. At each resolution, the inner region is contained in the outer region.

Return that plateau's actual left onset as k, and stop. A later change under
stronger cleanup does not invalidate the earlier qualifying interval. Do not
increase the required span to 1.33, 1.5, or 2, and do not continue searching for
fewer components after the first qualifying plateau has been found.

At a real next-change endpoint b, require 1.25k < b: equality would include a
changed result in the required closed interval. A final plateau observed through
64 can qualify when 1.25k <= 64, but this asserts no stability beyond 64. The
implementation uses the actual Float64 transition onsets and their correct
strict/inclusive comparisons. Keep the saved onset in computation; rounding it
down can retain a component that was removed at the transition.

If no plateau passes within the declared range, report the selection as
unresolved. Do not silently widen the range, change the span, or substitute a
fixed ratio such as 16. Returning k = 1 is valid: a class can need no further
cleanup under the same rule. It is the current outcome for CisPro, whose bridge
has already been removed by alpha.

### Why this bound is the current choice

The fixed 1.25× requirement accepts a modest interval of local stability. The
user selected this tradeoff after reviewing all six classes: residual fragments
are accepted for continued work, while the earlier General outer islands are
retained rather than removed by a demand for longer stability. The bound is an
empirical protocol choice, not a theorem that identifies a universally optimal
contour. It stays fixed when applying this working version to another dataset.

Choosing this convention used Top8000 development evidence, including the
available KDE comparisons. Once the convention is fixed, finding k uses only
our density field and geometry; a new dataset's KDE is not needed. Comparison
with published KDE follows construction and does not alter its selected k.

### Current Top8000 cleanup ratios

The following values are outputs of the same selector. Each is shared by the
two target levels within its class. They are not six constants to transplant
to a new dataset.

| Class | Selected k | Final raster (components, loops), 98% | Final raster (components, loops), 99.5% |
|---|---:|---|---|
| General | 6.796947492 | (3, 0) | (5, 5) |
| Gly | 6.027967697 | (3, 0) | (1, 2) |
| TransPro | 3.394636943 | (3, 0) | (1, 0) |
| CisPro | 1.000000000 | (2, 0) | (3, 0) |
| PrePro | 2.820726580 | (4, 0) | (3, 0) |
| IleVal | 5.116347524 | (2, 0) | (3, 0) |

Full-precision ratios and interval bounds are in the policy JSON and in the
saved study's `selections["1.25"]` records. Its originally designated primary
branch used a doubling; that historical label does not override the adopted
1.25× rule.

## 4. What the final regions and measurements mean

The 98% and 99.5% labels identify construction targets **before cleanup**.
Both gates L remain fixed, and cleanup is not followed by recalibration.
Deleting islands can lower coverage and filling holes can raise it, so final
coverage is measured again.

The alpha construction uses exact closed triangles. The cleanup uses periodic
raster cells. An observation on an exact triangle boundary may be inside that
region while its containing cell's center is outside. Report exact starting
counts separately, and compare cleanup changes against the same-grid alpha
mask. Pixel Betti numbers likewise describe the raster, not certified topology
of the exact triangle union.

The table reports the adopted 1.25× branch at 1536-square resolution. The
starting target labels the row; coverage columns measure the containing-cell
raster convention before and after cleanup. KDE is a subsequent assessment.

| Class | Target | Raster coverage after alpha → after cleanup | KDE Jaccard after alpha → after cleanup |
|---|---:|---:|---:|
| General | 98% | 97.305% → 97.245% | 0.8525 → 0.8772 |
| General | 99.5% | 99.298% → 99.256% | 0.8708 → 0.8795 |
| Gly | 98% | 97.122% → 96.918% | 0.7872 → 0.7942 |
| Gly | 99.5% | 99.146% → 99.137% | 0.8229 → 0.8307 |
| TransPro | 98% | 97.117% → 97.087% | 0.7923 → 0.8026 |
| TransPro | 99.5% | 99.200% → 99.197% | 0.8267 → 0.8273 |
| CisPro | 98% | 96.338% → 96.338% | 0.7287 → 0.7287 |
| CisPro | 99.5% | 98.325% → 98.325% | 0.7124 → 0.7124 |
| PrePro | 98% | 97.018% → 97.059% | 0.8683 → 0.8753 |
| PrePro | 99.5% | 99.158% → 99.148% | 0.8467 → 0.8481 |
| IleVal | 98% | 97.137% → 97.207% | 0.8452 → 0.8709 |
| IleVal | 99.5% | 99.294% → 99.269% | 0.8521 → 0.8590 |

All selected regions are nested at both tested resolutions, and their Betti
numbers agree between grids. These checks establish the stated selection
conditions, not complete boundary convergence or new-population coverage.

General retains both tracked outer islands at this choice; its small 98%
island is still absent. Its remaining five outer raster loops are protected
by alpha-excluded background. CisPro retains three outer raster components
because k = 1 makes no cleanup change, although the exact alpha region has two
components. PrePro retains a fourth inner component and IleVal a third outer
component. These are the reviewed residual differences accepted in the current
working choice; they have not been removed from the measurements or figures.

### Figures for the adopted result

In each linked comparison, **the left column is the adopted 1.25× result**;
the top row is 98% and the bottom row is 99.5%. Blue is our region and black is
the published KDE boundary. The middle and right columns retain the 1.33× and
1.5× comparisons; those are not current defaults.

- [General](../novak_work/validation_results/uniform_alpha_epsilon_span133_v1/General_window_comparison_kde.png).
- [Gly](../novak_work/validation_results/uniform_alpha_epsilon_span133_v1/Gly_window_comparison_kde.png).
- [TransPro](../novak_work/validation_results/uniform_alpha_epsilon_span133_v1/TransPro_window_comparison_kde.png).
- [CisPro](../novak_work/validation_results/uniform_alpha_epsilon_span133_v1/CisPro_window_comparison_kde.png).
- [PrePro](../novak_work/validation_results/uniform_alpha_epsilon_span133_v1/PrePro_window_comparison_kde.png).
- [IleVal](../novak_work/validation_results/uniform_alpha_epsilon_span133_v1/IleVal_window_comparison_kde.png).

The [stability study](UNIFORM_ALPHA_EPSILON_STABILITY.md) explains the alternative
window tests and feature audit. Its [assessment file](../novak_work/validation_results/uniform_alpha_epsilon_stability_v1/assessment.json)
contains the adopted rows under `variant = "span_1.25"`. The older ratio-16
three-stage displays remain historical illustrations and do not show the
current final-stage choice for every class.

## 5. Why this is not the paper's guaranteed output

[Bobrowski, Mukherjee and Taylor, *Topological consistency via kernel estimation*](https://doi.org/10.3150/15-BEJ744)
study homology surviving between density levels. Their result does not specify
our geometric operation of deleting whole components and filling holes. Raw
DTFE scores, the alpha-restricted domain, periodic raster, and multiplicative
window do not inherit their KDE-based sampling and bandwidth guarantee.
The paper motivates the window idea; it does not prove that a 1.25× stability
requirement selects a correct contour or realizes its inclusion-map homology.
The [earlier theorem comparison](ALPHA_THEN_EPSILON_EXPERIMENT.md#6-why-the-papers-guarantee-is-not-inherited)
provides the detailed distinction.

## 6. Implementation and adoption record

- [Fixed current policy and derived values](DTFE_ALPHA_EPSILON_POLICY.json).
- [Adopted alpha selection](../novak_work/diagnose_uniform_guarded_alpha.py).
- [Cleanup operation](../novak_work/rama_alpha_window_cleanup.py).
- [Exact event enumeration and cache](../novak_work/rama_alpha_cleanup_events.py).
- [Complete six-class event sweep](../novak_work/sweep_uniform_alpha_epsilon_events.py).
- [First-plateau selector](../novak_work/rama_epsilon_plateau_selection.py): call
  `select_first(intervals, 1.25, admissible_callback)` with the checks above.
- [Completed selection study](../novak_work/validation_results/uniform_alpha_epsilon_stability_v1/selection.json):
  use `selections["1.25"]` for each class.

The rule is implemented and evaluated on the available Top8000 inputs. This
adoption selects that existing branch; it does not relabel or rerun historical
experiments. Their 16, 1.33×, 1.5× and 2× settings remain recorded as tested.
The [pre-adoption guide snapshot](../archive/epsilon_span125_adoption_2026-09-24/SNAPSHOT.md)
preserves the preceding notes. The [new LaTeX exposition](../paper/dtfe_alpha_epsilon/README.md) describes this
adopted protocol with fresh figures and the 1.25× results. The older notebook
and standalone entrypoint remain separate migration work, as described in the
[project guide](../README.md).
