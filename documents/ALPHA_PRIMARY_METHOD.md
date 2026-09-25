# Core construction: unsmoothed whole-triangle DTFE and guarded alpha

This document specifies the first two stages of the adopted working procedure:
**periodic Delaunay/DTFE → common guarded alpha → epsilon cleanup with a fixed
1.25× stability requirement**. The [complete current procedure](UNIFORM_ALPHA_THEN_EPSILON.md)
and [machine-readable policy](DTFE_ALPHA_EPSILON_POLICY.json) specify the final
cleanup stage. There is no smoothing, area floor, or affine triangle-interior
clipping. The alpha construction below is unchanged by epsilon adoption.

The same rules apply to all six classes. Alpha and the cleanup ratio are
computed from our observations and geometry; a new dataset's KDE is used only
for subsequent assessment. The fixed 1.25× convention is a Top8000 development
choice. Adoption establishes the method for continued work, without claiming
that independent populations or higher dimensions have already been validated.

## 1. Build the periodic Delaunay geometry and raw density

Treat both angles as periodic with period 360°. Coordinates across a plotting
seam are neighbors on the same angular domain. Use the certified periodic
Delaunay mesh, including all corner incidences across the seams; translated
copies supply geometry but are not additional observations. Equal periodic
coordinates are represented once with their positive integer multiplicity
`w_i`, and `N = sum_i w_i` counts the construction observations.

Assign one third of each incident triangle's area to its vertex:

    A_i = (1/3) sum_{triangle corners at i} area(T)
    rho_i = w_i / A_i.

The assigned areas must be positive. This is raw DTFE: no area-floor correction
and no Gaussian smoothing are applied. The piecewise-affine density integrates
to `N`. Dividing every density and threshold by `N` gives probability-density
units without changing any selected region. The actual contour construction
below uses whole triangles, rather than contours through that affine field.

For each triangle define

    g_T = min_{i in T} rho_i
    R_T = circumradius(T), measured in degrees.

A triangle passes a density gate only if all three corners pass. This excludes
the partial-triangle wedges admitted by a direct affine contour, which caused
much of the earlier spiky CisPro geometry. That construction choice comes
before alpha and must not be mistaken for an effect of the alpha cutoff.

## 2. Calibrate both whole-triangle regions at infinity

Before excluding any large triangles, the inclusion height of observation `i` is

    h_i(infinity) = max {g_T : T incident to i}.

For `q = 49/50` and `199/200`, sort those heights downward and accumulate integer
weights. Choose the largest inclusive gate `t_q` covering at least `ceil(q N)`
observations. Include an entire equal-height group. Save the strict-above
count, tie weight, inclusive count, and overshoot. The baseline regions are

    S_q(infinity) = union of closed triangles with g_T >= t_q.

These are the 98% and 99.5% construction-calibrated regions. The inner region
is contained in the outer one because `t_98 >= t_99.5`. The labels refer to
counts of construction observations, not fractions of integrated density mass
or guaranteed coverage of new observations.

Use closed triangle unions: a shared vertex can connect two foreground
triangles, including across a periodic seam. Observations at shared edges or
vertices are covered if any incident selected triangle contains them. Their
weights are counted once, not once per incident triangle.

## 3. Apply the shared guarded minimum-area alpha rule

Hold both calibrated density gates fixed. At radius `a`, consider

    S_q(a) = union of closed triangles with R_T <= a and g_T >= t_q.

The same `a` is used at both levels. Because the gates remain fixed, these
regions are nested as alpha increases. One alpha candidate includes every
triangle tied at its radius event; alpha is a circumradius, not a squared
radius or density threshold.

A candidate must pass all four requirements:

1. **Preserve the complete 98% region.** Every triangle in `S_98(infinity)`
   remains selected. This also preserves its observation count and geometry.
2. **Retain the required 99.5% coverage.** The outer region covers at least
   `ceil(199 N / 200)` construction observations. It may lose observations
   from an initial tie overshoot while still satisfying the target.
3. **Create no new complement component.** Every connected component of the
   candidate's open complement contains at least one triangle from the
   baseline outer complement. This prevents a new isolated excluded region,
   even when a different old complement component merges elsewhere.
4. **Permit only supported outer splits.** If a baseline outer component
   splits, every surviving piece must contain at least one unchanged inner
   triangle. Existing outer components with no inner core may remain or
   shrink without being forced to disappear, but may not split into new
   coreless pieces.

The fourth guard does not require all inner components to remain separate.
Two inner cores may legitimately share one outer component, as in TransPro.
There is no requested number of CisPro components or special rule keyed to its
class name. Foreground components use closed-triangle vertex connectivity;
the open complement connects through shared edges of unselected triangles,
not through a foreground vertex. Both calculations identify periodic seams.

Among passing candidates, **minimize exact outer triangle area**. Since the
entire inner region is fixed, minimizing the sum of both areas gives the same
answer. Enumerate all active outer-triangle circumradius events that can
preserve the inner region and the required outer count, together with infinity.
Events below either exact feasibility bound cannot pass and can be skipped.
This is not the old eight-candidate radius grid.

For identical selected triangle sets, prefer the least restrictive evaluated
event, with infinity preferred when it gives the same geometry. Infinity itself
passes the guards by construction, so the procedure need not force a finite
cutoff. Save the full decision before reading any published-reference mask.
The radius is computed again from each new input dataset; the six values below
are results, not fitted constants to transplant unchanged.

### Why the alpha-stage gates remain properly calibrated

For any gate above `t_q`, the uncut region already covered too few observations.
A finite-alpha region at that higher gate is a subset and cannot cover more.
If the selected cutoff still meets the count at `t_q`, then `t_q` remains its
largest valid construction gate. Thus fixed gates preserve both calibration
and nesting; no downward recalibration is needed in this protocol.

The guards preserve specific geometric properties, not every aspect of
population topology. A component count alone does not establish the identities
of its features, and the complement guard is not a general guarantee about all
torus homology classes or all boundary distances. These are assessed separately.

## 4. Six-class results of the alpha stage and implementation

| Class | Selected alpha | Outer baseline area removed |
|---|---:|---:|
| General | 1.39511469° | 0.0229% |
| Gly | 5.98135638° | 0.0115% |
| TransPro | 2.95268237° | 0.0778% |
| CisPro | 14.55507444° | 11.5285% |
| PrePro | 3.09104371° | 0.6690% |
| IleVal | 2.36932142° | 0.0806% |

All six 98% regions are identical to their uncut baselines. Only CisPro's outer
component splits; its bridge is removed, both surviving pieces retain inner
triangles, and its 3,506 covered outer observations remain. No selected region
creates a new complement component. PrePro and IleVal each lose one covered
outer observation but still meet or exceed their required counts.

Post-construction reference assessment changes CisPro's outer Jaccard from
0.682526 to 0.712392. The other five decrease by at most 0.002389. Alpha therefore
improves the targeted CisPro geometry while causing small measured changes
elsewhere; it is not an overlap improvement in every class, and it does not
remove all pre-existing small islands or boundary discrepancies.

The [selection runner](../novak_work/diagnose_uniform_guarded_alpha.py),
[topology guards](../novak_work/rama_uniform_alpha_guards.py),
[sealed construction](../novak_work/validation_results/uniform_unsmoothed_alpha_guarded_v1/terminal.json),
and [published-reference assessment](../novak_work/validation_results/uniform_unsmoothed_alpha_guarded_assessment_v1/README.md)
record the implemented procedure. The [supporting audit](UNSMOOTHED_ALPHA_UNIFORM_RULE_AUDIT.md)
explains the mathematical choices, preceding failures, exact results, figures,
boundary measures, and thirteen focused guard tests. Its earlier candidate-status
wording records the assessment stage before the present adoption decision.

## 5. Adopted third stage: epsilon cleanup at the first 1.25× plateau

The whole-triangle union is the input to the adopted final cleanup stage.
Use the first joint plateau whose masks remain unchanged over [k,1.25k] at
both 98% and 99.5%, on both 768-square and 1536-square grids. Require matching
Betti numbers between grids, nonempty regions, and nesting at each resolution.
The same rule applies to every class, with one selected k shared by its levels.

Use thresholds kL and L/k for component removal and protected hole filling.
The fixed number 1.25 is the stability-span requirement, not k. Stop at the
first qualifying plateau; do not seek a later simpler region or substitute a
longer stability requirement. If none qualifies in [1,64], report unresolved.
The current CisPro result k = 1 validly requests no further cleanup after alpha.

The [complete procedure](UNIFORM_ALPHA_THEN_EPSILON.md) gives the cleanup rules,
endpoint and tie conventions, derived ratios, figures and measurements. The
alpha-stage construction counts above remain exact, but the final cleanup can
change coverage and is measured on a raster. It does not inherit the statistical
guarantee of the topology paper. The residual features of the 1.25× choice are
accepted for the current working version and remain visible in the records.

## 6. Scope, migration, and provenance

Together with the final cleanup specification, this is the adopted working
protocol for the present two-dimensional project. Resampling stability, independent-population coverage, Top2018 transfer,
matched end-to-end benchmarking, and higher-dimensional whole-simplex behavior
remain to be validated. Additional coordinates will be chi angles. No
end-to-end speed advantage or universally satisfactory shape is established.

The [new expository paper](../paper/dtfe_alpha_epsilon/README.md) now describes
this core construction and the adopted 1.25× cleanup. The older manuscript,
[notebook](../novak_work/03_density_adaptive_alpha_shapes.ipynb), and
[standalone entrypoint](../novak_work/alpha_dtfe/README.md) retain their earlier
regimes; creating the exposition does not migrate those implementations. The current selection
runner uses the available saved geometry; migration of the adopted recipe to
a portable new-data interface remains work to complete. Older status and
refinement documents retain their original experimental scope.

The [pre-adoption guide snapshot](../archive/uniform_alpha_adoption_2026-09-24/README.md)
preserves these guides immediately before this working-method decision. The
[earlier snapshot](../archive/unsmoothed_whole_triangle_policy_2026-09-24/README.md)
preserves the smoothed 3°/5° recipe. The smoothed 43.17493° CisPro result and the
[class-specific 14.99046° diagnostic](../novak_work/validation_results/unsmoothed_alpha_policy_v1/)
remain historical results, not alternate defaults. No experiment or archive
was removed or relabeled as an adopted run.

The [epsilon-adoption snapshot](../archive/epsilon_span125_adoption_2026-09-24/SNAPSHOT.md) preserves the preceding two-stage-only guide.
