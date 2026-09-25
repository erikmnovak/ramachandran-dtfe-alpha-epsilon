# Code supporting the DTFE–alpha–epsilon paper

The adopted protocol is described in the paper and in
[the policy JSON](documents/DTFE_ALPHA_EPSILON_POLICY.json).
This publication subset preserves the implementation files without rewriting
the scientific algorithms or presenting old experimental defaults as current.

## The three stages

1. **Periodic geometry and DTFE:** `novak_work/RamaAlphaGeometry.jl`,
   `RamaAlphaContours.jl`, and its `rama_alpha_contours/` includes provide
   the periodic triangulation, density-area accounting, and artifact utilities.
   These shared modules also contain older optional operations; the adopted
   protocol uses raw density, no smoothing, and no area floor.
2. **Guarded alpha selection:** `novak_work/diagnose_uniform_guarded_alpha.py`
   implements the adopted selection on saved geometry. The guards are in
   `rama_uniform_alpha_guards.py` and the exact triangle topology in
   `rama_compact_triangle_topology.py`.
3. **Epsilon cleanup:** `novak_work/rama_alpha_window_cleanup.py` defines
   island removal and protected hole filling. `rama_alpha_cleanup_events.py`
   enumerates actual changes, and `rama_epsilon_plateau_selection.py`
   implements the first qualifying plateau. Use `select_first` with span
   **1.25** and the nonempty/nesting/cross-grid topology checks in the policy.

The included study runner `analyze_uniform_alpha_epsilon_stability.py` retains
its original comparisons, including a historical primary doubling branch.
Its `selections["1.25"]` output is the adopted branch. Similarly, older helper
or experiment defaults are not replacements for the versioned policy.

The accompanying Python modules include their transitive local imports. Some
helpers originated in earlier studies; inclusion as a dependency does not adopt
those earlier methods. The Julia environment is in `novak_work/Project.toml`
and `Manifest.toml` (Julia 1.12).

## What can be reproduced from this checkout?

The paper builds from its included figures and tables. Its source, bibliography,
full-precision numerical results, policy, and figure input hashes are included.
The pure Python method tests can run with:

```sh
python -m pip install -r requirements-paper.txt
PYTHONPATH=novak_work:benchmark python -m unittest discover -s novak_work -p 'test_rama_*.py'
```

Full Top8000 experiment reruns and `generate_figures.py` require the omitted
observation data and saved geometry/event-mask artifacts named in the scripts
and `paper/dtfe_alpha_epsilon/figure_generation.json`. The runners do not silently
replace missing data with synthetic examples. This subset is not yet a portable
end-to-end contour application for arbitrary new datasets.

The six selected alpha radii, cleanup strengths, agreement measurements, and
small policy records remain available without downloading those large arrays.
