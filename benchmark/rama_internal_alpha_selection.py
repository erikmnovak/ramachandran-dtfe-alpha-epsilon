"""Choose one shared alpha against a fixed DTFE field, without a KDE input.

This pure pilot rule consumes already measured records. It does not calibrate,
sample a field, compute components, read files or choose a smoothing width.
An introduced join means one exact triangle component overlaps more than one
sampled direct-field component. Screening such joins does not screen splits,
missing components, holes or inaccurate boundaries, and is not a certificate
for the continuous bilinear level set.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
from numbers import Real

_TARGETS = (.98, .995)
_GRIDS = (768, 1536)


def _validated(candidates):
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("Supply a nonempty candidate list")
    ids, result = set(), []
    expected = {(q, n) for q in _TARGETS for n in _GRIDS}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be a record")
        name = candidate.get("id")
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("Candidate IDs must be distinct nonempty strings")
        ids.add(name)
        if "alpha_deg" not in candidate:
            raise ValueError("An explicit absolute alpha is required")
        alpha = candidate["alpha_deg"]
        if alpha is not None and alpha != "Inf":
            if (isinstance(alpha, bool) or not isinstance(alpha, Real)
                    or not math.isfinite(alpha) or alpha <= 0):
                raise ValueError("Alpha must be positive finite, 'Inf', or unavailable None")
        regions = candidate.get("regions")
        if not isinstance(regions, (list, tuple)) or len(regions) != 4:
            raise ValueError("Exactly two targets by two grids are required")
        seen, rows = set(), {}
        for region in regions:
            if not isinstance(region, dict):
                raise ValueError("Each measured region must be a record")
            q, n = region.get("target"), region.get("grid_size")
            if (isinstance(q, bool) or not isinstance(q, Real) or q not in _TARGETS
                    or type(n) is not int or n not in _GRIDS):
                raise ValueError("Use targets .98/.995 and integer grids 768/1536")
            key = (q, n)
            if key in seen:
                raise ValueError("Duplicate target/grid row")
            seen.add(key)
            defined = region.get("defined")
            if type(defined) is not bool:
                raise ValueError("Region defined status must be Boolean")
            keys = ("intersection_cells", "union_cells", "introduced_join_count")
            if any(k not in region for k in keys):
                raise ValueError("Every region requires explicit metric fields, including nulls")
            inter, union, joins = (region[k] for k in keys)
            if defined:
                if (any(type(x) is not int for x in (inter, union, joins))
                        or not 0 <= inter <= union <= n*n or union == 0
                        or joins < 0 or joins > n*n or alpha is None):
                    raise ValueError("Defined rows require valid integer overlaps, positive union, joins and alpha")
            elif any(x is not None for x in (inter, union, joins)):
                raise ValueError("Undefined rows must retain null metrics, never fabricated zeros")
            rows[key] = deepcopy(region)
        if seen != expected:
            raise ValueError("Missing target/grid combination")
        result.append((deepcopy(candidate), rows))
    return result


def _choose(validated, grids):
    """Rank complete measurements, preserving the full exact-score tie set."""
    ranked, reports = [], []
    for candidate, rows in validated:
        chosen = [rows[q, n] for q in _TARGETS for n in grids]
        complete = all(row["defined"] for row in chosen)
        violations = [dict(target=row["target"], grid_size=row["grid_size"],
                           introduced_join_count=row["introduced_join_count"])
                      for row in chosen if row["defined"] and row["introduced_join_count"]]
        losses = []
        if complete:
            losses = [Fraction(row["union_cells"]-row["intersection_cells"], row["union_cells"])
                      for row in chosen]
            worst, total = max(losses), sum(losses, Fraction())
            ranked.append((candidate, worst, total, not violations))
        reports.append(dict(id=candidate["id"], alpha_deg=candidate["alpha_deg"],
            complete=complete, admissible=complete and not violations,
            introduced_join_rows=violations,
            losses=[dict(target=row["target"], grid_size=row["grid_size"],
                         ratio=[value.numerator, value.denominator])
                    for row, value in zip(chosen, losses)],
            worst_loss_ratio=[worst.numerator, worst.denominator] if complete else None,
            summed_loss_ratio=[total.numerator, total.denominator] if complete else None))

    def _choice(pool):
        if not pool:
            return None
        best = min((worst, total) for _, worst, total, _ in pool)
        tied = [c for c, worst, total, _ in pool if (worst, total) == best]
        # The ID orders only indistinguishable absolute-alpha ties; it does not
        # silently choose between different overlap scores or different radii.
        tied.sort(key=lambda c: (c["alpha_deg"] == "Inf",
                                0 if c["alpha_deg"] == "Inf" else c["alpha_deg"], c["id"]))
        selected = tied[0]
        return dict(selected_id=selected["id"], alpha_deg=selected["alpha_deg"],
            score_tied_ids=sorted(c["id"] for c in tied),
            smallest_alpha_tied_ids=sorted(c["id"] for c in tied if c["alpha_deg"] == selected["alpha_deg"]),
            worst_loss_ratio=[best[0].numerator, best[0].denominator],
            summed_loss_ratio=[best[1].numerator, best[1].denominator])

    admissible = [row for row in ranked if row[3]]
    status = ("selected" if admissible else "unresolved_no_admissible_candidate"
              if ranked else "unresolved_no_complete_candidate")
    return dict(status=status, selection=_choice(admissible), candidates=reports,
                unconstrained_diagnostic=_choice(ranked) if not admissible else None)


def select_shared_alpha(candidates):
    """Select a shared radius from exact saved DTFE-to-DTFE measurements.

    Each candidate supplies ``id``, ``alpha_deg`` (positive finite, ``'Inf'``,
    or unavailable ``None``), and four ``regions``. A region identifies target
    .98/.995 and grid_size 768/1536, with Boolean ``defined``, integer
    ``intersection_cells``, ``union_cells`` and ``introduced_join_count``.
    Undefined measurements use explicit null metric values. Additional input
    diagnostics are copied into the output without affecting selection.

    Complete candidates with no introduced join at either target on either
    grid minimize, lexicographically, the worst and then summed exact
    XOR/union losses. The smallest absolute alpha breaks exact score ties;
    infinity is last. Every score-tied ID is retained. If no candidate passes,
    selection stays unresolved and an unconstrained diagnostic is separate.
    Single-grid choices are sensitivity records and never replace the joint
    choice. No epsilon, acceptance tolerance, KDE score or automatic fallback
    enters the rule. The caller owns construction calibration and the meaning
    of the exact-component/sampled-component overlap counts.
    """
    validated = _validated(candidates)
    result = _choose(validated, _GRIDS)
    result.update(format="dtfe-internal-alpha-selection-1",
        criteria=dict(targets=list(_TARGETS), grids=list(_GRIDS),
            screen="No exact triangle component joins multiple sampled direct-field components",
            ranking=["minimum worst exact XOR/union", "minimum sum of exact XOR/union",
                     "smallest absolute alpha; infinity last", "lexicographic ID only for equal alpha"],
            no_admissible="Unresolved; unconstrained best retained only as a diagnostic",
            connectivity_scope="Introduced joins only; splits, misses, unsampled components and holes are not screened"),
        input_candidates=[candidate for candidate, _ in validated],
        single_grid_sensitivity={str(n): _choose(validated, (n,)) for n in _GRIDS})
    return result
