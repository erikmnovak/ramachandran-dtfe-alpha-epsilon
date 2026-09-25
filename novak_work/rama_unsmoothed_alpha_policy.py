"""Explicit empirical alpha policy for raw 2D whole-triangle DTFE regions.

This is a class-specific research policy, not an estimator theorem or a
universal topology rule. Five classes keep infinity. CisPro protects the entire
two-component 98% region and chooses the largest circumradius event whose
99.5% region retains construction coverage, has two components without loops,
and puts the two protected inner components in distinct outer components.

The density thresholds are calibrated once at alpha=infinity and then fixed.
Consequently all candidate regions are nested as alpha increases. Maximizing
alpha retains the largest candidate region compatible with the constraints.
The numeric radius itself is learned from the current dataset; it is not a
hard-coded historical radius or a KDE-overlap optimum.
"""
from __future__ import annotations
import math

VERSION = 'raw_whole_triangle_class_policy_v1'
CLASSES = ('General', 'Gly', 'TransPro', 'CisPro', 'PrePro', 'IleVal')
TARGETS = ((98, 100), (995, 1000))


def default_alpha(category):
    """Return a declared infinity default, or None when CisPro needs a search."""
    if category not in CLASSES:
        raise ValueError('Unknown 2D residue class: '+str(category))
    return None if category == 'CisPro' else 'Inf'


def required_count(total, numerator, denominator):
    """Exact ceiling avoids floating point changes at integer rank boundaries."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError('Positive integer observation count required')
    if not 0 < numerator <= denominator:
        raise ValueError('Probability must be in (0,1]')
    return (numerator*total + denominator-1)//denominator


def cispro_rejection_reasons(row):
    """Return explicit failures, rather than replacing a failed search silently."""
    failures = []
    if row['inner_preserved'] is not True:
        failures.append('entire_98_percent_region_not_preserved')
    if row['outer_construction_count'] < row['outer_required_count']:
        failures.append('outer_construction_coverage_insufficient')
    labels = row['protected_core_outer_labels']
    if len(labels) != 2 or any(len(x) != 1 or x[0] < 0 for x in labels):
        failures.append('each_entire_inner_core_must_lie_in_one_outer_component')
    elif labels[0][0] == labels[1][0]:
        failures.append('protected_inner_cores_are_joined')
    if row['outer_exact_betti'] != [2, 0, 0]:
        failures.append('outer_region_must_have_two_components_and_no_loops')
    return failures


def select_cispro_event(rows, *, baseline_inner_betti):
    """Select the largest admissible *exact event*, with inclusive radius ties.

    Geometry, counts and labels must come from an exact periodic triangle
    complex, not a raster. The caller is responsible for evaluating every
    unique circumradius at which an admissible candidate could occur and Inf.
    No reference mask or Jaccard score is an input to this function.
    """
    if baseline_inner_betti[0] != 2:
        return dict(status='unresolved_inner_reference_not_two_components',
                    selection=None, admissible_events=0)
    valid = []
    for row in rows:
        value = math.inf if row['alpha_deg'] == 'Inf' else float(row['alpha_deg'])
        if not value > 0 or math.isnan(value):
            raise ValueError('Radius must be positive or Inf')
        if not cispro_rejection_reasons(row):
            valid.append((value, row))
    if not valid:
        return dict(status='unresolved_no_admissible_event', selection=None,
                    admissible_events=0)
    selected = max(valid, key=lambda item: item[0])[1]
    return dict(status='selected', selection=selected,
                admissible_events=len(valid),
                criterion='Largest exact radius event satisfying all fixed-threshold constraints')
