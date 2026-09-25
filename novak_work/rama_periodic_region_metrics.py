"""Topology and boundary distances for Boolean cells on a two-dimensional torus.

Masks are indexed ``[psi, phi]``. A True entry includes the entire CLOSED
square cell, including its edges and corners. Consequently diagonally touching
cells are connected. This convention describes the raster itself; sampled
pixels need not have the topology of the continuous contour that produced them.

Boundary distances compare the midpoints of exposed cell edges. They neither
measure continuous contour Hausdorff distance nor create edges at a display
seam. The angular period is 360 degrees by default.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree


def _mask(value, max_cells):
    """Validate the common raster contract before allocating geometry arrays."""
    if (isinstance(max_cells, (bool, np.bool_)) or
            not isinstance(max_cells, (int, np.integer)) or max_cells < 1):
        raise ValueError("max_cells must be a positive integer")
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Use a two-dimensional Boolean mask indexed [psi, phi]")
    if mask.shape[0] != mask.shape[1] or mask.shape[0] == 0:
        raise ValueError("Use a nonempty square grid with equal angular cell widths")
    if mask.size > max_cells:
        raise ValueError("The raster exceeds max_cells; increase the explicit memory budget")
    return mask


def _periodic_components(mask):
    """Merge ordinary eight-neighbor labels across both identified seams."""
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    parent = np.arange(count + 1, dtype=np.int64)
    # Every missing eight-neighbor adjacency crosses a row seam, a column seam,
    # or both. Including shifts -1, 0, +1 covers the diagonal corner contacts.
    pairs = []
    for shift in (-1, 0, 1):
        pairs.append(np.column_stack((labels[0, :], np.roll(labels[-1, :], shift))))
        pairs.append(np.column_stack((labels[:, 0], np.roll(labels[:, -1], shift))))
    pairs = np.concatenate(pairs)
    pairs = np.unique(pairs[(pairs[:, 0] != 0) & (pairs[:, 1] != 0)], axis=0)
    for left, right in pairs:
        while parent[left] != left:
            parent[left] = parent[parent[left]]
            left = parent[left]
        while parent[right] != right:
            parent[right] = parent[parent[right]]
            right = parent[right]
        if left != right:
            parent[right] = left
            count -= 1
    return int(count)


def periodic_region_topology(mask, *, max_cells=16_777_216):
    """Return Betti numbers of a union of closed periodic square cells.

    Count vertices, edges and faces in the torus cell decomposition, with each
    seam-identified cell counted once. Euler characteristic is ``V - E + F``.
    Eight-neighbor connected components give beta_0. A proper cell subcomplex
    of this connected torus has beta_2 = 0; the full torus has beta_2 = 1.
    Therefore beta_1 = beta_0 + beta_2 - Euler characteristic.

    These are exact integer counts for the represented cell union (Betti
    numbers over F2, also equal to rational Betti numbers here), not a claim
    about a continuous density-level region. ``max_cells`` bounds the raster size
    before the temporary Boolean and labeling arrays are allocated.
    """
    mask = _mask(mask, max_cells)
    faces = int(np.count_nonzero(mask))
    empty, whole = faces == 0, faces == mask.size
    if empty:
        vertices = edges = components = 0
    elif whole:
        vertices, edges, components = faces, 2 * faces, 1
    else:
        # An edge belongs when either incident cell belongs. A vertex belongs
        # when any of its four incident cells belongs, including seam copies.
        vertical_neighbors = mask | np.roll(mask, 1, axis=0)
        vertices = int(np.count_nonzero(vertical_neighbors | np.roll(vertical_neighbors, 1, axis=1)))
        edges = int(np.count_nonzero(vertical_neighbors))
        edges += int(np.count_nonzero(mask | np.roll(mask, 1, axis=1)))
        components = _periodic_components(mask)
    euler = vertices - edges + faces
    beta2 = int(whole)
    beta1 = components + beta2 - euler
    if beta1 < 0:
        raise ArithmeticError("Cell counts and periodic connectivity disagree")
    return dict(status="computed", betti=[components, beta1, beta2],
        components=components, euler_characteristic=euler,
        cell_counts=dict(vertices=vertices, edges=edges, faces=faces),
        grid_shape=list(mask.shape), empty=empty, whole_domain=whole,
        interpretation="Closed periodic raster-cell union; corner contact connects cells; not certified topology of the underlying continuous contour")


def _boundary_midpoints(mask):
    """Interface midpoints on the unit torus, avoiding squared-distance scaling."""
    width = 1. / mask.shape[0]
    row_v, column_v = np.nonzero(mask ^ np.roll(mask, -1, axis=1))
    row_h, column_h = np.nonzero(mask ^ np.roll(mask, -1, axis=0))
    points = np.empty((len(row_v) + len(row_h), 2), dtype=np.float64)
    split = len(row_v)
    # Coordinates are [phi, psi], measured from an arbitrary common origin.
    # Translating that origin does not change any torus distance.
    points[:split, 0] = (column_v + 1.) * width
    points[:split, 1] = (row_v + .5) * width
    points[split:, 0] = (column_h + .5) * width
    points[split:, 1] = (row_h + 1.) * width
    np.remainder(points, 1., out=points)
    return points


def _directed_boundary_summary(source, destination, period):
    if not len(source) or not len(destination):
        return dict(status="empty_source_boundary" if not len(source) else "empty_target_boundary",
            sample_count=len(source), mean=None, median=None, p95=None, max=None)
    # Query on the unit torus and scale the summaries afterward. Squaring tiny
    # angular coordinates inside the tree would otherwise underflow to zero.
    distances = cKDTree(destination, boxsize=1.).query(source, k=1, workers=1)[0]
    return dict(status="computed", sample_count=len(source), mean=float(np.mean(distances)) * period,
        median=float(np.median(distances)) * period, p95=float(np.quantile(distances, .95)) * period,
        max=float(np.max(distances)) * period)


def periodic_boundary_comparison(left, right, period=360., *, max_cells=16_777_216):
    """Compare exposed-edge midpoint sets using nearest periodic distances.

    Both masks use the same square grid. Every exposed edge has the same
    length, so each midpoint receives equal weight in the directed mean and
    quantiles. Perimeter is exactly edge count times cell width for this raster.
    Percentiles use NumPy's linear quantile convention.

    Empty and full-domain masks both have no boundary. When either boundary is
    absent, distance summaries and symmetric_max are None with an explicit
    undefined status—even if both masks are equal. When both are present,
    symmetric_max is the Hausdorff distance of their finite midpoint sets only.
    Distances use the units of ``period`` (degrees for the default 360).
    """
    left, right = _mask(left, max_cells), _mask(right, max_cells)
    if left.shape != right.shape:
        raise ValueError("Both masks must have the same grid shape")
    if isinstance(period, (bool, np.bool_)) or not isinstance(period, Real):
        raise ValueError("period must be a positive finite real number")
    period = float(period)
    if not math.isfinite(period) or period <= 0.:
        raise ValueError("period must be a positive finite real number")
    width = period / left.shape[0]
    if width / 2. == 0.:
        raise ValueError("period and grid size do not permit representable Float64 distances")
    a, b = _boundary_midpoints(left), _boundary_midpoints(right)
    perimeter_a, perimeter_b = len(a) * width, len(b) * width
    if not math.isfinite(perimeter_a) or not math.isfinite(perimeter_b):
        raise ValueError("The raster perimeter is not representable as Float64")
    forward = _directed_boundary_summary(a, b, period)
    backward = _directed_boundary_summary(b, a, period)
    present = bool(len(a) and len(b))
    return dict(status="computed" if present else "undefined_missing_boundary",
        left_to_right=forward, right_to_left=backward,
        symmetric_max=max(forward["max"], backward["max"]) if present else None,
        boundary_edge_counts=dict(left=len(a), right=len(b)),
        perimeters=dict(left=perimeter_a, right=perimeter_b), period=period,
        cell_width=width, grid_shape=list(left.shape),
        interpretation="Nearest periodic distances between exposed raster-edge midpoints; not continuous contour Hausdorff distance; seams add no boundary")
