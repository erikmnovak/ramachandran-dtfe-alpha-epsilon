"""Periodic multilinear density evaluation with exact constant plateaus.

This owner supplies the interpolation arithmetic for subsequent direct-field
experiments. Earlier sealed Gaussian-DTFE experiments retain their recorded
weighted-corner summation. Both formulas represent the same multilinear field
in real arithmetic; their floating-point treatment of an inclusive tie differs.
"""
from itertools import product
import numpy as np


def evaluate_periodic_density(density, points, *, phase=0.0):
    """Evaluate nonnegative cell-center densities on a 360-degree angular torus.

    ``density`` is a Float64 cube with two through six axes in physical angle
    order, for example phi, psi, chi1. With n cells per axis, sample j lies at
    -180 + (j + phase + 1/2)*360/n degrees. ``phase`` is a fraction of a cell,
    either shared by all axes or specified for each axis, in [0, 1).
    ``points`` has shape (number of queries, number of angles). Returned values
    have one entry per query. Only the queried cells' corner values are read;
    those values must be finite and nonnegative.

    Interpolate one coordinate at a time using a + t*(b-a). When a equals b,
    the difference is exactly zero, so any representable constant is preserved
    exactly, including at an inclusive density threshold. Clipping to the two
    endpoints enforces the convex range in floating point. No tolerance is
    added to the contour threshold and no almost-equal values are tied.

    Queries are evaluated in bounded batches; no array proportional to the
    number of density-grid cells is copied or scanned during a query.
    """
    values = np.asarray(density)
    dimension = values.ndim
    if (values.dtype != np.float64 or not 2 <= dimension <= 6 or
            min(values.shape) < 2 or len(set(values.shape)) != 1):
        raise ValueError("Expected a Float64 density cube with 2 to 6 angular axes")
    q = np.asarray(points, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != dimension or not np.isfinite(q).all():
        raise ValueError("Queries must be finite [point, angle] coordinates")
    offset = np.asarray(phase, dtype=np.float64)
    if offset.ndim == 0:
        offset = np.full(dimension, float(offset))
    if offset.shape != (dimension,) or not np.isfinite(offset).all() or np.any((offset < 0) | (offset >= 1)):
        raise ValueError("Phase must give cell fractions in [0, 1) for every angle")
    size = values.shape[0]
    output = np.empty(len(q), dtype=np.float64)
    for first in range(0, len(q), 65536):
        stop = min(first + 65536, len(q))
        # Reduce the angle before shifting it, avoiding overflow for a large
        # finite input while keeping all subsequent grid coordinates bounded.
        wrapped = np.remainder(q[first:stop], 360.)
        coordinate = np.remainder(wrapped + 180., 360.) * (size / 360.) - offset - .5
        lower = np.floor(coordinate).astype(np.int64)
        fractions = coordinate - lower
        corners = np.stack([values[tuple(((lower + corner) % size).T)]
                            for corner in product((0, 1), repeat=dimension)])
        if not np.isfinite(corners).all() or np.any(corners < 0):
            raise ValueError("Queried density corners must be finite and nonnegative")
        # Lexicographic corner order pairs the last physical coordinate first.
        for axis in reversed(range(dimension)):
            a, b = corners[::2], corners[1::2]
            interpolated = a + fractions[:, axis] * (b - a)
            corners = np.clip(interpolated, np.minimum(a, b), np.maximum(a, b))
        output[first:stop] = corners[0]
    return output
