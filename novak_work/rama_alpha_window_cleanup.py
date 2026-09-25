"""Experimental density-window cleanup inside a fixed alpha-eligible domain.

The alpha step has already happened when this module receives ``eligible``.
It supplies a spatial constraint. The scalar field can be the interpolated
DTFE density or a DTFE-derived whole-triangle gate score; the latter is not a
normalized density interpolation and may be zero outside the alpha domain.
For a central threshold
L and density ratio k >= 1, use high = k*L and low = L/k. We remove raw
components without a high-density core, then fill eligible shallow holes.

This is a geometric heuristic. Bobrowski et al. estimate image homology of
nested sets; they do not specify this cleaned region, prove this cleanup
correct, or establish a guarantee for DTFE with an alpha-domain restriction.
In particular, the returned region need not realize the image-homology ranks.
"""
import numpy as np

from rama_topology_cleanup_preview import periodic_labels


class AlphaWindowCleanup:
    """Reuse components of ``(density >= level) & eligible`` across ratios.

    Arrays are square rasters on the periodic angle domain. Foreground uses
    eight-neighbor connectivity, background four-neighbor connectivity. Those
    are display conventions; the resulting topology is raster topology, not
    an exact homology calculation on the underlying Delaunay complex.

    Inputs are copied so subsequent caller mutations cannot alter this fitted
    object. ``eligible`` must be Boolean and match the density array's shape.
    All cells are eligible for infinite alpha.
    """

    def __init__(self, density, level, eligible):
        values = np.asarray(density, dtype=float)
        domain = np.asarray(eligible)
        if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
            raise ValueError('Density must be a nonempty square two-dimensional raster')
        if domain.dtype != bool or domain.shape != values.shape:
            raise ValueError('Eligibility must be a Boolean raster matching density')
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Density or gate-score values must be nonnegative and finite')
        if not np.isscalar(level) or not np.isfinite(level) or level <= 0:
            raise ValueError('Level must be a positive finite scalar')
        self.density = values.copy()
        self.eligible = domain.copy()
        self.level = float(level)
        self.raw = (self.density >= self.level) & self.eligible
        self.foreground, _ = periodic_labels(self.raw, 8)
        self.peaks = np.full(int(self.foreground.max()) + 1, -np.inf)
        np.maximum.at(self.peaks, self.foreground.ravel(), self.density.ravel())

    def apply(self, factor):
        """Return the cleaned Boolean mask and an explicit record of edits.

        1. Delete an entire foreground component if its peak is below k*L.
        2. Relabel the complement *after* deletion. This prevents a discarded
           weak ring from leaving a newly filled island at its former centre.
        3. Fill a background component only if its minimum density is >= L/k,
           it does not wind around either periodic axis, it is not tied for
           largest background area, and it contains no alpha-ineligible cell.

        The last condition protects the *whole* background component, rather
        than filling its eligible subset and possibly restoring an alpha-cut
        connection. High-density cores are protected only inside the alpha
        domain. The returned mask is always a subset of ``eligible``.

        With k=1 the mask is unchanged. With an all-True domain this is exactly
        the earlier multiplicative_cleanup core/pit rule. A larger k can both
        remove foreground and fill background; no universal monotonicity or
        nesting of outputs at distinct construction levels is claimed here.
        """
        if not np.isscalar(factor) or not np.isfinite(factor) or factor < 1:
            raise ValueError('Use a finite density ratio factor >= 1')
        high, low = self.level * float(factor), self.level / float(factor)
        if not np.isfinite(high) or low <= 0:
            raise ValueError('Density-window thresholds must remain positive and finite')

        remove_labels = self.peaks < high
        remove_labels[0] = False
        intermediate = self.raw & ~remove_labels[self.foreground]
        background, winding = periodic_labels(~intermediate, 4)
        count = int(background.max()) + 1
        pits = np.full(count, np.inf)
        np.minimum.at(pits, background.ravel(), self.density.ravel())
        sizes = np.bincount(background.ravel(), minlength=count)
        sizes[0] = 0

        alpha_protected = np.zeros(count, bool)
        # Every excluded cell is background. Mark its whole component, which
        # also protects any eligible shallow cells connected to that gap.
        alpha_protected[np.unique(background[~self.eligible])] = True
        alpha_protected[0] = False
        protected = winding.any(axis=1) | alpha_protected
        protected[0] = True
        if sizes.max() > 0:
            # Preserve every maximum-size tie: array traversal order must not
            # decide which otherwise identical cavity becomes the exterior.
            protected |= (sizes == sizes.max()) & (sizes > 0)
        fill_labels = (pits >= low) & ~protected
        result = intermediate | fill_labels[background]

        if np.any(result & ~self.eligible):
            raise ArithmeticError('Cleanup restored alpha-ineligible cells')
        if np.any(self.eligible & (self.density >= high) & ~result):
            raise ArithmeticError('Cleanup removed an eligible high-density core')
        if np.any(result & (self.density < low)):
            raise ArithmeticError('Cleanup escaped its lower density threshold')
        removed = self.raw & ~result
        filled = ~self.raw & result
        edits = dict(
            removed_component_count=int(remove_labels.sum()),
            filled_background_component_count=int(fill_labels.sum()),
            alpha_protected_background_component_count=int(alpha_protected.sum()),
            alpha_ineligible_cells=int(np.count_nonzero(~self.eligible)),
            removed_cells=int(removed.sum()),
            filled_cells=int(filled.sum()),
            changed_cells=int(np.count_nonzero(result != self.raw)),
            removed=removed,
            filled=filled,
        )
        return result, edits
