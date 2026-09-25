"""Optional geometric illustration of a density-window topology filter.

This is NOT the contour construction in Bobrowski et al.: that paper returns
image homology, without choosing a unique boundary. This conservative display
rule removes a component only when it has no high-density core, and fills a
background component only when it has no low-density core. Winding background
components and the largest background component are protected. It neither
repairs bridges nor promises to realize the image-homology vector exactly.
"""
import numpy as np
from scipy.ndimage import label


def periodic_labels(mask, connectivity=8):
    """Periodic component labels plus winding flags in the two angle axes.

    Ordinary labels are merged across seams. Integer translation potentials
    distinguish an island crossing the display seam from a component that
    really winds around the torus; merely touching a plot edge is insufficient.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool or mask.shape[0] != mask.shape[1] or connectivity not in (4, 8):
        raise ValueError('Use a square Boolean mask and connectivity 4 or 8')
    structure = np.ones((3, 3), np.uint8) if connectivity == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    lab, count = label(mask, structure)
    parent = np.arange(count+1)
    delta = np.zeros((count+1, 2), np.int64)
    winding = np.zeros((count+1, 2), bool)

    def find(i):
        root = i; d = np.zeros(2, np.int64)
        while parent[root] != root:
            d += delta[root]; root = parent[root]
        while parent[i] != i:
            following = parent[i]; old = delta[i].copy()
            parent[i] = root; delta[i] = d
            d -= old; i = following
        return root

    n = len(lab); positions = np.arange(n)
    pairs = []
    for shift in ((-1, 0, 1) if connectivity == 8 else (0,)):
        wrapped = (positions+shift) % n
        translations = (positions+shift)//n
        pairs.append(np.column_stack((lab[0, positions], lab[-1, wrapped], translations, -np.ones(n, int))))
        pairs.append(np.column_stack((lab[positions, 0], lab[wrapped, -1], -np.ones(n, int), translations)))
    pairs = np.unique(np.concatenate(pairs), axis=0)
    for a, b, dx, dy in pairs:
        if not a or not b: continue
        ra, rb = find(a), find(b)
        # delta[label] is its tile translation relative to its root.
        relation = np.array([dx, dy])+delta[a]-delta[b]
        if ra == rb:
            winding[ra] |= relation != 0
        else:
            parent[rb] = ra; delta[rb] = relation
            winding[ra] |= winding[rb]
    roots = np.array([find(i) for i in range(count+1)])
    unique = np.unique(roots[1:])
    remap = np.zeros(count+1, np.int32); remap[unique] = np.arange(1, len(unique)+1)
    labels = remap[roots][lab]
    flags = np.zeros((len(unique)+1, 2), bool); flags[1:] = winding[unique]
    return labels, flags


class CleanupPreview:
    """Precompute central-region components once, then inspect many epsilons."""
    def __init__(self, density, level):
        self.density = np.asarray(density, dtype=float)
        if not np.isfinite(self.density).all() or not np.isfinite(level) or level <= 0:
            raise ValueError('Finite density and positive finite level required')
        self.level = float(level)
        self.raw = self.density >= level
        self.foreground, _ = periodic_labels(self.raw, 8)
        self.peaks = np.full(int(self.foreground.max())+1, -np.inf)
        np.maximum.at(self.peaks, self.foreground.ravel(), self.density.ravel())

    def apply(self, epsilon):
        if not np.isfinite(epsilon) or not 0 <= epsilon < self.level:
            raise ValueError('Use 0 <= epsilon < level')
        high, low = self.level+epsilon, self.level-epsilon
        remove_labels = self.peaks < high; remove_labels[0] = False
        intermediate = self.raw & ~remove_labels[self.foreground]
        # Relabel after removal, so deleting a weak ring cannot leave a newly
        # filled centre as an artificial island.
        background, winding = periodic_labels(~intermediate, 4)
        pits = np.full(int(background.max())+1, np.inf)
        np.minimum.at(pits, background.ravel(), self.density.ravel())
        sizes = np.bincount(background.ravel(), minlength=len(pits)); sizes[0] = 0
        protected = winding.any(axis=1); protected[0] = True
        if sizes.max() > 0:
            # Protect all maximum-size ties, independently of the grid origin.
            protected |= (sizes == sizes.max()) & (sizes > 0)
        fill_labels = (pits >= low) & ~protected
        result = intermediate | fill_labels[background]
        removed = self.raw & ~result
        filled = ~self.raw & result
        if np.any((self.density >= high) & ~result) or np.any(result & (self.density < low)):
            raise ArithmeticError('Preview escaped its density uncertainty band')
        return result, dict(removed_component_count=int(remove_labels.sum()),
            filled_background_component_count=int(fill_labels.sum()),
            removed_cells=int(removed.sum()), filled_cells=int(filled.sum()),
            changed_cells=int(np.count_nonzero(result != self.raw)),
            removed=removed, filled=filled)


def construction_coverage(vertices, vertex_density, weights, level, epsilon, edits):
    """Coverage of the explicitly defined raster-guided continuous edit.

    The continuous region removes only f<level+epsilon within marked removal
    cells, and adds only f>=level-epsilon within marked fill cells. Thus vertex
    memberships use actual DTFE values, never a nearest-pixel density value.
    The masks supply spatial edit locations only. No recalibration is applied.
    """
    n = len(edits['removed'])
    cell = np.floor(((np.asarray(vertices)+180.) % 360.)*n/360.).astype(np.int64) % n
    x, y = cell[:, 0], cell[:, 1]
    values = np.asarray(vertex_density)
    raw = values >= level
    after = (raw & ~(edits['removed'][y, x] & (values < level+epsilon))) | (
        edits['filled'][y, x] & (values >= level-epsilon))
    return dict(raw_count=int(np.sum(weights[raw])), after_count=int(np.sum(weights[after])),
                total_count=int(np.sum(weights)), changed_count=int(np.sum(weights[raw != after])))
