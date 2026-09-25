"""Compact exact topology of closed triangle regions in one periodic 2D mesh.

Integer base IDs and integer period lifts distinguish winding and parallel
edges. A one-time array/sparse-graph validation checks the whole oriented torus,
including circular vertex links. Region queries then count cells and connected
components without allocating Python objects per edge or triangle. The supplied
geometry certificate is still a provenance precondition: this module does not
recheck geometric embedding or a different interpolated density's topology.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integers(value, name, *, empty=False):
    if empty and isinstance(value, (list, tuple)) and not value:
        return np.empty(0, dtype=np.int64)
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers, not floats or booleans")
    if array.size and (int(array.min()) < -(1 << 63) or int(array.max()) >= (1 << 63)):
        raise ValueError(f"{name} must fit signed 64-bit integers")
    return array


def _components(left, right, size):
    """Undirected connectivity; sparse library does the graph traversal in C."""
    graph = coo_matrix((np.ones(len(left), dtype=bool), (left, right)),
                       shape=(size, size)).tocsr()
    return connected_components(graph, directed=False, return_labels=True)


def _ordered_rows(rows):
    # Structured scalar records avoid tuple/dictionary objects and permit an
    # exact lexicographic sort, including signed winding offsets.
    records = np.ascontiguousarray(rows).view(
        np.dtype([(f"f{i}", rows.dtype) for i in range(rows.shape[1])])).reshape(-1)
    return records, np.argsort(records, kind="stable")


def _unique_faces(bases, lifts):
    """Canonicalize all three corners, invariant under common translations."""
    count = len(bases)
    corner_order = np.lexsort((lifts[..., 1], lifts[..., 0], bases), axis=1)
    ordered_bases = np.take_along_axis(bases, corner_order, axis=1)
    ordered_lifts = np.take_along_axis(lifts, corner_order[..., None], axis=1)
    ordered_lifts -= ordered_lifts[:, :1, :].copy()
    rows = np.empty((count, 9), dtype=np.int64)
    rows[:, 0::3] = ordered_bases
    rows[:, 1::3] = ordered_lifts[..., 0]
    rows[:, 2::3] = ordered_lifts[..., 1]
    del corner_order, ordered_bases, ordered_lifts
    records, order = _ordered_rows(rows)
    sorted_records = records[order]
    if np.any(sorted_records[1:] == sorted_records[:-1]):
        raise ValueError("The complete mesh has duplicate periodic faces")


class PeriodicTriangleTopology:
    """Validate/compile one torus and measure selected closed triangle unions.

    Inputs are integer ``bases[F,3]`` and ``lifts[F,3,2]``; IDs are zero-based,
    lifts are integer period counts, and face orientations must be coherent.
    All vertices must appear. The explicit ``max_triangles`` budget is checked
    before broad allocation. The implementation uses signed 32-bit graph IDs,
    so at most INT32_MAX / 6 triangles are accepted; periodic lift differences
    must fit signed int64 (including negation). Inputs are copied as needed,
    and later caller mutations cannot alter the compiled mesh.

    For a proper selected subcomplex of a connected closed torus, beta_2=0;
    for the whole torus beta_2=1. Integer Euler characteristic then gives beta_1
    from beta_0. This counts vertex contact as connection. Returned component
    labels describe actual selected triangles, not raster pixels or densities.
    """

    def __init__(self, bases, lifts, *, nvertices, geometry_certificate,
                 max_triangles=1_000_000):
        nvertices = _positive(nvertices, "nvertices")
        budget = _positive(max_triangles, "max_triangles")
        if not isinstance(geometry_certificate, Mapping) or geometry_certificate.get("certified") is not True:
            raise ValueError("An explicit certified-torus geometry provenance record is required")
        bases = _integers(bases, "bases")
        lifts = _integers(lifts, "lifts")
        if bases.ndim != 2 or bases.shape[1] != 3 or not 0 < len(bases) <= budget:
            raise ValueError("Use nonempty [triangle, 3] bases within max_triangles")
        nfaces = len(bases)
        if 6 * nfaces > np.iinfo(np.int32).max or nvertices > 3 * nfaces:
            raise ValueError("Mesh exceeds the compact signed-32-bit graph index budget")
        if lifts.shape != (nfaces, 3, 2):
            raise ValueError("lifts must have shape [triangle, 3, 2]")
        if int(bases.min()) < 0 or int(bases.max()) >= nvertices:
            raise ValueError("Base vertex IDs are outside [0, nvertices)")
        # A common translation may be close to an int64 endpoint. Comparing
        # extrema as Python integers ensures subtraction cannot overflow.
        lifts = lifts.astype(np.int64, copy=False)
        for axis in range(2):
            if int(lifts[..., axis].max()) - int(lifts[..., axis].min()) > np.iinfo(np.int64).max:
                raise ValueError("Periodic lift differences exceed safe signed-64-bit arithmetic")
        bases = bases.astype(np.int32, copy=True)
        used = np.zeros(nvertices, dtype=bool)
        used[bases] = True
        if not np.all(used):
            raise ValueError("The complete mesh omits base vertices")
        del used
        _unique_faces(bases, lifts)

        # All oriented edge occurrences, in [face, edge] order. Canonical
        # reversal compares (base_left,base_right,dx,dy) lexicographically.
        edge_rows = np.empty((nfaces, 3, 4), dtype=np.int64)
        edge_rows[..., 0] = bases
        edge_rows[..., 1] = bases[:, (1, 2, 0)]
        edge_rows[..., 2:] = lifts[:, (1, 2, 0), :] - lifts
        left, right = edge_rows[..., 0], edge_rows[..., 1]
        dx, dy = edge_rows[..., 2], edge_rows[..., 3]
        same = left == right
        if np.any(same & (dx == 0) & (dy == 0)):
            raise ValueError("A triangle has a collapsed periodic edge")
        forward = (left < right) | (same & ((dx < 0) | ((dx == 0) & (dy < 0))))
        reverse = ~forward
        edge_rows[..., 0] = np.where(forward, bases, bases[:, (1, 2, 0)])
        edge_rows[..., 1] = np.where(forward, bases[:, (1, 2, 0)], bases)
        edge_rows[..., 2:][reverse] *= -1
        del left, right, dx, dy, same, reverse
        records, order = _ordered_rows(edge_rows.reshape(-1, 4))
        sorted_records = records[order]
        if len(order) % 2 or np.any(sorted_records[0::2] != sorted_records[1::2]) or np.any(sorted_records[1:-1:2] == sorted_records[2::2]):
            raise ValueError("Each periodic edge must have exactly two incident faces")
        signs = forward.reshape(-1)[order]
        if np.any(signs[0::2] == signs[1::2]):
            raise ValueError("Each periodic edge must have two oppositely oriented incident faces")
        nedges = len(order) // 2
        edge_ids = np.empty(3 * nfaces, dtype=np.int32)
        edge_ids[order] = np.repeat(np.arange(nedges, dtype=np.int32), 2)
        edge_ids = edge_ids.reshape(nfaces, 3)
        endpoints = edge_rows.reshape(-1, 4)[order[0::2], :2].astype(np.int32)
        del records, sorted_records, order, signs, edge_rows
        if nvertices - nedges + nfaces != 0:
            raise ValueError("The complete mesh must have torus Euler characteristic zero")
        count, labels = _components(endpoints[:, 0], endpoints[:, 1], nvertices)
        if count != 1:
            raise ValueError("The complete mesh is disconnected")
        del labels

        # An edge has TWO endpoint germs even for a torus loop. The triangular
        # corners connect these germs into a link. Degree two and one connected
        # link per vertex exclude pinched surfaces while retaining loop edges.
        incoming = (2 * edge_ids[:, (2, 0, 1)] + forward[:, (2, 0, 1)]).reshape(-1)
        outgoing = (2 * edge_ids + ~forward).reshape(-1)
        degrees = np.bincount(incoming, minlength=2 * nedges)
        degrees += np.bincount(outgoing, minlength=2 * nedges)
        if not np.all(degrees == 2):
            raise ValueError("A complete-mesh vertex link is not a circle")
        del degrees, forward
        _, link_labels = _components(incoming, outgoing, 2 * nedges)
        del incoming, outgoing
        lowest = np.full(nvertices, np.iinfo(np.int32).max, dtype=np.int32)
        highest = np.full(nvertices, -1, dtype=np.int32)
        np.minimum.at(lowest, endpoints.reshape(-1), link_labels)
        np.maximum.at(highest, endpoints.reshape(-1), link_labels)
        if np.any(lowest != highest):
            raise ValueError("A complete-mesh vertex has a disconnected link")
        del link_labels, lowest, highest
        for array in (bases, edge_ids, endpoints):
            array.flags.writeable = False
        self._bases, self._edge_ids, self._endpoints = bases, edge_ids, endpoints
        self.nvertices, self.ntriangles, self.nedges = nvertices, nfaces, nedges

    def region(self, selected_triangles, *, component_labels=False):
        """Measure unique selected IDs; optionally return all-face labels.

        ``triangle_component_labels`` is int32[F], with -1 for unselected
        triangles. Nonnegative labels are consecutive, ordered by the smallest
        base vertex in each component. They include closed corner/seam contact.
        Label arrays are optional because most scalar metric calls need only
        Betti numbers. The caller owns source/hash and saved-mask provenance.
        """
        if not isinstance(component_labels, (bool, np.bool_)):
            raise ValueError("component_labels must be Boolean")
        selected = _integers(selected_triangles, "selected_triangles", empty=True)
        if selected.ndim != 1 or len(selected) > self.ntriangles:
            raise ValueError("Use a one-dimensional vector of unique triangle IDs")
        if len(selected) and (int(selected.min()) < 0 or int(selected.max()) >= self.ntriangles):
            raise ValueError("Selected triangle IDs are outside the complete mesh")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Selected triangle IDs must be unique")
        selected = selected.astype(np.int64, copy=False)
        empty, whole = len(selected) == 0, len(selected) == self.ntriangles
        vertex_labels = None
        if empty:
            vertices = edges = components = 0
        elif whole:
            vertices, edges, components = self.nvertices, self.nedges, 1
        else:
            active = np.zeros(self.nvertices, dtype=bool)
            active[self._bases[selected]] = True
            retained = np.zeros(self.nedges, dtype=bool)
            retained[self._edge_ids[selected]] = True
            endpoints = self._endpoints[retained]
            vertices, edges = int(active.sum()), len(endpoints)
            total_components, labels = _components(endpoints[:, 0], endpoints[:, 1], self.nvertices)
            # Unused vertices appear as isolated nodes in the sparse graph.
            components = int(total_components - (self.nvertices - vertices))
            if component_labels:
                retained_labels = np.unique(labels[active])
                vertex_labels = np.full(self.nvertices, -1, dtype=np.int32)
                vertex_labels[active] = np.searchsorted(retained_labels, labels[active])
        faces = len(selected)
        euler = vertices - edges + faces
        beta2 = int(whole)
        beta1 = components + beta2 - euler
        if beta1 < 0:
            raise ArithmeticError("Selected-cell incidence and connectivity disagree")
        result = dict(status="computed", betti=[components, beta1, beta2], components=components,
            euler_characteristic=euler, cell_counts=dict(vertices=vertices, edges=edges, faces=faces),
            empty=empty, whole_domain=whole,
            full_mesh=dict(vertices=self.nvertices, edges=self.nedges, triangles=self.ntriangles,
                connected=True, euler_characteristic=0, opposite_edge_pairings=True,
                connected_circular_vertex_links=True, certified_geometry_provenance=True),
            interpretation="Exact integer topology over F2 of the closed periodic triangle union, conditional on the supplied certified embedded torus; not raster topology or topology of a different continuous score field")
        if component_labels:
            labels = np.full(self.ntriangles, -1, dtype=np.int32)
            labels[selected] = 0 if whole else (-1 if empty else vertex_labels[self._bases[selected, 0]])
            result["triangle_component_labels"] = labels
        return result
