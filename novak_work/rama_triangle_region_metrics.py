"""Integer topology of closed triangles in a certified periodic 2D mesh.

The exported base-vertex IDs and integer period lifts identify the actual
quotient cells. In particular, two edges with the same endpoint IDs can be
DIFFERENT torus edges, and an edge can be a nontrivial loop at one base vertex.
A selected triangle includes all its edges and vertices, so corner contact
connects regions. No rasterization or point-location tolerance is used here.

This module validates the complete oriented combinatorial torus, including
vertex links. Its interpretation as the continuous embedded triangle union
requires the supplied geometry certificate to be true. It does not independently
recertify coordinates, Delaunay predicates, or geometric embedding, and says
nothing about the continuous topology of a different interpolated or KDE contour.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integer_array(value, name, *, empty_sequence=False):
    array = np.asarray(value)
    # An ordinary [] is a useful canonical spelling for an empty index list;
    # nonempty float arrays (including encoded integer-valued f64) are rejected.
    if empty_sequence and isinstance(value, (list, tuple)) and len(value) == 0:
        array = np.asarray(value, dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers, not floats or booleans")
    if array.size and (int(array.min()) < np.iinfo(np.int64).min or int(array.max()) > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must be representable as signed 64-bit integers")
    return array


def _root(parent, vertex):
    while parent[vertex] != vertex:
        parent[vertex] = parent[parent[vertex]]
        vertex = int(parent[vertex])
    return vertex


def _join(parent, sizes, left, right):
    left, right = _root(parent, left), _root(parent, right)
    if left != right:
        if sizes[left] < sizes[right]:
            left, right = right, left
        parent[right] = left
        sizes[left] += sizes[right]


def _complete_torus(bases, lifts, nvertices):
    """Build quotient incidence once, validating all faces before selection."""
    nfaces = len(bases)
    edge_ids = np.empty((nfaces, 3), dtype=np.int64)
    edge_signs = np.empty((nfaces, 3), dtype=np.int8)
    edge_lookup, face_keys = {}, set()
    edge_endpoints, edge_occurrences, first_orientations = [], [], []
    used = np.zeros(nvertices, dtype=bool)
    parent, sizes = np.arange(nvertices), np.ones(nvertices, dtype=np.int64)
    for face in range(nfaces):
        ids = tuple(int(v) for v in bases[face])
        shifts = tuple(tuple(int(v) for v in row) for row in lifts[face])
        # Translation and corner-order independent face identity. Arithmetic
        # uses Python integers, so subtracting two int64 lifts cannot overflow.
        key = min(tuple(sorted((ids[k], shifts[k][0] - origin[0], shifts[k][1] - origin[1])
                               for k in range(3))) for origin in shifts)
        if key in face_keys:
            raise ValueError("The complete mesh has duplicate periodic faces")
        face_keys.add(key)
        for corner in range(3):
            after = (corner + 1) % 3
            left, right = ids[corner], ids[after]
            dx = shifts[after][0] - shifts[corner][0]
            dy = shifts[after][1] - shifts[corner][1]
            forward, backward = (left, right, dx, dy), (right, left, -dx, -dy)
            if forward == backward:
                raise ValueError("A triangle has a collapsed periodic edge")
            edge, sign = (forward, 1) if forward < backward else (backward, -1)
            index = edge_lookup.get(edge)
            if index is None:
                index = len(edge_lookup)
                edge_lookup[edge] = index
                edge_endpoints.append((edge[0], edge[1]))
                edge_occurrences.append(1)
                first_orientations.append(sign)
            else:
                edge_occurrences[index] += 1
                if edge_occurrences[index] > 2 or sign == first_orientations[index]:
                    raise ValueError("Each periodic edge must have two oppositely oriented incident faces")
            edge_ids[face, corner], edge_signs[face, corner] = index, sign
            used[left] = True
            _join(parent, sizes, left, right)
    if not np.all(used):
        raise ValueError("The complete mesh omits base vertices")
    if any(count != 2 for count in edge_occurrences):
        raise ValueError("The complete mesh has an unpaired periodic edge")
    if len({_root(parent, vertex) for vertex in range(nvertices)}) != 1:
        raise ValueError("The complete mesh is disconnected")
    nedges = len(edge_endpoints)
    if nvertices - nedges + nfaces != 0:
        raise ValueError("The complete mesh must have torus Euler characteristic zero")

    # Each edge has two endpoint GERMS, even if its endpoint base IDs coincide.
    # A triangular corner joins two germs in its vertex's link. Paired edges
    # give degree two; connected links rule out pinched/nonmanifold vertices.
    links = np.arange(2 * nedges)
    link_sizes = np.ones(2 * nedges, dtype=np.int64)
    link_degrees = np.zeros(2 * nedges, dtype=np.int8)
    for face in range(nfaces):
        for corner in range(3):
            before = (corner - 1) % 3
            incoming = 2 * int(edge_ids[face, before]) + int(edge_signs[face, before] > 0)
            outgoing = 2 * int(edge_ids[face, corner]) + int(edge_signs[face, corner] < 0)
            link_degrees[incoming] += 1
            link_degrees[outgoing] += 1
            _join(links, link_sizes, incoming, outgoing)
    if not np.all(link_degrees == 2):
        raise ValueError("A complete-mesh vertex link is not a circle")
    vertex_link_root = np.full(nvertices, -1, dtype=np.int64)
    for edge, endpoints in enumerate(edge_endpoints):
        for endpoint, vertex in enumerate(endpoints):
            root = _root(links, 2 * edge + endpoint)
            if vertex_link_root[vertex] == -1:
                vertex_link_root[vertex] = root
            elif vertex_link_root[vertex] != root:
                raise ValueError("A complete-mesh vertex has a disconnected link")
    return edge_ids, nedges


def periodic_triangle_region_topology(triangle_base_indices, triangle_lift_offsets,
                                     selected_triangles, *, nvertices,
                                     geometry_certificate, max_triangles=1_000_000):
    """Return exact Betti numbers of a selected closed periodic triangle union.

    ``triangle_base_indices`` is an integer [triangle, corner] array with three
    ZERO-BASED vertex IDs per row. ``triangle_lift_offsets`` is an integer
    [triangle, corner, phi_psi] array: a corner represents its base vertex plus
    360 times that integer pair. Common per-face translations do not matter.
    Face orientations must be coherent. ``selected_triangles`` is a 1D vector
    of unique zero-based integer face IDs (possibly []), not a Boolean mask.
    Portable f64 exports must be checked and converted to integers by the
    caller; this function deliberately rejects silent float-to-int conversion.

    ``geometry_certificate`` must be a mapping with ``certified is True``
    from the checked geometry export. The caller owns file/hash verification.
    This is a provenance precondition, not a new numerical embedding proof.
    The ENTIRE mesh is validated, even for an empty selection: unique faces,
    opposite edge pairings, no missing vertices, connectedness, Euler zero,
    and circular connected vertex links. Together these certify an abstract
    connected oriented torus. The input budget is checked before allocating
    the dictionaries and link/edge arrays, all linear in the triangle count.

    For the selected union, V, E and F count identified quotient cells and
    beta_0 counts connected components including shared vertices. A proper
    face subcomplex of this torus has beta_2=0; the whole torus has beta_2=1.
    Euler then gives beta_1=beta_0+beta_2-(V-E+F), over F2. The result describes
    the continuous triangle union under the certified embedding assumption;
    it is independent of sampling resolution and does not apply to a distinct
    underlying smooth score field.
    """
    bases, lifts, nvertices = _validate_mesh_inputs(
        triangle_base_indices, triangle_lift_offsets, nvertices,
        geometry_certificate, max_triangles)
    selected = _selected_faces(selected_triangles, len(bases))
    edge_ids, nedges = _complete_torus(bases, lifts, nvertices)
    return _selected_topology(bases, edge_ids, nedges, selected, nvertices)


def periodic_triangle_regions_topology(triangle_base_indices, triangle_lift_offsets,
                                      selections, *, nvertices,
                                      geometry_certificate, max_triangles=1_000_000):
    """Measure several regions of one certified mesh, validating the mesh once.

    Each element of ``selections`` is a vector of distinct zero-based triangle
    IDs with the same contract as ``periodic_triangle_region_topology``. Results
    have the same order and contents as separate calls to that function. Empty
    selections are empty regions; an empty collection still validates the mesh.
    A malformed selection raises rather than returning a partially valid batch.

    The complete torus does not change when a density threshold or radius
    changes. Its quotient edges and vertex links are therefore checked once.
    Each region still receives its own connectivity and integer cell counts;
    no topology is inferred from a neighboring parameter value. Selections are
    consumed one at a time, avoiding an additional copy of the whole family.
    """
    bases, lifts, nvertices = _validate_mesh_inputs(
        triangle_base_indices, triangle_lift_offsets, nvertices,
        geometry_certificate, max_triangles)
    try:
        iterator = iter(selections)
    except TypeError as exc:
        raise ValueError("selections must be an iterable of triangle-index vectors") from exc
    edge_ids, nedges = _complete_torus(bases, lifts, nvertices)
    return [_selected_topology(bases, edge_ids, nedges,
                _selected_faces(selected, len(bases)), nvertices)
            for selected in iterator]


def _validate_mesh_inputs(triangle_base_indices, triangle_lift_offsets,
                          nvertices, geometry_certificate, max_triangles):
    """Enforce common geometry contracts before broad incidence allocation."""
    budget = _positive_integer(max_triangles, "max_triangles")
    nvertices = _positive_integer(nvertices, "nvertices")
    if not isinstance(geometry_certificate, Mapping) or geometry_certificate.get("certified") is not True:
        raise ValueError("An explicit certified-torus geometry provenance record is required")
    bases = _integer_array(triangle_base_indices, "triangle_base_indices")
    if bases.ndim != 2 or bases.shape[1] != 3 or not 0 < len(bases) <= budget:
        raise ValueError("Use nonempty [triangle, 3] base indices within max_triangles")
    nfaces = len(bases)
    if nvertices > 3 * nfaces:
        raise ValueError("nvertices exceeds the available triangle incidences")
    lifts = _integer_array(triangle_lift_offsets, "triangle_lift_offsets")
    if lifts.shape != (nfaces, 3, 2):
        raise ValueError("Periodic lifts must have shape [triangle, 3, 2]")
    if bases.min() < 0 or bases.max() >= nvertices:
        raise ValueError("Base vertex indices are outside [0, nvertices)")
    return bases, lifts, nvertices


def _selected_faces(selected_triangles, nfaces):
    """Validate each region without changing its order or its caller's arrays."""
    selected = _integer_array(selected_triangles, "selected_triangles", empty_sequence=True)
    if selected.ndim != 1 or len(selected) > nfaces:
        raise ValueError("Use a 1D vector of unique selected triangle indices")
    if len(selected) and (selected.min() < 0 or selected.max() >= nfaces):
        raise ValueError("Selected triangle indices are outside the complete mesh")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Selected triangle indices must be unique")
    # Conversion follows explicit signed-range checks; caller arrays are never
    # mutated. No broad edge/link materialization occurs before these checks.
    return selected.astype(np.int64, copy=False)


def _selected_topology(bases, edge_ids, nedges, selected, nvertices):
    """Count quotient cells and vertex-contact connectivity on one region."""
    nfaces = len(bases)
    empty, whole = len(selected) == 0, len(selected) == nfaces
    if empty:
        vertices = edges = components = 0
    elif whole:
        vertices, edges, components = nvertices, nedges, 1
    else:
        active = np.zeros(nvertices, dtype=bool)
        retained_edges = np.zeros(nedges, dtype=bool)
        parent, sizes = np.arange(nvertices), np.ones(nvertices, dtype=np.int64)
        for face in selected:
            a, b, c = (int(v) for v in bases[face])
            active[[a, b, c]] = True
            retained_edges[edge_ids[face]] = True
            _join(parent, sizes, a, b)
            _join(parent, sizes, a, c)
        vertices, edges = int(active.sum()), int(retained_edges.sum())
        components = len({_root(parent, int(vertex)) for vertex in np.flatnonzero(active)})
    faces = len(selected)
    euler = vertices - edges + faces
    beta2 = int(whole)
    beta1 = components + beta2 - euler
    if beta1 < 0:
        raise ArithmeticError("Selected-cell incidence and connectivity disagree")
    return dict(status="computed", betti=[components, beta1, beta2], components=components,
        euler_characteristic=euler, cell_counts=dict(vertices=vertices, edges=edges, faces=faces),
        empty=empty, whole_domain=whole,
        full_mesh=dict(vertices=nvertices, edges=nedges, triangles=nfaces,
            connected=True, euler_characteristic=0, opposite_edge_pairings=True,
            connected_circular_vertex_links=True, certified_geometry_provenance=True),
        interpretation="Exact integer topology over F2 of the closed periodic triangle union, conditional on the supplied certified embedded torus; not raster topology or topology of a different continuous score field")
