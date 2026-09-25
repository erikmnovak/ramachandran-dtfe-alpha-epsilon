"""Hand, boundary-matrix, and cubical oracles for periodic triangle topology."""
import importlib.util
import unittest

import numpy as np

from rama_periodic_region_metrics import periodic_region_topology
from rama_triangle_region_metrics import periodic_triangle_region_topology, periodic_triangle_regions_topology


CERTIFICATE = {"certified": True, "source": "hand-constructed periodic torus oracle"}


def grid_torus(n):
    """Split each periodic square along its SW-to-NE diagonal, CCW faces."""
    bases, lifts = [], []
    for y in range(n):
        for x in range(n):
            for corners in (((x, y), (x + 1, y), (x + 1, y + 1)),
                            ((x, y), (x + 1, y + 1), (x, y + 1))):
                bases.append([(b % n) * n + a % n for a, b in corners])
                lifts.append([(a // n, b // n) for a, b in corners])
    return np.asarray(bases, dtype=np.int64), np.asarray(lifts, dtype=np.int64)


def square_selection(mask):
    return np.flatnonzero(np.repeat(mask.ravel(), 2))


def gf2_rank(columns):
    pivots = {}
    for column in columns:
        while column:
            pivot = column.bit_length() - 1
            if pivot in pivots:
                column ^= pivots[pivot]
            else:
                pivots[pivot] = column
                break
    return len(pivots)


def boundary_matrix_betti(bases, lifts, selected):
    """Independent F2 chain matrices, including torus loop/parallel edges."""
    vertices, edges, faces = set(), {}, []
    for index in selected:
        face = []
        for i in range(3):
            j = (i + 1) % 3
            a, b = int(bases[index, i]), int(bases[index, j])
            dx, dy = (int(lifts[index, j, k]) - int(lifts[index, i, k]) for k in range(2))
            key = min((a, b, dx, dy), (b, a, -dx, -dy))
            if key not in edges:
                edges[key] = len(edges)
            face.append(edges[key])
            vertices.update((a, b))
        faces.append(face)
    vertex_ids = {v: i for i, v in enumerate(sorted(vertices))}
    d1 = [(1 << vertex_ids[a]) ^ (1 << vertex_ids[b]) for a, b, _, _ in edges]
    d2 = [(1 << a) ^ (1 << b) ^ (1 << c) for a, b, c in faces]
    rank1, rank2 = gf2_rank(d1), gf2_rank(d2)
    return [len(vertices) - rank1, len(edges) - rank1 - rank2, len(faces) - rank2]


class PeriodicTriangleTopologyOracles(unittest.TestCase):
    def test_batch_matches_independent_boundary_matrix_oracle(self):
        rng = np.random.default_rng(9202607)
        for n in (1, 2, 7):
            bases, lifts = grid_torus(n)
            selections = [[], np.arange(len(bases))]
            selections += [np.flatnonzero(rng.random(len(bases)) < p)
                           for p in (.1, .5, .9) for _ in range(8)]
            results = periodic_triangle_regions_topology(bases, lifts, iter(selections),
                nvertices=n*n, geometry_certificate=CERTIFICATE)
            for selected, result in zip(selections, results):
                self.assertEqual(result["betti"], boundary_matrix_betti(bases, lifts, selected))
                self.assertEqual(result, periodic_triangle_region_topology(bases, lifts, selected,
                    nvertices=n*n, geometry_certificate=CERTIFICATE))

    def test_batch_hand_shapes_and_negative_contracts(self):
        n = 7
        bases, lifts = grid_torus(n)
        disk = np.zeros((n, n), bool); disk[2:5, 2:5] = True
        annulus = disk.copy(); annulus[3, 3] = False
        strip = np.zeros_like(disk); strip[3] = True
        regions = [[], square_selection(disk), square_selection(annulus),
                   square_selection(strip), np.arange(len(bases))]
        before = [np.asarray(s).copy() for s in regions]
        result = periodic_triangle_regions_topology(bases, lifts, regions,
            nvertices=n*n, geometry_certificate=CERTIFICATE)
        self.assertEqual([x["betti"] for x in result],
                         [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 2, 1]])
        for old, new in zip(before, regions):
            np.testing.assert_array_equal(old, new)
        self.assertEqual(periodic_triangle_regions_topology(bases, lifts, [],
            nvertices=n*n, geometry_certificate=CERTIFICATE), [])
        for bad in ([[], [0, 0]], [[-1]], [[0.]], None):
            with self.assertRaises(ValueError):
                periodic_triangle_regions_topology(bases, lifts, bad,
                    nvertices=n*n, geometry_certificate=CERTIFICATE)
        # Empty batch does not bypass the whole-mesh certificate or budget.
        for kwargs in ({"max_triangles": len(bases)-1}, {"geometry_certificate": {}}):
            options = dict(nvertices=n*n, geometry_certificate=CERTIFICATE)
            options.update(kwargs)
            with self.assertRaises(ValueError):
                periodic_triangle_regions_topology(bases, lifts, [], **options)
        with self.assertRaises(ValueError):
            periodic_triangle_regions_topology(bases[:-1], lifts[:-1], [],
                nvertices=n*n, geometry_certificate=CERTIFICATE)

    def result(self, n, selected, **kwargs):
        bases, lifts = grid_torus(n)
        return periodic_triangle_region_topology(bases, lifts, selected, nvertices=n*n,
                                                 geometry_certificate=CERTIFICATE, **kwargs)

    def check_mask(self, mask, expected):
        result = self.result(len(mask), square_selection(mask))
        self.assertEqual(result["betti"], expected)
        raster = periodic_region_topology(mask)
        self.assertEqual(result["betti"], raster["betti"])
        self.assertEqual(result["euler_characteristic"], raster["euler_characteristic"])
        # Splitting each selected square introduces one interior edge and one
        # additional face; vertex count and Euler characteristic are unchanged.
        self.assertEqual(result["cell_counts"]["vertices"], raster["cell_counts"]["vertices"])
        self.assertEqual(result["cell_counts"]["edges"], raster["cell_counts"]["edges"] + int(mask.sum()))
        self.assertEqual(result["cell_counts"]["faces"], 2 * int(mask.sum()))
        return result

    def test_empty_full_and_one_vertex_two_triangle_torus(self):
        for n in (1, 2, 7):
            self.check_mask(np.zeros((n, n), bool), [0, 0, 0])
            result = self.check_mask(np.ones((n, n), bool), [1, 2, 1])
            self.assertEqual(result["full_mesh"]["edges"], 3*n*n)
            self.assertTrue(result["full_mesh"]["connected_circular_vertex_links"])
        # All three edges loop at the same vertex, but have different winding
        # pairs. Selecting one face leaves a once-punctured torus, not a disk.
        result = self.result(1, [0])
        self.assertEqual(result["betti"], [1, 2, 0])
        self.assertEqual(result["cell_counts"], dict(vertices=1, edges=3, faces=1))
        self.assertEqual(self.result(1, []) ["betti"], [0, 0, 0])

    def test_seam_crossing_disk_annulus_and_punctured_torus(self):
        mask = np.zeros((7, 7), bool)
        mask[np.ix_([6, 0], [6, 0])] = True
        self.check_mask(mask, [1, 0, 0])
        mask[:] = False
        mask[1:6, 1:6] = True
        mask[2:5, 2:5] = False
        self.check_mask(mask, [1, 1, 0])
        mask[:] = True
        mask[3, 3] = False
        self.check_mask(mask, [1, 2, 0])

    def test_winding_strips_closed_corner_contacts_and_disjoint_disks(self):
        mask = np.zeros((8, 8), bool)
        mask[1, :] = True
        self.check_mask(mask, [1, 1, 0])
        mask[5, :] = True
        self.check_mask(mask, [2, 2, 0])
        for positions in (((2, 2), (3, 3)), ((0, 0), (7, 7)), ((0, 2), (7, 3))):
            mask[:] = False
            for position in positions:
                mask[position] = True
            self.check_mask(mask, [1, 0, 0])
        mask[:] = False
        mask[0, 0] = mask[3, 3] = True
        self.check_mask(mask, [2, 0, 0])
        self.assertEqual(self.result(4, [0])["betti"], [1, 0, 0])

    def test_exact_chain_matrix_oracle_on_all_small_and_random_face_subsets(self):
        rng = np.random.default_rng(9202601)
        for n in (1, 2, 3, 5):
            bases, lifts = grid_torus(n)
            count = len(bases)
            if n < 3:
                selections = [np.flatnonzero([(bits >> i) & 1 for i in range(count)])
                              for bits in range(1 << count)]
            else:
                selections = [np.flatnonzero(rng.random(count) < p)
                              for p in (.15, .5, .85) for _ in range(10)]
            for selected in selections:
                result = periodic_triangle_region_topology(bases, lifts, selected, nvertices=n*n,
                                                           geometry_certificate=CERTIFICATE)
                self.assertEqual(result["betti"], boundary_matrix_betti(bases, lifts, selected))

    def test_translation_corner_order_and_face_permutation_invariance(self):
        rng = np.random.default_rng(9202602)
        n = 5
        bases, lifts = grid_torus(n)
        mask = rng.random(len(bases)) < .5
        selected = np.flatnonzero(mask)
        baseline = self.result(n, selected)
        translated = lifts + rng.integers(-100, 101, size=(len(bases), 1, 2))
        for order in ((0, 1, 2), (1, 2, 0), (2, 1, 0)):
            permutation = rng.permutation(len(bases))
            result = periodic_triangle_region_topology(bases[permutation][:, order],
                translated[permutation][:, order], np.flatnonzero(mask[permutation]),
                nvertices=n*n, geometry_certificate=CERTIFICATE)
            self.assertEqual(result, baseline)
        # Common translations close to int64 limits require exact differences,
        # not overflowing subtraction on the input dtype.
        translated = lifts + (np.iinfo(np.int64).max - 2)
        self.assertEqual(periodic_triangle_region_topology(bases, translated, selected, nvertices=n*n,
                         geometry_certificate=CERTIFICATE), baseline)

    @unittest.skipUnless(importlib.util.find_spec("gudhi"), "Optional independent GUDHI oracle unavailable")
    def test_independent_periodic_cubical_oracle(self):
        import gudhi
        rng = np.random.default_rng(9202603)
        masks = [np.array([(bits >> i) & 1 for i in range(4)], bool).reshape(2, 2) for bits in range(16)]
        masks += [rng.random((n, n)) < p for n in (3, 5) for p in (.2, .5, .8) for _ in range(5)]
        for mask in masks:
            complex_ = gudhi.PeriodicCubicalComplex(top_dimensional_cells=np.where(mask, 0., 1.),
                                                   periodic_dimensions=[True, True])
            complex_.compute_persistence(homology_coeff_field=2)
            self.check_mask(mask, complex_.persistent_betti_numbers(0., 0.))

    def test_reject_missing_duplicate_reversed_and_malformed_faces(self):
        bases, lifts = grid_torus(3)
        def reject(b, l, count=9, expected=None):
            with self.assertRaisesRegex(ValueError, expected or "."):
                periodic_triangle_region_topology(b, l, [], nvertices=count, geometry_certificate=CERTIFICATE)
        reject(bases[:-1], lifts[:-1], expected="unpaired")
        reject(np.r_[bases, bases[:1]], np.r_[lifts, lifts[:1]], expected="duplicate")
        duplicate_b, duplicate_l = bases.copy(), lifts.copy()
        duplicate_b[1], duplicate_l[1] = bases[0, ::-1], lifts[0, ::-1]
        reject(duplicate_b, duplicate_l, expected="duplicate")
        reversed_b, reversed_l = bases.copy(), lifts.copy()
        reversed_b[0], reversed_l[0] = bases[0, ::-1], lifts[0, ::-1]
        reject(reversed_b, reversed_l, expected="oppositely oriented")
        wrong_lifts = lifts.copy(); wrong_lifts[0, 0, 0] += 1
        reject(bases, wrong_lifts)
        reject(bases, lifts, count=10, expected="omits")
        b, l = grid_torus(1)
        reject(np.r_[b, b + 1], np.r_[l, l], count=2, expected="disconnected")
        collapsed = l.copy(); collapsed[0, 1] = collapsed[0, 0]
        reject(b, collapsed, count=1, expected="collapsed")
        # A sphere is closed, oriented and connected but is not a torus.
        tetra = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        reject(tetra, np.zeros((4, 3, 2), int), count=4, expected="Euler")

    def test_reject_pinched_links_even_when_euler_and_pairings_pass(self):
        bases, lifts = grid_torus(3)
        # Attach a sphere to a torus at two distinct vertices. The result is
        # connected, oriented and has Euler zero, but those two links split.
        tetra = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        ids = np.array([0, 4, 9, 10])
        shifts = np.array([[0, 0], [10, 0], [0, 0], [0, 0]])
        with self.assertRaisesRegex(ValueError, "disconnected link"):
            periodic_triangle_region_topology(np.r_[bases, ids[tetra]], np.r_[lifts, shifts[tetra]], [],
                nvertices=11, geometry_certificate=CERTIFICATE)

    def test_strict_indices_provenance_shapes_and_early_budget(self):
        bases, lifts = grid_torus(2)
        def call(b=bases, l=lifts, s=(), **kwargs):
            options = dict(nvertices=4, geometry_certificate=CERTIFICATE)
            options.update(kwargs)
            return periodic_triangle_region_topology(b, l, s, **options)
        for invalid in ([0, 0], [-1], [8], [0.], np.array([True]), [[0]], np.array([], dtype=float)):
            with self.assertRaises(ValueError): call(s=invalid)
        for invalid in ({}, {"certified": False}, {"certified": 1}, None):
            with self.assertRaises(ValueError): call(geometry_certificate=invalid)
        for invalid in (0, -1, 4., True, 25):
            with self.assertRaises(ValueError): call(nvertices=invalid)
        for invalid in (0, -1, True, 8., 7):
            with self.assertRaises(ValueError): call(max_triangles=invalid)
        self.assertEqual(call(max_triangles=8)["betti"], [0, 0, 0])
        for invalid in (bases.astype(float), bases.astype(bool), bases[:, :2], bases[0], bases[:0]):
            with self.assertRaises(ValueError): call(b=invalid)
        for invalid in (lifts.astype(float), lifts[..., 0], lifts[:, :2]):
            with self.assertRaises(ValueError): call(l=invalid)
        huge = bases.astype(np.uint64); huge[0, 0] = np.uint64(1 << 63)
        with self.assertRaises(ValueError): call(b=huge)
        large_lifts = lifts.astype(np.uint64); large_lifts[0, 0, 0] = np.uint64(1 << 63)
        with self.assertRaises(ValueError): call(l=large_lifts)
        before_b, before_l = bases.copy(), lifts.copy()
        call(s=[3, 1])
        np.testing.assert_array_equal(bases, before_b)
        np.testing.assert_array_equal(lifts, before_l)


if __name__ == "__main__":
    unittest.main()
