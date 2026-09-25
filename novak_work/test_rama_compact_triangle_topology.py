"""Independent chain-matrix, contact, malformed-mesh and compact parity oracles."""
import unittest

import numpy as np

from rama_compact_triangle_topology import PeriodicTriangleTopology, _components
from rama_triangle_region_metrics import periodic_triangle_regions_topology
from test_rama_triangle_region_metrics import (
    CERTIFICATE, boundary_matrix_betti, grid_torus, square_selection,
)


def face_components(bases, selected):
    """Tiny independent set traversal: closed faces connect by ANY base vertex."""
    unseen = set(map(int, selected))
    result = []
    while unseen:
        todo = [unseen.pop()]
        component = set(todo)
        while todo:
            here = set(map(int, bases[todo.pop()]))
            neighbors = {i for i in unseen if here.intersection(map(int, bases[i]))}
            unseen.difference_update(neighbors)
            component.update(neighbors)
            todo.extend(neighbors)
        result.append(component)
    return {frozenset(row) for row in result}


class CompactTopologyOracles(unittest.TestCase):
    def compile(self, n):
        bases, lifts = grid_torus(n)
        return bases, lifts, PeriodicTriangleTopology(bases, lifts, nvertices=n*n,
                                                       geometry_certificate=CERTIFICATE)

    def test_all_tiny_subsets_against_chain_matrices_and_old_owner(self):
        for n in (1, 2):
            bases, lifts, compiled = self.compile(n)
            selected = [np.flatnonzero([(bits >> i) & 1 for i in range(len(bases))])
                        for bits in range(1 << len(bases))]
            previous = periodic_triangle_regions_topology(bases, lifts, selected,
                nvertices=n*n, geometry_certificate=CERTIFICATE)
            for ids, old in zip(selected, previous):
                self.assertEqual(compiled.region(ids), old)
                self.assertEqual(compiled.region(ids)["betti"], boundary_matrix_betti(bases, lifts, ids))

    def test_hand_shapes_and_component_labels(self):
        bases, _, compiled = self.compile(8)
        cases = []
        disk = np.zeros((8, 8), dtype=bool); disk[2:6, 2:6] = True
        cases.append((disk.copy(), [1, 0, 0]))
        disk[3:5, 3:5] = False; cases.append((disk.copy(), [1, 1, 0]))
        strip = np.zeros_like(disk); strip[1] = strip[5] = True
        cases.append((strip, [2, 2, 0]))
        for positions, expected in [([(0, 0), (7, 7)], [1, 0, 0]),
                                    ([(0, 0), (4, 4)], [2, 0, 0])]:
            mask = np.zeros_like(disk)
            for position in positions: mask[position] = True
            cases.append((mask, expected))
        cases += [(np.zeros_like(disk), [0, 0, 0]), (np.ones_like(disk), [1, 2, 1])]
        for mask, expected in cases:
            selected = square_selection(mask)
            result = compiled.region(selected, component_labels=True)
            self.assertEqual(result["betti"], expected)
            labels = result["triangle_component_labels"]
            self.assertEqual(labels.dtype, np.dtype("int32"))
            np.testing.assert_array_equal(np.flatnonzero(labels >= 0), selected)
            partition = {frozenset(np.flatnonzero(labels == label)) for label in range(expected[0])}
            self.assertEqual(partition, face_components(bases, selected))

    def test_random_regions_translations_and_orientation_parity(self):
        rng = np.random.default_rng(260923)
        for n in (3, 7, 24):
            bases, lifts, compiled = self.compile(n)
            selections = [np.flatnonzero(rng.random(len(bases)) < p)
                          for p in (.02, .2, .5, .95) for _ in range(3)]
            prior = periodic_triangle_regions_topology(bases, lifts, selections,
                nvertices=n*n, geometry_certificate=CERTIFICATE)
            shifts = rng.integers(-10000, 10000, size=(len(bases), 1, 2))
            shifted = PeriodicTriangleTopology(bases[:, ::-1], (lifts + shifts)[:, ::-1],
                nvertices=n*n, geometry_certificate=CERTIFICATE)
            for ids, old in zip(selections, prior):
                self.assertEqual(compiled.region(ids), old)
                self.assertEqual(shifted.region(ids), old)
                self.assertEqual(compiled.region(ids)["betti"], boundary_matrix_betti(bases, lifts, ids))
                result = compiled.region(ids, component_labels=True)
                labels = result["triangle_component_labels"]
                self.assertEqual({frozenset(np.flatnonzero(labels == i)) for i in range(old["components"])},
                                 face_components(bases, ids))

    def test_common_extreme_translation_and_input_mutation(self):
        bases, lifts = grid_torus(3)
        baseline = PeriodicTriangleTopology(bases, lifts, nvertices=9, geometry_certificate=CERTIFICATE)
        shifted = PeriodicTriangleTopology(bases, lifts + np.iinfo(np.int64).max - 2,
            nvertices=9, geometry_certificate=CERTIFICATE)
        expected = baseline.region([1, 4], component_labels=True)
        self.assertEqual(baseline.region([1, 4]), shifted.region([1, 4]))
        bases[:] = 0; lifts[:] = 9999
        np.testing.assert_array_equal(baseline.region([1, 4], component_labels=True)["triangle_component_labels"],
                                      expected["triangle_component_labels"])

    def test_reject_bad_incidence_and_pinched_links(self):
        bases, lifts = grid_torus(3)
        def reject(b, l, nv=9):
            with self.assertRaises(ValueError):
                PeriodicTriangleTopology(b, l, nvertices=nv, geometry_certificate=CERTIFICATE)
        reject(bases[:-1], lifts[:-1])
        reject(np.r_[bases, bases[:1]], np.r_[lifts, lifts[:1]])
        changed_b, changed_l = bases.copy(), lifts.copy()
        changed_b[0], changed_l[0] = bases[0, ::-1], lifts[0, ::-1]
        reject(changed_b, changed_l)
        changed_b[1], changed_l[1] = bases[0], lifts[0] + 4
        reject(changed_b, changed_l)
        reject(bases, lifts, 10)
        b, l = grid_torus(1)
        reject(np.r_[b, b + 1], np.r_[l, l], 2)
        l[0, 1] = l[0, 0]; reject(b, l, 1)
        sphere = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        reject(sphere, np.zeros((4, 3, 2), dtype=int), 4)
        ids = np.array([0, 4, 9, 10]); shifts = np.array([[0, 0], [10, 0], [0, 0], [0, 0]])
        with self.assertRaisesRegex(ValueError, "disconnected link"):
            PeriodicTriangleTopology(np.r_[bases, ids[sphere]], np.r_[lifts, shifts[sphere]],
                nvertices=11, geometry_certificate=CERTIFICATE)

    def test_parallel_endpoint_adjacency_does_not_overflow(self):
        # Periodic edges with different lifts can share endpoint base IDs.
        # 256 repeats would wrap to zero in an 8-bit numerical adjacency sum.
        left = np.zeros(256, dtype=np.int32)
        right = np.ones(256, dtype=np.int32)
        count, labels = _components(left, right, 3)
        self.assertEqual(count, 2)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_contracts_and_integer_overflow_guards(self):
        bases, lifts, compiled = self.compile(2)
        for bad in ([0, 0], [-1], [len(bases)], [0.], np.array([True]), [[0]], np.array([], float)):
            with self.assertRaises(ValueError): compiled.region(bad)
        with self.assertRaises(ValueError): compiled.region([], component_labels=1)
        options = dict(nvertices=4, geometry_certificate=CERTIFICATE)
        for extra in ({"nvertices": True}, {"nvertices": 0}, {"max_triangles": 7},
                      {"max_triangles": True}, {"geometry_certificate": {"certified": 1}}):
            with self.assertRaises(ValueError): PeriodicTriangleTopology(bases, lifts, **(options | extra))
        for b, l in [(bases.astype(float), lifts), (bases, lifts.astype(float)),
                     (bases[:, :2], lifts), (bases, lifts[..., 0])]:
            with self.assertRaises(ValueError): PeriodicTriangleTopology(b, l, **options)
        huge = bases.astype(np.uint64); huge[0, 0] = 1 << 63
        with self.assertRaises(ValueError): PeriodicTriangleTopology(huge, lifts, **options)
        overflow = lifts.copy(); overflow[0, 0, 0] = np.iinfo(np.int64).min
        with self.assertRaises(ValueError): PeriodicTriangleTopology(bases, overflow, **options)


if __name__ == "__main__":
    unittest.main()
