"""Hand-built tests for the alpha-restricted display cleanup, not a theorem."""
import unittest

import numpy as np

from rama_alpha_window_cleanup import AlphaWindowCleanup
from rama_topology_cleanup_preview import CleanupPreview
from probe_dtfe_extended_epsilon import multiplicative_cleanup


class AlphaWindowCleanupTests(unittest.TestCase):
    @staticmethod
    def fixture():
        values = np.full((20, 20), .1)
        values[3:14, 3:14] = 4.
        values[7:10, 7:10] = .75
        values[5, 5] = .75
        values[17, 17] = 1.2
        eligible = np.ones(values.shape, bool)
        eligible[8, 8] = False
        return values, eligible

    def test_alpha_gap_protects_whole_enclosed_background(self):
        values, eligible = self.fixture()
        result, edits = AlphaWindowCleanup(values, 1., eligible).apply(2.)
        # This shallow cavity would normally be filled. One forbidden cell
        # protects the entire connected cavity, not just that individual cell.
        self.assertFalse(result[7:10, 7:10].any())
        self.assertTrue(result[5, 5])
        self.assertFalse(result[17, 17])
        self.assertEqual(edits['filled_background_component_count'], 1)
        self.assertEqual(edits['removed_component_count'], 1)
        self.assertEqual(edits['alpha_protected_background_component_count'], 1)
        self.assertFalse(np.any(result & ~eligible))
        unrestricted, _ = AlphaWindowCleanup(values, 1., np.ones_like(eligible)).apply(2.)
        self.assertTrue(unrestricted[7:10, 7:10].all())

    def test_high_density_alpha_cut_is_not_restored(self):
        values = np.full((12, 12), 8.)
        eligible = np.ones(values.shape, bool)
        eligible[4:8, 4:8] = False
        result, edits = AlphaWindowCleanup(values, 1., eligible).apply(4.)
        np.testing.assert_array_equal(result, eligible)
        self.assertEqual(edits['changed_cells'], 0)

    def test_zero_scores_outside_alpha_domain(self):
        values, eligible = self.fixture()
        values[~eligible] = 0.
        result, _ = AlphaWindowCleanup(values, 1., eligible).apply(2.)
        self.assertFalse(result[7:10, 7:10].any())
        self.assertFalse(np.any(result & ~eligible))
        empty, _ = AlphaWindowCleanup(np.zeros_like(values), 1., eligible).apply(2.)
        self.assertFalse(empty.any())

    def test_factor_one_identity_including_threshold_ties(self):
        values, eligible = self.fixture()
        values[2, 2] = 1.
        result, edits = AlphaWindowCleanup(values, 1., eligible).apply(1.)
        np.testing.assert_array_equal(result, (values >= 1.) & eligible)
        self.assertEqual(edits['changed_cells'], 0)

    def test_infinite_alpha_matches_previous_multiplicative_cleanup(self):
        values, _ = self.fixture()
        rng = np.random.default_rng(42)
        fields = [values, np.exp(rng.normal(0., 1.4, (35, 35))),
                  np.full((10, 10), .1), np.full((10, 10), 10.)]
        for density in fields:
            new_model = AlphaWindowCleanup(density, 1., np.ones(density.shape, bool))
            previous_model = CleanupPreview(density, 1.)
            for factor in (1., 2., 4., 16., 128.):
                with self.subTest(shape=density.shape, factor=factor):
                    result, edits = new_model.apply(factor)
                    old_result, old_edits = multiplicative_cleanup(previous_model, factor)
                    np.testing.assert_array_equal(result, old_result)
                    for key, value in old_edits.items():
                        if isinstance(value, np.ndarray):
                            np.testing.assert_array_equal(edits[key], value)
                        else:
                            self.assertEqual(edits[key], value)

    def test_periodic_translation_equivariance(self):
        values, eligible = self.fixture()
        result, edits = AlphaWindowCleanup(values, 1., eligible).apply(2.)
        for shift in ((7, 9), (0, 13), (18, 18)):
            roll = lambda a: np.roll(a, shift, (0, 1))
            translated, after = AlphaWindowCleanup(roll(values), 1., roll(eligible)).apply(2.)
            np.testing.assert_array_equal(translated, roll(result))
            for key, value in edits.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(after[key], roll(value))
                else:
                    self.assertEqual(after[key], value)

    def test_relabel_after_weak_ring_removal(self):
        values = np.full((12, 12), .1)
        values[4:7, 4:7] = 1.1
        values[5, 5] = .9
        result, _ = AlphaWindowCleanup(values, 1., np.ones_like(values, bool)).apply(2.)
        self.assertFalse(result.any())

    def test_winding_background_and_largest_ties_are_protected(self):
        values = np.full((16, 16), 4.)
        values[0:2, :] = .75
        values[6, 6] = .75
        result, _ = AlphaWindowCleanup(values, 1., np.ones_like(values, bool)).apply(2.)
        self.assertFalse(result[0:2, :].any())
        self.assertTrue(result[6, 6])
        values = np.full((12, 12), 4.)
        values[2:4, 2:4] = .75
        values[8:10, 8:10] = .75
        result, _ = AlphaWindowCleanup(values, 1., np.ones_like(values, bool)).apply(2.)
        np.testing.assert_array_equal(result, values >= 1.)

    def test_fixed_domain_nested_level_fixture(self):
        # A fixture check only: the API does not promise universal nesting.
        values = np.full((24, 24), .1)
        values[3:18, 3:18] = 4.
        values[6:15, 6:15] = 8.
        values[9:12, 9:12] = .8
        eligible = np.ones(values.shape, bool)
        eligible[10, 10] = False
        inner, _ = AlphaWindowCleanup(values, 5., eligible).apply(1.5)
        outer, _ = AlphaWindowCleanup(values, 2., eligible).apply(1.5)
        self.assertFalse(np.any(inner & ~outer))
        self.assertFalse(np.any(outer & ~eligible))

    def test_inputs_and_previous_results_are_not_mutated(self):
        values, eligible = self.fixture()
        old_values, old_eligible = values.copy(), eligible.copy()
        model = AlphaWindowCleanup(values, 1., eligible)
        first, first_edits = model.apply(2.)
        saved, saved_removed = first.copy(), first_edits['removed'].copy()
        model.apply(16.)
        np.testing.assert_array_equal(first, saved)
        np.testing.assert_array_equal(first_edits['removed'], saved_removed)
        np.testing.assert_array_equal(values, old_values)
        np.testing.assert_array_equal(eligible, old_eligible)
        values[:] = 100.
        eligible[:] = False
        again, _ = model.apply(2.)
        np.testing.assert_array_equal(again, first)

    def test_reject_invalid_inputs(self):
        values, eligible = self.fixture()
        for bad in (values * np.nan, -values):
            with self.assertRaises(ValueError):
                AlphaWindowCleanup(bad, 1., eligible)
        for bad in (eligible.astype(int), eligible[:-1], np.ones((0, 0), bool)):
            with self.assertRaises(ValueError):
                AlphaWindowCleanup(values, 1., bad)
        for bad in (0., -1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                AlphaWindowCleanup(values, bad, eligible)
        model = AlphaWindowCleanup(values, 1., eligible)
        for bad in (0., .99, np.inf, np.nan, -2.):
            with self.assertRaises(ValueError):
                model.apply(bad)
        with self.assertRaises(ValueError):
            AlphaWindowCleanup(values, 1e308, eligible).apply(100.)


if __name__ == '__main__':
    unittest.main()
