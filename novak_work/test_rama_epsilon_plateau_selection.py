"""Adversarial endpoint, geometry-identity and joint-stream selector fixtures."""
import copy
import unittest
from rama_epsilon_plateau_selection import joint_intervals,select_first


def state(left,right,signature,observed=64.,betti=(3,0,0)):
    return dict(left_inclusive_k=float(left),right_exclusive_k=None if right is None else float(right),
        right_censored=right is None,observed_interval_right_k=float(observed if right is None else right),
        mask_sha256=signature,raster_betti=list(betti),mask_cells=100)


class EpsilonPlateauSelectionTests(unittest.TestCase):
    def test_union_of_changes_preserves_stream_order(self):
        streams=[[state(1,5,'a'),state(5,None,'b')],
                 [state(1,3,'c'),state(3,9,'d'),state(9,None,'e')]]
        result=joint_intervals(streams)
        self.assertEqual([r['left_inclusive_k'] for r in result],[1.,3.,5.,9.])
        self.assertEqual([[s['mask_sha256'] for s in r['states']] for r in result],
                         [['a','c'],['a','d'],['b','d'],['b','e']])
        self.assertEqual([r['right_exclusive_k'] for r in result],[3.,5.,9.,None])

    def test_closed_span_must_end_strictly_before_actual_change(self):
        result=joint_intervals([[state(1,2,'a'),state(2,None,'b')]])
        self.assertEqual(select_first(result,2.,lambda r:True)['left_inclusive_k'],2.)
        self.assertIsNone(select_first(result,2.,lambda r:r['states'][0]['mask_sha256']=='a'))

    def test_observed_censored_endpoint_is_included(self):
        result=joint_intervals([[state(1,32,'a'),state(32,None,'b')]])
        chosen=select_first(result,2.,lambda r:r['states'][0]['mask_sha256']=='b')
        self.assertEqual(chosen['left_inclusive_k'],32.)
        self.assertTrue(chosen['right_censored'])

    def test_redundant_geometry_boundaries_merge_including_terminal_point(self):
        result=joint_intervals([[state(1,4,'a'),state(4,64,'a'),state(64,None,'a')]])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['left_inclusive_k'],1.)
        self.assertIsNone(result[0]['right_exclusive_k'])
        self.assertEqual(result[0]['observed_interval_right_k'],64.)
        self.assertIsNotNone(select_first(result,64.,lambda r:True))

    def test_same_betti_with_swapped_features_does_not_merge(self):
        result=joint_intervals([[state(1,4,'island_A'),state(4,None,'island_B')],
                                [state(1,4,'hole_A'),state(4,None,'hole_B')]])
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['right_exclusive_k'],4.)

    def test_zero_width_terminal_change_is_recorded_not_selected(self):
        result=joint_intervals([[state(1,64,'a'),state(64,None,'b')]])
        self.assertEqual(len(result),2)
        self.assertEqual(result[-1]['left_inclusive_k'],64.)
        self.assertEqual(result[-1]['observed_interval_right_k'],64.)
        self.assertIsNone(select_first(result,64.,lambda r:True))
        self.assertIsNone(select_first(result,2.,lambda r:r['states'][0]['mask_sha256']=='b'))

    def test_no_qualifying_and_empty_inputs(self):
        result=joint_intervals([[state(1,None,'a')]])
        self.assertIsNone(select_first(result,2.,lambda r:False))
        self.assertEqual(joint_intervals([]),[])
        self.assertIsNone(select_first([],2.,lambda r:True))

    def test_inputs_are_not_mutated(self):
        streams=[[state(1,3,'a'),state(3,None,'a')]];before=copy.deepcopy(streams)
        result=joint_intervals(streams);select_first(result,2.,lambda r:True)
        self.assertEqual(streams,before)

    def test_validation_of_gaps_domains_and_span(self):
        with self.assertRaises(ValueError):joint_intervals([[state(1,3,'a'),state(4,None,'b')]])
        with self.assertRaises(ValueError):joint_intervals([[state(1,None,'a')],[state(1,None,'b',32.)]])
        with self.assertRaises(ValueError):joint_intervals([[]])
        with self.assertRaises(ValueError):select_first([],1.,lambda r:True)
        with self.assertRaises(ValueError):joint_intervals([[state(1,1,'a'),state(1,None,'b')]])

if __name__=='__main__':unittest.main()
