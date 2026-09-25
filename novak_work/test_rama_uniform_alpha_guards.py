"""Geometric torus fixtures test location-aware guards, not just Betti counts."""
import unittest
import numpy as np
from rama_compact_triangle_topology import PeriodicTriangleTopology
from rama_uniform_alpha_guards import (edge_adjacent_face_pairs,
    background_component_labels,evaluate_topology_guards)
from test_rama_triangle_region_metrics import grid_torus,CERTIFICATE,square_selection


class UniformAlphaGuardTests(unittest.TestCase):
    def setUp(self):
        self.n=8
        b,l=grid_torus(self.n)
        self.topology=PeriodicTriangleTopology(b,l,nvertices=self.n*self.n,
            geometry_certificate=CERTIFICATE)
        self.pairs=edge_adjacent_face_pairs(self.topology._edge_ids)

    def faces(self,squares):
        selected=np.zeros(self.topology.ntriangles,bool)
        selected[square_selection(squares)]=True
        return selected

    def region(self,mask):
        return self.topology.region(np.flatnonzero(mask),component_labels=True)

    def evaluate(self,inner,baseline,candidate):
        i,b,c=map(self.faces,(inner,baseline,candidate))
        return evaluate_topology_guards(i,b,c,self.region(b)['triangle_component_labels'],
            self.region(c)['triangle_component_labels'],background_component_labels(c,self.pairs))

    def test_baseline_identity_and_preexisting_coreless_component(self):
        b=np.zeros((8,8),bool);b[1:3,1:3]=True;b[5:7,5:7]=True
        i=np.zeros_like(b);i[1,1]=True
        result=self.evaluate(i,b,b)
        self.assertTrue(result['passed'])
        self.assertEqual(result['candidate_component_has_inner_core'],[True,False])

    def test_connected_outer_is_allowed_to_contain_two_inner_cores(self):
        b=np.zeros((8,8),bool);b[1:7,1:7]=True
        i=np.zeros_like(b);i[2,2]=True;i[5,5]=True
        result=self.evaluate(i,b,b)
        self.assertTrue(result['passed'])
        self.assertEqual(result['candidate_foreground_components'],1)

    def test_split_into_two_core_bearing_pieces_is_allowed(self):
        b=np.zeros((8,8),bool);b[1:7,1:7]=True
        i=np.zeros_like(b);i[2,2]=True;i[5,5]=True
        c=b.copy();c[:,3:5]=False
        result=self.evaluate(i,b,c)
        self.assertTrue(result['passed'])
        self.assertEqual(result['split_baseline_component_ids'],[0])
        self.assertEqual(result['candidate_foreground_components'],2)

    def test_split_coreless_piece_is_rejected(self):
        b=np.zeros((8,8),bool);b[1:7,1:7]=True
        i=np.zeros_like(b);i[2,2]=True
        c=b.copy();c[:,3:5]=False
        result=self.evaluate(i,b,c)
        self.assertFalse(result['passed'])
        self.assertTrue(result['no_new_background_components'])
        self.assertEqual(len(result['coreless_split_component_ids']),1)

    def test_new_hole_is_rejected(self):
        b=np.zeros((8,8),bool);b[1:7,1:7]=True
        i=np.zeros_like(b);i[2,2]=True
        c=b.copy();c[4,4]=False
        result=self.evaluate(i,b,c)
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['new_background_component_ids']),1)

    def test_new_hole_detected_when_an_old_hole_disappears(self):
        b=np.zeros((8,8),bool);b[1:7,1:7]=True;b[2,2]=False
        i=np.zeros_like(b);i[5,5]=True
        c=b.copy();c[1,2]=False;c[4,4]=False
        self.assertEqual(self.region(self.faces(b))['betti'][1],self.region(self.faces(c))['betti'][1])
        self.assertFalse(self.evaluate(i,b,c)['no_new_background_components'])

    def test_diagonal_background_contact_does_not_merge_holes(self):
        b=np.ones((8,8),bool);b[1,1]=False
        i=np.zeros_like(b);i[5,5]=True
        c=b.copy();c[2,2]=False
        result=self.evaluate(i,b,c)
        self.assertEqual(result['candidate_background_components'],2)
        self.assertFalse(result['no_new_background_components'])

    def test_periodic_background_connection_uses_seams(self):
        b=np.ones((8,8),bool);b[0,2]=False
        i=np.zeros_like(b);i[5,5]=True
        c=b.copy();c[7,2]=False
        result=self.evaluate(i,b,c)
        self.assertTrue(result['passed'])
        self.assertEqual(result['candidate_background_components'],1)

    def test_missing_core_region_is_rejected(self):
        b=np.zeros((8,8),bool);b[1:7,1:7]=True
        i=np.zeros_like(b);i[1,2]=True
        c=b.copy();c[1,2]=False
        result=self.evaluate(i,b,c)
        self.assertFalse(result['full_inner_region_preserved'])
        self.assertEqual(result['missing_inner_triangles'],2)

    def test_empty_and_whole_background(self):
        for fill,expected in [(True,0),(False,1)]:
            selected=np.full(self.topology.ntriangles,fill,bool)
            labels=background_component_labels(selected,self.pairs)
            self.assertEqual(len(np.unique(labels[labels>=0])),expected)
            np.testing.assert_array_equal(labels>=0,~selected)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):edge_adjacent_face_pairs(np.array([[0,0,0]]))
        with self.assertRaises(ValueError):background_component_labels(np.zeros(4),self.pairs)
        with self.assertRaises(ValueError):background_component_labels(np.zeros(4,bool),self.pairs)

if __name__=='__main__':unittest.main()
