"""Additional location/contact checks, separate from the sealed 11-test source."""
import unittest
import numpy as np
import test_rama_uniform_alpha_guards as fixture
from rama_uniform_alpha_guards import background_component_labels


class AdversarialUniformAlphaGuardTests(unittest.TestCase):
    def setUp(self):
        self.helper=fixture.UniformAlphaGuardTests()
        self.helper.setUp()

    def test_closed_foreground_corner_contact_differs_from_open_adjacency(self):
        h=self.helper
        squares=np.zeros((8,8),bool);squares[1,1]=True;squares[2,2]=True
        selected=h.faces(squares)
        self.assertEqual(h.region(selected)['betti'][0],1)
        open_pieces=background_component_labels(~selected,h.pairs)
        self.assertEqual(len(np.unique(open_pieces[open_pieces>=0])),2)

    def test_new_hole_rejected_despite_equal_background_component_counts(self):
        h=self.helper
        baseline=np.zeros((8,8),bool);baseline[1:7,1:7]=True;baseline[2,2]=False
        inner=np.zeros_like(baseline);inner[5,5]=True
        candidate=baseline.copy();candidate[1,2]=False;candidate[4,4]=False
        original=background_component_labels(h.faces(baseline),h.pairs)
        current=background_component_labels(h.faces(candidate),h.pairs)
        self.assertEqual(len(np.unique(original[original>=0])),len(np.unique(current[current>=0])))
        result=h.evaluate(inner,baseline,candidate)
        self.assertFalse(result['no_new_background_components'])
        self.assertEqual(len(result['new_background_component_ids']),1)

if __name__=='__main__':unittest.main()
