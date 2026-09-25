"""Parity, event endpoints and complete-interval tests for cached cleanup."""
import unittest
import numpy as np
from rama_alpha_window_cleanup import AlphaWindowCleanup
from rama_alpha_cleanup_events import CachedAlphaCleanup,_first_float_onset


class CleanupEventTests(unittest.TestCase):
    def compare(self,model,cached,k):
        a,ea=model.apply(k);b,eb=cached.apply(k)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(set(ea),set(eb))
        for key in ea:
            if isinstance(ea[key],np.ndarray):np.testing.assert_array_equal(ea[key],eb[key])
            else:self.assertEqual(ea[key],eb[key])
        return b

    def model(self,values,level=1.,eligible=None):
        if eligible is None:eligible=np.ones_like(values,bool)
        m=AlphaWindowCleanup(values,level,eligible)
        return m,CachedAlphaCleanup(m)

    def check_event_predecessors(self,cached):
        L=cached.model.level
        for d in cached.event_details:
            value=d['component_value'];x=d['onset_ratio'];prev=d['previous_ratio']
            self.assertEqual(prev,np.nextafter(x,-np.inf))
            if d['kind']=='foreground_removal':predicate=lambda k:value<L*k
            else:predicate=lambda k:value>=L/k
            self.assertFalse(predicate(prev));self.assertTrue(predicate(x))

    def test_foreground_strict_endpoint_rounding(self):
        values=np.zeros((8,8));values[3:5,3:5]=.3
        m,c=self.model(values,.1);events=c.events((1.,8.))
        self.check_event_predecessors(c)
        event=next(d for d in c.event_details if d['kind']=='foreground_removal')
        self.assertEqual(event['onset_ratio'],3.)
        self.assertTrue(self.compare(m,c,event['previous_ratio']).any())
        self.assertFalse(self.compare(m,c,event['onset_ratio']).any())

    def test_hole_inclusive_endpoint_rounding(self):
        values=np.zeros((8,8));values[2:6,2:6]=4.;values[3:5,3:5]=.1
        m,c=self.model(values,.3);c.events((1.,8.));self.check_event_predecessors(c)
        event=next(d for d in c.event_details if d['kind']=='background_fill')
        self.assertFalse(self.compare(m,c,event['previous_ratio'])[3,3])
        self.assertTrue(self.compare(m,c,event['onset_ratio'])[3,3])

    def test_weak_ring_removal_merges_and_protects_background(self):
        values=np.zeros((12,12));values[2:9,2:9]=2.;values[3:8,3:8]=.4
        values[10:12,10:12]=50.
        m,c=self.model(values)
        for k in (1.,1.99,2.,np.nextafter(2.,np.inf),2.5,16.,1.):self.compare(m,c,k)
        self.assertFalse(c.apply(2.5)[0][5,5])

    def test_alpha_gap_protection_and_periodic_seam(self):
        values=np.full((10,10),20.);values[:,0]=0.;values[4:6,4:6]=.3
        eligible=np.ones_like(values,bool);eligible[:,0]=False
        m,c=self.model(values,eligible=eligible)
        for k in (1.,2.,4.,16.,32.):self.compare(m,c,k)
        self.assertFalse(c.apply(16.)[0][:,0].any())

    def test_simultaneous_foreground_and_background_changes(self):
        values=np.zeros((12,12));values[2:10,2:10]=20.;values[3:9,3:9]=.5
        values[5:7,5:7]=2.
        m,c=self.model(values);events=c.events((1.,32.));self.check_event_predecessors(c)
        for event in events:
            self.compare(m,c,event)
            if event>1:self.compare(m,c,np.nextafter(event,-np.inf))
        self.compare(m,c,2.);self.compare(m,c,np.nextafter(2.,np.inf))

    def test_all_open_event_intervals_have_constant_masks(self):
        rng=np.random.default_rng(20260925)
        for trial in range(6):
            values=rng.choice([0.,.125,.3,.5,1.,1.25,2.,3.,5.,8.,17.],size=(8,8))
            eligible=rng.random((8,8))>.1;values[~eligible]=0.;values[0,0]=0.
            m,c=self.model(values);events=c.events((1.,32.));self.check_event_predecessors(c)
            for left,right in zip(events[:-1],events[1:]):
                a=self.compare(m,c,left)
                midpoint=float(np.sqrt(left*right));last=float(np.nextafter(right,-np.inf))
                if midpoint<right:np.testing.assert_array_equal(a,self.compare(m,c,midpoint))
                if last>=left:np.testing.assert_array_equal(a,self.compare(m,c,last))
            self.compare(m,c,events[-1])

    def test_current_partition_only_and_out_of_order_reuse(self):
        values=np.zeros((8,8));values[2:6,2:6]=20.;values[3:5,3:5]=.25
        m,c=self.model(values)
        for k in (1.,2.,4.,8.,16.):self.compare(m,c,k)
        self.assertEqual(c.statistics['partition_rebuilds'],0)
        self.assertEqual(c.statistics['partition_reuses'],5)
        self.compare(m,c,32.);self.assertEqual(c.statistics['partition_rebuilds'],1)
        self.compare(m,c,4.);self.assertEqual(c.statistics['partition_rebuilds'],2)

    def test_snapshot_does_not_mutate_or_follow_original_model(self):
        values=np.zeros((6,6));values[2:4,2:4]=3.
        m,c=self.model(values);expected=c.apply(2.)[0]
        m.density[:]=0.;m.raw[:]=False
        np.testing.assert_array_equal(c.apply(2.)[0],expected)

    def test_empty_start_and_domain_validation(self):
        m,c=self.model(np.zeros((5,5)))
        self.assertEqual(c.events(),[1.,64.]);self.assertFalse(self.compare(m,c,16.).any())
        for domain in ((0.,4.),(4.,4.),(1.,np.inf)):
            with self.assertRaises(ValueError):c.events(domain)

    def test_explicit_assumption_and_rounding_budget_failures(self):
        with self.assertRaises(ValueError):CachedAlphaCleanup(AlphaWindowCleanup(np.ones((4,4)),1.,np.ones((4,4),bool)))
        v=np.ones((4,4));e=np.ones((4,4),bool);e[0,0]=False
        with self.assertRaises(ValueError):CachedAlphaCleanup(AlphaWindowCleanup(v,1.,e))
        with self.assertRaises(ArithmeticError):_first_float_onset(1.,lambda x:x>2.,1.,4.)

if __name__=='__main__':unittest.main()
