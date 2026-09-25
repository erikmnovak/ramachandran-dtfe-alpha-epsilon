"""Event-based, cached evaluation of the existing alpha-window cleanup.

This is an acceleration and parameter bookkeeping layer, not a new cleanup.
The original AlphaWindowCleanup remains the mask/edits oracle. Requirements for
complete event enumeration here are the current experiment's verified facts:
scores are zero outside alpha eligibility, and the starting region is not the
whole raster torus. An empty starting region is permitted.

A foreground component disappears when peak < k*L (strict). Between such
removals the background partition is fixed; a fill predicate changes when
pit >= L/k (inclusive). Every background piece after removals contains an
original background piece, and newly removed cells had scores >= L. Therefore
its minimum is one of the ORIGINAL background minima. Initial peaks and pits
supply a complete (possibly redundant) list of event values.

We locate each predicate's FIRST Float64 transition by checking adjacent
representable numbers, not by assuming a rounded quotient has correct endpoint
semantics. Pathological inputs requiring more than 32 adjustments fail loudly.
"""
from __future__ import annotations
import math
import numpy as np
from rama_alpha_window_cleanup import AlphaWindowCleanup
from rama_topology_cleanup_preview import periodic_labels


def _first_float_onset(estimate,predicate,left,right,maximum_adjustments=32):
    """First representable k in (left,right] where a monotone predicate is true.

    Already-true predicates at the domain's left endpoint need no interior
    event. Bounds are always listed separately by the public events method.
    """
    if predicate(left) or not predicate(right):
        return None
    current=float(min(max(float(estimate),left),right))
    adjustments=0
    if predicate(current):
        while True:
            previous=float(np.nextafter(current,-np.inf))
            if previous<left or not predicate(previous):break
            current=previous;adjustments+=1
            if adjustments>maximum_adjustments:
                raise ArithmeticError('Float64 event onset needs more than 32 adjacent adjustments')
    else:
        while not predicate(current):
            current=float(np.nextafter(current,np.inf));adjustments+=1
            if current>right or adjustments>maximum_adjustments:
                raise ArithmeticError('Float64 event onset was not bracketed within 32 adjustments')
    previous=float(np.nextafter(current,-np.inf))
    if not (left<current<=right and predicate(current) and not predicate(previous)):
        raise ArithmeticError('Float64 event predecessor/onset verification failed')
    return current,previous,adjustments


class CachedAlphaCleanup:
    """Snapshot a fitted cleanup; reuse only its CURRENT background partition.

    ``events((1.,64.))`` returns sorted float boundaries including both domain
    endpoints. At each event value the new comparison state already applies.
    ``event_details`` records every predicate onset and its predecessor check.
    Multiple events can leave the final mask identical; the caller should merge
    consecutive equal masks when constructing maximal output plateaus.

    ``apply(k)`` returns the same mask and edits dictionary as the original
    implementation, including calls made out of order. No old partitions are
    accumulated. ``statistics`` exposes small computation counters only.
    """
    def __init__(self,model):
        if not isinstance(model,AlphaWindowCleanup):
            raise TypeError('Expected an AlphaWindowCleanup instance')
        if np.any(model.density[~model.eligible]!=0):
            raise ValueError('Event enumeration requires zero score outside alpha eligibility')
        if model.raw.all():
            raise ValueError('Event enumeration requires a starting region smaller than the whole torus')
        # Own a snapshot, so subsequent changes to the caller's model cannot
        # silently invalidate cached geometry or the declared event list.
        self.model=AlphaWindowCleanup(model.density,model.level,model.eligible)
        self.event_details=[]
        self.statistics=dict(initial_partition_builds=0,partition_rebuilds=0,
                             partition_reuses=0,apply_calls=0)
        self._remove_labels=np.zeros_like(self.model.peaks,dtype=bool)
        self._set_partition(self._remove_labels,initial=True)
        self._initial_pits=self._pits.copy()

    def _set_partition(self,remove_labels,initial=False):
        m=self.model
        self._remove_labels=remove_labels.copy()
        self._intermediate=m.raw & ~remove_labels[m.foreground]
        background,winding=periodic_labels(~self._intermediate,4)
        count=int(background.max())+1
        pits=np.full(count,np.inf)
        np.minimum.at(pits,background.ravel(),m.density.ravel())
        sizes=np.bincount(background.ravel(),minlength=count);sizes[0]=0
        alpha_protected=np.zeros(count,bool)
        alpha_protected[np.unique(background[~m.eligible])]=True
        alpha_protected[0]=False
        protected=winding.any(axis=1)|alpha_protected;protected[0]=True
        if sizes.max()>0:protected|=(sizes==sizes.max())&(sizes>0)
        self._background=background;self._pits=pits;self._protected=protected
        self._alpha_protected=alpha_protected
        key='initial_partition_builds' if initial else 'partition_rebuilds'
        self.statistics[key]+=1

    def events(self,domain=(1.,64.)):
        """List all possible mask-change onsets in a declared finite domain.

        Protected holes are deliberately included: their protection can change
        after a foreground removal. Zero pits never meet a positive low bound.
        Domain endpoints are bookkeeping boundaries, not necessarily events.
        """
        if len(domain)!=2:raise ValueError('Use a two-endpoint ratio domain')
        left,right=map(float,domain);L=self.model.level
        if not (math.isfinite(left) and math.isfinite(right) and 1<=left<right):
            raise ValueError('Use finite 1 <= lower < upper ratio bounds')
        if not math.isfinite(L*right) or L/right<=0:
            raise ValueError('Density-window thresholds must stay positive and finite')
        boundaries=[left,right];details=[]
        kinds=(('foreground_removal',self.model.peaks[1:]),
               ('background_fill',self._initial_pits[1:]))
        for kind,values in kinds:
            for component_id,value in enumerate(values,1):
                value=float(value)
                if not math.isfinite(value) or value<=0:continue
                if kind=='foreground_removal':
                    estimate=value/L
                    predicate=lambda k,v=value: v < L*float(k)
                else:
                    estimate=L/value
                    predicate=lambda k,v=value: v >= L/float(k)
                onset=_first_float_onset(estimate,predicate,left,right)
                if onset is None:continue
                current,previous,adjustments=onset
                boundaries.append(current)
                details.append(dict(kind=kind,component_id=component_id,
                    component_value=value,algebraic_ratio=estimate,
                    onset_ratio=current,previous_ratio=previous,
                    onset_predicate=True,previous_predicate=False,
                    nextafter_adjustments=adjustments))
        self.event_details=sorted(details,key=lambda d:(d['onset_ratio'],d['kind'],d['component_id']))
        return sorted(set(boundaries))

    def apply(self,factor):
        """Evaluate the original cleanup exactly, caching its expensive relabel."""
        if not np.isscalar(factor) or not np.isfinite(factor) or factor<1:
            raise ValueError('Use a finite density ratio factor >= 1')
        factor=float(factor);m=self.model;high=m.level*factor;low=m.level/factor
        if not np.isfinite(high) or low<=0:
            raise ValueError('Density-window thresholds must remain positive and finite')
        remove_labels=m.peaks<high;remove_labels[0]=False
        self.statistics['apply_calls']+=1
        if np.array_equal(remove_labels,self._remove_labels):
            self.statistics['partition_reuses']+=1
        else:self._set_partition(remove_labels)
        fill_labels=(self._pits>=low)&~self._protected
        result=self._intermediate|fill_labels[self._background]
        if np.any(result&~m.eligible):raise ArithmeticError('Cleanup restored alpha-ineligible cells')
        if np.any(m.eligible&(m.density>=high)&~result):
            raise ArithmeticError('Cleanup removed an eligible high-density core')
        if np.any(result&(m.density<low)):
            raise ArithmeticError('Cleanup escaped its lower density threshold')
        removed=m.raw&~result;filled=~m.raw&result
        return result,dict(removed_component_count=int(remove_labels.sum()),
            filled_background_component_count=int(fill_labels.sum()),
            alpha_protected_background_component_count=int(self._alpha_protected.sum()),
            alpha_ineligible_cells=int(np.count_nonzero(~m.eligible)),
            removed_cells=int(removed.sum()),filled_cells=int(filled.sum()),
            changed_cells=int(np.count_nonzero(result!=m.raw)),removed=removed,filled=filled)
