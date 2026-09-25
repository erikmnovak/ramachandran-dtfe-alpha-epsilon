"""Exact component guards for a common, unsmoothed alpha experiment.

These guards compare nested unions of closed triangles on an already certified
periodic triangulation. They do not compare with KDE or choose a radius. Closed
foreground triangles connect through vertices; the OPEN complement connects
through shared edges of unselected faces. Confusing those two conventions can
silently join holes through a point belonging to the closed foreground.

A legitimate density-level merger is allowed. The foreground guard only rejects
new pieces without any retained 98% core when a baseline outer component splits.
An existing outer component without a 98% core is allowed to remain unchanged or
shrink as one component. Together with fixed baseline gates, exact construction
coverage and full inner-region preservation are checked by the calling selector.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def edge_adjacent_face_pairs(edge_ids):
    """Recover the two incident face IDs of every certified periodic edge."""
    ids=np.asarray(edge_ids)
    if ids.ndim!=2 or ids.shape[1]!=3 or ids.dtype.kind not in 'iu' or not len(ids):
        raise ValueError('Expected nonempty integer edge IDs with shape [face,3]')
    flat=ids.ravel()
    if int(flat.min())<0 or int(flat.max())>=len(flat):
        raise ValueError('Edge IDs must fit the complete triangle-incidence table')
    counts=np.bincount(flat.astype(np.int64))
    if not np.all(counts==2):
        raise ValueError('Every contiguous edge ID must have exactly two incidences')
    face=np.repeat(np.arange(len(ids),dtype=np.int32),3)
    left=np.full(len(counts),np.iinfo(np.int32).max,dtype=np.int32)
    right=np.full(len(counts),-1,dtype=np.int32)
    np.minimum.at(left,flat,face);np.maximum.at(right,flat,face)
    return np.column_stack((left,right))


def _mask(value,name):
    result=np.asarray(value)
    if result.ndim!=1 or result.dtype!=bool:
        raise ValueError(name+' must be a one-dimensional Boolean array')
    return result


def background_component_labels(selected,face_pairs):
    """Label connected OPEN complement pieces; selected triangles get -1.

    Shared-edge adjacency is sufficient for the open complement of a closed
    subcomplex. Compacting to unselected faces avoids treating selected faces as
    millions of isolated graph vertices. Labels are contiguous signed int32.
    """
    selected=_mask(selected,'selected')
    pairs=np.asarray(face_pairs)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or pairs.dtype.kind not in 'iu':
        raise ValueError('face_pairs must have shape [edge,2] and integer entries')
    if pairs.size and (int(pairs.min())<0 or int(pairs.max())>=len(selected)):
        raise ValueError('Face pair outside the triangle array')
    inactive=np.flatnonzero(~selected)
    labels=np.full(len(selected),-1,dtype=np.int32)
    if not len(inactive):return labels
    remap=np.full(len(selected),-1,dtype=np.int32)
    remap[inactive]=np.arange(len(inactive),dtype=np.int32)
    keep=(~selected[pairs[:,0]])&(~selected[pairs[:,1]])
    links=remap[pairs[keep]]
    graph=coo_matrix((np.ones(len(links),dtype=bool),(links[:,0],links[:,1])),
        shape=(len(inactive),len(inactive))).tocsr()
    _,compact=connected_components(graph,directed=False,return_labels=True)
    labels[inactive]=compact
    return labels


def _labels(value,selected,name):
    labels=np.asarray(value)
    if labels.shape!=selected.shape or labels.dtype.kind not in 'iu':
        raise ValueError(name+' must be an integer label per triangle')
    if not np.array_equal(labels>=0,selected):
        raise ValueError(name+' must be nonnegative exactly on its selected region')
    active=labels[selected]
    count=int(active.max())+1 if len(active) else 0
    if count and not np.array_equal(np.unique(active),np.arange(count)):
        raise ValueError(name+' labels must be contiguous from zero')
    return labels,count


def evaluate_topology_guards(baseline_inner,baseline_outer,candidate_outer,
        baseline_foreground_labels,candidate_foreground_labels,candidate_background_labels):
    """Evaluate exact, location-aware guards without demanding core separation.

    Each candidate component belongs to one baseline component because outer
    triangles only disappear at fixed gates. A baseline component may split if
    EVERY remaining piece contains at least one unchanged inner triangle. This
    excludes new dangling pieces but does not prohibit two inner cores sharing
    one outer component. A new background component is one with no baseline
    background face: it is rejected even if a different old hole disappears,
    which a mere Betti-number comparison could miss.

    Returns JSON-compatible component IDs and pass/fail flags. The caller must
    independently verify observation coverage, exact fixed thresholds, and the
    certified geometry from which all labels were derived.
    """
    inner=_mask(baseline_inner,'baseline_inner')
    outer=_mask(baseline_outer,'baseline_outer')
    candidate=_mask(candidate_outer,'candidate_outer')
    if inner.shape!=outer.shape or candidate.shape!=outer.shape:
        raise ValueError('Region arrays must have identical shapes')
    if np.any(inner&~outer):raise ValueError('Baseline inner must lie in baseline outer')
    if np.any(candidate&~outer):raise ValueError('Candidate must be a subset at the fixed outer gate')
    base,nb=_labels(baseline_foreground_labels,outer,'baseline_foreground_labels')
    current,nc=_labels(candidate_foreground_labels,candidate,'candidate_foreground_labels')
    background,ng=_labels(candidate_background_labels,~candidate,'candidate_background_labels')
    # Candidate components are descendants of exactly one baseline component.
    parent=np.full(nc,-1,dtype=np.int32)
    if nc:
        ids,first=np.unique(current[candidate],return_index=True)
        parent[ids]=base[candidate][first]
        if not np.array_equal(parent[current[candidate]],base[candidate]):
            raise ValueError('A candidate component cannot cross baseline foreground components')
    children=np.bincount(parent,minlength=nb) if nc else np.zeros(nb,dtype=int)
    has_core=np.zeros(nc,dtype=bool)
    has_core[np.unique(current[inner&candidate])]=True
    bad=np.flatnonzero((children[parent]>=2)&~has_core) if nc else np.empty(0,dtype=int)
    seeded=np.zeros(ng,dtype=bool)
    seeded[np.unique(background[~outer])]=True
    new_background=np.flatnonzero(~seeded)
    missing_inner=int(np.count_nonzero(inner&~candidate))
    return dict(passed=bool(not missing_inner and not len(bad) and not len(new_background)),
        full_inner_region_preserved=missing_inner==0,missing_inner_triangles=missing_inner,
        no_new_background_components=len(new_background)==0,
        new_background_component_ids=new_background.astype(int).tolist(),
        no_new_coreless_split_pieces=len(bad)==0,
        coreless_split_component_ids=bad.astype(int).tolist(),
        split_baseline_component_ids=np.flatnonzero(children>=2).astype(int).tolist(),
        baseline_foreground_components=nb,candidate_foreground_components=nc,
        candidate_background_components=ng,
        candidate_component_parent_ids=parent.astype(int).tolist(),
        candidate_component_has_inner_core=has_core.tolist())
