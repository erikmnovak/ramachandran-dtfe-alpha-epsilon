#!/usr/bin/env python3
"""Choose a shared triangle radius against construction-calibrated DTFE fields.

This saved-data experiment performs no fit or geometry query. The direct field
and each triangle candidate have their own inclusive construction thresholds.
Only DTFE arrays enter the decision; the published-reference assessment is a
separate, later stage. Every numerical input is allowlisted before execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.ndimage import label

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'novak_work'))
from rama_compact_triangle_topology import PeriodicTriangleTopology
from rama_periodic_interpolation import evaluate_periodic_density
from rama_periodic_region_metrics import periodic_region_topology
from rama_construction_alpha_inputs import source_plan
from rama_internal_alpha_selection import select_shared_alpha
from rama_dtfe_artifacts import (read_f64, sha256_file, write_json,
                                check_sources, artifact_inventory)
from rama_direct_dtfe_field import run_limited

CLASSES = ('CisPro', 'PrePro', 'TransPro', 'Gly', 'IleVal', 'General')
KAPPAS = ('1/4', '1/2', '5/8', '3/4', '1', '3/2', '2', 'Inf')
TARGETS = (.98, .995)
GRIDS = (768, 1536)
LIMITS = dict(seconds_per_class=900, address_space_bytes=4*1024**3,
              process_group_rss_bytes=4*1024**3, threads=1)


def _read(path):
    return json.loads(Path(path).read_text())


def _integers(values, *, minimum=0):
    values = np.asarray(values)
    if (not np.isfinite(values).all() or np.any(values < minimum)
            or np.any(values >= 2.**63) or not np.equal(values, np.floor(values)).all()):
        raise ValueError('Expected exact bounded integer values')
    return values.astype(np.int64)


def _construction_levels(values, weights, targets=((49, 50), (199, 200))):
    """Highest inclusive level attaining each exact weighted ceiling rank.

    A duplicate coordinate contributes its complete integer multiplicity;
    threshold ties are indivisible. No tolerance merges nearby densities.
    """
    values = np.asarray(values)
    weights = _integers(weights, minimum=1)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('Finite nonnegative densities and positive weights required')
    order = np.argsort(-values, kind='stable')
    total = sum(map(int, weights))
    if total > np.iinfo(np.int64).max:
        raise ValueError('Total weight exceeds exact int64 accumulation range')
    cumulative = np.cumsum(weights[order])
    rows = []
    for numerator, denominator in targets:
        if not 0 < numerator < denominator:
            raise ValueError('Targets must lie strictly between zero and one')
        required = (numerator*total+denominator-1)//denominator
        gate = float(values[order[np.searchsorted(cumulative, required)]])
        above = int(weights[values > gate].sum())
        covered = int(weights[values >= gate].sum())
        if not above < required <= covered:
            raise ValueError('Inclusive weighted rank invariant failed')
        rows.append(dict(target=numerator/denominator, density_gate=gate,
            required_count=required, construction_count=covered, construction_total=total,
            strictly_above_count=above, tie_count=covered-above,
            tie_overshoot=covered-required, defined=True, status='calibrated'))
    return rows


def _component_labels(mask):
    """Closed periodic cell components, including corner and seam contacts."""
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 2 or len(set(mask.shape)) != 1:
        raise ValueError('A square Boolean grid is required')
    labels, count = label(mask, structure=np.ones((3, 3), np.uint8))
    parent = np.arange(count+1)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for shift in (-1, 0, 1):
        for a, b in ((labels[0], np.roll(labels[-1], shift)),
                     (labels[:, 0], np.roll(labels[:, -1], shift))):
            for left, right in np.unique(np.column_stack((a, b)), axis=0):
                if left and right:
                    left, right = root(left), root(right)
                    if left != right:
                        parent[right] = left
    roots = np.array([root(i) for i in range(count+1)])
    merged = roots[labels].ravel()
    sizes = np.bincount(merged, minlength=count+1)
    first = np.full(count+1, mask.size, np.int64)
    np.minimum.at(first, merged, np.arange(mask.size))
    active = np.flatnonzero(sizes[1:])+1
    active = active[np.argsort(first[active], kind='stable')]
    canonical = np.zeros(count+1, np.int32)
    canonical[active] = np.arange(1, len(active)+1)
    return canonical[merged].reshape(mask.shape), sizes[active], first[active]


def _component_grid(components, offsets, indices, n):
    """Map exact closed-union components through ALL containing triangles."""
    components, offsets, indices = map(np.asarray, (components, offsets, indices))
    if (components.ndim != 1 or components.dtype.kind not in 'iu'
            or np.any(components < -1) or offsets.shape != (n*n+1,)
            or offsets.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu'
            or offsets[0] != 0 or offsets[-1] != len(indices)
            or np.any(offsets[1:] <= offsets[:-1]) or np.any(indices < 0)
            or np.any(indices >= len(components))):
        raise ValueError('Invalid all-containing CSR or exact component labels')
    labels = components[indices]
    outside = int(components.max())+1
    starts = offsets[:-1].astype(np.intp)
    low = np.minimum.reduceat(np.where(labels < 0, outside, labels), starts)
    high = np.maximum.reduceat(labels, starts)
    covered = high >= 0
    if not np.array_equal(low[covered], high[covered]):
        raise ValueError('A shared closed face belongs to disconnected components')
    return np.where(covered, high+1, 0).reshape(n, n)


def _correspondence(spatial, direct_labels, direct_sizes, direct_first, ncomponents):
    """All positive-area component links; keep even unsampled exact islands."""
    if (spatial.shape != direct_labels.shape or spatial.dtype.kind not in 'iu'
            or spatial.min() < 0 or spatial.max() > ncomponents):
        raise ValueError('Invalid exact-component sample grid')
    nr = len(direct_sizes)
    if direct_labels.min() < 0 or direct_labels.max() != nr or len(direct_first) != nr:
        raise ValueError('Direct component labels differ')
    sizes = np.bincount(spatial.ravel(), minlength=ncomponents+1)[1:]
    first = np.full(ncomponents+1, spatial.size, np.int64)
    np.minimum.at(first, spatial.ravel(), np.arange(spatial.size))
    clinks, dlinks = [[] for _ in range(ncomponents)], [[] for _ in range(nr)]
    kept = np.zeros(nr, np.int64)
    shared = (spatial > 0) & (direct_labels > 0)
    keys, counts = np.unique(spatial[shared].astype(np.int64)*(nr+1)+direct_labels[shared], return_counts=True)
    links = []
    for key, count in zip(keys, counts):
        ci, di, count = int(key//(nr+1))-1, int(key%(nr+1))-1, int(count)
        clinks[ci].append(di); dlinks[di].append(ci); kept[di] += count
        links.append(dict(candidate_component=ci, direct_component=di, shared_cells=count,
            candidate_fraction=count/int(sizes[ci]), direct_fraction=count/int(direct_sizes[di])))
    return dict(candidate_components=[dict(component=i, sampled_cells=int(size),
            first_cell=int(first[i+1]) if size else None, direct_components=clinks[i])
            for i, size in enumerate(sizes)],
        direct_components=[dict(component=i, cells=int(size), first_cell=int(direct_first[i]),
            retained_cells=int(kept[i]), recall=float(kept[i]/size), candidate_components=dlinks[i])
            for i, size in enumerate(direct_sizes)], links=links,
        missed_direct_components=[i for i, x in enumerate(dlinks) if not x],
        unmatched_candidate_components=[i for i, x in enumerate(clinks) if not x],
        unsampled_candidate_components=[i for i, x in enumerate(sizes) if not x],
        candidate_components_joining_direct=[i for i, x in enumerate(clinks) if len(x)>1],
        direct_components_split_across_candidates=[i for i, x in enumerate(dlinks) if len(x)>1])


def _load_csr(directory, n, ntriangles, geometry_sha):
    metadata = _read(directory/'incidences.json')
    if (metadata['grid_size'] != n or metadata['phase'] != 0.
            or metadata['mask_axes'] != ['psi', 'phi'] or metadata['order'] != 'C'
            or metadata['geometry_sha256'] != geometry_sha):
        raise ValueError('Canonical query grid differs')
    arrays = []
    for key, dtype in (('offsets', '<u8'), ('indices', '<u4')):
        m = metadata[key]; path = directory/m['path']
        if (m['dtype'] != dtype or path.stat().st_size != np.dtype(dtype).itemsize*m['count']
                or sha256_file(path) != m['sha256']):
            raise ValueError('Changed CSR artifact')
        arrays.append(np.memmap(path, dtype=dtype, mode='r', shape=(m['count'],)))
    offsets, indices = arrays
    if (offsets.shape != (n*n+1,) or offsets[0] != 0 or offsets[-1] != len(indices)
            or np.any(offsets[1:] <= offsets[:-1]) or np.any(indices >= ntriangles)):
        raise ValueError('Invalid complete-mesh CSR bounds')
    bad = np.flatnonzero(indices[1:] <= indices[:-1])+1
    if not np.isin(bad, offsets[1:-1]).all():
        raise ValueError('Containing triangle rows must be sorted and unique')
    return offsets, indices


def _candidate_replay(estimator, selected, bases, weights, gates, radii):
    """Replay saved inclusive thresholds using max incident eligible gates.

    Triangle interpolation uses the original estimator's saved image-coordinate
    gates. It need not equal min of new canonical-vertex interpolation values.
    """
    rows = estimator['rows']; total = int(weights.sum()); nv = len(weights)
    if len(rows) != 16 or selected.shape != (16, len(bases)):
        raise ValueError('Expected eight two-target triangle candidates')
    for j, kappa in enumerate(KAPPAS):
        pair = rows[2*j:2*j+2]
        if (any(r['kappa'] != kappa or r['target'] != TARGETS[q]
                or r['outcome_index'] != 2*j+q or r['construction_total'] != total
                for q, r in enumerate(pair)) or pair[0]['alpha_deg'] != pair[1]['alpha_deg']):
            raise ValueError('Candidate identity, order or shared alpha differs')
        alpha = pair[0]['alpha_deg']
        native = pair[0]['native_alpha_deg']
        factor = {'1/4':.25,'1/2':.5,'5/8':.625,'3/4':.75,'1':1.,'3/2':1.5,'2':2.}.get(kappa)
        expected_alpha = 'Inf' if kappa == 'Inf' else None if native is None else native*factor
        if alpha != expected_alpha or pair[1]['native_alpha_deg'] != native:
            raise ValueError('Declared native-derived alpha differs')
        if alpha is None:
            if any(r['defined'] for r in pair) or selected[2*j:2*j+2].any():
                raise ValueError('Unavailable alpha cannot define a region')
            continue
        eligible = radii <= (math.inf if alpha == 'Inf' else alpha)
        heights = np.full(nv, -math.inf)
        np.maximum.at(heights, bases[eligible].ravel(), np.repeat(gates[eligible], 3))
        available = int(weights[np.isfinite(heights)].sum())
        for q, row in enumerate(pair):
            required = ((49*total+49)//50 if q == 0 else (199*total+199)//200)
            bits = selected[2*j+q]
            if row['required_count'] != required or row['eligible_weight'] != available:
                raise ValueError('Saved exact rank or support ceiling differs')
            if not row['defined']:
                if available >= required or bits.any():
                    raise ValueError('Infeasible state has sufficient support or nonempty placeholder')
                continue
            gate = row['density_gate']
            count = int(weights[heights >= gate].sum())
            above = int(weights[heights > gate].sum())
            predicate = eligible & (gates >= gate)
            covered = np.zeros(nv, bool); covered[bases[bits].ravel()] = True
            if (not above < required <= count or count != row['construction_count']
                    or above != row['strictly_above_count'] or count-above != row['tie_count']
                    or count-required != row['tie_overshoot'] or not np.array_equal(bits, predicate)
                    or int(bits.sum()) != row['selected_triangle_count']
                    or int(weights[covered].sum()) != count):
                raise ValueError('Saved triangle predicate or weighted inclusive rank differs')
        if all(r['defined'] for r in pair) and np.any(selected[2*j] & ~selected[2*j+1]):
            raise ValueError('Saved shared-alpha construction pair is not nested')
    return dict(status='passed', rows=16, ranks='Exact integer ceilings and inclusive ties',
                predicates='Original estimator radii and gates; canonical vertex multiplicities')


def _grid_density(field, n):
    """Native physical [phi,psi] interpolation, output C spatial [psi,phi]."""
    axis = -180+(np.arange(n)+.5)*360/n
    values = np.empty((n, n))
    for first in range(0, n, 32):
        psi, phi = np.meshgrid(axis[first:first+32], axis, indexing='ij')
        points = np.column_stack((phi.ravel(), psi.ravel()))
        values[first:first+32] = evaluate_periodic_density(field, points).reshape(-1, n)
    return axis, values


def _read_audit(allowed, output):
    """Reject unplanned project data reads; Python/library imports are separate."""
    observed = set(); allowed = {str((PROJECT/p).resolve()) for p in allowed}
    output = Path(output).resolve()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(args[0]).resolve(); mode, flags = args[1:3]
        reading = ('r' in mode or '+' in mode) if isinstance(mode, str) else (flags & os.O_ACCMODE) != os.O_WRONLY
        if not reading or not path.is_relative_to(PROJECT):
            return
        if path.is_relative_to(output) or path.suffix in ('.py', '.pyc', '.so'):
            return
        if str(path) not in allowed:
            raise PermissionError('Unplanned project data read: '+str(path))
        observed.add(str(path.relative_to(PROJECT)))
    sys.addaudithook(audit)
    return observed


def _worker(category, out, plan_path, plan_sha):
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS['address_space_bytes'],)*2)
    started = time.monotonic()
    if sha256_file(plan_path) != plan_sha:
        raise ValueError('Changed prospective plan')
    plan = _read(plan_path); case = plan['cases'][category]
    allowed = dict(plan['sources_and_inputs'])
    allowed[str(plan_path.relative_to(PROJECT))] = plan_sha
    observed = _read_audit(allowed, out)
    check_sources(plan['sources_and_inputs'])
    geometry_dir = PROJECT/case['geometry_directory']; meta = _read(geometry_dir/'metadata.json')
    constructor = _read(PROJECT/case['construct_root']/'report.json')
    if meta['geometry_certificate'] != constructor['geometry_certificate']:
        raise ValueError('Full embedded geometry certificate differs')
    nv, nt = case['n_vertices'], case['n_triangles']
    contracts = dict(vertices=((nv, 2), ['vertex','axis'], 'degree'),
        weights=((nv,), ['vertex'], 'multiplicity'), bases=((nt,3), ['triangle','corner'], 'zero_based_vertex_id'),
        lifts=((nt,3,2), ['triangle','corner','axis'], 'integer_periods'),
        areas=((nt,), ['triangle'], 'degree_squared'))
    arrays = {k: read_f64(geometry_dir, meta['arrays'][k], shape=s, axes=a, units=u)
              for k, (s,a,u) in contracts.items()}
    bases = _integers(arrays.pop('bases')); lifts = _integers(arrays.pop('lifts'), minimum=-2**63)
    weights = _integers(arrays.pop('weights'), minimum=1)
    vertices, areas = arrays['vertices'], arrays['areas']
    if weights.sum() != case['observation_count'] or np.any(bases >= nv) or np.any(areas <= 0):
        raise ValueError('Full construction population or geometry differs')
    ep = PROJECT/case['estimator_report']; estimator = _read(ep)
    if estimator['sigma_deg'] != case['sigma_deg'] or estimator['area_floor_quantile'] != .01:
        raise ValueError('Fixed estimator recipe differs')
    density = read_f64(ep.parent, estimator['arrays']['density'], shape=(384,384), axes=['psi','phi'], units='density')
    if np.any(density < 0):
        raise ValueError('Nonnegative saved field required')
    field = np.asarray(density.T, dtype=np.float64)
    vertex_density = evaluate_periodic_density(field, vertices)
    direct_rows = _construction_levels(vertex_density, weights)
    np.savez_compressed(out/'construction.npz', vertices=vertices, multiplicities=weights,
                        vertex_density=vertex_density, membership=vertex_density[:,None] >= np.array([r['density_gate'] for r in direct_rows]))
    write_json(out/'direct_thresholds.json', dict(status='frozen_before_grid_queries', rows=direct_rows,
        interpolation='Stable periodic nested affine bilinear interpolation; no threshold tolerance',
        density_axes=['psi','phi'], physical_axes=['phi','psi'], density_grid_size=384,
        field_source=case['estimator_report']))
    frozen_sha = sha256_file(out/'direct_thresholds.json')
    gates = read_f64(ep.parent, estimator['arrays']['gates'], shape=(nt,), axes=['triangle'], units='density')
    rm = case['radii_metadata']
    radii = read_f64(PROJECT/case['radii_directory'], rm, shape=(nt,), axes=rm['axes'], units=rm['units'])
    if np.any(gates < 0) or np.any(radii <= 0):
        raise ValueError('Nonnegative triangle gates and positive circumradii required')
    reconstructed = vertex_density[bases].min(axis=1)
    delta = reconstructed-gates
    positive = gates > 0
    arithmetic_diagnostic = dict(unequal_triangle_count=int(np.count_nonzero(delta)),
        maximum_absolute_difference=float(np.abs(delta).max()),
        maximum_relative_difference_positive_saved_gate=float(np.max(np.abs(delta[positive])/gates[positive])) if positive.any() else None,
        maximum_difference_divided_by_maximum_saved_gate=float(np.abs(delta).max()/gates.max()) if gates.max() else None,
        saved_gate_exceeds_stable_canonical_minimum_count=int(np.count_nonzero(delta < 0)),
        zero_saved_gate_count=int(np.count_nonzero(~positive)),
        scope='Numerical diagnostic only: original image-coordinate Julia triangle gates versus minimum stable Python canonical-vertex values; no replacement or acceptance tolerance')
    del reconstructed, delta, positive
    pm = estimator['arrays']['selected']; path = ep.parent/pm['path']; width=(nt+7)//8
    if (pm['dtype'] != 'u1' or pm['order'] != 'C' or pm['bitorder'] != 'little'
            or pm['shape'] != [16,width] or path.stat().st_size != 16*width
            or sha256_file(path) != pm['sha256']):
        raise ValueError('Saved selected triangle bits differ')
    packed = np.fromfile(path,np.uint8).reshape(16,width)
    unpacked = np.unpackbits(packed, axis=1, bitorder='little')
    if unpacked[:,nt:].any():
        raise ValueError('Nonzero selected bit padding')
    selected = unpacked[:,:nt].astype(bool); del unpacked
    replay = _candidate_replay(estimator, selected, bases, weights, gates, radii)
    write_json(out/'candidate_replay.json', replay)
    topology = PeriodicTriangleTopology(bases, lifts, nvertices=nv,
        geometry_certificate=meta['geometry_certificate'], max_triangles=3000000)
    candidate_ids = [f"sigma{case['sigma_deg']}_floor0.01_kappa{k}" for k in KAPPAS]
    methods = ['direct_field']+candidate_ids
    candidates = [dict(id=name, alpha_deg=estimator['rows'][2*j]['alpha_deg'], regions=[])
                  for j,name in enumerate(candidate_ids)]
    grid_data = {}
    for n in GRIDS:
        axis, values = _grid_density(field,n)
        values.astype('>f8').tofile(out/f'grid_{n}_phase0_density.f64')
        direct = [values >= row['density_gate'] for row in direct_rows]
        if np.any(direct[0] & ~direct[1]):
            raise ValueError('Direct regions are not nested')
        labels = [_component_labels(mask) for mask in direct]
        offsets, indices = _load_csr(PROJECT/case['grids'][str(n)],n,nt,case['geometry_sha256'])
        masks = np.zeros((9,2,(n*n+7)//8),np.uint8)
        masks[0] = np.packbits(np.stack(direct).reshape(2,-1),axis=1,bitorder='little')
        defined = np.zeros((9,2),bool); defined[0] = True
        direct_topology = [periodic_region_topology(mask) for mask in direct]
        grid_data[n] = dict(axis=axis,direct=direct,labels=labels,offsets=offsets,indices=indices,
            packed=masks,defined=defined,rows=[],direct_topology=direct_topology)
        for q,row in enumerate(direct_rows):
            grid_data[n]['rows'].append(dict(method='direct_field',target=TARGETS[q],defined=True,
                candidate_cells=int(direct[q].sum()),area_fraction=float(direct[q].mean()),
                raster_topology=direct_topology[q],construction_count=row['construction_count']))
        del values
    unique = {}; exact_rows = []
    for index,row in enumerate(estimator['rows']):
        j,q = divmod(index,2); name=candidate_ids[j]
        if not row['defined']:
            for n in GRIDS:
                measured=dict(method=name,target=TARGETS[q],grid_size=n,defined=False,status=row['status'],
                    intersection_cells=None,union_cells=None,introduced_join_count=None)
                candidates[j]['regions'].append(measured); grid_data[n]['rows'].append(measured)
            continue
        key=hashlib.sha256(packed[index].tobytes()).hexdigest()
        if key not in unique:
            result=topology.region(np.flatnonzero(selected[index]),component_labels=True)
            component_labels=result.pop('triangle_component_labels')
            exact=dict(result,selected_sha256=key,canonical_area_deg2=float(areas[selected[index]].sum()))
            cache={}
            for n,g in grid_data.items():
                spatial=_component_grid(component_labels,g['offsets'],g['indices'],n)
                mask=spatial>0
                cache[n]=dict(spatial=spatial,mask=mask,packed=np.packbits(mask.ravel(),bitorder='little'),
                              raster_topology=periodic_region_topology(mask))
            unique[key]=(exact,cache); exact_rows.append(exact)
        exact,cache=unique[key]
        for n,g in grid_data.items():
            c=cache[n]; mask=c['mask']; direct=g['direct'][q]
            inter=int(np.count_nonzero(mask & direct)); union=int(np.count_nonzero(mask | direct))
            links=_correspondence(c['spatial'],*g['labels'][q],exact['betti'][0])
            measured=dict(method=name,target=TARGETS[q],grid_size=n,defined=bool(union),
                status='measured' if union else 'undefined_empty_union',
                intersection_cells=inter if union else None,union_cells=union if union else None,
                introduced_join_count=len(links['candidate_components_joining_direct']) if union else None,
                candidate_cells=int(mask.sum()),direct_cells=int(direct.sum()),
                missing_cells=int(np.count_nonzero(direct & ~mask)),extra_cells=int(np.count_nonzero(mask & ~direct)),
                area_fraction=float(mask.mean()),direct_area_fraction=float(direct.mean()),
                jaccard=inter/union if union else None,component_correspondence=links,
                exact_triangle_topology=exact,raster_topology=c['raster_topology'],
                direct_raster_topology=g['direct_topology'][q])
            candidates[j]['regions'].append(measured);g['rows'].append(measured)
            g['packed'][j+1,q]=c['packed'];g['defined'][j+1,q]=True
    # A complete pair remains nested on every all-containing query lattice.
    for n,g in grid_data.items():
        for j in range(9):
            if g['defined'][j].all() and np.any(g['packed'][j,0] & ~g['packed'][j,1]):
                raise ValueError('Measured construction pair is not nested')
        stem=f'grid_{n}_phase0'
        np.savez_compressed(out/(stem+'.npz'),axis=g['axis'],targets=np.array(TARGETS),
            methods=np.array(methods),packed_masks=g['packed'],defined=g['defined'],
            mask_axes=np.array(['psi','phi']),bitorder=np.array('little'),grid_size=n,phase=0.)
        write_json(out/(stem+'.json'),dict(grid_size=n,phase=0.,mask_axes=['psi','phi'],
            order='C',bitorder='little',methods=methods,rows=g['rows']))
    decision=select_shared_alpha(candidates);write_json(out/'selection.json',decision)
    write_json(out/'report.json',dict(status='completed',category=category,observation_count=int(weights.sum()),
        sigma_deg=case['sigma_deg'],area_floor_quantile=.01,density_grid_size=384,
        candidate_ids=candidate_ids,methods=methods,targets=list(TARGETS),candidate_rows=estimator['rows'],
        direct_rows=direct_rows,unique_triangle_unions=len(unique),selection_status=decision['status'],
        selected_id=decision['selection']['selected_id'] if decision['selection'] else None,
        native_candidate_id=candidate_ids[4],unrestricted_candidate_id=candidate_ids[7],
        candidate_source=case['estimator_report'],geometry_sha256=case['geometry_sha256'],
        interpolation_arithmetic_diagnostic=arithmetic_diagnostic,
        canonical_triangle_topology=exact_rows,direct_thresholds_sha256=frozen_sha,
        scope='Native-derived eight-radius pilot; direct regions and triangle regions separately construction calibrated',
        topology_scope='Exact closed triangles conditional on certified embedded mesh; direct topology is sampled closed periodic cells',
        arithmetic_scope='Stable nested interpolation represents the same bilinear field in real arithmetic; historical Julia weighted-corner triangle gates remain unchanged, without a bitwise arithmetic equivalence claim',
        fitting=False,geometry_queries=False,top2018_used=False,published_reference_access=False,
        operational_elapsed_seconds=time.monotonic()-started))
    if sha256_file(out/'direct_thresholds.json')!=frozen_sha or sha256_file(plan_path)!=plan_sha:
        raise ValueError('Pre-query thresholds or prospective plan changed')
    check_sources(plan['sources_and_inputs'])
    write_json(out/'input_access.json',dict(project_data_files_read=sorted(observed),
        unplanned_project_data_reads_permitted=False,published_numerical_inputs=False))
    write_json(out/'terminal.json',dict(status='completed',category=category,plan_sha256=plan_sha,
        sources_and_inputs=plan['sources_and_inputs'],artifacts=artifact_inventory(out)))


def _plan():
    plan=source_plan();plan.update(schema='construction-alpha-pilot-1',limits=LIMITS,
        targets=list(TARGETS),grids=list(GRIDS),kappas=list(KAPPAS),
        no_fitting=True,no_geometry_queries=True,selection='Exact prescribed internal DTFE comparison')
    sources=[Path(__file__),Path(__file__).with_name('test_rama_construction_alpha_pilot.py'),
        Path(__file__).with_name('rama_internal_alpha_selection.py'),
        Path(__file__).with_name('test_rama_internal_alpha_selection.py'),
        Path(__file__).with_name('test_rama_construction_alpha_inputs.py'),
        Path(__file__).with_name('rama_dtfe_artifacts.py'),Path(__file__).with_name('rama_direct_dtfe_field.py'),
        Path(__file__).with_name('rama_top8000_development_data.py'),
        PROJECT/'novak_work/rama_periodic_interpolation.py',
        PROJECT/'novak_work/rama_compact_triangle_topology.py',
        PROJECT/'novak_work/rama_periodic_region_metrics.py']
    plan['sources_and_inputs'].update({str(p.relative_to(PROJECT)):sha256_file(p) for p in sources})
    plan['versions']=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=PROJECT/'novak_work/validation_results/construction_alpha_pilot_v1')
    parser.add_argument('--category',choices=CLASSES);parser.add_argument('--plan',type=Path)
    parser.add_argument('--plan-sha256');args=parser.parse_args();out=args.out.resolve()
    if args.category:
        try:_worker(args.category,out,args.plan.resolve(),args.plan_sha256)
        except BaseException as error:
            if not (out/'terminal.json').exists():
                write_json(out/'terminal.json',dict(status='failed',category=args.category,error=str(error),traceback=traceback.format_exc()))
            raise
        return
    if out.exists():raise ValueError('New study directory required; no silent retry or overwrite')
    plan=_plan();out.mkdir(parents=True);write_json(out/'plan.json',plan);plan_sha=sha256_file(out/'plan.json')
    jobs=[]
    for category in CLASSES:
        check_sources(plan['sources_and_inputs']);directory=out/category;directory.mkdir()
        result=run_limited([sys.executable,str(Path(__file__).resolve()),'--category',category,
            '--out',str(directory),'--plan',str(out/'plan.json'),'--plan-sha256',plan_sha],
            out/(category+'.log'),seconds=LIMITS['seconds_per_class'],rss_bytes=LIMITS['process_group_rss_bytes'])
        if not (directory/'terminal.json').exists():
            write_json(directory/'terminal.json',dict(status='failed',category=category,execution=result))
        terminal=_read(directory/'terminal.json')
        if terminal['status']=='completed':
            if (terminal['plan_sha256']!=plan_sha or terminal['artifacts']!=artifact_inventory(directory)
                    or terminal['sources_and_inputs']!=plan['sources_and_inputs']):
                raise ValueError('Class terminal binding differs')
        result.update(category=category,status=terminal['status']);jobs.append(result)
        write_json(out/(category+'_execution.json'),result)
        print(category,result['status'],result['returncode'],flush=True)
        if result['status']!='completed' or result['returncode']!=0:break
    check_sources(plan['sources_and_inputs'])
    if sha256_file(out/'plan.json')!=plan_sha:raise ValueError('Prospective plan changed')
    complete=len(jobs)==6 and all(x['status']=='completed' and x['returncode']==0 for x in jobs)
    write_json(out/'terminal.json',dict(status='completed' if complete else 'incomplete',jobs=jobs,
        plan_sha256=plan_sha,sources_and_inputs=plan['sources_and_inputs'],artifacts=artifact_inventory(out)))
    if not complete:raise SystemExit(1)


if __name__=='__main__':main()
