#!/usr/bin/env python3
"""Execute and seal the empirical raw-DTFE 2D alpha policy without KDE input.

Construction levels use inclusive weighted order statistics at alpha=infinity.
We freeze those levels throughout the CisPro event search. If a finite radius
still covers the target at that level, it is also the largest feasible calibrated
level for that finite radius: restricting triangles cannot increase inclusion
heights, and every higher level already failed at infinity. This preserves
both calibration semantics and nested candidate regions.

Each class is exported in its own process. Scores are minimum raw corner
count-density among triangle vertices, maximized over eligible containing
triangles, and normalized by observation count. They are selection scores,
not an interpolated probability density with a unit-integral guarantee.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark'))
from rama_construction_alpha_pilot import _load_csr
from rama_compact_triangle_topology import PeriodicTriangleTopology
from rama_unsmoothed_alpha_policy import (VERSION, CLASSES, TARGETS,
    default_alpha, required_count, select_cispro_event, cispro_rejection_reasons)

SOURCE=Path('novak_work/validation_results/construction_alpha_unfloored_v1')
DEFAULT=Path('novak_work/validation_results/unsmoothed_alpha_policy_v1')
GRIDS=(768,1536)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,data):
    with path.open('x') as stream:
        json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')


class Inputs:
    def __init__(self): self.pins={}
    def path(self,path,expected=None):
        p=ROOT/path;digest=sha(p)
        if expected is not None and digest!=expected: raise ValueError('Source changed: '+str(path))
        self.pins[str(p.relative_to(ROOT))]=digest
        return p
    def json(self,path,expected=None): return json.loads(self.path(path,expected).read_text())
    def array(self,folder,meta,integer=False):
        a=np.fromfile(self.path(Path(folder)/meta['path'],meta['sha256']),dtype=meta['dtype']).reshape(meta['shape'])
        assert meta['order']=='C' and np.isfinite(a).all()
        if integer:
            assert np.equal(a,np.floor(a)).all();a=a.astype(np.int64)
        return a


def inclusion_heights(bases,scores,nvertices):
    heights=np.zeros(nvertices)
    for corner in range(3):np.maximum.at(heights,bases[:,corner],scores)
    return heights


def construction_levels(heights,weights):
    total=int(weights.sum());order=np.argsort(-heights);cum=np.cumsum(weights[order]);rows=[]
    for numerator,denominator in TARGETS:
        required=required_count(total,numerator,denominator)
        L=float(heights[order[np.searchsorted(cum,required)]])
        count=int(weights[heights>=L].sum());strict=int(weights[heights>L].sum())
        assert L>0 and strict<required<=count
        rows.append(dict(target=numerator/denominator,required_count=required,
            construction_count=count,strictly_above_count=strict,total_count=total,
            density_gate=L/total,original_count_density_gate=L,
            construction_coverage=count/total))
    assert rows[0]['density_gate']>=rows[1]['density_gate']
    return rows


def cispro_search(bases,lifts,weights,areas,gates,radii,meta,levels,out):
    topology=PeriodicTriangleTopology(bases,lifts,nvertices=len(weights),
        geometry_certificate=meta['geometry_certificate'])
    Linner,Louter=[x['original_count_density_gate'] for x in levels]
    inner=gates>=Linner;outer=gates>=Louter
    inner_top=topology.region(np.flatnonzero(inner),component_labels=True)
    labels=inner_top.pop('triangle_component_labels')
    outer_top=topology.region(np.flatnonzero(outer))
    core_ids=[np.flatnonzero(labels==i) for i in range(inner_top['betti'][0])]
    core_population=[]
    for ids in core_ids:
        vertices=np.unique(bases[ids]);core_population.append(int(weights[vertices].sum()))
    # Below this bound, at least one entire protected inner triangle is lost.
    # Such an event cannot satisfy the policy, so skipping it is exact pruning.
    lower=float(radii[inner].max())
    finite=np.unique(radii[radii>=lower]);records=[]
    previous_inner=None;previous_outer=None
    for alpha in [*map(float,finite),'Inf']:
        value=np.inf if alpha=='Inf' else alpha
        eligible=radii<=value;selected_inner=inner&eligible;selected_outer=outer&eligible
        assert np.array_equal(selected_inner,inner)
        if previous_outer is not None:
            assert not np.any(previous_outer&~selected_outer)
        previous_inner=selected_inner;previous_outer=selected_outer
        measured=topology.region(np.flatnonzero(selected_outer),component_labels=True)
        outerlabels=measured.pop('triangle_component_labels')
        covered=np.zeros(len(weights),bool);covered[bases[selected_outer].ravel()]=True
        mapped=[np.unique(outerlabels[ids]).astype(int).tolist() for ids in core_ids]
        row=dict(alpha_deg=alpha,inner_preserved=True,
            outer_construction_count=int(weights[covered].sum()),
            outer_required_count=levels[1]['required_count'],
            protected_core_outer_labels=mapped,outer_exact_betti=measured['betti'],
            outer_triangle_count=int(selected_outer.sum()),outer_area_deg2=float(areas[selected_outer].sum()),
            inner_triangle_count=int(inner.sum()),inner_area_deg2=float(areas[inner].sum()))
        row['rejection_reasons']=cispro_rejection_reasons(row)
        records.append(row)
    decision=select_cispro_event(records,baseline_inner_betti=inner_top['betti'])
    decision.update(baseline_inner_topology=inner_top,baseline_outer_topology=outer_top,
        protected_core_construction_weights=core_population,
        protected_core_definition='Entire exact connected components of the alpha=infinity 98% whole-triangle region',
        baseline_inner_triangle_count=int(inner.sum()),baseline_outer_triangle_count=int(outer.sum()),
        minimum_radius_preserving_inner_deg=lower,
        all_unique_radius_events=int(len(np.unique(radii))),evaluated_events=len(records),
        skipped_lower_events_reason='Cannot preserve all baseline inner triangles',
        thresholds_fixed_during_search=True,candidate_regions_exactly_nested=True,
        topology_counts_vertex_contacts_and_periodic_seams=True,seed_coordinates_used=False)
    if decision['selection'] is not None:
        selected=decision['selection'];a=selected['alpha_deg']
        nextrows=[r for r in records if a!='Inf' and (r['alpha_deg']=='Inf' or r['alpha_deg']>a)]
        decision['first_larger_event']=nextrows[0] if nextrows else None
        decision['selected_constant_region_interval']={'left_inclusive_deg':a,
            'right_exclusive_deg':nextrows[0]['alpha_deg'] if nextrows else 'Inf'}
    write(out/'events.json',dict(policy=VERSION,records=records))
    np.savez_compressed(out/'protected_inner_components.npz',triangle_component_labels=labels,
        component_observation_weights=np.asarray(core_population),baseline_inner_selected=inner)
    return decision


def run_class(category,out):
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    started=time.monotonic();inputs=Inputs()
    terminal=inputs.json(SOURCE/'terminal.json');assert terminal['status']=='completed'
    plan=inputs.json(SOURCE/'plan.json',terminal['plan_sha256']);case=plan['cases'][category]
    folder=Path(case['geometry_directory']);meta=inputs.json(folder/'metadata.json',plan['sources_and_inputs'][str(folder/'metadata.json')])
    assert meta['geometry_certificate']['certified'] and meta['geometry_sha256']==case['geometry_sha256']
    bases=inputs.array(folder,meta['arrays']['bases'],True)
    weights=inputs.array(folder,meta['arrays']['weights'],True)
    vertices=inputs.array(folder,meta['arrays']['vertices'])
    total=int(weights.sum());assert total==case['observation_count'] and np.all(weights>0)
    rawpath=Path(case['construct_root'])/'sigma0_floor0/report.json';raw=inputs.json(rawpath)
    assert raw['sigma_deg']==0 and raw['area_floor_quantile']==0
    gates=inputs.array(rawpath.parent,raw['arrays']['gates']);radii=inputs.array(Path(case['radii_directory']),case['radii_metadata'])
    assert np.all(gates>0) and np.all(radii>0)
    infinite_heights=inclusion_heights(bases,gates,len(weights))
    levels=construction_levels(infinite_heights,weights)
    out.mkdir(exist_ok=False)
    if category=='CisPro':
        lifts=inputs.array(folder,meta['arrays']['lifts'],True);areas=inputs.array(folder,meta['arrays']['areas'])
        decision=cispro_search(bases,lifts,weights,areas,gates,radii,meta,levels,out)
        del lifts,areas
        alpha=decision['selection']['alpha_deg'] if decision['selection'] is not None else None
    else:
        alpha=default_alpha(category)
        decision=dict(status='selected',selection=dict(alpha_deg=alpha),
            criterion='Explicit empirical default: retain the whole unsmoothed triangle region',
            alpha_optimized_for_this_class=False)
    decision.update(category=category,policy_version=VERSION,alpha_deg=alpha,
        construction_levels=levels,threshold_calibration='Alpha=infinity inclusive exact weighted ranks, held fixed',
        no_smoothing=True,no_kde_read=True,not_a_universal_topology_selector=True,
        candidate_family='Raw minimum-corner whole-triangle superlevels with circumradius cutoff in degrees')
    write(out/'selection.json',decision)
    if alpha is None:
        write(out/'terminal.json',dict(status='unresolved',artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}));return
    eligible=radii<=(np.inf if alpha=='Inf' else alpha)
    triangle_scores=np.where(eligible,gates,0.)
    heights=inclusion_heights(bases,triangle_scores,len(weights))
    for row in levels:
        L=row['original_count_density_gate'];count=int(weights[heights>=L].sum());strict=int(weights[heights>L].sum())
        assert strict<row['required_count']<=count
        selected=eligible&(gates>=L);covered=np.zeros(len(weights),bool);covered[bases[selected].ravel()]=True
        assert np.array_equal(covered,heights>=L)
        row.update(alpha_deg=alpha,construction_count=count,strictly_above_count=strict,
            construction_coverage=count/total,selected_triangle_count=int(selected.sum()),
            sigma_deg=0,score_units='Raw count density divided by observation count',
            finite_level_still_largest_feasible=True)
    np.savez_compressed(out/'vertex_heights.npz',vertices=vertices,weights=weights,heights=heights/total)
    write(out/'levels.json',dict(status='exact_weighted_calibration_at_fixed_infinity_levels',rows=levels))
    selected_triangles=np.asarray([eligible&(gates>=r['original_count_density_gate']) for r in levels])
    np.savez_compressed(out/'selected_triangles.npz',packed_masks=np.packbits(selected_triangles,axis=1,bitorder='little'),
        triangle_count=len(bases),bitorder='little',targets=np.asarray([r['target'] for r in levels]))
    checks=[]
    for n in GRIDS:
        gp=Path(case['grids'][str(n)]);gm=inputs.json(gp/'incidences.json')
        for key in ('offsets','indices'):inputs.path(gp/gm[key]['path'],gm[key]['sha256'])
        offsets,indices=_load_csr(ROOT/gp,n,len(bases),case['geometry_sha256'])
        starts=offsets[:-1].astype(np.int64)
        score=np.maximum.reduceat(triangle_scores[indices],starts).reshape(n,n)/total
        support=np.logical_or.reduceat(eligible[indices],starts).reshape(n,n)
        assert np.array_equal(score>0,support)
        packed=[];baseline_packed=[];local=[]
        for row,selection in zip(levels,selected_triangles):
            mask=score>=row['density_gate']
            direct=np.logical_or.reduceat(selection[indices],starts).reshape(n,n)
            assert np.array_equal(mask,direct)
            baseline=np.logical_or.reduceat((gates>=row['original_count_density_gate'])[indices],starts).reshape(n,n)
            assert not np.any(mask&~baseline)
            if alpha=='Inf':assert np.array_equal(mask,baseline)
            if category=='CisPro' and row['target']==.98:assert np.array_equal(mask,baseline)
            packed.append(np.packbits(mask.ravel(),bitorder='little'))
            baseline_packed.append(np.packbits(baseline.ravel(),bitorder='little'))
            local.append(dict(target=row['target'],mask_cells=int(mask.sum()),baseline_cells=int(baseline.sum()),
                changed_cells=int(np.count_nonzero(mask!=baseline)),score_matches_closed_union=True,
                baseline_subset_verified=True))
        np.save(out/f'score_{n}.npy',score);np.save(out/f'eligible_{n}.npy',support)
        np.savez_compressed(out/f'masks_{n}.npz',packed_masks=np.asarray(packed),
            baseline_packed_masks=np.asarray(baseline_packed),grid_size=n,targets=np.asarray([r['target'] for r in levels]),
            bitorder='little',mask_axes=np.asarray(['psi','phi']))
        checks.append(dict(grid_size=n,rows=local))
        del offsets,indices,starts,score,support,mask,direct,baseline
    report=dict(status='completed',category=category,alpha_deg=alpha,policy_version=VERSION,
        no_smoothing=True,no_kde_read=True,thresholds_fixed=True,observation_count=total,
        vertex_count=len(weights),triangle_count=len(bases),geometry_sha256=case['geometry_sha256'],
        grids=checks,sources_and_inputs=inputs.pins,source_sha256=sha(__file__),
        elapsed_seconds=time.monotonic()-started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'report.json',report)
    write(out/'terminal.json',dict(status='completed',artifacts={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(category+': alpha='+str(alpha)+'; exact calibration and raster fields verified',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--class-name',choices=CLASSES);args=parser.parse_args();out=ROOT/args.out
    if args.class_name:run_class(args.class_name,out/args.class_name);return
    out.mkdir(exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('rama_unsmoothed_alpha_policy.py'),
        Path(__file__).with_name('test_rama_unsmoothed_alpha_policy.py'),Path(__file__).with_name('rama_compact_triangle_topology.py'),
        ROOT/'benchmark/rama_construction_alpha_pilot.py']
    write(out/'plan.json',dict(status='declared_before_execution',policy_version=VERSION,
        classes=CLASSES,grids=GRIDS,no_smoothing=True,no_kde=True,
        rule='Infinity for five classes. CisPro largest exact radius preserving entire baseline 98% region, target 99.5% coverage, two separate protected inner components and outer Betti (2,0,0). Thresholds fixed at infinity-calibrated values.',
        scope='Empirical class-specific policy for the present two-dimensional six-category system; unresolved if assumptions fail',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    for category in ('CisPro','PrePro','TransPro','Gly','IleVal','General'):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/matplotlib-dtfe-topology')
        subprocess.run([sys.executable,__file__,'--out',str(out),'--class-name',category],cwd=ROOT,env=env,check=True,timeout=600)
    decisions={c:json.loads((out/c/'selection.json').read_text()) for c in CLASSES}
    write(out/'all_decisions.json',dict(policy_version=VERSION,classes=decisions,
        all_classes_resolved=all(d['status']=='selected' for d in decisions.values()),no_kde_read=True))
    write(out/'terminal.json',dict(status='completed',artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        all_classes_resolved=all(d['status']=='selected' for d in decisions.values()),no_kde_read=True,manuscript_changed=False))
    print('Policy decisions and fields complete and sealed.',flush=True)

if __name__=='__main__':main()
