#!/usr/bin/env python3
"""Exhaust one uniform, fixed-threshold alpha rule on all six raw 2D classes.

Every class uses the same operations, without a class-specific component count:
(1) calibrate infinity whole-triangle 98%/99.5% regions;
(2) retain the whole inner region and sufficient outer construction coverage;
(3) forbid new complement components (new holes);
(4) when a baseline outer component splits, every resulting piece must contain
    a triangle belonging to the protected inner region;
(5) minimize outer area among admissible alpha regions, preferring infinity
    when it produces exactly the same selected triangles.

The regions are nested because density thresholds remain fixed. Hence the
first admissible distinct outer triangle set has the minimum area. Every active
outer circumradius event above the exact feasibility bound is evaluated, plus
infinity. Inactive triangle events cannot change the candidate contour.
This output is a diagnostic for visual assessment, not an automatic policy
promotion. No KDE/reference mask is loaded during selection or construction.
"""
from pathlib import Path
import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
import numpy as np
from run_unsmoothed_alpha_policy import (ROOT, SOURCE, Inputs, construction_levels,
    inclusion_heights, sha, write)
from rama_compact_triangle_topology import PeriodicTriangleTopology
from rama_unsmoothed_alpha_policy import CLASSES
from rama_uniform_alpha_guards import (edge_adjacent_face_pairs,
    background_component_labels, evaluate_topology_guards)

DEFAULT=Path('novak_work/validation_results/uniform_unsmoothed_alpha_guarded_v1')


def run(category,out):
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    started=time.monotonic();inputs=Inputs();terminal=inputs.json(SOURCE/'terminal.json')
    plan=inputs.json(SOURCE/'plan.json',terminal['plan_sha256']);case=plan['cases'][category]
    folder=Path(case['geometry_directory']);meta=inputs.json(folder/'metadata.json',plan['sources_and_inputs'][str(folder/'metadata.json')])
    assert meta['geometry_certificate']['certified'] and case['geometry_sha256']==meta['geometry_sha256']
    bases=inputs.array(folder,meta['arrays']['bases'],True)
    lifts=inputs.array(folder,meta['arrays']['lifts'],True)
    weights=inputs.array(folder,meta['arrays']['weights'],True)
    areas=inputs.array(folder,meta['arrays']['areas'])
    rawpath=Path(case['construct_root'])/'sigma0_floor0/report.json';raw=inputs.json(rawpath)
    assert raw['sigma_deg']==0 and raw['area_floor_quantile']==0
    gates=inputs.array(rawpath.parent,raw['arrays']['gates']);radii=inputs.array(Path(case['radii_directory']),case['radii_metadata'])
    heights=inclusion_heights(bases,gates,len(weights));levels=construction_levels(heights,weights)
    inner=gates>=levels[0]['original_count_density_gate']
    outer=gates>=levels[1]['original_count_density_gate']
    inner_bound=float(radii[inner].max())
    first_radius=np.full(len(weights),np.inf)
    qualifying=np.where(outer,radii,np.inf)
    for corner in range(3):np.minimum.at(first_radius,bases[:,corner],qualifying)
    order=np.argsort(first_radius);cum=np.cumsum(weights[order]);required=levels[1]['required_count']
    coverage_bound=float(first_radius[order[np.searchsorted(cum,required)]])
    assert np.isfinite(coverage_bound)
    alpha_min=max(inner_bound,coverage_bound)
    active_events=np.unique(radii[outer&(radii>=alpha_min)])
    assert len(active_events)>0 and active_events[0]==alpha_min
    del heights,qualifying,order,cum
    gc.collect()
    topology=PeriodicTriangleTopology(bases,lifts,nvertices=len(weights),
        geometry_certificate=meta['geometry_certificate'],max_triangles=3000000)
    face_pairs=edge_adjacent_face_pairs(topology._edge_ids)
    inner_top=topology.region(np.flatnonzero(inner))
    base_outer_top=topology.region(np.flatnonzero(outer),component_labels=True)
    baseline_fg=base_outer_top.pop('triangle_component_labels')
    records=[];selected_mask=None;selected_row=None;previous=None
    for alpha in [*map(float,active_events),'Inf']:
        value=np.inf if alpha=='Inf' else alpha
        selected=outer&(radii<=value)
        assert np.array_equal(inner&(radii<=value),inner)
        if previous is not None:assert not np.any(previous&~selected)
        previous=selected.copy()
        count=int(weights[first_radius<=value].sum());assert count>=required
        measured=topology.region(np.flatnonzero(selected),component_labels=True)
        foreground=measured.pop('triangle_component_labels')
        background=background_component_labels(selected,face_pairs)
        guards=evaluate_topology_guards(inner,outer,selected,baseline_fg,foreground,background)
        row=dict(alpha_deg=alpha,outer_construction_count=count,outer_required_count=required,
            inner_preserved=True,outer_exact_betti=measured['betti'],
            outer_triangle_count=int(selected.sum()),outer_area_deg2=float(areas[selected].sum()),
            guards=guards)
        # The guards module supplies a single explicit conjunction under 'passed'.
        admissible=bool(guards['passed'])
        row['admissible']=admissible
        records.append(row)
        if admissible:
            if selected_row is None:
                selected_row=row;selected_mask=selected.copy()
            elif np.array_equal(selected,selected_mask):
                # Equal geometry: later events are less restrictive, with Inf
                # preferred over finite events that realize the same contour.
                selected_row=row;selected_mask=selected.copy()
        del foreground,background
    assert records[-1]['admissible'],'Infinity must satisfy relative preservation guards'
    assert selected_row is not None
    selected_alpha=selected_row['alpha_deg']
    selected_pair=np.asarray([inner,selected_mask])
    baseline_pair=np.asarray([inner,outer])
    final_heights=inclusion_heights(bases,np.where(radii<=(np.inf if selected_alpha=='Inf' else selected_alpha),gates,0.),len(weights))
    for row,selection in zip(levels,selected_pair):
        L=row['original_count_density_gate'];covered=np.zeros(len(weights),bool);covered[bases[selection].ravel()]=True
        assert np.array_equal(covered,final_heights>=L)
        count=int(weights[covered].sum());strict=int(weights[final_heights>L].sum())
        assert strict<row['required_count']<=count
        row.update(alpha_deg=selected_alpha,construction_count=count,strictly_above_count=strict,
            construction_coverage=count/int(weights.sum()),selected_triangle_count=int(selection.sum()),
            region_area_deg2=float(areas[selection].sum()),baseline_area_deg2=float(areas[outer if row['target']==.995 else inner].sum()),
            fixed_level_is_still_largest_feasible=True)
    out.mkdir(exist_ok=False)
    write(out/'events.json',dict(records=records,active_finite_events=len(active_events)))
    decision=dict(status='selected_diagnostic',category=category,alpha_deg=selected_alpha,
        selection=selected_row,construction_levels=levels,
        objective='Minimum area subject to uniform preservation and topology guards; equal triangle-set ties choose least restrictive evaluated event, with infinity preferred.',
        exact_nested_set_minimum=True,all_active_events_evaluated=True,
        finite_event_count=len(active_events),total_event_count=len(records),
        admissible_event_count=sum(r['admissible'] for r in records),
        minimum_inner_preservation_radius_deg=inner_bound,
        minimum_outer_coverage_radius_deg=coverage_bound,
        minimum_joint_feasible_radius_deg=alpha_min,
        baseline_inner_topology=inner_top,baseline_outer_topology=base_outer_top,
        no_smoothing=True,no_kde_read=True,no_seed_coordinates=True,
        no_class_specific_rule=True,active_policy_changed=False,
        geometry_sha256=case['geometry_sha256'],observation_count=int(weights.sum()),
        triangle_count=len(bases),sources_and_inputs=inputs.pins,
        elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'selection.json',decision)
    write(out/'levels.json',dict(rows=levels,threshold_calibration='Inclusive infinity weighted ranks, held fixed'))
    np.savez_compressed(out/'selected_triangles.npz',packed_masks=np.packbits(selected_pair,axis=1,bitorder='little'),
        baseline_packed_masks=np.packbits(baseline_pair,axis=1,bitorder='little'),
        triangle_count=len(bases),targets=np.asarray([r['target'] for r in levels]),bitorder='little')
    write(out/'terminal.json',dict(status='completed',artifacts={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(f'{category}: alpha {selected_alpha}; {len(active_events)} active events; {sum(r["admissible"] for r in records)} admissible; outer Betti {selected_row["outer_exact_betti"]}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--class-name',choices=CLASSES);args=parser.parse_args();out=ROOT/args.out
    if args.class_name:run(args.class_name,out/args.class_name);return
    out.mkdir(exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('run_unsmoothed_alpha_policy.py'),
        Path(__file__).with_name('rama_unsmoothed_alpha_policy.py'),Path(__file__).with_name('rama_compact_triangle_topology.py'),
        Path(__file__).with_name('rama_uniform_alpha_guards.py'),Path(__file__).with_name('test_rama_uniform_alpha_guards.py')]
    write(out/'plan.json',dict(status='declared_before_execution',classes=CLASSES,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        no_kde_read=True,no_class_specific_rules=True,changes_active_policy=False,
        rule='Preserve all inner triangles and outer construction coverage at infinity-calibrated fixed levels; no new complement components; each piece of a split baseline outer component retains an inner triangle; minimize outer area; prefer infinity for identical geometry.',
        candidate_family='All active outer circumradius events at/above exact inner-preservation and outer-coverage bound, plus infinity.'))
    for category in ('CisPro','PrePro','TransPro','Gly','IleVal','General'):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        subprocess.run([sys.executable,__file__,'--out',str(out),'--class-name',category],cwd=ROOT,env=env,check=True,timeout=600)
    decisions={c:json.loads((out/c/'selection.json').read_text()) for c in CLASSES}
    write(out/'all_decisions.json',dict(classes=decisions,no_kde_read=True,no_class_specific_rules=True,
        active_policy_changed=False))
    write(out/'terminal.json',dict(status='completed',artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        no_kde_read=True,active_policy_changed=False))

if __name__=='__main__':main()
