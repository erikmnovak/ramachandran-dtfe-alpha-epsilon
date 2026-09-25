#!/usr/bin/env python3
"""Enumerate every represented cleanup geometry change for fixed adopted alpha.

The class, alpha and construction levels are never tuned in this experiment.
The existing density-ratio cleanup is swept over Float64 k in [1,64]. A tested
cache enumerates the actual first representable k at which a strict deletion
or inclusive filling predicate changes, including multiplication/division
roundoff. Thus evaluating those boundaries covers the constant predicate
intervals rather than merely sampling an evenly spaced list of ratios.

Adjacent states with identical full Boolean-mask SHA256 are merged into true
GEOMETRY plateaus, not just equal Betti-count plateaus. A plateau meeting k=64
is right-censored: no assertion is made about its continuation beyond the
observed domain. This runner chooses no plateau, uses no KDE reference, and
never changes or recalibrates the starting density levels after cleanup.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import numpy as np
from rama_alpha_window_cleanup import AlphaWindowCleanup
from rama_alpha_cleanup_events import CachedAlphaCleanup
from rama_periodic_region_metrics import periodic_region_topology

ROOT=Path(__file__).resolve().parents[1]
FIELDS=ROOT/'novak_work/validation_results/uniform_alpha_score_fields_v1'
DEFAULT=ROOT/'novak_work/validation_results/uniform_alpha_epsilon_events_v1'
CLASSES=('General','Gly','TransPro','CisPro','PrePro','IleVal')
GRIDS=(768,1536)
DOMAIN=(1.,64.)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')


def scalar_edits(edits):
    return {k:(v.item() if isinstance(v,np.generic) else v) for k,v in edits.items()
            if not isinstance(v,np.ndarray)}


def file_check(path,terminal):
    digest=sha(path)
    assert digest==terminal['artifacts'][str(path.relative_to(FIELDS))],str(path)
    return digest


def summarize_mask(mask,raw,point_rows,point_cols,weights):
    n=len(mask);inside=mask[point_rows,point_cols];baseline=raw[point_rows,point_cols]
    count=int(weights[inside].sum());total=int(weights.sum())
    return dict(raster_betti=periodic_region_topology(mask)['betti'],
        area_deg2=float(mask.sum()*(360/n)**2),mask_cells=int(mask.sum()),
        raster_count=count,raster_coverage=count/total,
        raw_raster_count=int(weights[baseline].sum()),
        lost_count=int(weights[baseline&~inside].sum()),
        gained_count=int(weights[~baseline&inside].sum()),
        removed_cells=int(np.count_nonzero(raw&~mask)),
        filled_cells=int(np.count_nonzero(~raw&mask)),
        total_count=total)


def component_record(model,point_rows,point_cols,weights,heights):
    """Record raw components for subsequent feature audit without KDE labels."""
    lab=model.foreground;count=len(model.peaks)
    sizes=np.bincount(lab.ravel(),minlength=count)
    assignments=lab[point_rows,point_cols]
    population=np.bincount(assignments,weights=weights,minlength=count).astype(np.int64)
    exactinside=heights>=model.level
    insidepop=np.bincount(assignments[exactinside],weights=weights[exactinside],minlength=count).astype(np.int64)
    rows=[]
    for i in range(1,count):
        rows.append(dict(component=i,pixel_count=int(sizes[i]),area_deg2=float(sizes[i]*(360/len(lab))**2),
            peak_density=float(model.peaks[i]),peak_over_level=float(model.peaks[i]/model.level),
            sampled_observation_weight=int(population[i]),
            sampled_exact_inside_observation_weight=int(insidepop[i])))
    return dict(rows=rows,component_population_convention='Containing raster cell; not exact closed-triangle component assignment',
        exact_construction_count=int(weights[exactinside].sum()),
        exact_inside_but_sampled_background_count=int(insidepop[0]))


def sweep_level(category,n,level,score,eligible,vertices,weights,heights,out):
    start=time.monotonic();L=level['density_gate'];q=level['target']
    point_cells=np.floor(((vertices+180.)%360.)*n/360.).astype(np.int64)%n
    cols,rows=point_cells.T
    model=AlphaWindowCleanup(score,L,eligible)
    assert model.raw.any() and not model.raw.all()
    assert int(weights[heights>=L].sum())==level['construction_count']
    cached=CachedAlphaCleanup(model)
    boundaries=[float(k) for k in cached.events(DOMAIN)]
    assert boundaries[0]==1. and boundaries[-1]==64. and all(a<b for a,b in zip(boundaries,boundaries[1:]))
    initial=component_record(model,rows,cols,weights,heights)
    print(f'{category} n={n} q={q}: {len(boundaries)} predicate boundaries',flush=True)
    masks=[];first_seen=[];metrics_cache={};plateaus=[];evaluations=[]
    critical_indices=sorted(set([1,len(boundaries)//2,max(0,len(boundaries)-2)]))
    validation_ks={1.,16.,64.}
    for i in critical_indices:
        if 0<i<len(boundaries)-1:
            k=boundaries[i];validation_ks.add(k)
            validation_ks.add(float(np.nextafter(k,-np.inf)))
    validations=[]
    for i,k in enumerate(boundaries):
        mask,edits=cached.apply(k)
        assert mask.shape==(n,n) and mask.dtype==bool
        assert not np.any(mask&~eligible)
        if k==1:assert np.array_equal(mask,model.raw)
        fingerprint=hashlib.sha256(mask.tobytes(order='C')).hexdigest()
        packed=np.packbits(mask.ravel(),bitorder='little')
        if fingerprint not in metrics_cache:
            index=len(masks);masks.append(packed);first_seen.append(k)
            metrics=summarize_mask(mask,model.raw,rows,cols,weights)
            metrics.update(mask_index=index,mask_sha256=fingerprint,
                packed_mask_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
            metrics_cache[fingerprint]=metrics
        else:
            metrics=metrics_cache[fingerprint]
            assert np.array_equal(packed,masks[metrics['mask_index']])
        if k in validation_ks:
            original,old_edits=model.apply(k)
            assert np.array_equal(mask,original)
            assert scalar_edits(edits)==scalar_edits(old_edits)
            validations.append(dict(k=k,at_boundary=True,exact_mask_match=True,all_scalar_edits_match=True))
        evaluation=dict(k=k,high_density_threshold=L*k,low_density_threshold=L/k,
            mask_index=metrics['mask_index'],mask_sha256=fingerprint,
            edits=scalar_edits(edits))
        evaluations.append(evaluation)
        right=boundaries[i+1] if i+1<len(boundaries) else None
        censored=right is None
        if plateaus and plateaus[-1]['mask_sha256']==fingerprint:
            state=plateaus[-1]
            state.update(right_exclusive_k=right,right_censored=censored,
                observed_interval_right_k=(DOMAIN[1] if censored else right),
                last_observed_k=k,last_event_index=i)
            state['predicate_event_count']+=1
        else:
            plateaus.append(dict(**metrics,left_inclusive_k=k,right_exclusive_k=right,
                right_censored=censored,observed_interval_right_k=(DOMAIN[1] if censored else right),
                last_observed_k=k,representative_k=k,
                first_event_index=i,last_event_index=i,predicate_event_count=1,
                high_density_threshold_at_left=L*k,low_density_threshold_at_left=L/k,
                edits_at_left=scalar_edits(edits),
                exact_starting_construction_count=level['construction_count']))
        if i and i%250==0:
            print(f'{category} n={n} q={q}: evaluated {i}/{len(boundaries)} boundaries, {len(plateaus)} geometry plateaus',flush=True)
    # Validate requested non-event controls and just-below tie locations after
    # the sweep, without requiring the cache to hold all previous partitions.
    done={v['k'] for v in validations}
    for k in sorted(validation_ks-done):
        observed,edits=cached.apply(k);original,old_edits=model.apply(k)
        assert np.array_equal(observed,original)
        assert scalar_edits(edits)==scalar_edits(old_edits)
        validations.append(dict(k=k,at_boundary=(k in boundaries),exact_mask_match=True,all_scalar_edits_match=True))
    # For each plateau, each predicate state has the same stored exact geometry.
    for state in plateaus:
        assert all(evaluations[i]['mask_sha256']==state['mask_sha256']
            for i in range(state['first_event_index'],state['last_event_index']+1))
        state['observed_log2_width']=float(np.log2(state['observed_interval_right_k']/state['left_inclusive_k']))
        state['positive_observed_width']=bool(state['observed_interval_right_k']>state['left_inclusive_k'])
    assert plateaus[0]['left_inclusive_k']==1 and plateaus[-1]['right_censored']
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'masks.npz',packed_masks=np.asarray(masks),first_seen_k=np.asarray(first_seen),
        grid_size=n,target=q,bitorder='little',mask_axes=np.asarray(['psi','phi']))
    write(out/'initial_components.json',initial)
    # Event-details provenance is supplied by the independently tested cache.
    event_details=getattr(cached,'event_details',None)
    write(out/'events.json',dict(status='completed',category=category,target=q,grid_size=n,
        alpha_deg=level['alpha_deg'],density_gate=L,ratio_domain_inclusive=list(DOMAIN),
        exact_starting_construction_count=level['construction_count'],total_observations=int(weights.sum()),
        threshold_recalibrated_after_cleanup=False,no_kde_read=True,no_epsilon_selected=True,
        predicate_boundary_count=len(boundaries),geometry_plateau_count=len(plateaus),
        unique_geometry_count=len(masks),states=plateaus,event_evaluations=evaluations,
        event_details=event_details,validation_against_original_apply=validations,
        raster_population_convention='Containing half-open raster cells; report separately from exact starting closed-triangle construction coverage.',
        mask_hash_convention='SHA256 of complete dense C-order Boolean array bytes; packed file bitorder is little.',
        plateau_convention='Left-inclusive/right-exclusive between actual Float64 predicate onsets. A final state is only observed through k=64 and is right-censored, never asserted stable beyond64.',
        elapsed_seconds=time.monotonic()-start))
    write(out/'terminal.json',dict(status='completed',artifacts={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(f'{category} n={n} q={q}: sealed {len(plateaus)} plateaus / {len(masks)} unique masks in {time.monotonic()-start:.1f}s',flush=True)
    return dict(grid_size=n,target=q,predicate_boundaries=len(boundaries),geometry_plateaus=len(plateaus),
        unique_geometries=len(masks),elapsed_seconds=time.monotonic()-start)


def run_class(category,out):
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    started=time.monotonic();terminal=json.loads((FIELDS/'terminal.json').read_text())
    assert terminal['status']=='completed';folder=FIELDS/category
    pins={}
    for name in ('levels.json','vertex_heights.npz'):
        pins[name]=file_check(folder/name,terminal)
    levels=json.loads((folder/'levels.json').read_text())['rows']
    with np.load(folder/'vertex_heights.npz',allow_pickle=False) as z:
        vertices,weights,heights=z['vertices'],z['weights'],z['heights']
    rows=[]
    for n in GRIDS:
        for name in (f'score_{n}.npy',f'eligible_{n}.npy'):
            pins[name]=file_check(folder/name,terminal)
        score=np.load(folder/f'score_{n}.npy',mmap_mode='r')
        eligible=np.load(folder/f'eligible_{n}.npy',mmap_mode='r')
        for level in levels:
            label=f'target_{round(1000*level["target"]):03d}'
            rows.append(sweep_level(category,n,level,score,eligible,vertices,weights,heights,
                out/f'grid_{n}'/label))
            gc.collect()
        del score,eligible
    write(out/'report.json',dict(status='completed',category=category,rows=rows,
        source_field_artifact_sha256=pins,elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        no_kde_read=True,no_epsilon_selected=True))
    write(out/'terminal.json',dict(status='completed',artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--class-name',choices=CLASSES);args=parser.parse_args();out=args.out.resolve()
    if args.class_name:run_class(args.class_name,out/args.class_name);return
    out.mkdir(exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('rama_alpha_cleanup_events.py'),
        Path(__file__).with_name('test_rama_alpha_cleanup_events.py'),
        Path(__file__).with_name('rama_alpha_window_cleanup.py'),
        Path(__file__).with_name('rama_topology_cleanup_preview.py'),
        Path(__file__).with_name('rama_periodic_region_metrics.py')]
    write(out/'plan.json',dict(status='declared_before_execution',classes=CLASSES,grids=GRIDS,
        ratio_domain_inclusive=list(DOMAIN),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        fields_terminal_sha256=sha(FIELDS/'terminal.json'),no_kde_read=True,no_epsilon_selected=True,
        enumeration='Every actual Float64 foreground-deletion or initial-background-pit-filling onset; merge adjacent identical dense masks into geometry plateaus.',
        memory_limit_bytes=2*1024**3,area_and_population_are_diagnostics_not_selection_constraints=True))
    for category in ('CisPro','PrePro','TransPro','Gly','IleVal','General'):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        subprocess.run([sys.executable,__file__,'--out',str(out),'--class-name',category],cwd=ROOT,env=env,check=True,timeout=1800)
    reports={c:json.loads((out/c/'report.json').read_text()) for c in CLASSES}
    write(out/'all_reports.json',dict(status='completed',classes=reports,no_kde_read=True,no_epsilon_selected=True))
    write(out/'terminal.json',dict(status='completed',artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        no_kde_read=True,no_epsilon_selected=True,alpha_and_density_thresholds_unchanged=True))
    print('All six exact-event sweeps completed and sealed.',flush=True)

if __name__=='__main__':main()
