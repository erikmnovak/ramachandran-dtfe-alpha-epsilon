#!/usr/bin/env python3
"""Probe density-window cleanup beyond the preceding epsilon search.

This is a diagnostic geometric rule, not Bobrowski et al.'s inferred homology.
No KDE, Jaccard, smoothing, or alpha geometry is read. Fixed original levels
are retained, so changed construction coverage must be reported explicitly.
Additive windows test the original rule. Multiplicative windows explore the
same core/pit logic on log-density: [L/factor, L*factor]. They are a new,
experimental extension, not covered by the paper's additive-no-critical-values
assumption. All output masks are periodic 1536 by 1536 raster displays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import numpy as np
from rama_topology_cleanup_preview import CleanupPreview, periodic_labels, construction_coverage
from rama_periodic_region_metrics import periodic_region_topology

PROJECT=Path(__file__).resolve().parents[1]
FIELDS=PROJECT/'novak_work/validation_results/raw_dtfe_topology_field_v1'
PREVIOUS=PROJECT/'novak_work/validation_results/dtfe_topology_window_v1'
CLASSES=('General','Gly','IleVal','PrePro','TransPro','CisPro')
ADDITIVE=(0.,.49,.6,.75,.9,.99)
FACTORS=(2.,4.,8.,16.,32.,64.,128.)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def multiplicative_cleanup(model,factor):
    """Apply precisely the existing core/pit rule with asymmetric thresholds."""
    high,low=model.level*factor,model.level/factor
    remove_labels=model.peaks<high
    remove_labels[0]=False
    intermediate=model.raw & ~remove_labels[model.foreground]
    background,winding=periodic_labels(~intermediate,4)
    pits=np.full(int(background.max())+1,np.inf)
    np.minimum.at(pits,background.ravel(),model.density.ravel())
    sizes=np.bincount(background.ravel(),minlength=len(pits));sizes[0]=0
    protected=winding.any(axis=1);protected[0]=True
    if sizes.max()>0:protected|=(sizes==sizes.max())&(sizes>0)
    fill_labels=(pits>=low)&~protected
    result=intermediate|fill_labels[background]
    removed=model.raw&~result;filled=~model.raw&result
    if np.any((model.density>=high)&~result) or np.any(result&(model.density<low)):
        raise ArithmeticError('Result escaped multiplicative density window')
    return result,dict(removed_component_count=int(remove_labels.sum()),
        filled_background_component_count=int(fill_labels.sum()),
        removed_cells=int(removed.sum()),filled_cells=int(filled.sum()),
        changed_cells=int(np.count_nonzero(result!=model.raw)),removed=removed,filled=filled)


def band_coverage(vertices,values,weights,level,high,low,edits):
    """Exact vertex-value coverage of the raster-guided continuous extension."""
    n=len(edits['removed'])
    cells=np.floor(((vertices+180.)%360.)*n/360.).astype(np.int64)%n
    x,y=cells[:,0],cells[:,1]
    raw=values>=level
    result=(raw&~(edits['removed'][y,x]&(values<high)))|(edits['filled'][y,x]&(values>=low))
    return dict(raw_count=int(weights[raw].sum()),after_count=int(weights[result].sum()),
        total_count=int(weights.sum()),changed_count=int(weights[raw!=result].sum()))


def analyse(category,output):
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
    signal.alarm(900)
    started=time.monotonic()
    folder=FIELDS/category;destination=output/category
    destination.mkdir(parents=True,exist_ok=True)
    paths=[folder/'terminal.json',folder/'mesh.npz',folder/'density_1536.npy',folder/'levels.json',PREVIOUS/category/'pl_persistence.npz']
    pins={str(p.relative_to(PROJECT)):sha(p) for p in paths}
    terminal=json.loads((folder/'terminal.json').read_text())
    assert terminal['status']=='completed'
    for p in paths[1:4]:
        assert terminal['artifacts'][p.name]==pins[str(p.relative_to(PROJECT))]
    levels=json.loads((folder/'levels.json').read_text())['rows']
    density=np.load(folder/'density_1536.npy',mmap_mode='r')
    with np.load(folder/'mesh.npz',allow_pickle=False) as z:
        vertices,values,weights=z['vertices'],z['vertex_density'],z['weights']
    with np.load(PREVIOUS/category/'pl_persistence.npz',allow_pickle=False) as z:
        diagrams=[z[f'H{k}'] for k in range(3)]
    rows=[];peak_rows=[];packed=[]
    for level in levels:
        L=level['density_gate'];model=CleanupPreview(density,L)
        peaks=model.peaks[1:]/L
        peak_rows.append(dict(target=level['target'],component_count=len(peaks),
            peak_over_level_quantiles={str(q):float(np.quantile(peaks,q)) for q in (0,.1,.25,.5,.75,.9,.99,1)},
            peak_at_least_factor={str(f):int((peaks>=f).sum()) for f in (1.49,2,4,8,16,32,64,128)},
            additive_epsilon_below_L_cannot_remove_count=int((peaks>=2.).sum())))
        settings=[('additive',t) for t in ADDITIVE]+[('multiplicative',f) for f in FACTORS]
        for mode,setting in settings:
            if mode=='additive':
                high,low=L*(1+setting),L*(1-setting)
                result,edits=model.apply(setting*L)
                coverage=construction_coverage(vertices,values,weights,L,setting*L,edits)
            else:
                high,low=L*setting,L/setting
                result,edits=multiplicative_cleanup(model,setting)
                coverage=band_coverage(vertices,values,weights,L,high,low,edits)
            assert coverage['raw_count']==level['construction_count']
            # Counts the paper-inspired image on the exact PL persistence, not
            # the raster's cleaned geometry. They answer different questions.
            image=[int(((d[:,0]<=-high)&(d[:,1]>-low)).sum()) for d in diagrams]
            row=dict(category=category,target=level['target'],mode=mode,setting=setting,
                density_gate=L,high=high,low=low,raster_betti=periodic_region_topology(result)['betti'],
                pl_image_ranks=image,area_deg2=float(result.sum()*(360/len(result))**2),
                **coverage,coverage=coverage['after_count']/coverage['total_count'],
                **{k:v for k,v in edits.items() if k not in ('removed','filled')})
            rows.append(row);packed.append(np.packbits(result.ravel(),bitorder='little'))
            print(category,f"{100*level['target']:g}%",mode,setting,'beta',row['raster_betti'],
                'coverage',f"{100*row['coverage']:.4f}%",flush=True)
    # Every setting shares one factor/relative epsilon across both target levels.
    count=len(ADDITIVE)+len(FACTORS)
    nesting=[]
    for i in range(count):
        inner=np.unpackbits(packed[i],bitorder='little').astype(bool)
        outer=np.unpackbits(packed[count+i],bitorder='little').astype(bool)
        nesting.append(dict(mode=rows[i]['mode'],setting=rows[i]['setting'],
                            inner_outside_outer_cells=int((inner&~outer).sum())))
    np.savez_compressed(destination/'masks.npz',packed_masks=np.asarray(packed),grid_size=len(density),bitorder='little')
    write(destination/'results.json',dict(status='completed',category=category,rows=rows,
        peaks=peak_rows,nesting_checks=nesting,input_sha256=pins,
        elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        KDE_access=False,no_smoothing=True,no_alpha=True))
    for path,digest in pins.items():assert sha(PROJECT/path)==digest
    write(destination/'terminal.json',dict(status='completed',
        artifact_sha256={p.name:sha(p) for p in (destination/'results.json',destination/'masks.npz')}))
    signal.alarm(0)


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).resolve(),PROJECT/'novak_work/rama_topology_cleanup_preview.py',PROJECT/'novak_work/rama_periodic_region_metrics.py']
    plan=dict(additive_relative_epsilon=ADDITIVE,multiplicative_factors=FACTORS,
        classes=CLASSES,selection='Prespecified diagnostic sweep; no KDE or Jaccard',
        multiplicative_scope='Exploratory log-density core/pit cleanup, not paper theorem',
        source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in sources})
    write(output/'plan.json',plan)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for category in CLASSES:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'class','--out',str(output),'--category',category],
            check=True,env=env,timeout=930)
    allrows=[r for c in CLASSES for r in json.loads((output/c/'results.json').read_text())['rows']]
    write(output/'results.json',dict(status='completed',rows=allrows))
    for path,digest in plan['source_sha256'].items():assert sha(PROJECT/path)==digest
    write(output/'terminal.json',dict(status='completed',
        artifact_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='terminal.json'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['run','class'])
    parser.add_argument('--category',choices=CLASSES)
    parser.add_argument('--out',type=Path,default=PROJECT/'novak_work/validation_results/dtfe_extended_epsilon_probe_v1')
    args=parser.parse_args()
    if args.stage=='run':main(args.out.resolve())
    else:analyse(args.category,args.out.resolve())
