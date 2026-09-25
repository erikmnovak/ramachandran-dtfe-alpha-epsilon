#!/usr/bin/env python3
"""Apply whole-triangle alpha first, then the experimental ratio cleanup.

The score fields and exact weighted construction levels are prepared by
select_unsmoothed_alpha_for_cleanup.py. Alpha is freshly selected without
smoothing. If the strict selector is unresolved, the unconstrained overlap
best is shown only as an explicitly labelled diagnostic, never a selection.
The present run keeps the candidate radii and construction levels fixed. Cleanup may remove components or fill shallow holes, but
never fills any background component touching the alpha-excluded domain.
Published reference masks are loaded only after all final regions are saved.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rama_alpha_window_cleanup import AlphaWindowCleanup
from rama_periodic_region_metrics import periodic_region_topology

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'novak_work/validation_results'
SELECTOR=BASE/'unsmoothed_alpha_selection_for_cleanup_v1'
FIELDS=SELECTOR
REFERENCE=BASE/'internal_alpha_unfloored_assessment_v1'
DEFAULT=BASE/'raw_alpha_then_epsilon_v1'
CLASSES=('General','Gly','TransPro','CisPro','PrePro','IleVal')
TARGETS=(.98,.995)
FACTOR=16.
REGION=ListedColormap(['white','#2e80aa'])
KDE=ListedColormap(['white','#cdcdcd'])
COLORS=['white','#e0e0e0','#91c3de','#f3bc94']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pin(path,pins):
    pins[str(path.relative_to(ROOT))]=sha(path)
    return path


def measure(mask,raw,vertices,weights):
    n=len(mask)
    cells=np.floor(((vertices+180.)%360.)*n/360.).astype(np.int64)%n
    x,y=cells.T;inside=mask[y,x];before=raw[y,x];total=int(weights.sum())
    return dict(raster_betti=periodic_region_topology(mask)['betti'],
                raster_count=int(weights[inside].sum()),raster_coverage=float(weights[inside].sum()/total),
                raw_raster_count=int(weights[before].sum()),total_count=total,
                lost_count=int(weights[before&~inside].sum()),gained_count=int(weights[~before&inside].sum()),
                area_deg2=float(mask.sum()*(360/n)**2),
                changed_area_deg2=float(np.count_nonzero(mask!=raw)*(360/n)**2))


def construct(category,output,pins):
    selection=json.loads(pin(SELECTOR/category/'selection.json',pins).read_text())
    accepted=selection['status']=='selected'
    chosen=selection['selection'] if accepted else selection['unconstrained_diagnostic']
    if chosen is None:raise ValueError(f'{category}: no complete candidate even for a diagnostic')
    folder=FIELDS/category/('selected_fields' if accepted else 'diagnostic_fields')
    field_terminal=json.loads(pin(folder/'terminal.json',pins).read_text())
    for name,digest in field_terminal['artifacts'].items():assert sha(folder/name)==digest,name
    destination=output/category;destination.mkdir()
    levels=json.loads(pin(folder/'levels.json',pins).read_text())['rows']
    with np.load(pin(folder/'vertex_heights.npz',pins),allow_pickle=False) as z:
        vertices,weights,heights=z['vertices'],z['weights'],z['heights']
    alpha=chosen['alpha_deg']
    packed=[];rows=[];audit=[];fine_masks=[]
    for n in (1536,768):
        score=np.load(pin(folder/f'score_{n}.npy',pins),mmap_mode='r')
        eligible=np.load(pin(folder/f'eligible_{n}.npy',pins),mmap_mode='r')
        assert score.shape==eligible.shape==(n,n) and eligible.dtype==bool
        with np.load(pin(SELECTOR/category/f'candidates_{n}.npz',pins),allow_pickle=False) as z:
            ci=z['methods'].tolist().index(chosen['selected_id'])
            assert z['defined'][ci].all()
            selected_masks=np.unpackbits(z['packed_masks'][ci],axis=1,bitorder='little',count=n*n).reshape(2,n,n).astype(bool)
        pair=[]
        for qi,level in enumerate(levels):
            q=level['target'];L=level['density_gate']
            exact=int(weights[heights>=L].sum())
            assert exact==level['construction_count']
            model=AlphaWindowCleanup(score,L,eligible)
            assert np.array_equal(model.raw,selected_masks[qi]),(category,n,q,'selector mask mismatch')
            identity,_=model.apply(1.)
            assert np.array_equal(identity,model.raw)
            final,edits=model.apply(FACTOR)
            assert not np.any(final&~eligible)
            pair.append(final)
            metrics=measure(final,model.raw,vertices,weights)
            if n==1536:
                for stage,mask in [('alpha_only',model.raw),('alpha_then_epsilon',final)]:
                    row=dict(category=category,target=q,stage=stage,alpha_deg=alpha,
                             selection_status=selection['status'],alpha_is_selected=accepted,
                             candidate_id=chosen['selected_id'],
                             density_gate=L,density_ratio=1. if stage=='alpha_only' else FACTOR,
                             exact_alpha_stage_count=exact,exact_alpha_stage_coverage=exact/int(weights.sum()),
                             **measure(mask,model.raw,vertices,weights))
                    if stage=='alpha_then_epsilon':
                        row['cleanup_details']={k:v for k,v in edits.items() if not isinstance(v,np.ndarray)}
                    rows.append(row);fine_masks.append(mask.copy())
                    packed.append(np.packbits(mask.ravel(),bitorder='little'))
            else:
                fine=fine_masks[qi*2+1]
                expanded=np.repeat(np.repeat(final,2,axis=0),2,axis=1)
                audit.append(dict(target=q,grid_size=n,**metrics,
                                  symmetric_difference_deg2=float(np.count_nonzero(expanded!=fine)*(360/1536)**2)))
            del model
        assert not np.any(pair[0]&~pair[1]),(category,n)
        del score,eligible,pair
        gc.collect()
    np.savez_compressed(destination/'masks.npz',packed_masks=np.asarray(packed),grid_size=1536,
                        mask_axes=np.asarray(['psi','phi']),bitorder='little')
    write(destination/'construction.json',dict(rows=rows,nesting_passed=True,alpha_exclusions_preserved=True))
    write(destination/'resolution.json',audit)
    print(category+': alpha then ratio cleanup constructed',flush=True)
    return rows


def unpack(path):
    with np.load(path,allow_pickle=False) as z:
        n=int(z['grid_size'])
        return np.unpackbits(z['packed_masks'],axis=1,bitorder='little',count=n*n).reshape(-1,n,n).astype(bool)


def outline(ax,mask):
    n=len(mask);axis=-180+(np.arange(-1,n+1)+.5)*360/n
    if mask.any() and not mask.all():
        ax.contour(axis,axis,np.pad(mask,1,mode='wrap'),levels=[.5],colors=['#303030'],linewidths=.75)


def style(ax,title):
    ax.set(xlim=(-180,180),ylim=(-180,180),aspect='equal',xticks=[-180,0,180],yticks=[-180,0,180],
           xlabel='φ (degrees)',ylabel='ψ (degrees)')
    ax.set_title(title,fontsize=11,pad=10)


def image_panel(ax,mask,title,kde=False):
    ax.imshow(mask,origin='lower',extent=(-180,180,-180,180),interpolation='nearest',
              cmap=KDE if kde else REGION,vmin=0,vmax=1)
    if kde:outline(ax,mask)
    style(ax,title)


def difference(ax,mask,reference,title):
    arr=np.zeros(mask.shape,np.uint8);arr[mask&reference]=1;arr[mask&~reference]=2;arr[reference&~mask]=3
    ax.imshow(arr,origin='lower',extent=(-180,180,-180,180),interpolation='nearest',
              cmap=ListedColormap(COLORS),vmin=0,vmax=3)
    outline(ax,reference);style(ax,title)


def legend(fig):
    fig.legend(handles=[Patch(facecolor=COLORS[1],label='Shared region'),
        Patch(facecolor=COLORS[2],label='Alpha + cleanup only'),Patch(facecolor=COLORS[3],label='Published KDE only'),
        Line2D([],[],color='#303030',label='Published KDE boundary')],loc='lower center',
        bbox_to_anchor=(.5,.025),ncol=4,frameon=False,fontsize=10)


def render(output,rows,pins):
    reference_terminal=json.loads(pin(REFERENCE/'terminal.json',pins).read_text())
    assert reference_terminal['status']=='completed'
    cases={};assessment=[]
    for c in CLASSES:
        masks=unpack(output/c/'masks.npz')
        rpath=REFERENCE/c/'grid_1536_phase0.npz'
        assert sha(rpath)==reference_terminal['artifacts'][str(Path(c)/rpath.name)]
        with np.load(pin(rpath,pins),allow_pickle=False) as z:
            assert z['targets'].tolist()==list(TARGETS) and z['mask_axes'].tolist()==['psi','phi']
            assert str(z['bitorder'])=='little'
            control=z['controls'].tolist().index('published_reference');assert z['defined'][control].all()
            reference=np.unpackbits(z['packed_masks'][control],axis=1,bitorder='little',count=1536**2).reshape(2,1536,1536).astype(bool)
        local=[r for r in rows if r['category']==c]
        assert len(local)==4
        for i,row in enumerate(local):
            ref=reference[TARGETS.index(row['target'])]
            inter=int((masks[i]&ref).sum());union=int((masks[i]|ref).sum())
            assessment.append(dict(**row,intersection_cells=inter,union_cells=union,jaccard=inter/union))
        measured=assessment[-4:];cases[c]=(masks,reference,measured)
        alpha=local[0]['alpha_deg'];alabel='∞' if alpha=='Inf' else f'{alpha:.4f}°'
        status_label='selected alpha' if local[0]['alpha_is_selected'] else 'diagnostic alpha: selector unresolved'
        fig,axes=plt.subplots(2,3,figsize=(15,11))
        for qi,q in enumerate(TARGETS):
            image_panel(axes[qi,0],reference[qi],'Published Top8000 KDE',kde=True)
            axes[qi,0].set_ylabel(f'{100*q:g}% starting level\nψ (degrees)')
            for mi,(offset,label) in enumerate([(0,'Alpha stage only'),(1,'Alpha then density-ratio cleanup ×16')],1):
                row=measured[2*qi+offset];b0,b1=row['raster_betti'][:2]
                image_panel(axes[qi,mi],masks[2*qi+offset],label+f'\n{b0} components · {b1} loops · raster coverage {100*row["raster_coverage"]:.2f}%\nJaccard = {row["jaccard"]:.3f}')
        fig.suptitle(c+f' — raw alpha then cleanup (α = {alabel})\n{status_label}',fontsize=18,y=.975)
        fig.text(.5,.02,'No smoothing. ×16 tests density contrast, not an angular distance. Alpha exclusions are protected; displayed coverage is measured again.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.065,right=.98,top=.84,bottom=.1,wspace=.28,hspace=.42)
        fig.savefig(output/f'{c}_stages.png',dpi=230,facecolor='white');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11.5,6.2))
        for qi,q in enumerate(TARGETS):
            row=measured[2*qi+1]
            difference(axes[qi],masks[2*qi+1],reference[qi],f'{100*q:g}% starting level · Jaccard = {row["jaccard"]:.3f}')
        fig.suptitle(c+f' — raw alpha + cleanup versus KDE (α = {alabel})\n{status_label}',fontsize=16,y=.97)
        legend(fig);fig.subplots_adjust(left=.075,right=.98,top=.81,bottom=.2,wspace=.23)
        fig.savefig(output/f'{c}_differences.png',dpi=250,facecolor='white');plt.close(fig)
        print(c+': figures rendered',flush=True)
    for qi,q in enumerate(TARGETS):
        for mode in ('final','kde_comparison'):
            fig,axes=plt.subplots(2,3,figsize=(14.5,11.5))
            for ax,c in zip(axes.flat,CLASSES):
                masks,reference,measured=cases[c];row=measured[2*qi+1]
                b0,b1=row['raster_betti'][:2]
                title=c+(' (selected)' if row['alpha_is_selected'] else ' (diagnostic)')+f'\n{b0} components · {b1} loops · raster coverage {100*row["raster_coverage"]:.2f}% · J = {row["jaccard"]:.3f}'
                if mode=='final':image_panel(ax,masks[2*qi+1],title)
                else:difference(ax,masks[2*qi+1],reference[qi],title)
            fig.suptitle(f'Alpha first, then density-ratio cleanup ×16 — {100*q:g}% starting level',fontsize=18,y=.98)
            if mode=='kde_comparison':legend(fig)
            else:fig.text(.5,.035,'Blue is the final region. Alpha was searched without smoothing. “Diagnostic” means the connectivity guard rejected every candidate.',ha='center',fontsize=11)
            fig.subplots_adjust(left=.065,right=.98,top=.87,bottom=.12,wspace=.28,hspace=.35)
            fig.savefig(output/f'all_six_{mode}_{100*q:g}.png',dpi=240,facecolor='white');plt.close(fig)
    write(output/'assessment.json',dict(rows=assessment,reference='published_reference',settings_changed=False))


def main(output):
    fields_terminal=FIELDS/'terminal.json'
    terminal=json.loads(fields_terminal.read_text());assert terminal['status']=='completed'
    for name,digest in terminal['artifacts'].items():
        assert sha(FIELDS/name)==digest,name
    pins={str(fields_terminal.relative_to(ROOT)):sha(fields_terminal)}
    diagnostic_terminal=FIELDS/'diagnostic_fields_terminal.json'
    if diagnostic_terminal.exists():
        d=json.loads(pin(diagnostic_terminal,pins).read_text())
        for name,digest in d['artifacts'].items():assert sha(FIELDS/name)==digest,name
    for p in (Path(__file__).resolve(),ROOT/'novak_work/rama_alpha_window_cleanup.py',ROOT/'novak_work/rama_periodic_region_metrics.py'):
        pin(p,pins)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'plan.json',dict(classes=CLASSES,targets=TARGETS,density_ratio=FACTOR,
        alpha_policy='Fresh unsmoothed selection; unresolved classes display explicitly labelled unconstrained diagnostics',
        score='Maximum eligible containing-triangle minimum raw DTFE corner density',
        calibration='Exact original weighted whole-triangle construction levels before cleanup',
        alpha_domain_protection='Never fill a background component touching any alpha-excluded cell',
        smoothing=False,post_cleanup_recalibration=False))
    rows=[]
    for c in CLASSES:rows.extend(construct(c,output,pins))
    write(output/'construction.json',dict(rows=rows,KDE_access=False))
    write(output/'construction_terminal.json',dict(status='completed',source_inputs=pins,
        artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}))
    render(output,rows,pins)
    for name,value in pins.items():assert sha(ROOT/name)==value,name
    write(output/'terminal.json',dict(status='completed',source_inputs=pins,settings_changed_during_assessment=False,
        artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}))
    print('Completed '+str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT)
    main(parser.parse_args().out)
