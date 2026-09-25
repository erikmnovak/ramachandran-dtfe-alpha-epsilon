#!/usr/bin/env python3
"""Select first stable cleanup intervals, seal them, then assess against KDE.

This is an experiment on fixed adopted alpha fields. It does not change the
alpha protocol or promote epsilon cleanup to a default. All topology tests
refer to closed periodic raster cells. A plateau requires unchanged *masks*,
not merely unchanged Betti numbers. The primary span is a doubling; shorter
spans are prespecified sensitivity analyses, never selected using the KDE.
"""
from pathlib import Path
import argparse
import bisect
import hashlib
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import run_unsmoothed_alpha_then_epsilon as r
from rama_periodic_region_metrics import periodic_region_topology, periodic_boundary_comparison
from rama_epsilon_plateau_selection import joint_intervals, select_first

ROOT=r.ROOT
BASE=r.BASE
EVENTS=BASE/'uniform_alpha_epsilon_events_v1'
OUT=BASE/'uniform_alpha_epsilon_stability_v1'
PLAN=ROOT/'documents/UNIFORM_ALPHA_EPSILON_STABILITY_PLAN.json'
GRIDS=(768,1536)
SPANS=(1.25,1.5,2.)
# The order is fixed in stored intervals and masks: q98 coarse/fine, q995 coarse/fine.
KEYS=tuple((q,n) for q in r.TARGETS for n in GRIDS)
METRICS=('raster_betti','area_deg2','mask_cells','raster_count','raster_coverage',
    'raw_raster_count','lost_count','gained_count','removed_cells','filled_cells',
    'total_count','exact_starting_construction_count','mask_index','mask_sha256')


def read(path):
    return json.loads(path.read_text())


def verify(folder):
    terminal=read(folder/'terminal.json')
    assert terminal['status']=='completed'
    for name,h in terminal['artifacts'].items():assert r.sha(folder/name)==h,name
    return terminal


class ClassStates:
    """Load packed masks once per class; unpack only requested geometries."""
    def __init__(self,category):
        self.category=category;self.reports={};self.packed={};self.cache={}
        for key in KEYS:
            q,n=key;folder=EVENTS/category/f'grid_{n}'/f'target_{round(q*1000)}'
            self.reports[key]=read(folder/'events.json')
            with np.load(folder/'masks.npz',allow_pickle=False) as z:
                assert int(z['grid_size'])==n and float(z['target'])==q
                assert str(z['bitorder'])=='little' and z['mask_axes'].tolist()==['psi','phi']
                self.packed[key]=z['packed_masks']

    def state(self,key,k):
        states=self.reports[key]['states']
        return states[bisect.bisect_right([s['left_inclusive_k'] for s in states],k)-1]

    def mask(self,key,state):
        cachekey=(key,state['mask_index'])
        if cachekey not in self.cache:
            n=key[1]
            mask=np.unpackbits(self.packed[key][state['mask_index']],bitorder='little',count=n*n).reshape(n,n).astype(bool)
            assert hashlib.sha256(mask.tobytes()).hexdigest()==state['mask_sha256']
            # A complete event sweep can contain hundreds of large masks.
            # Bound this short-lived cache rather than retaining every state.
            if len(self.cache)>=12:self.cache.pop(next(iter(self.cache)))
            self.cache[cachekey]=mask
        return self.cache[cachekey]

    def checks(self,interval,keys=KEYS):
        states=interval['states'];pairs=dict(zip(keys,states))
        targets=sorted(set(q for q,n in keys))
        grid_agreement=all(pairs[q,768]['raster_betti']==pairs[q,1536]['raster_betti'] for q in targets)
        nonempty=all(s['mask_cells']>0 for s in states)
        nesting={}
        # Independent per-level selection deliberately has no shared-level
        # nesting constraint. Its combined nesting is assessed afterward.
        if len(targets)==2:
            for n in GRIDS:
                nesting[str(n)]=int(np.count_nonzero(self.mask((.98,n),pairs[.98,n])&~self.mask((.995,n),pairs[.995,n])))
        return dict(grid_betti_agreement=grid_agreement,nonempty=nonempty,
            inner_cells_outside_outer=nesting,
            admissible=grid_agreement and nonempty and not any(nesting.values()))

    def compact(self,interval,keys=KEYS):
        result={k:v for k,v in interval.items() if k!='states'}
        result['states']=[dict(target=q,grid_size=n,**{k:s[k] for k in METRICS}) for (q,n),s in zip(keys,interval['states'])]
        result['checks']=self.checks(interval,keys)
        return result


def select():
    assert not OUT.exists(),'Use a new output version; do not overwrite sealed choices.'
    verify(EVENTS)
    OUT.mkdir()
    sources=[Path(__file__),ROOT/'novak_work/rama_epsilon_plateau_selection.py',
        ROOT/'novak_work/test_rama_epsilon_plateau_selection.py',PLAN]
    pins={str(p.relative_to(ROOT)):r.sha(p) for p in sources}
    r.write(OUT/'plan.json',dict(declared_plan=read(PLAN),source_sha256=pins,
        events_terminal_sha256=r.sha(EVENTS/'terminal.json'),no_KDE_read=True))
    results=[]
    for category in r.CLASSES:
        data=ClassStates(category)
        intervals=joint_intervals([data.reports[key]['states'] for key in KEYS])
        joint=[data.compact(i) for i in intervals]
        selections={};per_level={}
        for span in SPANS:
            chosen=select_first(intervals,span,lambda i:data.checks(i)['admissible'])
            selections[str(span)]=(dict(status='selected',selected_k=chosen['left_inclusive_k'],
                interval=data.compact(chosen)) if chosen is not None else dict(status='unresolved',selected_k=None))
            individual=[];target_masks={}
            for q in r.TARGETS:
                keys=((q,768),(q,1536))
                pairs=joint_intervals([data.reports[key]['states'] for key in keys])
                c=select_first(pairs,span,lambda i:data.checks(i,keys)['admissible'])
                individual.append(dict(target=q,status='unresolved',selected_k=None) if c is None else
                    dict(target=q,status='selected',selected_k=c['left_inclusive_k'],interval=data.compact(c,keys)))
                if c is not None:
                    for key,state in zip(keys,c['states']):target_masks[key]=data.mask(key,state)
            nesting=None
            if len(target_masks)==4:
                nesting={str(n):int(np.count_nonzero(target_masks[.98,n]&~target_masks[.995,n])) for n in GRIDS}
            per_level[str(span)]=dict(selections=individual,combined_inner_cells_outside_outer=nesting)
        at16=next(i for i in intervals if i['left_inclusive_k']<=16 and
            (i['right_exclusive_k'] is None or 16<i['right_exclusive_k']))
        # Save all chosen figures' masks before any reference is opened. Ratio
        # 16 and k=1 are fixed controls, not substitutes for unresolved results.
        controls=[('no_cleanup',1.),('previous_16',16.)]
        for span in SPANS:
            s=selections[str(span)]
            if s['status']=='selected':controls.append((f'span_{span:g}',s['selected_k']))
        folder=OUT/category;folder.mkdir();saved=[];records=[]
        for name,k in controls:
            for qi,q in enumerate(r.TARGETS):
                fine=data.state((q,1536),k);coarse=data.state((q,768),k)
                mask=data.mask((q,1536),fine);small=data.mask((q,768),coarse)
                saved.append(np.packbits(mask.ravel(),bitorder='little'))
                records.append(dict(variant=name,k=k,target=q,grid_size=1536,
                    **{x:fine[x] for x in METRICS},
                    cross_grid_xor_area_deg2=float(np.count_nonzero(mask!=np.repeat(np.repeat(small,2,0),2,1))*(360/1536)**2),
                    same_fine_mask_as_16=fine['mask_sha256']==data.state((q,1536),16)['mask_sha256']))
        np.savez_compressed(folder/'selected_masks.npz',packed_masks=np.asarray(saved),grid_size=1536,
            bitorder='little',mask_axes=['psi','phi'],variants=[x['variant'] for x in records],targets=[x['target'] for x in records])
        r.write(folder/'selected_regions.json',dict(rows=records))
        result=dict(category=category,primary_span=2.,selections=selections,per_level_diagnostic=per_level,
            interval_containing_16=data.compact(at16),joint_intervals=joint)
        r.write(folder/'selection.json',result);results.append(result)
        print(category+': '+str({s:v['selected_k'] for s,v in selections.items()}),flush=True)
        del data
    r.write(OUT/'selection.json',dict(classes=results,no_KDE_read=True,cleanup_adopted=False))
    for name,h in pins.items():assert r.sha(ROOT/name)==h,name
    r.write(OUT/'selection_terminal.json',dict(status='completed',no_KDE_read=True,
        source_sha256=pins,events_terminal_sha256=r.sha(EVENTS/'terminal.json'),
        artifacts={str(p.relative_to(OUT)):r.sha(p) for p in OUT.rglob('*') if p.is_file()}))


def kde_line(ax,mask):
    n=len(mask);axis=-180+(np.arange(-1,n+1)+.5)*360/n
    if mask.any() and not mask.all():
        ax.contour(axis,axis,np.pad(mask,1,mode='wrap'),levels=[.5],colors=['#111111'],linewidths=.95,zorder=3)


def btext(b):
    return f'components {b[0]}, loops {b[1]}'


def interval_text(i):
    if i['right_censored']:return f'[{i["left_inclusive_k"]:.5g}, 64] observed'
    return f'[{i["left_inclusive_k"]:.5g}, {i["right_exclusive_k"]:.5g})'


def stability_plot(category,result):
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    for qi,q in enumerate(r.TARGETS):
        for bi,name in enumerate(('Components (β₀)','Loops (β₁)')):
            ax=axes[qi,bi]
            for n,color,style in ((768,'#777777','--'),(1536,'#216f9a','-')):
                states=read(EVENTS/category/f'grid_{n}'/f'target_{round(q*1000)}'/'events.json')['states']
                x=[s['left_inclusive_k'] for s in states]+[64.]
                y=[s['raster_betti'][bi] for s in states]+[states[-1]['raster_betti'][bi]]
                ax.step(x,y,where='post',label=f'{n} × {n} raster',color=color,ls=style,lw=1.8)
            chosen=result['selections']['2.0']
            if chosen['status']=='selected':
                i=chosen['interval'];k=chosen['selected_k']
                ax.axvspan(k,i['observed_interval_right_k'],color='#2e9d69',alpha=.12)
                ax.axvline(k,color='#2e9d69',lw=1.8,label=f'First doubling plateau: {k:.4g}')
            ax.axvline(16,color='#ba5d20',ls=':',lw=2,label='Previous ratio 16')
            ax.set_xscale('log',base=2);ax.set_yscale('symlog',linthresh=3)
            ax.set_xticks([1,2,4,8,16,32,64],[str(x) for x in [1,2,4,8,16,32,64]])
            ax.set_xlim(1,64);ax.set_ylim(bottom=-.15);ax.grid(alpha=.2)
            ax.set_xlabel('Density-window ratio k');ax.set_ylabel(name)
            ax.set_title(f'{100*q:g}% starting level')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.935),ncol=2,frameon=False)
    fig.suptitle(category+' — where the cleanup changes and first becomes stable',fontsize=16,y=.985)
    fig.text(.5,.025,'Selection uses unchanged full masks at both levels and resolutions; equal counts alone do not qualify.\nGreen shading is the selected joint plateau. The study stops at k = 64.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.80,bottom=.13,left=.08,right=.98,hspace=.35,wspace=.23)
    fig.savefig(OUT/f'{category}_stability.png',dpi=210,facecolor='white');plt.close(fig)


def assess():
    # Explicit phase barrier: a complete no-KDE selection snapshot must exist
    # and all its artifact hashes must still agree before reference access.
    terminal=read(OUT/'selection_terminal.json');assert terminal['no_KDE_read']
    for name,h in terminal['artifacts'].items():assert r.sha(OUT/name)==h,name
    reference_terminal=verify(r.REFERENCE)
    results=read(OUT/'selection.json')['classes'];assessment=[];cases={};pins={}
    for result in results:
        category=result['category'];records=read(OUT/category/'selected_regions.json')['rows']
        with np.load(OUT/category/'selected_masks.npz',allow_pickle=False) as z:
            masks=np.unpackbits(z['packed_masks'],axis=1,bitorder='little',count=1536**2).reshape(-1,1536,1536).astype(bool)
        path=r.REFERENCE/category/'grid_1536_phase0.npz'
        assert r.sha(path)==reference_terminal['artifacts'][str(Path(category)/path.name)]
        pins[str(path.relative_to(ROOT))]=r.sha(path)
        with np.load(path,allow_pickle=False) as z:
            assert z['targets'].tolist()==list(r.TARGETS) and z['mask_axes'].tolist()==['psi','phi']
            ci=z['controls'].tolist().index('published_reference');assert z['defined'][ci].all()
            reference=np.unpackbits(z['packed_masks'][ci],axis=1,bitorder='little',count=1536**2).reshape(2,1536,1536).astype(bool)
        lookup={};metric_cache={}
        for record,mask in zip(records,masks):
            q=record['target'];qi=r.TARGETS.index(q);ref=reference[qi]
            key=(q,record['mask_sha256'])
            if key not in metric_cache:
                assert periodic_region_topology(mask)['betti']==record['raster_betti']
                metric_cache[key]=dict(jaccard=float((mask&ref).sum()/(mask|ref).sum()),
                    published_reference_betti=periodic_region_topology(ref)['betti'],
                    boundary=periodic_boundary_comparison(mask,ref))
            row=dict(category=category,**record,**metric_cache[key]);assessment.append(row)
            lookup[record['variant'],q]=(mask,row)
        cases[category]=(lookup,reference,result)
        stability_plot(category,result)
        variants=('no_cleanup','span_1.25','span_2','previous_16')
        labels=('Alpha region; no cleanup','First stable over 1.25×\nSensitivity check','First stable over 2×\nPrimary rule','Previous cleanup')
        fig,axes=plt.subplots(2,4,figsize=(18.5,10.7))
        for qi,q in enumerate(r.TARGETS):
            for si,(variant,label) in enumerate(zip(variants,labels)):
                ax=axes[qi,si]
                if (variant,q) not in lookup:
                    ax.text(.5,.5,'No qualifying interval\nin the declared range',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
                mask,row=lookup[variant,q]
                r.image_panel(ax,mask,label+f' · k = {row["k"]:.5g}\n'+btext(row['raster_betti'])+f' · J = {row["jaccard"]:.3f}')
                kde_line(ax,reference[qi])
                if si==0:ax.set_ylabel(f'{100*q:g}% starting level\nψ (degrees)')
        fig.suptitle(category+' — unchanged alpha construction, different epsilon cleanup windows',fontsize=17,y=.985)
        fig.legend(handles=[Patch(facecolor='#2e80aa',label='Our region'),Line2D([],[],color='#111111',label='Published KDE boundary (assessment only)')],
            loc='upper center',bbox_to_anchor=(.5,.952),ncol=2,frameon=False,fontsize=12)
        fig.text(.5,.026,'Ratios were selected before KDE assessment. Percentages identify the starting construction level; cleanup coverage is measured again.\nSmaller component counts do not by themselves mean better contours.',ha='center',fontsize=11)
        fig.subplots_adjust(top=.82,bottom=.11,left=.047,right=.988,wspace=.26,hspace=.37)
        fig.savefig(OUT/f'{category}_epsilon_comparison_kde.png',dpi=215,facecolor='white');plt.close(fig)
        print(category+': stability curves and KDE comparison rendered',flush=True)
    for qi,q in enumerate(r.TARGETS):
        fig,axes=plt.subplots(2,3,figsize=(14,11.8))
        for ax,category in zip(axes.flat,r.CLASSES):
            lookup,reference,result=cases[category]
            if ('span_2',q) not in lookup:
                ax.text(.5,.5,category+'\nNo qualifying interval',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
            mask,row=lookup['span_2',q]
            r.image_panel(ax,mask,category+f' · k = {row["k"]:.5g}\n'+btext(row['raster_betti'])+f' · J = {row["jaccard"]:.3f}')
            kde_line(ax,reference[qi])
        fig.suptitle(f'First stable cleanup over a doubling of k — {100*q:g}% starting level',fontsize=17,y=.98)
        fig.legend(handles=[Patch(facecolor='#2e80aa',label='Our region'),Line2D([],[],color='#111111',label='Published KDE boundary (assessment only)')],
            loc='upper center',bbox_to_anchor=(.5,.947),ncol=2,frameon=False,fontsize=12)
        fig.text(.5,.035,'Experimental stability selection; no smoothing, clipping, or KDE fitting. Stability does not establish that every feature is correct.',ha='center',fontsize=10)
        fig.subplots_adjust(top=.85,bottom=.12,left=.065,right=.98,wspace=.28,hspace=.36)
        fig.savefig(OUT/f'all_six_first_stable_{100*q:g}_kde.png',dpi=220,facecolor='white');plt.close(fig)
    r.write(OUT/'assessment.json',dict(rows=assessment,selection_sealed_before_KDE=True,
        selection_terminal_sha256=r.sha(OUT/'selection_terminal.json'),reference_sources=pins))
    lines=['# Epsilon-ratio stability after the adopted alpha rule','',
        'The first stable interval is selected from unchanged masks at both contour levels and both grid resolutions. The primary interval must span a doubling of the ratio. KDE is opened only after selection and masks have been sealed. This is an experiment, not an adopted cleanup default.','',
        '| Class | First 1.25× span | First 1.5× span | First 2× span | Primary joint plateau | Same fine masks as ratio 16? |',
        '|---|---:|---:|---:|---|---|']
    for result in results:
        c=result['category'];sel=result['selections'];primary=sel['2.0'];lookup,ref,_=cases[c]
        values=['unresolved' if sel[str(s)]['selected_k'] is None else f'{sel[str(s)]["selected_k"]:.6g}' for s in SPANS]
        plateau='unresolved' if primary['status']!='selected' else interval_text(primary['interval'])
        same='unresolved' if primary['status']!='selected' else str(all(lookup['span_2',q][1]['same_fine_mask_as_16'] for q in r.TARGETS))
        lines.append('| '+c+' | '+' | '.join(values)+f' | {plateau} | {same} |')
    lines+=['','Ratios shown rounded here; exact Float64 onsets in `selection.json` and saved masks define the results. Rounding down a deletion onset can retain a tied component. A final interval through 64 is observed only through that limit; it is not a claim of permanent stability.','',
        '- [All PNGs in one ZIP](epsilon_stability_figures.zip)',
        '- [All six classes, primary selection, 98%](all_six_first_stable_98_kde.png)',
        '- [All six classes, primary selection, 99.5%](all_six_first_stable_99.5_kde.png)','']
    for c in r.CLASSES:
        lines.append(f'- {c}: [stability curves]({c}_stability.png), [four-window comparison with KDE]({c}_epsilon_comparison_kde.png)')
    lines+=['','The shape figures show no cleanup, the 1.25× sensitivity selection, the primary doubling selection, and the previous fixed ratio 16. The 1.5× result and independently selected target-level ratios are recorded in the JSON.','',
        'Coverage values in this study sample the containing raster cells; they are not exact closed-triangle construction counts. The original exact counts are separately recorded. Matching Betti numbers on two grids does not establish boundary convergence. See `assessment.json` for coverage, area, boundary distances and overlap; none were optimized by the plateau selector.','',
        '- [Full explanation and limitations](../../../documents/UNIFORM_ALPHA_EPSILON_STABILITY.md)',
        '- [Prespecified plan](../../../documents/UNIFORM_ALPHA_EPSILON_STABILITY_PLAN.json)',
        '- [Event sweep](../uniform_alpha_epsilon_events_v1/)',
        '- [Analysis source](../../analyze_uniform_alpha_epsilon_stability.py)','']
    (OUT/'README.md').write_text('\n'.join(lines))
    with zipfile.ZipFile(OUT/'epsilon_stability_figures.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')):z.write(p,p.name)
        z.write(OUT/'README.md','README.md')
    for name,h in terminal['artifacts'].items():assert r.sha(OUT/name)==h,name
    r.write(OUT/'terminal.json',dict(status='completed',selection_sealed_before_KDE=True,
        source_sha256=terminal['source_sha256'],reference_sources=pins,
        artifacts={str(p.relative_to(OUT)):r.sha(p) for p in OUT.rglob('*') if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('select','assess','both'),default='both');args=parser.parse_args()
    if args.phase in ('select','both'):select()
    if args.phase in ('assess','both'):assess()
