#!/usr/bin/env python3
"""Rebuild the exposition's figures from the adopted, saved Top8000 results.

This is a rendering/verification script, not a new fit or parameter search.
Every numerical input is hashed, and the displayed 1.25x masks are checked
against both their saved hashes and the independently saved Jaccard scores.
The sole synthetic illustration is explicitly labelled as a DTFE schematic.
Run from anywhere inside the project with Python, NumPy, SciPy and Matplotlib.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-dtfe-exposition')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'novak_work/validation_results'
FIG = HERE / 'figures'
TABLE = HERE / 'tables'
CLASSES = ('General', 'Gly', 'TransPro', 'CisPro', 'PrePro', 'IleVal')
TARGETS = (.98, .995)
BLUE = '#237ca6'
PALE = '#c7deea'
ORANGE = '#cc7138'
GREEN = '#378663'
INK = '#252b31'
PINS = {}
CHECKS = []
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': True,
    'axes.spines.right': True, 'axes.edgecolor': INK,
    'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    PINS[str(path.relative_to(ROOT))] = digest(path)
    return path


def read_json(path):
    return json.loads(pin(path).read_text())


def unpack(packed, n=1536):
    return np.unpackbits(packed, bitorder='little', count=n*n).reshape(n, n).astype(bool)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=320, bbox_inches='tight', pad_inches=.035)
    plt.close(fig)
    print(name, flush=True)


def outline(ax, mask, color=BLUE, lw=.65, **kwargs):
    # Periodic padding allows the boundary to meet each displayed seam
    # without introducing a spurious line along the square's edge.
    n = mask.shape[0]
    axis = -180 + (np.arange(-1, n+1) + .5) * 360/n
    ax.contour(axis, axis, np.pad(mask.astype(float), 1, mode='wrap'),
               levels=[.5], colors=[color], linewidths=lw, **kwargs)


def paint(ax, mask, color=PALE, alpha=1):
    ax.imshow(np.ma.masked_where(~mask, mask), origin='lower',
              extent=(-180, 180, -180, 180), interpolation='nearest',
              cmap=ListedColormap([color]), vmin=0, vmax=1, alpha=alpha,
              rasterized=True)


def axes_angles(ax, crop=None, labels=True):
    ax.set_aspect('equal')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-180, 0, 180])
    if crop:
        ax.set_xlim(*crop[:2]); ax.set_ylim(*crop[2:])
        ax.set_xticks(np.linspace(crop[0], crop[1], 3))
        ax.set_yticks(np.linspace(crop[2], crop[3], 3))
    if labels:
        ax.set_xlabel(r'$\phi$ (degrees)')
        ax.set_ylabel(r'$\psi$ (degrees)')
    ax.tick_params(length=3, pad=2)


def beta(row):
    return tuple(row['raster_betti'][:2])


def selected_mask(category, target, variant='span_1.25'):
    return MASKS[category, target, variant]


def load_results():
    policy = read_json(ROOT / 'documents/DTFE_ALPHA_EPSILON_POLICY.json')
    assert policy['epsilon']['stability_span_factor'] == 1.25
    assert policy['epsilon']['ratio_domain_inclusive'] == [1, 64]
    for path, expected in policy['source_sha256'].items():
        assert digest(ROOT / path) == expected, path
    assessment = read_json(BASE / 'uniform_alpha_epsilon_stability_v1/assessment.json')
    rows = {(r['category'], r['target'], r['variant']): r for r in assessment['rows']}
    masks, refs, decisions, alphas = {}, {}, {}, {}
    for c in CLASSES:
        folder = BASE / 'uniform_alpha_epsilon_stability_v1' / c
        with np.load(pin(folder / 'selected_masks.npz')) as z:
            assert list(z['mask_axes']) == ['psi', 'phi']
            for j, (v, q) in enumerate(zip(z['variants'], z['targets'])):
                mask = unpack(z['packed_masks'][j], int(z['grid_size']))
                assert hashlib.sha256(mask.tobytes()).hexdigest() == rows[c, q, str(v)]['mask_sha256']
                masks[c, float(q), str(v)] = mask
        with np.load(pin(BASE / 'internal_alpha_unfloored_assessment_v1' / c / 'grid_1536_phase0.npz')) as z:
            i = list(z['controls']).index('published_reference')
            for j, q in enumerate(z['targets']):
                assert z['defined'][i, j]
                refs[c, float(q)] = unpack(z['packed_masks'][i, j])
        decision = read_json(folder / 'selection.json')['selections']['1.25']
        assert decision['status'] == 'selected'
        interval = decision['interval']
        assert interval['checks']['admissible']
        k = decision['selected_k']
        assert k == policy['current_top8000_derived_results'][c]['k']
        right = interval['observed_interval_right_k']
        assert 1.25*k <= right if interval['right_censored'] else 1.25*k < right
        decisions[c] = decision
        alphas[c] = read_json(BASE / 'uniform_unsmoothed_alpha_guarded_v1' / c / 'selection.json')
        for q in TARGETS:
            m, r = masks[c, q, 'span_1.25'], refs[c, q]
            j = np.count_nonzero(m & r) / np.count_nonzero(m | r)
            assert abs(j-rows[c, q, 'span_1.25']['jaccard']) < 1e-14
            CHECKS.append(dict(category=c, target=q, mask_hash_verified=True, jaccard_recomputed=j))
        assert not np.any(masks[c, .98, 'span_1.25'] & ~masks[c, .995, 'span_1.25'])
    return rows, masks, refs, decisions, alphas


def dtfe_schematic():
    rng = np.random.default_rng(29)
    points = np.concatenate([rng.uniform(.05, .95, (24, 2)),
                             rng.normal([.37, .55], [.065, .085], (26, 2))])
    tri = Delaunay(points).simplices
    polys = points[tri]
    u, v = polys[:, 1]-polys[:, 0], polys[:, 2]-polys[:, 0]
    areas = abs(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0])/2
    assigned = np.bincount(tri.ravel(), weights=np.repeat(areas/3, 3), minlength=len(points))
    rho = 1/assigned
    gates = rho[tri].min(axis=1)
    # A schematic threshold, not a fitted Top8000 parameter.
    L = np.quantile(gates, .57)
    focus = np.argmin(np.sum((points-[.64, .42])**2, axis=1))
    fig, axes = plt.subplots(1, 3, figsize=(6.55, 2.7))
    for ax in axes:
        ax.add_collection(PolyCollection(polys, facecolors='none', edgecolors='#919aa1', linewidths=.35))
        ax.scatter(*points.T, s=7, c=INK, zorder=4)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title('(a) Local triangles')
    for p in polys[np.any(tri == focus, axis=1)]:
        at = np.flatnonzero(np.all(p == points[focus], axis=1))[0]
        a, b, c = p[at], p[(at+1)%3], p[(at+2)%3]
        axes[1].add_patch(Polygon([a, (a+b)/2, p.mean(axis=0), (a+c)/2],
                                  facecolor=ORANGE, edgecolor='white', lw=.45, zorder=2))
    axes[1].scatter(*points[focus], s=30, facecolor='white', edgecolor=INK, zorder=6)
    axes[1].annotate(r'$A_i$', points[focus], xytext=(.76, .12),
                      arrowprops=dict(arrowstyle='-', color=INK), fontsize=12)
    axes[1].set_title('(b) Assigned area')
    axes[2].add_collection(PolyCollection(polys[gates >= L], facecolors=PALE,
                                          edgecolors=BLUE, linewidths=.6, zorder=3))
    axes[2].set_title('(c) Whole-triangle selection')
    fig.text(.5, .03, r'Schematic only: $\rho_i=1/A_i$; a whole triangle passes when $\min_i\rho_i\geq L$.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.01, right=.99, top=.87, bottom=.16, wspace=.1)
    save(fig, 'dtfe_schematic')


def cispro_alpha():
    c = 'CisPro'; q = .995
    with np.load(pin(BASE / 'uniform_unsmoothed_alpha_guarded_assessment_v1/CisPro_masks.npz')) as z:
        index = list(z['targets']).index(q)*len(z['stages']) + list(z['stages']).index('infinity')
        before = unpack(z['packed_masks'][index])
    after = selected_mask(c, q, 'no_cleanup')
    folder = BASE / 'remaining_full_smoothing_assessment_v1/CisPro/grid/geometry'
    meta = read_json(folder/'metadata.json')
    def array(key):
        a = meta['arrays'][key]
        return np.fromfile(pin(folder/a['path']), dtype=a['dtype']).reshape(a['shape'])
    vertices, bases, lifts = array('vertices'), array('bases').astype(int), array('lifts')
    polys = vertices[bases] + 360*lifts
    with np.load(pin(BASE/'uniform_unsmoothed_alpha_guarded_v1/CisPro/selected_triangles.npz')) as z:
        nt = int(z['triangle_count'])
        selected = np.unpackbits(z['packed_masks'][1], bitorder='little', count=nt).astype(bool)
        baseline = np.unpackbits(z['baseline_packed_masks'][1], bitorder='little', count=nt).astype(bool)
    removed = polys[baseline & ~selected]
    assert len(removed) == 11
    removed -= 360*np.floor((removed.mean(axis=1)[:, None, :] + 180)/360)
    fig, axes = plt.subplots(1, 3, figsize=(6.55, 3.25))
    for ax, m in zip(axes[:2], [before, after]):
        paint(ax, m); outline(ax, m); axes_angles(ax)
    axes[0].set_title('(a) Whole triangles\n'+r'$\alpha=\infty$')
    axes[1].set_title('(b) Guarded alpha\n'+r'$\alpha=14.5551^\circ$')
    paint(axes[2], after)
    outline(axes[2], after)
    for dx in (-360, 0, 360):
        for dy in (-360, 0, 360):
            axes[2].add_collection(PolyCollection(removed+[dx, dy], facecolors='#f1cfb6',
                edgecolors=ORANGE, linewidths=.8, zorder=3))
    crop = (-125, -25, 20, 120)
    axes_angles(axes[2], crop)
    axes[2].set_title('(c) The sparse bridge\nRemoved triangles in orange')
    for ax in axes[:2]:
        ax.add_patch(Rectangle((crop[0], crop[2]), crop[1]-crop[0], crop[3]-crop[2],
                               fill=False, edgecolor=ORANGE, linestyle='--', lw=.9))
    axes[1].set_ylabel(''); axes[2].set_ylabel('')
    fig.text(.5, .035, 'CisPro, 99.5% construction target. No smoothing; density threshold held fixed.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.065, right=.99, top=.82, bottom=.21, wspace=.32)
    save(fig, 'cispro_alpha')


def general_cleanup():
    fig, axes = plt.subplots(2, 2, figsize=(6.1, 6.7))
    k = DECISIONS['General']['selected_k']
    for i, q in enumerate(TARGETS):
        for j, v in enumerate(('no_cleanup', 'span_1.25')):
            m = selected_mask('General', q, v)
            paint(axes[i, j], m); outline(axes[i, j], m, lw=.4)
            axes_angles(axes[i, j]); row = ROWS['General', q, v]
            title = 'After alpha, before cleanup' if j == 0 else f'After cleanup, k = {k:.3f}'
            axes[i, j].set_title(title+'\n'+f'{q*100:g}% target; '+r'$(\beta_0,\beta_1)=$'+f'{beta(row)}')
            if j: axes[i, j].set_ylabel('')
    fig.subplots_adjust(left=.10, right=.98, bottom=.085, top=.93, hspace=.34, wspace=.27)
    save(fig, 'general_cleanup')


def event_mask(category, target, k):
    folder = BASE/'uniform_alpha_epsilon_events_v1'/category/'grid_1536'/f'target_{round(1000*target)}'
    events = read_json(folder/'events.json')
    states = events['states']
    i = np.searchsorted([s['left_inclusive_k'] for s in states], k, side='right')-1
    state = states[i]
    with np.load(pin(folder/'masks.npz')) as z:
        m = unpack(z['packed_masks'][state['mask_index']])
    assert hashlib.sha256(m.tobytes()).hexdigest() == state['mask_sha256']
    return m, state


def feature_fates():
    # These observed transitions locate specific components; none is used
    # to choose k. The selected ratio was already fixed without a KDE mask.
    event98 = 4.894829841963749
    event995 = 11.94771719576887
    k = DECISIONS['General']['selected_k']
    cases = [(.98, (25, 80, -155, -100), [1, np.nextafter(event98, 0), event98],
              ['Starting island', 'Immediately before loss', 'Immediately after loss']),
             (.995, (45, 100, -80, -25), [1, k, event995],
              ['Starting island', 'Selected cleanup', 'Stronger cleanup'])]
    fig, axes = plt.subplots(2, 3, figsize=(6.55, 5.2))
    for i, (q, crop, ks, titles) in enumerate(cases):
        for j, (value, title) in enumerate(zip(ks, titles)):
            m, state = event_mask('General', q, value)
            paint(axes[i,j], m); outline(axes[i,j], m, lw=.6)
            axes_angles(axes[i,j], crop)
            label = f'k = {value:.4f}'
            if i == 0 and j == 1: label = r'$k\uparrow 4.89483$'
            if i == 0 and j == 2: label = r'$k=4.89483\ldots$ (event onset)'
            axes[i,j].set_title(title+'\n'+label, fontsize=9)
            if j == 2:
                axes[i,j].text(.5, .5, 'Island absent', color='#666666',
                                ha='center', va='center', transform=axes[i,j].transAxes)
            if j: axes[i,j].set_ylabel('')
        axes[i,0].set_ylabel(f'{q*100:g}% target\n'+r'$\psi$ (degrees)')
    fig.subplots_adjust(left=.095, right=.99, top=.92, bottom=.09, hspace=.56, wspace=.30)
    save(fig, 'general_feature_fates')


def stability_plot():
    fig = plt.figure(figsize=(6.55, 5.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1], hspace=.5)
    ax = fig.add_subplot(gs[0]); zoom = fig.add_subplot(gs[1])
    k = DECISIONS['General']['selected_k']
    b = DECISIONS['General']['interval']['right_exclusive_k']
    for q, color in zip(TARGETS, [BLUE, ORANGE]):
        folder = BASE/'uniform_alpha_epsilon_events_v1/General/grid_1536'/f'target_{round(1000*q)}'
        states = read_json(folder/'events.json')['states']
        x = [s['left_inclusive_k'] for s in states]+[64]
        for col, ls, name in [(0, '-', 'components'), (1, '--', 'loops')]:
            y = [s['raster_betti'][col] for s in states]
            # +1 permits zeros and large counts on the same logarithmic scale.
            ax.step(x, np.array(y+[y[-1]])+1, where='post', color=color, ls=ls, lw=1.1,
                    label=f'{q*100:g}%: {name}')
    ax.axvspan(k, 1.25*k, color=GREEN, alpha=.16)
    ax.axvline(k, color=GREEN, lw=1)
    ax.set(xscale='log', yscale='log', xlim=(1,64), xlabel='Cleanup ratio k', ylabel='Count + 1 (log scale)')
    ax.set_xticks([1,2,4,8,16,32,64], labels=['1','2','4','8','16','32','64'])
    ax.set_yticks([1,2,6,11,101,1001], labels=['1','2','6','11','101','1001'])
    ax.grid(alpha=.17); ax.legend(ncol=2, loc='upper right')
    ax.set_title('(a) General: removing transient features, then stopping', loc='left')
    # Display every event of the four streams locally. The long blank band
    # is a joint plateau of complete masks, not merely a plateau of counts.
    for i, (q,n) in enumerate([(q,n) for q in TARGETS for n in (768,1536)]):
        ev = read_json(BASE/'uniform_alpha_epsilon_events_v1/General'/f'grid_{n}'/f'target_{round(1000*q)}'/'events.json')
        changes = [s['left_inclusive_k'] for s in ev['states'][1:] if 5.8 <= s['left_inclusive_k'] <= 9.2]
        zoom.hlines(i, 5.8, 9.2, color='#d2d7da', lw=1)
        zoom.vlines(changes, i-.17, i+.17, color=INK, lw=.75)
    zoom.axvspan(k, b, color='#e2ece6')
    zoom.axvspan(k, 1.25*k, color=GREEN, alpha=.25)
    zoom.axvline(k, color=GREEN); zoom.axvline(1.25*k, color=GREEN, ls='--')
    zoom.axvline(b, color=ORANGE, ls=':')
    zoom.set_yticks(range(4), ['98%, 768', '98%, 1536', '99.5%, 768', '99.5%, 1536'])
    zoom.set(xlim=(5.8,9.2), ylim=(-.5,3.7), xlabel='Cleanup ratio k; each tick on a row is an actual region change')
    zoom.set_title('(b) One unchanged interval for both levels and both grids', loc='left')
    zoom.text(k, 3.32, f'{k:.3f}', ha='left', color=GREEN, fontsize=8)
    zoom.text(1.25*k-.03, 3.32, f'1.25k = {1.25*k:.3f}', ha='right', color=GREEN, fontsize=8)
    zoom.text(b+.04, -.4, f'next change\n{b:.3f}', ha='left', color=ORANGE, fontsize=8)
    fig.subplots_adjust(left=.17, right=.98, top=.94, bottom=.105)
    save(fig, 'general_stability')


def reference_panels(q):
    fig, axes = plt.subplots(2, 3, figsize=(6.55, 5.85))
    for ax, c, letter in zip(axes.flat, CLASSES, 'abcdef'):
        m, ref = selected_mask(c, q), REFS[c,q]
        paint(ax, m); outline(ax, m, lw=.42); outline(ax, ref, color=INK, lw=.8)
        axes_angles(ax)
        row = ROWS[c,q,'span_1.25']
        ax.set_title(f'({letter}) {c}: J = {row["jaccard"]:.3f}\n'+
                     r'$(\beta_0,\beta_1)=$'+f'{beta(row)}', fontsize=9)
    for i in [1,2,4,5]: axes.flat[i].set_ylabel('')
    fig.legend(handles=[Patch(facecolor=PALE, edgecolor=BLUE, label='Final triangle-based region'),
                        Line2D([0],[0],color=INK,lw=1.1,label='Published KDE boundary')],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5,.003))
    fig.subplots_adjust(left=.075, right=.99, top=.94, bottom=.115, wspace=.28, hspace=.34)
    save(fig, f'kde_comparison_{round(1000*q)}')


def tables():
    params, scores, coverage = [], [], []
    for c in CLASSES:
        a = ALPHAS[c]; k = DECISIONS[c]['selected_k']
        params.append(f'{c} & {a["observation_count"]:,} & {a["alpha_deg"]:.4f} & {k:.4f} \\\\')
        for q in TARGETS:
            before, after = ROWS[c,q,'no_cleanup'], ROWS[c,q,'span_1.25']
            beta_before = beta(before); beta_after = beta(after)
            reference = tuple(after['published_reference_betti'][:2])
            scores.append(f'{c if q == .98 else ""} & {100*q:g} & {before["jaccard"]:.3f} & {after["jaccard"]:.3f} & '
                          f'${beta_before}$ & ${beta_after}$ & ${reference}$ \\\\')
            if q == .995: scores.append(r'\addlinespace[2pt]')
            coverage.append(dict(category=c, target=q,
                exact_alpha_count=after['exact_starting_construction_count'], N=after['total_count'],
                raster_before=before['raster_coverage'], raster_after=after['raster_coverage']))
    (TABLE/'parameters.tex').write_text('\n'.join(params)+'\n')
    (TABLE/'agreement.tex').write_text('\n'.join(scores)+'\n')
    (HERE/'numerical_results.json').write_text(json.dumps(dict(parameters={c:dict(
        N=ALPHAS[c]['observation_count'], alpha_deg=ALPHAS[c]['alpha_deg'],
        k=DECISIONS[c]['selected_k']) for c in CLASSES}, coverage=coverage,
        assessment=[ROWS[c,q,'span_1.25'] for c in CLASSES for q in TARGETS]), indent=2)+'\n')


if __name__ == '__main__':
    FIG.mkdir(exist_ok=True); TABLE.mkdir(exist_ok=True)
    ROWS, MASKS, REFS, DECISIONS, ALPHAS = load_results()
    dtfe_schematic()
    cispro_alpha()
    general_cleanup()
    feature_fates()
    stability_plot()
    for q in TARGETS: reference_panels(q)
    tables()
    manifest = dict(protocol='dtfe-alpha-epsilon-span125-v1',
        purpose='Render adopted saved results; no new fitting or parameter selection',
        schematic='dtfe_schematic only: seeded illustrative points, not research evidence',
        reference='Published Top8000 percentile grids, periodic bilinear scores, assessed after selection',
        input_sha256=PINS, checks=CHECKS,
        script_sha256=digest(__file__),
        outputs={str(p.relative_to(HERE)):digest(p) for p in sorted(FIG.iterdir()) if p.is_file()})
    (HERE/'figure_generation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verified all twelve adopted masks and recomputed their Jaccard scores.', flush=True)
