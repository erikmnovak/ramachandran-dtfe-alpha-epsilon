"""Explicit saved-field inputs for the construction-calibrated alpha pilot.

The joined index identifies the six full populations. Only construction
arrays, certified geometry and point-to-triangle incidences are admitted as
numerical inputs. Old grid reports, reference masks, published tables and
reference-agreement summaries are deliberately not opened or hashed here.
Terminal manifests are read as provenance metadata; their other artifacts are
not recursively consumed. The worker receives this small, auditable allowlist.
"""
from pathlib import Path
import hashlib
import json

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = Path('novak_work/validation_results')
INDEX = RESULTS/'six_class_full_smoothing_completed_v1/index.json'
INDEX_SHA = 'b6699aa5d3e046f71af9f5cc559577333de7f3aaeb2e52af29906f81afe42baa'
COUNTS = dict(General=1055696, Gly=112542, IleVal=191138, PrePro=59498,
              TransPro=63227, CisPro=3523)
WIDTHS = {c: 3 if c == 'General' else 5 for c in COUNTS}


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _relative(path, root):
    path = Path(path)
    absolute = (root/path).resolve() if not path.is_absolute() else path.resolve()
    if not absolute.is_relative_to(root):
        raise ValueError('Source escapes the project root')
    return absolute.relative_to(root).as_posix()


def source_plan(*, root=PROJECT):
    """Verify exact saved inputs and return a reference-free numerical plan.

    The caller adds its numerical source hashes and writes the complete plan
    before launching any class. This routine performs no density evaluation,
    region measurement or scientific parameter selection.
    """
    root = Path(root).resolve()
    pins = {}

    def pin(path, expected=None):
        name = _relative(path, root)
        # These data are never permitted in the pilot's numerical allowlist.
        if name.startswith('data_points/') or any(x in Path(name).name for x in
                ('reference_masks', 'unique_masks', 'outcomes', 'pairs.csv')):
            raise ValueError('Published or mixed-assessment data are not selector inputs')
        digest = _sha(root/name)
        if expected is not None and digest != expected:
            raise ValueError('Changed bound input: '+name)
        if name in pins and pins[name] != digest:
            raise ValueError('Source changed while planning: '+name)
        pins[name] = digest
        return name

    def read(path, expected=None):
        name = pin(path, expected)
        value = (root/name).read_bytes()
        if hashlib.sha256(value).hexdigest() != pins[name]:
            raise ValueError('Metadata changed while reading')
        return json.loads(value)

    def artifact(directory, terminal, relative):
        relative = Path(relative).as_posix()
        if relative not in terminal['artifacts']:
            raise ValueError('Artifact is not bound by its completed terminal: '+relative)
        return pin(Path(directory)/relative, terminal['artifacts'][relative])

    index = read(INDEX, INDEX_SHA)
    if index['status'] != 'completed' or index['top2018_used'] or set(index['categories']) != set(COUNTS):
        raise ValueError('Completed six-class Top8000 index required')
    cases = {}
    for category, count in COUNTS.items():
        entry = index['categories'][category]
        if (entry['original_observation_count'] != count or entry['constructor_weight'] != count
                or entry['mirror_augmented']):
            raise ValueError('Full original construction population required')
        terminal_path = Path(entry['construct_terminal'])
        construct = terminal_path.parent
        terminal = read(terminal_path, entry['construct_terminal_sha256'])
        if terminal['status'] != 'completed' or terminal['category'] != category:
            raise ValueError('Incomplete constructor')
        report_name = artifact(construct, terminal, 'report.json')
        report = read(report_name)
        if report['n_observations'] != count or report['category'] != category:
            raise ValueError('Construction report population differs')
        estimator_id = f'sigma{WIDTHS[category]}_floor0.01'
        chosen = next(x for x in report['estimators'] if x['estimator_id'] == estimator_id)
        estimator_name = artifact(construct, terminal, chosen['report'])
        estimator = read(estimator_name, chosen['report_sha256'])
        if (estimator['sigma_deg'] != WIDTHS[category] or estimator['area_floor_quantile'] != .01
                or estimator['category'] != category or estimator['status'] != 'completed'):
            raise ValueError('Chosen estimator contract differs')
        for metadata in estimator['arrays'].values():
            name = artifact(construct, terminal, Path(chosen['report']).parent/metadata['path'])
            if pins[name] != metadata['sha256']:
                raise ValueError('Estimator array hash differs')

        if entry['layout'] == 'split_reports':
            grid_terminal_path = Path(entry['grid_terminal'])
            grid_terminal = read(grid_terminal_path, entry['grid_terminal_sha256'])
            grid_base = grid_terminal_path.parent
            grid_root = grid_base
        elif entry['layout'] == 'combined_assessment':
            grid_terminal_path = Path(entry['assessment_terminal'])
            grid_terminal = read(grid_terminal_path, entry['assessment_terminal_sha256'])
            grid_base = grid_terminal_path.parent
            grid_root = grid_base/'grid'
        else:
            raise ValueError('Unknown saved input layout')
        if grid_terminal['status'] != 'completed':
            raise ValueError('Incomplete grid export')
        geometry = grid_root/'geometry'
        meta_name = artifact(grid_base, grid_terminal, geometry.relative_to(grid_base)/'metadata.json')
        meta = read(meta_name)
        if (meta['n_triangles'] != report['n_triangles'] or meta['n_vertices'] != report['n_unique']
                or not meta['geometry_certificate']['certified']):
            raise ValueError('Certified geometry dimensions differ')
        for array in meta['arrays'].values():
            name = artifact(grid_base, grid_terminal, geometry.relative_to(grid_base)/array['path'])
            if pins[name] != array['sha256']:
                raise ValueError('Geometry array hash differs')
        grids = {}
        for size in (768, 1536):
            directory = grid_root/f'grid_{size}_phase0'
            relative = directory.relative_to(grid_base)
            incidence = read(artifact(grid_base, grid_terminal, relative/'incidences.json'))
            if (incidence['grid_size'] != size or incidence['phase'] != 0.
                    or incidence['order'] != 'C' or incidence['mask_axes'] != ['psi','phi']
                    or incidence['geometry_sha256'] != meta['geometry_sha256']):
                raise ValueError('CSR grid convention or triangle order differs')
            for key in ('offsets','indices'):
                array = incidence[key]
                name = artifact(grid_base, grid_terminal, relative/array['path'])
                if pins[name] != array['sha256']:
                    raise ValueError('CSR array hash differs')
            grids[str(size)] = _relative(directory, root)

        if 'radii' in report.get('arrays', {}):
            radii_root = construct
            radii = report['arrays']['radii']
            name = artifact(construct, terminal, radii['path'])
        else:
            # The original two-class sweep did not export the shared radii.
            # Its later density-refinement study did, with the identical mesh
            # and mandatory replay of the384-grid construction anchors.
            radii_root = RESULTS/'fine_smoothing_refinement_v1'/category
            refine_terminal = read(radii_root/'terminal.json')
            if refine_terminal['status'] != 'completed':
                raise ValueError('Incomplete saved radius export')
            refinement = read(artifact(radii_root, refine_terminal, 'report.json'))
            if (refinement['category'] != category or refinement['n_observations'] != count
                    or refinement['n_triangles'] != meta['n_triangles']
                    or refinement['geometry_metadata'] != report['geometry_metadata']
                    or _relative(refinement['canonical_geometry'], root) != _relative(geometry, root)):
                raise ValueError('Refinement radius vector belongs to another mesh/order')
            artifact(radii_root, refine_terminal, 'anchor_check.json')
            radii = refinement['arrays']['radii']
            name = artifact(radii_root, refine_terminal, radii['path'])
        if pins[name] != radii['sha256'] or radii['shape'] != [meta['n_triangles']]:
            raise ValueError('Shared radius array differs')
        cases[category] = dict(category=category, observation_count=count,
            sigma_deg=WIDTHS[category], area_floor_quantile=.01,
            construct_root=_relative(construct,root), estimator_report=estimator_name,
            geometry_directory=_relative(geometry,root), grids=grids,
            radii_directory=_relative(radii_root,root), radii_metadata=radii,
            geometry_sha256=meta['geometry_sha256'], n_vertices=meta['n_vertices'],
            n_triangles=meta['n_triangles'])
    pin(Path(__file__).resolve())
    pin('documents/CONSTRUCTION_DTFE_ALPHA_PILOT.md')
    return dict(status='frozen_before_execution',cases=cases,sources_and_inputs=pins,
        input_scope='Construction fields, geometry and CSR only; terminal manifests are provenance metadata',
        no_published_numerical_inputs=True, top2018_used=False)
