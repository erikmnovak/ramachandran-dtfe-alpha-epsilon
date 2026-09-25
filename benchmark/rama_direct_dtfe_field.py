#!/usr/bin/env python3
"""Fixed-setting 2D DTFE direct-field construction study, without comparators.

Julia owns the frozen interpolation arithmetic and weighted construction ranks.
This runner owns immutable provenance, original identities, staged data access,
packed audit masks and per-class resource supervision. It never fits a model.
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
import traceback

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
CAPSULE=PROJECT/'novak_work/validation_results/dtfe_development_v1'
CLASSES=('CisPro','TransPro','Gly','General')
GRIDS=((768,0.),(768,.5),(1536,0.),(1536,.5))
TARGETS=np.array(['49/50','199/200'])
JULIA=Path('/home/eriknovak/.julia/juliaup/julia-1.12.1+0.x64.linux.gnu/bin/julia')
SETTINGS=dict(sigma_deg=8.,area_floor_quantile=.01,alpha_restriction=False,density_grid_size=384,
    grids=[dict(size=n,phase=p)for n,p in GRIDS],targets=TARGETS.tolist(),
    calibration='Highest inclusive weighted constructor-vertex density level meeting ceil(q*N)',
    fitting=False,integration=False,smoothing=False,comparator_access=False)
LIMITS=dict(seconds_per_class=900,address_space_bytes_per_process=4*1024**3,
    process_group_rss_bytes=4*1024**3,poll_seconds=.1,threads=1,
    scope='Hard process-group wall deadline and per-process address-space limit; aggregate RSS checked every0.1s')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    if path.exists():raise ValueError('Refusing to overwrite '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(pins):
    for name,expected in pins.items():
        if sha(PROJECT/name)!=expected:raise ValueError('Bound source/input changed: '+name)


def write_f64(path,array,axes,units):
    values=np.asarray(array,dtype='>f8')
    if path.exists() or not np.isfinite(values).all():raise ValueError('Invalid/new-file f64 contract')
    values.tofile(path)
    return dict(path=path.name,shape=list(values.shape),dtype='>f8',order='C',axes=axes,units=units,
                bytes=values.nbytes,sha256=sha(path))


def read_f64(path,metadata):
    if (metadata['path']!=path.name or metadata['dtype']!='>f8' or metadata['order']!='C'
            or metadata['bytes']!=path.stat().st_size or path.stat().st_size!=8*np.prod(metadata['shape'])
            or sha(path)!=metadata['sha256']):raise ValueError('Portable output mismatch')
    values=np.fromfile(path,dtype='>f8').reshape(metadata['shape']).astype(float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite output')
    return values


def pack_masks(density,gates):
    density=np.asarray(density);gates=np.asarray(gates)
    if (density.ndim!=2 or gates.shape!=(2,) or gates[0]<gates[1]
            or not np.isfinite(density).all() or not np.isfinite(gates).all()
            or np.any(density<0) or np.any(gates<0)):raise ValueError('Bad grid/gate contract')
    membership=density[...,None]>=gates
    if np.any(membership[...,0]&~membership[...,1]):raise ValueError('Non-nested masks')
    return np.packbits(membership.reshape(-1,2).T,axis=1,bitorder='little')


def source_plan():
    capsule=read(CAPSULE/'manifest.json')
    pins={str((CAPSULE/'manifest.json').relative_to(PROJECT)):sha(CAPSULE/'manifest.json')}
    cases={}
    for category in CLASSES:
        familydir=CAPSULE/'families'/category
        family=read(familydir/'family.json')
        geometryfile=familydir/family['geometry_model']['file']
        if sha(geometryfile)!=family['geometry_model']['sha256']:raise ValueError('Geometry metadata differs')
        geometry=read(geometryfile)
        paths=[familydir/'family.json',geometryfile,CAPSULE/capsule['cases'][category]['dataset']]
        if sha(paths[-1])!=capsule['cases'][category]['dataset_sha256']:raise ValueError('Role dataset differs')
        arrays=[(familydir,family['arrays']['smoothed_density'])]
        arrays += [(geometryfile.parent,geometry['arrays'][key]) for key in ('vertices','multiplicities','phi_centers','psi_centers')]
        for directory,meta in arrays:
            path=directory/meta['path']
            if sha(path)!=meta['sha256'] or path.stat().st_size!=meta['bytes']:raise ValueError('Input array differs')
            paths.append(path)
        for path in paths:
            relative=str(path.relative_to(CAPSULE))
            if capsule['artifacts'].get(relative)!=sha(path):raise ValueError('Capsule artifact differs')
            pins[str(path.relative_to(PROJECT))]=sha(path)
        pins.update(family['source_hashes'])
        cases[category]=dict(family_directory=str(familydir.relative_to(PROJECT)),
            dataset=str(paths[2].relative_to(PROJECT)),dataset_sha256=sha(paths[2]))
    for filename in ('rama_direct_dtfe_field.py','rama_direct_dtfe_field_worker.jl','test_rama_direct_dtfe_field.py','test_rama_direct_dtfe_field.jl'):
        path=PROJECT/'benchmark'/filename;pins[str(path.relative_to(PROJECT))]=sha(path)
    for filename in ('Project.toml','Manifest.toml'):
        path=PROJECT/'novak_work'/filename;pins[str(path.relative_to(PROJECT))]=sha(path)
    verify(pins)
    return dict(status='frozen_before_execution',sources_and_inputs=pins,cases=cases,
        settings=SETTINGS,resource_limits=LIMITS)


class JuliaSession:
    def __init__(self,directory,julia):
        self.directory=directory;self.counter=0
        self.log=(directory/'julia_stderr.txt').open('w')
        self.process=subprocess.Popen([str(julia),'--startup-file=no','--threads=1','--project='+str(PROJECT/'novak_work'),
            str(PROJECT/'benchmark/rama_direct_dtfe_field_worker.jl')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,bufsize=1)

    def request(self,action,**values):
        self.counter+=1
        request=dict(action=action,request_id=self.counter,**values)
        path=self.directory/f'request_{self.counter:02d}.json';write(path,request)
        self.process.stdin.write(str(path)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Julia exited without response; see stderr log')
        result=json.loads(line);write(self.directory/f'response_{self.counter:02d}.json',result)
        if result.get('status')!='completed' or result.get('request_id')!=self.counter:raise RuntimeError('Julia request failed: '+line)
        return result.get('result')

    def close(self):
        self.request('close');self.process.stdin.close()
        if self.process.wait(timeout=30)!=0:raise RuntimeError('Julia exit failure')
        self.log.close()


def case_run(category,directory,plan_path,julia,plan_sha256):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    if sha(plan_path)!=plan_sha256:raise ValueError('Frozen execution plan changed')
    plan=read(plan_path);pins=plan['sources_and_inputs'];verify(pins)
    if plan.get('settings')!=SETTINGS or plan.get('resource_limits')!=LIMITS:raise ValueError('Plan differs from fixed execution contract')
    case=plan['cases'][category];started=time.monotonic()
    dataset=PROJECT/case['dataset']
    # Only construction arrays are opened before thresholds are sealed.
    with np.load(dataset,allow_pickle=False) as source:
        fit={key:source['refit_'+key].copy()for key in ('points','ids','groups')}
    if len(set(fit['ids']))!=len(fit['ids']):raise ValueError('Repeated construction identities')
    meta=write_f64(directory/'original_refit_points.f64',fit['points'],['observation','phi_psi'],'degree')
    session=JuliaSession(directory,julia)
    frozen=session.request('load',family_directory=str(PROJECT/case['family_directory']),output_directory=str(directory),
        category=category,original_points_file=str(directory/'original_refit_points.f64'),original_points_metadata=meta)
    if read(directory/'frozen_thresholds.json')!=frozen:raise ValueError('Frozen threshold response differs')
    frozen_hash=sha(directory/'frozen_thresholds.json')
    gates=np.array([row['density_gate']for row in frozen['rows']])
    construction={key:read_f64(directory/meta['path'],meta)for key,meta in frozen['construction_arrays'].items()}
    construction['multiplicities']=construction['multiplicities'].astype(np.int64)
    np.savez_compressed(directory/'construction.npz',**construction)
    write(directory/'field/metadata.json',dict(arrays=frozen['field_arrays'],density_grid_size=384,
        axes=['psi','phi'],sigma_deg=8.,area_floor_quantile=.01,source_estimator_column_zero_based=5,
        normalization_mass=frozen['normalization_mass'],saved_field_mass=frozen['saved_field_mass']))
    # The separate data stage starts only after the immutable cutoff exists.
    with np.load(dataset,allow_pickle=False) as source:
        extra={key:source['assessment_'+key].copy()for key in ('points','ids','groups')}
    if len(set(extra['ids']))!=len(extra['ids']) or set(fit['groups'])&set(extra['groups']):raise ValueError('Extra identity/group contract differs')
    observations={};rows=[dict(row)for row in frozen['rows']]
    for role,values in (('refit',fit),('assessment',extra)):
        path=directory/f'{role}_query_points.f64'
        record=write_f64(path,values['points'],['observation','phi_psi'],'degree')
        result=session.request('query',points_file=str(path),points_metadata=record,output_file=str(directory/f'{role}_density.f64'))
        density=read_f64(directory/result['path'],result);membership=density[:,None]>=gates
        for key,array in values.items():observations[role+'_'+key]=array
        observations[role+'_density']=density;observations[role+'_membership']=membership
        for q,row in enumerate(rows):row[role+'_count']=int(membership[:,q].sum());row[role+'_total']=len(density)
    observations['density_gates']=gates;observations['targets']=TARGETS
    np.savez_compressed(directory/'observations.npz',**observations)
    grids=[]
    for n,phase in GRIDS:
        name=f'grid_{n}_phase{phase:g}'
        result=session.request('grid',size=n,phase=phase,output_file=str(directory/(name+'_density.f64')))
        density=read_f64(directory/result['path'],result)
        packed=pack_masks(density,gates)
        axis=-180+(np.arange(n)+.5+phase)*360/n
        np.savez_compressed(directory/(name+'.npz'),axis=axis,grid_size=n,phase=phase,targets=TARGETS,
            packed_masks=packed,mask_axes=np.array(['psi','phi','target']),bitorder=np.array('little'))
        write(directory/(name+'_density.json'),result)
        grids.append(dict(file=name+'.npz',density_file=result['path'],density_metadata=name+'_density.json',size=n,phase=phase,
            counts=[int(np.count_nonzero(density>=gate))for gate in gates],density_gates=gates.tolist()))
    session.close()
    if sha(directory/'frozen_thresholds.json')!=frozen_hash:raise ValueError('Construction thresholds changed')
    verify(pins)
    write(directory/'report.json',dict(status='completed',category=category,rows=rows,grids=grids,
        settings=plan['settings'],constructor_correspondence=frozen['correspondence'],
        frozen_thresholds_sha256=frozen_hash,source_dataset_sha256=case['dataset_sha256'],
        operational_seconds=time.monotonic()-started,peak_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        representation='Inclusive superlevel of unchanged periodic bilinear Gaussian-DTFE field; no alpha or whole-triangle gate',
        scope='Recorded Top8000 development populations. Original refit coverage is separate from augmented Gly construction coverage. No comparator read.'))
    artifacts={str(p.relative_to(directory)):sha(p)for p in sorted(directory.rglob('*'))if p.is_file()}
    write(directory/'terminal.json',dict(status='completed',sources_and_inputs=pins,artifacts=artifacts))


def _group_rss(pgid):
    total=0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            tail=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(tail[2])!=pgid:continue
            for line in (entry/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):total+=1024*int(line.split()[1]);break
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return total


def run_limited(command,log,*,seconds=900,rss_bytes=4*1024**3):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',JULIA_NUM_THREADS='1')
    started=time.monotonic();peak=0;failure=None
    with log.open('w')as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        while process.poll() is None:
            rss=_group_rss(process.pid);peak=max(peak,rss)
            if time.monotonic()-started>seconds or rss>rss_bytes:
                failure='wall_budget' if time.monotonic()-started>seconds else 'process_group_rss_budget'
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                break
            time.sleep(.1)
        code=process.wait()
    return dict(returncode=code,failure=failure,operational_wall_seconds=time.monotonic()-started,peak_polled_group_rss_bytes=peak)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=PROJECT/'novak_work/validation_results/dtfe_direct_field_v1')
    p.add_argument('--julia',type=Path,default=JULIA)
    p.add_argument('--category',choices=CLASSES);p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256')
    a=p.parse_args();out=a.out.resolve()
    if a.category:
        try:case_run(a.category,out,a.plan,a.julia,a.plan_sha256)
        except BaseException as error:
            if not (out/'terminal.json').exists():write(out/'terminal.json',dict(status='failed',error=str(error),traceback=traceback.format_exc()))
            raise
        return
    if out.exists():raise ValueError('Study exists; no overwrite or silent retry')
    plan=source_plan();plan['julia_executable']=str(a.julia);plan['julia_executable_sha256']=sha(a.julia)
    out.mkdir(parents=True);write(out/'plan.json',plan);plan_sha256=sha(out/'plan.json')
    jobs=[]
    for category in CLASSES:
        verify(plan['sources_and_inputs']);directory=out/category;directory.mkdir()
        command=[sys.executable,str(Path(__file__).resolve()),'--out',str(directory),'--julia',str(a.julia),'--category',category,'--plan',str(out/'plan.json'),'--plan-sha256',plan_sha256]
        result=run_limited(command,out/(category+'.log'))
        result['category']=category
        if not (directory/'terminal.json').exists():write(directory/'terminal.json',dict(status='failed',error='Worker terminated without completed terminal',execution=result))
        result['status']=read(directory/'terminal.json')['status'];jobs.append(result)
        write(out/(category+'_execution.json'),result)
        print(category,result['status'],result['returncode'],flush=True)
    verify(plan['sources_and_inputs'])
    if sha(out/'plan.json')!=plan_sha256:raise ValueError('Execution plan changed')
    if sha(a.julia)!=plan['julia_executable_sha256']:raise ValueError('Julia executable changed')
    status='completed' if all(j['status']=='completed' and j['returncode']==0 for j in jobs) else 'incomplete'
    artifacts={str(f.relative_to(out)):sha(f)for f in sorted(out.rglob('*'))if f.is_file()}
    write(out/'terminal.json',dict(status=status,jobs=jobs,sources_and_inputs=plan['sources_and_inputs'],artifacts=artifacts))
    if status!='completed':raise SystemExit(1)


if __name__=='__main__':main()
