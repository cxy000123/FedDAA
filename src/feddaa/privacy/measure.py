"""Measure single-round output distributions from a DAAN checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import secrets
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def flat(state,keys):
    return torch.cat([state[k].detach().cpu().double().reshape(-1) for k in keys]).numpy()


def load_updates(a,privacy):
    from feddaa import federated as fed
    folder=a.checkpoint.parent
    if digest(a.checkpoint)!=read(folder/'validation.json')['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    config=checkpoint['config'];spec=checkpoint['model_spec']
    if spec['model'] not in ('daan', 'lead_fusion'):
        raise ValueError('Expected the DAAN checkpoint')
    # fed.make_model also checks the exact architecture of archived checkpoints.
    if config.get('scope')!='all':raise ValueError('Expected all-parameter DP')
    if digest(a.fixed_scalers)!=config['fixed_scalers_sha256']:
        raise ValueError('Scaler mismatch; use the scalers paired with the checkpoint')
    hashes=[digest(a.data_dir/f'client_{i}_households.parquet') for i in range(3)]
    if hashes!=config['data_sha256']:raise ValueError('Dataset mismatch')
    stores,scalers,device=fed.context(SimpleNamespace(data_dir=a.data_dir,
                           fixed_scalers=a.fixed_scalers,device=a.device))
    store=stores[a.region]
    if a.households>store.n_customers:raise ValueError('Probe household count exceeds public region roster')
    if [s.n_customers for s in stores]!=privacy['regional_households']:
        raise ValueError('Roster mismatch')
    factory=lambda:fed.make_model(spec,checkpoint['past_dim'],checkpoint['future_dim'],config['history_steps'])
    model=factory();keys=fed.parameter_keys(model,config['scope'])
    state=checkpoint['state'];updates=[]
    # Public roster order, not selection by update norms or forecast errors.
    ids=sorted(store.customer_ids)[:a.households]
    for i,customer in enumerate(ids):
        torch.manual_seed(20261013+i)
        data=store.build_data(customer,scalers,config['train_stride_hours'],
                              config['future_weather_mode'],include_test=False)
        if not len(data['splits']['train']['origin']):
            raise ValueError(f'No valid training windows for probe household {i+1}')
        local=fed.train_local(state,data,keys,config['local_epochs'],config['batch_size'],
            config['lr'],config['weight_decay'],device,
            config['gradient_clip'],model_factory=factory)
        updates.append({k:local[k]-state[k] for k in keys})
        del data,local
        print(f'Local probe update {i+1}/{len(ids)} (training split only)',flush=True)
    template={k:torch.zeros_like(state[k],device='cpu') for k in keys}
    return updates,template,keys,{'synthetic':False,'checkpoint_sha256':digest(a.checkpoint),
        'probe_households':len(ids),'public_region_roster':store.n_customers,
        'remaining_household_contributions':'zero in this fixed-update mechanism diagnostic',
        'input_split':'train','checkpoint_is_fixed':'no model retraining or test selection'}


def run_probe(updates,template,keys,privacy,repetitions,progress=None):
    from feddaa import federated as base
    C=float(privacy['clip_norm']);sigma=float(privacy['noise_multiplier'])
    N=float(privacy['normalization']);bound=float(privacy['max_protected_update_norm'])
    aliases=[f'H{i+1:02d}' for i in range(len(updates))]
    vectors=[flat(u,keys) for u in updates]
    if any(np.sqrt(np.sum(np.square(v)))<=1e-12 for v in vectors):
        raise ValueError('Zero update has no target-aligned direction')
    # Directions fixed from the local updates BEFORE inspecting DP releases.
    directions=np.stack([v/np.sqrt(np.sum(np.square(v))) for v in vectors])
    backgrounds=[[u for j,u in enumerate(updates) if j!=i] for i in range(len(updates))]
    controls=[];control_detail=[]
    for mode in ['non_dp','clip_only']:
        included=base.regional_update(updates,template,mode,C,sigma,N,0,bound)
        included_vec=flat(included,keys)
        for i,background in enumerate(backgrounds):
            suppressed=base.regional_update(background,template,mode,C,sigma,N,0,bound)
            suppressed_vec=flat(suppressed,keys)
            baseline=float(np.einsum('i,i->', suppressed_vec, directions[i]))
            projection=float(np.einsum('i,i->', included_vec, directions[i]))
            for neighbor,value in [('included',projection),('suppressed',baseline)]:
                controls.append({'household':aliases[i],'mode':mode,'neighbor':neighbor,
                                 'repetition':0,'projection_raw':value,
                                 'projection_centered':value-baseline,'centering_baseline':baseline})
            control_detail.append({'household':aliases[i],'mode':mode,
                                   'projection_change':projection-baseline,
                                   'whole_vector_change_norm':float(np.sqrt(np.sum(np.square(included_vec-suppressed_vec))))})
    secure=secrets.SystemRandom()
    inc=np.empty((repetitions,len(updates)),dtype=float)
    exc=np.empty_like(inc);norms=[]
    for repetition in range(repetitions):
        # One included release may be projected along all fixed user directions.
        # Suppressed releases use independent fresh noise, including across users.
        release=base.regional_update(updates,template,'dp',C,sigma,N,secure.getrandbits(63),bound)
        vector=flat(release,keys);inc[repetition]=np.einsum('ij,j->i', directions, vector)
        norms.append(float(np.sqrt(np.sum(np.square(vector)))))
        for i,background in enumerate(backgrounds):
            release=base.regional_update(background,template,'dp',C,sigma,N,secure.getrandbits(63),bound)
            vector=flat(release,keys);exc[repetition,i]=np.einsum('i,i->', vector, directions[i])
            norms.append(float(np.sqrt(np.sum(np.square(vector)))))
        if progress and ((repetition+1)%100==0 or repetition+1==repetitions):
            progress(repetition+1,repetitions)
    if max(norms)>bound+1e-5:raise AssertionError('Released output norm exceeds configured final bound')
    samples=list(controls)
    # Within each mode/user, translate BOTH neighbor datasets by the same baseline.
    # Translation changes neither separation nor overlap; no scaling or matching is used.
    for i,alias in enumerate(aliases):
        baseline=float(exc[:,i].mean())
        for neighbor,values in [('included',inc[:,i]),('suppressed',exc[:,i])]:
            for repetition,value in enumerate(values):
                samples.append({'household':alias,'mode':'dp','neighbor':neighbor,
                    'repetition':repetition,'projection_raw':float(value),
                    'projection_centered':float(value-baseline),'centering_baseline':baseline})
    info={'directions':'known target-update directions, fixed before DP sampling',
          'cohort':'fixed probe cohort; no resampling of household membership in this experiment',
          'dp_noise':'fresh independent secret seeds for each release',
          'postprocessing':'actual final whole-vector norm bound applied',
          'centering':'controls subtract deterministic suppressed output; DP subtracts empirical suppressed mean; both neighbors share the same shift',
          'max_measured_dp_release_norm':max(norms),'configured_release_norm_bound':bound,
          'dp_outputs_per_neighbor_per_user':repetitions,
          'included_release_shared_across_directions':True}
    return pd.DataFrame(samples),pd.DataFrame(control_detail),info


def report(output):
    samples=pd.read_csv(output/'empirical_projections_INTERNAL.csv')
    controls=pd.read_csv(output/'deterministic_controls.csv')
    meta=read(output/'experiment_manifest.json')
    H=int(meta['probe_households']);T=int(meta['repetitions'])
    aliases=[f'H{i+1:02d}' for i in range(H)]
    expected_count=H*(2*T+4)
    if len(samples)!=expected_count:raise ValueError('Incomplete sample file')
    for alias in aliases:
        for neighbor in ['included','suppressed']:
            part=samples[(samples.household==alias)&(samples['mode']=='dp')&(samples.neighbor==neighbor)]
            if len(part)!=T or part.repetition.nunique()!=T:raise ValueError('Missing/duplicate DP repetitions')
    if not np.isfinite(samples[['projection_raw','projection_centered']].to_numpy()).all():
        raise ValueError('Nonfinite samples')
    summary=samples.groupby(['household','mode','neighbor']).projection_centered.agg(
        count='count',mean='mean',std='std',median='median',minimum='min',maximum='max').reset_index()
    summary.to_csv(output/'empirical_summary.csv',index=False)
    selected=meta['display_target']
    if selected not in aliases:raise ValueError('Unknown display target')
    def deterministic(mode):
        return controls[controls['mode']==mode].set_index('household').reindex(aliases).projection_change.to_numpy()
    raw=deterministic('non_dp');clipped=deterministic('clip_only')
    retention=np.divide(clipped,raw,out=np.full(H,np.nan),where=abs(raw)>1e-12)*100
    if not np.isfinite(retention).all():raise ValueError('Zero raw projection in retention ratio')
    df=pd.DataFrame({'household':aliases,'raw_projection_change':raw,
                     'clipped_projection_change':clipped,'clipping_retained_percent':retention})
    df.to_csv(output/'contribution_comparison.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--fixed-scalers',type=Path,default=Path('data/scalers.json'))
    p.add_argument('--output',type=Path,default=Path('runs/privacy'))
    p.add_argument('--region',type=int,choices=[0,1,2],default=2)
    p.add_argument('--households',type=int,default=8)
    p.add_argument('--repetitions',type=int,default=1000)
    p.add_argument('--display-target',default='H01',help='Display alias fixed before simulation, not selected for best overlap')
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--plot-only',action='store_true')
    a=p.parse_args()
    if a.plot_only:report(a.output);return
    if not 1<=a.households<=8:p.error('Choose 1 to 8 probe households')
    if a.repetitions<100:p.error('Use at least 100 repetitions')
    if a.threads<1:p.error('Threads must be positive')
    if a.display_target not in [f'H{i+1:02d}' for i in range(a.households)]:p.error('Display target outside cohort')
    if (a.output/'experiment_manifest.json').exists():
        raise FileExistsError('Use --plot-only to redraw completed results, or choose a new output directory')
    torch.set_num_threads(a.threads)
    privacy=read(a.checkpoint.parent/'privacy_report.json')
    updates,template,keys,source=load_updates(a,privacy)
    a.output.mkdir(parents=True,exist_ok=True)
    manifest={'source':source,'probe_households':a.households,'region':a.region,
        'repetitions':a.repetitions,'display_target':a.display_target,
        'experiment':'measured fixed-cohort, single-round conditional Gaussian mechanism with actual final bounding',
        'all_plot_values':'actual deterministic controls and sampled mechanism outputs; violin KDE is a visualization of samples',
        'target_directions':'privileged diagnostic information from known local updates, not an attacker observation assumption',
        'privacy_scope':'not a new epsilon=1 release or an end-to-end training audit',
        'publication':'internal research outputs; repeated releases and non-DP controls must not be treated as one DP-protected release',
        'source_training_noise_configuration':privacy,
        'script_sha256':digest(Path(__file__)),'status':'running'}
    (a.output/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    samples,controls,info=run_probe(updates,template,keys,privacy,a.repetitions,
                   progress=lambda n,total:print(f'Measured DP repetitions {n}/{total}',flush=True))
    samples.to_csv(a.output/'empirical_projections_INTERNAL.csv',index=False)
    controls.to_csv(a.output/'deterministic_controls.csv',index=False)
    manifest.update(info);manifest['status']='complete'
    (a.output/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report(a.output)
    print('ALL EMPIRICAL DAAN PRIVACY FIGURES COMPLETE',flush=True)


if __name__ == "__main__":
    main()
