#!/usr/bin/env python3
"""Frozen seed012 DP checkpoints: future-weather perturbation test, no training."""
import argparse, hashlib, json, math, os, shutil
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
MODELS={'timexer':'TimeXer','tcn':'TCN','tide':'TiDE','mlp':'MLP','daan':'DAAN'}
SEEDS=[0,1,2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen(p,value):
    if p.exists():
        if read(p)!=value:raise ValueError('Changed frozen settings; use a new output: '+str(p))
    else:p.write_text(json.dumps(value,indent=2),encoding='utf-8')
def source(a,model,seed):
    group='daan' if model=='daan' else 'original'
    rid=Path(model)/'dp_eps_1'/f'seed_{seed}'
    return a.main_results/'training'/group/'results'/rid,a.main_results/'tests'/group/'test'/rid,rid
def jobs(a):
    return [(model,level,noise,seed) for model in a.models for level in a.levels
            for noise in a.noise_seeds for seed in SEEDS]
def target(a,model,level,noise,rid):
    return a.output/'results'/model/f'level_{level:g}'/f'noise_seed_{noise}'/'test'/rid
def summarize(a):
    import numpy as np
    import pandas as pd
    rows=[];missing=[]
    for model,level,noise,seed in jobs(a):
        _,_,rid=source(a,model,seed);folder=target(a,model,level,noise,rid)
        if not (folder/'complete.json').exists() or not (folder/'evaluation.json').exists():
            missing.append({'model':model,'level':level,'noise_seed':noise,'training_seed':seed});continue
        v=read(folder/'evaluation.json')
        if v.get('evaluation_split')!='test' or v['run']['seed']!=seed or v['run']['epsilon']!=1:
            raise ValueError('Unexpected test source: '+str(folder))
        rows.append({'model':MODELS[model],'level':level,'noise_seed':noise,'training_seed':seed,
                     **v['system'],'household_rmse':v['household_macro_rmse_per_kw'],
                     'household_p90':v['household_p90_rmse_per_kw'],
                     'worst_region_nrmse_percent':v['worst_region_nrmse_percent']})
    (a.output/'missing_runs.json').write_text(json.dumps(missing,indent=2),encoding='utf-8')
    if not rows:return missing
    data=pd.DataFrame(rows);data.to_csv(a.output/'weather_seed_metrics.csv',index=False)
    fields=['rmse','mae','r2','daylight_rmse','household_rmse','household_p90','worst_region_nrmse_percent']
    records=[]
    for model in a.models:
        for level in a.levels:
            frame=data[(data.model==MODELS[model])&(data.level==level)]
            groups=frame.groupby('training_seed')
            ready=set(frame.training_seed)==set(SEEDS) and all(set(g.noise_seed)==set(a.noise_seeds) for _,g in groups)
            byseed=groups[fields].mean()
            item={'model':MODELS[model],'level':level,'n_training_seeds':len(byseed),
                  'status':'complete' if ready else 'pending'}
            for f in fields:
                item[f+'_mean']=float(byseed[f].mean()) if ready else np.nan
                item[f+'_std']=float(byseed[f].std(ddof=1)) if ready else np.nan
            records.append(item)
    summary=pd.DataFrame(records)
    clean=summary[summary.level==0].set_index('model').rmse_mean
    summary['rmse_change_percent']=[100*(v/clean[n]-1) if np.isfinite(clean[n]) else np.nan
                                   for n,v in zip(summary.model,summary.rmse_mean)]
    summary.to_csv(a.output/'weather_summary.csv',index=False)
    md=['# Weather sensitivity, training seeds 0 / 1 / 2','',
        '| Model | Noise SD / train weather SD | Seeds | RMSE | MAE | R2 | RMSE change (%) |',
        '|---|---|---|---|---|---|---|']
    for _,r in summary.iterrows():
        vals=[f'{r.rmse_mean:.6f}',f'{r.mae_mean:.6f}',f'{r.r2_mean:.6f}',f'{r.rmse_change_percent:.4f}'] if r.status=='complete' else ['pending']*4
        md.append('| '+' | '.join([r.model,f'{100*r.level:g}%',str(r.n_training_seeds)]+vals)+' |')
    (a.output/'weather_results.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f'Weather test completions: {len(rows)}/{len(jobs(a))}',flush=True)
    return missing
def perturb_future(future,origins,scalers,level,noise_seed,rho=.8):
    import numpy as np
    from feddaa.data.windows import WEATHER_COLUMNS,SOLAR_FEATURE_NAMES
    result=np.array(future,copy=True)
    if level==0: return result
    nw=len(WEATHER_COLUMNS); solar_start=4*nw+3
    mean=np.asarray(scalers.weather_mean); std=np.asarray(scalers.weather_std)
    names=['temp_c_mean','dewpoint_c_mean','wind_speed_mps_mean','precip_mm_mean','silo_solar_exposure_mj_m2_mean']
    columns=[WEATHER_COLUMNS.index(name) for name in names]
    raw=result[:,:,:nw].astype(float)*std+mean
    for day,date in enumerate(origins):
        for name,column in zip(names,columns):
            digest=hashlib.sha256((str(date)+'|'+name+'|'+str(noise_seed)).encode()).digest()
            rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
            if name=='silo_solar_exposure_mj_m2_mean':
                error=np.repeat(rng.normal(),48)  # Daily exposure is repeated within each day.
            else:
                innovations=rng.normal(size=48); error=np.empty(48); error[0]=innovations[0]
                for t in range(1,48): error[t]=rho*error[t-1]+np.sqrt(1-rho*rho)*innovations[t]
            raw[day,:,column]+=level*std[column]*error
    for name in ['wind_speed_mps_mean','precip_mm_mean','silo_solar_exposure_mj_m2_mean']:
        column=WEATHER_COLUMNS.index(name); raw[:,:,column]=np.maximum(raw[:,:,column],0.)
    ti,di=WEATHER_COLUMNS.index('temp_c_mean'),WEATHER_COLUMNS.index('dewpoint_c_mean')
    raw[:,:,di]=np.minimum(raw[:,:,di],raw[:,:,ti])
    exposure=raw[:,:,WEATHER_COLUMNS.index('silo_solar_exposure_mj_m2_mean')]
    sky=future[:,:,solar_start+SOLAR_FEATURE_NAMES.index('clear_sky_shape')]
    daylight=future[:,:,solar_start+SOLAR_FEATURE_NAMES.index('daylight_flag')]
    day_length=future[:,:,solar_start+SOLAR_FEATURE_NAMES.index('day_length_fraction')]
    shares=np.divide(sky,sky.sum(1,keepdims=True),out=np.zeros_like(sky),where=sky.sum(1,keepdims=True)>0)
    ratio=exposure/np.maximum(day_length*24.,1e-3)
    raw[:,:,WEATHER_COLUMNS.index('silo_clear_sky_ratio')]=np.where(daylight>.5,ratio,0.)
    raw[:,:,WEATHER_COLUMNS.index('silo_halfhour_radiation_shape')]=exposure*shares
    changed=columns+[WEATHER_COLUMNS.index('silo_clear_sky_ratio'),WEATHER_COLUMNS.index('silo_halfhour_radiation_shape')]
    result[:,:,changed]=((raw[:,:,changed]-mean[changed])/std[changed]).astype(result.dtype)
    return result


def evaluate(a, model, level, noise, pending):
    from feddaa.evaluate import evaluate as evaluate_runs, predict

    def noisy_predict(network, data, split, variants, batch_size, device):
        original = data["splits"][split]
        changed = dict(original)
        changed["future"] = perturb_future(original["future"], original["origin"],
                                           data["scalers"], level, noise, a.rho)
        modified = dict(data, splits=dict(data["splits"], **{split: changed}))
        return predict(network, modified, split, variants, batch_size, device)

    output = a.output / "results" / model / f"level_{level:g}" / f"noise_seed_{noise}"
    evaluate_runs(SimpleNamespace(output=output, data_dir=a.data_dir, fixed_scalers=a.fixed_scalers,
                                  device=a.device, threads=a.threads, split="test"),
                  pending, noisy_predict, compare_prior=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-results',type=Path,default=Path('runs/main'))
    p.add_argument('--output',type=Path,default=Path('runs/main/weather'))
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--fixed-scalers',type=Path,default=Path('data/scalers.json'))
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--models',nargs='+',choices=list(MODELS),default=['daan','timexer','tcn','tide','mlp'])
    p.add_argument('--levels',type=float,nargs='+',default=[0,.05,.1,.15,.2])
    p.add_argument('--noise-seeds',type=int,nargs='+',default=[2026]);p.add_argument('--rho',type=float,default=.8)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    if a.threads<1 or not 0<=a.rho<1 or 0 not in a.levels or any(not math.isfinite(v) or v<0 for v in a.levels):
        p.error('Invalid threads/rho/noise levels (include 0)')
    for values in [a.models,a.levels,a.noise_seeds]:
        if len(values)!=len(set(values)):p.error('Duplicate settings')
    for key in ['main_results','output','data_dir','fixed_scalers']:setattr(a,key,getattr(a,key).resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    if a.report_only:summarize(a);return
    checkpoints={}
    for model in a.models:
        group='daan' if model=='daan' else 'original'
        plan=read(a.main_results/'training'/group/'plan.json')
        if plan['future_weather_mode']!='observed':raise ValueError('Expected observed future weather baseline')
        for seed in SEEDS:
            folder,clean,rid=source(a,model,seed)
            for path in [folder/'complete.json',folder/'validation.json',folder/'privacy_report.json',clean/'complete.json',clean/'evaluation.json']:
                if not path.exists():raise FileNotFoundError('Finish seed012 main experiment: '+str(path))
            privacy=read(folder/'privacy_report.json')
            if privacy['target_epsilon']!=1 or privacy['delta']!=1e-5:raise ValueError('DP conditions differ')
            value=read(clean/'evaluation.json');sha=read(folder/'validation.json')['checkpoint_sha256']
            if value.get('evaluation_split')!='test' or value['run']['mode']!='dp' or value['run']['seed']!=seed or value['checkpoint_sha256']!=sha:
                raise ValueError('Clean-test source mismatch')
            checkpoints[f'{model}/{seed}']=sha
    settings={'models':a.models,'training_seeds':SEEDS,'levels':a.levels,'weather_noise_seeds':a.noise_seeds,
        'rho':a.rho,'checkpoint_hashes':checkpoints,'script_sha256':digest(Path(__file__)),
        'perturbation_sha256':digest(Path(__file__)),
        'future_weather':'additive errors scaled by original TRAINING weather SD',
        'frozen_fields':'history, lagged weather, PV-derived inputs, capacity, calendar and labels',
        'modified_fields':'future temperature, dewpoint, wind, precipitation, exposure and derived radiation features',
        'selection':'fixed checkpoints; no retraining or test-based model selection'}
    if a.prepare_only:
        frozen(a.output/'planned_settings.json',settings)
        print(f'Planned {len(jobs(a))} evaluations; {len(a.models)*3*len(a.noise_seeds)} reuse clean NEW-seed tests; remaining are perturbation tests',flush=True);return
    hashes=[digest(a.data_dir/f'client_{i}_households.parquet') for i in range(3)]
    scaler=digest(a.fixed_scalers)
    for model in a.models:
        for seed in SEEDS:
            folder,clean,_=source(a,model,seed)
            if digest(folder/'model.pt')!=checkpoints[f'{model}/{seed}']:raise ValueError('Checkpoint changed')
            marker=read(clean/'complete.json')
            if marker['data_sha256']!=hashes or marker['fixed_scalers_sha256']!=scaler:raise ValueError('Data/scaler mismatch')
    settings.update(data_hashes=hashes,scaler_hash=scaler)
    frozen(a.output/'experiment.json',settings)
    for model in a.models:
        for level in a.levels:
            for noise in a.noise_seeds:
                pending=[]
                for seed in SEEDS:
                    folder,clean,rid=source(a,model,seed);dest=target(a,model,level,noise,rid)
                    if (dest/'complete.json').exists():
                        if read(dest/'complete.json')!=read(clean/'complete.json'):
                            raise ValueError('Completed weather test signature mismatch: '+str(dest))
                        if not all((dest/name).exists() for name in ['evaluation.json','predictions.npz','households.csv']):
                            raise FileNotFoundError('Completed weather test files missing: '+str(dest))
                        continue
                    if level==0:
                        dest.mkdir(parents=True,exist_ok=True)
                        for name in ['evaluation.json','predictions.npz','households.csv','complete.json']:
                            shutil.copy2(clean/name,dest/name)
                        print('Reused NEW seed clean test:',model,seed,flush=True)
                    else:pending.append((folder,checkpoints[f'{model}/{seed}']))
                if pending:
                    print('STAGE WEATHER',model,'level',level,'noise_seed',noise,flush=True)
                    evaluate(a,model,level,noise,pending)
                summarize(a)
    if summarize(a):raise RuntimeError('Missing weather tests')
    print('ALL SEED012 WEATHER TESTS COMPLETE',flush=True)
if __name__=='__main__':main()

