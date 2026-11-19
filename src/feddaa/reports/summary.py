"""Summarize the three training seeds for each test condition."""

import csv
import json
from pathlib import Path
import statistics


SEEDS=[0,1,2]


NAMES={'timexer':'TimeXer','tcn':'TCN','tide':'TiDE','mlp':'MLP','daan':'DAAN'}


FIELDS=['rmse','mae','r2','nrmse_percent','daylight_rmse','daylight_mae',
        'worst_region_nrmse_percent','household_rmse','household_p90','parameters','training_minutes']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def frozen(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if read(p)!=v:raise ValueError('Changed frozen configuration; use a new output: '+str(p))
    else:p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def write_csv(p,rows):
    if rows:
        with p.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def paths(output,label,run):
    return output/'training'/label/'results'/run['id'],output/'tests'/label/'test'/run['id']


def summarize(output,plans):
    rows=[];missing=[]
    for label,plan in plans.items():
        for run in plan['runs']:
            train,test=paths(output,label,run)
            required=[test/'complete.json',test/'evaluation.json',train/'complete.json',
                      train/'privacy_report.json',train/'validation.json']
            if not all(p.exists() for p in required):
                missing.append({'group':label,'run':run['id']});continue
            value=read(test/'evaluation.json');privacy=read(train/'privacy_report.json')
            if value.get('evaluation_split')!='test' or value['run']!=run:raise ValueError('Test condition mismatch')
            if read(test/'complete.json')['checkpoint_sha256']!=value['checkpoint_sha256']:
                raise ValueError('Test checkpoint signature mismatch')
            if read(train/'validation.json')['checkpoint_sha256']!=value['checkpoint_sha256']:
                raise ValueError('Validation checkpoint signature mismatch')
            if run['mode']=='dp' and privacy['achieved_epsilon']>run['epsilon']+1e-6:
                raise ValueError('Privacy budget exceeded')
            s=value['system'];rows.append({'model':NAMES[run['model']],'mode':run['mode'],
                'epsilon':run['epsilon'] if run['mode']=='dp' else -1.,
                'delta':plan['delta'] if run['mode']=='dp' else -1.,'seed':run['seed'],
                **{f:s[f] for f in ['rmse','mae','r2','nrmse_percent','daylight_rmse','daylight_mae']},
                'worst_region_nrmse_percent':value['worst_region_nrmse_percent'],
                'household_rmse':value['household_macro_rmse_per_kw'],
                'household_p90':value['household_p90_rmse_per_kw'],
                'parameters':value['parameter_count'],'training_minutes':value['source_validation']['training_seconds']/60,
                'test_file':str(test/'evaluation.json')})
    write_csv(output/'test_seed_metrics.csv',rows)
    conditions=[]
    for plan in plans.values():
        for run in plan['runs']:
            k=(NAMES[run['model']],run['mode'],run['epsilon'] if run['mode']=='dp' else -1.)
            if k not in conditions:conditions.append(k)
    summaries=[]
    for model,mode,eps in conditions:
        group=[r for r in rows if (r['model'],r['mode'],r['epsilon'])==(model,mode,eps)]
        complete=len(group)==3 and {r['seed'] for r in group}==set(SEEDS)
        item={'model':model,'mode':mode,'epsilon':eps,'n_seeds':len(group),
              'status':'complete' if complete else 'pending'}
        for f in FIELDS:
            item[f+'_mean']=statistics.mean(r[f] for r in group) if complete else None
            item[f+'_std']=statistics.stdev(r[f] for r in group) if complete else None
        summaries.append(item)
    write_csv(output/'test_summary.csv',summaries)
    (output/'missing_runs.json').write_text(json.dumps(missing,indent=2),encoding='utf-8')
    text=['# Seeds 0 / 1 / 2: test results','',
        '| Model | Mode | Epsilon | Seeds | RMSE mean +/- SD | MAE | R2 | Minutes |',
        '|---|---|---|---|---|---|---|---|']
    for r in summaries:
        values=([f"{r['rmse_mean']:.6f} +/- {r['rmse_std']:.6f}",f"{r['mae_mean']:.6f}",
                 f"{r['r2_mean']:.6f}",f"{r['training_minutes_mean']:.2f}"] if r['status']=='complete' else ['pending']*4)
        text.append('| '+' | '.join([r['model'],r['mode'],str(r['epsilon']),str(r['n_seeds'])]+values)+' |')
    (output/'test_results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Completed test runs:',len(rows),'/',sum(len(p['runs']) for p in plans.values()),flush=True)
    return missing
