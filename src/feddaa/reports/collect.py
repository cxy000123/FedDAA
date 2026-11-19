"""Build updated figures only from the corrected seed012 study."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys
import numpy as np
import pandas as pd

FAMILIES = ['timexer', 'tcn', 'tide', 'mlp', 'daan']
MODES = ['non_dp', 'clip_only', 'dp']
COUNTS = dict(timexer=103969, tcn=98113, tide=2979816, mlp=3000817, daan=9073)

def select_representative_dates(origins, actual):
    """Pick distinct days closest to P10, P50 and P90 of observed daily generation."""
    if actual.shape != (len(origins), 48) or not np.isfinite(actual).all():
        raise ValueError("Expected 48 finite observations for each test date")
    if len(origins) < 3 or len(set(origins)) != len(origins):
        raise ValueError("Expected at least three unique test dates")

    # Sort chronologically so an equal distance to a quantile favors the earlier date.
    order = np.argsort(origins)
    dates = np.asarray(origins)[order]
    daily_generation = actual[order].sum(axis=1)
    targets = np.quantile(daily_generation, [0.10, 0.50, 0.90])
    available = np.ones(len(dates), dtype=bool)
    selected = []
    for target in targets:
        nearest = np.argmin(np.where(available, np.abs(daily_generation - target), np.inf))
        selected.append(str(dates[nearest]))
        available[nearest] = False
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    a = p.parse_args()
    base = a.input.resolve()
    out = base / 'figures'
    data = out / 'data'
    data.mkdir(parents=True, exist_ok=True)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    records, regions, households, curves = [], [], [], []
    reference = None
    reference_origins = None
    dates = None
    reference_roster = None
    for family in FAMILIES:
        group, name = ('daan' if family == 'daan' else 'original'), family
        for mode in MODES:
            tag = 'dp_eps_1' if mode == 'dp' else mode
            for seed in [0, 1, 2]:
                folder = base / 'tests' / group / 'test' / name / tag / f'seed_{seed}'
                e = read(folder / 'evaluation.json')
                run = e['run']
                if (run['model'], run['mode'], run['seed'], run['epsilon'], e['evaluation_split']) != (
                        name, mode, seed, 1.0 if mode == 'dp' else None, 'test'):
                    raise ValueError('Wrong figure source: ' + str(folder))
                if e['parameter_count'] != COUNTS[family]:
                    raise ValueError('Not the final retuned architecture: ' + str(folder))
                records.append(dict(family=family, mode=mode, seed=seed, parameters=e['parameter_count'],
                    training_minutes=e['source_validation']['training_seconds'] / 60,
                    household_rmse=e['household_macro_rmse_per_kw'], household_p90=e['household_p90_rmse_per_kw'],
                    worst_region_nrmse_percent=e['worst_region_nrmse_percent'],
                    **{k: e['system'][k] for k in ['rmse', 'mae', 'r2', 'daylight_rmse']}))
                if mode != 'dp':
                    continue
                for r in e['regions']:
                    regions.append(dict(family=family, seed=seed, region=r['client_id'] + 1,
                                        nrmse_percent=r['nrmse_percent']))
                hh = pd.read_csv(folder / 'households.csv', dtype={'customer': str})
                hh = hh[hh.variant == 'learned'].copy()
                roster = set(zip(hh.region, hh.customer))
                if len(roster) != len(hh) or not roster:
                    raise ValueError('Expected distinct test households')
                if reference_roster is None:
                    reference_roster = roster
                if roster != reference_roster:
                    raise ValueError('Test household roster differs across models')
                hh['family'], hh['seed'] = family, seed
                households.append(hh[['family', 'seed', 'region', 'customer', 'rmse']])
                with np.load(folder / 'predictions.npz', allow_pickle=False) as z:
                    actual = z['actual']
                    predicted = z['predicted'][list(z['variants']).index('learned')]
                    origins = list(z['origins'])
                if reference is None:
                    reference = actual.copy()
                    reference_origins = origins
                    dates = select_representative_dates(origins, actual)
                if (origins != reference_origins or actual.shape != reference.shape
                        or not np.allclose(actual, reference, rtol=0, atol=1e-10)):
                    raise ValueError('Test dates or targets differ across models or seeds')
                for date in dates:
                    i = origins.index(date)
                    curves.extend(dict(family=family, seed=seed, date=date, step=t,
                                       actual=actual[i, t], predicted=predicted[i, t]) for t in range(48))
    df = pd.DataFrame(records)
    df.to_csv(data / 'test_seed_metrics.csv', index=False)
    summary = []
    fields = ['rmse', 'mae', 'r2', 'daylight_rmse', 'household_rmse', 'household_p90',
              'worst_region_nrmse_percent', 'training_minutes']
    for family in FAMILIES:
        for mode in MODES:
            frame = df[(df.family == family) & (df['mode'] == mode)]
            if len(frame) != 3 or frame.parameters.nunique() != 1:
                raise ValueError('Incomplete or mixed architecture in figures')
            row = dict(family=family, mode=mode, parameters=int(frame.parameters.iloc[0]))
            for field in fields:
                row[field + '_mean'], row[field + '_std'] = frame[field].mean(), frame[field].std(ddof=1)
            summary.append(row)
    pd.DataFrame(summary).to_csv(data / 'test_summary.csv', index=False)
    pd.DataFrame(regions).to_csv(data / 'regions.csv', index=False)
    hh = pd.concat(households)
    hh.to_csv(data / 'households_seed.csv', index=False)
    user_keys = ['family', 'region', 'customer']
    per_user = hh.groupby(user_keys)
    if hh.duplicated(user_keys + ['seed']).any() or not per_user.seed.agg(
        lambda seeds: set(seeds) == {0, 1, 2}
    ).all():
        raise ValueError('Figure 3-3 requires RMSE from seeds 0, 1 and 2 for each included household')
    per_user.rmse.mean().reset_index().to_csv(data / 'households_mean.csv', index=False)
    pd.DataFrame(curves).to_csv(data / 'daily_curves.csv', index=False)
    pd.DataFrame({'date': dates}).to_csv(data / 'representative_dates.csv', index=False)
    weather = base / "weather/weather_summary.csv"
    if weather.is_file():
        ws = pd.read_csv(weather)
        if len(ws) == 25 and (ws.status == "complete").all():
            shutil.copy2(weather, data / "weather_summary.csv")
    subprocess.run([sys.executable, "-m", "feddaa.reports.figures",
                    "--input", str(data), "--output", str(out)], check=True)
    print('Updated corrected-study figures saved:', out, flush=True)

if __name__ == '__main__':
    main()
