"""Recreate publication figures from the included data/ CSVs.
Usage: python plots/main.py [--input FOLDER] [--output FOLDER]
Dependencies: numpy pandas matplotlib. No checkpoints or training data required.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

from feddaa import CJK_FONT
from feddaa.reports.extras import FAMILIES, MODEL_LABELS, cost, weather


def mean_daily_curves(curves, dates, families):
    """Figure 3-4 uses one test prediction per slot from each seed 0, 1 and 2."""
    seeds = (0, 1, 2)
    keys = ["date", "family", "step", "seed"]
    expected = pd.MultiIndex.from_product([dates, families, range(48), seeds], names=keys)
    selected = curves.loc[curves["seed"].isin(seeds) & curves["date"].isin(dates)
                          & curves["family"].isin(families)].set_index(keys)
    if selected.index.has_duplicates or not expected.difference(selected.index).empty:
        raise ValueError("Figure 3-4 requires one prediction per date, model, slot and seed (0, 1, 2).")
    values = selected.loc[expected, ["actual", "predicted"]].copy()
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Figure 3-4 requires finite observations and predictions.")
    actual_by_slot = values["actual"].groupby(level=["date", "step"])
    if not np.allclose(actual_by_slot.min(), actual_by_slot.max(), rtol=0, atol=1e-10):
        raise ValueError("Figure 3-4 observations differ across models or seeds.")
    values["abs_error"] = (values["predicted"] - values["actual"]).abs()
    return actual_by_slot.first(), values.groupby(level=keys[:-1])[["predicted", "abs_error"]].mean()


def rmse_growth(summary):
    """Use each model's three-run mean without DP as its baseline (Eq. 28)."""
    means = summary.pivot(index="family", columns="mode", values="rmse_mean")
    return 100 * means[["clip_only", "dp"]].div(means["non_dp"], axis=0).sub(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('runs/main/figures/data'))
    p.add_argument('--output', type=Path, default=Path('runs/main/figures'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    configure_font = CJK_FONT
    font_manager.fontManager.addfont(configure_font)
    chinese = font_manager.FontProperties(fname=configure_font).get_name()
    available = {f.name for f in font_manager.fontManager.ttflist}
    latin = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({'font.family':[latin, chinese, 'DejaVu Serif'],
        'font.size':10, 'axes.labelsize':11, 'xtick.labelsize':10, 'ytick.labelsize':10,
        'axes.unicode_minus':False, 'axes.linewidth':.8, 'pdf.fonttype':42,
        'svg.fonttype':'none', 'legend.frameon':False, 'savefig.facecolor':'white'})
    families = FAMILIES
    names = MODEL_LABELS
    colors = ['#53789C','#6F9690','#B59667','#8B81A0','#BD5955']
    modes = ['non_dp','clip_only','dp']
    labels = ['无DP','仅裁剪','DP（ε=1）']
    sm = pd.read_csv(a.input / 'test_summary.csv')
    def tidy(ax):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', color='#D9D9D9', linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out', length=3)
    def save(fig,name):
        for ext in ['png','pdf','svg']:
            fig.savefig(a.output / (name+'.'+ext), dpi=300, bbox_inches='tight', pad_inches=.06)
        plt.close(fig)
    # Figure 3-1: three-seed means; no error bars.
    fig, ax = plt.subplots(figsize=(7.7,3.65), layout='constrained')
    x = np.arange(5); w=.23
    for i,(m,label,c) in enumerate(zip(modes,labels,['#A9BCCB','#6B938D','#BD6B60'])):
        v = sm[sm['mode']==m].set_index('family').loc[families]
        ax.bar(x+(i-1)*w,v.rmse_mean,w,label=label,color=c,edgecolor='#444444',linewidth=.45,zorder=3)
    ax.set_xticks(x,names); ax.set_ylabel('系统RMSE / kWh'); ax.set_ylim(0,.092)
    ax.yaxis.set_major_locator(MultipleLocator(.02))
    ax.legend(ncols=3,loc='upper center'); tidy(ax)
    save(fig,'图3-1_三种训练条件的系统RMSE')
    # Figure 3-2: relative growth of the three-run mean system RMSE.
    growth = rmse_growth(sm)
    fig, ax = plt.subplots(figsize=(7.7,3.65),layout='constrained')
    for i,(m,label,c) in enumerate(zip(modes[1:],labels[1:],['#6B938D','#BD6B60'])):
        values = growth.loc[families, m]
        positions = x + (i - .5) * .32
        ax.bar(positions, values, .30, label=label, color=c, edgecolor='#444444', linewidth=.45, zorder=3)
        for position, value in zip(positions, values):
            ax.annotate(f'{value:.1f}', (position, value), xytext=(0, 4),
                        textcoords='offset points', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x,names); ax.set_ylabel('相对无DP的RMSE增长率 / %'); ax.set_ylim(0,145)
    ax.legend(ncols=2,loc='upper center'); tidy(ax)
    save(fig,'图3-2_相对无DP的RMSE增长率')
    # Figure 3-3: region means and empirical CDF of per-user three-run means.
    region = pd.read_csv(a.input / 'regions.csv')
    hh = pd.read_csv(a.input / 'households_mean.csv')
    fig, axes = plt.subplots(1,2,figsize=(10.4,3.75),layout='constrained')
    for i,(f,name,c) in enumerate(zip(families,names,colors)):
        v=region[region.family==f].groupby('region').nrmse_percent.agg(['mean','std']).loc[[1,2,3]]
        axes[0].bar(np.arange(3)+(i-2)*.15,v['mean'],.14,color=c,label=name,
            edgecolor='#444444',linewidth=.35,zorder=3)
        vals=np.sort(hh[hh.family==f].rmse.to_numpy())
        if len(vals) == 0:
            raise ValueError(f'No household errors for {name}')
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=c,label=name,
                     linewidth=1.8 if f=='daan' else 1.2)
    axes[0].set_xticks(np.arange(3),['区域1','区域2','区域3'])
    axes[0].set_ylabel('NRMSE / %'); axes[0].set_ylim(0,15)
    axes[0].set_title('（a）区域预测误差',fontsize=11,pad=8)
    axes[1].set_xlabel('用户三种子平均RMSE / (kWh/kW)'); axes[1].set_ylabel('累积概率')
    axes[1].set_ylim(0,1.01); axes[1].set_xlim(.025,.105)
    axes[1].set_title('（b）用户误差经验累积分布',fontsize=11,pad=8)
    axes[1].legend(loc='lower right',fontsize=9)
    for ax in axes: tidy(ax)
    save(fig,'图3-3_区域误差与用户累积分布')
    # Figure 3-4: mean prediction and mean absolute error across seeds 0, 1 and 2.
    dates = pd.read_csv(a.input / 'representative_dates.csv')['date'].tolist()
    curves = pd.read_csv(a.input / 'daily_curves.csv')
    actual, means = mean_daily_curves(curves, dates, families)
    fig,axes=plt.subplots(2,3,figsize=(11.0,5.4),sharex=True,layout='constrained')
    hours = np.arange(48) / 2
    for col,date in enumerate(dates):
        axes[0,col].plot(hours,actual.loc[date],color='#222222',lw=1.7,label='实际值',zorder=5)
        for f,name,c in zip(families,names,colors):
            agg = means.loc[(date, f)]
            axes[0,col].plot(hours,agg.predicted,color=c,lw=1.5 if f=='daan' else 1.0,label=name)
            axes[1,col].plot(hours,agg.abs_error,color=c,lw=1.5 if f=='daan' else 1.0)
        axes[0,col].set_title(date,fontsize=10)
        axes[1,col].set_xlabel('时刻 / h'); axes[1,col].set_xticks([0,6,12,18,24]); axes[1,col].set_xlim(0,24)
        for ax in axes[:,col]: tidy(ax)
    axes[0,0].set_ylabel('半小时发电量 / kWh'); axes[1,0].set_ylabel('绝对误差 / kWh')
    handles,leglabels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,leglabels,ncols=6,loc='outside lower center',fontsize=9)
    save(fig,'图3-4_代表日期曲线_同步更新')
    cost(a.input, a.output)
    if (a.input / 'weather_summary.csv').is_file():
        weather(a.input, a.output)
    print('Figures saved to', a.output)

if __name__=='__main__':
    main()
