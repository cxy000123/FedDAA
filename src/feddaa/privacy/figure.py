"""Redraw the measured DAAN privacy mechanism results with Chinese labels.
No training, resampling, or changes to measured values are performed.
Dependencies: numpy, pandas, matplotlib and a Chinese font.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

from feddaa import CJK_FONT
FIGURE_NAME = '图3-5_裁剪与加噪对单用户贡献及聚合输出的影响_中文版'


def chinese_font(font_file):
    selected = font_file or (CJK_FONT if CJK_FONT.is_file() else None)
    if selected is not None:
        if not selected.is_file():
            raise FileNotFoundError(f"Chinese font file not found: {selected}")
        font_manager.fontManager.addfont(selected)
        return font_manager.FontProperties(fname=selected).get_name()
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("SimSun", "Noto Serif CJK SC", "Source Han Serif SC", "Microsoft YaHei",
                 "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Zen Hei"):
        if name in available:
            return name
    raise RuntimeError("No Chinese font found. Use the updated archive or supply --font-file.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/privacy/seed_0'))
    p.add_argument('--output',type=Path,default=Path('runs/privacy/seed_0/chinese_figure'))
    p.add_argument('--font-file', type=Path, help='Optional path to a Chinese TTF, OTF or TTC font')
    a=p.parse_args()
    samples=pd.read_csv(a.input/'empirical_projections_INTERNAL.csv')
    controls=pd.read_csv(a.input/'deterministic_controls.csv')
    meta=json.loads((a.input/'experiment_manifest.json').read_text(encoding='utf-8'))
    assert meta['status']=='complete' and not meta['source']['synthetic']
    n=int(meta['probe_households']); repetitions=int(meta['repetitions'])
    aliases=[f'H{i+1:02d}' for i in range(n)]
    assert len(samples)==n*(2*repetitions+4)
    assert np.isfinite(samples[['projection_raw','projection_centered']].to_numpy()).all()
    for alias in aliases:
        for neighbor in ['included','suppressed']:
            subset=samples[(samples.household==alias)&(samples['mode']=='dp')&(samples.neighbor==neighbor)]
            assert len(subset)==repetitions and subset.repetition.nunique()==repetitions
    def values(mode):
        return controls[controls['mode']==mode].set_index('household').loc[aliases].projection_change.to_numpy()
    raw,clipped=values('non_dp'),values('clip_only')
    assert np.all(raw!=0)
    retention=100*clipped/raw
    distributions=[samples[(samples.household==alias)&(samples['mode']=='dp')&
        (samples.neighbor=='included')].projection_centered.to_numpy() for alias in aliases]
    chinese = chinese_font(a.font_file)
    available = {font.name for font in font_manager.fontManager.ttflist}
    latin = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({'font.family':[latin,chinese],
        'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'axes.unicode_minus':False,
        'pdf.fonttype':42,'svg.fonttype':'path','axes.linewidth':.8})
    blue,orange,red='#4c72b0','#dd8452','#c44e52'
    fig=plt.figure(figsize=(13,9))
    grid=fig.add_gridspec(2,2,height_ratios=[1.4,1])
    ax=fig.add_subplot(grid[0,:]); bx=fig.add_subplot(grid[1,0]); cx=fig.add_subplot(grid[1,1])
    x=np.arange(n)
    bars=ax.bar(x-.26,raw,.22,color=blue,label='无差分隐私：实测输出变化')
    ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=8)
    ax.bar(x,clipped,.22,color=orange,label='仅裁剪：实测输出变化')
    violin=ax.violinplot(distributions,positions=x+.26,widths=.24,showmeans=False,
        showmedians=True,showextrema=False,points=100)
    for body in violin['bodies']:
        body.set_facecolor(red); body.set_edgecolor(red); body.set_alpha(.65)
    violin['cmedians'].set_color('#7f1d1d')
    handles,labels=ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=red,alpha=.65))
    labels.append('差分隐私：输出分布（横线为中位数）')
    ax.legend(handles,labels,ncols=2,fontsize=8.5,loc='upper left')
    ax.set_xticks(x,aliases)
    ax.set_xlabel('用户编号（沿各目标用户的更新方向投影）')
    ax.set_ylabel('相对屏蔽目标用户基准的输出投影')
    low=min(0.,min(float(d.min()) for d in distributions))
    high=max(float(raw.max()),max(float(d.max()) for d in distributions))
    ax.set_ylim(low-.08*(high-low),high+.32*(high-low))
    ax.axhline(0,color='#555555',lw=.8)
    ax.set_title('（a）无差分隐私、仅裁剪与差分隐私发布结果',pad=10)
    bx.plot(x,retention,'o-',color='#55a868')
    for i,value in enumerate(retention):
        bx.annotate(f'{value:.2f}%',(i,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
    bx.set_xticks(x,aliases); bx.set_ylim(0,max(5,float(retention.max())*1.2))
    bx.set_xlabel('用户编号'); bx.set_ylabel('裁剪后的用户贡献保留比例 / %')
    bx.set_title('（b）裁剪后的用户贡献保留比例',pad=10)
    target=meta['display_target']
    part=samples[(samples.household==target)&(samples['mode']=='dp')]
    bins=np.histogram_bin_edges(part.projection_centered.to_numpy(),bins=35)
    for neighbor,label,color in [('suppressed','屏蔽目标用户',blue),('included','包含目标用户',red)]:
        data=part[part.neighbor==neighbor].projection_centered.to_numpy()
        cx.hist(data,bins=bins,density=True,color=color,alpha=.35,label=label)
        cx.hist(data,bins=bins,density=True,histtype='step',color=color,lw=1.4)
    cx.set_xlabel('相对屏蔽目标用户基准的输出投影')
    cx.set_ylabel('经验概率密度')
    cx.set_title(f'（c）用户{target}的差分隐私输出分布（每种情况{repetitions}次）',pad=10)
    cx.legend(fontsize=8.5)
    for axis in [ax,bx,cx]:
        axis.grid(axis='y' if axis is ax else 'both',alpha=.2)
        axis.set_axisbelow(True)
    fig.text(.5,.012,'固定用户组的单轮机制对照；采用实际高斯噪声与发布前输出范数限幅。',
             ha='center',fontsize=8.5)
    fig.tight_layout(rect=[0,.04,1,1],h_pad=1.5,w_pad=1.5)
    a.output.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(a.output/f'{FIGURE_NAME}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

    names = ("empirical_projections_INTERNAL.csv", "deterministic_controls.csv", "experiment_manifest.json")
    hashes = {name: hashlib.sha256((a.input / name).read_bytes()).hexdigest() for name in names}
    (a.output / "图3-5_数据来源.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(f"中文图 3-5：{a.output / (FIGURE_NAME + '.png')}", flush=True)


if __name__ == "__main__":
    main()
