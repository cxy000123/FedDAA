"""Weather and model-cost figures."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

FAMILIES = ['timexer', 'tcn', 'tide', 'mlp', 'daan']
ARCHIVE_NAMES = ['TimeXer', 'TCN', 'TiDE', 'MLP', 'DAAN']
MODEL_LABELS = ['FedTX', 'FedTCN', 'FedTD', 'FedMLP', 'FedDAA']
COLORS = ['#53789C', '#6F9690', '#B59667', '#8B81A0', '#BD5955']

def save(fig, name, output):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output / (name + '.' + ext), dpi=300, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)

def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#D9D9D9', linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=3)

def weather(inputs, output):
    data = pd.read_csv(inputs / 'weather_summary.csv')
    assert len(data) == 25 and (data.status == 'complete').all()
    assert (data.n_training_seeds == 3).all()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.35), layout='constrained')
    keys = ['rmse_mean', 'mae_mean', 'rmse_change_percent']
    labels = ['系统RMSE / kWh', '系统MAE / kWh', 'RMSE变化率 / %']
    titles = ['(a) 系统RMSE', '(b) 系统MAE', '(c) 相对无扰动的变化']
    for name, model_label, color in zip(ARCHIVE_NAMES, MODEL_LABELS, COLORS):
        frame = data[data.model == name].sort_values('level')
        for ax, key in zip(axes, keys):
            ax.plot(frame.level * 100, frame[key], 'o-', ms=4, color=color,
                    lw=1.8 if name == 'DAAN' else 1.2, label=model_label)
    for ax, label, title in zip(axes, labels, titles):
        tidy(ax)
        ax.set_xlabel('\u6270\u52a8\u6807\u51c6\u5dee / \u8bad\u7ec3\u96c6\u6c14\u8c61\u6807\u51c6\u5dee / %')
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20]))
    axes[0].legend(loc='center right', fontsize=8)
    save(fig, '图3-6_气象输入敏感性分析', output)

def cost(inputs, output):
    data = pd.read_csv(inputs / 'test_summary.csv')
    dp = data[data['mode'] == 'dp'].set_index('family').loc[FAMILIES]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.75), layout='constrained')
    x = np.arange(5)
    axes[0].bar(x, dp.parameters, color=COLORS, edgecolor='#444444', linewidth=.4, zorder=3)
    axes[0].set_yscale('log')
    axes[0].set_ylim(1000, 3e7)
    axes[0].set_xticks(x, MODEL_LABELS)
    axes[0].set_ylabel('\u53c2\u6570\u91cf / \u4e2a\uff08\u5bf9\u6570\u523b\u5ea6\uff09')
    axes[0].set_title('(a) \u6a21\u578b\u53c2\u6570\u91cf', fontsize=11)
    for xx, value in zip(x, dp.parameters):
        axes[0].text(xx, value * 1.15, f'{int(value):,}', ha='center', va='bottom', fontsize=8)
    for i, (mode, label, color) in enumerate(zip(
        ['non_dp', 'clip_only', 'dp'],
        ['\u65e0DP', '\u4ec5\u88c1\u526a', 'DP\uff08\u03b5=1\uff09'],
        ['#A9BCCB', '#6B938D', '#BD6B60'])):
        frame = data[data['mode'] == mode].set_index('family').loc[FAMILIES]
        axes[1].bar(x + (i - 1) * .24, frame.training_minutes_mean, .23,
                    color=color, label=label, edgecolor='#444444', linewidth=.4, zorder=3)
    axes[1].set_xticks(x, MODEL_LABELS)
    axes[1].set_ylabel('\u8bad\u7ec3\u65f6\u95f4 / min')
    axes[1].set_ylim(0, data.training_minutes_mean.max() * 1.25)
    axes[1].set_title('(b) \u4e09\u79cd\u8bad\u7ec3\u6761\u4ef6\u7684\u8017\u65f6', fontsize=11)
    axes[1].legend(ncols=3, loc='upper center', fontsize=8)
    for ax in axes:
        tidy(ax)
    save(fig, '图3-7_模型参数量与训练时长', output)
