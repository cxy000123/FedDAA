# FedDAA 光伏联邦预测

本项目实现日内对齐注意力网络 DAAN，以及可信区域聚合下的光伏发电量预测实验。主实验包括 FedDAA、FedTX、FedTCN、FedTD 和 FedMLP，各条件使用随机种子 0、1、2 各运行一次，比较无 DP、仅裁剪及 DP（ε=1，δ=10⁻⁵）三种设置。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `src/feddaa/models/` | DAAN 与四种对比模型 |
| `src/feddaa/data/` | 分区、归一化参数和预测窗口 |
| `src/feddaa/privacy/` | 差分隐私机制、预算核算和单户贡献实测 |
| `src/feddaa/reports/` | 指标汇总、论文表格和绘图 |
| `src/feddaa/config/` | 共享实验设置、各模型运行清单和正式输入校验值 |
| `src/feddaa/train.py`、`federated.py` | 实验入口和联邦训练 |
| `src/feddaa/evaluate.py`、`weather.py` | 测试集评估和气象敏感性分析 |

## 环境

论文使用 PyTorch 和 NVIDIA GeForce RTX 5090。代码需要 Python 3.10 或更新版本。GPU 运行须安装与 CUDA 兼容的 PyTorch，可用 `--device cpu` 做输入核对。以下命令均在仓库根目录执行。

```bash
python -m pip install -e .
python -m feddaa.verify
```

## 1. 准备数据

将 `ausgrid_weather_silo_2010_2013.parquet` 放在代码目录的上一级。这是已按邮编与日期或时间匹配的合并数据：半小时发电量来自 Ausgrid 电表的 GG 通道，气温、露点、风速、降水来自气象站记录，日太阳辐照量来自 SILO。SILO 日值匹配到当天的半小时时段。

源文件 SHA-256：`133521546a03799acf7821b9db95909f9557df8c1050bd7a383da96578c30c68`。

```text
项目目录/
├── ausgrid_weather_silo_2010_2013.parquet
└── feddaa/
    ├── README.md
    ├── pyproject.toml
    └── src/
```

```bash
python -m feddaa.data.prepare --source ../ausgrid_weather_silo_2010_2013.parquet --output-dir data
python -m feddaa.data.scalers --data-dir data --output data/scalers.json
```

预处理按地理位置将 300 户划入 150、80、70 户三个互不重叠的区域。训练使用随机种子 0、1、2。数据按预测目标所在日期划分为训练集、验证集和测试集，历史窗口使用预测时点之前的观测。

| 划分 | 日期（包含首尾） |
| --- | --- |
| 训练 | 2010-07-01 至 2012-03-31 |
| 验证 | 2012-04-01 至 2012-06-30 |
| 测试 | 2012-07-01 至 2013-06-30 |

每户发电量先按固定装机容量归一化，再用训练期的固定均值和标准差做标准化。历史或预测目标缺少 GG 值时只丢弃该窗口；容量与基础气象变量的缺测值采用线性插值。另存位置可通过 `--data-dir` 和 `--fixed-scalers` 指定。

## 2. 训练和测试

五种模型统一训练 50 轮，使用过去 7 天预测未来 24 小时（48 个半小时时段）。模型均预测相对于过去 7 天同时刻均值的残差，预测时段输入使用实测气象。推理时把发电量限制为非负值，并将太阳位置确定的夜间预测置零。

区域内采用独立泊松抽样，期望每轮抽取 4 户。无 DP 不裁剪单户更新、不加噪、不对区域发布投影；仅裁剪对完整单户更新做二范数裁剪并对区域发布投影；DP 在聚合后额外加入高斯噪声。区域更新按固定期望户数归一化，上层按各区域固定户数加权。差分隐私预算按各区域的泊松抽样高斯机制累计 50 轮 RDP 后取最大的区域上界。各运行使用相同采样协议；抽样与 DP 噪声使用独立随机性，不由公开训练种子决定。

```bash
python -m feddaa.train --device cuda --prepare-only
python -u -m feddaa.train --device cuda
```

完整运行包括 45 次联邦训练和对应的测试，随后固定各模型的 DP 检查点，按训练期气象标准差的 0%、5%、10%、15%、20% 扰动未来气象；同一扰动水平各模型共用扰动，历史输入和目标保持原样。结果写入 `runs/main/`，主实验完成后自动生成图件。使用相同输入与输出目录再次运行会跳过已通过校验的任务。气象阶段的运行记录含源码校验值。如需沿用整理前的气象结果，已生成的汇总表可直接用于绘图，也可运行 `python -m feddaa.weather --report-only` 从已有测试记录重算汇总。

## 3. 生成图表

```bash
python -m feddaa.reports.tables
python -m feddaa.reports.figures
python -u -m feddaa.privacy.demo --device cuda --seed 0
```

- 测试指标先在每次运行内计算，再按种子 0、1、2 汇总均值与样本标准差。区域对有效用户取平均，系统按固定区域户数加权。区域 NRMSE 按实测极差归一化，用户 RMSE 按装机容量归一化。
- 表格写入 `reports/section33_tables.md`，图件写入 `runs/main/figures/`。训练用时从模型初始化开始，到第 50 轮完成为止，不含验证和测试；FP32 完整参数载荷按参数量乘 4 字节计算。
- 图 3-4 使用测试集实际日发电量，自动选取最接近第 10、50 和 90 百分位数的三个日期，并绘制种子 0、1、2 的平均预测值和平均绝对误差。
- 图 3-5 固定种子 0 的 DAAN DP 检查点，选区域 3 的 8 户；固定其他更新，分别在加入和屏蔽目标更新的条件下各抽样 1000 次。图件写入 `runs/privacy/seed_0/chinese_figure/`。

## 数据版本

随代码提供的校验清单记录正式实验使用的分区 Parquet 和归一化参数的 SHA-256。
