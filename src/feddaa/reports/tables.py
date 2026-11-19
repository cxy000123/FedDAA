"""Rebuild section 3.3 tables from the frozen, unmodified experiment metrics."""

import argparse
import csv
from pathlib import Path


MODELS = {"TimeXer": "FedTX", "TCN": "FedTCN", "TiDE": "FedTD", "MLP": "FedMLP", "DAAN": "FedDAA"}
MODES = ("non_dp", "clip_only", "dp")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def table(title, headers, rows):
    def line(values):
        return "| " + " | ".join(map(str, values)) + " |"
    return "\n".join([f"## {title}", "", line(headers), line(["---"] * len(headers)),
                      *(line(row) for row in rows)])


def fixed(row, *fields):
    return [f"{float(row[key]):.6f}" for key in fields]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/main"))
    parser.add_argument("--output", type=Path, default=Path("reports/section33_tables.md"))
    args = parser.parse_args()
    rows = read_rows(args.input / "test_summary.csv")
    assert len(rows) == 15 and all(x["status"] == "complete" and x["n_seeds"] == "3" for x in rows)
    lookup = {(row["model"], row["mode"]): row for row in rows}
    weather = {(row["model"], float(row["level"])): row
               for row in read_rows(args.input / "weather/weather_summary.csv")}
    assert len(weather) == 25 and all(x["status"] == "complete" for x in weather.values())
    output = ["# 3.3 节表格复算", "各指标为种子 0、1、2 的平均值，数值按原始实验记录归入相应模型。"]
    output.append(table("表 3-1 无 DP", ["模型", "RMSE/kWh", "MAE/kWh", "R²", "日间RMSE/kWh"],
        [(MODELS[m], *fixed(lookup[m, "non_dp"], "rmse_mean", "mae_mean", "r2_mean", "daylight_rmse_mean"))
         for m in MODELS]))
    output.append(table("表 3-2 仅裁剪与 DP（ε=1）",
        ["模型", "仅裁剪 RMSE/kWh", "DP RMSE/kWh", "DP MAE/kWh", "DP R²"],
        [(MODELS[m], *fixed(lookup[m, "clip_only"], "rmse_mean"),
          *fixed(lookup[m, "dp"], "rmse_mean", "mae_mean", "r2_mean")) for m in MODELS]))
    output.append(table("表 3-3 区域、用户与日间误差",
        ["模型", "最差区域 NRMSE/%", "用户平均 RMSE/(kWh/kW)", "用户 P90 RMSE/(kWh/kW)", "日间 RMSE/kWh"],
        [(MODELS[m], *fixed(lookup[m, "dp"], "worst_region_nrmse_percent_mean", "household_rmse_mean",
                    "household_p90_mean", "daylight_rmse_mean")) for m in MODELS]))
    output.append(table("表 3-6 气象输入扰动",
        ["模型", "0% RMSE/kWh", "20% RMSE/kWh", "0% MAE/kWh", "20% MAE/kWh"],
        [(MODELS[m], *(fixed(weather[m, level], metric)[0]
               for metric in ("rmse_mean", "mae_mean") for level in (0, 0.2))) for m in MODELS]))
    costs, increases = [], []
    for model in MODELS:
        count = int(float(lookup[model, "dp"]["parameters_mean"]))
        time = [f'{float(lookup[model, mode]["training_minutes_mean"]):.2f}±'
                f'{float(lookup[model, mode]["training_minutes_std"]):.2f}' for mode in MODES]
        costs.append((MODELS[model], f"{count:,}", f"{count * 4:,}", *time))
        for mode, label in (("clip_only", "仅裁剪"), ("dp", "DP")):
            baseline = float(lookup[model, "non_dp"]["rmse_mean"])
            ratio = 100 * (float(lookup[model, mode]["rmse_mean"]) / baseline - 1)
            increases.append((MODELS[model], label, f"{ratio:.2f}"))
    output.append(table("表 3-7 模型开销", ["模型", "参数量", "FP32载荷/字节",
        "无 DP/min", "仅裁剪/min", "DP/min"], costs))
    output.append(table("图 3-2 系统 RMSE 增长率", ["模型", "条件", "RMSE 均值的相对增幅/%"], increases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(output) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
