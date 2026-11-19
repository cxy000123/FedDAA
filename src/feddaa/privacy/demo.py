"""Regenerate Figure 3-5 using a fixed DP checkpoint from the final DAAN study."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def checkpoint_source(main, seed):
    run_id = f"daan/dp_eps_1/seed_{seed}"
    plan = main / "training/daan/plan.json"
    folder = main / "training/daan/results" / run_id
    checkpoint = folder / "model.pt"
    done, val, privacy = (read(folder / name) for name in
                          ("complete.json", "validation.json", "privacy_report.json"))
    expected_run = {"id": run_id, "model": "daan", "mode": "dp", "epsilon": 1.0, "seed": seed}
    signature = digest(checkpoint)
    if (expected_run not in read(plan)["runs"] or done["run"] != expected_run
            or val["run"] != expected_run or val["evaluation_split"] != "val"
            or done["plan_sha256"] != digest(plan) or val["plan_sha256"] != done["plan_sha256"]
            or val["checkpoint_sha256"] != signature):
        raise ValueError("DAAN checkpoint, plan and validation records do not match")
    expected_privacy = {"mode": "dp", "training_dp": True, "target_epsilon": 1.0, "delta": 1e-5,
                        "total_parameters": 9073, "scope": "all",
                        "rounds": 50, "clip_norm": .05, "normalization": 4,
                        "regional_households": [150, 80, 70],
                        "max_protected_update_norm": 1.0}
    if (any(privacy.get(k) != v for k, v in expected_privacy.items())
            or not 0 < privacy["achieved_epsilon"] <= 1.000001
            or not privacy["noise_multiplier"] > 0
            or privacy["fixed_scalers_sha256"] != done["fixed_scalers_sha256"]):
        raise ValueError("DAAN privacy parameters differ from the Figure 3-5 protocol")
    return checkpoint, signature


def save_manuscript_values(output, seed, checkpoint_sha):
    import pandas as pd

    samples = pd.read_csv(output / "empirical_projections_INTERNAL.csv")
    controls = pd.read_csv(output / "contribution_comparison.csv")
    dp = samples[samples["mode"] == "dp"]
    key = ["household", "neighbor"]
    groups = dp.groupby(key)["projection_centered"]
    averages = groups.mean().unstack("neighbor")
    gaps = (averages["included"] - averages["suppressed"]).abs()

    def extent(series):
        return [float(series.min()), float(series.max())]

    h01 = dp[dp["household"] == "H01"].groupby("neighbor")["projection_centered"]
    result = {
        "training_seed": seed, "checkpoint_sha256": checkpoint_sha,
        "raw_projection_change_range": extent(controls["raw_projection_change"]),
        "clipped_projection_change_range": extent(controls["clipped_projection_change"]),
        "clipping_retained_percent_range": extent(controls["clipping_retained_percent"]),
        "absolute_dp_mean_gap_range": extent(gaps),
        "dp_within_group_std_range": extent(groups.std(ddof=1)),
        "h01_middle_95_percent": {label: [float(series.quantile(q)) for q in (.025, .975)]
                                   for label, series in h01},
    }
    (output / "fig35_manuscript_values.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-results", type=Path, default=Path("runs/main"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fixed-scalers", type=Path, default=Path("data/scalers.json"))
    parser.add_argument("--seed", type=int, choices=(0,), default=0, help="Figure 3-5 uses training seed 0")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--prepare-only", action="store_true", help="Verify the checkpoint without reading household data")
    stages.add_argument("--redraw", action="store_true", help="Regenerate plots from an already completed probe")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    main = args.main_results.resolve()
    output = (args.output or Path(f"runs/privacy/seed_{args.seed}")).resolve()
    checkpoint, signature = checkpoint_source(main, args.seed)
    print(f"Verified DAAN DP checkpoint seed={args.seed}: {signature}", flush=True)
    if args.prepare_only:
        return
    if output == main or main in output.parents or output in main.parents:
        parser.error("Use a dedicated Figure 3-5 output directory")
    manifest = output / "experiment_manifest.json"
    if not args.redraw:
        subprocess.run([sys.executable, "-u", "-m", "feddaa.privacy.measure",
            "--checkpoint", str(checkpoint), "--data-dir", str(args.data_dir.resolve()),
            "--fixed-scalers", str(args.fixed_scalers.resolve()), "--output", str(output),
            "--device", args.device, "--threads", str(args.threads),
            "--region", "2", "--households", "8", "--repetitions", "1000", "--display-target", "H01"],
            check=True)
    result = read(manifest)
    if (result["status"] != "complete" or result["source"]["checkpoint_sha256"] != signature
            or result["source"]["synthetic"] or result["region"] != 2 or result["probe_households"] != 8
            or result["repetitions"] != 1000 or result["display_target"] != "H01"):
        raise ValueError("Existing probe differs from the selected seed or protocol")
    save_manuscript_values(output, args.seed, signature)
    subprocess.run([sys.executable, "-m", "feddaa.privacy.figure",
                    "--input", str(output), "--output", str(output / "chinese_figure")], check=True)
    print(f"Figure 3-5 and replacement statistics: {output}", flush=True)


if __name__ == "__main__":
    main()
