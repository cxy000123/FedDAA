"""Train and test the five paper models with seeds 0, 1 and 2."""

import argparse
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from feddaa import CONFIG_DIR

COUNTS = {"daan": 9073, "timexer": 103969, "tcn": 98113, "tide": 2979816, "mlp": 3000817}
REGIONS = range(3)


def plans():
    from feddaa import federated as fed

    protocol = fed.read_json(CONFIG_DIR / "protocol.json")
    configs = {name: {**protocol, **fed.read_json(CONFIG_DIR / filename)}
               for name, filename in (("daan", "daan.json"), ("original", "baselines.json"))}
    seeds = (0, 1, 2)
    if tuple(protocol["seeds"]) != seeds or protocol["target_epsilon"] != 1:
        raise ValueError("The main experiment requires seeds 0, 1, 2 and epsilon=1")
    for config in configs.values():
        expected = [
            {"id": f"{spec['model']}/{label}/seed_{seed}", "model": spec["model"],
             "mode": mode, "epsilon": epsilon, "seed": seed}
            for spec in config["models"]
            for mode, label, epsilon in (("non_dp", "non_dp", None),
                                         ("clip_only", "clip_only", None),
                                         ("dp", "dp_eps_1", protocol["target_epsilon"]))
            for seed in seeds
        ]
        if config["runs"] != expected:
            raise ValueError("Plan does not match the paper's full model × condition × seed design")
    return configs


def preflight(args, configs):
    from feddaa import federated as fed

    fed.check_training_runtime(args.device)
    stores, scalers, _ = fed.context(args)
    roster = [s.n_customers for s in stores]
    config = configs["daan"]
    if roster != config["regional_households"]:
        raise ValueError(f"Expected the fixed paper roster {config['regional_households']}, found {roster}")
    dimensions = set()
    for store in stores:
        example = store.build_data(store.customer_ids[0], scalers, config["train_stride_hours"],
                                   config["future_weather_mode"], include_test=True)
        dimensions.add((example["past_dim"], example["future_dim"]))
    if len(dimensions) != 1:
        raise ValueError("Feature dimensions differ across regions")
    (past_dim, future_dim), = dimensions
    for plan in configs.values():
        for spec in plan["models"]:
            model = fed.make_model(spec, past_dim, future_dim, plan["history_steps"])
            count = sum(p.numel() for p in model.parameters())
            if count != COUNTS[spec["model"]]:
                raise ValueError(f"Unexpected parameter count: {spec['model']}={count}")
            print(f"{spec['model']}: {count:,} parameters")
    print(f"Eligible households by region: {roster}. Planned training runs: "
          f"{sum(len(plan['runs']) for plan in configs.values())}.")


def run_group(args, name, config):
    from feddaa import federated as fed
    from feddaa.reports import summary
    from feddaa.evaluate import evaluate

    plan = args.output / "training" / name / "plan.json"
    summary.frozen(plan, config)
    folder = plan.parent / "results"
    fed.run_matrix(SimpleNamespace(plan=plan, data_dir=args.data_dir, fixed_scalers=args.fixed_scalers,
                                   device=args.device, output_root=folder, run_id=None,
                                   resume=True, restart_unsaved=True))
    runs = []
    for run in config["runs"]:
        training = folder / run["id"]
        validation = fed.read_json(training / "validation.json")
        if validation["run"] != run:
            raise ValueError(f"Validation run differs: {training}")
        runs.append((training, validation["checkpoint_sha256"]))
    evaluate(SimpleNamespace(output=args.output / "tests" / name, data_dir=args.data_dir,
                             fixed_scalers=args.fixed_scalers, device=args.device, threads=args.threads,
                             split="test"), runs)


def main():
    import os

    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fixed-scalers", type=Path, default=Path("data/scalers.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/main"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true", help="Check regional inputs, windows and model sizes")
    parser.add_argument("--skip-weather", action="store_true", help="Finish after clean-test figures")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    for name in ("data_dir", "fixed_scalers", "output"):
        setattr(args, name, getattr(args, name).resolve())

    import torch

    torch.set_num_threads(args.threads)
    configs = plans()
    preflight(args, configs)
    if args.prepare_only:
        return

    from feddaa import federated as fed
    from feddaa.reports import summary

    summary.frozen(args.output / "inputs.json", {
        "fixed_scalers_sha256": fed.digest(args.fixed_scalers),
        "data_sha256": [fed.digest(args.data_dir / f"client_{i}_households.parquet") for i in REGIONS],
    })
    for name, plan in configs.items():
        run_group(args, name, plan)
    if summary.summarize(args.output, configs):
        raise RuntimeError("Main tests are incomplete")
    if not args.skip_weather:
        subprocess.run([sys.executable, "-u", "-m", "feddaa.weather",
                        "--main-results", str(args.output), "--output", str(args.output / "weather"),
                        "--data-dir", str(args.data_dir), "--fixed-scalers", str(args.fixed_scalers),
                        "--device", args.device, "--threads", str(args.threads)], check=True)
    subprocess.run([sys.executable, "-m", "feddaa.reports.collect", "--input", str(args.output)], check=True)


if __name__ == "__main__":
    main()
