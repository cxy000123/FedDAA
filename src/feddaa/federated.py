"""Local optimization, trusted aggregation and validation."""

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import time
import uuid

import torch

from feddaa.models.baselines import make_baseline
from feddaa.models.daan import make_daan
from feddaa.data.windows import training_loader


def make_model(spec, past_dim, future_dim, history_steps):
    # Also accept the original identifier when inspecting an old checkpoint.
    if spec["model"] in ("daan", "lead_fusion"):
        return make_daan(spec, past_dim, future_dim, history_steps)
    return make_baseline(spec["model"], past_dim, future_dim, history_steps,
                         spec["hidden_dim"], spec["dropout"])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def get_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def assert_finite_state(state: dict[str, torch.Tensor], label: str) -> None:
    invalid = [key for key, value in state.items() if not torch.isfinite(value).all()]
    if invalid:
        raise FloatingPointError(
            f"{label} contains NaN/Inf in {len(invalid)} tensors; first={invalid[0]}"
        )


def train_local(
    base_state: dict[str, torch.Tensor],
    data: dict[str, object],
    trainable_keys: tuple[str, ...],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    gradient_clip: float,
    model_factory,
) -> dict[str, torch.Tensor]:
    model = model_factory().to(device)
    model.load_state_dict({key: value.to(device) for key, value in base_state.items()})
    trainable = []
    selected = set(trainable_keys)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("The selected DP parameter scope has no trainable parameters.")
    loader = training_loader(data["splits"]["train"], batch_size, device, shuffle=True)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for past, future, target, baseline in loader:
            past, future = past.to(device), future.to(device)
            target, baseline = target.to(device), baseline.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(torch.square(baseline + model(past, future) - target))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite local loss during DP training.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
            optimizer.step()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    assert_finite_state({key: state[key] for key in trainable_keys}, "local state")
    return state


def parameter_keys(model, scope):
    import torch
    if scope == "all":
        return tuple(name for name, _ in model.named_parameters())
    if scope == "output_layer":
        # TiDE outputs all 48 steps jointly; per-lead models output one step.
        layers = [(name, layer) for name, layer in model.named_modules()
                  if isinstance(layer, torch.nn.Linear)]
        prefix = layers[-1][0] + "."
        return tuple(name for name, _ in model.named_parameters() if name.startswith(prefix))
    raise ValueError(f"Unsupported trainable scope: {scope}")


def regional_update(deltas, template, mode, clip_norm, sigma, normalization, noise_seed, max_norm):
    import torch
    from feddaa.privacy.dp import private_household_sum, clip_household_delta, clip_released_delta
    if mode == "dp":
        value, _ = private_household_sum(deltas, template, clip_norm, sigma, normalization, noise_seed)
    else:
        value = {k: torch.zeros_like(v) for k, v in template.items()}
        for delta in deltas:
            contribution = clip_household_delta(delta, clip_norm)[0] if mode == "clip_only" else delta
            for key in value:
                value[key].add_(contribution[key] / normalization)
    # Identical final bounding in the clipped controls separates clipping from noise.
    return clip_released_delta(value, max_norm)[0] if mode != "non_dp" else value


def evaluate(model, stores, scalers, config, device, split_name):
    import numpy as np
    import pandas as pd
    from feddaa.data.windows import VALID_START, TEST_START, TEST_END_EXCLUSIVE
    from feddaa.data.windows import predict_model_split, metric_values
    if split_name not in ("val", "test"):
        raise ValueError(split_name)
    start, end = (VALID_START, TEST_START) if split_name == "val" else (TEST_START, TEST_END_EXCLUSIVE)
    reference = pd.date_range(start, end, freq="D", inclusive="left")
    lookup = {pd.Timestamp(t): i for i, t in enumerate(reference)}
    system_actual = np.zeros((len(reference), 48))
    system_predicted = np.zeros_like(system_actual)
    region_rows, household_rows = [], []
    public_daylight = np.full_like(system_actual, np.nan)
    for store in stores:
        actual_sum, pred_sum = np.zeros_like(system_actual), np.zeros_like(system_actual)
        counts = np.zeros(len(reference), dtype=int)
        for i, customer_id in enumerate(store.customer_ids):
            data = store.build_data(customer_id, scalers, config["train_stride_hours"],
                                    config["future_weather_mode"], include_test=split_name == "test")
            split = data["splits"][split_name]
            if not len(split["origin"]):
                continue
            predicted = predict_model_split(model, data, split_name, config["batch_size"], device)
            if config["zero_night_output"]:
                predicted = np.where(split["daylight"] > .5, predicted, 0.)
            actual = np.asarray(split["target_raw"], dtype=np.float64)
            if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
                raise FloatingPointError("Non-finite evaluation values")
            positions = np.asarray([lookup[pd.Timestamp(t)] for t in split["origin"]])
            if len(np.unique(positions)) != len(positions):
                raise ValueError("Duplicate evaluation origins")
            actual_sum[positions] += actual
            pred_sum[positions] += predicted
            counts[positions] += 1
            public_daylight[positions] = split["daylight"]
            # Capacity-normalized household errors prevent large installations dominating.
            capacity = np.maximum(split["capacity"], 1e-6)
            household_rows.append(metric_values(actual / capacity, predicted / capacity, split["daylight"]))
            if (i + 1) % 25 == 0 or i + 1 == store.n_customers:
                print(f"{split_name}: region {store.client_id}, households {i+1}/{store.n_customers}", flush=True)
        if (counts == 0).any():
            raise ValueError("Uncovered regional evaluation origins")
        regional_actual = actual_sum / counts[:, None]
        regional_predicted = pred_sum / counts[:, None]
        system_actual += regional_actual * store.n_customers
        system_predicted += regional_predicted * store.n_customers
        region_rows.append({"client_id": store.client_id, "n_customers": store.n_customers,
                            **metric_values(regional_actual, regional_predicted, public_daylight)})
    if not np.isfinite(public_daylight).all():
        raise ValueError("Uncovered daylight flags")
    total = sum(s.n_customers for s in stores)
    system = metric_values(system_actual / total, system_predicted / total, public_daylight)
    worst = max(row["nrmse_percent"] for row in region_rows)
    return {"evaluation_split": split_name, "test_split_accessed": split_name == "test",
            "system": system, "regions": region_rows,
            "household_macro_rmse_per_kw": float(np.mean([r["rmse"] for r in household_rows])),
            "household_p90_rmse_per_kw": float(np.quantile([r["rmse"] for r in household_rows], .9)),
            "worst_region_nrmse_percent": worst}


def context(args):
    from feddaa.data.windows import LocalScalers, RegionHouseholdStore
    stores = [RegionHouseholdStore(args.data_dir / f"client_{i}_households.parquet", i) for i in range(3)]
    # The same user must not be protected as independent disjoint records in two regions.
    rosters = [set(store.customer_ids) for store in stores]
    if any(rosters[i] & rosters[j] for i in range(3) for j in range(i)):
        raise ValueError("Households overlap between regions; max-region accounting is invalid.")
    return stores, LocalScalers.from_dict(read_json(args.fixed_scalers)), get_device(args.device)


def run_one(run, spec, config, stores, scalers, device, output):
    import torch
    from feddaa.privacy.dp import poisson_sample_households, calibrate_noise_multiplier, epsilon_from_rdp
    wall_started = time.perf_counter()
    probe = stores[0].build_data(stores[0].customer_ids[0], scalers, config["train_stride_hours"],
                                 config["future_weather_mode"], include_test=False)
    past_dim, future_dim = int(probe["past_dim"]), int(probe["future_dim"])
    del probe
    factory = lambda: make_model(spec, past_dim, future_dim, config["history_steps"])
    torch.manual_seed(run["seed"])
    training_started = time.perf_counter()
    model = factory()
    keys = parameter_keys(model, config["scope"])
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    count = sum(state[k].numel() for k in keys)
    cohort = config["expected_households"]
    if any(cohort > s.n_customers for s in stores):
        raise ValueError("Expected cohort exceeds a regional roster")
    rates = [cohort / s.n_customers for s in stores]
    sigma = calibrate_noise_multiplier(run["epsilon"], max(rates), config["rounds"], config["delta"]) if run["mode"] == "dp" else 0.
    epsilons = [epsilon_from_rdp(q, sigma, config["rounds"], config["delta"])[0] for q in rates] if sigma else []
    secure = secrets.SystemRandom()
    total_households = sum(s.n_customers for s in stores)
    for round_id in range(config["rounds"]):
        global_delta = {k: torch.zeros_like(state[k]) for k in keys}
        for store, rate in zip(stores, rates):
            selected = poisson_sample_households(store.customer_ids, rate, secure.getrandbits(63))
            deltas = []
            for customer_id in selected:
                data = store.build_data(customer_id, scalers, config["train_stride_hours"],
                                        config["future_weather_mode"], include_test=False)
                if not len(data["splits"]["train"]["origin"]):
                    # A sampled household without valid windows contributes zero.
                    del data
                    continue
                local = train_local(state, data, keys, config["local_epochs"],
                    config["batch_size"], config["lr"], config["weight_decay"], device,
                    config["gradient_clip"], model_factory=factory)
                deltas.append({k: local[k] - state[k] for k in keys})
                del data, local
            update = regional_update(deltas, {k: state[k] for k in keys}, run["mode"],
                config["clip_norm"], sigma, cohort, secure.getrandbits(63), config["max_protected_update_norm"])
            assert_finite_state(update, "regional update")
            for k in keys:
                global_delta[k].add_(update[k], alpha=store.n_customers / total_households)
        for k in keys:
            state[k].add_(global_delta[k])
        assert_finite_state(state, "global state")
        print(f"{run['id']}: round {round_id+1}/{config['rounds']}", flush=True)
    training_seconds = time.perf_counter() - training_started
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state": state, "model_spec": spec, "past_dim": past_dim, "future_dim": future_dim,
                  "config": config, "run": run}
    torch.save(checkpoint, output / "model.pt")
    model.load_state_dict(state)
    result = evaluate(model.to(device), stores, scalers, config, device, "val")
    privacy = {"mode": run["mode"], "training_dp": bool(sigma), "target_epsilon": run["epsilon"],
               "adjacency": "add/remove one household contribution; fixed public roster",
               "accountant": "Poisson-subsampled Gaussian RDP; integer orders 2..256",
               "achieved_epsilon": max(epsilons) if epsilons else None,
               "delta": config["delta"] if sigma else None, "noise_multiplier": sigma,
               "regional_epsilons": epsilons, "regional_sample_rates": rates,
               "regional_households": [s.n_customers for s in stores],
               "rounds": config["rounds"], "scope": config["scope"],
               "clip_norm": config["clip_norm"] if run["mode"] != "non_dp" else None,
               "normalization": cohort,
               "max_protected_update_norm": config["max_protected_update_norm"] if run["mode"] != "non_dp" else None,
               "fixed_scalers_sha256": config["fixed_scalers_sha256"],
               "trainable_parameters": count, "total_parameters": sum(p.numel() for p in model.parameters()),
               "noise_l2_rms_before_postprocessing": sigma * config["clip_norm"] * math.sqrt(count) / cohort,
               "claim": config["privacy_scope"], "limitations": config["privacy_limitations"]}
    write_json(output / "privacy_report.json", privacy)
    write_json(output / "validation.json", {**result, "run": run,
               "training_seconds": training_seconds, "wall_seconds": time.perf_counter() - wall_started,
               "checkpoint_sha256": digest(output / "model.pt"), "plan_sha256": config["plan_sha256"]})
    write_json(output / "complete.json", {"run": run, "plan_sha256": config["plan_sha256"],
               "fixed_scalers_sha256": config["fixed_scalers_sha256"],
               "data_sha256": config["data_sha256"]})


def prepare_run_output(output, expected, resume, restart_unsaved=False):
    """Skip verified completions or retry a failed attempt with no saved model.

    Never replace checkpoints, evaluation results, or mismatched experiment data.
    The prior start record is retained in the same directory; no directory deletion.
    """
    if output.exists() and any(output.iterdir()):
        complete_path = output / "complete.json"
        if resume and complete_path.is_file() and read_json(complete_path) == expected:
            if digest(output / "model.pt") != read_json(output / "validation.json")["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed: {output}")
            print(f"Skip verified completed run: {expected['run']['id']}", flush=True)
            return False
        started = output / "started.json"
        files = list(output.iterdir())
        only_start_records = all(
            item.is_file() and not item.is_symlink() and (
                item.name == "started.json" or
                (item.name.startswith("started.interrupted.") and item.suffix == ".json")
            ) for item in files
        )
        if (resume and restart_unsaved and not complete_path.exists() and
                only_start_records and started.is_file() and read_json(started) == expected):
            backup = output / f"started.interrupted.{uuid.uuid4().hex}.json"
            started.rename(backup)
            print(f"Restart unsaved run from round 1; previous start record retained: {backup}", flush=True)
        else:
            raise FileExistsError(
                f"Existing/incomplete/mismatched run: {output}. "
                "--resume --restart-unsaved only restarts matching runs containing start records alone; "
                "saved models and evaluation artifacts are never overwritten."
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "started.json", expected)
    return True


def check_training_runtime(device_name):
    """Probe the outer-product BMM path before any experiment output is created."""
    import torch
    device = get_device(device_name)
    disabled = os.environ.get("TORCH_DISABLE_NATIVE_JIT", "0")
    print(f"PyTorch {torch.__version__}; device={device}; TORCH_DISABLE_NATIVE_JIT={disabled}", flush=True)
    # Use deterministic tensors: do not perturb training random initialization.
    left = torch.ones((2, 16, 1), device=device, requires_grad=True)
    right = torch.ones((2, 1, 16), device=device, requires_grad=True)
    try:
        torch.bmm(left, right).square().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except RuntimeError as error:
        if "Failed to find C compiler" in str(error):
            raise RuntimeError(
                "GPU 算子预检失败：Triton 找不到 C 编译器。请使用更新后的直接运行脚本，"
                "确保 DISABLE_NATIVE_JIT=True；若当前 PyTorch 版本不支持此开关，"
                "需在服务器安装 C 编译器并设置 CC 指向其可执行文件。"
            ) from error
        raise
    print("Tensor forward/backward runtime check passed.", flush=True)
    return {"torch_version": str(torch.__version__), "device": str(device),
            "torch_disable_native_jit": disabled}


def run_matrix(args):
    runtime = check_training_runtime(args.device)
    config = read_json(args.plan)
    config["runtime"] = runtime
    config["plan_sha256"] = digest(args.plan)
    config["fixed_scalers_sha256"] = digest(args.fixed_scalers)
    provenance_path = args.fixed_scalers.with_suffix(args.fixed_scalers.suffix + ".provenance.json")
    config["scaler_provenance"] = read_json(provenance_path) if provenance_path.exists() else {"source": "unverified"}
    provenance = config["scaler_provenance"]
    if provenance.get("sha256", config["fixed_scalers_sha256"]) != config["fixed_scalers_sha256"]:
        raise ValueError("Scaler provenance digest mismatch")
    if provenance.get("fit_period_end_exclusive", "") > "2012-04-01 00:00:00":
        raise ValueError("Scalers include validation dates; use prepare-scalers for training-only preprocessing")
    config["data_sha256"] = [digest(args.data_dir / f"client_{i}_households.parquet") for i in range(3)]
    stores, scalers, device = context(args)
    selected_runs = config["runs"]
    if args.run_id:
        selected_runs = [r for r in selected_runs if r["id"] == args.run_id]
        if not selected_runs:
            raise ValueError(f"Unknown run id: {args.run_id}")
    for run in selected_runs:
        output = args.output_root / run["id"]
        expected = {"run": run, "plan_sha256": config["plan_sha256"],
                    "fixed_scalers_sha256": config["fixed_scalers_sha256"], "data_sha256": config["data_sha256"]}
        if not prepare_run_output(output, expected, args.resume, getattr(args, "restart_unsaved", False)):
            continue
        spec = next(m for m in config["models"] if m["model"] == run["model"])
        run_one(run, spec, config, stores, scalers, device, output)
