"""Evaluate paper checkpoints on the held-out forecasting split."""

import json
import platform
from pathlib import Path
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(actual, predicted, daylight=None):
    import numpy as np
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Invalid evaluation arrays")
    error = p-a
    span = np.ptp(a)
    total = np.sum((a-a.mean())**2)
    result = dict(n=int(a.size), rmse=float(np.sqrt(np.mean(error**2))),
                  mae=float(np.mean(abs(error))),
                  nrmse_percent=float(100*np.sqrt(np.mean(error**2))/span) if span else None,
                  r2=float(1-np.sum(error**2)/total) if total else None)
    if daylight is not None:
        mask = np.asarray(daylight) > .5
        day = metrics(a[mask], p[mask])
        result.update(daylight_rmse=day["rmse"], daylight_mae=day["mae"], daylight_r2=day["r2"])
    return result


def aggregate(actual_sums, prediction_sums, counts, sizes):
    """Match the project's public-region-size weighted mean evaluation."""
    import numpy as np
    if np.any(counts <= 0):
        raise ValueError("Uncovered regional dates")
    regional_actual = actual_sums/counts[:, :, None]
    regional_predicted = prediction_sums/counts[None, :, :, None]
    weights = np.asarray(sizes, dtype=float)/sum(sizes)
    actual = np.sum(regional_actual*weights[:, None, None], axis=0)
    predicted = np.sum(regional_predicted*weights[None, :, None, None], axis=1)
    return actual, predicted, regional_actual, regional_predicted


def predict(model, data, split_name, variants, batch_size, device):
    import numpy as np
    import torch
    from feddaa.data.windows import restore_gg
    split = data["splits"][split_name]
    result = {v: [] for v in variants}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(split["past"]), batch_size):
            stop = start+batch_size
            past = torch.from_numpy(split["past"][start:stop]).to(device)
            future = torch.from_numpy(split["future"][start:stop]).to(device)
            prediction = model(past, future)
            base = split["baseline_scaled"][start:stop]
            for variant in variants:
                residual = 0 if variant == "baseline" else prediction.cpu().numpy()
                result[variant].append(base+residual)
    return {v: restore_gg(np.concatenate(parts), split["capacity"], data["scalers"])
            for v, parts in result.items()}


def evaluate(a, runs, predict_run=predict, *, compare_prior=True):
    import numpy as np
    import pandas as pd
    import torch
    from feddaa import federated as fed
    from feddaa.data.windows import VALID_START, TEST_START, TEST_END_EXCLUSIVE
    torch.set_num_threads(a.threads)
    fed.check_training_runtime(a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    scaler_hash = fed.digest(a.fixed_scalers)
    hashes = [fed.digest(a.data_dir/f"client_{i}_households.parquet") for i in range(3)]
    context_args = SimpleNamespace(data_dir=a.data_dir, fixed_scalers=a.fixed_scalers, device=a.device)
    stores, scalers, device = fed.context(context_args)
    for folder, checkpoint_hash in runs:
        checkpoint_path = folder/"model.pt"
        if fed.digest(checkpoint_path) != checkpoint_hash:
            raise ValueError(f"Checkpoint changed: {folder}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config, run = checkpoint["config"], checkpoint["run"]
        if config["fixed_scalers_sha256"] != scaler_hash or config["data_sha256"] != hashes:
            raise ValueError(f"Data or scaler mismatch: {folder}")
        if config["plan_sha256"] != read(folder/"complete.json")["plan_sha256"]:
            raise ValueError("Checkpoint and completion plan hashes differ")
        if read(folder/"validation.json").get("evaluation_split") != "val":
            raise ValueError("Expected the validation record of the final 50-round checkpoint")
        if config.get("scope") != "all":
            raise ValueError("Only all-parameter checkpoints supported")
        spec = checkpoint["model_spec"]
        variants = ["learned", "baseline"]
        target = a.output/a.split/run["id"]
        target.mkdir(parents=True, exist_ok=True)
        signature = {"checkpoint_sha256": checkpoint_hash, "fixed_scalers_sha256": scaler_hash,
                     "data_sha256": hashes, "split": a.split, "variants": variants,
                     "format_version": 2}
        done = target/"complete.json"
        if done.exists():
            if read(done) != signature:
                raise ValueError(f"Analysis configuration changed: use a new output ({target})")
            if not all((target/x).exists() for x in ("evaluation.json", "predictions.npz", "households.csv")):
                raise ValueError(f"Completed outputs missing: {target}")
            print(f"Skip completed analysis: {run['id']}", flush=True)
            continue
        model = fed.make_model(spec, checkpoint["past_dim"], checkpoint["future_dim"],
                                  config["history_steps"])
        model.load_state_dict(checkpoint["state"])
        model.to(device)
        start, end = (VALID_START, TEST_START) if a.split == "val" else (TEST_START, TEST_END_EXCLUSIVE)
        origins = pd.date_range(start, end, freq="D", inclusive="left")
        lookup = {pd.Timestamp(t): i for i, t in enumerate(origins)}
        days = len(origins)
        actual_sums = np.zeros((3, days, 48))
        pred_sums = np.zeros((len(variants), 3, days, 48))
        counts = np.zeros((3, days), dtype=int)
        daylight = np.full((days, 48), np.nan)
        household_rows = []
        for k, store in enumerate(stores):
            for j, customer in enumerate(store.customer_ids):
                data = store.build_data(customer, scalers, config["train_stride_hours"],
                                        config["future_weather_mode"], include_test=a.split == "test")
                split = data["splits"][a.split]
                if not len(split["origin"]):
                    del data
                    continue
                predictions = predict_run(model, data, a.split, variants, config["batch_size"], device)
                actual = np.asarray(split["target_raw"], dtype=float)
                positions = np.asarray([lookup[pd.Timestamp(t)] for t in split["origin"]])
                if len(np.unique(positions)) != len(positions):
                    raise ValueError("Duplicate evaluation origins")
                if not np.isfinite(actual).all():
                    raise ValueError("Nonfinite observations")
                actual_sums[k, positions] += actual
                counts[k, positions] += 1
                daylight[positions] = split["daylight"]
                capacity = np.maximum(split["capacity"], 1e-6)
                for vi, variant in enumerate(variants):
                    predicted = predictions[variant]
                    if config["zero_night_output"]:
                        predicted = np.where(split["daylight"] > .5, predicted, 0)
                    pred_sums[vi, k, positions] += predicted
                    row = {"variant": variant, "region": k, "customer": str(customer),
                           "capacity_kw": float(np.mean(capacity)),
                           **metrics(actual/capacity, predicted/capacity, split["daylight"])}
                    household_rows.append(row)
                if (j+1) % 25 == 0 or j+1 == len(store.customer_ids):
                    print(f"{run['id']} {a.split}: region={k} users={j+1}/{store.n_customers}", flush=True)
                del data
        if not np.isfinite(daylight).all():
            raise ValueError("Missing daylight observations")
        y, predictions, regional_y, regional_predictions = aggregate(
            actual_sums, pred_sums, counts, [s.n_customers for s in stores])
        evaluations = {}
        for vi, variant in enumerate(variants):
            regions = [{"client_id": k, "n_customers": store.n_customers,
                        **metrics(regional_y[k], regional_predictions[vi, k], daylight)}
                       for k, store in enumerate(stores)]
            hh = [r["rmse"] for r in household_rows if r["variant"] == variant]
            worst = max(r["nrmse_percent"] for r in regions)
            system = metrics(y, predictions[vi], daylight)
            evaluations[variant] = {"system": system, "regions": regions,
                "household_macro_rmse_per_kw": float(np.mean(hh)),
                "household_p90_rmse_per_kw": float(np.quantile(hh, .9)),
                "worst_region_nrmse_percent": worst}
        baseline = evaluations["learned"]
        result = {"run": run, "evaluation_split": a.split,
                  "test_split_accessed": a.split == "test",
                  "checkpoint_sha256": checkpoint_hash,
                  "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                      "cuda": torch.version.cuda, "device": str(device),
                      "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()},
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "source_validation": read(folder/"validation.json"), "variants": evaluations,
                  **baseline}
        original_test = folder/"test.json"
        if compare_prior and a.split == "test" and original_test.exists():
            previous = read(original_test)
            if previous["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Prior test checkpoint mismatch")
            for metric in ("rmse", "mae", "nrmse_percent", "r2"):
                if not np.isclose(previous["system"][metric], baseline["system"][metric],
                                  rtol=1e-5, atol=1e-7):
                    raise ValueError(f"Original test differs: {metric}")
        fed.write_json(target/"evaluation.json", result)
        pd.DataFrame(household_rows).to_csv(target/"households.csv", index=False)
        np.savez_compressed(target/"predictions.npz", origins=origins.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
            variants=np.asarray(variants), actual=y, predicted=predictions,
            regional_actual=regional_y, regional_predicted=regional_predictions,
            daylight=daylight, regional_counts=counts)
        fed.write_json(done, signature)
        print(f"Completed: {target}; RMSE={baseline['system']['rmse']:.6f}", flush=True)
        del model, checkpoint
    print("Frozen evaluation finished. Run the plot command next.", flush=True)
