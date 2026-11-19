#!/usr/bin/env python3
"""Partition raw GG household records into three private regional silos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import numpy as np
from sklearn.cluster import KMeans


EXPECTED_CLIENT_SIZES = {0: 150, 1: 80, 2: 70}


EXPECTED_STEPS_PER_HOUSEHOLD = 52_608
RAW_COLUMNS = [
    "Customer",
    "Generator Capacity",
    "Postcode",
    "Consumption Category",
    "datetime",
    "kWh",
    "temp_c",
    "dewpoint_c",
    "wind_speed_mps",
    "precip_mm",
    "silo_grid_lat",
    "silo_grid_lon",
    "silo_solar_exposure_mj_m2",
]
OUTPUT_COLUMNS = [
    "client_id",
    "customer_id",
    "postcode",
    "datetime",
    "gg_kwh",
    "generator_capacity_kw",
    "temp_c",
    "dewpoint_c",
    "wind_speed_mps",
    "precip_mm",
    "silo_grid_lat",
    "silo_grid_lon",
    "silo_solar_exposure_mj_m2",
]


def load_gg_households(parquet_path: Path, batch_size: int) -> pd.DataFrame:
    columns = [
        "Customer",
        "Postcode",
        "Consumption Category",
        "Generator Capacity",
        "silo_grid_lat",
        "silo_grid_lon",
    ]
    pieces: list[pd.DataFrame] = []
    source = pq.ParquetFile(parquet_path)
    for batch_id, batch in enumerate(source.iter_batches(batch_size=batch_size, columns=columns), start=1):
        frame = batch.to_pandas()
        frame = frame[frame["Consumption Category"].eq("GG")].drop_duplicates("Customer")
        if not frame.empty:
            pieces.append(frame)
        print(f"scanned parquet batch {batch_id}", flush=True)
    households = pd.concat(pieces, ignore_index=True).drop_duplicates("Customer")
    if len(households) != sum(EXPECTED_CLIENT_SIZES.values()):
        raise ValueError(f"Expected 300 GG households, found {len(households)}.")
    return households


def assign_geographic_clients(households: pd.DataFrame, seed: int) -> pd.DataFrame:
    coordinates = households[["silo_grid_lat", "silo_grid_lon"]].to_numpy(dtype=np.float64)
    labels = KMeans(n_clusters=3, random_state=seed, n_init=20).fit_predict(coordinates)
    households = households.copy()
    households["_cluster"] = labels

    # The original clients are ordered south to north: Sydney, Central Coast, Newcastle/Hunter.
    ordered = (
        households.groupby("_cluster")["silo_grid_lat"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    cluster_to_client = {cluster: client_id for client_id, cluster in enumerate(ordered)}
    households["client_id"] = households["_cluster"].map(cluster_to_client).astype(int)

    sizes = households.groupby("client_id")["Customer"].nunique().to_dict()
    if sizes != EXPECTED_CLIENT_SIZES:
        raise ValueError(f"Geographic clustering produced {sizes}, expected {EXPECTED_CLIENT_SIZES}.")
    return households.drop(columns="_cluster")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../ausgrid_weather_silo_2010_2013.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument("--batch-size", type=int, default=750_000)
    parser.add_argument("--seed", type=int, default=0, help="Seed for geographic K-means only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    return args


def region_path(output_dir: Path, client_id: int) -> Path:
    return output_dir / f"client_{client_id}_households.parquet"


def prepare_outputs(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for client_id in EXPECTED_CLIENT_SIZES
        if (path := region_path(args.output_dir, client_id)).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Private silo files already exist; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    prepare_outputs(args)

    households = assign_geographic_clients(
        load_gg_households(args.source, args.batch_size),
        args.seed,
    )
    households["Customer"] = households["Customer"].astype(str)
    membership = households[
        ["Customer", "Postcode", "client_id", "silo_grid_lat", "silo_grid_lon"]
    ].rename(columns={"Customer": "customer_id", "Postcode": "postcode"})
    membership.to_csv(args.output_dir / "household_membership.csv", index=False)
    customer_to_client = households.set_index("Customer")["client_id"].to_dict()

    writers: dict[int, pq.ParquetWriter] = {}
    row_counts = {client_id: 0 for client_id in EXPECTED_CLIENT_SIZES}
    source = pq.ParquetFile(args.source)
    try:
        for batch_id, batch in enumerate(
            source.iter_batches(batch_size=args.batch_size, columns=RAW_COLUMNS),
            start=1,
        ):
            frame = batch.to_pandas()
            frame = frame[frame["Consumption Category"].eq("GG")].copy()
            if frame.empty:
                continue
            frame["Customer"] = frame["Customer"].astype(str)
            frame["client_id"] = frame["Customer"].map(customer_to_client)
            if frame["client_id"].isna().any():
                raise ValueError("A GG household has no regional assignment.")
            frame["client_id"] = frame["client_id"].astype("int8")
            frame = frame.rename(
                columns={
                    "Customer": "customer_id",
                    "Postcode": "postcode",
                    "kWh": "gg_kwh",
                    "Generator Capacity": "generator_capacity_kw",
                }
            )
            for client_id, regional in frame.groupby("client_id", sort=True):
                client_id = int(client_id)
                table = pa.Table.from_pandas(
                    regional[OUTPUT_COLUMNS],
                    preserve_index=False,
                )
                if client_id not in writers:
                    writers[client_id] = pq.ParquetWriter(
                        region_path(args.output_dir, client_id),
                        table.schema,
                        compression="zstd",
                    )
                writers[client_id].write_table(table)
                row_counts[client_id] += len(regional)
            print(f"partitioned source batch {batch_id}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    maximum_rows = {
        client_id: size * EXPECTED_STEPS_PER_HOUSEHOLD
        for client_id, size in EXPECTED_CLIENT_SIZES.items()
    }
    if any(
        row_counts[client_id] > maximum_rows[client_id]
        for client_id in EXPECTED_CLIENT_SIZES
    ):
        raise ValueError(
            f"Private-silo row counts exceed the expected maximum: {row_counts}."
        )
    complete_start = pd.Timestamp("2010-07-01 00:00:00")
    complete_end = pd.Timestamp("2013-06-30 23:30:00")
    complete_by_client: dict[int, list[str]] = {}
    incomplete_by_client: dict[int, list[str]] = {}
    for client_id in EXPECTED_CLIENT_SIZES:
        coverage = pd.read_parquet(
            region_path(args.output_dir, client_id),
            columns=["customer_id", "datetime"],
        )
        coverage["customer_id"] = coverage["customer_id"].astype(str)
        coverage["datetime"] = pd.to_datetime(coverage["datetime"])
        summary = coverage.groupby("customer_id")["datetime"].agg(
            ["size", "nunique", "min", "max"]
        )
        complete = summary[
            summary["size"].eq(EXPECTED_STEPS_PER_HOUSEHOLD)
            & summary["nunique"].eq(EXPECTED_STEPS_PER_HOUSEHOLD)
            & summary["min"].eq(complete_start)
            & summary["max"].eq(complete_end)
        ].index.astype(str)
        complete_by_client[client_id] = sorted(complete, key=int)
        incomplete_by_client[client_id] = sorted(
            set(summary.index.astype(str)) - set(complete), key=int
        )
    complete_lookup = {
        (client_id, customer_id)
        for client_id, customer_ids in complete_by_client.items()
        for customer_id in customer_ids
    }
    membership["complete_fixed_timeline"] = [
        (int(row.client_id), str(row.customer_id)) in complete_lookup
        for row in membership.itertuples(index=False)
    ]
    membership.to_csv(args.output_dir / "household_membership.csv", index=False)
    manifest = {
        "experiment": "exp2.v2",
        "source": str(args.source.resolve()),
        "privacy_boundary": "regional_client",
        "target_granularity": "raw_household_GG",
        "client_households": EXPECTED_CLIENT_SIZES,
        "complete_timeline_households": {
            client_id: len(values)
            for client_id, values in complete_by_client.items()
        },
        "incomplete_timeline_customer_ids": incomplete_by_client,
        "gg_missing_policy": "retain households; discard windows with any missing history or target GG",
        "train_start": "2010-07-01",
        "validation_start": "2012-04-01",
        "client_rows": row_counts,
        "test_start": "2012-07-01",
        "test_end_exclusive": "2013-07-01",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Exp2.v2 private silos to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
