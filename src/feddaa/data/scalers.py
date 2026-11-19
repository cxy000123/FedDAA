"""Fit fixed normalization parameters on observed training records."""

import argparse
from pathlib import Path

from feddaa.data.windows import VALID_START, RegionHouseholdStore, aggregate_scaler_moments, scaler_moments
from feddaa.federated import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/scalers.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    moments = []
    for region in range(3):
        store = RegionHouseholdStore(args.data_dir / f"client_{region}_households.parquet", region)
        for customer in store.customer_ids:
            frame = store.household_frame(customer)
            train = frame["datetime"].lt(VALID_START) & frame["gg_observed"].gt(0.5)
            moments.append(scaler_moments(frame.loc[train]))
        print(f"Region {region + 1}: {store.n_customers} households", flush=True)
        del store
    write_json(args.output, aggregate_scaler_moments(moments).to_dict())
    write_json(args.output.with_suffix(args.output.suffix + ".provenance.json"), {
        "sha256": digest(args.output),
        "source": str(args.data_dir.resolve()),
        "fit_period_end_exclusive": str(VALID_START),
        "validation_data_used": False,
        "test_data_used": False,
        "privacy_scope": "non_private_fixed_preprocessing",
    })
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
