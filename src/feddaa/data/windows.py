"""Regional Parquet inputs, training scalers and 7-day forecasting windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


STEPS_PER_DAY = 48


TRAIN_START = pd.Timestamp("2010-07-01")


VALID_START = pd.Timestamp("2012-04-01")


TEST_START = pd.Timestamp("2012-07-01")


TEST_END_EXCLUSIVE = pd.Timestamp("2013-07-01")


WEATHER_COLUMNS = [
    "temp_c_mean",
    "dewpoint_c_mean",
    "wind_speed_mps_mean",
    "precip_mm_mean",
    "silo_solar_exposure_mj_m2_mean",
    "silo_solar_exposure_lag1d_mean",
    "silo_solar_exposure_lag2d_mean",
    "silo_solar_exposure_lag7d_mean",
    "silo_clear_sky_ratio",
    "silo_halfhour_radiation_shape",
]


SOLAR_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "weekday_sin",
    "weekday_cos",
    "solar_elevation_sin",
    "solar_elevation_positive",
    "clear_sky_shape",
    "daylight_flag",
    "solar_noon_distance",
    "day_length_fraction",
]


@dataclass
class LocalScalers:
    gg_per_kw_mean: float
    gg_per_kw_std: float
    weather_mean: list[float]
    weather_std: list[float]
    capacity_log_mean: float
    capacity_log_std: float

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "gg_per_kw_mean": self.gg_per_kw_mean,
            "gg_per_kw_std": self.gg_per_kw_std,
            "weather_mean": self.weather_mean,
            "weather_std": self.weather_std,
            "capacity_log_mean": self.capacity_log_mean,
            "capacity_log_std": self.capacity_log_std,
        }

    @classmethod
    def from_dict(cls, values: dict) -> "LocalScalers":
        return cls(
            gg_per_kw_mean=float(values["gg_per_kw_mean"]),
            gg_per_kw_std=float(values["gg_per_kw_std"]),
            weather_mean=[float(value) for value in values["weather_mean"]],
            weather_std=[float(value) for value in values["weather_std"]],
            capacity_log_mean=float(values["capacity_log_mean"]),
            capacity_log_std=float(values["capacity_log_std"]),
        )


def solar_calendar_matrix(values: pd.Series | pd.DatetimeIndex | np.ndarray, latitude_deg: float) -> np.ndarray:
    dt = pd.DatetimeIndex(values)
    hour = dt.hour.to_numpy(dtype=np.float64) + dt.minute.to_numpy(dtype=np.float64) / 60.0
    day = dt.dayofyear.to_numpy(dtype=np.float64) - 1.0 + hour / 24.0
    weekday = dt.weekday.to_numpy(dtype=np.float64) + hour / 24.0

    latitude = np.deg2rad(latitude_deg)
    declination = np.deg2rad(23.44) * np.sin(2.0 * np.pi * (284.0 + day) / 365.2425)
    hour_angle = np.deg2rad(15.0 * (hour - 12.0))
    sin_elevation = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    positive_elevation = np.clip(sin_elevation, 0.0, None)
    daylight = (sin_elevation > 0.0).astype(np.float64)
    cos_sunset = np.clip(-np.tan(latitude) * np.tan(declination), -1.0, 1.0)
    day_length_fraction = (2.0 * np.arccos(cos_sunset) / (2.0 * np.pi)).astype(np.float64)
    solar_noon_distance = np.abs(hour - 12.0) / 12.0

    return np.column_stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * day / 365.2425),
            np.cos(2.0 * np.pi * day / 365.2425),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
            sin_elevation,
            positive_elevation,
            positive_elevation**1.5,
            daylight,
            solar_noon_distance,
            day_length_fraction,
        ]
    ).astype(np.float32)


def scaler_moments(frame: pd.DataFrame) -> dict[str, float | int | list[float]]:
    """Return aggregate moments that can be shared without exposing raw rows."""
    train = frame[(frame["datetime"] >= TRAIN_START) & (frame["datetime"] < VALID_START)]
    capacity = train["generator_capacity_mean_kw"].to_numpy(dtype=np.float64)
    gg_per_kw = train["gg_mean_kwh"].to_numpy(dtype=np.float64) / np.maximum(capacity, 1e-6)
    weather = train[WEATHER_COLUMNS].to_numpy(dtype=np.float64)
    capacity_log = np.log1p(capacity)
    return {
        "count": int(len(train)),
        "gg_sum": float(gg_per_kw.sum()),
        "gg_sum_sq": float(np.square(gg_per_kw).sum()),
        "weather_sum": weather.sum(axis=0).tolist(),
        "weather_sum_sq": np.square(weather).sum(axis=0).tolist(),
        "capacity_log_sum": float(capacity_log.sum()),
        "capacity_log_sum_sq": float(np.square(capacity_log).sum()),
    }


def aggregate_scaler_moments(items: list[dict]) -> LocalScalers:
    count = sum(int(item["count"]) for item in items)
    if count <= 0:
        raise ValueError("Cannot aggregate empty scaler moments.")
    gg_sum = sum(float(item["gg_sum"]) for item in items)
    gg_sum_sq = sum(float(item["gg_sum_sq"]) for item in items)
    weather_sum = np.sum(
        [np.asarray(item["weather_sum"], dtype=np.float64) for item in items],
        axis=0,
    )
    weather_sum_sq = np.sum(
        [np.asarray(item["weather_sum_sq"], dtype=np.float64) for item in items],
        axis=0,
    )
    capacity_sum = sum(float(item["capacity_log_sum"]) for item in items)
    capacity_sum_sq = sum(float(item["capacity_log_sum_sq"]) for item in items)

    gg_mean = gg_sum / count
    weather_mean = weather_sum / count
    capacity_mean = capacity_sum / count
    gg_std = np.sqrt(max(gg_sum_sq / count - gg_mean**2, 1e-12))
    weather_std = np.sqrt(np.maximum(weather_sum_sq / count - weather_mean**2, 1e-12))
    capacity_std = np.sqrt(max(capacity_sum_sq / count - capacity_mean**2, 1e-12))
    weather_std[weather_std < 1e-6] = 1.0
    return LocalScalers(
        gg_per_kw_mean=float(gg_mean),
        gg_per_kw_std=max(float(gg_std), 1e-6),
        weather_mean=weather_mean.tolist(),
        weather_std=weather_std.tolist(),
        capacity_log_mean=float(capacity_mean),
        capacity_log_std=max(float(capacity_std), 1e-6),
    )


FORECAST_STEPS = 48


HISTORY_DAYS = 7


HISTORY_STEPS = HISTORY_DAYS * STEPS_PER_DAY


WEATHER_LAG_DAYS = (1, 2, 7)


EXPECTED_TEST_WINDOWS = 365


FUTURE_WEATHER_MODE = "observed"


def split_for_origin(
    origin: pd.Timestamp,
    target_end: pd.Timestamp,
) -> str | None:
    if TRAIN_START <= origin and target_end <= VALID_START:
        return "train"
    if VALID_START <= origin and target_end <= TEST_START:
        return "val"
    if TEST_START <= origin and target_end <= TEST_END_EXCLUSIVE:
        return "test"
    return None


def make_bucket() -> dict[str, list]:
    return {
        key: []
        for key in (
            "past",
            "future",
            "target_scaled",
            "target_raw",
            "capacity",
            "baseline_scaled",
            "daylight",
            "origin",
        )
    }


def build_windows(
    frame: pd.DataFrame,
    client_id: int,
    scalers: LocalScalers,
    latitude_deg: float = -33.8688,
    train_stride_hours: int = 6,
    future_weather_mode: str = FUTURE_WEATHER_MODE,
    include_test: bool = True,
) -> dict[str, object]:
    """Build leakage-safe 7d/base windows for one client."""
    if (
        train_stride_hours <= 0
        or 24 % train_stride_hours
    ):
        raise ValueError(
            "train_stride_hours must be a positive divisor of 24."
        )
    if future_weather_mode not in {
        "lagged_mean",
        "observed",
    }:
        raise ValueError(
            "future_weather_mode must be 'lagged_mean' "
            "or 'observed'."
        )

    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime").reset_index(drop=True)
    if not include_test:
        frame = frame[frame["datetime"] < TEST_START].reset_index(drop=True)

    gg_observed = (
        frame["gg_observed"].to_numpy(dtype=np.float32) > 0.5
        if "gg_observed" in frame.columns
        else frame["gg_mean_kwh"].notna().to_numpy()
    )
    capacity = frame[
        "generator_capacity_mean_kw"
    ].to_numpy(dtype=np.float32)
    gg_raw = np.nan_to_num(np.clip(
        frame["gg_mean_kwh"].to_numpy(dtype=np.float32),
        0.0,
        None,
    ), nan=0.0)
    gg_per_kw = gg_raw / np.maximum(capacity, 1e-6)
    gg_scaled = (
        (gg_per_kw - scalers.gg_per_kw_mean)
        / scalers.gg_per_kw_std
    ).astype(np.float32)
    weather = frame[WEATHER_COLUMNS].to_numpy(
        dtype=np.float32
    )
    weather_scaled = (
        (
            weather
            - np.asarray(
                scalers.weather_mean,
                dtype=np.float32,
            )
        )
        / np.asarray(
            scalers.weather_std,
            dtype=np.float32,
        )
    ).astype(np.float32)
    capacity_scaled = (
        (
            np.log1p(capacity)
            - scalers.capacity_log_mean
        )
        / scalers.capacity_log_std
    ).astype(np.float32)
    solar = solar_calendar_matrix(
        frame["datetime"],
        latitude_deg,
    )
    past_all = np.column_stack(
        [
            gg_scaled,
            capacity_scaled,
            weather_scaled,
            solar,
        ]
    ).astype(np.float32)

    split_names = (
        ("train", "val", "test")
        if include_test
        else ("train", "val")
    )
    buckets = {
        name: make_bucket() for name in split_names
    }
    datetime_values = frame["datetime"].to_numpy()
    for origin_index in range(
        HISTORY_STEPS,
        len(frame) - FORECAST_STEPS + 1,
    ):
        origin = pd.Timestamp(
            datetime_values[origin_index]
        )
        target_end = origin + pd.Timedelta(
            minutes=30 * FORECAST_STEPS
        )
        split = split_for_origin(origin, target_end)
        if split is None or origin.minute != 0:
            continue
        if split == "test" and not include_test:
            continue
        if (
            split == "train"
            and origin.hour % train_stride_hours
        ):
            continue
        if split != "train" and origin.hour != 0:
            continue
        required_start = origin_index - HISTORY_STEPS
        required_stop = origin_index + FORECAST_STEPS
        if required_start < 0 or not np.all(gg_observed[required_start:required_stop]):
            # Never interpolate a household GG gap or join windows across it.
            continue

        lagged_weather = []
        for lag in WEATHER_LAG_DAYS:
            lag_start = (
                origin_index
                - lag * STEPS_PER_DAY
            )
            lagged_weather.append(
                weather_scaled[
                    lag_start
                    : lag_start + FORECAST_STEPS
                ]
            )
        future_weather = (
            weather_scaled[
                origin_index
                : origin_index + FORECAST_STEPS
            ]
            if future_weather_mode == "observed"
            else np.mean(
                np.stack(lagged_weather, axis=0),
                axis=0,
            ).astype(np.float32)
        )
        previous_day = gg_scaled[
            origin_index - STEPS_PER_DAY
            : origin_index
        ]
        seven_day = gg_scaled[
            origin_index - 7 * STEPS_PER_DAY
            : origin_index
        ]
        seven_day_mean = seven_day.reshape(
            7,
            STEPS_PER_DAY,
        ).mean(axis=0)
        future = np.column_stack(
            [
                future_weather,
                *lagged_weather,
                previous_day,
                seven_day_mean,
                capacity_scaled[
                    origin_index
                    : origin_index + FORECAST_STEPS
                ],
                solar[
                    origin_index
                    : origin_index + FORECAST_STEPS
                ],
            ]
        ).astype(np.float32)

        bucket = buckets[split]
        bucket["past"].append(
            past_all[
                origin_index - HISTORY_STEPS
                : origin_index
            ]
        )
        bucket["future"].append(future)
        bucket["target_scaled"].append(
            gg_scaled[
                origin_index
                : origin_index + FORECAST_STEPS
            ]
        )
        bucket["target_raw"].append(
            gg_raw[
                origin_index
                : origin_index + FORECAST_STEPS
            ]
        )
        bucket["capacity"].append(
            capacity[
                origin_index
                : origin_index + FORECAST_STEPS
            ]
        )
        bucket["baseline_scaled"].append(
            seven_day_mean
        )
        bucket["daylight"].append(
            solar[
                origin_index
                : origin_index + FORECAST_STEPS,
                SOLAR_FEATURE_NAMES.index(
                    "daylight_flag"
                ),
            ]
        )
        bucket["origin"].append(np.datetime64(origin))

    splits: dict[str, dict[str, np.ndarray]] = {}
    window_shapes = {"past": (HISTORY_STEPS, past_all.shape[-1]),
                     "future": (FORECAST_STEPS, 4 * len(WEATHER_COLUMNS) + 3 + solar.shape[-1])}
    for split, bucket in buckets.items():
        splits[split] = {
            key: (np.stack(values).astype(np.float32) if values else
                  np.empty((0, *window_shapes.get(key, (FORECAST_STEPS,))), dtype=np.float32))
            for key, values in bucket.items()
            if key != "origin"
        }
        splits[split]["origin"] = np.asarray(bucket["origin"], dtype="datetime64[ns]")

    if include_test:
        test_origins = pd.DatetimeIndex(
            splits["test"]["origin"]
        )
        expected_last = (
            TEST_END_EXCLUSIVE - pd.Timedelta(days=1)
        )
        complete_observation = bool(np.all(gg_observed))
        if complete_observation and (
            len(test_origins) != EXPECTED_TEST_WINDOWS
            or test_origins.min() != TEST_START
            or test_origins.max() != expected_last
        ):
            raise ValueError(
                "Full-year test coverage check failed: "
                f"expected {EXPECTED_TEST_WINDOWS} daily "
                f"origins from {TEST_START.date()} through "
                f"{expected_last.date()}, got "
                f"{len(test_origins)} origins from "
                f"{test_origins.min()} through "
                f"{test_origins.max()}."
            )

    return {
        "client_id": client_id,
        "splits": splits,
        "past_dim": window_shapes["past"][-1],
        "future_dim": window_shapes["future"][-1],
        "scalers": scalers,
        "n_customers": int(
            frame["n_customers"].iloc[0]
        ),
        "future_weather_mode": future_weather_mode,
    }


def training_loader(
    split: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split["past"]),
            torch.from_numpy(split["future"]),
            torch.from_numpy(
                split["target_scaled"]
            ),
            torch.from_numpy(
                split["baseline_scaled"]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def restore_gg(
    scaled: np.ndarray,
    capacity: np.ndarray,
    scalers: LocalScalers,
) -> np.ndarray:
    gg_per_kw = (
        scaled * scalers.gg_per_kw_std
        + scalers.gg_per_kw_mean
    )
    return np.clip(
        gg_per_kw * np.maximum(capacity, 1e-6),
        0.0,
        None,
    )


def predict_model_split(
    model: torch.nn.Module,
    data: dict[str, object],
    split_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    split = data["splits"][split_name]
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(
            0,
            len(split["past"]),
            batch_size,
        ):
            stop = start + batch_size
            residual = model(
                torch.from_numpy(
                    split["past"][start:stop]
                ).to(device),
                torch.from_numpy(
                    split["future"][start:stop]
                ).to(device),
            ).cpu().numpy()
            outputs.append(
                split["baseline_scaled"][start:stop]
                + residual
            )
    return restore_gg(
        np.concatenate(outputs, axis=0),
        split["capacity"],
        data["scalers"],
    )


def metric_values(
    actual: np.ndarray,
    predicted: np.ndarray,
    daylight: np.ndarray | None = None,
) -> dict[str, float | int]:
    actual_values = np.asarray(
        actual,
        dtype=np.float64,
    ).reshape(-1)
    predicted_values = np.asarray(
        predicted,
        dtype=np.float64,
    ).reshape(-1)
    error = predicted_values - actual_values
    rmse = float(
        np.sqrt(np.mean(np.square(error)))
    )
    target_range = float(
        actual_values.max() - actual_values.min()
    )
    denominator = float(
        np.sum(
            np.square(
                actual_values - actual_values.mean()
            )
        )
    )
    values: dict[str, float | int] = {
        "n": int(len(actual_values)),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "nrmse_percent": (
            100.0 * rmse / target_range
            if target_range > 0
            else float("nan")
        ),
        "r2": (
            1.0
            - float(np.sum(np.square(error)))
            / denominator
            if denominator > 0
            else float("nan")
        ),
    }
    if daylight is None:
        return values

    mask = np.asarray(daylight).reshape(-1) > 0.5
    if not mask.any():
        values.update(
            {
                "daylight_rmse": float("nan"),
                "daylight_mae": float("nan"),
                "daylight_r2": float("nan"),
            }
        )
        return values
    daylight_actual = actual_values[mask]
    daylight_error = error[mask]
    daylight_denominator = float(
        np.sum(
            np.square(
                daylight_actual
                - daylight_actual.mean()
            )
        )
    )
    values.update(
        {
            "daylight_rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(daylight_error)
                    )
                )
            ),
            "daylight_mae": float(
                np.mean(np.abs(daylight_error))
            ),
            "daylight_r2": (
                1.0
                - float(
                    np.sum(
                        np.square(daylight_error)
                    )
                )
                / daylight_denominator
                if daylight_denominator > 0
                else float("nan")
            ),
        }
    )
    return values


RAW_REGION_COLUMNS = [
    "client_id",
    "customer_id",
    "datetime",
    "gg_kwh",
    "generator_capacity_kw",
    "temp_c",
    "dewpoint_c",
    "wind_speed_mps",
    "precip_mm",
    "silo_solar_exposure_mj_m2",
]


EXPECTED_START = pd.Timestamp("2010-07-01 00:00:00")


EXPECTED_END = pd.Timestamp("2013-06-30 23:30:00")


EXPECTED_INDEX = pd.date_range(EXPECTED_START, EXPECTED_END, freq="30min")


class RegionHouseholdStore:
    """A private regional silo that exposes one household at a time."""

    def __init__(
        self,
        path: Path,
        client_id: int,
        latitude_deg: float = -33.8688,
    ) -> None:
        self.path = path
        self.client_id = client_id
        self.latitude_deg = latitude_deg
        raw = pd.read_parquet(path, columns=RAW_REGION_COLUMNS)
        if raw.empty:
            raise ValueError(f"Private region file is empty: {path}")
        raw["client_id"] = pd.to_numeric(
            raw["client_id"], errors="raise"
        ).astype(int)
        found = set(raw["client_id"].unique())
        if found != {client_id}:
            raise ValueError(
                f"{path} contains client ids {sorted(found)}, expected {client_id}."
            )
        raw["customer_id"] = raw["customer_id"].astype(str)
        raw["datetime"] = pd.to_datetime(raw["datetime"])
        raw = raw.sort_values(
            ["customer_id", "datetime"]
        ).reset_index(drop=True)
        if raw.duplicated(["customer_id", "datetime"]).any():
            raise ValueError(f"Duplicate household timestamps in {path}.")
        if not raw["datetime"].isin(EXPECTED_INDEX).all():
            raise ValueError(f"Out-of-period or unaligned household timestamps in {path}.")
        # Keep every household in the fixed roster. Reindexing marks absent readings;
        # the window builder then skips only windows lacking their required GG values.
        self._raw = raw
        self.customer_ids = tuple(sorted(raw["customer_id"].unique(), key=int))

    @property
    def n_customers(self) -> int:
        return len(self.customer_ids)

    def household_frame(self, customer_id: str) -> pd.DataFrame:
        raw = self._raw[
            self._raw["customer_id"].eq(str(customer_id))
        ].copy()
        if raw.empty:
            raise KeyError(customer_id)
        raw = raw.sort_values("datetime")
        raw = raw.set_index("datetime").reindex(EXPECTED_INDEX)
        raw.index.name = "datetime"
        raw["gg_observed"] = raw["gg_kwh"].notna().astype(np.float32)
        raw["client_id"] = self.client_id
        raw["customer_id"] = str(customer_id)
        raw = raw.reset_index()
        target = pd.to_numeric(raw["gg_kwh"], errors="coerce")
        if target[raw["gg_observed"].gt(0.5)].isna().any():
            raise ValueError(
                f"Customer {customer_id} has an invalid observed GG target."
            )
        raw["gg_kwh"] = target
        raw["generator_capacity_kw"] = pd.to_numeric(
            raw["generator_capacity_kw"], errors="coerce"
        ).interpolate(limit_direction="both")
        weather_numeric = [
            "temp_c",
            "dewpoint_c",
            "wind_speed_mps",
            "precip_mm",
            "silo_solar_exposure_mj_m2",
        ]
        raw[weather_numeric] = raw[weather_numeric].apply(pd.to_numeric, errors="coerce")
        # Do not let interpolation fill the end of a training split with a later
        # validation or test observation. At an edge use the nearest available
        # observation within the same split.
        period = np.searchsorted([VALID_START.value, TEST_START.value],
                                 raw["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64"),
                                 side="right")
        raw[weather_numeric] = raw[weather_numeric].groupby(period).transform(
            lambda values: values.interpolate(limit_direction="both")
        )
        if raw[["generator_capacity_kw", *weather_numeric]].isna().any().any():
            raise ValueError(f"Customer {customer_id} has unresolved missing data.")
        capacity = raw["generator_capacity_kw"]
        if not np.isfinite(capacity).all() or capacity.iloc[0] <= 0 or capacity.nunique() != 1:
            raise ValueError(f"Customer {customer_id} needs one fixed positive installation capacity.")

        frame = pd.DataFrame(
            {
                "client_id": self.client_id,
                "datetime": raw["datetime"].to_numpy(),
                "n_customers": 1,
                "gg_mean_kwh": raw["gg_kwh"].clip(lower=0).to_numpy(),
                "gg_observed": raw["gg_observed"].to_numpy(),
                "generator_capacity_mean_kw": raw[
                    "generator_capacity_kw"
                ].to_numpy(),
                "temp_c_mean": raw["temp_c"].to_numpy(),
                "dewpoint_c_mean": raw["dewpoint_c"].to_numpy(),
                "wind_speed_mps_mean": raw["wind_speed_mps"].to_numpy(),
                "precip_mm_mean": raw["precip_mm"].to_numpy(),
                "silo_solar_exposure_mj_m2_mean": raw[
                    "silo_solar_exposure_mj_m2"
                ].to_numpy(),
            }
        )
        exposure = frame["silo_solar_exposure_mj_m2_mean"]
        for lag in (1, 2, 7):
            frame[f"silo_solar_exposure_lag{lag}d_mean"] = exposure.shift(
                lag * STEPS_PER_DAY
            )

        solar = solar_calendar_matrix(frame["datetime"], self.latitude_deg)
        clear_shape = solar[
            :, SOLAR_FEATURE_NAMES.index("clear_sky_shape")
        ].astype(np.float64)
        daylight = solar[
            :, SOLAR_FEATURE_NAMES.index("daylight_flag")
        ]
        day_length = solar[
            :, SOLAR_FEATURE_NAMES.index("day_length_fraction")
        ]
        dates = frame["datetime"].dt.normalize()
        daily_shape = pd.Series(clear_shape).groupby(dates).transform("sum")
        halfhour_share = np.divide(
            clear_shape,
            daily_shape.to_numpy(),
            out=np.zeros_like(clear_shape),
            where=daily_shape.to_numpy() > 0,
        )
        frame["silo_halfhour_radiation_shape"] = (
            exposure.to_numpy() * halfhour_share
        )
        frame["silo_clear_sky_ratio"] = exposure.to_numpy() / np.maximum(
            day_length * 24.0,
            1e-3,
        )
        frame.loc[daylight < 0.5, "silo_clear_sky_ratio"] = 0.0
        frame[list(WEATHER_COLUMNS)] = frame[list(WEATHER_COLUMNS)].interpolate(
            limit_direction="both"
        )
        return frame

    def build_data(
        self,
        customer_id: str,
        scalers: LocalScalers,
        train_stride_hours: int,
        future_weather_mode: str,
        include_test: bool,
    ) -> dict[str, object]:
        frame = self.household_frame(customer_id)
        if not include_test:
            frame = frame[frame["datetime"] < TEST_START].reset_index(drop=True)
        data = build_windows(
            frame,
            self.client_id,
            scalers,
            latitude_deg=self.latitude_deg,
            train_stride_hours=train_stride_hours,
            future_weather_mode=future_weather_mode,
            include_test=include_test,
        )
        data["customer_id"] = str(customer_id)
        return data
