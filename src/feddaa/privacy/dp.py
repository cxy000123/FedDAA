#!/usr/bin/env python3
"""Household-level DP primitives and a Poisson-subsampled RDP accountant."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch


DEFAULT_ORDERS = tuple(range(2, 257))


def poisson_sample_households(
    customer_ids: tuple[str, ...], sample_rate: float, seed: int
) -> list[str]:
    """Independently include every household with probability ``sample_rate``."""
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1].")
    rng = np.random.default_rng(seed)
    mask = rng.random(len(customer_ids)) < sample_rate
    return [customer_ids[index] for index in np.flatnonzero(mask)]


def _l2_norm(delta: dict[str, torch.Tensor]) -> float:
    squared = sum(
        float(torch.sum(value.detach().cpu().double().square()))
        for value in delta.values()
    )
    return math.sqrt(squared)


def clip_household_delta(
    delta: dict[str, torch.Tensor], clip_norm: float
) -> tuple[dict[str, torch.Tensor], float, float]:
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive.")
    raw_norm = _l2_norm(delta)
    scale = min(1.0, clip_norm / (raw_norm + 1e-12))
    return (
        {key: value.detach().cpu().float() * scale for key, value in delta.items()},
        raw_norm,
        scale,
    )


def private_household_sum(
    deltas: list[dict[str, torch.Tensor]],
    template: dict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    normalization: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Release a Gaussian-noised sum of individually clipped updates.

    The neighbouring relation is add/remove-one household. The L2 sensitivity
    of the clipped sum is ``clip_norm``. Division by the public, fixed
    ``normalization`` is post-processing and therefore does not change DP.
    Empty Poisson samples correctly release pure Gaussian noise.
    """
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive for DP training.")
    if normalization <= 0:
        raise ValueError("normalization must be positive.")
    clipped: list[dict[str, torch.Tensor]] = []
    raw_norms: list[float] = []
    scales: list[float] = []
    for delta in deltas:
        item, raw_norm, scale = clip_household_delta(delta, clip_norm)
        clipped.append(item)
        raw_norms.append(raw_norm)
        scales.append(scale)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: dict[str, torch.Tensor] = {}
    for key, value in template.items():
        total = torch.zeros_like(value, device="cpu", dtype=torch.float32)
        for item in clipped:
            total.add_(item[key])
        noise = torch.normal(
            mean=0.0,
            std=clip_norm * noise_multiplier,
            size=total.shape,
            generator=generator,
        )
        result[key] = (total + noise) / normalization
    diagnostics = {
        "sampled_households": float(len(deltas)),
        "clipped_households": float(sum(scale < 1.0 for scale in scales)),
        "mean_raw_update_norm": float(np.mean(raw_norms)) if raw_norms else 0.0,
        "max_raw_update_norm": float(np.max(raw_norms)) if raw_norms else 0.0,
        "mean_clip_scale": float(np.mean(scales)) if scales else 1.0,
    }
    return result, diagnostics


def clip_released_delta(
    delta: dict[str, torch.Tensor], max_norm: float
) -> tuple[dict[str, torch.Tensor], float, float]:
    """Bound a released DP update by post-processing (no additional privacy cost)."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")
    raw_norm = _l2_norm(delta)
    scale = min(1.0, max_norm / (raw_norm + 1e-12))
    return (
        {key: value.detach().cpu().float() * scale for key, value in delta.items()},
        raw_norm,
        scale,
    )


def _log_add(log_x: float, log_y: float) -> float:
    if log_x == -math.inf:
        return log_y
    if log_y == -math.inf:
        return log_x
    maximum = max(log_x, log_y)
    return maximum + math.log(math.exp(log_x - maximum) + math.exp(log_y - maximum))


def poisson_gaussian_rdp(
    sample_rate: float,
    noise_multiplier: float,
    orders: Iterable[int] = DEFAULT_ORDERS,
) -> dict[int, float]:
    """Exact integer-order RDP bound for the Poisson-sampled Gaussian mechanism."""
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1].")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")
    result: dict[int, float] = {}
    log_q = math.log(sample_rate)
    log_one_minus_q = math.log1p(-sample_rate) if sample_rate < 1 else -math.inf
    sigma_sq = noise_multiplier**2
    for order_value in orders:
        order = int(order_value)
        if order < 2 or order != order_value:
            raise ValueError("RDP orders must be integers >= 2.")
        log_a = -math.inf
        for index in range(order + 1):
            if sample_rate == 1 and index != order:
                continue
            log_binomial = (
                math.lgamma(order + 1)
                - math.lgamma(index + 1)
                - math.lgamma(order - index + 1)
            )
            term = log_binomial + index * log_q
            if order != index:
                term += (order - index) * log_one_minus_q
            term += (index * index - index) / (2.0 * sigma_sq)
            log_a = _log_add(log_a, term)
        result[order] = log_a / (order - 1)
    return result


def epsilon_from_rdp(
    sample_rate: float,
    noise_multiplier: float,
    rounds: int,
    delta: float,
    orders: Iterable[int] = DEFAULT_ORDERS,
) -> tuple[float, int]:
    if rounds <= 0:
        raise ValueError("rounds must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    per_round = poisson_gaussian_rdp(sample_rate, noise_multiplier, orders)
    candidates = {
        order: rounds * rdp + math.log(1.0 / delta) / (order - 1)
        for order, rdp in per_round.items()
    }
    best_order = min(candidates, key=candidates.get)
    return candidates[best_order], best_order


def calibrate_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    rounds: int,
    delta: float,
    tolerance: float = 1e-5,
) -> float:
    """Find the smallest sigma whose accounted epsilon is at most the target."""
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive.")
    low, high = 1e-3, 1.0
    while epsilon_from_rdp(sample_rate, high, rounds, delta)[0] > target_epsilon:
        high *= 2.0
        if high > 1e6:
            raise RuntimeError("Could not bracket a noise multiplier.")
    while high - low > tolerance * max(1.0, high):
        middle = (low + high) / 2.0
        if epsilon_from_rdp(sample_rate, middle, rounds, delta)[0] <= target_epsilon:
            high = middle
        else:
            low = middle
    return high
