"""Synthetic M-Pesa-style transaction generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRANSACTION_TYPES = [
    "send_money",
    "withdraw",
    "buy_goods",
    "paybill",
    "airtime",
]

CHANNELS = ["app", "ussd", "agent", "api"]


def generate_synthetic_data(
    n: int = 5000,
    fraud_rate: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic mobile-money transactions with several fraud signals."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    fraud_count = max(1, int(round(n * fraud_rate)))

    hour = rng.integers(0, 24, size=n)
    day_of_week = rng.integers(0, 7, size=n)
    transaction_type = rng.choice(
        TRANSACTION_TYPES,
        size=n,
        p=[0.36, 0.18, 0.2, 0.16, 0.1],
    )
    channel = rng.choice(CHANNELS, size=n, p=[0.44, 0.31, 0.18, 0.07])
    account_age_days = rng.integers(7, 3650, size=n)
    is_weekend = (day_of_week >= 5).astype(int)
    recipient_is_new = rng.binomial(1, 0.18, size=n)
    failed_pin_attempts = rng.choice([0, 1, 2, 3], size=n, p=[0.82, 0.1, 0.05, 0.03])
    location_risk_score = np.clip(rng.normal(0.25, 0.15, size=n), 0.0, 1.0)
    transaction_velocity_1h = np.maximum(1, rng.poisson(1.8, size=n))

    base_amount = rng.lognormal(mean=7.2, sigma=0.75, size=n)
    type_multiplier = pd.Series(transaction_type).map(
        {
            "send_money": 1.0,
            "withdraw": 1.25,
            "buy_goods": 0.8,
            "paybill": 0.9,
            "airtime": 0.35,
        }
    )
    amount = base_amount * type_multiplier.to_numpy()

    balance_before = amount + rng.lognormal(mean=8.0, sigma=0.85, size=n)
    balance_after = np.clip(balance_before - amount, 0, None)

    is_fraud = np.zeros(n, dtype=int)
    fraud_indices = rng.choice(n, size=fraud_count, replace=False)
    is_fraud[fraud_indices] = 1

    amount[fraud_indices] *= rng.uniform(6.0, 20.0, size=fraud_count)
    hour[fraud_indices] = rng.choice([0, 1, 2, 3, 4, 23], size=fraud_count)
    recipient_is_new[fraud_indices] = 1
    failed_pin_attempts[fraud_indices] = rng.choice([2, 3, 4, 5], size=fraud_count)
    location_risk_score[fraud_indices] = rng.uniform(0.7, 1.0, size=fraud_count)
    transaction_velocity_1h[fraud_indices] = rng.integers(5, 15, size=fraud_count)

    fraud_type_override = rng.choice(
        ["send_money", "withdraw", "api_cashout"],
        size=fraud_count,
        p=[0.5, 0.3, 0.2],
    )
    fraud_channel_override = np.where(fraud_type_override == "api_cashout", "api", channel[fraud_indices])
    transaction_type = transaction_type.astype(object)
    transaction_type[fraud_indices] = np.where(
        fraud_type_override == "api_cashout",
        "withdraw",
        fraud_type_override,
    )
    channel[fraud_indices] = fraud_channel_override

    amount = np.round(amount, 2)
    balance_before = np.maximum(balance_before, amount + rng.uniform(25, 400, size=n))
    balance_after = np.round(np.clip(balance_before - amount, 0, None), 2)

    df = pd.DataFrame(
        {
            "transaction_id": [f"MPESA-{i:07d}" for i in range(1, n + 1)],
            "amount": amount,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "transaction_type": transaction_type,
            "channel": channel,
            "account_age_days": account_age_days,
            "recipient_is_new": recipient_is_new,
            "failed_pin_attempts": failed_pin_attempts,
            "location_risk_score": location_risk_score.round(3),
            "transaction_velocity_1h": transaction_velocity_1h,
            "balance_before": balance_before.round(2),
            "balance_after": balance_after,
            "is_fraud": is_fraud,
        }
    )
    return df
