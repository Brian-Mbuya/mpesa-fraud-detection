"""Visualization utilities for the fraud detection demo."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_amount_distribution(df: pd.DataFrame) -> None:
    """Plot amount distributions for normal and fraudulent transactions."""

    plt.figure(figsize=(10, 5))
    plt.hist(df.loc[df["is_fraud"] == 0, "amount"], bins=40, alpha=0.65, label="Normal")
    plt.hist(df.loc[df["is_fraud"] == 1, "amount"], bins=40, alpha=0.65, label="Fraud")
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hourly_fraud_rate(df: pd.DataFrame) -> None:
    """Plot the observed fraud rate by hour."""

    hourly = df.groupby("hour")["is_fraud"].mean().mul(100)
    plt.figure(figsize=(10, 5))
    plt.plot(hourly.index, hourly.values, marker="o", linewidth=2)
    plt.title("Fraud Rate by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Fraud Rate (%)")
    plt.xticks(range(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
