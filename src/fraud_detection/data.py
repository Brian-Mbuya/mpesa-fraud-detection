import pandas as pd
import numpy as np


def generate_synthetic_data(n=5000, seed=42, fraud_count=120):
    """Generate synthetic transaction data with an injected fraud signal.

    Returns a pandas DataFrame with columns: amount, time_of_day, transaction_type, is_fraud
    """
    np.random.seed(seed)

    amount = np.random.exponential(scale=1500, size=n)
    time_of_day = np.random.randint(0, 24, n)
    transaction_type = np.random.choice(
        ['send_money', 'withdraw', 'buy_goods', 'paybill'],
        size=n,
        p=[0.4, 0.2, 0.25, 0.15],
    )

    is_fraud = np.zeros(n, dtype=int)
    fraud_indices = np.random.choice(n, size=fraud_count, replace=False)
    is_fraud[fraud_indices] = 1

    # fraud patterns: very high amount late night
    amount[fraud_indices] = np.random.uniform(50000, 150000, size=len(fraud_indices))
    time_of_day[fraud_indices] = np.random.choice([0, 1, 2, 3], size=len(fraud_indices))

    df = pd.DataFrame({
        'amount': amount,
        'time_of_day': time_of_day,
        'transaction_type': transaction_type,
        'is_fraud': is_fraud,
    })

    return df
