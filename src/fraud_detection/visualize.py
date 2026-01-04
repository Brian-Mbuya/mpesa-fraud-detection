import matplotlib.pyplot as plt
import pandas as pd


def plot_amount_distribution(df, bins=50):
    plt.figure()
    df['amount'].hist(bins=bins)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.show()


def plot_avg_amount_by_label(df):
    plt.figure()
    df.groupby('is_fraud')['amount'].mean().plot(kind='bar')
    plt.title("Average Amount: Fraud vs Normal")
    plt.xlabel("Fraud (1) vs Normal (0)")
    plt.ylabel("Average Amount")
    plt.show()
