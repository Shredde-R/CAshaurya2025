from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import build_feature_dataset, get_numeric_columns, summarize_dataset


OUTPUT_DIR = Path("eda_outputs")


def save_distribution_plots(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_columns = get_numeric_columns(df)
    n_cols = 2
    n_rows = int((len(numeric_columns) + n_cols - 1) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, max(5, n_rows * 3.4)))
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns):
        axes[i].hist(df[col], bins=30, color="steelblue", alpha=0.85, edgecolor="black")
        axes[i].set_title(f"{col} distribution")

    for ax in axes[len(numeric_columns):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_time_series_plots(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_columns = get_numeric_columns(df)
    fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(16, max(8, len(numeric_columns) * 2.5)), sharex=True)
    if len(numeric_columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_columns):
        ax.plot(df.index, df[col], linewidth=0.8, color="teal")
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(output_dir / "timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_correlation_plot(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_columns = get_numeric_columns(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[numeric_columns].corr()
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for row in range(len(corr.index)):
        for col in range(len(corr.columns)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_resampled_tide_plot(df: pd.DataFrame, output_dir: Path) -> None:
    daily_tide = df["Water_Level"].resample("D").mean()
    weekly_tide = df["Water_Level"].resample("W").mean()
    monthly_tide = df["Water_Level"].resample("ME").mean()

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    daily_tide.plot(ax=axes[0], title="Daily Mean Water Level", color="navy")
    weekly_tide.plot(ax=axes[1], title="Weekly Mean Water Level", color="darkgreen")
    monthly_tide.plot(ax=axes[2], title="Monthly Mean Water Level", color="darkred")
    fig.tight_layout()
    fig.savefig(output_dir / "water_level_resampled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_severity_plot(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    df["severity_class"].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#7fc97f", "#fdc086", "#ef3b2c"])
    ax.set_title("Severity Class Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "severity_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_eda_summary(df: pd.DataFrame, missing_timestamps: pd.DatetimeIndex) -> None:
    summary = summarize_dataset(df, missing_timestamps)
    print("Dataset overview")
    print(f"Rows: {summary['rows']}")
    print(f"Date range: {summary['start']} to {summary['end']}")
    print(f"Missing timestamps filled after hourly reindex: {summary['missing_timestamps_count']}")
    print()
    print("Missing timestamps sample")
    print(missing_timestamps[:10])
    print()
    print("Numeric summary")
    print(summary["numeric_summary"])
    print()
    print("Correlation with Water_Level")
    numeric_columns = get_numeric_columns(df)
    print(df[numeric_columns].corr()["Water_Level"].sort_values(ascending=False))
    print()
    print("Autocorrelation for Water_Level")
    for lag in [1, 2, 3, 6, 12, 24]:
        print(f"lag_{lag}: {df['Water_Level'].autocorr(lag=lag):.4f}")
    print()
    print("Severity distribution")
    print(df["severity_class"].value_counts(normalize=True).sort_index().round(4))


def main() -> None:
    plt.style.use("ggplot")
    OUTPUT_DIR.mkdir(exist_ok=True)

    feature_df, missing_timestamps = build_feature_dataset()
    print_eda_summary(feature_df, missing_timestamps)

    save_distribution_plots(feature_df, OUTPUT_DIR)
    save_time_series_plots(feature_df, OUTPUT_DIR)
    save_correlation_plot(feature_df, OUTPUT_DIR)
    save_resampled_tide_plot(feature_df, OUTPUT_DIR)
    save_severity_plot(feature_df, OUTPUT_DIR)

    print()
    print(f"Saved plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
