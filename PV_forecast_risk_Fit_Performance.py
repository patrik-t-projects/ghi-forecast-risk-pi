import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PV_forecast_risk_functions import fit_risk_score_from_features
from config import DATA_DIR


# =========================
# USER SELECTION
# =========================
START_DATE = "2026-03-21"
END_DATE   = "2026-05-10"
MODEL = "ICON1"
GHI_threshold = 0

FIGSIZE = (10, 5)

Folder = str(DATA_DIR)

# Feature columns used as input to the fitted MODEL
feature_cols = [
    f"{MODEL} range",
    f"{MODEL} std",
    f"{MODEL} ens spread",
    f"{MODEL} CC ens spread",
    f"{MODEL} horizon spread",
    f"{MODEL} forecast update total abs",
]


def build_risk_feature_dataframe(
    df_rad,
    df_metrics,
    model="ICON1",
    start_date="2026-03-21",
    end_date="2026-05-08",
    ghi_threshold=0,
    target_horizons=(0,),
):
    """
    Builds the feature dataframe used for fitting and plotting.

    Inputs
    ------
    df_rad:
        Radiation dataframe. Must contain:
        - Datetime
        - MeteoSwiss stations
        - {model} prev day 0/1/2/3
        - {model} max
        - {model} min
        - {model} std
        - {model} ens spread
        - {model} CC ens spread

    df_metrics:
        Metrics dataframe. Must contain:
        - Datetime
        - MeteoSwiss - {model}_d0, d1, d2, d3 depending on target_horizons

    Returns
    -------
    df_features:
        Clean dataframe containing:
        - Datetime UTC
        - forecast error columns
        - risk feature columns
        - MeteoSwiss stations

    feature_info:
        Dictionary containing:
        - feature_cols
        - metric_cols
        - horizon_colors
        - model
        - start_date
        - end_date
        - ghi_threshold
    """

    # =========================
    # Forecast error columns
    # =========================
    metric_cols = [
        f"MeteoSwiss - {model}_d{horizon}"
        for horizon in target_horizons
    ]

    horizon_colors = {
        f"MeteoSwiss - {model}_d0": "tab:green",
        f"MeteoSwiss - {model}_d1": "tab:orange",
        f"MeteoSwiss - {model}_d2": "tab:red",
        f"MeteoSwiss - {model}_d3": "tab:purple",
    }

    # =========================
    # Make copies
    # =========================
    df_rad = df_rad.copy()
    df_metrics = df_metrics.copy()

    # =========================
    # Datetime handling
    # =========================
    if "Datetime UTC" not in df_rad.columns:
        df_rad["Datetime UTC"] = pd.to_datetime(df_rad["Datetime"], utc=True)

    if "Datetime" in df_rad.columns:
        df_rad = df_rad.drop(columns="Datetime")

    if "Datetime UTC" not in df_metrics.columns:
        df_metrics["Datetime UTC"] = pd.to_datetime(df_metrics["Datetime"], utc=True)

    if "Datetime" in df_metrics.columns:
        df_metrics = df_metrics.drop(columns="Datetime")

    # =========================
    # Select date range
    # =========================
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)

    df_rad_sel = df_rad[
        (df_rad["Datetime UTC"] >= start_dt)
        & (df_rad["Datetime UTC"] <= end_dt)
    ].copy()

    df_metrics_sel = df_metrics[
        (df_metrics["Datetime UTC"] >= start_dt)
        & (df_metrics["Datetime UTC"] <= end_dt)
    ].copy()

    # ========================================
    # Time-resolved forecast-risk diagnostics
    # ========================================
    horizon_forecast_cols = [
        f"{model} prev day 0",
        f"{model} prev day 1",
        f"{model} prev day 2",
        f"{model} prev day 3",
    ]

    df_features_part = df_rad_sel[["Datetime UTC"]].copy()

    # 1) Horizon spread over the four forecasts
    df_features_part[f"{model} horizon spread"] = df_rad_sel[horizon_forecast_cols].std(axis=1)

    # 2) Forecast updates between horizons
    df_features_part[f"{model} update d1_to_d0"] = (
        df_rad_sel[f"{model} prev day 0"]
        - df_rad_sel[f"{model} prev day 1"]
    )

    df_features_part[f"{model} update d2_to_d1"] = (
        df_rad_sel[f"{model} prev day 1"]
        - df_rad_sel[f"{model} prev day 2"]
    )

    df_features_part[f"{model} update d3_to_d2"] = (
        df_rad_sel[f"{model} prev day 2"]
        - df_rad_sel[f"{model} prev day 3"]
    )

    df_features_part[f"{model} forecast update total abs"] = (
        df_features_part[f"{model} update d1_to_d0"].abs()
        + df_features_part[f"{model} update d2_to_d1"].abs()
        + df_features_part[f"{model} update d3_to_d2"].abs()
    )

    # 3) Ensemble range and spread variables
    df_features_part[f"{model} range"] = (
        df_rad_sel[f"{model} max"]
        - df_rad_sel[f"{model} min"]
    )

    df_features_part[f"{model} std"] = df_rad_sel[f"{model} std"]
    df_features_part[f"{model} ens spread"] = df_rad_sel[f"{model} ens spread"]
    df_features_part[f"{model} CC ens spread"] = df_rad_sel[f"{model} CC ens spread"]

    # ==================================================
    # Feature columns used as input to the fitted model
    # IMPORTANT: does NOT include fitted risk score.
    # ==================================================
    feature_cols = [
        f"{model} range",
        f"{model} std",
        f"{model} ens spread",
        f"{model} CC ens spread",
        f"{model} horizon spread",
        f"{model} forecast update total abs",
    ]

        # ===============================
    # Merge forecast errors + features
    # ===============================

    available_metric_cols = [
        col for col in metric_cols
        if col in df_metrics_sel.columns and not df_metrics_sel[col].isna().all()
    ]

    if len(available_metric_cols) > 0:
        df_features = df_metrics_sel[["Datetime UTC"] + available_metric_cols].merge(
            df_features_part[["Datetime UTC"] + feature_cols],
            on="Datetime UTC",
            how="inner",
        )
    else:
        df_features = df_features_part[["Datetime UTC"] + feature_cols].copy()

    # ===================================
    # Add measured radiation if available
    # ===================================

    if (
        "MeteoSwiss stations" in df_rad_sel.columns
        and not df_rad_sel["MeteoSwiss stations"].isna().all()
    ):
        df_features = df_features.merge(
            df_rad_sel[["Datetime UTC", "MeteoSwiss stations"]],
            on="Datetime UTC",
            how="left",
        )

        # Apply daylight mask only if measured radiation is available
        daylight_mask_fit = df_features["MeteoSwiss stations"] > ghi_threshold
        df_features = df_features[daylight_mask_fit].copy()

    else:
        # Future case: no measured MeteoSwiss stations available
        # Do not apply daylight mask here.
        pass

    df_features = df_features.sort_values("Datetime UTC").reset_index(drop=True)

    # =========================
    # Info dictionary
    # =========================
    feature_info = {
        "model": model,
        "feature_cols": feature_cols,
        "metric_cols": available_metric_cols,
        "requested_metric_cols": metric_cols,
        "horizon_colors": horizon_colors,
        "start_date": start_date,
        "end_date": end_date,
        "ghi_threshold": ghi_threshold,
        "target_horizons": target_horizons,
    }

    return df_features, feature_info




# ============================================================
# Helper function:
# Heatmap + scatterplots for any time resolution
# ============================================================

def plot_risk_metric_performance(
    df_source,
    error_cols,
    risk_cols,
    error_label_prefix,
    title_prefix,
    model,
    start_date,
    end_date,
):
    """
    Creates:
    1) Heatmap of Kendall-tau correlation
    2) Scatterplot subplots for all risk metrics vs selected error columns

    df_source:
        DataFrame containing error_cols and risk_cols.

    error_cols:
        Columns containing forecast error values.

    risk_cols:
        Columns containing risk metric values.

    error_label_prefix:
        Label used for x-axis.

    title_prefix:
        Example: "Hourly" or "Daily"

    model:
        ICON1 or ICON2
    """

    # =========================
    # Performance table
    # =========================
    performance_table = pd.DataFrame(index=risk_cols, columns=error_cols, dtype=float)
    rank_rmse_table = pd.DataFrame(index=risk_cols, columns=error_cols, dtype=float)

    for risk_col in risk_cols:
        for error_col in error_cols:

            tmp = df_source[[risk_col, error_col]].copy()

            # Remove rows where this specific risk metric or error is missing
            tmp = tmp.dropna(subset=[risk_col, error_col])

            # Remove artificial zero-risk values from periods where the risk metric was not available
            tmp = tmp[tmp[risk_col] != 0].copy()

            # Use absolute values consistently with the scatterplots
            x_error = tmp[error_col].abs()
            y_risk = tmp[risk_col].abs()

            # Rank-based deviation from perfect monotonic ordering
            error_rank = x_error.rank(method="average", pct=True)
            risk_rank = y_risk.rank(method="average", pct=True)
            rank_error = risk_rank - error_rank
            rank_rmse = np.sqrt((rank_error ** 2).mean())

            # Need at least 3 points and at least 2 unique values on both sides
            if (
                len(tmp) < 3
                or x_error.nunique() < 2
                or y_risk.nunique() < 2
            ):
                performance_table.loc[risk_col, error_col] = np.nan
                rank_rmse_table.loc[risk_col, error_col] = np.nan
                continue

            kendall_tau_corr = y_risk.corr(x_error, method="kendall")

            performance_table.loc[risk_col, error_col] = 100 * (kendall_tau_corr + 1) / 2
            rank_rmse_table.loc[risk_col, error_col] = 100 * rank_rmse

    combined_output_table = pd.concat(
        {
            "Kendall-tau [%]": performance_table,
            "Rank RMSE [%]": rank_rmse_table,
        },
        axis=1,
    )

    print(f"\n{title_prefix} risk metric performance [%]:")
    print(combined_output_table)

    # =========================
    # Plot performance heatmap
    # =========================
    fig_hm, ax_hm = plt.subplots(figsize=(10, 5))

    heatmap_data = performance_table.astype(float).values

    im = ax_hm.imshow(heatmap_data, aspect="auto", vmin=-100, vmax=100)

    ax_hm.set_xticks(np.arange(len(error_cols)))
    ax_hm.set_yticks(np.arange(len(risk_cols)))

    ax_hm.set_xticklabels(error_cols, rotation=35, ha="right")
    ax_hm.set_yticklabels(risk_cols)

    ax_hm.set_title(
        f"{model}: {title_prefix} risk metric performance vs forecast error\n"
        f"Kendall-tau correlation in %, {start_date} to {end_date}"
    )

    cbar = plt.colorbar(im, ax=ax_hm)
    cbar.set_label("Performance score [%]")

    # Text annotations
    for i in range(len(risk_cols)):
        for j in range(len(error_cols)):
            value = heatmap_data[i, j]

            if np.isnan(value):
                text = "n/a"
            else:
                text = f"{value:.1f}%"

            ax_hm.text(j, i, text, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    # =========================
    # Scatterplots
    # x-axis: abs forecast error
    # y-axis: risk metric
    # =========================
    for error_col in error_cols:

        n_metrics = len(risk_cols)
        ncols = 3
        nrows = int(np.ceil(n_metrics / ncols))

        fig_scatter, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=FIGSIZE,
        )

        axes = np.array(axes).reshape(-1)

        for i, risk_col in enumerate(risk_cols):
            ax = axes[i]

            tmp = df_source[[error_col, risk_col]].copy()

            # Remove rows where this specific risk metric or error is missing
            tmp = tmp.dropna(subset=[error_col, risk_col])

            # Remove artificial zero-risk values from periods where the risk metric was not available
            tmp = tmp[tmp[risk_col] != 0].copy()

            ax.scatter(
                tmp[error_col].abs(),
                tmp[risk_col].abs(),
                s=8,
                alpha=0.8,
            )

            ax.set_xlabel(error_label_prefix)
            ax.set_ylabel(f"{title_prefix} {risk_col}")
            ax.grid(True, alpha=0.3)

            # Show Kendall-tau performance in title
            x_error = tmp[error_col].abs()
            y_risk = tmp[risk_col].abs()

            if (
                len(tmp) >= 3
                and x_error.nunique() > 1
                and y_risk.nunique() > 1
            ):
                kendall_tau_corr = x_error.corr(y_risk, method="kendall")
                ax.set_title(f"{risk_col}\nKendall-tau = {100 * kendall_tau_corr:.1f}%")
            else:
                ax.set_title(f"{risk_col}\nKendall-tau = n/a")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            f"{model}: {title_prefix} {error_col} vs risk metrics\n"
            f"{start_date} to {end_date}",
            y=0.98,
        )

        plt.tight_layout()
        plt.show()

    return performance_table, rank_rmse_table


# ============================================================
# Main script execution
# ============================================================

if __name__ == "__main__":

    # =========================
    # Load data
    # =========================
    df_rad = pd.read_csv(os.path.join(Folder, "station_radiation_CH_interp_year_2026.csv"), sep=";")
    df_metrics = pd.read_csv(os.path.join(Folder, "station_metrics_CH_interp_year_2026.csv"), sep=";")

    # =========================
    # Build features
    # =========================
    df_features, feature_info = build_risk_feature_dataframe(
        df_rad=df_rad,
        df_metrics=df_metrics,
        model=MODEL,
        start_date=START_DATE,
        end_date=END_DATE,
        ghi_threshold=GHI_threshold,
        target_horizons=(0,),
    )

    # =========================
    # Fit risk score
    # =========================
    fit_result, plot_context = fit_risk_score_from_features(
        df_features=df_features,
        feature_cols=feature_info["feature_cols"],
        metric_cols=feature_info["metric_cols"],
        model=feature_info["model"],
        start_date=feature_info["start_date"],
        end_date=feature_info["end_date"],
        ghi_threshold=feature_info["ghi_threshold"],
        figsize=FIGSIZE,
        horizon_colors=feature_info["horizon_colors"],
        use_rank_features=True,
        nonnegative_weights=True,
        verbose=True,
        top_risk_penalty_weight=0.5,
        top_quantile=0.9,
    )

    # ======================================
    # Unpack for your existing plotting code
    # ======================================
    df_plot = plot_context["df_plot"]
    risk_metric_cols = plot_context["risk_metric_cols"]
    metric_cols = plot_context["metric_cols"]
    horizon_colors = plot_context["horizon_colors"]

    MODEL = plot_context["MODEL"]
    START_DATE = plot_context["START_DATE"]
    END_DATE = plot_context["END_DATE"]
    GHI_threshold = plot_context["GHI_threshold"]
    FIGSIZE = plot_context["FIGSIZE"]


    # =========================
    # Plot
    # =========================

    # fig, ax1 = plt.subplots(figsize=FIGSIZE)

    # # Left y-axis: metrics
    # for col in metric_cols:
    #     ax1.plot(
    #         df_plot["Datetime UTC"],
    #         df_plot[col].abs(),
    #         label=col,
    #         color=horizon_colors[col],
    #         linewidth=1.8,
    #     )

    # ax1.set_xlabel("Datetime UTC")
    # ax1.set_ylabel("GHI forecast error CH")
    # ax1.grid(True, alpha=0.3)

    # # Right y-axis: radiation / ensemble statistics
    # ax2 = ax1.twinx()

    # for col in risk_metric_cols:
    #     ax2.plot(
    #         df_plot["Datetime UTC"],
    #         df_plot[col],
    #         label=col,
    #         linestyle="--",
    #         linewidth=1.2,
    #         alpha=0.8,
    #     )

    # ax2.set_ylabel(f"{MODEL} GHI ensemble statistics CH")

    # # Combined legend
    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(
    #     lines_1 + lines_2,
    #     labels_1 + labels_2,
    #     loc="upper left",
    #     fontsize=9,
    #     ncol=2,
    # )

    # plt.title(
    #     f"{MODEL}: Abs.GHI forecast error and statistics from "
    #     f"{START_DATE} to {END_DATE}"
    # )

    # plt.tight_layout()
    # plt.show()

    # =========================
    # Hourly + Daily forecast-risk score analysis
    # =========================

    # Keep hourly version untouched
    df_hourly = df_plot.copy()

    # Already daylight-filtered before fitting
    df_hourly_source = df_hourly.copy()

    # ============================================================
    # 1) Hourly / all timestamp performance
    # ============================================================
    hourly_error_cols = metric_cols.copy()
    hourly_risk_cols = risk_metric_cols.copy()

    hourly_performance_table, hourly_rank_rmse_table = plot_risk_metric_performance(
        df_source=df_hourly_source,
        error_cols=hourly_error_cols,
        risk_cols=hourly_risk_cols,
        error_label_prefix="Hourly abs. forecast error",
        title_prefix="Hourly",
        model=MODEL,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # ============================================================
    # 2) Daily performance
    # ============================================================
    df_daily_source = df_hourly_source.copy()
    df_daily_source["Date"] = df_daily_source["Datetime UTC"].dt.floor("D")

    # =========================
    # Daily summed absolute errors
    # =========================
    daily_error_cols = []

    for metric_col in metric_cols:
        daily_abs_error_col = f"{metric_col} daily abs error"
        df_daily_source[daily_abs_error_col] = df_daily_source[metric_col].abs()
        daily_error_cols.append(daily_abs_error_col)

    # =========================
    # Daily summed risk metrics
    # =========================
    daily_risk_cols = []

    for rad_col in risk_metric_cols:
        daily_risk_col = f"{rad_col} daily score"
        df_daily_source[daily_risk_col] = df_daily_source[rad_col].abs()
        daily_risk_cols.append(daily_risk_col)

    # =========================
    # Aggregate to daily sums
    # =========================
    daily_agg_cols = daily_error_cols + daily_risk_cols

    df_daily = df_daily_source.groupby("Date", as_index=False)[daily_agg_cols].sum(min_count=1)

    # Drop days without useful data
    df_daily = df_daily.dropna(subset=daily_agg_cols, how="all").copy()

    # =========================
    # Daily heatmap + scatterplots
    # =========================
    daily_performance_table, daily_rank_rmse_table = plot_risk_metric_performance(
        df_source=df_daily,
        error_cols=daily_error_cols,
        risk_cols=daily_risk_cols,
        error_label_prefix="Daily summed abs. forecast error",
        title_prefix="Daily",
        model=MODEL,
        start_date=START_DATE,
        end_date=END_DATE,
    )
