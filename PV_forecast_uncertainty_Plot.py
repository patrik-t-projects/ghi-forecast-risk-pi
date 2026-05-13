import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from PV_forecast_uncertainty_functions import choose_f0, limits_for_f0
from config import DATA_DIR, data_path


# ==========================
# Time window definition
# ==========================
START_DATE = pd.Timestamp("2025-03-29 00:00:00", tz="UTC")
END_DATE   = pd.Timestamp("2025-03-29 23:59:59", tz="UTC")
date_label = "year_powerfit"
bplot_year = 0

GHI_threshold = 30

# =====================
# Plot configuration
# =====================
FIGSIZE = (14, 8)
FIGSIZE_scatter = (16, 8)
FONTSIZE = 20
FIG_POSITION = (2200, 100)  # screen position (x, y) in pixels


folder = str(DATA_DIR)


elcom_colors = {
    "dark_blue": (75/255, 96/255, 165/255),
    "light_blue": (146/255, 162/255, 206/255),
    "green": (145/255, 195/255, 74/255),
    "yellow": (255/255, 217/255, 102/255),
    "orange": (249/255, 170/255, 0/255),
    "grey": (191/255, 191/255, 191/255),
    "lila": (150/255, 120/255, 200/255),
    "red": (192/255, 0/255, 0/255),
}

elcom_color_text = (55/255, 55/255, 55/255)


plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE + 2,
    "axes.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE - 1,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
})


# Load CSV
print("Loading saved csv...")
metrics_out_path = os.path.join(folder, f"station_metrics_CH_interp_{date_label}.csv")
df_station_metrics = pd.read_csv(metrics_out_path, sep=";")
df_station_metrics["Datetime"] = pd.to_datetime(df_station_metrics["Datetime"], utc=True)
df_station_metrics = df_station_metrics[(df_station_metrics["Datetime"] >= START_DATE) & (df_station_metrics["Datetime"] <= END_DATE)]

rad_out_path = os.path.join(folder, f"station_radiation_CH_interp_{date_label}.csv")
df_station_radiation = pd.read_csv(rad_out_path, sep=";")
df_station_radiation["Datetime"] = pd.to_datetime(df_station_radiation["Datetime"], utc=True)
df_station_radiation = df_station_radiation[(df_station_radiation["Datetime"] >= START_DATE) & (df_station_radiation["Datetime"] <= END_DATE)]
ghi_series = df_station_radiation["MeteoSwiss stations"]
ghi_mask = ghi_series > GHI_threshold

# Load correlation per day file
out_path = os.path.join(folder, f"daily_correlation_CH_{date_label}.csv")
df_raw = pd.read_csv(out_path, sep=";")
df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"], utc=True)
df_daily_corr = df_raw[(df_raw["Datetime"] >= START_DATE) & (df_raw["Datetime"] <= END_DATE)]


# Load CAB data for imbalances
filepath = data_path("polished_data_SwissGrid_CAB_15min.csv")
df_raw = pd.read_csv(filepath, sep=";")
df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"])
df_CAB = df_raw[["Datetime", "Total system imbalance MW", "AE-Price long EUR/MWh", "AE-Price short EUR/MWh"]]
df_CAB = df_CAB.set_index('Datetime')
df_CAB = df_CAB.sort_index()
df_CAB.index = df_CAB.index.tz_localize('Europe/Zurich', ambiguous=False, nonexistent='shift_forward').tz_convert('UTC')
df_CAB = df_CAB[(df_CAB.index >= START_DATE) & (df_CAB.index <= END_DATE)]

print("Finished loading csv files.")


metric_cols = [
               "MeteoSwiss - ICON_d0",
               "MeteoSwiss - ICON_d1",
               "MeteoSwiss - ICON_d2",
               "MeteoSwiss - ICON_d3",
               # "Ensemble spread"
               ]
metric_color_map = {
        "MeteoSwiss - ICON_d0": elcom_colors["green"],
        "MeteoSwiss - ICON_d1": elcom_colors["orange"],
        "MeteoSwiss - ICON_d2": elcom_colors["red"],
        "MeteoSwiss - ICON_d3": elcom_colors["lila"],
    }

rad_cols = [
            "MeteoSwiss stations",
            "ICON prev day 0",
            "ICON prev day 1",
            "ICON prev day 2",
            "ICON prev day 3",
            # "ICON1 prev day 0",
            # "ICON1 prev day 1",
            # "ICON1 prev day 2",
            # "ICON1 prev day 3",
            # "Era5 reanalysed",
            # "Era5 ensemble mean",
            "Photovoltaik"
            ]
rad_color_map = {
        "MeteoSwiss stations": elcom_colors["dark_blue"],
        "ICON prev day 0": elcom_colors["green"],
        "ICON prev day 1": elcom_colors["orange"],
        "ICON prev day 2": elcom_colors["red"],
        "ICON prev day 3": elcom_colors["lila"],
        "Photovoltaik": elcom_colors["yellow"],
    }

power_cols = [
              # "MeteoSwiss - ICON_d0 power_ratio",
              # "MeteoSwiss - ICON_d1 power_ratio",
              # "MeteoSwiss - ICON_d2 power_ratio",
              # "MeteoSwiss - ICON_d3 power_ratio",
              "MeteoSwiss - ICON_d0 power_fit",
              "MeteoSwiss - ICON_d1 power_fit",
              "MeteoSwiss - ICON_d2 power_fit",
              "MeteoSwiss - ICON_d3 power_fit",
              ]

power_color_map = {
        "MeteoSwiss - ICON_d0 power_ratio": elcom_colors["green"],
        "MeteoSwiss - ICON_d1 power_ratio": elcom_colors["orange"],
        "MeteoSwiss - ICON_d2 power_ratio": elcom_colors["red"],
        "MeteoSwiss - ICON_d3 power_ratio": elcom_colors["lila"],
        "MeteoSwiss - ICON_d0 power_fit": elcom_colors["green"],
        "MeteoSwiss - ICON_d1 power_fit": elcom_colors["orange"],
        "MeteoSwiss - ICON_d2 power_fit": elcom_colors["red"],
        "MeteoSwiss - ICON_d3 power_fit": elcom_colors["lila"],
    }

# rename_map = {
#     "MeteoSwiss - ICON_d0": "Intraday Prognose",
#     "MeteoSwiss - ICON_d1": "D-1 Prognose",
#     "MeteoSwiss - ICON_d2": "D-2 Prognose",
#     "MeteoSwiss - ICON_d3": "D-3 Prognose",
# }
rename_map = {
    "MeteoSwiss - ICON_d0": "Intraday forecast",
    "MeteoSwiss - ICON_d1": "D-1 forecast",
    "MeteoSwiss - ICON_d2": "D-2 forecast",
    "MeteoSwiss - ICON_d3": "D-3 forecast",
}


if bplot_year != 1:

    # ====================================
    # Plot weighted radiation data
    # ====================================
    fig = plt.figure(figsize=FIGSIZE)
    x, y = FIG_POSITION
    manager = fig.canvas.manager
    window = manager.window
    window.move(x, y)

    ax_left = plt.gca()

    # Compute global min/max across all radiation columns
    y_min = df_station_radiation[rad_cols[:-1]].min().min()
    y_max = df_station_radiation[rad_cols[:-1]].max().max()

    for col in rad_cols[:-1]:
        ax_left.plot(df_station_radiation["Datetime"], df_station_radiation[col], label=col, color=rad_color_map.get(col), linewidth=2)

    ax_left.set_xlabel("Time [UTC]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.set_xlim(START_DATE, END_DATE)
    ax_left.set_ylim(min(0, 1.1 * y_min), 1.1 * y_max)
    ax_left.set_ylabel("Power-weighted GHI [W/m^2]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.grid(True, color=elcom_color_text)
    # ax_left.set_title("Actual vs. Forecasted GHI (CH-wide & power-weighted)", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.set_title("Actual power-weighted GHI (CH-wide)", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_left.tick_params(axis="both", colors=elcom_color_text)

    # # ---- Right y-axis: System imbalance as bar plot ----
    # ax_right = ax_left.twinx()
    # bar_width = pd.Timedelta("12min")
    # ax_right.bar(df_CAB.index, df_CAB["Total system imbalance MW"], width=bar_width, color="grey", alpha=0.35, label="Unausgeglichenheit")
    # ax_right.set_ylabel("RZ-Unausgeglichenheit [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    # ax_right.tick_params(axis="y", color=elcom_color_text)

    # Photovoltaik
    ax_right = ax_left.twinx()
    ax_right.plot(df_station_radiation["Datetime"], df_station_radiation["Photovoltaik"], label="PV-produktion SEC", color=rad_color_map.get("Photovoltaik"))
    # ax_right.plot(power_res.index, power_res, label="PV-produktion SEC", color=rad_color_map.get("Photovoltaik"))
    ax_right.set_ylabel("PV-production CH [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_right.tick_params(axis="y", color=elcom_color_text)
    ax_right.set_ylim(0, 1.1 * df_station_radiation["Photovoltaik"].max())

    # ---- Spines ----
    for spine in ax_left.spines.values():
        spine.set_color(elcom_color_text)
    for spine in ax_right.spines.values():
        spine.set_color(elcom_color_text)

    # ---- Combined legend ----
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    legend = ax_left.legend(lines_left + lines_right, labels_left + labels_right, fontsize=FONTSIZE, loc="upper right", frameon=True)
    for text in legend.get_texts():
        text.set_color(elcom_color_text)
    legend.get_frame().set_edgecolor(elcom_color_text)
    # plt.legend(fontsize=FONTSIZE, labelcolor=elcom_color_text)
    plt.tight_layout()
    plt.show()


    # ====================================
    # Print daily correlation values
    # ====================================

    df_corr_table = df_daily_corr.reset_index()
    print(df_corr_table.to_string(index=False))


    # ====================================
    # Plot metrics data GHI
    # ====================================

    fig = plt.figure(figsize=FIGSIZE)
    x, y = FIG_POSITION
    manager = fig.canvas.manager
    window = manager.window
    window.move(x, y)

    ax_left = plt.gca()

    # ---- Axis limits  ----
    dLmin = df_station_metrics[metric_cols].min().min()
    dLmax = df_station_metrics[metric_cols].max().max()

    dRmin = df_CAB["Total system imbalance MW"].min()
    dRmax = df_CAB["Total system imbalance MW"].max()

    f0 = choose_f0(dLmin, dLmax, dRmin, dRmax, marginL=1.05, marginR=1.05, wL=0.8, wR=1.2)

    yL_min, yL_max, _ = limits_for_f0(dLmin, dLmax, f0, margin=1.05)
    yR_min, yR_max, _ = limits_for_f0(dRmin, dRmax, f0, margin=1.05)

    for col in metric_cols:
        plt.plot(df_station_metrics["Datetime"], df_station_metrics[col], label=rename_map.get(col), color=metric_color_map.get(col))

    ax_left.set_xlabel("Time [UTC]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.set_xlim(START_DATE, END_DATE)
    ax_left.set_ylabel("GHI-forecast error [W/m^2]", fontsize=FONTSIZE, color=elcom_color_text)
    # ax_left.set_ylabel("GHI Prognose Abweichung [W/m^2]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.grid(True)
    ax_left.set_ylim(yL_min, yL_max)
    ax_left.set_title("GHI forecast uncertainty (CH-wide & power-weighted)", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_left.tick_params(axis="both", colors=elcom_color_text)

    # # ---- Right y-axis: Imbalances ----
    ax_right = ax_left.twinx()
    bar_width = pd.Timedelta("12min")
    ax_right.set_ylim(yR_min, yR_max)

    ax_right.bar(df_CAB.index, df_CAB["Total system imbalance MW"], color="grey", width=bar_width, alpha=0.35, label="Imbalance")
    # ax_right.bar(df_CAB.index, df_CAB["Total system imbalance MW"], color="grey", width=bar_width, alpha=0.35, label="Unausgeglichenheit")
    ax_right.set_ylabel("RZ-Imbalance [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    # ax_right.set_ylabel("RZ-Unausgeglichenheit [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_right.tick_params(axis="y", colors=elcom_color_text)

    # ---- Spines ----
    for spine in ax_left.spines.values():
        spine.set_color(elcom_color_text)
    for spine in ax_right.spines.values():
        spine.set_color(elcom_color_text)

    # ---- Combined legend ----
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    legend = ax_left.legend(lines_left + lines_right, labels_left + labels_right, fontsize=FONTSIZE, loc="upper right", frameon=True)
    for text in legend.get_texts():
        text.set_color(elcom_color_text)
    legend.get_frame().set_edgecolor(elcom_color_text)
    plt.tight_layout()
    plt.show()


    # ====================================
    # Plot metrics data Power
    # ====================================

    fig = plt.figure(figsize=FIGSIZE)
    x, y = FIG_POSITION
    manager = fig.canvas.manager
    window = manager.window
    window.move(x, y)

    ax_left = plt.gca()

    # Left-axis metrics
    y_min_left = df_station_metrics[power_cols].min().min()
    y_max_left = df_station_metrics[power_cols].max().max()
    # Right-axis metric
    y_min_right = df_CAB["Total system imbalance MW"].min()
    y_max_right = df_CAB["Total system imbalance MW"].max()
    # Global extrema
    y_min = min(y_min_left, y_min_right)
    y_max = max(y_max_left, y_max_right)

    for col in power_cols:
        linestyle = "-" if "power_ratio" in col else "-"
        marker = "o" if "power_ratio" in col else "x"
        plt.plot(df_station_metrics["Datetime"], df_station_metrics[col], color=power_color_map.get(col), linestyle=linestyle, markersize=4, label=rename_map.get(col.replace(" power_ratio", "")))

    # ax_left.set_xlabel("Time [UTC]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.set_xlabel("Zeit [UTC]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.set_xlim(START_DATE, END_DATE)
    ax_left.set_ylim(min(0, 1.1 * y_min), 1.1 * y_max)
    # ax_left.set_ylabel("Estimated power error [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.set_ylabel("PV-Prognose Abweichung [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.grid(True)
    # ax_left.set_title("PV-Power forecast uncertainty (CH-wide & power-weighted)", fontsize=FONTSIZE, color=elcom_color_text)
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_left.tick_params(axis="both", colors=elcom_color_text)

    # # ---- Right y-axis: Imbalances ----
    ax_right = ax_left.twinx()
    bar_width = pd.Timedelta("12min")
    # ax_right.bar(df_CAB.index, df_CAB["Total system imbalance MW"], color="grey", width=bar_width, alpha=0.35, label="Imbalance")
    ax_right.bar(df_CAB.index, df_CAB["Total system imbalance MW"], color="grey", width=bar_width, alpha=0.35, label="RZ-Unausgeglichenheit")
    # ax_right.set_ylabel("RZ-Imbalance [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_right.set_ylabel("RZ-Unausgeglichenheit [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    ax_right.tick_params(axis="y", colors=elcom_color_text)
    ax_right.set_ylim(min(0, 1.1 * y_min), 1.1 * y_max)

    # ---- Spines ----
    for spine in ax_left.spines.values():
        spine.set_color(elcom_color_text)
    for spine in ax_right.spines.values():
        spine.set_color(elcom_color_text)

    # ---- Combined legend ----
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    legend = ax_left.legend(lines_left + lines_right, labels_left + labels_right, fontsize=FONTSIZE, loc="best", frameon=True)
    for text in legend.get_texts():
        text.set_color(elcom_color_text)
    legend.get_frame().set_edgecolor(elcom_color_text)
    plt.tight_layout()
    plt.show()


    # ====================================
    # Scatterplot GHI - subplots (1x4)
    # ====================================

    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE_scatter, sharex=True, sharey=True)
    x_pos, y_pos = FIG_POSITION
    manager = fig.canvas.manager
    window = manager.window
    window.move(x_pos, y_pos)

    metric_title_map = {
        "MeteoSwiss - ICON_d0": "H-3",
        "MeteoSwiss - ICON_d1": "D-1",
        "MeteoSwiss - ICON_d2": "D-2",
        "MeteoSwiss - ICON_d3": "D-3",
    }

    x_base = df_CAB["Total system imbalance MW"]
    if date_label != "year":
        x_base = x_base[:-1]

    df_station_metrics = df_station_metrics.set_index("Datetime")
    df_station_radiation = df_station_radiation.set_index("Datetime")

    for ax, metric in zip(axes, metric_cols):
        y_s = df_station_metrics[metric]
        g_s = df_station_radiation["MeteoSwiss stations"]

        xy = pd.concat([x_base, y_s, g_s], axis=1, join="inner")
        xy.columns = ["imb", "y", "ghi"]

        mask = np.isfinite(xy["imb"]) & np.isfinite(xy["y"]) & (xy["ghi"] > GHI_threshold)
        xy = xy[mask]

        x = xy["imb"].values
        y = xy["y"].values
        # Scatter
        ax.scatter(x, y, color=metric_color_map[metric], alpha=0.7, label=metric, s=10)

        # Linear fit y = a*x + b
        a, b = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = a * x_fit + b

        ax.plot(x_fit, y_fit, color=metric_color_map[metric], linewidth=2)

        ax.set_title(metric_title_map.get(metric, metric), fontsize=FONTSIZE, color=elcom_color_text)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=FONTSIZE, colors=elcom_color_text)

        # --- Annotation box (top-left): Weighted Pearson GHI + Permutation p-value ---
        row = df_corr_table[df_corr_table["Metric"] == metric].iloc[0]
        pear = row["Weighted Pearson GHI"]
        pval = row["Permutation p-value"]

        txt = (
            f"Weighted Pearson: {pear:.2f}\n"
            f"p-value: {pval:.2g}"
        )

        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=FONTSIZE-2,
            color=elcom_color_text,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
        )
    # End loop

    # Shared labels
    axes[0].set_ylabel("GHI Forecast Error [W/m^2]", fontsize=FONTSIZE, color=elcom_color_text)
    for ax in axes:
        ax.set_xlabel("RZ Imbalance [MW]", fontsize=FONTSIZE, color=elcom_color_text)

    start_str = START_DATE.strftime("%d.%m %H:%M")
    end_str = END_DATE.strftime("%d.%m %H:%M")
    fig.suptitle(f"Correlation of GHI forecast error & Imbalances ({start_str} - {end_str}) \n (GHI > {GHI_threshold} W/m^2)", fontsize=FONTSIZE, color=elcom_color_text, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98)
    plt.show()


    # ====================================
    # Scatterplot Power - subplots (1x4)
    # ====================================

    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE_scatter, sharex=True, sharey=True)
    x_pos, y_pos = FIG_POSITION
    manager = fig.canvas.manager
    window = manager.window
    window.move(x_pos, y_pos)

    metric_title_map = {
        "MeteoSwiss - ICON_d0 power_ratio": "H-3",
        "MeteoSwiss - ICON_d1 power_ratio": "D-1",
        "MeteoSwiss - ICON_d2 power_ratio": "D-2",
        "MeteoSwiss - ICON_d3 power_ratio": "D-3",
        # "MeteoSwiss - ICON_d0 power_fit": "H-3",
        # "MeteoSwiss - ICON_d1 power_fit": "D-1",
        # "MeteoSwiss - ICON_d2 power_fit": "D-2",
        # "MeteoSwiss - ICON_d3 power_fit": "D-3",
    }

    x_base = df_CAB["Total system imbalance MW"]
    if date_label != "year":
        x_base = x_base[:-1]

    for ax, metric in zip(axes, power_cols):
        y_s = df_station_metrics[metric]
        g_s = df_station_radiation["MeteoSwiss stations"]

        xy = pd.concat([x_base, y_s, g_s], axis=1, join="inner")
        xy.columns = ["imb", "y", "ghi"]

        mask = np.isfinite(xy["imb"]) & np.isfinite(xy["y"]) & (xy["ghi"] > GHI_threshold)
        xy = xy[mask]

        x = xy["imb"].values
        y = xy["y"].values

        # Scatter
        ax.scatter(x, y, color=power_color_map[metric], alpha=0.7, label=metric, s=10)

        # Linear fit y = a*x + b
        a, b = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = a * x_fit + b

        ax.plot(x_fit, y_fit, color=power_color_map[metric], linewidth=2)

        ax.set_title(metric_title_map.get(metric, metric), fontsize=FONTSIZE, color=elcom_color_text)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=FONTSIZE, colors=elcom_color_text)

        # --- Annotation box (top-left): Weighted Pearson GHI + Permutation p-value ---
        base_metric = metric.replace(" power_ratio", "")
        row = df_corr_table[df_corr_table["Metric"] == base_metric].iloc[0]
        driver_contr = row["Driver contribution Power ratio"]
        P_expl = row["P_expl Power ratio"]
        RMSE = row["RMSE Power ratio"]
        pval = row["Permutation p-value"]
        driver_contr_base = row["Driver contribution Power ratio_base"]
        P_expl_base = row["P_expl Power ratio_base"]
        RMSE_base = row["RMSE Power ratio_base"]

        txt = (
            f"Driver contr.: {driver_contr:.2f}\n"
            # f"Driver contr. base: {driver_contr_base:.2f}\n"
            f"P_expl: {P_expl:.2f}\n"
            # f"P_expl base: {P_expl_base:.2f}\n"
            # f"RMSE: {RMSE:.2f}\n"
            # f"RMSE base: {RMSE_base:.2f}\n"
            f"p-value: {pval:.2f}\n"
        )

        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=FONTSIZE-2,
            color=elcom_color_text,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
        )

    # Shared labels
    axes[0].set_ylabel("PV-Prognose Abweichung [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    for ax in axes:
        ax.set_xlabel("RZ-Unausgeglichenheit [MW]", fontsize=FONTSIZE, color=elcom_color_text)

    start_str = START_DATE.strftime("%d.%m %H:%M")
    end_str = END_DATE.strftime("%d.%m %H:%M")
    fig.suptitle(f"Correlation of PV-Power forecast error & Imbalances ({start_str} - {end_str}) \n (GHI > {GHI_threshold} W/m^2)", fontsize=FONTSIZE, color=elcom_color_text, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98)
    plt.show()


elif bplot_year == 1:

    # # Scatterplot all year
    #
    # # ====================================
    # # Scatterplot GHI - subplots (1x4)
    # # ====================================
    # fig, axes = plt.subplots(1, 4, figsize=FIGSIZE_scatter, sharex=True, sharey=True)
    # x_pos, y_pos = FIG_POSITION
    # manager = fig.canvas.manager
    # window = manager.window
    # window.move(x_pos, y_pos)
    #
    # metric_title_map = {
    #     "MeteoSwiss - ICON_d0": "H-3",
    #     "MeteoSwiss - ICON_d1": "D-1",
    #     "MeteoSwiss - ICON_d2": "D-2",
    #     "MeteoSwiss - ICON_d3": "D-3",
    # }
    #
    # x_base = df_CAB["Total system imbalance MW"]
    # x_base = x_base[~x_base.index.duplicated(keep="first")]
    #
    # df_station_radiation = df_station_radiation.set_index("Datetime")
    # df_station_metrics = df_station_metrics.set_index("Datetime")
    # df_station_metrics = df_station_metrics[~df_station_metrics.index.duplicated(keep="first")]
    #
    # for ax, metric in zip(axes, metric_cols):
    #     y_df = df_station_metrics[metric]
    #     g_s = df_station_radiation["MeteoSwiss stations"]
    #
    #     xy = pd.concat([x_base, y_df, g_s], axis=1, join="inner").dropna()
    #     xy.columns = ["imb", "y", "ghi"]
    #     xy = xy[xy["ghi"] > GHI_threshold]
    #
    #     x = xy["imb"].values
    #     y = xy["y"].values
    #
    #     # Scatter
    #     ax.scatter(x, y, color=metric_color_map[metric], alpha=0.7, label=metric, s=1)
    #
    #     # Linear fit y = a*x + b
    #     a, b = np.polyfit(x, y, 1)
    #     x_fit = np.linspace(x.min(), x.max(), 200)
    #     y_fit = a * x_fit + b
    #
    #     ax.plot(x_fit, y_fit, color=metric_color_map[metric], linewidth=2)
    #
    #     ax.set_title(metric_title_map.get(metric, metric), fontsize=FONTSIZE, color=elcom_color_text)
    #     ax.grid(True)
    #     ax.tick_params(axis="both", labelsize=FONTSIZE, colors=elcom_color_text)
    #
    #     # --- Annotation box (top-left): Weighted Pearson GHI + Permutation p-value ---
    #     df_corr_agg = df_daily_corr.groupby("Metric", as_index=False).agg({"Weighted Pearson GHI": "median", "Permutation p-value": "median"})
    #     row = df_corr_agg[df_corr_agg["Metric"] == metric].iloc[0]
    #     pear = row["Weighted Pearson GHI"]
    #     pval = row["Permutation p-value"]
    #
    #     txt = (
    #         f"Weighted Pearson: {pear:.2f}\n"
    #         f"p-value: {pval:.2g}"
    #     )
    #
    #     ax.text(
    #         0.02, 0.98, txt,
    #         transform=ax.transAxes,
    #         va="top", ha="left",
    #         fontsize=FONTSIZE-2,
    #         color=elcom_color_text,
    #         bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    #     )
    # # End loop
    #
    # # Shared labels
    # axes[0].set_ylabel("GHI Forecast Error [W/m^2]", fontsize=FONTSIZE, color=elcom_color_text)
    # for ax in axes:
    #     ax.set_xlabel("RZ Imbalance [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    #
    # start_str = START_DATE.strftime("%d.%m %H:%M")
    # end_str = END_DATE.strftime("%d.%m %H:%M")
    # fig.suptitle(f"Correlation of GHI forecast error & Imbalances ({start_str} - {end_str}) \n (GHI > {GHI_threshold} W/m^2)", fontsize=FONTSIZE, color=elcom_color_text, y=0.96)
    # plt.tight_layout(rect=[0, 0, 1, 0.99])
    # fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98)
    # plt.show()
    #
    #
    # # ====================================
    # # Scatterplot Power - subplots (1x4)
    # # ====================================
    # fig, axes = plt.subplots(1, 4, figsize=FIGSIZE_scatter, sharex=True, sharey=True)
    # x_pos, y_pos = FIG_POSITION
    # manager = fig.canvas.manager
    # window = manager.window
    # window.move(x_pos, y_pos)
    #
    # metric_title_map = {
    #     "MeteoSwiss - ICON_d0 power_ratio": "H-3",
    #     "MeteoSwiss - ICON_d1 power_ratio": "D-1",
    #     "MeteoSwiss - ICON_d2 power_ratio": "D-2",
    #     "MeteoSwiss - ICON_d3 power_ratio": "D-3",
    # }
    #
    # x_base = df_CAB["Total system imbalance MW"]
    # x_base = x_base[~x_base.index.duplicated(keep="first")]
    #
    # for ax, metric in zip(axes, power_cols):
    #     y_df = df_station_metrics[metric]
    #     g_s = df_station_radiation["MeteoSwiss stations"]
    #
    #     xy = pd.concat([x_base, y_df, g_s], axis=1, join="inner").dropna()
    #     xy.columns = ["imb", "y", "ghi"]
    #     xy = xy[xy["ghi"] > GHI_threshold]
    #
    #     x = xy["imb"].values
    #     y = xy["y"].values
    #
    #     # Scatter
    #     ax.scatter(x, y, color=power_color_map[metric], alpha=0.7, label=metric, s=1)
    #
    #     # Linear fit y = a*x + b
    #     a, b = np.polyfit(x, y, 1)
    #     x_fit = np.linspace(x.min(), x.max(), 200)
    #     y_fit = a * x_fit + b
    #
    #     ax.plot(x_fit, y_fit, color=power_color_map[metric], linewidth=2)
    #
    #     ax.set_title(metric_title_map.get(metric, metric), fontsize=FONTSIZE, color=elcom_color_text)
    #     ax.grid(True)
    #     ax.tick_params(axis="both", labelsize=FONTSIZE, colors=elcom_color_text)
    #
    #     # --- Annotation box (top-left): Weighted Pearson GHI + Permutation p-value ---
    #     base_metric = metric.replace(" power_ratio", "")
    #     df_corr_agg = df_daily_corr.groupby("Metric", as_index=False).agg({"Weighted Pearson GHI": "median", "Permutation p-value": "median"})
    #     row = df_corr_agg[df_corr_agg["Metric"] == base_metric].iloc[0]
    #     pear = row["Weighted Pearson GHI"]
    #     pval = row["Permutation p-value"]
    #
    #     txt = (
    #         f"Weighted Pearson: {pear:.2f}\n"
    #         f"p-value: {pval:.2g}"
    #     )
    #
    #     ax.text(
    #         0.02, 0.98, txt,
    #         transform=ax.transAxes,
    #         va="top", ha="left",
    #         fontsize=FONTSIZE - 2,
    #         color=elcom_color_text,
    #         bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    #     )
    # # End loop
    #
    # # Shared labels
    # axes[0].set_ylabel("GHI Forecast Error [W/m^2]", fontsize=FONTSIZE, color=elcom_color_text)
    # for ax in axes:
    #     ax.set_xlabel("RZ Imbalance [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    #
    # start_str = START_DATE.strftime("%d.%m %H:%M")
    # end_str = END_DATE.strftime("%d.%m %H:%M")
    # fig.suptitle(f"Correlation of GHI forecast error & Imbalances ({start_str} - {end_str}) \n (GHI > {GHI_threshold} W/m^2)", fontsize=FONTSIZE, color=elcom_color_text, y=0.96)
    # plt.tight_layout(rect=[0, 0, 1, 0.99])
    # fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98)
    # plt.show()


    # ==================================================
    # Full year scatter plot of metric values
    # ==================================================

    correlation_col = "Driver contribution Power ratio"
    # correlation_col = "P_expl Power ratio"

    CORR_THRESHOLD = 0.85
    stat_col = "Permutation p-value"
    Stat_VAL_THRESHOLD = 0.1

    metric_title_map = {
        "MeteoSwiss - ICON_d0": "H-3",
        "MeteoSwiss - ICON_d1": "D-1",
        "MeteoSwiss - ICON_d2": "D-2",
        "MeteoSwiss - ICON_d3": "D-3",
    }

    # ==================================================
    # Boxplot GHI correlation distributions per weekday
    # ==================================================
    import seaborn as sns

    fig = plt.figure(figsize=FIGSIZE)
    x, y = FIG_POSITION
    manager = fig.canvas.manager
    window = manager.window
    window.move(x, y)
    ax = plt.gca()

    # -----------------------------
    # Prepare dataframe
    # -----------------------------
    df_daily_corr["Datetime"] = pd.to_datetime(df_daily_corr["Datetime"])
    df_daily_corr["weekday"] = df_daily_corr["Datetime"].dt.day_name()

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    df_daily_corr["weekday"] = pd.Categorical(df_daily_corr["weekday"], categories=weekday_order, ordered=True)

    # --------------------------------
    # Apply filtering
    # --------------------------------
    df_filtered = df_daily_corr[(df_daily_corr[correlation_col] > CORR_THRESHOLD) & (df_daily_corr[stat_col] < Stat_VAL_THRESHOLD)].copy()
    # df_filtered[correlation_col] = df_filtered[correlation_col]**2

    # -----------------------------
    # Boxplot
    # -----------------------------
    metric_order = [
        "MeteoSwiss - ICON_d0",
        "MeteoSwiss - ICON_d1",
        "MeteoSwiss - ICON_d2",
        "MeteoSwiss - ICON_d3",
    ]

    palette = {
        metric: metric_color_map.get(metric)
        for metric in metric_order
    }

    sns.boxplot(
        data=df_filtered,
        x="weekday",
        y=correlation_col,
        hue="Metric",
        order=weekday_order,
        hue_order=metric_order,
        palette=palette,
        width=0.7,
        fliersize=2,
        ax=ax
    )

    # -----------------------------
    # Axis formatting
    # -----------------------------
    ax.set_xticklabels(weekday_order, fontsize=FONTSIZE, color=elcom_color_text)
    ax.set_ylabel(correlation_col, fontsize=FONTSIZE, color=elcom_color_text)
    ax.set_xlabel("Day of week", labelpad=18, fontsize=FONTSIZE, color=elcom_color_text)
    ax.set_title(f"Distribution of {correlation_col} by weekday\n (Corr > {CORR_THRESHOLD} | p_val > {Stat_VAL_THRESHOLD} | GHI > {GHI_threshold} W/m^2)", fontsize=FONTSIZE, color=elcom_color_text)
    ax.grid(True, axis="y", color=elcom_color_text, alpha=0.3)
    ax.tick_params(axis="both", colors=elcom_color_text)

    # -----------------------------
    # Spines
    # -----------------------------
    for spine in ax.spines.values():
        spine.set_color(elcom_color_text)

    # -----------------------------
    # Legend
    # -----------------------------
    handles, labels = ax.get_legend_handles_labels()
    mapped_labels = [metric_title_map.get(label, label) for label in labels]

    legend = ax.legend(handles, mapped_labels, fontsize=FONTSIZE, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=True)

    for text in legend.get_texts():
        text.set_color(elcom_color_text)
    legend.get_frame().set_edgecolor(elcom_color_text)

    plt.tight_layout()
    plt.show()



    # # ==================================================
    # # Parameters
    # # ==================================================
    # correlation_col = "Driver contribution Power ratio"
    # # correlation_col = "P_expl Power ratio"
    # stat_col = "Permutation p-value"
    #
    # Stat_VAL_THRESHOLD = 0.99
    #
    # thresholds = np.linspace(0.0, 1.0, 11)
    #
    # metric_order = [
    #     "MeteoSwiss - ICON_d0",
    #     "MeteoSwiss - ICON_d1",
    #     "MeteoSwiss - ICON_d2",
    #     "MeteoSwiss - ICON_d3",
    # ]
    #
    # weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    #
    # # ==================================================
    # # Prepare dataframe
    # # ==================================================
    # df_daily_corr["Datetime"] = pd.to_datetime(df_daily_corr["Datetime"])
    # df_daily_corr["weekday"] = df_daily_corr["Datetime"].dt.day_name()
    #
    # df_daily_corr["weekday"] = pd.Categorical(df_daily_corr["weekday"], categories=weekday_order, ordered=True)
    # df_daily_corr[correlation_col] = df_daily_corr[correlation_col]**2
    #
    # # ==================================================
    # # Loop over weekdays -> one figure per weekday
    # # ==================================================
    # for weekday in weekday_order:
    #
    #     df_day = df_daily_corr[df_daily_corr["weekday"] == weekday]
    #     records = []
    #
    #     for metric in metric_order:
    #
    #         df_metric = df_day[df_day["Metric"] == metric]
    #         y_values = []
    #
    #         for thr in thresholds:
    #
    #             df_filtered = df_metric[(df_metric[correlation_col] > thr) & (df_metric[stat_col] < Stat_VAL_THRESHOLD)]
    #
    #             for val in df_filtered[correlation_col]:
    #                 records.append({
    #                     "Threshold": thr,
    #                     "Correlation": val,
    #                     "Metric": metric
    #                 })
    #     # End loop
    #
    #     df_plot = pd.DataFrame(records)
    #
    #     # --------------------------------
    #     # Plot
    #     # --------------------------------
    #     fig, ax = plt.subplots(figsize=FIGSIZE)
    #
    #     sns.boxplot(
    #         data=df_plot,
    #         x="Threshold",
    #         y="Correlation",
    #         hue="Metric",
    #         palette={m: metric_color_map[m] for m in metric_order},
    #         fliersize=2,
    #         ax=ax
    #     )
    #
    #     # -----------------------------
    #     # Formatting
    #     # -----------------------------
    #     ax.set_xlabel("Correlation threshold", fontsize=FONTSIZE, color=elcom_color_text)
    #     ax.set_ylabel(correlation_col, fontsize=FONTSIZE, color=elcom_color_text)
    #     ax.set_title(f"{weekday}: Threshold sweep\n(p-value < {Stat_VAL_THRESHOLD})", fontsize=FONTSIZE, color=elcom_color_text)
    #
    #     ax.grid(True, alpha=0.3)
    #     ax.tick_params(axis="both", colors=elcom_color_text)
    #     ax.set_xticklabels([f"{float(t.get_text()):.2f}" for t in ax.get_xticklabels()], rotation=0)
    #
    #     for spine in ax.spines.values():
    #         spine.set_color(elcom_color_text)
    #
    #     legend = ax.legend(fontsize=FONTSIZE)
    #     for text in legend.get_texts():
    #         text.set_color(elcom_color_text)
    #
    #     plt.tight_layout()
    #     plt.show()



    # # =====================================================
    # # Scatterplot Ensemble Spread vs GHI forecat error
    # # =====================================================
    #
    # fig = plt.figure(figsize=FIGSIZE_scatter)
    # x_pos, y_pos = FIG_POSITION
    # manager = fig.canvas.manager
    # window = manager.window
    # window.move(x_pos, y_pos)
    #
    # ax = plt.gca()
    #
    # # --- Prepare data ---
    # df_station_metrics = df_station_metrics.set_index("Datetime")
    # df_station_radiation = df_station_radiation.set_index("Datetime")
    #
    # x_base = df_station_metrics["MeteoSwiss - ICON_d0"].abs()
    # x_base = x_base[~x_base.index.duplicated(keep="first")]
    # y_s = df_station_metrics["Ensemble spread"]
    # g_s = df_station_radiation["MeteoSwiss stations"]
    #
    # # Align everything
    # xy = pd.concat([x_base, y_s, g_s], axis=1, join="inner")
    # xy.columns = ["imb", "spread", "ghi"]
    #
    # # Apply filters (same logic as before)
    # mask = (np.isfinite(xy["imb"]) & np.isfinite(xy["spread"]) & (xy["ghi"] > GHI_threshold))
    # xy = xy[mask]
    #
    # x = xy["imb"].values
    # y = xy["spread"].values
    #
    # # --- Scatter ---
    # ax.scatter(x, y, color=elcom_colors["dark_blue"], alpha=0.7, s=10)
    #
    # # --- Linear fit ---
    # if len(x) > 2:
    #     a, b = np.polyfit(x, y, 1)
    #     x_fit = np.linspace(x.min(), x.max(), 200)
    #     y_fit = a * x_fit + b
    #     ax.plot(x_fit, y_fit, color=elcom_colors["dark_blue"], linewidth=2)
    #
    # # --- Axis formatting ---
    # ax.set_xlabel("GHI forecast error Intraday [MW]", fontsize=FONTSIZE, color=elcom_color_text)
    # ax.set_ylabel("GHI Ensemble Spread [W/m^2]", fontsize=FONTSIZE, color=elcom_color_text)
    #
    # ax.set_title(f"Correlation of GHI Ensemble Spread & GHI forecast error Intraday\n ({START_DATE.strftime('%d.%m')} - {END_DATE.strftime('%d.%m')} | GHI > {GHI_threshold} W/m^2)", fontsize=FONTSIZE, color=elcom_color_text)
    #
    # pear = np.corrcoef(x, y)[0, 1]
    #
    # ax.text(
    #     0.02, 0.98,
    #     f"Pearson: {pear:.2f}",
    #     transform=ax.transAxes,
    #     va="top", ha="left",
    #     fontsize=FONTSIZE - 2,
    #     color=elcom_color_text,
    #     bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    # )
    #
    # ax.grid(True)
    # ax.tick_params(axis="both", colors=elcom_color_text)
    #
    # # --- Spines ---
    # for spine in ax.spines.values():
    #     spine.set_color(elcom_color_text)
    #
    # plt.tight_layout()
    # plt.show()
