import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PV_forecast_uncertainty_functions import weighted_pearson, explainable_power, driver_contribution, compute_rmse, predict_pv_power_from_ghi
from config import DATA_DIR


# ==========================
# Time window definition
# ==========================
START_DATE = pd.Timestamp(globals().get("START_DATE", "2026-03-21 00:00:00"), tz="UTC")
END_DATE   = pd.Timestamp(globals().get("END_DATE", "2026-05-05 23:59:59"), tz="UTC")
date_label = "year_2026"

GHI_threshold = 30

folder = str(DATA_DIR)


# Geodata from PV plants
GEOCODED_filepath = os.path.join(folder, "ElectricityProductionPlant_geocoded.csv")
print(f"Loading geocoded plants...", flush=True)
df_plants_geocoded = pd.read_csv(GEOCODED_filepath, sep=",", parse_dates=["BeginningOfOperation"])
df_plants_geocoded["BeginningOfOperation"] = (df_plants_geocoded["BeginningOfOperation"].dt.tz_localize("UTC").dt.floor("D"))

# Weathers station radiation data
weather_stations_filepath = os.path.join(folder, "meteoswiss_gre_hourly_2026.csv")
print(f"Loading Weather stations Data...", flush=True)
df_weather_stations_data = pd.read_csv(weather_stations_filepath, sep=";", parse_dates=["Datetime"])
df_weather_stations_data["Datetime"] = pd.to_datetime(df_weather_stations_data["Datetime"], utc=True)

# Historical and Previous Model Runs Data from Open Meteo for Weather stations
open_meteo_filepath = os.path.join(folder, "Data_OpenMeteo_ENSEMBLE_PrevRuns_weather_stations_2026.csv")
print(f"Loading OPEN METEO Data for weather stations...", flush=True)
df_OPEN_METEO_data = pd.read_csv(open_meteo_filepath, sep=";", parse_dates=["Datetime"])
df_OPEN_METEO_data["Datetime"] = (pd.to_datetime(df_OPEN_METEO_data["Datetime"]).dt.tz_localize("Europe/Zurich", nonexistent="shift_forward", ambiguous="infer").dt.tz_convert("UTC"))

# Pre-filter by time window (do this once)
df_weather_tw = df_weather_stations_data[(df_weather_stations_data["Datetime"] >= START_DATE) & (df_weather_stations_data["Datetime"] <= END_DATE)]
df_open_meteo_tw = df_OPEN_METEO_data[(df_OPEN_METEO_data["Datetime"] >= START_DATE) & (df_OPEN_METEO_data["Datetime"] <= END_DATE)]


# =========================================
# Unique list of nearest weather stations
# =========================================
print(
    f"Processing time range: {START_DATE.date()} at {END_DATE.date()} "
    f"({len(pd.date_range(START_DATE, END_DATE, freq='D'))} days, "
    f"{len(pd.date_range(START_DATE, END_DATE, freq='h'))} hours)",
    flush=True
)

station_list = (df_plants_geocoded["Closest weather station"].dropna().unique())

print(f"Found {len(station_list)} unique weather stations")


# ==================================================================
# Prepare geocoded plants and total Power per hour for normalization
# ==================================================================
time_index_full = pd.date_range(start=START_DATE, end=END_DATE, freq="h", tz="UTC")
df_power_out_path = os.path.join(folder, f"power_station_weights_CH_{date_label}.csv")
if os.path.exists(df_power_out_path):
    print(f"Power weights file already exists at loading:\n{df_power_out_path}", flush=True)
    df_power = pd.read_csv(df_power_out_path, sep=";", parse_dates=["Datetime"], index_col=False)
    df_power["Datetime"] = pd.to_datetime(df_power["Datetime"], utc=True)
    missing_times = time_index_full.difference(df_power["Datetime"].unique())
else:
    print("No power weights file found at creating new one", flush=True)
    df_power = pd.DataFrame()
    missing_times = time_index_full

if len(missing_times) > 0:
    print(f"Computing missing power weights from {missing_times.min()} to {missing_times.max()}")

    df_plants_active = df_plants_geocoded[df_plants_geocoded["BeginningOfOperation"] <= END_DATE.floor("D")].copy()
    plants = df_plants_active[["Closest weather station", "BeginningOfOperation", "TotalPower"]].rename(columns={"Closest weather station": "station_abbr"})

    records = []

    for station, group in plants.groupby("station_abbr"):
        for ts in missing_times:
            active_power = group.loc[group["BeginningOfOperation"] <= ts.floor("D"), "TotalPower"].sum()

            records.append({
                "Datetime": ts,
                "station_abbr": station,
                "total_power_station": active_power
            })

    df_power_new = pd.DataFrame(records)

    df_total_new = df_power_new.groupby("Datetime", as_index=False)["total_power_station"].sum().rename(columns={"total_power_station": "total_power_all"})

    df_power_new = df_power_new.merge(df_total_new, on="Datetime", how="left")
    df_power_new["power_weight"] = df_power_new["total_power_station"] / df_power_new["total_power_all"]
    
    df_power = pd.concat([df_power, df_power_new], ignore_index=True)
    df_power = df_power.drop_duplicates(["station_abbr", "Datetime"], keep="last")
    df_power = df_power.sort_values(["station_abbr", "Datetime"])

    df_power.to_csv(df_power_out_path, sep=";", index=False)
    print(f"Updated power weights saved to {df_power_out_path}")
else:
    print("Power weights already cover full time range")
# End check


#
# =========================================
# Loop through weather stations
# =========================================
all_station_metrics = []
all_station_radiation = []
all_station_GHI_10min = []

for i, station in enumerate(station_list, start=1):
    print(f"\n[{i}/{len(station_list)}] Processing station: {station}")

    # Radiation data from MeteoSwiss
    df_station_weather = df_weather_tw[df_weather_tw["station_abbr"] == station].sort_values("Datetime").reset_index(drop=True)
    df_station_open_meteo = df_open_meteo_tw[df_open_meteo_tw["station_abbr"] == station].sort_values("Datetime").reset_index(drop=True)
    df_power_station_filtered = df_power[df_power["station_abbr"] == station].sort_values("Datetime").reset_index(drop=True)

    print(
        f"    Time span: {df_station_weather['Datetime'].min().date()} at "
        f"{df_station_weather['Datetime'].max().date()} "
        f"({df_station_weather['Datetime'].dt.floor('D').nunique()} days)",
        flush=True
    )

    # Before merging make sure weather station GHI values are hourly sampled
    if "gre000z0" in df_weather_tw.columns:
        df_station_weather_10min = df_station_weather.set_index("Datetime").sort_index()
        df_station_weather = df_station_weather_10min["gre000z0"].resample("h", label="right", closed="right").mean().rename("gre000h0").reset_index()


    # Merge weather station GHI with Open METEO forecast data
    df_station = (df_station_open_meteo
                  .merge(df_station_weather, on="Datetime", how="left", suffixes=("_open", "_weather"))
                  .merge(df_power_station_filtered, on="Datetime", how="left")
                 )

    # Check for lost timestamps
    lost_weather = len(df_station_weather) - len(df_station)
    weather_times = set(df_station_weather["Datetime"])
    merged_times = set(df_station["Datetime"])
    if lost_weather > 0:
        print(f"    âš ï¸ Dropped {lost_weather} MeteoSwiss hours (missing data)")

        missing_times = sorted(weather_times - merged_times)
        if missing_times:
            print(f"    âŒ Missing timestamps ({len(missing_times)}):")


    # ----------------------------------
    # Calculate all metrics
    # ----------------------------------

    Bias_GRE_ICON1_d0 = df_station["gre000h0"].values - df_station["shortwave_radiation"].values
    Bias_GRE_ICON1_d1 = df_station["gre000h0"].values - df_station["shortwave_radiation_previous_day1"].values
    Bias_GRE_ICON1_d2 = df_station["gre000h0"].values - df_station["shortwave_radiation_previous_day2"].values
    Bias_GRE_ICON1_d3 = df_station["gre000h0"].values - df_station["shortwave_radiation_previous_day3"].values

    # Get weights and create metric fict to later create dataframe
    weights = df_station["power_weight"].values
    df_metrics_per_station = pd.DataFrame(
        {
            "Datetime": df_station["Datetime"].values,
            "station_abbr": station,
            "MeteoSwiss - ICON1_d0": Bias_GRE_ICON1_d0 * weights,
            "MeteoSwiss - ICON1_d1": Bias_GRE_ICON1_d1 * weights,
            "MeteoSwiss - ICON1_d2": Bias_GRE_ICON1_d2 * weights,
            "MeteoSwiss - ICON1_d3": Bias_GRE_ICON1_d3 * weights,
        }
    )

    all_station_metrics.append(df_metrics_per_station)

    # ----------------------------------
    # Radiation signals (power-weighted)
    # ----------------------------------
    R_obs = df_station["gre000h0"].values

    R_ICON1_prev0 = df_station["shortwave_radiation"].values
    R_ICON1_prev1 = df_station["shortwave_radiation_previous_day1"].values
    R_ICON1_prev2 = df_station["shortwave_radiation_previous_day2"].values
    R_ICON1_prev3 = df_station["shortwave_radiation_previous_day3"].values

    R_ICON1_mean = df_station["shortwave_radiation_icon_ch1_ens_mean"].values
    R_ICON1_std = df_station["shortwave_radiation_icon_ch1_ens_std"].values
    R_ICON1_min = df_station["shortwave_radiation_icon_ch1_ens_min"].values
    R_ICON1_max = df_station["shortwave_radiation_icon_ch1_ens_max"].values
    R_ICON2_mean = df_station["shortwave_radiation_icon_ch2_ens_mean"].values
    R_ICON2_std = df_station["shortwave_radiation_icon_ch2_ens_std"].values
    R_ICON2_min = df_station["shortwave_radiation_icon_ch2_ens_min"].values
    R_ICON2_max = df_station["shortwave_radiation_icon_ch2_ens_max"].values

    R_ICON1_ens_spread = df_station["shortwave_radiation_spread_meteoswiss_icon_ch1_ensemble_mean"].values
    R_ICON2_ens_spread = df_station["shortwave_radiation_spread_meteoswiss_icon_ch2_ensemble_mean"].values

    C_ICON1_ens_spread = df_station["cloud_cover_spread_meteoswiss_icon_ch1_ensemble_mean"].values
    C_ICON2_ens_spread = df_station["cloud_cover_spread_meteoswiss_icon_ch2_ensemble_mean"].values

    df_rad_per_station = pd.DataFrame(
        {
            "Datetime": df_station["Datetime"].values,
            "station_abbr": station,
            "MeteoSwiss stations": R_obs * weights,
            "ICON1 prev day 0": R_ICON1_prev0 * weights,
            "ICON1 prev day 1": R_ICON1_prev1 * weights,
            "ICON1 prev day 2": R_ICON1_prev2 * weights,
            "ICON1 prev day 3": R_ICON1_prev3 * weights,

            "ICON1 mean": R_ICON1_mean * weights,
            "ICON1 std": R_ICON1_std * weights,
            "ICON1 min": R_ICON1_min* weights,
            "ICON1 max": R_ICON1_max * weights,
            "ICON2 mean": R_ICON2_mean * weights,
            "ICON2 std": R_ICON2_std * weights,
            "ICON2 min": R_ICON2_min* weights,
            "ICON2 max": R_ICON2_max * weights,

            "ICON1 ens spread": R_ICON1_ens_spread * weights,
            "ICON2 ens spread": R_ICON2_ens_spread * weights,

            "ICON1 CC ens spread": C_ICON1_ens_spread * weights,
            "ICON2 CC ens spread": C_ICON2_ens_spread * weights,
        }
    )

    all_station_radiation.append(df_rad_per_station)


    # ==================================================================
    # Make new dataframe just for 10min GHI data
    # ==================================================================
    if "gre000z0" in df_weather_tw.columns:
        df_power_station_filtered = df_power_station_filtered.set_index("Datetime").sort_index()
        df_power_station_10min = df_power_station_filtered.resample("10min").ffill().reset_index()
        df_station_10min = df_station_weather_10min.merge(df_power_station_10min, on="Datetime", how="inner").reset_index()
        R_obs_10min = df_station_10min["gre000z0"].values
        weights_10min = df_station_10min["power_weight"].values

        df_GHI_10min = pd.DataFrame(
            {
                "Datetime": df_station_10min["Datetime"].values,
                "MeteoSwiss stations": R_obs_10min * weights_10min,
            }
        )

        all_station_GHI_10min.append(df_GHI_10min)

# End loop


# Concat df and groupby weather station
df_station_metrics = pd.concat(all_station_metrics, ignore_index=True)
# Save metrics per station as csv
out_path = os.path.join(folder, f"GHI_metrics_per_station_{date_label}.csv")
df_station_metrics.to_csv(out_path, sep=";")
print(f"Saved station radiation to {out_path}")

df_station_metrics_CH = df_station_metrics.groupby("Datetime", as_index=False).sum(numeric_only=True, min_count=1)
df_station_radiation = pd.concat(all_station_radiation, ignore_index=False)
df_station_radiation_CH = df_station_radiation.groupby("Datetime", as_index=False).sum(numeric_only=True, min_count=1)


# ====================================
# Compute daily correlation metrics
# ====================================

# Interpolate to 15min values
df_station_metrics_CH = df_station_metrics_CH.set_index("Datetime").sort_index()
df_station_radiation_CH = df_station_radiation_CH.set_index("Datetime").sort_index()
df_station_metrics_CH_interp = df_station_metrics_CH.resample("15min").interpolate(method="time", limit_area="inside")
df_station_radiation_CH_interp = df_station_radiation_CH.resample("15min").interpolate(method="time", limit_area="inside")
# Make the interpolated series time-zone aware
df_station_radiation_CH_interp.index = df_station_radiation_CH_interp.index.tz_localize("UTC")
df_station_metrics_CH_interp.index = df_station_metrics_CH_interp.index.tz_localize("UTC")
assert df_station_metrics_CH_interp.index.equals(df_station_radiation_CH_interp.index), "Datetime indices do not match!"

# # Get PV production data from Energy Charts
# df_Energy_Chart_data = df_Energy_Chart_data.set_index("Datetime").sort_index()
# df_resample = df_Energy_Chart_data.copy()

# if "gre000z0" in df_weather_tw.columns:
#     df_station_GHI_10min = pd.concat(all_station_GHI_10min, ignore_index=False)
#     df_station_GHI_10min_CH = df_station_GHI_10min.groupby("Datetime", as_index=False).sum(numeric_only=True).set_index("Datetime").sort_index()
#     df_station_GHI_10min_CH_interp = df_station_GHI_10min_CH.resample("1min").ffill().resample("15min", label="right", closed="right").mean()
#     df_Energy_Chart_data_interp = df_Energy_Chart_data.copy()
# else:
#     df_Energy_Chart_data_resample = df_resample["Photovoltaik"].resample("h", label="right", closed="right").mean()
#     df_Energy_Chart_data_interp = df_Energy_Chart_data_resample.resample("15min").interpolate(method="time")

# # Merge PV production data onto "df_station_radiation_CH_interp"
# df_merged = pd.merge(df_station_radiation_CH_interp, df_Energy_Chart_data_interp, on='Datetime', how='left')
# df_station_radiation_CH_interp = df_merged

# Save radiation interpolated files as csv
rad_out_path = os.path.join(folder, f"station_radiation_CH_interp_{date_label}.csv")
df_station_radiation_CH_interp.to_csv(rad_out_path, sep=";")
print(f"Saved station radiation to {rad_out_path}")


# # ==============================================================
# # Calculate the correlations per metric per day
# # ==============================================================

# rows = []

# metrics = [
#     "MeteoSwiss - ICON_d0",
#     "MeteoSwiss - ICON_d1",
#     "MeteoSwiss - ICON_d2",
#     "MeteoSwiss - ICON_d3",
# ]

# # Remove last timestamp for daily loop
# df_station_metrics_CH_interp = df_station_metrics_CH_interp.loc[
#     (df_station_metrics_CH_interp.index >= START_DATE) &
#     (df_station_metrics_CH_interp.index < END_DATE)
# ]


# # Daily loop
# for day, df_day_metrics in df_station_metrics_CH_interp.groupby(df_station_metrics_CH_interp.index.date):

#     # same timestamps for this day
#     print(day)
#     idx = df_day_metrics.index
#     df_day_radiation = df_station_radiation_CH_interp.loc[idx]

#     # Per-day weights on the full day index
#     ghi_day = df_day_radiation["MeteoSwiss stations"].reindex(idx).values
#     pv_day = df_day_radiation["Photovoltaik"].reindex(idx).values
#     imb = df_CAB["Total system imbalance MW"].reindex(idx).values

#     # PV-active mask
#     ghi_mask = np.isfinite(ghi_day) & (ghi_day > GHI_threshold)
#     weights = ghi_day

#     # PV weights for GHI to MW conversion
#     PV_weights_ratio_day = pv_day / np.maximum(ghi_day, GHI_threshold)


#     # ==========================================================
#     # Daily baseline subtraction (non-PV periods)
#     # ==========================================================
#     baseline_mask = (~ghi_mask) & np.isfinite(imb)
#     imb_baseline = np.nanmean(imb[baseline_mask])
#     imb_anomaly = imb - imb_baseline


#     # ==========================================================================
#     # Fit PV-prod to actual GHI, then apply the fit to the forecasted value
#     # ==========================================================================
#     df_fit_day = pd.concat([df_day_radiation["Photovoltaik"].reindex(idx), df_day_radiation["MeteoSwiss stations"]], axis=1).dropna()
#     df_fit_day.columns = ["P_pv", "GHI"]
#     df_fit_day["hour"] = df_fit_day.index.hour
#     # Split into morning / afternoon
#     idx_max = df_fit_day["P_pv"].idxmax()
#     df_morning = df_fit_day[df_fit_day.index < idx_max]
#     df_afternoon = df_fit_day[df_fit_day.index >= idx_max]
#     GHI_fit = df_fit_day["GHI"].values
#     morning_mask = df_fit_day.index <= df_fit_day["P_pv"].idxmax()


#     # # --- logistic function fit ---
#     # from scipy.optimize import curve_fit
#     # def logistic(G, P_max, k, G0):
#     #     sigma = 1 / (1 + np.exp(-k * (G - G0)))
#     #     sigma0 = 1 / (1 + np.exp(-k * (0 - G0)))
#     #     return P_max * (sigma - sigma0) / (1 - sigma0)
#     #
#     # def logistic_derivative(G, P_max, k, G0):
#     #     f = logistic(G, P_max, k, G0)
#     #     return k * f * (1 - f / P_max)
#     #
#     # def fit_logistic(G, P):
#     #     # initial guesses
#     #     P_max0 = np.max(P)
#     #     k0 = 0.01
#     #     G00 = np.median(G)
#     #
#     #     popt, _ = curve_fit(logistic, G, P, p0=[P_max0, k0, G00], maxfev=10000, bounds=([0, 0, 0], [np.inf, 1, 1000]))
#     #     return popt  # (P_max, k, G0)
#     #
#     #
#     # # --- filter data ---
#     # df_fit_day_part = df_fit_day[df_fit_day["GHI"] > 5]
#     # # --- MORNING FIT ---
#     # df_morning_fit = df_morning[df_morning["GHI"] > 5]
#     # G_m = df_morning_fit["GHI"].values
#     # P_m = df_morning_fit["P_pv"].values
#     # Pmax_m, k_m, G0_m = fit_logistic(G_m, P_m)
#     # # --- AFTERNOON FIT ---
#     # df_afternoon_fit = df_afternoon[df_afternoon["GHI"] > 5]
#     # G_a = df_afternoon_fit["GHI"].values
#     # P_a = df_afternoon_fit["P_pv"].values
#     # Pmax_a, k_a, G0_a = fit_logistic(G_a, P_a)
#     # P_fit_day = np.zeros_like(GHI_fit)
#     # P_fit_day[morning_mask] = logistic(GHI_fit[morning_mask], Pmax_m, k_m, G0_m)
#     # P_fit_day[~morning_mask] = logistic(GHI_fit[~morning_mask], Pmax_a, k_a, G0_a)


#     # ==========================================================================
#     # ML approach
#     # ==========================================================================
#     ML_model = joblib.load(f"pv_ghi_xgb_model_interp.pkl")
#     pv_pred = predict_pv_power_from_ghi(df_day_radiation, ML_model, ghi_col="MeteoSwiss stations")
#     total_installed_capacity = df_power_station_filtered.set_index("Datetime")["total_power_all"] / 1000  # in MW
#     # print(f"Total installed capacity: {total_installed_capacity.iloc[0]:.2f} MW")
#     pv_pred = pv_pred * total_installed_capacity.reindex(pv_pred.index, method="ffill")
#     # df_compare = pd.DataFrame({"Photovoltaik": df_day_radiation["Photovoltaik"], "pv_pred": pv_pred})
#     # print(df_compare.iloc[32:72])

#     # --- Fitted TEMP function approach --- #
#     a = 0.32234211642475497
#     gamma = 0.11582514573821052
#     k = 4.971876728789709e-16
#     T_air = df_day_radiation["T_air"]
#     P_fit_day = a * GHI_fit * (1 - gamma * (T_air + k * GHI_fit - 25)) / 1000  # In MW
#     P_fit_day = P_fit_day * total_installed_capacity.reindex(pv_pred.index, method="ffill")
#     # df_compare = pd.DataFrame({"Photovoltaik": df_day_radiation["Photovoltaik"], "PV_fit_day": P_fit_day})
#     # print(df_compare.iloc[32:72])


#     # Plot fit
#     fig, axs = plt.subplots(1, 2, figsize=(12, 5))

#     # --- Left plot: GHI vs P_pv with fit ---
#     sc = axs[0].scatter(df_fit_day["GHI"], df_fit_day["P_pv"], c=df_fit_day["hour"], cmap="viridis", label="Values")
#     fig.colorbar(sc, ax=axs[0], label="Hour")
#     axs[0].plot(GHI_fit, P_fit_day, marker="x", color="tab:red", label="Fit")
#     axs[0].plot(GHI_fit, pv_pred, marker="x", color="tab:grey", label="ML Fit")
#     axs[0].set_xlabel("GHI")
#     axs[0].set_ylabel("P_pv")
#     axs[0].grid(True)
#     axs[0].legend()
#     axs[0].set_title("GHI vs PV with linear Fit")

#     # --- Right plot: Ratio P/GHI over time ---
#     hours = (df_fit_day.index.hour + df_fit_day.index.minute / 60 + df_fit_day.index.second / 3600)
#     pv_pred_reindexed = pv_pred.reindex(df_fit_day.index)/pv_pred.max()
#     P_fit_day_reindexed = P_fit_day.reindex(df_fit_day.index)/P_fit_day.max()
#     ax1 = axs[1]
#     ax1.plot(hours, df_fit_day["P_pv"] / np.maximum(df_fit_day["GHI"], 1e-4), color="black", marker="o", markersize=3, label="Ratio")
#     ax1.plot(hours, df_fit_day["P_pv"] / np.maximum(df_fit_day["GHI"], GHI_threshold), color="black", linestyle="--", markersize=3, label=f"Ratio (ghi>{GHI_threshold})")
#     ax1.plot(hours, P_fit_day / np.maximum(GHI_fit, GHI_threshold), color="black", linestyle="--", marker="x", markersize=3, label=f"Ratio fit (ghi>{GHI_threshold})")
#     # ax1.plot(hours, pv_pred_reindexed / np.maximum(GHI_fit, GHI_threshold), color="black", linestyle="--", marker="x", markersize=3, label=f"Ratio ML (ghi>{GHI_threshold})")
#     ax1.set_xticks(range(0, 24))
#     ax1.set_xlabel("Hour of Day")
#     ax1.set_ylabel("Ratio PV/GHI")
#     ax1.legend(loc="upper left")
#     ax1.grid(True)
#     # Right y-axis at GHI
#     ax2 = ax1.twinx()
#     ax2.plot(hours, df_fit_day["GHI"]/df_fit_day["GHI"].max(), marker="x", linestyle="--", label="GHI")
#     ax2.plot(hours, df_fit_day["P_pv"]/df_fit_day["P_pv"].max(), marker="x", linestyle="--", label="PV")
#     ax2.plot(hours, P_fit_day_reindexed, marker="x", linestyle="--", label="Fit PV")
#     ax2.plot(hours, pv_pred_reindexed, marker="x", linestyle="--", label="ML PV")
#     ax2.set_ylabel("GHI & P_PV [norm]")
#     ax2.legend(loc="upper right")
#     ax1.set_title("PV / GHI Ratio Over Time")
#     plt.tight_layout()
#     plt.show()


#     # Loop over metrics
#     for metric in metrics:

#         ghi_metric = df_day_metrics[metric].reindex(idx).values

#         # --- Power error --- #
#         # Ratio
#         power_error_ratio = PV_weights_ratio_day * ghi_metric

#         # Fit
#         # # Logistic function
#         # P_fit_act = np.zeros_like(ghi_day)
#         # P_fit_forc = np.zeros_like(ghi_day)
#         # P_fit_act[morning_mask] = logistic(ghi_day[morning_mask], Pmax_m, k_m, G0_m)
#         # P_fit_forc[morning_mask] = logistic(ghi_day[morning_mask]-ghi_metric[morning_mask], Pmax_m, k_m, G0_m)
#         # P_fit_act[~morning_mask] = logistic(ghi_day[~morning_mask], Pmax_a, k_a, G0_a)
#         # P_fit_forc[~morning_mask] = logistic(ghi_day[~morning_mask]-ghi_metric[~morning_mask], Pmax_a, k_a, G0_a)
#         # power_error_fit = P_fit_act - P_fit_forc
#         # Temp model fit
#         GHI_forecast = ghi_day - ghi_metric
#         P_fit_day_forecast = a * GHI_forecast * (1 - gamma * (T_air + k * GHI_forecast - 25)) / 1000  # In MW
#         P_fit_day_forecast = P_fit_day_forecast * total_installed_capacity.reindex(pv_pred.index, method="ffill")
#         power_error_fit = P_fit_day - P_fit_day_forecast
#         # # ML model
#         # df_GHI = pd.DataFrame({"MeteoSwiss stations": df_day_radiation["MeteoSwiss stations"] - df_day_metrics[metric].reindex(idx)}, index=df_day_radiation.index)
#         # pv_pred_forecast = predict_pv_power_from_ghi(df_GHI, ML_model, ghi_col="MeteoSwiss stations")
#         # pv_pred_forecast = pv_pred_forecast * total_installed_capacity.reindex(pv_pred_forecast.index, method="ffill")
#         # power_error_fit = pv_pred - pv_pred_forecast


#         # Append power error to metrics dataframe
#         df_station_metrics_CH_interp.loc[idx, f"{metric} power_ratio"] = power_error_ratio
#         df_station_metrics_CH_interp.loc[idx, f"{metric} power_fit"] = power_error_fit


#     # --- Calculate correlation value --- #
#     cols_ratio = [f"{metric} power_ratio" for metric in metrics]
#     cols_fit = [f"{metric} power_fit" for metric in metrics]
#     norm_ratio = np.nanmax(np.abs(df_station_metrics_CH_interp[cols_ratio].values))
#     norm_fit = np.nanmax(np.abs(df_station_metrics_CH_interp[cols_fit].values))

#     for metric in metrics:

#         ghi_metric = df_day_metrics[metric].reindex(idx).values
#         power_error_ratio = df_station_metrics_CH_interp.loc[idx, f"{metric} power_ratio"].values
#         power_error_fit = df_station_metrics_CH_interp.loc[idx, f"{metric} power_fit"].values

#         # Pairwise-valid masks for clean correlations
#         m_ghi = ghi_mask & np.isfinite(ghi_metric) & np.isfinite(imb)
#         m_pwr_ratio = ghi_mask & np.isfinite(power_error_ratio) & np.isfinite(imb)
#         m_pwr_fit = ghi_mask & np.isfinite(power_error_fit) & np.isfinite(imb)

#         # Weighted pearson correlation
#         w_pearson_ghi = weighted_pearson(ghi_metric[m_ghi], imb[m_ghi], weights[m_ghi])

#         # Explainable imbalance power
#         Pexpl_power_ratio, k_power_ratio = explainable_power(power_error_ratio[m_ghi], imb[m_ghi], ghi_day[m_ghi], ghi_min=GHI_threshold)
#         Pexpl_power_fit, k_power_fit = explainable_power(power_error_fit[m_ghi], imb[m_ghi], ghi_day[m_ghi], ghi_min=GHI_threshold)
#         Pexpl_power_ratio_base, k_power_ratio_base = explainable_power(power_error_ratio[m_ghi], imb_anomaly[m_ghi], ghi_day[m_ghi], ghi_min=GHI_threshold)
#         Pexpl_power_fit_base, k_power_fit_base = explainable_power(power_error_fit[m_ghi], imb_anomaly[m_ghi], ghi_day[m_ghi], ghi_min=GHI_threshold)

#         # Driver contribution metric (normalized)
#         driver_contrib_power_ratio = driver_contribution(imb[m_ghi] / max(imb[m_ghi]), power_error_ratio[m_ghi] / norm_ratio)
#         driver_contrib_power_fit = driver_contribution(imb[m_ghi] / max(imb[m_ghi]), power_error_fit[m_ghi] / norm_fit)
#         driver_contrib_power_ratio_base = driver_contribution(imb[m_ghi] / max(imb_anomaly[m_ghi]), power_error_ratio[m_ghi] / norm_ratio)
#         driver_contrib_power_fit_base = driver_contribution(imb[m_ghi] / max(imb_anomaly[m_ghi]), power_error_fit[m_ghi] / norm_fit)

#         # RMSE (normalized)
#         RMSE_ratio = compute_rmse(imb[m_ghi], power_error_ratio[m_ghi])
#         RMSE_fit = compute_rmse(imb[m_ghi], power_error_fit[m_ghi])
#         RMSE_ratio_base = compute_rmse(imb_anomaly[m_ghi], power_error_ratio[m_ghi])
#         RMSE_fit_base = compute_rmse(imb_anomaly[m_ghi], power_error_fit[m_ghi])


#         #
#         # ============================
#         # Permutation test
#         # ============================
#         S_obs = w_pearson_ghi
#         func_obs = weighted_pearson
#         N_PERM = 1000
#         S_null = []
#         n = len(imb)
#         mask = m_ghi

#         for _ in range(N_PERM):
#             shift = np.random.randint(1, n)
#             e_shift = np.roll(ghi_metric, shift)

#             # keep the SAME timestamps, but drop any NaNs introduced by the roll
#             mm = mask & np.isfinite(e_shift)

#             S_perm = func_obs(e_shift[mm], imb[mm], weights[mm])
#             S_null.append(S_perm)

#         S_null = np.array(S_null)
#         p_value = (1 + np.sum(S_null >= S_obs)) / (1 + len(S_null))


#         #
#         # Append
#         rows.append(
#             {
#                 "Datetime": pd.to_datetime(day),
#                 "Metric": metric,
#                 # Weighted Pearson
#                 "Weighted Pearson GHI": w_pearson_ghi,
#                 # explainable imbalance energy
#                 "P_expl Power ratio": Pexpl_power_ratio,
#                 "P_expl Power fit": Pexpl_power_fit,
#                 "P_expl Power ratio_base": Pexpl_power_ratio_base,
#                 "P_expl Power fit_base": Pexpl_power_fit_base,
#                 # "k GHI": k_ghi,
#                 "k Power ratio": k_power_ratio,
#                 "k Power fit": k_power_fit,
#                 # Permutation test
#                 "Permutation p-value": p_value,
#                 # Driver contribution
#                 "Driver contribution Power ratio": driver_contrib_power_ratio,
#                 "Driver contribution Power fit": driver_contrib_power_fit,
#                 "Driver contribution Power ratio_base": driver_contrib_power_ratio_base,
#                 "Driver contribution Power fit_base": driver_contrib_power_fit_base,
#                 # RMSE
#                 "RMSE Power ratio": RMSE_ratio,
#                 "RMSE Power fit": RMSE_fit,
#                 "RMSE Power ratio_base": RMSE_ratio_base,
#                 "RMSE Power fit_base": RMSE_fit_base,
#             }
#         )
# # End for loop


# =========================================
# Save full interpolated metrics (all days)
# =========================================
metrics_out_path = os.path.join(folder, f"station_metrics_CH_interp_{date_label}.csv")
df_station_metrics_CH_interp.to_csv(metrics_out_path, sep=";")
print(f"Saved station metrics to {metrics_out_path}")

# # Make a daily correlations dataframe and save
# df_daily_metrics_CH = pd.DataFrame(rows)
# # Save daily metrics to csv
# out_path = os.path.join(folder, f"daily_correlation_CH_{date_label}.csv")
# df_daily_metrics_CH.to_csv(out_path, index=False, sep=";")
