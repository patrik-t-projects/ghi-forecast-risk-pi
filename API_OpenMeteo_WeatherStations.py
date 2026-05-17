from __future__ import annotations
import pandas as pd
import os
from OpenMeteo_API_functions import open_meteo_api, compress_ensemble_runs
from config import DATA_DIR, data_path


"""
===============================================================================
OPEN-METEO WEATHER DATA PIPELINE FOR METEOSWISS STATIONS (SWITZERLAND)
===============================================================================

PURPOSE
-------
This script downloads hourly solar radiation data from the Open-Meteo API
for a set of geocoded MeteoSwiss weather stations in Switzerland.

For each station and coordinate, it retrieves:
- Historical reanalysis data (ERA5, ERA5 ensemble)
- Archived previous forecast model runs (ICON, MeteoSwiss ICON)

The resulting datasets are merged into a single, station-level,
hourly time series suitable for solar resource analysis, PV production
modeling, or validation of forecast models.


PIPELINE SUMMARY
----------------

1. INPUT PREPARATION
   - Load MeteoSwiss station metadata from CSV
   - Extract station identifier and WGS84 coordinates
   - Ensure identifier uniqueness and clean formatting
   - Skip stations already processed or previously failed (autosave logic)

2. API DATA SOURCES
   - Historical reanalysis: ERA5 and ERA5 ensemble
   - Archived forecast runs: ICON and MeteoSwiss ICON (previous runs API)
   - Hourly solar radiation variables only

3. BATCHED API ACCESS
   - Query stations in configurable batches
   - Throttle requests to respect Open-Meteo rate limits
   - Cache partial results to disk after each batch (restart-safe)

4. TIME WINDOW HANDLING
   - Historical data: single request per batch over full time range
   - Previous model runs: split into monthly / weekly / N-day windows
   - Fixed global start and end dates (no commissioning logic)

5. ERROR HANDLING & RECOVERY
   - Immediate stop on API failure to protect IP
   - Autosave all successful data before exit
   - Log failed stations and coordinates to a separate CSV

6. DATA ASSEMBLY
   - Parse hourly time series per station
   - Merge historical and previous-run datasets on station and timestamp
   - Attach station coordinates for downstream spatial processing

7. OUTPUT
   - Clean, hourly, station-level dataset
   - CSV output ready for aggregation, validation, or PV modeling workflows


DESIGN GOALS
------------
- Restartable execution with autosave and skip logic
- Minimal and efficient API usage
- Transparent failure tracking
- Scalable to multi-year, multi-model weather datasets

===============================================================================
"""


# =========================
# USER SETTINGS
# =========================
START_DATE = globals().get("START_DATE", "2026-03-21")
END_DATE   = globals().get("END_DATE", "2026-05-17")

# Open-Meteo
TZ = "Europe/Berlin"


# --- ENSEMBLE MODEL RUNS ---
ENSEMBLE_RUNS_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_RUNS_MODELS = ["meteoswiss_icon_ch1_ensemble", "meteoswiss_icon_ch2_ensemble"]
ENSEMBLE_RUNS_HOURLY_VARS = ["shortwave_radiation", "cloud_cover"]

# --- ENSEMBLE MEAN ---
ENSEMBLE_MEAN_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_MEAN_MODELS = ["meteoswiss_icon_ch1_ensemble_mean", "meteoswiss_icon_ch2_ensemble_mean"]
ENSEMBLE_MEAN_HOURLY_VARS = ["shortwave_radiation_spread", "cloud_cover_spread"]

# --- PREVIOUS MODEL RUNS ---
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODELS_PREV = ["meteoswiss_icon_seamless"]
HOURLY_BASE_VARS = ["shortwave_radiation", "shortwave_radiation_previous_day1", "shortwave_radiation_previous_day2", "shortwave_radiation_previous_day3"]


# Throttling between calls
# These values have to be identical with the ones in "OpenMeteo_API_functions"!
BATCH_SIZE = 5
SLEEP_BETWEEN_CALLS_SEC = 0.5
# PREVIOUS_RUNS_WINDOW = "month"   # "month", "week", "ndays"
# PREVIOUS_RUNS_N_DAYS = 3  # Number of days for ndays

# Output
OUTPUT_DIR = str(DATA_DIR)
OUTPUT_FILE = globals().get("OUTPUT_FILE", "Data_OpenMeteo_ENSEMBLE_PrevRuns_weather_stations_2026_2")
# =========================


# ==============================
# Build output dataframe
# ==============================
PLANTS_CSV = data_path("meteoswiss_stations_metadata.csv")
plants = pd.read_csv(PLANTS_CSV, sep=";")
plants = plants.rename(columns={
    "station_coordinates_wgs84_lat": "lat",
    "station_coordinates_wgs84_lon": "lon",
})

print(f"Fetching weather for {len(plants)} plants from {START_DATE} to {END_DATE}", flush=True)

# --- Ensemble model runs ---
df_ensemble_runs = open_meteo_api(
    type_str="ENSEMBLE RUNS",
    url=ENSEMBLE_RUNS_URL,
    plants=plants,
    start_date=START_DATE,
    end_date=END_DATE,
    timezone=TZ,
    hourly_vars=ENSEMBLE_RUNS_HOURLY_VARS,
    models=ENSEMBLE_RUNS_MODELS,
    batch_size=BATCH_SIZE,
    sleep_sec=SLEEP_BETWEEN_CALLS_SEC,
    autosave_path=os.path.join(OUTPUT_DIR, "autosave_ens_runs.csv"),
)
# Aggregate mean, std, min, max
df_ensemble_runs = compress_ensemble_runs(df_ensemble_runs)


# --- Ensemble mean & spread ---
df_ensemble_mean = open_meteo_api(
    type_str="ENSEMBLE MEAN",
    url=ENSEMBLE_MEAN_URL,
    plants=plants,
    start_date=START_DATE,
    end_date=END_DATE,
    timezone=TZ,
    hourly_vars=ENSEMBLE_MEAN_HOURLY_VARS,
    models=ENSEMBLE_MEAN_MODELS,
    batch_size=BATCH_SIZE,
    sleep_sec=SLEEP_BETWEEN_CALLS_SEC,
    autosave_path=os.path.join(OUTPUT_DIR, "autosave_ens_mean.csv"),
)

# --- Previous model runs ---
df_prev = open_meteo_api(
    type_str="PREVIOUS RUNS",
    url=PREVIOUS_RUNS_URL,
    plants=plants,
    start_date=START_DATE,
    end_date=END_DATE,
    timezone=TZ,
    hourly_vars=HOURLY_BASE_VARS,
    models=MODELS_PREV,
    batch_size=BATCH_SIZE,
    sleep_sec=SLEEP_BETWEEN_CALLS_SEC,
    autosave_path=os.path.join(OUTPUT_DIR, "autosave_prev.csv"),
)



# ==================================
# Retry FAILED ENSEMBLE RUNS stations
# ==================================
FAILED_ENSEMBLE_RUNS_PATH = os.path.join(OUTPUT_DIR, "autosave_ens_runs_failed.csv")

if os.path.exists(FAILED_ENSEMBLE_RUNS_PATH):
    failed_ensemble_runs = pd.read_csv(FAILED_ENSEMBLE_RUNS_PATH)

    if not failed_ensemble_runs.empty:
        print(f"ðŸ” Retrying ensemble runs for {len(failed_ensemble_runs)} failed stations", flush=True)

        retry_plants_ens_runs = (plants.merge(failed_ensemble_runs[["station_abbr"]], on="station_abbr", how="inner"))

        df_ensemble_runs_retry = open_meteo_api(
            type_str="ENSEMBLE RUNS",
            url=ENSEMBLE_RUNS_URL,
            plants=retry_plants_ens_runs,
            start_date=START_DATE,
            end_date=END_DATE,
            timezone=TZ,
            hourly_vars=ENSEMBLE_RUNS_HOURLY_VARS,
            models=ENSEMBLE_RUNS_MODELS,
            batch_size=BATCH_SIZE,
            sleep_sec=SLEEP_BETWEEN_CALLS_SEC,
            autosave_path=os.path.join(OUTPUT_DIR, "autosave_ens_runs_retry.csv"),
        )

        if not df_ensemble_runs_retry.empty:
            df_ensemble_runs_retry = compress_ensemble_runs(df_ensemble_runs_retry)
            df_ensemble_runs = pd.concat([df_ensemble_runs, df_ensemble_runs_retry], ignore_index=True)
# End retry ENSEMBLE RUNS

# ==================================
# Retry FAILED ENSEMBLE MEAN stations
# ==================================
FAILED_ENSEMBLE_MEAN_PATH = os.path.join(OUTPUT_DIR, "autosave_ens_mean_failed.csv")

if os.path.exists(FAILED_ENSEMBLE_MEAN_PATH):
    failed_ensemble_mean = pd.read_csv(FAILED_ENSEMBLE_MEAN_PATH)

    if not failed_ensemble_mean.empty:
        print(f"ðŸ” Retrying ensemble mean for {len(failed_ensemble_mean)} failed stations", flush=True)

        retry_plants_ens_mean = (plants.merge(failed_ensemble_mean[["station_abbr"]], on="station_abbr", how="inner"))

        df_ensemble_mean_retry = open_meteo_api(
            type_str="ENSEMBLE MEAN",
            url=ENSEMBLE_MEAN_URL,
            plants=retry_plants_ens_mean,
            start_date=START_DATE,
            end_date=END_DATE,
            timezone=TZ,
            hourly_vars=ENSEMBLE_MEAN_HOURLY_VARS,
            models=ENSEMBLE_MEAN_MODELS,
            batch_size=BATCH_SIZE,
            sleep_sec=SLEEP_BETWEEN_CALLS_SEC,
            autosave_path=os.path.join(OUTPUT_DIR, "autosave_ens_mean_retry.csv"),
        )

        if not df_ensemble_mean_retry.empty:
            df_ensemble_mean = pd.concat([df_ensemble_mean, df_ensemble_mean_retry], ignore_index=True)
# End retry ENSEMBLE RUNS

# ==================================
# Retry FAILED previous-run stations
# ==================================
FAILED_PREV_PATH = os.path.join(OUTPUT_DIR, "autosave_prev_failed.csv")
if os.path.exists(FAILED_PREV_PATH):
    failed_prev = pd.read_csv(FAILED_PREV_PATH)

    if not failed_prev.empty:
        print(f"ðŸ” Retrying previous-run weather for {len(failed_prev)} failed stations", flush=True)

        retry_plants_prev = plants.merge(failed_prev[["station_abbr"]], on="station_abbr", how="inner")

        df_prev_retry = open_meteo_api(
            type_str="PREVIOUS RUNS",
            url=PREVIOUS_RUNS_URL,
            plants=retry_plants_prev,
            start_date=START_DATE,
            end_date=END_DATE,
            timezone=TZ,
            hourly_vars=HOURLY_BASE_VARS,
            models=MODELS_PREV,
            batch_size=BATCH_SIZE,
            sleep_sec=SLEEP_BETWEEN_CALLS_SEC,
            autosave_path=os.path.join(OUTPUT_DIR, "autosave_prev_retry.csv"),
        )

        if not df_prev_retry.empty:
            df_prev = pd.concat([df_prev, df_prev_retry], ignore_index=True)
# End retry previous model runs


# ===============================
# Coverage & completeness check
# ===============================

expected_stations = set(plants["station_abbr"])
expected_start = pd.to_datetime(START_DATE)
expected_end = pd.to_datetime(END_DATE)

print("\n=== COVERAGE CHECK ===")

for name, df in [("ENSEMBLE-RUNS", df_ensemble_runs), ("ENSEMBLE-MEAN", df_ensemble_mean), ("PREVIOUS-RUN", df_prev)]:
    print(f"\n--- {name} ---")

    found_stations = set(df["station_abbr"])
    missing = expected_stations - found_stations

    print(f"Stations expected : {len(expected_stations)}")
    print(f"Stations found    : {len(found_stations)}")

    if missing:
        print(f"âš ï¸ Missing stations ({len(missing)}): {sorted(missing)}")
    else:
        print("âœ… No missing stations")

    ranges = df.groupby("station_abbr")["Datetime"].agg(start="min", end="max").reset_index()

    incomplete = ranges[(ranges["start"] > expected_start) | (ranges["end"] < expected_end)]

    if incomplete.empty:
        print("âœ… All stations have full date coverage")
    else:
        print(f"âš ï¸ Incomplete date coverage ({len(incomplete)} stations):")
        print(incomplete.sort_values("station_abbr").to_string(index=False))
# End completeness check


# ===============================
# Merge everything
# ===============================
df_ensemble_runs = df_ensemble_runs.drop_duplicates(subset=["station_abbr", "Datetime"])
df_ensemble_mean = df_ensemble_mean.drop_duplicates(subset=["station_abbr", "Datetime"])
df_prev = df_prev.drop_duplicates(subset=["station_abbr", "Datetime"])

out = (df_prev.merge(df_ensemble_runs, on=["station_abbr", "Datetime"], how="left", suffixes=("", "_ens_run"))
              .merge(df_ensemble_mean, on=["station_abbr", "Datetime"], how="left", suffixes=("", "_ens_mean"))
              .merge(plants[["station_abbr", "lat", "lon"]], on="station_abbr", how="left")
    )

out = out.drop_duplicates(subset=["station_abbr", "Datetime"], keep="last")

# Save file
save_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE}.csv")
out.to_csv(save_path, sep=";", index=False, encoding="utf-8")
print(f"Saved dataset to: {save_path}")
