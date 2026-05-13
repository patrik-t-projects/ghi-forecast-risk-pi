import pandas as pd
import time
import io
import requests
from config import data_path


# -------------------------
# Settings
# -------------------------
time_window = "recent"
resolution = "h"

# -------------------------
# Parameters to extract
# -------------------------
parameter_gre = "gre000h0"
parameter_ods = "ods000h0"
parameter_tre = "tre200h0"
parameter_rre = "rre150h0"
parameter_sre = "sre000h0"
parameter_hto = "htoauth0"

parameters = [
    parameter_gre,
    # parameter_ods,
    # parameter_tre,
    # parameter_rre,
    # parameter_sre,
    # parameter_hto,
]
required_parameter = parameter_gre


# Save paths
stations_path = data_path("meteoswiss_stations_metadata.csv")
out_path = data_path("meteoswiss_gre_hourly_2026.csv")


# -------------------------
# Load station metadata
# -------------------------
stations_url = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_stations.csv"
stations = pd.read_csv(stations_url, sep=";", encoding="windows-1252", low_memory=False)

stations = stations[[
    "station_abbr",
    "station_wigos_id",
    "station_coordinates_wgs84_lat",
    "station_coordinates_wgs84_lon",
]]

# -------------------------
# Load data inventory
# -------------------------
inventory_url = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_datainventory.csv"
inventory = pd.read_csv(inventory_url, sep=";", encoding="windows-1252")

# Stations that measure global radiation
stations_with_rad = sorted(inventory.loc[inventory["parameter_shortname"] == parameter_gre, "station_abbr"].str.lower().unique())
print(f"Found {len(stations_with_rad)} stations with {parameter_gre}.")

df_stations_with_rad = stations[stations["station_abbr"].str.lower().isin(stations_with_rad)].copy()

# Save only stations with radiation parameters
df_stations_with_rad.to_csv(stations_path, sep=";", index=False, encoding="utf-8")
print(f"Saved station metadata CSV to: {stations_path}")


# Function for getting csv data with retries
def read_csv_with_retries(url, retries=5, wait=10, timeout=30):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            return pd.read_csv(
                io.StringIO(response.content.decode("windows-1252")),
                sep=";",
                encoding="windows-1252",
            )

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt}/{retries} failed: {e}")
            time.sleep(wait)

    raise last_error


# -------------------------
# Download + filter data
# -------------------------
dfs = []
failed_stations = []

for i, st in enumerate(stations_with_rad, 1):
    url = f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/{st}/ogd-smn_{st}_{resolution}_{time_window}.csv"

    try:
        print(f"[{i}] Downloading data for {st.upper()}")
        df = read_csv_with_retries(url, retries=5, wait=10)
    except Exception as e:
        print(f"[{i}] {st.upper()} - download failed after retries: {e}")
        failed_stations.append(st.upper())
        continue

    df["Datetime"] = pd.to_datetime(df["reference_timestamp"], utc=True, dayfirst=True)
    # Filter only year 2026
    df = df[df["Datetime"].dt.year == 2026]

    if df.empty:
        print("Returned empty dataframe")
        continue

    # Always include GRE
    if required_parameter not in df.columns:
        print(f"[{i}] {st.upper()} - no {required_parameter} column")
        continue

    # Initialize output dataframe
    df_out = pd.DataFrame()
    df_out["Datetime"] = df["Datetime"]

    # Loop through all parameters
    for param in parameters:    
        if param in df.columns:
            df_out[param] = df[param]
        else:
            df_out[param] = pd.NA

    df_out["station_abbr"] = st.upper()

    dfs.append(df_out)


# -------------------------
# Combine metadata
# -------------------------
if not dfs:
    raise RuntimeError("No radiation data loaded.")
radiation = pd.concat(dfs, ignore_index=True)

print(f"Done. Final dataset: {len(radiation)} rows.")
print(radiation.columns)
print(radiation.head())

print("Saving file to csv...")

print(f"Date range in saved CSV: {radiation['Datetime'].min()} to {radiation['Datetime'].max()}")
radiation.to_csv(out_path, sep=";", index=False, encoding="utf-8")

print(f"Saved dataset to: {out_path}")
