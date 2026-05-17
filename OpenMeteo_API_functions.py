from __future__ import annotations
import time
from typing import Dict, Optional, List
import pandas as pd
import requests
import os
from requests.exceptions import RequestException
import subprocess
import random
import json
from pathlib import Path



# Throttling between calls
BATCH_SIZE_ENSEMBLE = 5
BATCH_SIZE_PREVIOUS   = 5
SLEEP_BETWEEN_CALLS_SEC = 0.5
PREVIOUS_RUNS_WINDOW = "month"   # "month", "week", "ndays"
PREVIOUS_RUNS_N_DAYS = 3  # Number of days for ndays


VPN_COUNTRIES = [
    "Switzerland",
    "Germany",
    "France",
    "Netherlands",
    "Austria",
    "Italy",
    "Belgium",
    "Spain",
]

VPN_STATE_FILE = Path.home() / "nordvpn_country_history.json"
VPN_RECENT_COUNTRY_LIMIT = 3   # do not reuse last 3 countries


def run_command(command):
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def load_vpn_history():
    if not VPN_STATE_FILE.exists():
        return []

    try:
        with open(VPN_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_vpn_history(history):
    with open(VPN_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-20:], f, indent=2)


def get_nordvpn_status():
    stdout, stderr, code = run_command("nordvpn status")
    return stdout


def choose_new_vpn_country():
    history = load_vpn_history()
    recent_countries = history[-VPN_RECENT_COUNTRY_LIMIT:]

    available = [c for c in VPN_COUNTRIES if c not in recent_countries]

    if not available:
        available = VPN_COUNTRIES.copy()

    return random.choice(available)


def change_vpn_country():
    old_status = get_nordvpn_status()
    print("\nCurrent NordVPN status:")
    print(old_status)

    new_country = choose_new_vpn_country()

    print(f"\nChanging NordVPN location to: {new_country}")

    stdout, stderr, code = run_command(f'nordvpn connect "{new_country}"')

    print(stdout)
    if stderr:
        print(stderr)

    if code != 0:
        raise RuntimeError(f"NordVPN failed to connect to {new_country}")

    history = load_vpn_history()
    history.append(new_country)
    save_vpn_history(history)

    print("\nNew NordVPN status:")
    print(get_nordvpn_status())

    return new_country


def is_429_error(error):
    text = str(error).lower()
    return (
        "429" in text
        or "too many requests" in text
        or "rate limit" in text
        or "rate-limit" in text
    )


def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def open_meteo_api(
    *,
    type_str: str,
    url: str,
    plants: pd.DataFrame,
    start_date: str,
    end_date: str,
    timezone: str,
    hourly_vars: List[str],
    models: Optional[List[str]] = None,
    batch_size: int,
    sleep_sec: float = 0.0,
    timeout_sec: int = 30,
    autosave_path: Optional[str] = None,
    max_attempts: int = 10,
    retry_sleep_sec: float = 10.0,
    change_vpn_on_error: bool = True,
) -> pd.DataFrame:

    api_name = type_str
    failed_path = autosave_path.replace(".csv", "_failed.csv") if autosave_path else None
    weather_frames: List[pd.DataFrame] = []

    # -------------------------
    # Helper: autosave progress
    # -------------------------
    def _autosave_partial():
        nonlocal weather_frames
        if autosave_path and weather_frames:
            df_new = pd.concat(weather_frames, ignore_index=True)

            if os.path.exists(autosave_path):
                df_old = pd.read_csv(autosave_path)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new

            df_all = df_all.drop_duplicates(subset=["station_abbr", "Datetime"], keep="last")
            df_all.to_csv(autosave_path, index=False)
            print(f"Autosaved {len(df_all)} rows -> {autosave_path}", flush=True)
            weather_frames = []

    
    # -------------------------
    # Helper: log failed batch
    # -------------------------
    def _log_failed_batch(batch_df: pd.DataFrame, status):
        if not failed_path:
            return

        rows = []
        for _, r in batch_df.iterrows():
            rows.append({
                "station_abbr": r["station_abbr"],
                "lat": r["lat"],
                "lon": r["lon"],
                "status": status,
            })

        df_new = pd.DataFrame(rows)

        if os.path.exists(failed_path):
            df_old = pd.read_csv(failed_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv(failed_path, index=False)
        print(f"Logged {len(rows)} failed plants -> {failed_path}", flush=True)


    def _get_completed_stations_from_autosave(path, start_date, end_date):
        if not path or not os.path.exists(path):
            return set()

        df = pd.read_csv(path, usecols=["station_abbr", "Datetime"])
        if df.empty:
            return set()

        df["Datetime"] = pd.to_datetime(df["Datetime"])

        request_start = pd.to_datetime(start_date)
        request_end = pd.to_datetime(end_date) + pd.Timedelta(hours=23)

        coverage = df.groupby("station_abbr")["Datetime"].agg(start="min", end="max").reset_index()

        completed = coverage.loc[(coverage["start"] <= request_start) & (coverage["end"] >= request_end), "station_abbr"]

        return set(completed)


    # ----------------------------------
    # Skip already successful / failed
    # ----------------------------------
    skip_ids = set()

    if autosave_path and os.path.exists(autosave_path):
        skip_ids |= _get_completed_stations_from_autosave(autosave_path, start_date, end_date)

    # if failed_path and os.path.exists(failed_path):
    #     skip_ids |= set(pd.read_csv(failed_path, usecols=["station_abbr"])["station_abbr"])

    if skip_ids:
        before = len(plants)
        plants = plants[~plants["station_abbr"].isin(skip_ids)].copy()
        after = len(plants)
        print(f"Skipping {before - after} completed/failed plants (remaining: {after})", flush=True)

    if plants.empty:
        print("All plants already processed", flush=True)
        if autosave_path and os.path.exists(autosave_path):
            df_all = pd.read_csv(autosave_path)
            df_all["Datetime"] = pd.to_datetime(df_all["Datetime"])
            return df_all

        return pd.DataFrame()

    # =========================
    # Main batch loop
    # =========================
    for batch_idx, idxs in enumerate(chunked(plants.index.tolist(), batch_size), start=1):
        batch = plants.loc[idxs]

        print(
            f"Open-Meteo batch {batch_idx}: "
            f"{len(batch)} plants [{batch['station_abbr'].iloc[0]} ... {batch['station_abbr'].iloc[-1]}] | "
            f"lat/lon=({batch['lat'].iloc[0]:.5f}, {batch['lon'].iloc[0]:.5f})",
            flush=True,
        )

        params = {
            "latitude": ",".join(batch["lat"].astype(float).astype(str)),
            "longitude": ",".join(batch["lon"].astype(float).astype(str)),
            "timezone": timezone,
            "hourly": ",".join(hourly_vars),
        }

        if models:
            params["models"] = ",".join(models)
        if "previous-runs-api" in url:
            params["time_mode"] = "time_interval"

        # -------------------------
        # Time slicing logic
        # -------------------------
        if "previous-runs-api" in url:
            if PREVIOUS_RUNS_WINDOW == "month":
                date_starts = pd.date_range(start=pd.to_datetime(start_date).replace(day=1), end=pd.to_datetime(end_date), freq="MS")
            elif PREVIOUS_RUNS_WINDOW == "week":
                date_starts = pd.date_range(start=start_date, end=end_date, freq="7D")
            elif PREVIOUS_RUNS_WINDOW == "ndays":
                date_starts = pd.date_range(start=start_date, end=end_date, freq=f"{PREVIOUS_RUNS_N_DAYS}D")
            else:
                raise ValueError(f"Unknown PREVIOUS_RUNS_WINDOW: {PREVIOUS_RUNS_WINDOW}")
        else:
            # Normal historical/ensemble request with explicit dates,
            # or forecast request with Open-Meteo default forecast horizon.
            date_starts = [None]

        for d_start in date_starts:
            if "previous-runs-api" in url:
                if PREVIOUS_RUNS_WINDOW == "month":
                    d_end = (d_start + pd.offsets.MonthEnd(1)).date()
                elif PREVIOUS_RUNS_WINDOW == "week":
                    d_end = (d_start + pd.Timedelta(days=6)).date()
                elif PREVIOUS_RUNS_WINDOW == "ndays":
                    d_end = (d_start + pd.Timedelta(days=PREVIOUS_RUNS_N_DAYS - 1)).date()

                actual_start = max(d_start.date(), pd.to_datetime(start_date).date())
                d_end = min(d_end, pd.to_datetime(end_date).date())
                params["start_date"] = actual_start.isoformat()
                params["end_date"] = d_end.isoformat()
            else:
                if start_date is not None:
                    params["start_date"] = start_date
                if end_date is not None:
                    params["end_date"] = end_date

            js = None

            for attempt in range(1, max_attempts + 1):
                try:
                    print(f"{api_name} request: {params.get('start_date', 'default forecast start')} -> {params.get('end_date', 'default forecast horizon')}", flush=True)
                    r = requests.get(url, params=params, timeout=timeout_sec)
                    r.raise_for_status()
                    js = r.json()
                    break

                except RequestException as e:
                    status = getattr(e.response, "status_code", None)
                    print(f"API error on attempt {attempt}/{max_attempts} (status={status}) {e}", flush=True)

                    if attempt < max_attempts:
                        print("Retrying request with a new VPN location.", flush=True)
                        if change_vpn_on_error:
                            try:
                                change_vpn_country()
                            except Exception as vpn_error:
                                print(f"VPN change failed: {vpn_error}", flush=True)
                                print("Continuing retry loop without successful VPN change.", flush=True)

                        time.sleep(retry_sleep_sec)
                        continue

                    print("Stopping Open-Meteo run after failed request.", flush=True)

                    _autosave_partial()
                    _log_failed_batch(batch, status)
                    raise 

            if js is None:
                _autosave_partial()
                _log_failed_batch(batch, "no_json_response")
                raise RuntimeError("Open-Meteo request failed without a valid JSON response.")
            
            locations = js if isinstance(js, list) else [js]

            for run_id, loc in enumerate(locations):
                hourly = loc.get("hourly", {})
                times = hourly.get("time", [])
                if not times:
                    continue

                df = pd.DataFrame({"Datetime": pd.to_datetime(times)})
                df["station_abbr"] = batch.iloc[run_id]["station_abbr"]

                for key, values in hourly.items():
                    if key != "time":
                        df[key] = values

                weather_frames.append(df)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    # -------------------------
    # Final autosave
    # -------------------------
    _autosave_partial()

    if autosave_path and os.path.exists(autosave_path):
        df_all = pd.read_csv(autosave_path)
        df_all["Datetime"] = pd.to_datetime(df_all["Datetime"])
        return df_all

    return pd.DataFrame()


def compress_ensemble_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    base_cols = ["station_abbr", "Datetime"]
    out = df[base_cols].copy()

    icon_ch1_cols = [
        c for c in df.columns
        if (
            c == "shortwave_radiation_meteoswiss_icon_ch1_ensemble"
            or (
                c.startswith("shortwave_radiation_member")
                and c.endswith("_meteoswiss_icon_ch1_ensemble")
            )
        )
    ]

    icon_ch2_cols = [
        c for c in df.columns
        if (
            c == "shortwave_radiation_meteoswiss_icon_ch2_ensemble"
            or (
                c.startswith("shortwave_radiation_member")
                and c.endswith("_meteoswiss_icon_ch2_ensemble")
            )
        )
    ]

    if icon_ch1_cols:
        out["shortwave_radiation_icon_ch1_ens_mean"] = df[icon_ch1_cols].mean(axis=1)
        out["shortwave_radiation_icon_ch1_ens_std"] = df[icon_ch1_cols].std(axis=1, ddof=0)
        out["shortwave_radiation_icon_ch1_ens_min"] = df[icon_ch1_cols].min(axis=1)
        out["shortwave_radiation_icon_ch1_ens_max"] = df[icon_ch1_cols].max(axis=1)
    else:
        print("Warning: No MeteoSwiss ICON-CH1 ensemble columns found", flush=True)


    if icon_ch2_cols:
        out["shortwave_radiation_icon_ch2_ens_mean"] = df[icon_ch2_cols].mean(axis=1)
        out["shortwave_radiation_icon_ch2_ens_std"] = df[icon_ch2_cols].std(axis=1, ddof=0)
        out["shortwave_radiation_icon_ch2_ens_min"] = df[icon_ch2_cols].min(axis=1)
        out["shortwave_radiation_icon_ch2_ens_max"] = df[icon_ch2_cols].max(axis=1)
    else:
        print("Warning: No MeteoSwiss ICON-CH2 ensemble columns found", flush=True)

    
    print(f"Compressed ensemble runs: {len(icon_ch1_cols)} ICON-CH1 cols, {len(icon_ch2_cols)} ICON-CH2 cols", flush=True)

    return out
