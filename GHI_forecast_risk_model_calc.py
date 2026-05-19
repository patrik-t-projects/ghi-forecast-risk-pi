import numpy as np
import pandas as pd

from config import DATA_DIR, DOWNLOADS_DIR, REPORTS_DIR
from GHI_forecast_risk_functions import compute_future_quantile_error_band, fit_risk_score_from_features


MODEL = "ICON1"
HIST_START_DATE = "2026-03-21"
RISK_WINDOW = 0.1
QUANTILE = 0.95
MIN_SAMPLES = 5
GHI_THRESHOLD = 0

OUTPUT_PATH = REPORTS_DIR / "GHI_forecast_risk_report_data.csv"

PREV_DAY_COLS = [
    f"{MODEL} prev day 0",
    f"{MODEL} prev day 1",
    f"{MODEL} prev day 2",
    f"{MODEL} prev day 3",
]

METRIC_COLS = [
    f"MeteoSwiss - {MODEL}_d0",
    f"MeteoSwiss - {MODEL}_d1",
    f"MeteoSwiss - {MODEL}_d2",
    f"MeteoSwiss - {MODEL}_d3",
]

FEATURE_COLS = [
    f"{MODEL} range",
    f"{MODEL} std",
    f"{MODEL} ens spread",
    f"{MODEL} CC ens spread",
    f"{MODEL} horizon spread",
    f"{MODEL} forecast update total abs",
]

REQUIRED_RADIATION_COLS = [
    "Datetime",
    "MeteoSwiss stations",
    *PREV_DAY_COLS,
    f"{MODEL} std",
    f"{MODEL} min",
    f"{MODEL} max",
    f"{MODEL} ens spread",
    f"{MODEL} CC ens spread",
]

REQUIRED_METRIC_COLS = [
    "Datetime",
    *METRIC_COLS,
]

ERROR_COL = f"MeteoSwiss - {MODEL}_d0"
ABS_ERROR_COL = f"abs {ERROR_COL}"
PREDICTED_RISK_COL = f"{MODEL} predicted risk score"
FITTED_RISK_COL = f"{MODEL} fitted risk score including day"

ICON2_BACKUP_SUFFIXES = [
    "mean",
    "std",
    "min",
    "max",
    "ens spread",
    "CC ens spread",
]

EXTRA_OUTPUT_COLS = [
    f"{MODEL} mean",
    f"{MODEL} min",
    f"{MODEL} max",
]


def clean_float(value):
    if pd.isna(value):
        return None
    return float(value)


def performance_percent(error, risk):
    tmp = pd.DataFrame({"error": error, "risk": risk}).replace([np.inf, -np.inf], np.nan).dropna()
    tmp = tmp[tmp["risk"] != 0].copy()

    if len(tmp) < 3:
        return np.nan

    x_error = tmp["error"].abs()
    y_risk = tmp["risk"].abs()

    if x_error.nunique() < 2 or y_risk.nunique() < 2:
        return np.nan

    tau = y_risk.corr(x_error, method="kendall")

    return 100 * (tau + 1) / 2 if np.isfinite(tau) else np.nan


def load_raw_data():
    df_rad = pd.read_csv(DATA_DIR / "station_radiation_CH_interp_year_2026.csv", sep=";")
    df_metrics = pd.read_csv(DATA_DIR / "station_metrics_CH_interp_year_2026.csv", sep=";")

    missing_rad = [col for col in REQUIRED_RADIATION_COLS if col not in df_rad.columns]
    missing_metrics = [col for col in REQUIRED_METRIC_COLS if col not in df_metrics.columns]

    if missing_rad:
        raise ValueError(f"Missing required radiation columns: {missing_rad}")

    if missing_metrics:
        raise ValueError(f"Missing required metric columns: {missing_metrics}")

    df_rad["Datetime UTC"] = pd.to_datetime(df_rad["Datetime"], utc=True)
    df_metrics["Datetime UTC"] = pd.to_datetime(df_metrics["Datetime"], utc=True)

    df_rad = df_rad.drop(columns="Datetime")
    df_metrics = df_metrics.drop(columns="Datetime")

    raw = df_rad.merge(df_metrics, on="Datetime UTC", how="left")
    raw = raw.sort_values("Datetime UTC").reset_index(drop=True)

    for suffix in ICON2_BACKUP_SUFFIXES:
        icon1_col = f"{MODEL} {suffix}"
        icon2_col = f"ICON2 {suffix}"

        if icon1_col in raw.columns and icon2_col in raw.columns:
            raw[icon1_col] = raw[icon1_col].combine_first(raw[icon2_col])

    swissgrid_path = DOWNLOADS_DIR / "control-area-balance-2026.csv"

    if swissgrid_path.exists():
        df_imbalance = pd.read_csv(swissgrid_path, sep=";", encoding="utf-8-sig")

        if {"Date Time [UTC]", "Total System Imbalance"}.issubset(df_imbalance.columns):
            df_imbalance = df_imbalance[["Date Time [UTC]", "Total System Imbalance"]].copy()
            df_imbalance["Datetime UTC"] = pd.to_datetime(
                df_imbalance["Date Time [UTC]"],
                dayfirst=True,
                utc=True,
            )
            df_imbalance = df_imbalance.drop(columns="Date Time [UTC]")
            raw = raw.merge(df_imbalance, on="Datetime UTC", how="left")

    if "Total System Imbalance" not in raw.columns:
        raw["Total System Imbalance"] = np.nan

    raw[f"{MODEL} range"] = raw[f"{MODEL} max"] - raw[f"{MODEL} min"]

    raw[f"{MODEL} horizon spread"] = raw[PREV_DAY_COLS].std(axis=1)

    raw[f"{MODEL} update d1_to_d0"] = raw[f"{MODEL} prev day 0"] - raw[f"{MODEL} prev day 1"]
    raw[f"{MODEL} update d2_to_d1"] = raw[f"{MODEL} prev day 1"] - raw[f"{MODEL} prev day 2"]
    raw[f"{MODEL} update d3_to_d2"] = raw[f"{MODEL} prev day 2"] - raw[f"{MODEL} prev day 3"]
    raw[f"{MODEL} forecast update total abs"] = (
        raw[f"{MODEL} update d1_to_d0"].abs()
        + raw[f"{MODEL} update d2_to_d1"].abs()
        + raw[f"{MODEL} update d3_to_d2"].abs()
    )

    if ERROR_COL not in raw.columns:
        raw[ERROR_COL] = raw["MeteoSwiss stations"] - raw[f"{MODEL} prev day 0"]

    raw[ABS_ERROR_COL] = raw[ERROR_COL].abs()

    return raw


def predict_percentile_risk(df_features_target, df_features_hist, fit_result):
    output_col = fit_result["output_col"]
    feature_cols = fit_result["feature_cols"]
    model_info = fit_result["hourly_model_info"]
    weights = np.asarray(model_info["weights"], dtype=float)

    x_hist = df_features_hist[feature_cols].copy()

    if fit_result.get("use_rank_features", False):
        x_hist = x_hist.rank(method="average", pct=True)

    hist_raw_risk = x_hist.values @ weights

    if "intercept" in model_info:
        hist_raw_risk += float(model_info["intercept"])

    hist_raw_risk = np.sort(hist_raw_risk[np.isfinite(hist_raw_risk)])

    x_target = df_features_target[feature_cols].copy()

    if fit_result.get("use_rank_features", False):
        x_target_ranked = x_target.copy()

        for col in feature_cols:
            hist_values = df_features_hist[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            hist_values = np.sort(hist_values)
            target_values = x_target[col].to_numpy(dtype=float)

            x_target_ranked[col] = [
                np.searchsorted(hist_values, val, side="right") / len(hist_values)
                if len(hist_values) > 0 and np.isfinite(val)
                else np.nan
                for val in target_values
            ]

        x_target = x_target_ranked

    target_raw_risk = x_target.values @ weights

    if "intercept" in model_info:
        target_raw_risk += float(model_info["intercept"])

    if len(hist_raw_risk) == 0:
        return pd.Series(np.nan, index=df_features_target.index)

    return pd.Series(
        [
            100 * np.searchsorted(hist_raw_risk, val, side="right") / len(hist_raw_risk)
            if np.isfinite(val)
            else np.nan
            for val in target_raw_risk
        ],
        index=df_features_target.index,
    )


def fit_model(df_features):
    return fit_risk_score_from_features(
        df_features=df_features,
        feature_cols=FEATURE_COLS,
        metric_cols=[ERROR_COL],
        model=MODEL,
        start_date=str(df_features["Datetime UTC"].min().date()),
        end_date=str(df_features["Datetime UTC"].max().date()),
        ghi_threshold=GHI_THRESHOLD,
        figsize=(10, 5),
        use_rank_features=True,
        nonnegative_weights=True,
        verbose=False,
        top_risk_penalty_weight=0.5,
        top_quantile=0.9,
    )


def valid_fit_rows(df):
    cols = ["Datetime UTC", ERROR_COL] + FEATURE_COLS
    out = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    out = out[out[FEATURE_COLS].abs().sum(axis=1) > 0].copy()
    return out


def load_existing_output():
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(OUTPUT_PATH)

    if "Datetime UTC" in df.columns:
        df["Datetime UTC"] = pd.to_datetime(df["Datetime UTC"], utc=True)

    return df


def append_output(rows):
    new_df = pd.DataFrame(rows)

    if new_df.empty:
        return load_existing_output()

    if OUTPUT_PATH.exists():
        old_df = pd.read_csv(OUTPUT_PATH)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined = combined.drop_duplicates(subset=["Datetime UTC", "model"], keep="last")
    combined["Datetime UTC"] = pd.to_datetime(combined["Datetime UTC"], utc=True)
    combined = combined.sort_values(["Datetime UTC", "model"]).reset_index(drop=True)
    combined["Datetime UTC"] = combined["Datetime UTC"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    return combined


def rows_for_day(raw, target_day):
    target_start = pd.Timestamp(target_day).tz_convert("UTC")
    target_end = target_start + pd.Timedelta(days=1)
    hist_start = pd.to_datetime(HIST_START_DATE, utc=True)

    df_hist = raw[
        (raw["Datetime UTC"] >= hist_start)
        & (raw["Datetime UTC"] < target_start)
    ].copy()
    df_target = raw[
        (raw["Datetime UTC"] >= target_start)
        & (raw["Datetime UTC"] < target_end)
    ].copy()
    df_including_day = raw[
        (raw["Datetime UTC"] >= hist_start)
        & (raw["Datetime UTC"] < target_end)
    ].copy()

    output_source_cols = [
        "Datetime UTC",
        "MeteoSwiss stations",
        "Total System Imbalance",
        *PREV_DAY_COLS,
        *METRIC_COLS,
        ABS_ERROR_COL,
        *EXTRA_OUTPUT_COLS,
        *FEATURE_COLS,
    ]
    target_out = df_target[output_source_cols].copy()
    target_out = target_out.replace([np.inf, -np.inf], np.nan)
    df_features_target = target_out.dropna(subset=FEATURE_COLS).copy()
    df_fit_hist = valid_fit_rows(df_hist)
    df_fit_including_day = valid_fit_rows(df_including_day)
    target_has_actual_error = target_out[ERROR_COL].notna().any()

    target_out[PREDICTED_RISK_COL] = np.nan
    target_out[FITTED_RISK_COL] = np.nan
    target_out["q95_error_band"] = np.nan

    prediction_performance = np.nan
    fitted_performance = np.nan

    if len(df_fit_hist) >= max(10, len(FEATURE_COLS) + 3) and not df_features_target.empty:
        fit_result_pred, _ = fit_model(df_fit_hist)

        df_target_q95 = compute_future_quantile_error_band(
            df_error_hist=df_fit_hist[["Datetime UTC", ERROR_COL]],
            df_features_hist=df_fit_hist[["Datetime UTC"] + FEATURE_COLS],
            df_features_future=df_features_target[["Datetime UTC"] + FEATURE_COLS],
            fit_result=fit_result_pred,
            error_col=ERROR_COL,
            risk_window=RISK_WINDOW,
            quantile=QUANTILE,
            min_samples=MIN_SAMPLES,
            smooth=True,
            enforce_monotonic=True,
        )
        df_target_q95[PREDICTED_RISK_COL] = predict_percentile_risk(df_target_q95, df_fit_hist, fit_result_pred)

        target_out = target_out.drop(columns=[PREDICTED_RISK_COL, "q95_error_band"]).merge(
            df_target_q95[["Datetime UTC", PREDICTED_RISK_COL, "q95_error_band"]],
            on="Datetime UTC",
            how="left",
        )

        prediction_performance = performance_percent(
            target_out[ABS_ERROR_COL],
            target_out[PREDICTED_RISK_COL],
        )

    fitted_target = pd.DataFrame(columns=["Datetime UTC", FITTED_RISK_COL])

    if target_has_actual_error and len(df_fit_including_day) >= max(10, len(FEATURE_COLS) + 3):
        fit_result_including_day, plot_context_including_day = fit_model(df_fit_including_day)
        output_col = fit_result_including_day["output_col"]
        fitted_all = plot_context_including_day["df_fit"][["Datetime UTC", ERROR_COL, output_col]].copy()
        fitted_all[ABS_ERROR_COL] = fitted_all[ERROR_COL].abs()
        fitted_performance = performance_percent(fitted_all[ABS_ERROR_COL], fitted_all[output_col])
        fitted_target = fitted_all[
            (fitted_all["Datetime UTC"] >= target_start)
            & (fitted_all["Datetime UTC"] < target_end)
        ][["Datetime UTC", output_col]].rename(columns={output_col: FITTED_RISK_COL})

    target_out = target_out.drop(columns=[FITTED_RISK_COL]).merge(fitted_target, on="Datetime UTC", how="left")
    target_out["q95_lower"] = (target_out[f"{MODEL} prev day 0"] - target_out["q95_error_band"]).clip(lower=0)
    target_out["q95_upper"] = target_out[f"{MODEL} prev day 0"] + target_out["q95_error_band"]

    rows = []
    run_timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    for _, row in target_out.iterrows():
        out = {
            "Datetime UTC": pd.Timestamp(row["Datetime UTC"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date": target_start.strftime("%Y-%m-%d"),
            "model": MODEL,
            "run_timestamp_utc": run_timestamp,
            "MeteoSwiss stations": clean_float(row["MeteoSwiss stations"]),
            ABS_ERROR_COL: clean_float(row[ABS_ERROR_COL]),
            "Total System Imbalance": clean_float(row["Total System Imbalance"]),
            PREDICTED_RISK_COL: clean_float(row[PREDICTED_RISK_COL]),
            FITTED_RISK_COL: clean_float(row[FITTED_RISK_COL]),
            "q95_error_band": clean_float(row["q95_error_band"]),
            "q95_lower": clean_float(row["q95_lower"]),
            "q95_upper": clean_float(row["q95_upper"]),
            "predicted_kendall_tau_percent_day": clean_float(prediction_performance),
            "fitted_kendall_tau_percent_including_day": clean_float(fitted_performance),
        }

        for col in PREV_DAY_COLS + METRIC_COLS + EXTRA_OUTPUT_COLS:
            out[col] = clean_float(row[col])

        for col in FEATURE_COLS:
            out[col] = clean_float(row[col])

        rows.append(out)

    return rows


def main():
    raw = load_raw_data()
    existing = load_existing_output()

    existing_by_day = {}

    if not existing.empty:
        existing_by_day = {
            day: group.copy()
            for day, group in existing.groupby(existing["Datetime UTC"].dt.floor("D").dt.strftime("%Y-%m-%d"))
        }

    target_days = raw["Datetime UTC"].dt.floor("D").drop_duplicates().sort_values()

    all_rows = []

    for target_day in target_days:
        day_key = target_day.strftime("%Y-%m-%d")

        raw_day = raw[raw["Datetime UTC"].dt.floor("D").dt.strftime("%Y-%m-%d") == day_key]
        existing_day = existing_by_day.get(day_key)
        raw_day_has_actual = raw_day[ERROR_COL].notna().any()
        existing_day_is_complete = False

        if existing_day is not None and not existing_day.empty:
            has_prediction = existing_day[PREDICTED_RISK_COL].notna().any()
            has_q95 = existing_day["q95_error_band"].notna().any()
            has_actual_outputs = (
                existing_day[ERROR_COL].notna().any()
                and existing_day[FITTED_RISK_COL].notna().any()
                and existing_day["predicted_kendall_tau_percent_day"].notna().any()
                and existing_day["fitted_kendall_tau_percent_including_day"].notna().any()
            )
            has_clean_future_outputs = (
                existing_day[ERROR_COL].isna().all()
                and existing_day[FITTED_RISK_COL].isna().all()
                and existing_day["predicted_kendall_tau_percent_day"].isna().all()
                and existing_day["fitted_kendall_tau_percent_including_day"].isna().all()
            )
            existing_day_is_complete = has_prediction and has_q95 and (
                has_actual_outputs if raw_day_has_actual else has_clean_future_outputs
            )

        if existing_day_is_complete:
            continue

        try:
            rows = rows_for_day(raw, target_day)
            all_rows.extend(rows)
            print(f"{day_key}: prepared {len(rows)} row(s)")
        except Exception as exc:
            print(f"{day_key}: skipped ({exc})")

    output = append_output(all_rows)

    print(f"Saved {len(all_rows)} new row(s) to {OUTPUT_PATH}")
    print(f"Output now has {len(output)} row(s).")


if __name__ == "__main__":
    main()
