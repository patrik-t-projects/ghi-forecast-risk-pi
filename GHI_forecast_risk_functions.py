from scipy.optimize import differential_evolution
import numpy as np
import pandas as pd


def fit_risk_score(
    df,
    feature_cols,
    target_error_cols,
    output_col="fitted risk score",
    use_rank_features=True,
    nonnegative_weights=True,
    verbose=True,
    random_state=42,
    maxiter=300,
    top_risk_penalty_weight=0.5,
    top_quantile=0.90,
):
    """
    Fit a forecast-risk score by directly maximizing a rank-based objective.

    Model:

        risk_score = w1*x1 + w2*x2 + ... + wn*xn

    Objective:

        score = Kendall-tau + top_risk_penalty_weight * top_hit_rate

    where top_hit_rate is the fraction of true high-error points that are also
    classified as high-risk points.

    The target is:

        mean(abs(target_error_cols))

    If several horizons are given, the model fits to the mean absolute error
    across those horizons.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns and target error columns.

    feature_cols : list[str]
        Forecast-only risk metric columns.

    target_error_cols : list[str]
        Forecast error columns. The function uses abs(error). If several
        columns are passed, their mean absolute error is used as target.

    output_col : str
        Name of the fitted risk-score column.

    use_rank_features : bool
        If True, feature values are converted to percentile ranks before fitting.

    nonnegative_weights : bool
        If True, fitted weights are constrained to be nonnegative.

    verbose : bool
        If True, print fitted model information.

    random_state : int
        Random seed for differential evolution.

    maxiter : int
        Maximum number of differential evolution iterations.

    top_risk_penalty_weight : float
        Weight of the top-hit-rate term in the objective.
        0.0 means pure Kendall-tau optimization.

    top_quantile : float
        Quantile used to define high-error and high-risk points.
        Example: 0.90 means top 10%.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with output_col added.

    model_info : dict
        Fitted weights and training performance.
    """

    df_out = df.copy()

    # -------------------------
    # Checks
    # -------------------------
    missing_features = [c for c in feature_cols if c not in df_out.columns]
    missing_targets = [c for c in target_error_cols if c not in df_out.columns]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if missing_targets:
        raise ValueError(f"Missing target error columns: {missing_targets}")

    if not (0.0 < top_quantile < 1.0):
        raise ValueError("top_quantile must be between 0 and 1.")

    # -------------------------
    # Build target
    # -------------------------
    target_col = "__risk_model_target_abs_error__"
    df_out[target_col] = df_out[target_error_cols].abs().mean(axis=1)

    fit_cols = feature_cols + [target_col]
    fit_df = df_out[fit_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    # Remove rows where all features are zero
    fit_df = fit_df[(fit_df[feature_cols].abs().sum(axis=1) > 0)].copy()

    # if len(fit_df) < len(feature_cols) + 2:
    #     raise ValueError(
    #         f"Not enough valid rows to fit model. "
    #         f"Valid rows: {len(fit_df)}, features: {len(feature_cols)}"
    #     )

    # -------------------------
    # Feature preprocessing
    # -------------------------
    X_raw = fit_df[feature_cols].astype(float)
    y = fit_df[target_col].astype(float)

    if use_rank_features:
        # Convert each feature to percentile rank in [0, 1].
        # This makes the model naturally suited to rank-based evaluation.
        X_proc = X_raw.rank(method="average", pct=True)
    else:
        X_proc = X_raw.copy()

    X = X_proc.values
    y_arr = y.values

    n_features = len(feature_cols)

    # -------------------------
    # Helper: normalize weights
    # -------------------------
    def normalize_weights(params):
        params = np.asarray(params, dtype=float)

        if nonnegative_weights:
            params = np.maximum(params, 0.0)

        weight_sum = params.sum()

        if weight_sum <= 1e-12:
            return np.ones_like(params) / len(params)

        return params / weight_sum

    # -------------------------
    # Helper: top-hit-rate
    # -------------------------
    def top_hit_rate(risk, error, q=0.90):
        """
        Fraction of true high-error points that are also high-risk points.

        Example:
            q = 0.90:
            - high-error points = top 10% error points
            - high-risk points  = top 10% risk points
        """

        risk = np.asarray(risk, dtype=float)
        error = np.asarray(error, dtype=float)

        valid = np.isfinite(risk) & np.isfinite(error)
        risk = risk[valid]
        error = error[valid]

        if len(risk) < 5:
            return 0.0

        risk_thr = np.quantile(risk, q)
        error_thr = np.quantile(error, q)

        high_risk = risk >= risk_thr
        high_error = error >= error_thr

        n_high_error = high_error.sum()

        if n_high_error == 0:
            return 0.0

        return (high_risk & high_error).sum() / n_high_error

    # -------------------------
    # Objective:
    # maximize Kendall tau + weighted top-hit-rate
    # -------------------------
    def objective(params):
        weights = normalize_weights(params)
        risk_raw = X @ weights

        if np.nanstd(risk_raw) < 1e-12:
            return 1e6

        tau = pd.Series(risk_raw).corr(pd.Series(y_arr), method="kendall")

        if not np.isfinite(tau):
            return 1e6

        hit_rate = top_hit_rate(
            risk=risk_raw,
            error=y_arr,
            q=top_quantile,
        )

        score = tau + top_risk_penalty_weight * hit_rate

        # differential_evolution minimizes
        return -score

    # -------------------------
    # Optimize weights
    # -------------------------
    if nonnegative_weights:
        bounds = [(0.0, 1.0)] * n_features
    else:
        bounds = [(-1.0, 1.0)] * n_features

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=random_state,
        maxiter=maxiter,
        polish=True,
        updating="immediate",
        workers=1,
    )

    if not result.success and verbose:
        print(f"Warning: optimizer did not fully converge: {result.message}")

    weights = normalize_weights(result.x)

    # -------------------------
    # Apply model to all rows
    # -------------------------
    X_all_raw = df_out[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)

    if use_rank_features:
        # For now: rank within the provided df.
        # Later, for future prediction, use historical reference percentiles.
        X_all_proc = X_all_raw.rank(method="average", pct=True)
    else:
        X_all_proc = X_all_raw.copy()

    risk_raw_all = X_all_proc.values @ weights

    risk_series = pd.Series(risk_raw_all, index=df_out.index)

    # Output percentile-style risk score in [0, 100]
    df_out[output_col] = 100 * risk_series.rank(method="average", pct=True)

    # -------------------------
    # Training performance
    # -------------------------
    train_risk = X @ weights

    train_tau = pd.Series(train_risk).corr(pd.Series(y_arr), method="kendall")

    train_top_hit_rate = top_hit_rate(
        risk=train_risk,
        error=y_arr,
        q=top_quantile,
    )

    error_rank = pd.Series(y_arr).rank(method="average", pct=True)
    risk_rank = pd.Series(train_risk).rank(method="average", pct=True)
    rank_rmse = np.sqrt(((risk_rank - error_rank) ** 2).mean())

    model_info = {
        "output_col": output_col,
        "feature_cols": feature_cols,
        "target_error_cols": target_error_cols,
        "target_definition": "mean absolute error across target_error_cols",
        "weights": pd.Series(weights, index=feature_cols),
        "use_rank_features": use_rank_features,
        "nonnegative_weights": nonnegative_weights,
        "training_kendall_tau": train_tau,
        "training_rank_rmse_percent": 100 * rank_rmse,
        "training_top_hit_rate": train_top_hit_rate,
        "top_risk_penalty_weight": top_risk_penalty_weight,
        "top_quantile": top_quantile,
        "n_fit_rows": len(fit_df),
        "optimizer_success": result.success,
        "optimizer_message": result.message,
    }

    if verbose:
        print("\nFitted rank-optimized risk model")
        print("--------------------------------")
        print(f"Output column: {output_col}")
        print(f"Target columns: {target_error_cols}")
        print(f"Number of fit rows: {len(fit_df)}")
        print(f"Training Kendall tau: {train_tau:.4f}")
        print(f"Training Rank RMSE: {100 * rank_rmse:.2f}%")
        print(f"Training Top Hit Rate @ q={top_quantile:.2f}: {train_top_hit_rate:.3f}")
        print(f"Top-risk penalty weight: {top_risk_penalty_weight:.3f}")
        print("\nWeights:")
        print(model_info["weights"].sort_values(ascending=False))

    df_out = df_out.drop(columns=[target_col])

    return df_out, model_info


def fit_risk_score_from_features(
    df_features,
    feature_cols,
    metric_cols,
    model="ICON1",
    start_date="2026-03-21",
    end_date="2026-05-08",
    ghi_threshold=0,
    figsize=(10, 5),
    horizon_colors=None,
    use_rank_features=True,
    nonnegative_weights=True,
    verbose=True,
    top_risk_penalty_weight=0.5,
    top_quantile=0.9,
):
    """
    Fits the risk score model using an already prepared df_features.

    Inputs
    ------
    df_features:
        Output dataframe from build_risk_feature_dataframe.

    feature_cols:
        Risk feature columns used as model inputs.
        Important: should NOT include the fitted risk score column.

    metric_cols:
        Forecast error columns used as fitting targets.

    Returns
    -------
    fit_result:
        Clean fitted model package for reuse in another .py file.

    plot_context:
        Dictionary containing everything needed for the plotting section.
    """

    df_features = df_features.copy()

    if horizon_colors is None:
        horizon_colors = {
            f"MeteoSwiss - {model}_d0": "tab:green",
            f"MeteoSwiss - {model}_d1": "tab:orange",
            f"MeteoSwiss - {model}_d2": "tab:red",
            f"MeteoSwiss - {model}_d3": "tab:purple",
        }

    output_col = f"{model} fitted risk score"

    # ====================================
    # Fit risk score
    # ====================================
    df_fit, hourly_model_info = fit_risk_score(
        df=df_features,
        feature_cols=feature_cols,
        target_error_cols=metric_cols,
        output_col=output_col,
        use_rank_features=use_rank_features,
        nonnegative_weights=nonnegative_weights,
        verbose=verbose,
        top_risk_penalty_weight=top_risk_penalty_weight,
        top_quantile=top_quantile,
    )

    # ==================================================
    # Risk columns used for plotting
    # IMPORTANT: includes fitted risk score.
    # ==================================================
    risk_metric_cols_for_plot = feature_cols + [output_col]

    df_plot = df_fit.copy()

    # ==================================================
    # Output 1:
    # Clean fitted model result for reuse in another .py
    # ==================================================
    fit_result = {
        "model": model,
        "output_col": output_col,
        "feature_cols": feature_cols,
        "target_error_cols": metric_cols,
        "hourly_model_info": hourly_model_info,
        "use_rank_features": use_rank_features,
        "nonnegative_weights": nonnegative_weights,
        "top_risk_penalty_weight": top_risk_penalty_weight,
        "top_quantile": top_quantile,
    }

    # ==================================================
    # Output 2:
    # Everything needed for the plotting section
    # ==================================================
    plot_context = {
        "df_plot": df_plot,
        "df_fit": df_fit,
        "df_features": df_features,
        "feature_cols": feature_cols,
        "risk_metric_cols": risk_metric_cols_for_plot,
        "metric_cols": metric_cols,
        "horizon_colors": horizon_colors,
        "MODEL": model,
        "START_DATE": start_date,
        "END_DATE": end_date,
        "GHI_threshold": ghi_threshold,
        "FIGSIZE": figsize,
        "output_col": output_col,
    }

    return fit_result, plot_context


def compute_future_quantile_error_band(
    df_error_hist,
    df_features_hist,
    df_features_future,
    fit_result,
    error_col,
    risk_window=0.1,
    quantile=0.95,
    min_samples=5,
    smooth=True,
    enforce_monotonic=True,
):
    """
    Compute future risk scores and convert them into q95 forecast-error bands
    using local historical risk-score windows.

    Parameters
    ----------
    df_error_hist : pd.DataFrame
        Historical dataframe containing the realized forecast error.
        Must contain 'Datetime UTC' and error_col.

    df_features_hist : pd.DataFrame
        Historical feature dataframe used for fitting.
        Must contain 'Datetime UTC'. If the fitted risk-score column is missing,
        it will be computed from fit_result.

    df_features_future : pd.DataFrame
        Future feature dataframe. The function computes the future risk score here.

    fit_result : dict
        Output from fit_risk_score_from_features.

    error_col : str
        Historical forecast-error column, e.g. 'MeteoSwiss - ICON1_d0'.
        The q95 band is computed from abs(error_col).

    risk_window : float
        Local risk window half-width, e.g. 10 means R* Â± 10.

    quantile : float
        Error quantile to estimate. For 95% band use 0.95.

    min_samples : int
        Minimum number of historical samples needed inside the local window.
        If too few are found, the nearest historical samples are used instead.

    smooth : bool
        If True, smooths the future q95 bands over increasing risk score.

    enforce_monotonic : bool
        If True, forces higher risk scores to have non-decreasing error bands.

    Returns
    -------
    df_features_future_out : pd.DataFrame
        Future feature dataframe including:
        - fitted risk score column
        - q95_error_band
    """

    import numpy as np
    import pandas as pd

    output_col = fit_result["output_col"]
    feature_cols = fit_result["feature_cols"]
    model_info = fit_result["hourly_model_info"]

    # -----------------------------
    # Helper: apply fitted risk model
    # -----------------------------
    def apply_risk_model(df_features):
        df_out = df_features.copy()

        X = df_out[feature_cols].copy()

        # Same transformation as used during fitting
        if fit_result.get("use_rank_features", False):
            X = X.rank(method="average", pct=True)

        weights = np.asarray(model_info["weights"], dtype=float)

        df_out[output_col] = X.values @ weights

        if "intercept" in model_info:
            df_out[output_col] += float(model_info["intercept"])

        return df_out

    # -----------------------------
    # Historical risk scores
    # -----------------------------
    if output_col not in df_features_hist.columns:
        df_features_hist_risk = apply_risk_model(df_features_hist)
    else:
        df_features_hist_risk = df_features_hist.copy()

    # -----------------------------
    # Future risk scores
    # -----------------------------
    df_features_future_out = apply_risk_model(df_features_future)

    # -----------------------------
    # Join historical risk + error
    # -----------------------------
    hist = df_features_hist_risk[["Datetime UTC", output_col]].merge(
        df_error_hist[["Datetime UTC", error_col]],
        on="Datetime UTC",
        how="inner",
    )

    hist = hist.dropna(subset=[output_col, error_col]).copy()
    hist["abs_error"] = hist[error_col].abs()

    hist_risk = hist[output_col].to_numpy(dtype=float)
    hist_abs_error = hist["abs_error"].to_numpy(dtype=float)

    if len(hist) == 0:
        raise ValueError("No valid historical samples found after merging risk scores and forecast errors.")

    # -----------------------------
    # Local-window q95 calibration
    # -----------------------------
    future_risk = df_features_future_out[output_col].to_numpy(dtype=float)
    q_values = []

    for r_star in future_risk:
        mask = np.abs(hist_risk - r_star) <= risk_window

        # Fallback if local window has too few points:
        # use nearest historical risk samples.
        if mask.sum() < min_samples:
            nearest_idx = np.argsort(np.abs(hist_risk - r_star))[:min_samples]
            local_errors = hist_abs_error[nearest_idx]
        else:
            local_errors = hist_abs_error[mask]

        q_values.append(np.quantile(local_errors, quantile))

    df_features_future_out["q95_error_band_raw"] = q_values

    # -----------------------------
    # Optional smoothing / monotonicity
    # -----------------------------
    if smooth or enforce_monotonic:
        tmp = df_features_future_out[[output_col, "q95_error_band_raw"]].copy()
        tmp["_original_index"] = np.arange(len(tmp))
        tmp = tmp.sort_values(output_col).reset_index(drop=True)

        band = tmp["q95_error_band_raw"].copy()

        if smooth:
            # Small rolling smoothing in risk-score order.
            # Keeps the local-window idea but reduces sample noise.
            band = band.rolling(window=5, center=True, min_periods=1).median()

        if enforce_monotonic:
            # Risk score should order uncertainty:
            # higher risk should not produce a lower q95 band.
            band = np.maximum.accumulate(band.to_numpy(dtype=float))

        tmp["q95_error_band"] = band

        tmp = tmp.sort_values("_original_index")
        df_features_future_out["q95_error_band"] = tmp["q95_error_band"].to_numpy()

    else:
        df_features_future_out["q95_error_band"] = df_features_future_out["q95_error_band_raw"]

    return df_features_future_out


# ===================================
# Fill missing ICON1 future data with ICON2
# ===================================

def fill_model_gaps_with_backup_model(
    df,
    primary_model="ICON1",
    backup_model="ICON2",
    cols_to_fill=None,
):
    """
    Fills missing primary-model columns with backup-model columns.

    Example:
        ICON1 std is filled with ICON2 std where ICON1 std is NaN.

    The output keeps the primary-model column names.
    """

    df_out = df.copy()

    if cols_to_fill is None:
        cols_to_fill = [
            # "prev day 0",
            # "prev day 1",
            # "prev day 2",
            # "prev day 3",
            "mean",
            "max",
            "min",
            "std",
            "ens spread",
            "CC ens spread",
        ]

    for suffix in cols_to_fill:
        primary_col = f"{primary_model} {suffix}"
        backup_col = f"{backup_model} {suffix}"

        if primary_col not in df_out.columns:
            raise ValueError(f"Missing primary column: {primary_col}")

        if backup_col not in df_out.columns:
            raise ValueError(f"Missing backup column: {backup_col}")

        df_out[primary_col] = df_out[primary_col].fillna(df_out[backup_col])

    return df_out
