import numpy as np
import pandas as pd
import pvlib


def weighted_pearson(x, y, w):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < 3:
        return np.nan

    x = x[m]
    y = y[m]
    w = w[m]

    w_sum = np.sum(w)
    mx = np.sum(w * x) / w_sum
    my = np.sum(w * y) / w_sum

    cov_xy = np.sum(w * (x - mx) * (y - my))
    var_x = np.sum(w * (x - mx) ** 2)
    var_y = np.sum(w * (y - my) ** 2)

    if var_x <= 0 or var_y <= 0:
        return np.nan

    return cov_xy / np.sqrt(var_x * var_y)
# End func


def explainable_power(x, y, ghi_day=None, ghi_min=30.0):
    """
    Fraction of imbalance |y| that can be reasonably attributed
    to PV power error x on a given day, using only PV-active samples.

    x : PV power error [MW]
    y : system imbalance [MW]
    ghi_day : GHI time series [W/m^2] (same length as x, y)
    ghi_min : PV-active threshold [W/m^2]

    Returns
    -------
    P_expl : float in [0, 1]
        Explainable fraction of imbalance
    k_raw : float
        Raw coupling strength (diagnostic)
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Base validity mask
    m = np.isfinite(x) & np.isfinite(y)

    # PV-active mask
    if ghi_day is not None:
        ghi = np.asarray(ghi_day)
        m = m & np.isfinite(ghi) & (ghi >= ghi_min)

    if m.sum() < 3:
        return np.nan, np.nan

    x_m = x[m]
    y_m = y[m]

    denom = np.sum(x_m**2)
    if denom <= 0:
        return np.nan, np.nan

    # Raw statistical coupling (PV-active only)
    k_raw = np.sum(x_m * y_m) / denom

    # Bounded attributable imbalance
    i_pv = np.sign(y_m) * np.minimum(np.abs(k_raw * x_m), np.abs(y_m))

    # Explainable fraction (PV-active imbalance only)
    total = np.sum(np.abs(y_m))
    P_expl = np.sum(np.abs(i_pv)) / total if total > 0 else np.nan

    return P_expl, k_raw
# End func


def driver_contribution(imbalances, metric):

    imbalances = np.asarray(imbalances)
    metric = np.asarray(metric)

    a = np.sum(imbalances * metric) / np.sum(metric ** 2)
    driver_signal = a * metric

    var_driver = np.sum(driver_signal**2)
    var_total = np.sum(imbalances**2)

    return var_driver / var_total
# End func


def compute_rmse(imbalances, metric):
    metric = np.asarray(metric)
    imbalances = np.asarray(imbalances)

    return np.sqrt(np.mean((imbalances - metric)**2))
# End func


# Functions for nice 0 alignement in plots
def limits_for_f0(dmin, dmax, f0, margin=1.05):
    """
    Smallest [ymin,ymax] that contains [dmin,dmax] while placing 0 at fraction f0.
    Adds multiplicative margin to the required total range.
    """
    # Required total range T so that:
    # ymax = (1-f0)*T >= dmax
    # ymin = -f0*T <= dmin
    req_pos = dmax / (1 - f0) if dmax > 0 else 0.0
    req_neg = (-dmin) / f0     if dmin < 0 else 0.0
    T = max(req_pos, req_neg) * margin

    ymin = -f0 * T
    ymax = (1 - f0) * T
    return ymin, ymax, T
# End func


def choose_f0(dLmin, dLmax, dRmin, dRmax, marginL=1.05, marginR=1.05, wL=1.0, wR=1.0):
    """
    Pick f0 in (0,1) that balances both axes so neither gets 'smushed'.
    Objective: minimize max(weighted expansion factors).
    """
    f_grid = np.linspace(0.05, 0.95, 901)

    # data spans used to normalize "expansion"
    spanL = max(1e-12, dLmax - dLmin)
    spanR = max(1e-12, dRmax - dRmin)

    best = None
    for f0 in f_grid:
        _, _, TL = limits_for_f0(dLmin, dLmax, f0, margin=marginL)
        _, _, TR = limits_for_f0(dRmin, dRmax, f0, margin=marginR)

        # Expansion factors (bigger means more whitespace / more smushing risk)
        eL = (TL / spanL) * wL
        eR = (TR / spanR) * wR

        score = max(eL, eR)  # minimize the worst one
        if best is None or score < best[0]:
            best = (score, f0)

    return best[1]  # f0
# End func



def predict_pv_power_from_ghi(df, model, ghi_col="MeteoSwiss stations", lat=46.95, lon=7.44, tz="UTC"):
    """
    Predict PV power from weighted GHI using a trained ML model.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Datetime (index or column) and GHI column
    model : trained ML model (e.g. XGBoost loaded via joblib)
    ghi_col : str
        Column name for weighted GHI
    lat, lon : float
        Location coordinates (default: Bern)
    tz : str
        Timezone (default: UTC)

    Returns
    -------
    pd.Series
        Predicted PV power (same index as input)
    """

    df = df.copy()

    # --------------------------------------------------
    # Ensure Datetime index
    # --------------------------------------------------
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df = df.set_index("Datetime")

    df = df.sort_index()

    # --------------------------------------------------
    # Solar position (pvlib)
    # --------------------------------------------------
    solpos = pvlib.solarposition.get_solarposition(
        time=df.index,
        latitude=lat,
        longitude=lon
    )

    df["solar_zenith"] = solpos["zenith"].values
    df["solar_azimuth"] = solpos["azimuth"].values

    # ------------------------------------------
    # Time features
    # ------------------------------------------
    df["doy"] = df.index.dayofyear
    df["month"] = df.index.month
    df["hour"] = df.index.hour

    # Weekday: 1 (Mon) -> 7 (Sun)
    df["day"] = df.index.dayofweek

    # Afternoon flag
    df["afternoon"] = (df["solar_azimuth"] > 180).astype(int)

    # --------------------------------------------------
    # Feature matrix
    # --------------------------------------------------
    X = pd.DataFrame({
        "GHI_weighted": df[ghi_col].values,
        "month": df["month"].values,
        "day": df["day"].values,
        "hour": df["hour"].values,
        "solar_zenith": df["solar_zenith"].values,
        "solar_azimuth": df["solar_azimuth"].values,
        "doy": df["doy"].values,
        "afternoon": df["afternoon"].values,
    }, index=df.index)

    # Optional: remove night / invalid values
    X["GHI_weighted"] = np.maximum(X["GHI_weighted"], 0)

    # --------------------------------------------------
    # Model prediction
    # --------------------------------------------------
    y_pred = model.predict(X)

    return pd.Series(y_pred, index=df.index, name="pv_power_pred")
# End func
