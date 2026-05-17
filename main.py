from datetime import date, datetime, timedelta
import time
from zoneinfo import ZoneInfo
import runpy
import pandas as pd
from config import EMAIL_TO, data_path, report_path, script_path
from email_report import send_html_report_email


def run_pipeline():
    OPEN_METEO_FULL_FILE = data_path("Data_OpenMeteo_ENSEMBLE_PrevRuns_weather_stations_2026.csv")
    METEOSWISS_WEATHER_FILE = data_path("meteoswiss_gre_hourly_2026.csv")


    def read_datetime_csv(path, sep=";"):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Copy the data folder to this project or update DATA_DIR in config.py."
            )

        df = pd.read_csv(path, sep=sep)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df


    def zurich_to_utc_string(value, fmt):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("Europe/Zurich")
        else:
            ts = ts.tz_convert("Europe/Zurich")
        return ts.tz_convert("UTC").strftime(fmt)


    old = read_datetime_csv(OPEN_METEO_FULL_FILE)


    # ==============================================
    # MeteoSwiss weather stations API call
    # ==============================================
    print("\nChecking MeteoSwiss weather station data...")

    yesterday = (datetime.now() - timedelta(days=1)).date()
    df_weather = read_datetime_csv(METEOSWISS_WEATHER_FILE)

    min_date = df_weather["Datetime"].dt.date.min()
    max_date = df_weather["Datetime"].dt.date.max()

    print(f"Existing MeteoSwiss file date range: {min_date} to {max_date}")

    if max_date >= yesterday:
        print(f"Existing file already includes yesterday ({yesterday}). Skipping API call.")
    else:
        print(f"Existing file only goes up to {max_date}. Running API script...")
        print("\nRunning MeteoSwiss_Stations_Weather_data_API.py...")
        runpy.run_path(str(script_path("MeteoSwiss_Stations_Weather_data_API.py")))


    # ==============================================
    # OpenMeteo API call for weather stations
    # ==============================================
    print("\nRunning API_OpenMeteo_WeatherStations.py...")

    last_date = old["Datetime"].max().date()
    up_to_date = date.today() + timedelta(days=5)

    if last_date >= up_to_date:
        print("OpenMeteo data is already up to date. Skipping API call.")
        updated = old
    else:
        START_DATE = date.today().isoformat()
        END_DATE = (date.today() + timedelta(days=5)).isoformat()

        output_stem = f"Data_OpenMeteo_ENSEMBLE_PrevRuns_weather_stations_{START_DATE}_{END_DATE}"
        globals_dict = {
            "START_DATE": START_DATE,
            "END_DATE": END_DATE,
            "OUTPUT_FILE": output_stem,
        }

        print(f"Existing data range: {old['Datetime'].min()} to {old['Datetime'].max()}")
        print(f"Fetching new data from {START_DATE} to {END_DATE}")

        runpy.run_path(str(script_path("API_OpenMeteo_WeatherStations.py")), init_globals=globals_dict)
            

        # ==============================================
        # Append new OpenMeteo data to full 2026 file
        # ==============================================
        new_path = data_path(f"{output_stem}.csv")

        new = pd.read_csv(new_path, sep=";")
        new["Datetime"] = pd.to_datetime(new["Datetime"])

        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE) + pd.Timedelta(days=1)
        new = new[(new["Datetime"] >= start_dt) & (new["Datetime"] < end_dt)].copy()

        updated = pd.concat([old, new], ignore_index=True)
        updated = updated.drop_duplicates(subset=["station_abbr", "Datetime"], keep="last")
        updated = updated.sort_values(["station_abbr", "Datetime"]).reset_index(drop=True)
        updated.to_csv(OPEN_METEO_FULL_FILE, sep=";", index=False, encoding="utf-8")

        print(f"Updated full 2026 file saved to: {OPEN_METEO_FULL_FILE}")
        print(f"New date range: {updated['Datetime'].min()} to {updated['Datetime'].max()}")


    # ==============================================
    # Run PV_Forecast_uncertainty_calc
    # ==============================================
    print("\nRunning PV_Forecast_uncertainty_calc.py...")

    START_DATE = zurich_to_utc_string(updated["Datetime"].min(), "%Y-%m-%d %H:%M:%S")
    END_DATE = zurich_to_utc_string(updated["Datetime"].max(), "%Y-%m-%d %H:%M:%S")

    print(f"Existing data range: {updated['Datetime'].min()} to {updated['Datetime'].max()}")
    print(f"Calculating from {START_DATE} to {END_DATE}")

    runpy.run_path(
        str(script_path("PV_Forecast_uncertainty_calc.py")),
        init_globals={"START_DATE": START_DATE, "END_DATE": END_DATE},
    )

    # ==============================================
    # Run GHI_forecast_risk_plot_html
    # ==============================================
    print("\nRunning GHI_forecast_risk_plot_html.py...")

    START_DATE = zurich_to_utc_string(updated["Datetime"].min(), "%Y-%m-%d")
    END_DATE = zurich_to_utc_string(updated["Datetime"].max(), "%Y-%m-%d")

    print(f"Existing data range: {updated['Datetime'].min()} to {updated['Datetime'].max()}")
    print(f"Calculating from {START_DATE} to {END_DATE}")

    runpy.run_path(
        str(script_path("GHI_forecast_risk_plot_html.py")),
        init_globals={
            "START_DATE": START_DATE,
            "END_DATE": END_DATE,
            "INCLUDE_SCATTERPLOTS": True,
            "OUTPUT_SUFFIX": "_with_scatterplots",
            "OPEN_BROWSER": False,
        },
    )

    runpy.run_path(
        str(script_path("GHI_forecast_risk_plot_html.py")),
        init_globals={
            "START_DATE": START_DATE,
            "END_DATE": END_DATE,
            "INCLUDE_SCATTERPLOTS": False,
            "OUTPUT_SUFFIX": "_without_scatterplots",
            "OPEN_BROWSER": False,
        },
    )

    time.sleep(5)
    print("\nFinished running all scripts.")


    print("\nSend EMAIL with html report...")

    html_file_paths = [
        report_path(f"GHI_forecast_risk_ICON1_{START_DATE}_to_{END_DATE}_with_scatterplots.html"),
        report_path(f"GHI_forecast_risk_ICON1_{START_DATE}_to_{END_DATE}_without_scatterplots.html"),
    ]

    send_html_report_email(
        to_address=EMAIL_TO,
        html_file_path=html_file_paths,
    )

    print(f"\nSuccessfully sent email to {EMAIL_TO}")



if __name__ == "__main__":
    zurich_tz = ZoneInfo("Europe/Zurich")
    last_successful_run_date = date(2026, 5, 14)

    print("Waiting for 08:00 Europe/Zurich time...")

    while True:
        now = datetime.now(zurich_tz)
        # print(now.hour, now.minute)

        # Start trying from 08:00 onward, once per day until successful
        if (now.hour >= 8 and last_successful_run_date != now.date()):
            print(f"\nIt is {now.strftime('%Y-%m-%d %H:%M:%S %Z')}. Starting daily run...")

            try:
                run_pipeline()

                # Only mark the day as done if the full pipeline succeeded
                last_successful_run_date = now.date()

                print("\nDaily run finished successfully.")

            except Exception as e:
                print(f"\nDaily run failed: {e}")
                print("Will retry in 30 seconds...")

        time.sleep(30)
