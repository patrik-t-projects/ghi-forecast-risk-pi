# PV Forecast Risk on Raspberry Pi

This project downloads historical and forecast GHI data (model based and actual measured data), estimates GHI forecast error, builds a forecast risk report, and emails the generated HTML report, all autoated rom a Raspberry Pi.

## What the Pipeline Does

1. Gets newest data from MeteoSwiss weather stations (https://www.meteoschweiz.admin.ch/service-und-publikationen/service/open-data.html).
2. Gets newest data from Open-Meteo ensemble and previous-run forecast data (https://open-meteo.com/en/docs).
3. Calculates GHI forecast error.
4. Fits a risk model on historic data of GHI forecast data using ensemble model runs etc. to predict the GHI gorecast risk.
5. Builds an interactive HTML report.
6. Sends the report by email.

The daily scheduler in `main.py` waits for 08:00 Europe/Zurich time and runs the full pipeline once per day.

## Project Structure

- `main.py` - daily entry point and pipeline orchestration.
- `config_example.py` - safe template showing the required configuration fields.
- `email_report.py` - sends the HTML report by email.
- `MeteoSwiss_Stations_Weather_data_API.py` - downloads MeteoSwiss station data.
- `API_OpenMeteo_WeatherStations.py` - downloads Open-Meteo weather station data.
- `PV_Forecast_uncertainty_calc.py` - calculates forecast uncertainty metrics.
- `PV_forecast_risk_plot_html.py` - creates the interactive HTML risk report.
- `PV_forecast_risk_functions.py`, `PV_forecast_uncertainty_functions.py`, `OpenMeteo_API_functions.py` - helper functions.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The version ranges in `requirements.txt` keep installs on compatible major versions while still allowing patch and minor updates.

Then edit `config_example.py` for your local paths, email sender, email recipient, and SMTP app password.

## Configuration

Required values:

- `DATA_DIR` - folder containing input and generated CSV data.
- `REPORTS_DIR` - folder where HTML reports and tracking files are written.
- `DOWNLOADS_DIR` - folder containing Swissgrid download files.
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` - SMTP settings.
- `EMAIL_FROM`, `EMAIL_TO` - report sender and recipient.

## Run

Run the daily scheduler:

```bash
python main.py
```

Run individual scripts directly only when you want to update or debug one pipeline step.

## GitHub Upload Notes

The `.gitignore` file tells Git to ignore private local settings, Python cache files, virtual environments, generated CSV/HTML outputs, logs, and editor/OS metadata. It helps prevent accidental commits of credentials, machine-specific paths, bulky generated data, and files that can be recreated.

## NordVPN Note

`OpenMeteo_API_functions.py` can change NordVPN country after repeated Open-Meteo request failures. The country names are selected from a hardcoded list, and NordVPN is called with an argument list instead of through a shell. That keeps the command safer if the country source is changed later, because user-controlled text would not be interpreted as shell syntax.
