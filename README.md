# PV Forecast Risk on Raspberry Pi

This project downloads weather and PV forecast data, estimates PV forecast uncertainty, builds a forecast risk report, and emails the generated HTML report from a Raspberry Pi.

## What the Pipeline Does

1. Updates MeteoSwiss radiation station data.
2. Updates Open-Meteo ensemble and previous-run forecast data.
3. Calculates PV forecast uncertainty.
4. Builds an interactive HTML risk report (example can be found here: https://patrik-t-projects.github.io/ghi-forecast-risk-pi/GHI_forecast_risk_example.html)
5. Sends the report by email.

The daily scheduler in `main.py` waits for 08:00 Europe/Zurich time and runs the full pipeline once per day.

## Project Structure

- `main.py` - daily entry point and pipeline orchestration.
- `config.py` - local private configuration for paths and email credentials.
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

Create your private config:

```bash
cp config_example.py config.py
```

Then edit `config.py` for your Raspberry Pi paths, email sender, email recipient, and SMTP app password.

## Configuration

Keep `config.py` private. It contains machine-specific paths and can contain email credentials.

Required values:

- `DATA_DIR` - folder containing input and generated CSV data.
- `REPORTS_DIR` - folder where HTML reports and tracking files are written.
- `DOWNLOADS_DIR` - folder containing Swissgrid download files.
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` - SMTP settings.
- `EMAIL_FROM`, `EMAIL_TO` - report sender and recipient.

Use `config_example.py` as the public, non-sensitive template.

## Run

Run the daily scheduler:

```bash
python main.py
```

Run individual scripts directly only when you want to update or debug one pipeline step.

## GitHub Upload Notes

Do not upload your real `config.py`, local data files, generated reports, logs, or credentials. Upload `config_example.py` instead so others can create their own local config.

Good files to upload:

- Python source files, except private `config.py`
- `config_example.py`
- `README.md`
- `requirements.txt`

Keep private:

- `config.py`
- data folders
- output/report folders
- email passwords and app passwords
- local Raspberry Pi paths if they identify your private setup
