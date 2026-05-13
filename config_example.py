from pathlib import Path
import os
import sys
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/path/to/pv_forecast_risk/data").expanduser().resolve()
REPORTS_DIR = Path("/path/to/pv_forecast_risk/output").expanduser().resolve()
DOWNLOADS_DIR = Path("/path/to/downloads").expanduser().resolve()

SMTP_HOST = "smtp.example.com"
SMTP_PORT = 587
SMTP_USER = "your-email@example.com"
SMTP_PASSWORD = "your-email-app-password"
EMAIL_FROM = "your-email@example.com"
EMAIL_TO = "recipient@example.com"


DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def data_path(filename):
    return DATA_DIR / filename


def script_path(filename):
    return PROJECT_ROOT / filename


def report_path(filename):
    return REPORTS_DIR / filename


def maybe_open_browser(path):
    value = os.environ.get("PV_FORECAST_OPEN_BROWSER")
    if value is not None:
        should_open = value.strip().lower() in {"1", "true", "yes", "on"}
    else:
        should_open = sys.platform.startswith("win") or bool(os.environ.get("DISPLAY"))

    if should_open:
        webbrowser.open(Path(path).resolve().as_uri(), new=2)
