from __future__ import annotations
from email.message import EmailMessage
from pathlib import Path
import mimetypes
import smtplib
from config import EMAIL_FROM, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER


def send_html_report_email(
    to_address: str,
    html_file_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    subject: str = "GHI Forecast Risk Report",
    body: str = "Attached are the latest GHI forecast risk HTML reports.",
    *,
    smtp_host: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    from_address: str | None = None,
    use_tls: bool = True,
) -> None:
    """Send the generated HTML report as an email attachment."""

    if isinstance(html_file_path, (str, Path)):
        report_paths = [Path(html_file_path).expanduser().resolve()]
    else:
        report_paths = [Path(path).expanduser().resolve() for path in html_file_path]

    for report_path in report_paths:
        if not report_path.exists():
            raise FileNotFoundError(f"HTML report not found: {report_path}")

    smtp_host = smtp_host or SMTP_HOST
    smtp_port = smtp_port or SMTP_PORT
    smtp_user = smtp_user or SMTP_USER
    smtp_password = smtp_password or SMTP_PASSWORD
    from_address = from_address or EMAIL_FROM or smtp_user

    missing = [
        name
        for name, value in {
            "PV_FORECAST_SMTP_HOST": smtp_host,
            "PV_FORECAST_SMTP_USER": smtp_user,
            "PV_FORECAST_SMTP_PASSWORD": smtp_password,
            "PV_FORECAST_EMAIL_FROM or PV_FORECAST_SMTP_USER": from_address,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing email configuration: {', '.join(missing)}")

    message = EmailMessage()
    message["From"] = from_address
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(body)

    html_text = report_paths[0].read_text(encoding="utf-8")
    message.add_alternative(html_text, subtype="html")

    for report_path in report_paths:
        content_type, _ = mimetypes.guess_type(report_path)
        maintype, subtype = (content_type or "text/html").split("/", 1)
        message.add_attachment(
            report_path.read_bytes(),
            maintype=maintype,
            subtype=subtype,
            filename=report_path.name,
        )

    with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
        if use_tls:
            server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(message)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Email a PV forecast risk HTML report.")
    parser.add_argument("to_address", help="Recipient email address")
    parser.add_argument("html_file_path", help="Path to the generated HTML report")
    parser.add_argument("--subject", default="PV Forecast Risk Report")
    parser.add_argument("--body", default="Attached is the latest PV forecast risk HTML report.")
    parser.add_argument("--no-tls", action="store_true", help="Disable STARTTLS")
    args = parser.parse_args()

    send_html_report_email(
        to_address=args.to_address,
        html_file_path=args.html_file_path,
        subject=args.subject,
        body=args.body,
        use_tls=not args.no_tls,
    )
