"""SMTP email provider for alert notifications."""

from __future__ import annotations

import contextlib
import smtplib
from email.message import EmailMessage

import structlog

from app.core.config import Settings, get_settings
from app.providers import ProviderError

logger = structlog.get_logger()


def send_email(
    *,
    to_address: str,
    subject: str,
    text_content: str,
    html_content: str | None = None,
    settings: Settings | None = None,
) -> None:
    """
    Send an email via SMTP.

    Raises ProviderError if the SMTP connection or delivery fails, allowing
    the durable job processor to retry appropriately.
    """
    cfg = settings or get_settings()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.app_email_from.strip("\"'")
    msg["To"] = to_address
    msg.set_content(text_content)

    if html_content:
        msg.add_alternative(html_content, subtype="html")

    try:
        server_context: smtplib.SMTP
        if cfg.smtp_port == 465:
            server_context = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=15)
        else:
            server_context = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15)

        with server_context as server:
            password = cfg.smtp_password.get_secret_value()
            if cfg.smtp_port != 465 and (cfg.smtp_username or password or cfg.smtp_port == 587):
                with contextlib.suppress(smtplib.SMTPNotSupportedError):
                    server.starttls()
            if cfg.smtp_username and password:
                server.login(cfg.smtp_username, password)
            server.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        logger.error(
            "smtp_send_failed",
            to_address=to_address,
            subject=subject,
            error=str(exc),
        )
        raise ProviderError(f"SMTP delivery failed: {exc}") from exc
