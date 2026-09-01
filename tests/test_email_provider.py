from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.providers import ProviderError
from app.providers.email import send_email


def test_send_email_success() -> None:
    settings = Settings(
        smtp_host="localhost",
        smtp_port=1025,
        app_email_from="Adept <test@adept.local>",
    )
    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        send_email(
            to_address="lead@example.com",
            subject="Test Subject",
            text_content="Plain text content",
            html_content="<p>HTML content</p>",
            settings=settings,
        )

        mock_smtp.assert_called_once_with("localhost", 1025, timeout=15)
        mock_server.send_message.assert_called_once()
        sent_msg = mock_server.send_message.call_args[0][0]
        assert sent_msg["Subject"] == "Test Subject"
        assert sent_msg["To"] == "lead@example.com"
        assert sent_msg["From"] == "Adept <test@adept.local>"


def test_send_email_smtp_failure_raises_provider_error() -> None:
    settings = Settings(
        smtp_host="localhost",
        smtp_port=1025,
    )
    with patch("smtplib.SMTP", side_effect=OSError("Connection refused")):
        with pytest.raises(ProviderError) as exc_info:
            send_email(
                to_address="lead@example.com",
                subject="Test Subject",
                text_content="Plain text content",
                settings=settings,
            )
        assert "SMTP delivery failed" in str(exc_info.value)


def test_send_email_ssl_port_465() -> None:
    settings = Settings(
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="user@example.com",
        smtp_password=SecretStr("password"),
        app_email_from="Adept <test@adept.local>",
    )
    with patch("smtplib.SMTP_SSL") as mock_smtp_ssl:
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server

        send_email(
            to_address="lead@example.com",
            subject="Test Subject",
            text_content="Plain text content",
            settings=settings,
        )

        mock_smtp_ssl.assert_called_once_with("smtp.gmail.com", 465, timeout=15)
        mock_server.login.assert_called_once_with("user@example.com", "password")
        mock_server.send_message.assert_called_once()


def test_settings_spring_mail_aliases() -> None:
    settings = Settings.model_validate(
        {
            "spring_mail_host": "smtp.gmail.com",
            "spring_mail_port": 587,
            "spring_mail_username": "user@gmail.com",
            "spring_mail_password": "secret_password",
        }
    )
    assert settings.smtp_host == "smtp.gmail.com"
    assert settings.smtp_port == 587
    assert settings.smtp_username == "user@gmail.com"
    assert settings.smtp_password.get_secret_value() == "secret_password"


def test_send_email_strips_quotes_from_from_header() -> None:
    settings = Settings(
        smtp_host="localhost",
        smtp_port=1025,
        app_email_from='"Adept <no-reply@adeptindustries.dev>"',
    )
    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        send_email(
            to_address="lead@example.com",
            subject="Test Subject",
            text_content="Plain text content",
            settings=settings,
        )

        sent_msg = mock_server.send_message.call_args[0][0]
        assert sent_msg["From"] == "Adept <no-reply@adeptindustries.dev>"
