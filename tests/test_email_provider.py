from unittest.mock import MagicMock, patch

import pytest

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
