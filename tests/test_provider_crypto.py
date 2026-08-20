import base64
import os

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.providers import ProviderConfigurationError
from app.providers.crypto import decrypt_integration_secret, encrypt_integration_secret


def _settings(key: bytes) -> Settings:
    return Settings(
        postgres_password=SecretStr("test"),
        app_integration_encryption_active_key_version=1,
        app_integration_encryption_key_v1_base64=SecretStr(base64.b64encode(key).decode("ascii")),
    )


def test_jira_secret_round_trip_matches_api_aes_gcm_layout() -> None:
    settings = _settings(os.urandom(32))
    ciphertext, version = encrypt_integration_secret("refresh-secret", settings)

    combined = base64.b64decode(ciphertext)
    assert len(combined[:12]) == 12
    assert decrypt_integration_secret(ciphertext, version, settings) == "refresh-secret"


def test_jira_secret_invalid_tag_is_permanent_configuration_error() -> None:
    settings = _settings(os.urandom(32))
    ciphertext, version = encrypt_integration_secret("refresh-secret", settings)
    tampered = bytearray(base64.b64decode(ciphertext))
    tampered[-1] ^= 1

    with pytest.raises(ProviderConfigurationError, match="ciphertext is invalid"):
        decrypt_integration_secret(base64.b64encode(tampered).decode(), version, settings)
