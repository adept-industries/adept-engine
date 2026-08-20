from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import Settings
from app.providers import ProviderConfigurationError

GCM_IV_LENGTH_BYTES = 12


def decrypt_integration_secret(ciphertext: str, key_version: int, settings: Settings) -> str:
    key = _key_for_version(settings, key_version)
    try:
        combined = base64.b64decode(ciphertext, validate=True)
        if len(combined) <= GCM_IV_LENGTH_BYTES:
            raise ValueError("encrypted payload is too short")
        plaintext = AESGCM(key).decrypt(
            combined[:GCM_IV_LENGTH_BYTES],
            combined[GCM_IV_LENGTH_BYTES:],
            None,
        )
        return plaintext.decode("utf-8")
    except (InvalidTag, ValueError, UnicodeDecodeError) as exc:
        raise ProviderConfigurationError("Jira credential ciphertext is invalid") from exc


def encrypt_integration_secret(
    plaintext: str, settings: Settings, *, key_version: int | None = None
) -> tuple[str, int]:
    version = key_version or settings.app_integration_encryption_active_key_version
    key = _key_for_version(settings, version)
    nonce = os.urandom(GCM_IV_LENGTH_BYTES)
    combined = nonce + AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(combined).decode("ascii"), version


def _key_for_version(settings: Settings, key_version: int) -> bytes:
    if key_version != 1:
        raise ProviderConfigurationError(
            f"integration encryption key version {key_version} is not configured in the engine"
        )
    encoded = settings.app_integration_encryption_key_v1_base64.get_secret_value()
    if not encoded:
        raise ProviderConfigurationError(
            "APP_INTEGRATION_ENCRYPTION_KEY_V1_BASE64 is required for Jira jobs"
        )
    try:
        key = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ProviderConfigurationError(
            "APP_INTEGRATION_ENCRYPTION_KEY_V1_BASE64 is invalid Base64"
        ) from exc
    if len(key) != 32:
        raise ProviderConfigurationError(
            "APP_INTEGRATION_ENCRYPTION_KEY_V1_BASE64 must decode to exactly 32 bytes"
        )
    return key
