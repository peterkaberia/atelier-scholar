"""
Local Secret Encryption Module

Provides symmetric encryption for API keys entered via the Settings UI before
they're persisted to SQLite. The encryption key itself is generated once on
first run and stored in a local file outside git - appropriate for a local,
single-user tool where a full KMS/secrets-manager story would be overkill.
See docs/adr/0001-local-encryption-key-for-ui-settings.md for the reasoning.
"""

import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

# Overridable for tests/deployments that want the key somewhere else.
SECRET_KEY_PATH = os.getenv("ATELIER_SECRET_KEY_PATH", ".atelier_secret.key")


def _get_or_create_key() -> bytes:
    """Reads the local Fernet key, generating and persisting one on first run."""
    if os.path.exists(SECRET_KEY_PATH):
        with open(SECRET_KEY_PATH, "rb") as f:
            return f.read()

    key = Fernet.generate_key()
    with open(SECRET_KEY_PATH, "wb") as f:
        f.write(key)
    return key


@lru_cache(maxsize=1)
def _get_cipher() -> Fernet:
    return Fernet(_get_or_create_key())


def encrypt(value: str) -> str:
    """Encrypts a plaintext secret for storage."""
    return _get_cipher().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt(token: str) -> str:
    """
    Decrypts a stored secret. Returns an empty string (rather than raising)
    if the token is malformed or was encrypted under a since-rotated key,
    since callers treat "no usable secret" the same as "not configured".
    """
    try:
        return _get_cipher().decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError):
        return ""
