"""
Google OAuth2 helper.
Adapted from _refs/google-api-python-client/samples/calendar_api/calendar_sample.py

On first run, opens a browser for OAuth consent. Saves token to token.json.
Requires credentials.json downloaded from Google Cloud Console.
"""
import logging
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger("jarvis")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar",  # read + write
]

_TOKEN_PATH = Path("token.json")
_CREDS_PATH = Path(os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json"))


def get_google_creds() -> Credentials | None:
    """Return valid Google credentials, refreshing or re-authorizing as needed."""
    creds: Credentials | None = None

    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                logger.warning("Token refresh failed: %s", exc)
                creds = None

        if not creds:
            if not _CREDS_PATH.exists():
                logger.error(
                    "credentials.json not found. Download it from Google Cloud Console "
                    "and place it at %s", _CREDS_PATH
                )
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(_CREDS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)

        _TOKEN_PATH.write_text(creds.to_json())

    return creds
