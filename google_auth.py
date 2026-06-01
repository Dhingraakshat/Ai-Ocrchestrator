"""
Google OAuth2 helper.
Adapted from _refs/google-api-python-client/samples/calendar_api/calendar_sample.py

Local dev:  reads credentials.json + token.json from disk.
Production: reads GOOGLE_TOKEN_JSON and GOOGLE_CREDENTIALS_JSON env vars
            (paste the raw JSON content of each file as the env var value).

To prepare for deployment:
  1. Run locally once so token.json is created.
  2. Set GOOGLE_TOKEN_JSON = contents of token.json
  3. Set GOOGLE_CREDENTIALS_JSON = contents of credentials.json
"""
import json
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

    # --- Try env var first (production / Railway) ---
    token_json_str = os.getenv("GOOGLE_TOKEN_JSON", "")
    if token_json_str:
        try:
            creds = Credentials.from_authorized_user_info(json.loads(token_json_str), SCOPES)
        except Exception as exc:
            logger.warning("Failed to load GOOGLE_TOKEN_JSON: %s", exc)
            creds = None
    elif _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Persist refreshed token back to disk (local dev only)
                if not token_json_str:
                    _TOKEN_PATH.write_text(creds.to_json())
                return creds
            except Exception as exc:
                logger.warning("Token refresh failed: %s", exc)
                creds = None

        if not creds:
            # --- Try env var for credentials.json (production) ---
            creds_json_str = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
            if creds_json_str:
                try:
                    from google_auth_oauthlib.flow import Flow
                    flow = Flow.from_client_config(json.loads(creds_json_str), SCOPES)
                    logger.error(
                        "Google token is missing or expired. "
                        "Re-run locally and update GOOGLE_TOKEN_JSON in Railway Variables."
                    )
                    return None
                except Exception as exc:
                    logger.error("Failed to parse GOOGLE_CREDENTIALS_JSON: %s", exc)
                    return None
            elif _CREDS_PATH.exists():
                flow = InstalledAppFlow.from_client_secrets_file(str(_CREDS_PATH), SCOPES)
                creds = flow.run_local_server(port=0)
                _TOKEN_PATH.write_text(creds.to_json())
            else:
                logger.error(
                    "No Google credentials found. Set GOOGLE_TOKEN_JSON + "
                    "GOOGLE_CREDENTIALS_JSON env vars, or place credentials.json locally."
                )
                return None

    return creds
