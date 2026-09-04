"""Load Zerodha Kite Connect credentials and authenticate the API client."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
SESSION_PATH = PROJECT_ROOT / ".kite_session"
KITE_API_ROOT = "https://api.kite.trade"


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "****"
    return f"****{value[-4:]}"


def load_credentials() -> tuple[str, str]:
    if not ENV_PATH.exists():
        raise RuntimeError(f"Missing {ENV_PATH}. Create it and add KITE_API_KEY and KITE_API_SECRET.")

    load_dotenv(ENV_PATH)

    api_key = (os.getenv("KITE_API_KEY") or "").strip()
    api_secret = (os.getenv("KITE_API_SECRET") or "").strip()

    missing = [
        name
        for name, value in (("KITE_API_KEY", api_key), ("KITE_API_SECRET", api_secret))
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing " + ", ".join(missing) + f" in {ENV_PATH}. Paste your values after the '='."
        )

    return api_key, api_secret


def create_kite_client(api_key: str) -> KiteConnect:
    try:
        return KiteConnect(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize KiteConnect: {exc}") from exc


def _load_saved_access_token() -> str | None:
    env_token = (os.getenv("KITE_ACCESS_TOKEN") or "").strip()
    if env_token:
        return env_token
    if not SESSION_PATH.exists():
        return None
    try:
        payload = json.loads(SESSION_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    token = str(payload.get("access_token") or "").strip()
    return token or None


def save_access_token(access_token: str) -> None:
    SESSION_PATH.write_text(json.dumps({"access_token": access_token}))
    SESSION_PATH.chmod(0o600)


def _login_required_message(kite: KiteConnect) -> str:
    return (
        "Kite login required. Open this URL in a browser, complete login, "
        "then re-run with --request-token <token> from the redirected URL:\n"
        f"{kite.login_url()}"
    )


def authenticate(request_token: str | None = None) -> KiteConnect:
    """Return a KiteConnect client with a valid access token.

    Access tokens are reused from KITE_ACCESS_TOKEN or .kite_session when still
    valid. Otherwise a request_token from the Kite login redirect is exchanged.
    """
    api_key, api_secret = load_credentials()
    kite = create_kite_client(api_key)
    token = (request_token or os.getenv("KITE_REQUEST_TOKEN") or "").strip()

    if token:
        try:
            session = kite.generate_session(token, api_secret=api_secret)
        except Exception as exc:
            raise RuntimeError(f"Failed to exchange request token: {exc}") from exc
        access_token = session.get("access_token")
        if not access_token:
            raise RuntimeError("Kite session response did not include an access_token.")
        save_access_token(access_token)
        kite.profile()
        return kite

    access_token = _load_saved_access_token()
    if access_token:
        kite.set_access_token(access_token)
        try:
            kite.profile()
            return kite
        except TokenException:
            if SESSION_PATH.exists():
                SESSION_PATH.unlink()

    raise RuntimeError(_login_required_message(kite))


def test_connectivity(kite: KiteConnect) -> None:
    if not kite.login_url():
        raise RuntimeError("KiteConnect did not generate a login URL.")

    try:
        response = requests.get(
            KITE_API_ROOT,
            headers={"X-Kite-Version": "3"},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the Kite API at {KITE_API_ROOT}: {exc}") from exc

    if response.status_code >= 500:
        raise RuntimeError(
            f"Kite API returned HTTP {response.status_code}. Try again later."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Kite Connect authentication and connectivity.")
    parser.add_argument(
        "--request-token",
        default=None,
        help="Request token from the Kite login redirect URL.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Complete request-token -> access-token login and call profile().",
    )
    args = parser.parse_args()

    try:
        api_key, api_secret = load_credentials()
        kite = create_kite_client(api_key)
        test_connectivity(kite)
        if args.login or args.request_token:
            kite = authenticate(request_token=args.request_token)
            profile = kite.profile()
            print(f"Authenticated as: {profile.get('user_name', 'unknown')}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _ = api_secret
    print("Kite Connect client initialized.")
    print(f"API key loaded: {_mask(api_key)}")
    print("API secret loaded: ****")
    print(f"Kite API reachable: {KITE_API_ROOT}")
    print("Connectivity test passed. No market data was downloaded and no orders were placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
