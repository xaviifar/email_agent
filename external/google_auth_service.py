"""
Google Auth Service Module

Singleton service for Google OAuth 2.0 authentication. Handles the full
OAuth flow (authorization URL generation, code exchange, token refresh) and
provides ``google.oauth2.credentials.Credentials`` objects ready for use with
the Gmail API.

Architecture note:
    Token refresh is handled here (via ``get_credentials``) rather than inside
    ``GmailClient``. This prevents ``aiogoogle`` from attempting its own
    internal refresh flow, which fails with ``invalid_grant`` errors.
"""

import os
import json
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GoogleAuthService:
    """Singleton Google OAuth 2.0 service.

    Manages client configuration loading (from environment variables or a
    JSON secrets file), OAuth flow execution, and credential refresh.

    Attributes:
        client_secrets_file: Path to the ``client_secret.json`` file.
        scopes: List of OAuth scopes requested during authorisation.
        redirect_uri: OAuth redirect URI registered in Google Cloud Console.
        client_config: Cached client configuration dict.
    """

    _instance = None

    def __new__(cls):
        """Enforce singleton pattern — only one instance is ever created."""
        if cls._instance is None:
            cls._instance = super(GoogleAuthService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialise the service and load client configuration.

        Skips re-initialisation on subsequent calls (singleton guard).
        Loads OAuth configuration from environment variables.
        """
        if getattr(self, "_initialized", False):
            return

        self.client_secrets_file = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
        self.scopes = [
            scope.strip()
            for scope in os.getenv("GOOGLE_SCOPES", "").split(",")
            if scope.strip()
        ]
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost")
        self.client_config = self._load_client_config()
        self._initialized = True

    def _load_client_config(self) -> Dict[str, str]:
        """Load OAuth client configuration from environment variables or file.

        Tries environment variables first (``GOOGLE_CLIENT_ID`` /
        ``GOOGLE_CLIENT_SECRET``). Falls back to the JSON secrets file at
        ``client_secrets_file`` if env vars are absent.

        Returns:
            A dict with keys ``client_id``, ``client_secret``, ``token_uri``,
            and ``auth_uri``. Returns an empty dict if neither source is
            available.
        """
        file_config: Dict[str, str] = {}
        if os.path.exists(self.client_secrets_file):
            try:
                with open(self.client_secrets_file, "r") as f:
                    secrets = json.load(f)
                    key = "installed" if "installed" in secrets else "web"
                    file_config = secrets[key]
            except Exception as e:
                logger.error(f"Error loading client secrets file: {e}")

        env_config: Dict[str, str] = {}
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if client_id and client_secret:
            env_config = {
                "client_id": client_id,
                "client_secret": client_secret,
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            }

        if env_config and file_config:
            env_id = env_config.get("client_id")
            env_secret = env_config.get("client_secret")
            file_id = file_config.get("client_id")
            file_secret = file_config.get("client_secret")
            if env_id != file_id or env_secret != file_secret:
                logger.warning(
                    "Google OAuth config mismatch between env vars and client_secret.json. "
                    "Using client_secret.json to avoid invalid_client during token exchange."
                )
                return file_config
            return env_config

        if env_config:
            return env_config

        if file_config:
            return file_config

        logger.warning("No Google Credentials found in Env or File.")
        return {}

    def get_client_config(self) -> Dict[str, str]:
        """Return the cached OAuth client configuration.

        Returns:
            Dict with ``client_id``, ``client_secret``, ``token_uri``, and
            ``auth_uri``.
        """
        return self.client_config

    def _build_oauth_state_token(self, code_verifier: str, expires_in_seconds: int = 600) -> str:
        """Create a signed state payload containing PKCE data and expiry."""
        now = datetime.now(timezone.utc)
        payload = {
            "nonce": secrets.token_urlsafe(24),
            "code_verifier": code_verifier,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=expires_in_seconds)).timestamp()),
        }
        return jwt.encode(
            payload,
            self._get_secret_key(),
            algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        )

    def parse_oauth_state_token(self, state_token: str) -> Dict[str, str]:
        """Decode and validate the signed OAuth state payload."""
        try:
            payload = jwt.decode(
                state_token,
                self._get_secret_key(),
                algorithms=[os.getenv("JWT_ALGORITHM", "HS256")],
            )
        except JWTError as exc:
            raise ValueError("Invalid OAuth state. Please retry login.") from exc

        code_verifier = payload.get("code_verifier")
        if not code_verifier:
            raise ValueError("OAuth session expired. Please retry login.")

        return payload

    @staticmethod
    def _get_secret_key() -> str:
        secret_key = os.getenv("SECRET_KEY")
        if not secret_key:
            raise ValueError("SECRET_KEY no está configurado en .env")
        return secret_key

    def get_authorization_url(self) -> Tuple[str, str]:
        """Generate the Google OAuth 2.0 authorisation URL.

        Returns:
            A tuple of ``(authorization_url, state)`` where ``state`` is a
            signed payload containing CSRF and PKCE data for the callback.

        Raises:
            ValueError: If the client configuration is missing.
        """
        config = self.get_client_config()
        if not config:
            raise ValueError("Google Client Config missing")

        flow = Flow.from_client_config(
            {"web": config},
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
        )
        code_verifier = secrets.token_urlsafe(96)
        flow.code_verifier = code_verifier
        state_token = self._build_oauth_state_token(code_verifier)

        authorization_url, state = flow.authorization_url(
            access_type="offline",
            #include_granted_scopes="true",
            prompt="consent",
            state=state_token,
        )
        return authorization_url, state

    def exchange_code_for_token(self, code: str, *, state: str | None = None, code_verifier: str | None = None) -> Dict:
        """Exchange an authorisation code for OAuth tokens.

        Args:
            code: The authorisation code received from Google's callback.
            state: OAuth state returned by Google callback.
            code_verifier: PKCE verifier generated during login.

        Returns:
            A dict containing ``token``, ``refresh_token``, ``token_uri``,
            ``client_id``, ``client_secret``, and ``scopes``.
        """
        config = self.get_client_config()
        flow = Flow.from_client_config(
            {"web": config},
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
            state=state,
        )
        if code_verifier:
            flow.code_verifier = code_verifier
        flow.fetch_token(code=code)
        credentials = flow.credentials

        return {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes,
        }

    def get_credentials(self, token_data: Optional[Dict] = None) -> Credentials:
        """Build and refresh ``google.oauth2.credentials.Credentials``.

        Always attempts a token refresh when a ``refresh_token`` is present.
        This guarantees that ``GmailClient`` receives a fresh access token and
        never needs to perform its own (unreliable) refresh.

        Note:
            Scopes are intentionally *not* passed to ``Credentials()``. Passing
            them causes ``google-auth`` to include them in the refresh request,
            which triggers ``invalid_grant`` if they do not exactly match the
            originally granted scopes.

        Args:
            token_data: Optional token dictionary. When omitted, it is loaded
                from the JSON file configured by ``TOKEN_FILE``.

        Returns:
            A refreshed ``Credentials`` object with a valid access token.

        Raises:
            ValueError: If the token refresh fails or no valid access token
                is available after refresh (user must re-authenticate).
        """
        token_file = os.getenv("TOKEN_FILE", "token.json")

        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                token_data = json.load(f)

            creds = Credentials(
                token=token_data.get("token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri"),
                client_id=token_data.get("client_id"),
                client_secret=token_data.get("client_secret"),
                scopes=token_data.get("scopes"),
            )

            if creds.valid:
                return creds

            if creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    return creds
                except Exception:
                    logger.warning(
                        "Refresh token inválido. Se realizará un nuevo login."
                    )

        # No existe token o el refresh token ya no sirve
        creds = self.login_local()

        with open(token_file, "w") as f:
            json.dump({
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }, f)

        return creds

    def get_user_info(self, token_data: Dict) -> Dict:
        """Retrieve the authenticated user's Google profile information.

        Args:
            token_data: Token dict as returned by ``exchange_code_for_token``.

        Returns:
            A dict with Google profile fields (``id``, ``email``, ``name``,
            ``picture``, etc.).
        """
        creds = self.get_credentials(token_data)
        from googleapiclient.discovery import build
        service = build("oauth2", "v2", credentials=creds)
        return service.userinfo().get().execute()

    def login_local(self) -> Credentials:

        flow = InstalledAppFlow.from_client_secrets_file(
            self.client_secrets_file,
            self.scopes,
        )

        credentials = flow.run_local_server(
            port=0,
            access_type="offline",
            prompt="consent",
        )

        return credentials


google_auth_service = GoogleAuthService()