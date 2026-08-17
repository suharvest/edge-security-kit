"""Single-account cookie session auth (HUB_SPEC §7).

Sessions are server-side and in-memory only: a hub restart invalidates them by
design. The cookie is HttpOnly + SameSite=Strict so the same credential covers
REST and the WS handshake (browsers cannot set an Authorization header on a WS
upgrade, which is why Basic Auth is not used).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

import bcrypt

from .clock import Clock
from .storage import Storage

COOKIE_NAME = "hub_session"
DEFAULT_USERNAME = "admin"
# §7: no fixed default. A first start with no HUB_ADMIN_PASSWORD generates a
# random secret; 24 url-safe chars is ~142 bits, well over the 16-char floor.
INITIAL_PASSWORD_CHARS = 24


def generate_initial_password() -> str:
    return secrets.token_urlsafe(INITIAL_PASSWORD_CHARS)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


@dataclass
class Session:
    token: str
    username: str
    last_used_ms: int


class AuthManager:
    def __init__(
        self,
        storage: Storage,
        clock: Clock | None = None,
        idle_days: float = 7.0,
    ) -> None:
        self.storage = storage
        self.clock = clock or Clock()
        self.idle_ms = idle_days * 86_400_000
        self._sessions: dict[str, Session] = {}

    def ensure_default_account(
        self, username: str = DEFAULT_USERNAME, password: str | None = None
    ) -> tuple[str, str | None]:
        """Create the single account if the auth table is empty.

        Returns ``(username, plaintext_or_None)``; the plaintext is returned only
        when the account was just created, so the caller can log it and persist
        it to ``initial-password.txt`` (§7). With no ``password`` argument the
        secret is randomly generated — there is no fixed default credential.
        """
        if self.storage.any_auth() is not None:
            return (self.storage.any_auth() or {})["username"], None
        plaintext = password or generate_initial_password()
        self.storage.set_auth(username, hash_password(plaintext), must_change=True)
        return username, plaintext

    def login(self, username: str, password: str) -> Session | None:
        record = self.storage.get_auth(username)
        if record is None or not verify_password(password, record["password_hash"]):
            return None
        token = secrets.token_urlsafe(32)
        session = Session(token=token, username=username, last_used_ms=self.clock.wall_ms())
        self._sessions[token] = session
        return session

    def resolve(self, token: str | None) -> Session | None:
        if not token:
            return None
        session = self._sessions.get(token)
        if session is None:
            return None
        now = self.clock.wall_ms()
        if now - session.last_used_ms > self.idle_ms:
            del self._sessions[token]
            return None
        session.last_used_ms = now
        return session

    def logout(self, token: str | None) -> None:
        if token:
            self._sessions.pop(token, None)

    def change_password(
        self, username: str, old_password: str, new_password: str, keep_token: str | None = None
    ) -> str | None:
        """Rotate the password; every other session is invalidated (§4).

        Returns ``None`` on success, otherwise an error string.
        """
        record = self.storage.get_auth(username)
        if record is None or not verify_password(old_password, record["password_hash"]):
            return "invalid credentials"
        if len(new_password) < 8:
            return "new password must be at least 8 characters"
        self.storage.set_auth(username, hash_password(new_password), must_change=False)
        for token in [t for t in self._sessions if t != keep_token]:
            del self._sessions[token]
        return None

    def must_change(self, username: str) -> bool:
        record = self.storage.get_auth(username)
        return bool(record and record["must_change"])

    def account(self) -> dict[str, Any] | None:
        return self.storage.any_auth()
