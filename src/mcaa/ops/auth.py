"""Admin token check for governance and research endpoints.

Set ``MCAA_ADMIN_TOKEN`` to require the header ``X-Admin-Token`` on admin
endpoints (document approval, analytics, export, deletion). Without it the
server runs in *dev mode*: admin endpoints are open and /api/health says so.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException


def admin_token() -> str | None:
    return os.environ.get("MCAA_ADMIN_TOKEN") or None


def auth_mode() -> str:
    return "token" if admin_token() else "disabled (dev mode)"


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = admin_token()
    if expected is None:
        return
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="admin token required (X-Admin-Token)")
