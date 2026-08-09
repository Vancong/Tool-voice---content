"""Session manager for Gemini Web Playwright automation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from src.gemini_web.models import GeminiWebConfig, SessionStatus
from src.utils.logger import get_logger


class SessionManager:
    """Handles saving, loading, verifying, and clearing browser sessions for Gemini Web."""

    def __init__(self, config: Optional[GeminiWebConfig] = None) -> None:
        self._config = config or GeminiWebConfig()
        self._logger = get_logger(name="gemini_session").bind(module="gemini_web", component="session")
        self._session_path = Path(self._config.session_file).resolve()
        self._profile_dir = Path(self._config.user_data_dir).resolve()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self._session_path.parent.mkdir(parents=True, exist_ok=True)
        self._profile_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_path(self) -> Path:
        return self._session_path

    @property
    def profile_dir(self) -> Path:
        return self._profile_dir

    def has_session_file(self) -> bool:
        """Check if session state file exists and has valid Google authentication cookies."""
        if not self._session_path.exists():
            return False
        try:
            with open(self._session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cookies = data.get("cookies", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                auth_names = {"sid", "hsid", "ssid", "sapisid", "osid", "__secure-1psid", "__secure-3psid", "__secure-1psidts", "__secure-3psidts"}
                has_auth = any(c.get("name", "").lower() in auth_names for c in cookies if isinstance(c, dict))
                return has_auth
        except Exception:
            return False

    def get_status(self) -> SessionStatus:
        """Get high-level status of session."""
        if self.has_session_file():
            return SessionStatus.LOGGED_IN
        return SessionStatus.NOT_LOGGED_IN

    def import_cookies_from_raw_string(self, raw_input: str, target_domain: str = ".google.com", target_file: Optional[Path] = None) -> int:
        """Parse JSON or HTTP Header Cookie string and save to target_file in Playwright format."""
        raw_input = raw_input.strip()
        if not raw_input:
            return 0

        cookies_list = []
        out_path = target_file or self._session_path

        # 1. Try parsing as JSON (Cookie-Editor / EditThisCookie / Playwright format)
        if raw_input.startswith("[") or raw_input.startswith("{"):
            try:
                data = json.loads(raw_input)
                if isinstance(data, dict) and "cookies" in data:
                    raw_cookies = data["cookies"]
                elif isinstance(data, list):
                    raw_cookies = data
                else:
                    raw_cookies = []

                for c in raw_cookies:
                    if isinstance(c, dict) and "name" in c and "value" in c:
                        domain = c.get("domain") or target_domain
                        if not domain.startswith("."):
                            domain = "." + domain.lstrip(".")
                        cookie_obj = {
                            "name": str(c["name"]).strip(),
                            "value": str(c["value"]).strip(),
                            "domain": domain,
                            "path": c.get("path") or "/",
                            "secure": bool(c.get("secure", True)),
                            "httpOnly": bool(c.get("httpOnly", False)),
                        }
                        if "sameSite" in c and c["sameSite"] in ["Strict", "Lax", "None"]:
                            cookie_obj["sameSite"] = c["sameSite"]
                        if "expirationDate" in c or "expires" in c:
                            exp = c.get("expirationDate") or c.get("expires")
                            if exp and float(exp) > 0:
                                cookie_obj["expires"] = float(exp)
                        cookies_list.append(cookie_obj)
            except Exception as exc:
                self._logger.warning("Failed parsing cookie string as JSON: {}", exc)

        # 2. Fallback: Parse as raw Cookie HTTP Header string (NAME=VALUE; NAME2=VALUE2)
        if not cookies_list:
            parts = raw_input.split(";")
            for part in parts:
                if "=" in part:
                    name, val = part.split("=", 1)
                    name = name.strip()
                    val = val.strip()
                    if name and val:
                        cookies_list.append({
                            "name": name,
                            "value": val,
                            "domain": target_domain,
                            "path": "/",
                            "secure": True,
                            "httpOnly": False,
                        })

        if cookies_list:
            # If out_path exists and has existing cookies, merge them
            existing_cookies = []
            if out_path.exists():
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        existing_cookies = old_data.get("cookies", []) if isinstance(old_data, dict) else []
                except Exception:
                    pass

            # Merge: new cookies overwrite old ones with same name & domain
            cookie_dict = {(c.get("domain", ""), c.get("name", "")): c for c in existing_cookies}
            for c in cookies_list:
                cookie_dict[(c.get("domain", ""), c.get("name", ""))] = c

            merged_cookies = list(cookie_dict.values())
            storage_data = {"cookies": merged_cookies, "origins": []}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(storage_data, f, indent=2)
            self._logger.info("Imported {} cookies into {}", len(merged_cookies), out_path)
            return len(cookies_list)

        return 0

    def clear_session(self) -> bool:
        """Clear session files and browser profile."""
        try:
            if self._session_path.exists():
                self._session_path.unlink()
                self._logger.info("Removed session file: {}", self._session_path)

            if self._profile_dir.exists():
                shutil.rmtree(self._profile_dir, ignore_errors=True)
                self._profile_dir.mkdir(parents=True, exist_ok=True)
                self._logger.info("Cleared browser profile dir: {}", self._profile_dir)
            return True
        except Exception as exc:
            self._logger.error("Failed to clear session: {}", exc)
            return False
