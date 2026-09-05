"""Load, parse, and hash MARLA experiment configuration files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from marla.config.models import Config
from marla.config.validation import validate_config_semantics

# Each marker as its underscore-separated segment sequence, e.g.
# "api_key" -> ("api", "key").
_REDACTED_KEY_MARKERS = tuple(
    tuple(marker.split("_")) for marker in ("password", "secret", "token", "api_key", "apikey")
)
_REDACTED_PLACEHOLDER = "***REDACTED***"


class ConfigError(Exception):
    """Raised when an experiment configuration file is invalid.

    Carries the full list of problems (Pydantic validation errors and/or
    semantic validation problems) so callers such as ``marla validate`` can
    print them all at once instead of failing on the first one.
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(problems))


def _format_pydantic_error(exc: ValidationError) -> list[str]:
    messages = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        messages.append(f"{loc}: {error['msg']}")
    return messages


def parse_config(raw: dict[str, Any]) -> Config:
    """Parse an already-loaded YAML dict into a validated :class:`Config`."""
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_pydantic_error(exc)) from exc


def load_config(path: str | Path) -> Config:
    """Load and validate an experiment configuration file.

    Performs both Pydantic schema validation and the semantic checks in
    :mod:`marla.config.validation` (e.g. scenario file existence).
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError([f"Configuration file not found: {config_path}"])

    with config_path.open("r", encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError([f"Failed to parse YAML: {exc}"]) from exc

    if not isinstance(raw, dict):
        raise ConfigError(["Configuration file must contain a YAML mapping at the top level"])

    config = parse_config(raw)

    problems = validate_config_semantics(config, config_path.parent)
    if problems:
        raise ConfigError(problems)

    return config


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered.endswith("_env"):
        # By convention these store an environment variable *name*
        # (e.g. password_env: "MARLA_GATEKEEPER_PASSWORD"), never the
        # secret value itself, so there is nothing to redact.
        return False

    # Matched as a contiguous run of whole underscore-separated segments,
    # never as a raw substring -- a plain `marker in lowered` check redacts
    # any key that merely *contains* the marker inside a longer word, e.g.
    # "max_new_tokens" (a generation-length setting, not a secret) via
    # "token" inside "tokens". That corrupts it into a string where an int
    # is expected the moment such a config.yaml is reloaded (see
    # research/aamas2027's checkpoint-evaluation harness, which does
    # exactly that). Segment matching still catches "auth_token"
    # (segment ("token",)) and "openai_api_key" (adjacent segment pair
    # ("api", "key")) correctly.
    segments = lowered.split("_")
    for marker_segments in _REDACTED_KEY_MARKERS:
        n = len(marker_segments)
        if any(tuple(segments[i : i + n]) == marker_segments for i in range(len(segments) - n + 1)):
            return True
    return False


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, val in value.items():
            if isinstance(key, str) and _is_secret_key(key):
                redacted[key] = _REDACTED_PLACEHOLDER
            else:
                redacted[key] = _redact(val)
        return redacted
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def redacted_config_dict(config: Config) -> dict[str, Any]:
    """Return a plain-dict dump of the config with any secret-shaped keys redacted.

    MARLA never stores actual secrets in configuration (only environment
    variable *names* via ``password_env``), so this is a defensive measure
    against future fields rather than something expected to trigger today.
    """
    return _redact(config.model_dump(mode="json"))


def config_hash(config: Config) -> str:
    """Stable SHA-256 hash of the resolved configuration, for metadata.json."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
