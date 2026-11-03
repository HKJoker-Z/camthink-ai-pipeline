"""``${VAR}`` and ``${VAR:-default}`` substitution in loaded YAML values.

Only string leaves are substituted. A reference to an unset variable without a default is a
configuration error so that secrets are never silently empty.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from ai_pipeline.core.errors import ConfigError

_VAR_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def substitute_env(value: Any, *, env: Mapping[str, str] | None = None, path: str = "") -> Any:
    env = os.environ if env is None else env

    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name = m.group("name")
            if name in env:
                return env[name]
            if m.group("default") is not None:
                return m.group("default")
            raise ConfigError(
                f"Environment variable {name!r} referenced at {path or '<root>'} is not set",
                code="ENV_VAR_MISSING",
                details={"variable": name, "path": path},
                hint=f"Export {name} or add it to .env (see .env.example), "
                f"or use ${{{name}:-default}} to supply a default.",
            )

        return _VAR_RE.sub(repl, value)
    if isinstance(value, dict):
        return {
            k: substitute_env(v, env=env, path=f"{path}.{k}" if path else str(k))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [substitute_env(v, env=env, path=f"{path}[{i}]") for i, v in enumerate(value)]
    return value
