"""
Utilities for MCP server loading.

Kept separate from loader.py to keep orchestration minimal and helpers reusable.
"""

from __future__ import annotations

import fnmatch
import os
import shlex
from pathlib import Path
from typing import Any, Callable, Dict, List
import re

import yaml
from mcp2py import load as mcp2py_load
from precursor.config.loader import get_user_name
from platformdirs import user_data_dir

# ----------------------------
# YAML loading (explicit path)
# ----------------------------
def load_yaml_override(path: str) -> Dict[str, Any]:
    """Load a YAML file from an explicit path (expands ~)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"mcp_servers.yaml not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ----------------------------
# Env & process setup helpers
# ----------------------------
def apply_env(spec: Dict[str, Any]) -> None:
    """Apply per-server env vars and inject USER_NAME for GUM."""
    for k, v in (spec.get("env") or {}).items():
        if v is not None:
            # Expand ~ and ${VARS} for portability
            os.environ[k] = os.path.expandvars(os.path.expanduser(str(v)))

    # Special-case: inject USER_NAME for the GUM server
    if (spec.get("id") or "").lower() == "gum":
        user_name = get_user_name()
        if user_name:
            os.environ["USER_NAME"] = user_name

    # Default SLIDEV_DIR for the slides server if not explicitly provided
    if (spec.get("id") or "").lower() == "slides":
        if not os.environ.get("SLIDEV_DIR"):
            default_dir = os.path.join(user_data_dir(appname="precursor", appauthor="precursor", version=None), "slides")

            if not os.path.exists(default_dir):
                os.makedirs(default_dir, exist_ok=True)

            # Do not create directories here; the Node server ensures existence on boot
            os.environ["SLIDEV_DIR"] = str(default_dir)


# ----------------------------
# Command normalization + spawn
# ----------------------------
def _build_cmd_string(load_field: Any) -> str:
    """
    Normalize the YAML 'load' field into a single shell command string.

    Accepts:
      - str: e.g. "python -m precursor.mcp_servers.drive.server"
      - dict: {command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem", "/Users/michael/Projects", "/Users/michael/Desktop"]}

    Returns
    -------
    str
        A shell-safe command string suitable for passing to mcp2py.load().
    """
    if isinstance(load_field, str):
        # Expand ~ and ${VARS} for each token; do not rely on shell expansion.
        # Many launchers pass a single command string to subprocess without a shell.
        parts = shlex.split(load_field)
        expanded_parts = [
            os.path.expandvars(os.path.expanduser(str(p))) for p in parts
        ]
        # IMPORTANT: Do NOT add shell quotes here. mcp2py may not spawn via a shell,
        # so quotes would be passed literally to the process.
        return " ".join(expanded_parts)

    if isinstance(load_field, dict):
        cmd = load_field.get("command")
        args = load_field.get("args") or []
        if not cmd:
            raise ValueError("Invalid 'load' mapping: missing 'command'.")
        if not isinstance(args, list):
            raise ValueError("Invalid 'load' mapping: 'args' must be a list of strings.")
        # Expand ~ and ${VARS} in both command and args
        cmd_expanded = os.path.expandvars(os.path.expanduser(str(cmd)))
        args_expanded = [os.path.expandvars(os.path.expanduser(str(a))) for a in args]
        parts = [cmd_expanded] + args_expanded
        # IMPORTANT: Do NOT add shell quotes here. mcp2py may not spawn via a shell,
        # so quotes would be passed literally to the process.
        return " ".join(parts)

    raise TypeError(f"Unsupported 'load' type: {type(load_field)!r}")


_DEFAULT_SERVER_TIMEOUT: float = 60.0  # seconds


def start_server(
    spec: Dict[str, Any],
    *,
    timeout: float = _DEFAULT_SERVER_TIMEOUT,
) -> Any:
    """Spawn one MCP server via mcp2py with helpful error context.

    Args:
        spec: Server specification dict from mcp_servers.yaml.
        timeout: Maximum seconds to wait for the server to initialize.
                 Defaults to 60s. Set to 0 or negative to wait indefinitely.
    """
    cmd: str | None = None
    sid = spec.get("id", "<unknown-id>")
    try:
        cmd = _build_cmd_string(spec["load"])
        if timeout > 0:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    mcp2py_load, cmd, auto_auth=True, headers=spec.get("headers"),
                )
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(
                        f"MCP server '{sid}' did not initialize within {timeout}s"
                    )
        else:
            return mcp2py_load(cmd, auto_auth=True, headers=spec.get("headers"))
    except Exception as e:
        # Build a richer error with expanded command and unresolved env var hints
        msg = f"Failed to start MCP '{sid}' with command: {spec['load']}"
        if cmd:
            msg += f" (expanded: {cmd})"
            # Detect unresolved environment variables like $VAR or ${VAR}
            unresolved = []
            for match in re.findall(r"\$(\{?[A-Za-z_][A-Za-z0-9_]*\}?)", cmd):
                var_name = match.strip("{}")
                if not os.environ.get(var_name):
                    unresolved.append(var_name)
            if unresolved:
                uniq = ", ".join(sorted(set(unresolved)))
                msg += f" — unresolved env vars: {uniq}"
        raise RuntimeError(msg) from e


# ----------------------------
# Pattern compilation
# ----------------------------
def _as_list(x: Any, default: List[str]) -> List[str]:
    if x is None:
        return list(default)
    if isinstance(x, str):
        return [x]
    if isinstance(x, list) and all(isinstance(s, str) for s in x):
        return x
    try:
        return [str(x)]
    except Exception:
        return list(default)


def compile_allow_fn(defaults: Dict[str, Any]) -> Callable[[str], bool]:
    """
    Compile a callable allow(name) based on glob-style patterns from defaults:
      - allow_patterns: list[str] (default ["*"])
      - deny_patterns: list[str] (default [])
    """
    allow_patterns = _as_list(defaults.get("allow_patterns"), ["*"])
    deny_patterns = _as_list(defaults.get("deny_patterns"), [])

    def match_any(name: str, patterns: List[str]) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    def allow(name: str) -> bool:
        if not match_any(name, allow_patterns):
            return False
        if match_any(name, deny_patterns):
            return False
        return True

    return allow