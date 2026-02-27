"""
MCP server loader.

Reads `config/mcp_servers.yaml`, starts enabled MCP servers via mcp2py,
and returns both the loaded servers and a compiled allow_fn for tool filtering.

Servers are loaded **concurrently** (via a thread pool) and each server has a
configurable startup timeout.  If any single server fails or times out, the
remaining servers are still returned so the agent can run with a degraded
toolset rather than hanging forever.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from precursor.config.loader import load_mcp_servers_yaml
from precursor.mcp_loader.utils import (
    apply_env,
    start_server,
    compile_allow_fn,
    load_yaml_override,
    _DEFAULT_SERVER_TIMEOUT,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedServer:
    id: str
    client: Any  # mcp2py client (exposes .tools)


@dataclass
class MCPConfigBundle:
    servers: List[LoadedServer]
    allow_fn: Callable[[str], bool]  # e.g., allow("drive.search_files") -> True/False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_enabled_specs(
    cfg: Dict[str, Any],
    wanted: Optional[set] = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (defaults, filtered_specs) from a parsed config dict."""
    defaults: Dict[str, Any] = cfg.get("defaults") or {}
    servers_cfg: List[Dict[str, Any]] = cfg.get("servers") or []

    specs: List[Dict[str, Any]] = []
    for spec in servers_cfg:
        if wanted is not None and str(spec.get("id")) not in wanted:
            continue
        enabled = spec.get("enabled", defaults.get("enabled", True))
        if not enabled:
            continue
        if "id" not in spec or "load" not in spec:
            continue
        specs.append(spec)
    return defaults, specs


def _start_one(spec: Dict[str, Any], timeout: float) -> LoadedServer:
    """Apply env and start a single server.  Runs inside a worker thread."""
    apply_env(spec)
    client = start_server(spec, timeout=timeout)
    return LoadedServer(id=str(spec["id"]), client=client)


def _load_servers_concurrent(
    specs: List[Dict[str, Any]],
    timeout: float,
) -> List[LoadedServer]:
    """Start all *specs* concurrently, skipping any that fail or time out."""
    if not specs:
        return []

    servers: List[LoadedServer] = []

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        future_to_sid = {
            pool.submit(_start_one, spec, timeout): spec.get("id", "?")
            for spec in specs
        }
        for future in as_completed(future_to_sid):
            sid = future_to_sid[future]
            try:
                servers.append(future.result())
                logger.info("mcp_loader: server '%s' started successfully", sid)
            except Exception:
                logger.exception(
                    "mcp_loader: server '%s' failed to start — skipping", sid,
                )

    return servers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_enabled_mcp_servers(
    config_path: str | None = None,
    *,
    timeout: float = _DEFAULT_SERVER_TIMEOUT,
) -> MCPConfigBundle:
    """
    Load all enabled MCP servers + global allow/deny settings.

    Servers are started **concurrently**.  Any server that fails or exceeds
    *timeout* seconds is logged and skipped so the agent can proceed with the
    remaining servers.

    Args:
        config_path: Optional explicit path to mcp_servers.yaml. If None,
                     uses PRECURSOR_MCP_SERVERS_FILE or the package default.
        timeout: Per-server startup timeout in seconds (default: 60).

    Returns:
        MCPConfigBundle(servers=[LoadedServer], allow_fn=callable)
    """
    cfg = (
        load_mcp_servers_yaml()
        if config_path is None
        else load_yaml_override(config_path)
    )

    defaults, specs = _prepare_enabled_specs(cfg)
    servers = _load_servers_concurrent(specs, timeout)

    if not servers:
        logger.warning(
            "mcp_loader: no MCP servers loaded — agent will have no tools"
        )

    allow_fn = compile_allow_fn(defaults)
    return MCPConfigBundle(servers=servers, allow_fn=allow_fn)


def load_selected_mcp_servers(
    server_ids: List[str],
    config_path: str | None = None,
    *,
    timeout: float = _DEFAULT_SERVER_TIMEOUT,
) -> MCPConfigBundle:
    """
    Load only the specified MCP servers (if enabled) + global allow/deny settings.

    Same concurrency and fault-tolerance guarantees as
    :func:`load_enabled_mcp_servers`.

    Args:
        server_ids: List of server ids to load (e.g., ["filesystem", "drive"])
        config_path: Optional explicit path to mcp_servers.yaml. If None,
                     uses PRECURSOR_MCP_SERVERS_FILE or the package default.
        timeout: Per-server startup timeout in seconds (default: 60).

    Returns:
        MCPConfigBundle(servers=[LoadedServer], allow_fn=callable)
    """
    cfg = (
        load_mcp_servers_yaml()
        if config_path is None
        else load_yaml_override(config_path)
    )

    defaults, specs = _prepare_enabled_specs(cfg, wanted=set(server_ids))
    servers = _load_servers_concurrent(specs, timeout)

    if not servers:
        logger.warning(
            "mcp_loader: none of the requested servers (%s) could be loaded",
            server_ids,
        )

    allow_fn = compile_allow_fn(defaults)
    return MCPConfigBundle(servers=servers, allow_fn=allow_fn)
