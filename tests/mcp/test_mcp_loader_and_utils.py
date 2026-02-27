from __future__ import annotations

import os
import types
import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def fake_mcp2py(monkeypatch):
    """
    Provide a minimal fake `mcp2py` module so `precursor.mcp_loader.utils`
    can import `from mcp2py import load as mcp2py_load` without requiring
    the real dependency.
    """
    fake = types.ModuleType("mcp2py")
    calls = []

    def fake_load(cmd: str, auto_auth: bool = True, headers=None):
        calls.append({"cmd": cmd, "auto_auth": auto_auth, "headers": headers})
        # minimal fake client with a 'tools' iterable
        class Client:
            def __init__(self):
                self.tools = []
        return Client()

    fake.load = fake_load
    monkeypatch.setitem(sys.modules, "mcp2py", fake)
    return {"module": fake, "calls": calls}


def _import_loader_and_utils():
    # Import fresh to ensure our fake mcp2py is in place
    utils = importlib.import_module("precursor.mcp_loader.utils")
    loader = importlib.import_module("precursor.mcp_loader.loader")
    return loader, utils


def test_load_enabled_mcp_servers_happy_path(fake_mcp2py, monkeypatch):
    import sys
    # Ensure default config file is used (no override)
    monkeypatch.delenv("PRECURSOR_MCP_SERVERS_FILE", raising=False)

    loader, utils = _import_loader_and_utils()

    # Monkeypatch mcp2py_load in utils to return fake clients with fake tools
    calls = []

    def fake_load(cmd: str, auto_auth: bool = True, headers=None):
        calls.append({"cmd": cmd, "headers": headers})
        class Client:
            def __init__(self):
                # Simulate a couple of tools; names don't matter here
                def t1(): ...
                def t2(): ...
                self.tools = [t1, t2]
        return Client()

    monkeypatch.setattr(utils, "mcp2py_load", fake_load)

    bundle = loader.load_enabled_mcp_servers()

    ids = {s.id for s in bundle.servers}
    # All enabled servers in mcp_servers.yaml should load
    assert "gum" in ids
    assert "drive" in ids
    assert "filesystem" in ids
    assert "coder" in ids

    allow = bundle.allow_fn
    assert allow("drive.search_files")
    assert allow("coder.run_code_task")
    # deny_patterns default empty; everything allowed
    assert allow("anything.anytool")

    # sanity: start_server invoked for all enabled entries
    assert len(calls) == len(ids)


def test_start_server_handles_string_vs_dict_load(monkeypatch):
    loader, utils = _import_loader_and_utils()

    recorded = []

    def fake_load(cmd: str, auto_auth: bool = True, headers=None):
        recorded.append(cmd)
        class Client:
            tools = []
        return Client()

    monkeypatch.setattr(utils, "mcp2py_load", fake_load)

    # Spec with string command
    spec1 = {"id": "gum", "load": "python -m precursor.mcp_servers.gum.server"}
    utils.start_server(spec1)
    # Spec with dict command
    spec2 = {
        "id": "filesystem",
        "load": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        },
    }
    utils.start_server(spec2)

    assert recorded[0] == "python -m precursor.mcp_servers.gum.server"
    # The dict is normalized into a shell-quoted string; check key substrings
    assert "npx" in recorded[1]
    assert "@modelcontextprotocol/server-filesystem" in recorded[1]
    assert "/tmp" in recorded[1]


def test_start_server_timeout(monkeypatch):
    """start_server raises RuntimeError (wrapping TimeoutError) when mcp2py hangs."""
    import time
    _, utils = _import_loader_and_utils()

    def hanging_load(cmd: str, auto_auth: bool = True, headers=None):
        time.sleep(30)  # simulate a hang

    monkeypatch.setattr(utils, "mcp2py_load", hanging_load)

    spec = {"id": "slow", "load": "python -m fake_slow_server"}
    with pytest.raises(RuntimeError, match="did not initialize within"):
        utils.start_server(spec, timeout=1)


def test_load_enabled_skips_failed_servers(fake_mcp2py, monkeypatch, tmp_path):
    """Servers that raise during startup are skipped; others still load."""
    loader, utils = _import_loader_and_utils()

    override_yaml = tmp_path / "mcp_servers.yaml"
    override_yaml.write_text(
        """defaults:
  enabled: true
  allow_patterns: ["*"]
  deny_patterns: []
servers:
  - id: good
    load: "python -m good_server"
    enabled: true
  - id: bad
    load: "python -m bad_server"
    enabled: true
""",
        encoding="utf-8",
    )

    def selective_load(cmd: str, auto_auth: bool = True, headers=None):
        if "bad" in cmd:
            raise ConnectionError("boom")
        class Client:
            tools = []
        return Client()

    monkeypatch.setattr(utils, "mcp2py_load", selective_load)

    bundle = loader.load_enabled_mcp_servers(config_path=str(override_yaml))
    ids = {s.id for s in bundle.servers}
    assert ids == {"good"}  # bad was skipped, good still loaded


def test_apply_env_injects_user_name_for_gum(monkeypatch):
    # Patch get_user_name to a predictable value
    from precursor import config as cfg_pkg  # to ensure package import path
    import precursor.config.loader as cfg_loader
    monkeypatch.setattr(cfg_loader, "get_user_name", lambda: "Michael Ryan")

    # Import utils after patch
    _, utils = _import_loader_and_utils()

    # Ensure clean env keys
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("USER_NAME", raising=False)

    utils.apply_env({"id": "gum", "env": {"FOO": "bar"}})
    assert os.environ.get("FOO") == "bar"
    assert os.environ.get("USER_NAME") == "Michael Ryan"


def test_filesystem_server_uses_dict_load(tmp_path, monkeypatch):
    loader, utils = _import_loader_and_utils()

    # Build a tiny YAML override with only filesystem server (dict load)
    override_yaml = tmp_path / "mcp_servers.yaml"
    override_yaml.write_text(
        """defaults:
  enabled: true
  allow_patterns: ["*"]
  deny_patterns: []
servers:
  - id: filesystem
    load:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/Users/michael/Projects", "/Users/michael/Desktop"]
    enabled: true
""",
        encoding="utf-8",
    )

    recorded = []

    def fake_load(cmd: str, auto_auth: bool = True, headers=None):
        recorded.append(cmd)
        class Client:
            tools = []
        return Client()

    monkeypatch.setattr(utils, "mcp2py_load", fake_load)

    bundle = loader.load_enabled_mcp_servers(config_path=str(override_yaml))

    # One server present
    assert [s.id for s in bundle.servers] == ["filesystem"]
    # The dict is converted to a single shell command string
    assert len(recorded) == 1
    assert "npx" in recorded[0]
    assert "@modelcontextprotocol/server-filesystem" in recorded[0]
    assert "/Users/michael/Desktop" in recorded[0]


