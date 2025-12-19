from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def scratchpad_test_env(tmp_path, monkeypatch):
    """
    Per-test sandbox for scratchpad (and other DB-backed features).

    - sets PRECURSOR_PROJECTS_FILE to a temp projects.yaml
    - patches precursor.scratchpad.store._get_data_dir to a temp dir
      so we never touch the real user's DB or real config.

    This fixture intentionally lives at tests/ root so all test modules
    (e.g. tests/messaging/*) can reuse it.
    """
    # temp dirs
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    projects_yaml = config_dir / "projects.yaml"
    projects_yaml.write_text(
        """projects:
  - name: "Test Project Alpha"
    description: "A test project used in unit tests."
    agent_enabled: true
  - name: "Misc"
    description: "Fallback"
    agent_enabled: true
""",
        encoding="utf-8",
    )

    # point the loader to this projects.yaml
    monkeypatch.setenv("PRECURSOR_PROJECTS_FILE", str(projects_yaml))

    # patch data dir for the sqlite DB
    import precursor.scratchpad.store as store

    def _fake_get_data_dir() -> Path:
        return data_dir

    monkeypatch.setattr(store, "_get_data_dir", _fake_get_data_dir)

    yield {
        "data_dir": data_dir,
        "config_dir": config_dir,
        "projects_yaml": projects_yaml,
    }


