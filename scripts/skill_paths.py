"""Shared discovery helpers for local Claude/Codex skills."""
from __future__ import annotations

import os
from pathlib import Path


def _candidate_skill_roots(skill_name: str) -> list[Path]:
    home = Path.home()
    env_key = f"{skill_name.upper().replace('-', '_')}_ROOT"
    candidates = []
    if os.environ.get(env_key):
        candidates.append(Path(os.environ[env_key]).expanduser())
    candidates.extend(
        [
            home / ".claude" / "skills" / skill_name,
            home / ".codex" / "skills" / skill_name,
            home / "AppData" / "Roaming" / "npm" / "node_modules" / skill_name,
        ]
    )
    return candidates


def find_skill_root(skill_name: str) -> Path | None:
    for candidate in _candidate_skill_roots(skill_name):
        if candidate.is_dir():
            return candidate
    return None


def find_skill_scripts(skill_name: str) -> Path | None:
    root = find_skill_root(skill_name)
    if not root:
        return None
    scripts = root / "scripts"
    return scripts if scripts.is_dir() else None


def find_hithink_cli() -> str | None:
    scripts = find_skill_scripts("hithink-market-query")
    if not scripts:
        return None
    cli = scripts / "cli.py"
    return str(cli) if cli.is_file() else None
