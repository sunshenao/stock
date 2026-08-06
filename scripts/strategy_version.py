"""Create and verify immutable strategy-version manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VERSION_DIR = PROJECT_ROOT / "codex" / "strategy_versions"
STRATEGY_FILES = (
    "scripts/risk_rules.py",
    "scripts/market_state.py",
    "scripts/daily_risk.py",
    "scripts/execution_model.py",
    "scripts/event_risk.py",
    "scripts/market_data_cache.py",
    "scripts/etf_analyzer.py",
    "scripts/backtest.py",
    "scripts/backtest_current.py",
    "scripts/workflow.py",
    "scripts/selection_guard.py",
    "scripts/decision_contract.py",
    "scripts/ops_contract.py",
    "scripts/universe_history.py",
    "scripts/etf.txt",
    "scripts/etf_universe_history.csv",
    "codex/stock_selection_logic.md",
    "codex/contracts/weekly_evidence.template.json",
    "codex/contracts/account_state.template.json",
    "codex/contracts/ops_packet.template.json",
    "codex/contracts/weekly_review_packet.template.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    version: str,
    *,
    frozen_at: str,
    forward_start: str,
    minimum_forward_weeks: int = 26,
) -> dict:
    files = {}
    for relative in STRATEGY_FILES:
        path = PROJECT_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"strategy file missing: {relative}")
        files[relative] = _sha256(path)
    combined = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items())).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "strategy_version": version,
        "frozen_at": frozen_at,
        "forward_start": forward_start,
        "minimum_forward_weeks": int(minimum_forward_weeks),
        "parameter_change_policy": (
            "Freeze signal, position, execution and universe rules during the "
            "forward window. Emergency bug fixes require a new version."
        ),
        "combined_sha256": combined,
        "files": files,
    }


def write_manifest(manifest: dict) -> Path:
    version = str(manifest["strategy_version"])
    path = DEFAULT_VERSION_DIR / version / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    latest = DEFAULT_VERSION_DIR / "current.json"
    latest.write_text(
        json.dumps({
            "strategy_version": version,
            "manifest": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "combined_sha256": manifest["combined_sha256"],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_current_manifest() -> dict | None:
    latest = DEFAULT_VERSION_DIR / "current.json"
    if not latest.exists():
        return None
    pointer = json.loads(latest.read_text(encoding="utf-8"))
    manifest_path = PROJECT_ROOT / pointer["manifest"]
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_manifest(manifest: dict) -> tuple[bool, list[str]]:
    errors = []
    for relative, expected in manifest.get("files", {}).items():
        path = PROJECT_ROOT / relative
        if not path.exists():
            errors.append(f"missing: {relative}")
            continue
        actual = _sha256(path)
        if actual != expected:
            errors.append(f"hash changed: {relative}")
    return not errors, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the current strategy files")
    parser.add_argument("--version", required=True)
    parser.add_argument("--frozen-at", default=datetime.now().astimezone().isoformat(timespec="seconds"))
    parser.add_argument("--forward-start", required=True)
    parser.add_argument("--minimum-forward-weeks", type=int, default=26)
    args = parser.parse_args()
    path = write_manifest(build_manifest(
        args.version,
        frozen_at=args.frozen_at,
        forward_start=args.forward_start,
        minimum_forward_weeks=args.minimum_forward_weeks,
    ))
    print(path)


if __name__ == "__main__":
    main()
