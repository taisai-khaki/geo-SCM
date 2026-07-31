from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "reports" / "structural_regime_completion"
PROOF_LOG = OUT / "reproducibility_proof.log"
VERSION_FILE = OUT / "reproducibility_versions.csv"
SUMMARY_FILE = OUT / "reproducibility_proof_summary.csv"

PRIMARY = [
    sys.executable,
    "scripts/run_final_audited_analysis.py",
    "--bootstrap-reps", "999",
    "--mi-reps", "20",
    "--stability-reps", "100",
    "--seed", "20260731",
]
POST = [
    sys.executable,
    "scripts/run_post_period_sensitivity.py",
    "--bootstrap-reps", "999",
    "--seed", "20260731",
]
PYTEST = [sys.executable, "-m", "pytest", "-q"]
DISPLAY = {
    "pytest": "python -m pytest -q",
    "primary": "python scripts/run_final_audited_analysis.py --bootstrap-reps 999 --mi-reps 20 --stability-reps 100 --seed 20260731",
    "post_period": "python scripts/run_post_period_sensitivity.py --bootstrap-reps 999 --seed 20260731",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparable_files() -> list[Path]:
    excluded = {
        "output_file_hashes.csv",
        "reproduction_log.txt",
        "reproducibility_proof.log",
        "reproducibility_proof_summary.csv",
        "reproducibility_versions.csv",
    }
    return [
        path
        for path in sorted(OUT.iterdir())
        if path.is_file()
        and path.name not in excluded
        and path.suffix.lower() in {".csv", ".png", ".md"}
    ]


def snapshot() -> dict[str, str]:
    return {
        str(path.relative_to(BASE)): sha256(path)
        for path in comparable_files()
    }


def run_command(label: str, command: list[str], log: list[str]) -> int:
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(
        command,
        cwd=BASE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log.extend(
        [
            f"COMMAND: {DISPLAY[label]}",
            f"START_UTC: {started}",
            f"EXIT_CODE: {result.returncode}",
            "STDOUT_BEGIN",
            result.stdout.rstrip(),
            "STDOUT_END",
            "STDERR_BEGIN",
            result.stderr.rstrip(),
            "STDERR_END",
            "",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return result.returncode


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = [
        "REPRODUCIBILITY PROOF",
        f"BASE_DIR: {BASE}",
        f"PROOF_STARTED_UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    records = []
    run_command("primary", PRIMARY, log)
    records.append({"pass": "pass_1", "command": "primary", "exit_code": 0})
    run_command("post_period", POST, log)
    records.append({"pass": "pass_1", "command": "post_period", "exit_code": 0})
    PROOF_LOG.write_text(
        "\n".join(log) + "\nINTERIM_STATUS: primary and post-period commands completed; pytest pending.\n",
        encoding="utf-8",
    )
    interim_versions = [
        {"python": platform.python_version(), "package": "python", "version": platform.python_version()},
    ]
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pytest"]:
        interim_versions.append(
            {
                "python": platform.python_version(),
                "package": package,
                "version": importlib.metadata.version(package),
            }
        )
    pd.DataFrame(interim_versions).to_csv(VERSION_FILE, index=False)
    run_command("pytest", PYTEST, log)
    records.append({"pass": "pass_1", "command": "pytest", "exit_code": 0})
    first = snapshot()

    run_command("primary", PRIMARY, log)
    records.append({"pass": "pass_2", "command": "primary", "exit_code": 0})
    run_command("post_period", POST, log)
    records.append({"pass": "pass_2", "command": "post_period", "exit_code": 0})
    second = snapshot()

    differing = sorted(
        key for key in set(first) | set(second) if first.get(key) != second.get(key)
    )
    deterministic = not differing
    log.extend(
        [
            f"DETERMINISM_STATUS: {'PASS' if deterministic else 'FAIL'}",
            f"FILES_COMPARED: {len(set(first) | set(second))}",
            f"DIFFERING_FILES: {', '.join(differing) if differing else 'none'}",
            f"PROOF_FINISHED_UTC: {datetime.now(timezone.utc).isoformat()}",
        ]
    )
    proof_text = "\n".join(log) + "\n"
    PROOF_LOG.write_text(proof_text, encoding="utf-8")

    versions = [
        {"python": platform.python_version(), "package": "python", "version": platform.python_version()},
    ]
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pytest"]:
        versions.append(
            {
                "python": platform.python_version(),
                "package": package,
                "version": importlib.metadata.version(package),
            }
        )
    pd.DataFrame(versions).to_csv(VERSION_FILE, index=False)

    records.append(
        {
            "pass": "comparison",
            "command": "all comparable output files",
            "exit_code": 0 if deterministic else 1,
            "files_compared": len(set(first) | set(second)),
            "differing_files": len(differing),
        }
    )
    pd.DataFrame(records).to_csv(SUMMARY_FILE, index=False)

    sys.path.insert(0, str(BASE / "scripts"))
    import run_final_audited_analysis as final

    final.write_authoritative_manifest(OUT)
    final.write_reproduction(BASE, OUT)
    if not deterministic:
        raise SystemExit(f"Determinism failed for {differing}")
    print(f"Reproducibility proof passed; compared {len(first)} output files")


if __name__ == "__main__":
    main()
