"""Pre-flight: would the Windows bundle install cleanly, or fall back to source builds?

`uv pip install` only discovers a missing wheel on the target platform, so a bad
resolution costs a full Windows CI run to find. This reproduces that check from
any machine in a couple of seconds by evaluating the exported requirements
against the bundle's target environment and looking for a matching wheel in the
lockfile.

It exists because two such failures shipped in a row:
  - torchvision resolved to a win_arm64-only "+d801a34" local build
  - numpy 2.0.2 (capped by cellpose's numpy<2.1) has no cp313 wheel

Usage:
    python scripts/windows/check_wheels.py
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker

ROOT = Path(__file__).resolve().parent.parent.parent

# Must match the interpreters scripts/windows/make_dist.ps1 embeds.
TARGETS = [
    ("main", ROOT, "3.12", ROOT / "uv.lock"),
    ("worker", ROOT / "src" / "training", "3.10", ROOT / "src" / "training" / "uv.lock"),
]

# Pure-Python sdists that have no wheel but build anywhere without a compiler.
ALLOWED_SOURCE_BUILDS = {"proxy-tools", "mobile-sam"}


def win_env(py_version: str) -> dict:
    full = {"3.10": "3.10.11", "3.12": "3.12.10"}[py_version]
    return {
        "sys_platform": "win32", "platform_machine": "AMD64",
        "platform_system": "Windows", "os_name": "nt",
        "python_version": py_version, "python_full_version": full,
        "platform_release": "", "platform_version": "",
        "implementation_name": "cpython", "implementation_version": full,
        "platform_python_implementation": "CPython",
    }


def export(project_dir: Path, py_version: str) -> str:
    out = subprocess.run(
        ["uv", "export", "--no-dev", "--python", py_version, "--no-emit-project"],
        cwd=project_dir, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"uv export failed in {project_dir}:\n{out.stderr}")
    return out.stdout


def selected_packages(requirements: str, env: dict) -> dict[str, str]:
    found = {}
    for line in requirements.splitlines():
        m = re.match(r"^([A-Za-z0-9._-]+)==([^\s;\\]+)(?:\s*;\s*(.+?))?\s*\\?$", line.rstrip())
        if not m:
            continue
        name, version, marker = m.groups()
        if marker and not Marker(marker).evaluate(env):
            continue
        found[name.lower().replace("_", "-")] = version
    return found


def wheel_matches(filename: str, py_version: str) -> bool:
    """Does this wheel's tag apply to CPython <py_version> on win_amd64?"""
    minor = int(py_version.split(".")[1])
    parts = filename[:-4].rsplit("-", 3)
    if len(parts) < 4:
        return False
    py_tag, abi_tag, platform_tag = parts[1], parts[2], parts[3]
    if platform_tag == "any":
        return True
    if platform_tag != "win_amd64":
        return False
    if abi_tag == "none":  # e.g. watchdog-6.0.0-py3-none-win_amd64.whl
        return True
    if abi_tag == "abi3":
        m = re.match(r"cp3(\d+)", py_tag)
        return bool(m) and int(m.group(1)) <= minor
    return py_tag == f"cp3{minor}" and abi_tag.startswith(f"cp3{minor}")


def check(label: str, project_dir: Path, py_version: str, lock_path: Path) -> list[str]:
    env = win_env(py_version)
    packages = selected_packages(export(project_dir, py_version), env)

    lock = tomllib.loads(lock_path.read_text())
    index = {
        (p["name"].lower().replace("_", "-"), p["version"]): (
            [w["url"].rsplit("/", 1)[-1] for w in p.get("wheels", [])],
            p.get("source", {}),
        )
        for p in lock["package"]
    }

    problems = []
    for name, version in sorted(packages.items()):
        entry = index.get((name, version))
        if entry is None:
            problems.append(f"{name}=={version} is not in {lock_path.name}")
            continue
        wheels, source = entry
        if any(wheel_matches(w, py_version) for w in wheels):
            continue
        if name in ALLOWED_SOURCE_BUILDS:
            continue
        reason = "git source" if "git" in source else ("sdist only" if not wheels else
                 f"{len(wheels)} wheels, none for cp3{py_version.split('.')[1]}/win_amd64")
        problems.append(f"{name}=={version} would build from source ({reason})")

    status = "FAIL" if problems else "ok"
    print(f"[{status}] {label}: {len(packages)} packages, python {py_version}, win_amd64")
    for p in problems:
        print(f"       {p}")
    return problems


def main() -> int:
    print("Checking Windows bundle wheel coverage...\n")
    failures = []
    for label, project_dir, py_version, lock_path in TARGETS:
        failures += check(label, project_dir, py_version, lock_path)

    print()
    if failures:
        print(f"{len(failures)} problem(s). The Windows build would compile these on the")
        print("runner, which usually fails in the embeddable environment.")
        return 1
    print("All packages resolve to binary wheels on Windows AMD64.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
