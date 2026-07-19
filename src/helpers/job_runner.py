"""Generic subprocess job runner for the cellpose/densenet workers.

Each "job" is a worker subprocess that reads inputs from input.npz and writes
results to output.npz inside a private tempdir. State lives under a string key
in st.session_state so multiple jobs can be in flight side-by-side.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import streamlit as st

ss = st.session_state


# -----------------------------------------------------------------------------
# Worker path resolution (portable / frozen / dev)
# -----------------------------------------------------------------------------


def _resolve_worker_paths() -> dict:
    """Detect runtime mode and resolve paths to the unified worker."""
    here = Path(__file__).resolve()
    src_dir = here.parent.parent          # .../src
    repo_root = src_dir.parent            # repo root

    # Portable mode: bundled python + worker source under bin/
    win_py = repo_root / "bin" / "python_worker" / "python.exe"
    mac_py = repo_root / "bin" / "python_worker" / "bin" / "python"
    portable_py = next((p for p in (win_py, mac_py) if p.exists()), None)
    portable_worker = (
        repo_root / "src" / "training" / "unified_worker.py" if portable_py else None
    )

    # Frozen mode (PyInstaller): packaged worker executable
    frozen_worker = None
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", here.parent))
        frozen_worker = base / "workers" / "unified_worker.exe"

    # Dev mode (uv): training project lives at src/training with its own pyproject
    dev_training = src_dir / "training"
    dev_worker = dev_training / "unified_worker.py"

    return {
        "portable_py": str(portable_py) if portable_py else None,
        "portable_worker": str(portable_worker) if portable_worker else None,
        "frozen_worker": str(frozen_worker) if frozen_worker else None,
        "dev_training": str(dev_training),
        "dev_worker": str(dev_worker),
    }


_PATHS = _resolve_worker_paths()
if _PATHS["portable_py"]:
    print(f"[MyCol] Portable mode detected. Worker: {_PATHS['portable_py']}")


def build_worker_cmd(worker_name: str, in_path, out_path) -> list[str]:
    """Construct the subprocess command for `worker_name` ('finetune' | 'validation' | 'densenet')."""
    in_str, out_str = str(in_path), str(out_path)

    if _PATHS["portable_py"]:
        return [_PATHS["portable_py"], _PATHS["portable_worker"], worker_name, in_str, out_str]
    if getattr(sys, "frozen", False):
        return [_PATHS["frozen_worker"], worker_name, in_str, out_str]
    return [
        "uv", "run", "--project", _PATHS["dev_training"], "python",
        _PATHS["dev_worker"], worker_name, in_str, out_str,
    ]


# -----------------------------------------------------------------------------
# Job lifecycle
# -----------------------------------------------------------------------------


def start_worker_job(
    job_key: str,
    worker_name: str,
    inputs: dict,
    metadata: dict | None = None,
) -> None:
    """Spawn a worker subprocess and stash job state under st.session_state[job_key]."""
    tmpdir = tempfile.mkdtemp(prefix=f"{worker_name}_")
    in_path = Path(tmpdir) / "input.npz"
    out_path = Path(tmpdir) / "output.npz"
    log_path = Path(tmpdir) / f"{worker_name}.log"

    np.savez_compressed(in_path, **inputs)

    cmd = build_worker_cmd(worker_name, in_path, out_path)
    log_file = open(log_path, "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    ss[job_key] = {
        "process": process,
        "log_file": log_file,
        "tmpdir": tmpdir,
        "in_path": str(in_path),
        "out_path": str(out_path),
        "log_path": str(log_path),
        "status": "running",
        **(metadata or {}),
    }


def check_worker_job_status(
    job_key: str,
    on_complete: Callable[[dict, dict], None],
) -> str | None:
    """Poll the job; on success run `on_complete(npz_data, job)`. Returns status string."""
    job = ss.get(job_key)
    if not job:
        return None

    _refresh_log(job)
    returncode = job["process"].poll()
    if returncode is None:
        return "running"

    _close_log(job)
    _refresh_log(job)

    try:
        if returncode != 0:
            return _fail(job, f"Training failed with exit code {returncode}")

        out_path = Path(job["out_path"])
        if not out_path.exists():
            return _fail(job, "Training failed: Output file not found.")

        with np.load(out_path, allow_pickle=True) as data:
            on_complete(data, job)

        job["status"] = "complete"
        return "complete"
    finally:
        shutil.rmtree(job["tmpdir"], ignore_errors=True)


def cancel_worker_job(job_key: str) -> None:
    """Terminate the worker and clean up its tempdir."""
    job = ss.get(job_key)
    if not job:
        return

    proc = job["process"]
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    shutil.rmtree(job["tmpdir"], ignore_errors=True)
    job["status"] = "cancelled"


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _refresh_log(job: dict) -> None:
    log_path = Path(job["log_path"])
    if not log_path.exists():
        return
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
        if content:
            job["log_content"] = content
    except Exception:
        pass


def _close_log(job: dict) -> None:
    lf = job.get("log_file")
    if not lf:
        return
    try:
        lf.flush()
        lf.close()
    except Exception:
        pass


def _fail(job: dict, reason: str) -> str:
    job["status"] = "failed"
    job["error"] = f"{reason}\n\nLog:\n{job.get('log_content', 'No log available')}"
    return "failed"
