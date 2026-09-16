"""Entry point for the packaged Mycol bundle.

The native launcher (tools/launcher) runs this with the bundled interpreter. In
release builds the launcher starts Python with CREATE_NO_WINDOW, so nothing
printed here reaches a console — everything is mirrored to a log file that the
error path points users at.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STARTUP_TIMEOUT_S = 120


def _log_path() -> Path:
    """A writable, conventional location for the session log.

    Never inside the bundle: an app in /Applications, or a folder unpacked under
    Program Files, is not writable by the user running it.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) / "Mycol" if base else None
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Logs" / "Mycol"
    else:
        base = os.environ.get("XDG_STATE_HOME")
        root = Path(base) / "Mycol" if base else Path.home() / ".local" / "state" / "Mycol"

    if root is not None:
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root / "mycol.log"
        except OSError:
            pass
    return Path(tempfile.gettempdir()) / "mycol.log"


class _Tee:
    """Write to the real stream (if any) and the log file."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        if self._stream is not None:
            try:
                self._stream.write(data)
            except (OSError, ValueError):
                pass
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        if self._stream is not None:
            try:
                self._stream.flush()
            except (OSError, ValueError):
                pass
        self._fh.flush()


def _free_port() -> int:
    """Ask the OS for an unused port.

    The port used to be hardcoded, which failed silently whenever something else
    already held it — and with the console hidden the user saw nothing at all.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bundled_model_env(root_dir: Path, env: dict) -> None:
    """Point the model loaders at weights baked in by scripts/fetch_models.py.

    Without this the first segmentation triggers downloads from cellpose.org and
    huggingface.co, which is exactly where locked-down institutional networks
    break an otherwise working install.
    """
    models = root_dir / "models"
    if not models.is_dir():
        print("[Bootstrap] No bundled models/ directory; using network downloads.")
        return

    env["MYCOL_MODELS_DIR"] = str(models)
    print(f"[Bootstrap] Bundled models: {models}")

    cellpose_dir = models / "cellpose"
    if cellpose_dir.is_dir():
        # cellpose.models reads this at import time to set MODEL_DIR.
        env["CELLPOSE_LOCAL_MODELS_PATH"] = str(cellpose_dir)
        print(f"[Bootstrap] Cellpose models: {cellpose_dir}")


def _wait_for_server(port: int, process: subprocess.Popen) -> bool:
    """Poll until Streamlit accepts connections, or it dies trying."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def main() -> int:
    root_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(root_dir))
    os.chdir(root_dir)

    log_file = _log_path()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
        sys.stdout = _Tee(real_stdout, fh)
        sys.stderr = _Tee(real_stderr, fh)
        try:
            return _run(root_dir, log_file, fh)
        finally:
            # Leaving the tees installed would point them at a closed file for
            # the rest of interpreter shutdown.
            sys.stdout, sys.stderr = real_stdout, real_stderr


def _run(root_dir: Path, log_file: Path, log_fh) -> int:
    print("[Bootstrap] Starting Mycol...")
    print(f"[Bootstrap] Python: {sys.executable}")
    print(f"[Bootstrap] Root:   {root_dir}")
    print(f"[Bootstrap] Log:    {log_file}")

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Error: Streamlit not found in the bundled environment.")
        print("The bin/python_main environment is incomplete or was moved.")
        return 1

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    _bundled_model_env(root_dir, env)

    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless", "true",
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        # A packaged app never edits its own source, and the watcher is a
        # startup cost (and a source of spurious reloads) on Windows.
        "--server.fileWatcherType", "none",
        # Keeps first launch off the network on locked-down machines.
        "--browser.gatherUsageStats", "false",
    ]

    print(f"[Bootstrap] Starting Streamlit on port {port}...")
    # The child gets the real file handle, not the tee: subprocess needs a
    # fileno(), and Streamlit's output is only ever read back from the log.
    process = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT
    )

    try:
        if not _wait_for_server(port, process):
            code = process.poll()
            print(f"Error: Streamlit did not start (exit code {code}).")
            print(f"See the log for details: {log_file}")
            return code or 1

        url = f"http://127.0.0.1:{port}"
        print(f"[Bootstrap] Server ready at {url}")
        _open_ui(url, process)
    finally:
        print("[Bootstrap] Cleaning up...")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    return 0


def _open_ui(url: str, process: subprocess.Popen) -> None:
    """Show the app in a native window, falling back to the system browser.

    pywebview raises rather than merely failing to import when no GUI backend is
    available (a missing WebView2 runtime on Windows, for instance), so both
    cases have to be handled — the previous `except ImportError or RuntimeError`
    only ever caught ImportError, because `or` returns its first truthy operand.
    """
    try:
        import webview

        print("[Bootstrap] Opening WebView...")
        webview.create_window("Mycol", url, width=1400, height=900)
        webview.start()
        return
    except Exception as exc:  # noqa: BLE001 - any backend failure falls back
        print(f"[Bootstrap] WebView unavailable ({exc!r}); using system browser.")

    import webbrowser

    webbrowser.open(url)
    # Without a window to close, the Streamlit process defines the app lifetime.
    process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
