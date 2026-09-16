"""Entry point for the packaged Mycol bundle.

Release builds have no console, so all output is mirrored to a log file.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

STARTUP_TIMEOUT_S = 120

# Disconnected grace before quitting; long enough to survive a page reload.
# MYCOL_IDLE_TIMEOUT=0 disables.
IDLE_TIMEOUT_S = 30
_IDLE_POLL_S = 3


def _log_path() -> Path:
    """Writable log location. Never inside the bundle - that may be read-only."""
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
    """Ask the OS for an unused port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bundled_model_env(root_dir: Path, env: dict) -> None:
    """Point the model loaders at weights baked in by scripts/fetch_models.py."""
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
            # The tees would otherwise point at a closed file.
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
        # A packaged app never edits its own source.
        "--server.fileWatcherType", "none",
        # Keeps first launch off the network on locked-down machines.
        "--browser.gatherUsageStats", "false",
    ]

    print(f"[Bootstrap] Starting Streamlit on port {port}...")
    # The child needs a real fileno(), which the tee does not have.
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
        _open_ui(url, port, process)
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


def _idle_timeout() -> float:
    """Seconds of no browser connection before quitting; 0 disables."""
    raw = os.environ.get("MYCOL_IDLE_TIMEOUT", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return float(IDLE_TIMEOUT_S)


def _established_connections(port: int) -> int | None:
    """Established TCP connections to `port`, or None if it cannot be determined.

    Streamlit's websocket stays open for a backgrounded tab and closes with a
    shut one, so this tracks open tabs rather than user activity.
    """
    try:
        if sys.platform == "win32":
            cmd = ["netstat", "-ano", "-p", "TCP"]
            needle = f":{port}"
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
                creationflags=0x08000000,  # CREATE_NO_WINDOW
            )
            return sum(
                1 for line in result.stdout.splitlines()
                if needle in line and "ESTABLISHED" in line.upper()
            )
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:ESTABLISHED"],
            capture_output=True, text=True, timeout=10,
        )
        # lsof exits 1 with no output when nothing matches: a real zero.
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        return max(0, len(lines) - 1)  # drop the header
    except Exception:
        return None


def _wait_until_closed(port: int, process: subprocess.Popen) -> None:
    """Block until Streamlit exits, or every browser has been gone a while."""
    timeout = _idle_timeout()
    if timeout <= 0:
        print("[Bootstrap] Idle shutdown disabled; waiting for the server to exit.")
        process.wait()
        return

    print(f"[Bootstrap] Will quit {timeout:.0f}s after the last browser disconnects.")
    seen_browser = False
    gone_since = None

    while process.poll() is None:
        time.sleep(_IDLE_POLL_S)
        count = _established_connections(port)

        if count is None:
            # A failed detector must never take down a working app.
            continue
        if count > 0:
            seen_browser = True
            gone_since = None
            continue
        if not seen_browser:
            # The browser may still be starting.
            continue

        now = time.monotonic()
        if gone_since is None:
            gone_since = now
        elif now - gone_since >= timeout:
            print("[Bootstrap] No browser connected; shutting down.")
            return


def _ui_mode() -> str:
    """"browser" or "window". MYCOL_UI overrides.

    Browser by default: embedded webviews offer no zoom, and WebView2 is one
    more thing that can be missing on Windows.
    """
    override = os.environ.get("MYCOL_UI", "").strip().lower()
    return override if override in ("browser", "window") else "browser"


def _open_in_browser(url: str, port: int, process: subprocess.Popen) -> None:
    """Open the app in the default browser and wait for it to be closed."""
    print(f"[Bootstrap] Opening {url} in the default browser...")
    webbrowser.open(url)
    print("[Bootstrap] Close the browser tab when you are finished.")
    _wait_until_closed(port, process)


def _open_ui(url: str, port: int, process: subprocess.Popen) -> None:
    """Show the app in the browser or a native window."""
    if _ui_mode() == "browser":
        _open_in_browser(url, port, process)
        return

    try:
        import webview

        print("[Bootstrap] Opening WebView...")
        webview.create_window("Mycol", url, width=1400, height=900)
        webview.start()
        return
    except Exception as exc:  # noqa: BLE001 - any backend failure falls back
        print(f"[Bootstrap] WebView unavailable ({exc!r}); using system browser.")

    _open_in_browser(url, port, process)


if __name__ == "__main__":
    raise SystemExit(main())
