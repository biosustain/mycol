import webview
import subprocess


if __name__ == "__main__":
    subprocess.Popen(
        ["uv", "run", "streamlit", "run", "app.py", "--server.headless", "true"]
    )
    webview.create_window("Mycol", "http://localhost:8501")
    webview.start()
