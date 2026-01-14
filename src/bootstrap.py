import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(root_dir))
    os.chdir(root_dir)

    print("[Bootstrap] Starting Mycol...")
    print("[Bootstrap] Python: ", sys.executable)
    print("[Bootstrap] Root: ", root_dir)
        
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: Streamlit not found in the embedded environment.")
        print("Ensure 'python_main' has dependencies installed.")
        sys.exit(1)
        
    print(f"[Bootstrap] Running from: {root_dir}")
    print(f"[Bootstrap] Python: {sys.executable}")
    
    server_port = 8502
    
    cmd = [
        sys.executable, 
        "-m", "streamlit", 
        "run", 
        "app.py", 
        "--server.headless", "true",
        "--server.port", str(server_port)
    ]
    
    print("[Bootstrap] Starting Streamlit server...")
    
    # Start Streamlit in background
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Force Light Theme ????
    #env["STREAMLIT_THEME_BASE"] = "light"
    #env["STREAMLIT_THEME_PRIMARY_COLOR"] = "#FF4B4B" 
    #env["STREAMLIT_THEME_BACKGROUND_COLOR"] = "#FFFFFF"
    #env["STREAMLIT_THEME_TEXT_COLOR"] = "#004280"
    
    process = subprocess.Popen(cmd, env=env)
    

    # wait 2 seconds to warm up
    time.sleep(2)
    
    try:
        import webview
    except ImportError:
        print("Warning: pywebview not found. Opening system browser...")
        import webbrowser
        webbrowser.open(f"http://localhost:{server_port}")
        process.wait()
        return

    print("[Bootstrap] Opening WebView...")
    window = webview.create_window('Mycol', f'http://localhost:{server_port}')
    webview.start()
    
    print("[Bootstrap] Cleaning up...")
    process.terminate()
    process.wait()

if __name__ == "__main__":
    main()
