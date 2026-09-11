import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "streamlit_app.py"
    print("=" * 65)
    print("  Launching Financial Risk Intelligence Streamlit Dashboard")
    print("  Local URL: http://localhost:8501")
    print("=" * 65)
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.headless=true"
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping Streamlit dashboard.")
