"""Startup script for Risk_Modeler Streamlit application.

This script launches the Streamlit web interface for the Risk_Modeler tool.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit application."""
    app_path = Path(__file__).parent / "streamlit_app.py"

    # Launch Streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
        "--theme.base",
        "light",
        "--theme.primaryColor",
        "#1f77b4",
        "--theme.backgroundColor",
        "#ffffff",
        "--theme.secondaryBackgroundColor",
        "#f0f2f6",
    ]

    print("🚀 Starting Risk_Modeler Web Interface...")
    print("📍 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Risk_Modeler stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
