"""Streamlit Community Cloud entrypoint with CPU inference and read-only setup."""
import os
from pathlib import Path
import runpy

os.environ["LESIONLAB_HOSTED"] = "1"
runpy.run_path(str(Path(__file__).resolve().parent / "streamlit_app.py"), run_name="__main__")
