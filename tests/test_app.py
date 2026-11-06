from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
