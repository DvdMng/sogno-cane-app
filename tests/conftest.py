"""Make the package importable when running tests from a source checkout."""
import gc
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _private_home(tmp_path_factory, monkeypatch):
    """Every test gets its own SOGNO_CANE home. The installed app keeps its
    settings and its session in ~/.sogno_cane: a test closing a window must
    never write there. Tests that set their own home still override this."""
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path_factory.mktemp("home")))


@pytest.fixture(autouse=True, scope="session")
def _gc_only_in_the_main_thread():
    """Qt objects must never be finalised by a background thread (UDP
    receiver, engine): the automatic GC runs in whichever thread trips it,
    and a Qt destructor there deadlocks against the main thread — it did,
    in this suite. The app does the same (sogno_cane.ui.gui_gc)."""
    gc.disable()
    yield
    gc.enable()


@pytest.fixture(autouse=True)
def _collect_in_the_main_thread():
    yield
    gc.collect()
