"""Make the package importable when running tests from a source checkout."""
import gc
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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
