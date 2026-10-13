import subprocess
import sys

import pytest

from zeitnot.engine.power import disable_power_throttling


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power throttling only")
def test_disable_for_self_and_child() -> None:
    assert disable_power_throttling() is True
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert disable_power_throttling(child.pid) is True
    finally:
        child.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_noop_elsewhere() -> None:
    assert disable_power_throttling() is False
