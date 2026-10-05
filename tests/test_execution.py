import mmap
import sys
from types import ModuleType

import pytest

from nanochat.execution import execute_code


def test_execution_captures_output():
    result = execute_code("import sys; print('hello'); print('warning', file=sys.stderr)")
    assert result.success
    assert result.stdout == "hello\n"
    assert result.stderr == "warning\n"


def test_execution_does_not_inherit_parent_state(monkeypatch):
    monkeypatch.setitem(sys.modules, "parent_only_module", ModuleType("parent_only_module"))
    with mmap.mmap(-1, 512 * 1024 * 1024):
        result = execute_code(
            "import sys; assert 'parent_only_module' not in sys.modules; "
            "assert 'torch' not in sys.modules; print(len(bytearray(1024 * 1024)))"
        )
    assert result.success, result
    assert result.stdout == "1048576\n"


def test_execution_reports_exception():
    result = execute_code("raise ValueError('bad value')")
    assert not result.success
    assert result.error == "ValueError: bad value"
    assert not result.timeout


def test_execution_reports_timeout():
    result = execute_code("while True: pass", timeout=0.1)
    assert not result.success
    assert result.timeout


def test_execution_kills_worker_that_disables_alarm():
    result = execute_code(
        "import signal; signal.setitimer(signal.ITIMER_REAL, 0)\nwhile True: pass",
        timeout=0.1,
    )
    assert not result.success
    assert result.timeout
    assert result.error == "Execution timed out (process killed)"


def test_execution_reports_worker_exit_without_result():
    result = execute_code("import os; os._exit(1)")
    assert not result.success
    assert not result.timeout
    assert result.error == "Execution failed (no result returned)"


@pytest.mark.skipif(sys.platform == "darwin", reason="Memory limits are skipped on macOS")
def test_execution_enforces_memory_limit():
    result = execute_code("data = bytearray(512 * 1024 * 1024)")
    assert not result.success
    assert result.memory_exceeded


def test_execution_allows_disabling_memory_limit():
    result = execute_code("print(42)", maximum_memory_bytes=None)
    assert result.success, result
    assert result.stdout == "42\n"