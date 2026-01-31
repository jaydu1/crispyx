"""Test configuration to keep loky backend usable in restricted environments.

Some macOS sandboxed environments deny os.sysconf("SC_SEM_NSEMS_MAX"), which
joblib/loky uses to check semaphore limits. We keep loky as the backend but
skip the check when sysconf is not permitted.
"""

from __future__ import annotations


def pytest_sessionstart(session):  # pragma: no cover - test harness setup
    try:
        from joblib.externals.loky import process_executor
    except Exception:
        return

    if getattr(process_executor._check_system_limits, "_crispyx_patched", False):
        return

    original = process_executor._check_system_limits

    def _check_system_limits_safe():
        try:
            return original()
        except PermissionError:
            return None

    _check_system_limits_safe._crispyx_patched = True  # type: ignore[attr-defined]
    process_executor._check_system_limits = _check_system_limits_safe
