"""The SDK's heavy dependencies load only when they are used."""

from __future__ import annotations

import subprocess
import sys

from tests.conftest import run_script_over


def test_importing_the_sdk_and_driving_over_udp_loads_no_livekit_and_no_cli_libraries(
    loopback_edge, tmp_path
):
    """A real loopback LAN connect and preflight keep media and CLI imports lazy. A child
    interpreter over real sockets, on the real clock: what it imports is the point."""
    code, out, err = run_script_over(
        loopback_edge,
        """
        import sys
        robot.get_state()
        robot.close()
        loaded = sorted(m for m in ("livekit", "questionary", "rich") if m in sys.modules)
        print("loaded", ",".join(loaded))
        """,
        tmp_path / "child.err",
    )
    assert code == 0, err
    assert out.strip().endswith("loaded"), f"loaded: {out.strip()}"


def test_the_cli_entry_point_imports_without_its_ui_libraries():
    script = "import sys, menlo.cli; print('rich' in sys.modules, 'questionary' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=True
    )
    assert out.stdout.strip() == "False False"
