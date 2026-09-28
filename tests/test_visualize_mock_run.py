"""Tests for the reusable live mock-run visualizer."""

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VISUALIZER = REPOSITORY_ROOT / "scripts" / "visualize_mock_run.py"


def test_visualizer_streams_mock_calls_and_summary() -> None:
    result = subprocess.run(
        [sys.executable, str(VISUALIZER), "--scenario", "bills", "--delay", "0"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.stderr == ""
    assert "OFFLINE / MOCKED" in result.stdout
    assert "No external LLM or API calls" in result.stdout
    assert "Real LangGraph workflow" in result.stdout
    assert "User request" in result.stdout
    assert "Scenario data" in result.stdout
    assert "[MOCK] get_customer" in result.stdout
    assert "[MOCK] list_bills" in result.stdout
    assert "[PASS] Check overdue bills" in result.stdout
    assert "[DONE] 1 scenario passed; 2 tool calls were mocked" in result.stdout


def test_visualizer_can_force_ansi_colors() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VISUALIZER),
            "--scenario",
            "profile",
            "--delay",
            "0",
            "--color",
            "always",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "\N{ESCAPE}[" in result.stdout
    assert "[PASS] Update profile" in result.stdout
