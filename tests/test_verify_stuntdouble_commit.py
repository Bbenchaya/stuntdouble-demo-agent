"""Consumer-level regression tests for the generic commit verifier."""

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPOSITORY_ROOT / "scripts" / "verify_stuntdouble_commit.py"
MERGED_COMMIT = "b6e3807"


def test_commit_runtime_verifier_accepts_commit_hash() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--commit",
            MERGED_COMMIT,
            "--color",
            "never",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.stderr == ""
    assert f"Target commit   {MERGED_COMMIT}" in result.stdout
    assert "[PASS] Requested commit is installed" in result.stdout
    assert "[PASS] returns_fn honors matching and echoes" in result.stdout
    assert "[PASS] Structured AIMessage content is parsed" in result.stdout
    assert "SCENARIO 4/5: MCP timeout, cleanup, and retry" in result.stdout
    assert (
        "Simulate an unresponsive server, verify cleanup, then reconnect."
        in result.stdout
    )
    assert "SCENARIO 5/5: MCP stderr flood and response noise" in result.stdout
    assert (
        "Flood stderr and mix unrelated stdout messages with valid replies."
        in result.stdout
    )
    assert "[MCP] Starting local stdio server (mode=silent)" in result.stdout
    assert "[MCP] Handshake timed out as expected" in result.stdout
    assert "[MCP] Restarting local stdio server (mode=clean)" in result.stdout
    assert "[MCP] Starting local stdio server (mode=noisy)" in result.stdout
    assert "[MCP] tools/list returned 1 tool" in result.stdout
    assert "echo — Return the supplied message" in result.stdout
    assert (
        '[MCP] tools/call echo {"message": "commit regression works"}' in result.stdout
    )
    assert "[MCP] Stopped local MCP server" in result.stdout
    assert "[PASS] Silent MCP server times out and can retry" in result.stdout
    assert "[PASS] Noisy MCP responses are handled safely" in result.stdout
    assert "[DONE] 5 runtime regression checks passed" in result.stdout


def test_commit_runtime_verifier_rejects_non_hash_revision() -> None:
    result = subprocess.run(
        [sys.executable, str(VERIFIER), "--commit", "main"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "commit must be a 7-40 character hexadecimal Git hash" in result.stderr
