#!/usr/bin/env python3
"""Run local StuntDouble runtime regression checks against a Git commit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from stuntdouble import InputNotMatchedError, MockToolsRegistry
from stuntdouble.mirroring.integrations.llm import LLMProvider
from stuntdouble.mirroring.mcp_client import MCPClient, MCPServerConfig, MCPTool
from stuntdouble.mirroring.models import ToolDefinition

REPOSITORY_URL = "https://github.com/intuit/stunt-double.git"
ISOLATED_CHILD_ENV = "STUNTDOUBLE_COMMIT_CHECK_ISOLATED"
MCP_SERVER = Path(__file__).with_name("local_mcp_test_server.py")

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_MAGENTA = "\033[35m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"


@dataclass(frozen=True)
class RegressionCheck:
    key: str
    name: str
    headline: str
    scenario: str
    behavior: str
    run: Callable[[], str]


class StructuredLocalLLM:
    """Local LangChain-compatible client returning structured content blocks."""

    def invoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(
            content=[
                {"type": "text", "text": '{"id":"CUST-001",'},
                {"type": "text", "text": '"name":"Acme Corp"}'},
            ]
        )


def _commit_hash(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{7,40}", normalized) is None:
        raise argparse.ArgumentTypeError(
            "commit must be a 7-40 character hexadecimal Git hash"
        )
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        type=_commit_hash,
        help="Git commit to test; defaults to the currently installed Git revision",
    )
    parser.add_argument(
        "--only",
        choices=("all", "pin", "builder", "llm", "mcp-timeout", "mcp-noise"),
        default="all",
        help="Run one regression area (default: all)",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="ANSI color mode (default: auto)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show a traceback if a check fails",
    )
    return parser.parse_args()


def _colors_enabled(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return (
        sys.stdout.isatty()
        and os.environ.get("TERM") != "dumb"
        and "NO_COLOR" not in os.environ
    )


def _paint(text: str, *styles: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{''.join(styles)}{text}{ANSI_RESET}"


def _installed_commit() -> str | None:
    direct_url_text = importlib.metadata.distribution("stuntdouble").read_text(
        "direct_url.json"
    )
    if direct_url_text is None:
        return None
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    return commit.lower() if isinstance(commit, str) else None


def _check_installed_commit(expected_commit: str) -> str:
    installed_commit = _installed_commit()
    if installed_commit is None:
        raise AssertionError("stuntdouble is not installed from a direct Git source")
    if not installed_commit.startswith(expected_commit):
        raise AssertionError(
            f"expected {expected_commit}, installed {installed_commit}"
        )
    return f"installed Git revision {installed_commit[:12]}"


def _run_isolated(expected_commit: str, args: argparse.Namespace) -> int:
    requirement = f"stuntdouble @ git+{REPOSITORY_URL}@{expected_commit}"
    command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        requirement,
        "python",
        str(Path(__file__).resolve()),
        "--commit",
        expected_commit,
        "--only",
        args.only,
        "--color",
        args.color,
    ]
    if args.verbose:
        command.append("--verbose")

    print(
        f"[SETUP] Installed revision does not match {expected_commit}; "
        "creating an isolated uv environment",
        flush=True,
    )
    environment = os.environ.copy()
    environment[ISOLATED_CHILD_ENV] = "1"
    return subprocess.run(command, env=environment, check=False).returncode


def _check_builder_behavior() -> str:
    registry = MockToolsRegistry()
    registry.mock("quote").when(tier="premium").echoes_input("customer_id").returns_fn(
        lambda **kwargs: {"discount_percent": 20}
    )
    mock = registry.resolve("quote", {})
    if mock is None:
        raise AssertionError("builder did not register the mock")

    result = mock(tier="premium", customer_id="CUST-001")
    expected = {"discount_percent": 20, "customer_id": "CUST-001"}
    if result != expected:
        raise AssertionError(f"expected {expected}, got {result}")

    try:
        mock(tier="basic", customer_id="CUST-001")
    except InputNotMatchedError:
        pass
    else:
        raise AssertionError("non-matching .when() input was accepted")

    return ".when() rejected a mismatch and .echoes_input() copied customer_id"


def _check_structured_llm_content() -> str:
    provider = LLMProvider(llm_client=StructuredLocalLLM())
    tool = ToolDefinition(
        name="get_customer",
        description="Retrieve a customer",
        input_schema={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    )
    result = provider.generate_with_llm(tool, {"customer_id": "CUST-001"})
    expected = {"id": "CUST-001", "name": "Acme Corp"}
    if result != expected:
        raise AssertionError(f"expected {expected}, got {result}")
    return "two local AIMessage text blocks were joined and parsed as JSON"


def _mcp_config(mode: str, read_timeout: float) -> MCPServerConfig:
    return MCPServerConfig(
        name=f"local-{mode}",
        command=[sys.executable, str(MCP_SERVER)],
        args=["--mode", mode],
        read_timeout=read_timeout,
    )


def _mcp_log(message: str) -> None:
    print(f"       [MCP] {message}", flush=True)


def _log_mcp_start(mode: str, action: str = "Starting") -> None:
    _mcp_log(f"{action} local stdio server (mode={mode})")
    print(
        f"             Command: {sys.executable} {MCP_SERVER} --mode {mode}",
        flush=True,
    )


def _log_mcp_connection(client: MCPClient) -> None:
    process_id = client._process.pid if client._process is not None else "unknown"
    _mcp_log(f"Handshake complete; connected process pid={process_id}")


def _log_mcp_tools(tools: list[MCPTool]) -> None:
    tool_label = "tool" if len(tools) == 1 else "tools"
    _mcp_log(f"tools/list returned {len(tools)} {tool_label}")
    for tool in tools:
        print(f"             • {tool.name} — {tool.description}", flush=True)
        required = tool.input_schema.get("required", [])
        if required:
            print(f"               Required input: {', '.join(required)}", flush=True)


def _check_mcp_timeout_and_retry() -> str:
    config = _mcp_config("silent", read_timeout=0.3)
    client = MCPClient(config)
    _log_mcp_start("silent")
    started = time.monotonic()
    try:
        client.connect()
    except (TimeoutError, RuntimeError):
        pass
    else:
        client.disconnect()
        raise AssertionError("silent server unexpectedly completed the handshake")
    elapsed = time.monotonic() - started
    _mcp_log(f"Handshake timed out as expected after {elapsed:.2f}s")

    if elapsed >= 3:
        raise AssertionError(f"timeout took too long: {elapsed:.2f}s")
    if client._connected or client._process is not None:
        raise AssertionError("failed connection left MCP client resources open")
    _mcp_log("Failed process cleanup confirmed")

    client.config.name = "local-retry"
    client.config.args = ["--mode", "clean"]
    _log_mcp_start("clean", action="Restarting")
    try:
        client.connect()
        _log_mcp_connection(client)
        tools = client.list_tools()
        _log_mcp_tools(tools)
        if [tool.name for tool in tools] != ["echo"]:
            raise AssertionError(f"retry returned unexpected tools: {tools}")
    finally:
        client.disconnect()
        _mcp_log("Stopped local MCP server")

    return f"timed out in {elapsed:.2f}s, cleaned up, then reconnected successfully"


def _check_mcp_noise_handling() -> str:
    client = MCPClient(_mcp_config("noisy", read_timeout=3))
    _log_mcp_start("noisy")
    try:
        tools = client.list_tools()
        _log_mcp_connection(client)
        _log_mcp_tools(tools)
        tool_names = [tool.name for tool in tools]
        if tool_names != ["echo"]:
            raise AssertionError(
                f"notification or stale response leaked through: {tool_names}"
            )
        if not client._stderr_buffer:
            raise AssertionError(
                "the MCP client's stderr drain captured no diagnostics"
            )
        stderr_size = sum(len(line) for line in client._stderr_buffer)
        _mcp_log(
            f"stderr drain captured {len(client._stderr_buffer)} lines "
            f"({stderr_size:,} characters retained)"
        )

        arguments = {"message": "commit regression works"}
        _mcp_log(f"tools/call echo {json.dumps(arguments, sort_keys=True)}")
        result = client.call_tool("echo", arguments)
        _mcp_log(f"tools/call result {json.dumps(result, sort_keys=True)}")
        expected_content = [{"type": "text", "text": "commit regression works"}]
        if result.get("content") != expected_content:
            raise AssertionError(f"unexpected tool result: {result}")
    finally:
        client.disconnect()
        _mcp_log("Stopped local MCP server")

    return "drained >64 KB stderr and ignored notification, stale ID, and log noise"


def _build_checks(expected_commit: str) -> tuple[RegressionCheck, ...]:
    return (
        RegressionCheck(
            key="pin",
            name="Requested commit is installed",
            headline="Commit selection",
            scenario="Confirm the harness is executing the requested StuntDouble revision.",
            behavior="Git revision selection",
            run=partial(_check_installed_commit, expected_commit),
        ),
        RegressionCheck(
            key="builder",
            name="returns_fn honors matching and echoes",
            headline="Conditional function mock",
            scenario="Match premium input, echo a customer ID, and reject a mismatch.",
            behavior="MockBuilder behavior",
            run=_check_builder_behavior,
        ),
        RegressionCheck(
            key="llm",
            name="Structured AIMessage content is parsed",
            headline="Structured local LLM response",
            scenario="Join multiple text blocks into one valid JSON mock response.",
            behavior="LLM response extraction",
            run=_check_structured_llm_content,
        ),
        RegressionCheck(
            key="mcp-timeout",
            name="Silent MCP server times out and can retry",
            headline="MCP timeout, cleanup, and retry",
            scenario="Simulate an unresponsive server, verify cleanup, then reconnect.",
            behavior="Timeout and failed-connect cleanup",
            run=_check_mcp_timeout_and_retry,
        ),
        RegressionCheck(
            key="mcp-noise",
            name="Noisy MCP responses are handled safely",
            headline="MCP stderr flood and response noise",
            scenario="Flood stderr and mix unrelated stdout messages with valid replies.",
            behavior="stderr drain and response-ID matching",
            run=_check_mcp_noise_handling,
        ),
    )


def main() -> int:
    args = _parse_args()
    installed_commit = _installed_commit()
    expected_commit = args.commit or installed_commit
    if expected_commit is None:
        print(
            "[FAIL] No Git commit was provided and the installed package has no Git metadata.",
            file=sys.stderr,
        )
        return 2

    if installed_commit is None or not installed_commit.startswith(expected_commit):
        if os.environ.get(ISOLATED_CHILD_ENV) == "1":
            print(
                f"[FAIL] Isolated environment installed {installed_commit}, expected {expected_commit}.",
                file=sys.stderr,
            )
            return 2
        return _run_isolated(expected_commit, args)

    colors = _colors_enabled(args.color)
    checks = _build_checks(expected_commit)
    selected = [check for check in checks if args.only in ("all", check.key)]

    logging.getLogger("stuntdouble").setLevel(logging.CRITICAL)

    border = "━" * 72
    print(_paint(border, ANSI_CYAN, enabled=colors))
    print(
        _paint(
            "  🔬  STUNTDOUBLE COMMIT RUNTIME REGRESSION HARNESS",
            ANSI_BOLD,
            ANSI_CYAN,
            enabled=colors,
        )
    )
    print(_paint(border, ANSI_CYAN, enabled=colors))
    print(f"  Target commit   {expected_commit}")
    print("  Network         Only needed to fetch an uncached commit")
    print("  External LLM    Not called")
    print("  MCP transport   Dedicated local stdio subprocess")
    print(f"  Checks          {len(selected)}")
    print(_paint(border, ANSI_CYAN, enabled=colors))

    passed = 0
    for position, check in enumerate(selected, start=1):
        print()
        print(
            _paint(
                f"╭─ SCENARIO {position}/{len(selected)}: {check.headline}",
                ANSI_BOLD,
                ANSI_MAGENTA,
                enabled=colors,
            )
        )
        print(f"│  {check.scenario}")
        print(_paint("╰" + "─" * 71, ANSI_MAGENTA, enabled=colors))
        print(
            _paint(
                f"[RUN ] {check.name}",
                ANSI_BOLD,
                ANSI_YELLOW,
                enabled=colors,
            )
        )
        print(
            _paint(
                f"       Behavior: {check.behavior}",
                ANSI_DIM,
                enabled=colors,
            )
        )
        started = time.monotonic()
        try:
            detail = check.run()
        except Exception as error:
            elapsed = time.monotonic() - started
            print(
                _paint(
                    f"[FAIL] {check.name} ({elapsed:.2f}s)",
                    ANSI_BOLD,
                    ANSI_RED,
                    enabled=colors,
                )
            )
            print(f"       {type(error).__name__}: {error}")
            if args.verbose:
                traceback.print_exc(file=sys.stdout)
            return 1

        elapsed = time.monotonic() - started
        print(
            _paint(
                f"[PASS] {check.name} ({elapsed:.2f}s)",
                ANSI_BOLD,
                ANSI_GREEN,
                enabled=colors,
            )
        )
        print(f"       {detail}")
        passed += 1

    check_label = "check" if passed == 1 else "checks"
    print()
    print(
        _paint(
            f"✓ [DONE] {passed} runtime regression {check_label} passed for {expected_commit[:12]}",
            ANSI_BOLD,
            ANSI_GREEN,
            enabled=colors,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
