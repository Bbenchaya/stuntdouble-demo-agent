#!/usr/bin/env python3
"""Stream StuntDouble mock-tool workflows as they run."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from stuntdouble import (
    CallRecorder,
    MockToolsRegistry,
    create_mockable_tool_wrapper,
    inject_scenario_metadata,
)
from support_agent.agent import ALL_TOOLS, build_graph
from support_agent.mocked_tools import register_support_agent_mocks

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIRECTORY = REPOSITORY_ROOT / "scenarios"

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_BLUE = "\033[34m"
ANSI_GREEN = "\033[32m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_MAGENTA = "\033[35m"


class FakeChatWithTools(GenericFakeChatModel):
    """Scripted chat model that accepts tool binding as a no-op."""

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> FakeChatWithTools:
        return self


@dataclass(frozen=True)
class ScenarioRun:
    key: str
    name: str
    scenario_file: str
    prompt: str
    responses: tuple[AIMessage, ...]


def _tool_call(name: str, arguments: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": f"call_{name}"}],
    )


def _final_response(content: str) -> AIMessage:
    return AIMessage(content=content)


SCENARIOS = (
    ScenarioRun(
        key="bills",
        name="Check overdue bills",
        scenario_file="check_bills.json",
        prompt="Check my overdue bills",
        responses=(
            _tool_call("get_customer", {"customer_id": "CUST-001"}),
            _tool_call("list_bills", {"customer_id": "CUST-001", "status": "overdue"}),
            _final_response("You have two overdue bills totaling $5,700."),
        ),
    ),
    ScenarioRun(
        key="payment",
        name="Make payment",
        scenario_file="make_payment.json",
        prompt="Pay bill BILL-001",
        responses=(
            _tool_call("get_bill_details", {"bill_id": "BILL-001"}),
            _tool_call("make_payment", {"bill_id": "BILL-001", "amount": 4500}),
            _final_response("Payment submitted for $4,500."),
        ),
    ),
    ScenarioRun(
        key="profile",
        name="Update profile",
        scenario_file="update_profile.json",
        prompt="Update my email",
        responses=(
            _tool_call("get_profile", {"customer_id": "CUST-001"}),
            _tool_call(
                "update_profile", {"customer_id": "CUST-001", "email": "new@acme.com"}
            ),
            _final_response("Profile email updated to new@acme.com."),
        ),
    ),
)


def _nonnegative_float(value: str) -> float:
    delay = float(value)
    if delay < 0:
        raise argparse.ArgumentTypeError("delay must be non-negative")
    return delay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("all", *(scenario.key for scenario in SCENARIOS)),
        default="all",
        help="Workflow to run (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=_nonnegative_float,
        default=0.35,
        help="Seconds to pause between streamed events (default: 0.35)",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="ANSI color mode (default: auto)",
    )
    return parser.parse_args()


async def _emit(message: str, delay: float) -> None:
    print(message, flush=True)
    if delay:
        await asyncio.sleep(delay)


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


def _pretty_json(value: Any) -> str:
    formatted = json.dumps(value, default=str, indent=2, sort_keys=True)
    return "\n".join(f"        {line}" for line in formatted.splitlines())


def _banner(scenario_count: int, colors: bool) -> str:
    border = "━" * 62
    return "\n".join(
        (
            _paint(border, ANSI_CYAN, enabled=colors),
            _paint(
                "  🎭  STUNTDOUBLE MOCK WORKFLOW VISUALIZER",
                ANSI_BOLD,
                ANSI_CYAN,
                enabled=colors,
            ),
            _paint(border, ANSI_CYAN, enabled=colors),
            f"  Execution mode  {_paint('OFFLINE / MOCKED', ANSI_BOLD, ANSI_GREEN, enabled=colors)}",
            "  Agent graph     Real LangGraph workflow",
            "  LLM             No external LLM or API calls",
            "  Tools           StuntDouble mocks; no real backend calls",
            f"  Scenarios       {scenario_count}",
            _paint(border, ANSI_CYAN, enabled=colors),
        )
    )


async def _run_scenario(
    scenario: ScenarioRun,
    position: int,
    scenario_count: int,
    delay: float,
    colors: bool,
) -> int:
    heading = f"▶ Scenario {position}/{scenario_count}: {scenario.name}"
    await _emit(f"\n{_paint(heading, ANSI_BOLD, ANSI_MAGENTA, enabled=colors)}", delay)
    await _emit(
        "\n".join(
            (
                f"  User request    {scenario.prompt}",
                f"  Scenario data   scenarios/{scenario.scenario_file}",
                "  Runtime         Loading local mock data",
            )
        ),
        delay,
    )
    scenario_metadata = json.loads(
        (SCENARIOS_DIRECTORY / scenario.scenario_file).read_text()
    )
    registry = MockToolsRegistry()
    recorder = CallRecorder()
    register_support_agent_mocks(registry, ALL_TOOLS)
    wrapper = create_mockable_tool_wrapper(
        registry,
        recorder=recorder,
        tools=ALL_TOOLS,
        validate_signatures=True,
    )
    graph = build_graph(
        FakeChatWithTools(messages=iter(scenario.responses)), wrapper=wrapper
    )

    await _emit(
        _paint("  [RUN] Invoking the agent graph", ANSI_BLUE, enabled=colors),
        delay,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=scenario.prompt)]},
        config=inject_scenario_metadata({}, scenario_metadata),
    )

    for call_number, call in enumerate(recorder.calls, start=1):
        await _emit(
            "\n".join(
                (
                    _paint(
                        f"  [MOCK] {call.tool_name}  ({call_number}/{len(recorder.calls)})",
                        ANSI_BOLD,
                        ANSI_YELLOW,
                        enabled=colors,
                    ),
                    _paint("      Input", ANSI_DIM, enabled=colors),
                    _pretty_json(call.args),
                    _paint("      Resolved result", ANSI_DIM, enabled=colors),
                    _pretty_json(call.result),
                )
            ),
            delay,
        )

    await _emit(
        _paint(
            f"  [AGENT] {result['messages'][-1].content}",
            ANSI_BOLD,
            ANSI_BLUE,
            enabled=colors,
        ),
        delay,
    )
    await _emit(
        _paint(
            f"  [PASS] {scenario.name} — {len(recorder.calls)} mocked calls",
            ANSI_BOLD,
            ANSI_GREEN,
            enabled=colors,
        ),
        delay,
    )
    return len(recorder.calls)


async def _run(selected_scenario: str, delay: float, color_mode: str) -> int:
    selected = [
        scenario for scenario in SCENARIOS if selected_scenario in ("all", scenario.key)
    ]
    colors = _colors_enabled(color_mode)
    await _emit(_banner(len(selected), colors), delay)

    total_calls = 0
    for position, scenario in enumerate(selected, start=1):
        total_calls += await _run_scenario(
            scenario, position, len(selected), delay, colors
        )

    scenario_label = "scenario" if len(selected) == 1 else "scenarios"
    call_label = "call was" if total_calls == 1 else "calls were"
    await _emit(
        "\n"
        + _paint(
            f"✓ [DONE] {len(selected)} {scenario_label} passed; {total_calls} tool {call_label} mocked",
            ANSI_BOLD,
            ANSI_GREEN,
            enabled=colors,
        ),
        0,
    )
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args.scenario, args.delay, args.color))


if __name__ == "__main__":
    raise SystemExit(main())
