"""Run the support agent with mocked tools.

Usage:
    python -m support_agent                     # Default: fake LLM, check_bills workflow
    python -m support_agent make_payment
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import TypedDict

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

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"


class FakeChatWithTools(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


# ── Workflow definitions (fake LLM only: scripted tool calls + final answer) ──


class DemoWorkflow(TypedDict):
    user_message: str
    responses: list[AIMessage]
    scenario_file: str


WORKFLOWS: dict[str, DemoWorkflow] = {
    "check_bills": {
        "user_message": "I need to check my overdue bills for customer CUST-001.",
        "responses": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_customer",
                        "args": {"customer_id": "CUST-001"},
                        "id": "call_1",
                    },
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_bills",
                        "args": {"customer_id": "CUST-001", "status": "overdue"},
                        "id": "call_2",
                    },
                ],
            ),
            AIMessage(
                content="I found 2 overdue bills for Acme Corp:\n"
                "  - BILL-001: $4,500 (14 days overdue)\n"
                "  - BILL-002: $1,200 (7 days overdue)\n"
                "Total overdue: $5,700"
            ),
        ],
        "scenario_file": "check_bills.json",
    },
    "make_payment": {
        "user_message": "I'd like to pay bill BILL-001.",
        "responses": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_bill_details",
                        "args": {"bill_id": "BILL-001"},
                        "id": "call_1",
                    },
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "make_payment",
                        "args": {"bill_id": "BILL-001", "amount": 4500.0},
                        "id": "call_2",
                    },
                ],
            ),
            AIMessage(
                content="Payment of $4,500 submitted for BILL-001.\n"
                "Status: processing. You'll receive a confirmation shortly."
            ),
        ],
        "scenario_file": "make_payment.json",
    },
    "update_profile": {
        "user_message": "Please update my email to new@acme.com.",
        "responses": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_profile",
                        "args": {"customer_id": "CUST-001"},
                        "id": "call_1",
                    },
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "update_profile",
                        "args": {
                            "customer_id": "CUST-001",
                            "email": "new@acme.com",
                        },
                        "id": "call_2",
                    },
                ],
            ),
            AIMessage(
                content="Your email has been updated from old@acme.com to new@acme.com."
            ),
        ],
        "scenario_file": "update_profile.json",
    },
}


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Offline StuntDouble support agent demo"
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        default="check_bills",
        choices=list(WORKFLOWS.keys()),
        help="Demo workflow name",
    )
    return parser.parse_args(argv)


async def run_workflow(workflow_name: str):
    workflow = WORKFLOWS[workflow_name]

    registry = MockToolsRegistry()
    recorder = CallRecorder()
    register_support_agent_mocks(registry, ALL_TOOLS)
    wrapper = create_mockable_tool_wrapper(
        registry, recorder=recorder, tools=ALL_TOOLS, validate_signatures=True
    )

    scenario = json.loads((SCENARIOS_DIR / workflow["scenario_file"]).read_text())
    config = inject_scenario_metadata({}, scenario)

    llm = FakeChatWithTools(messages=iter(workflow["responses"]))
    graph = build_graph(llm, wrapper=wrapper)

    print(f"{'=' * 60}")
    print(f"  Workflow: {workflow_name}")
    print("  LLM: Fake (scripted, offline)")
    print(f"  Scenario: {scenario['scenario_id']}")
    print(f"{'=' * 60}")
    print()
    print(f"  User: {workflow['user_message']}")
    print()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=workflow["user_message"])]},
        config=config,
    )

    print(f"  Agent: {result['messages'][-1].content}")
    print()

    print(f"  {'─' * 56}")
    print("  Tool calls recorded by StuntDouble:")
    print()
    for i, call in enumerate(recorder.calls, 1):
        status = "MOCKED" if call.was_mocked else "REAL"
        print(f"    {i}. {call.tool_name} [{status}]")
        print(f"       args: {call.args}")
        if call.result is not None:
            result_str = json.dumps(call.result, indent=None, default=str)
            if len(result_str) > 80:
                result_str = result_str[:77] + "..."
            print(f"       result: {result_str}")
        print()

    print(f"  Total: {len(recorder.calls)} tool call(s)")
    print(f"{'=' * 60}")


def main():
    args = _parse_args()
    asyncio.run(run_workflow(args.workflow))


if __name__ == "__main__":
    main()
