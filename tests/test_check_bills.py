"""Check Bills workflow tests — demonstrates fluent builder API."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from stuntdouble import (
    CallRecorder,
    MockToolsRegistry,
    create_mockable_tool_wrapper,
    inject_scenario_metadata,
    register_data_driven,
)
from support_agent.agent import ALL_TOOLS, build_graph
from support_agent.mocked_tools import register_support_agent_mocks
from tests.conftest import (
    FakeChatWithTools,
    build_test_graph,
    make_final_response,
    make_tool_call,
    mock_config,
)

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


class TestCheckBillsHappyPath:
    """Happy path: customer lookup -> bill listing -> summarize."""

    async def test_check_bills_workflow(self, registry, recorder, mock_wrapper):
        registry.mock("get_customer").returns(
            {"id": "CUST-001", "name": "Acme Corp", "tier": "enterprise"}
        )
        registry.mock("list_bills").returns(
            {"bills": [{"id": "B001", "amount": 4500}, {"id": "B002", "amount": 1200}]}
        )

        responses = [
            make_tool_call("get_customer", {"customer_id": "CUST-001"}),
            make_tool_call(
                "list_bills", {"customer_id": "CUST-001", "status": "overdue"}
            ),
            make_final_response("You have 2 overdue bills totaling $5,700."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = mock_config({"scenario_id": "check-bills-test"})

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Check my overdue bills")]},
            config=config,
        )

        recorder.assert_called("get_customer")
        recorder.assert_called("list_bills")
        recorder.assert_call_order("get_customer", "list_bills")
        assert "5,700" in result["messages"][-1].content


class TestCheckBillsInputMatching:
    """Input matching via data-driven scenario with multiple cases."""

    async def test_customer_tier_routing(self, registry, recorder, mock_wrapper):
        register_data_driven(registry, "get_customer")
        register_data_driven(registry, "list_bills")

        scenario = {
            "scenario_id": "vip-customer-test",
            "mocks": {
                "get_customer": [
                    {
                        "input": {"customer_id": "VIP-001"},
                        "output": {
                            "id": "VIP-001",
                            "name": "VIP Corp",
                            "tier": "platinum",
                        },
                    },
                    {
                        "output": {
                            "id": "CUST-999",
                            "name": "Standard Co",
                            "tier": "standard",
                        }
                    },
                ],
                "list_bills": [{"output": {"bills": []}}],
            },
        }

        responses = [
            make_tool_call("get_customer", {"customer_id": "VIP-001"}),
            make_tool_call(
                "list_bills", {"customer_id": "VIP-001", "status": "overdue"}
            ),
            make_final_response("No overdue bills for VIP Corp."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = inject_scenario_metadata({}, scenario)

        await graph.ainvoke(
            {"messages": [HumanMessage(content="Check VIP bills")]},
            config=config,
        )

        recorder.assert_called_with("get_customer", customer_id="VIP-001")


class TestCheckBillsCallRecording:
    """Call recording: detailed assertion patterns."""

    async def test_call_count_and_summary(self, registry, recorder, mock_wrapper):
        registry.mock("get_customer").returns({"id": "CUST-001", "name": "Acme"})
        registry.mock("list_bills").returns({"bills": []})

        responses = [
            make_tool_call("get_customer", {"customer_id": "CUST-001"}),
            make_tool_call("list_bills", {"status": "overdue"}),
            make_final_response("No bills found."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = mock_config({"scenario_id": "recording-test"})

        await graph.ainvoke(
            {"messages": [HumanMessage(content="Any bills?")]},
            config=config,
        )

        recorder.assert_called_once("get_customer")
        recorder.assert_called_once("list_bills")
        recorder.assert_not_called("make_payment")
        assert len(recorder.calls) == 2


class TestCheckBillsEdgeCases:
    """Edge case: empty results."""

    async def test_no_overdue_bills(self, registry, recorder, mock_wrapper):
        registry.mock("get_customer").returns({"id": "CUST-001", "name": "Acme"})
        registry.mock("list_bills").returns({"bills": []})

        responses = [
            make_tool_call("get_customer", {"customer_id": "CUST-001"}),
            make_tool_call("list_bills", {"status": "overdue"}),
            make_final_response("No overdue bills found."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = mock_config({"scenario_id": "empty-bills-test"})

        await graph.ainvoke(
            {"messages": [HumanMessage(content="Check bills")]},
            config=config,
        )

        recorder.assert_called("list_bills")


class TestCheckBillsPythonMockFactories:
    """Python mock factories (mocked_tools.py) reading the same JSON scenarios."""

    async def test_check_bills_json_via_factories(self):
        scenario = json.loads((SCENARIOS_DIR / "check_bills.json").read_text())
        registry = MockToolsRegistry()
        recorder = CallRecorder()
        register_support_agent_mocks(registry, ALL_TOOLS)
        wrapper = create_mockable_tool_wrapper(
            registry, recorder=recorder, tools=ALL_TOOLS, validate_signatures=True
        )
        responses = [
            make_tool_call("get_customer", {"customer_id": "CUST-001"}),
            make_tool_call(
                "list_bills", {"customer_id": "CUST-001", "status": "overdue"}
            ),
            make_final_response("You have overdue bills."),
        ]
        graph = build_graph(
            FakeChatWithTools(messages=iter(responses)), wrapper=wrapper
        )
        config = inject_scenario_metadata({}, scenario)

        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Check overdue bills for customer CUST-001.")
                ]
            },
            config=config,
        )

        recorder.assert_called("get_customer")
        recorder.assert_called("list_bills")
        assert "overdue" in result["messages"][-1].content.lower()
