"""Make Payment workflow tests — demonstrates data-driven JSON scenario pattern."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from stuntdouble import inject_scenario_metadata, register_data_driven
from tests.conftest import build_test_graph, make_final_response, make_tool_call

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def load_scenario(registry, filename):
    """Load a JSON scenario file and register all mocks as data-driven."""
    scenario = json.loads((SCENARIOS_DIR / filename).read_text())
    for tool_name in scenario["mocks"]:
        register_data_driven(registry, tool_name)
    return scenario


class TestMakePaymentHappyPath:
    """Happy path: get bill details -> make payment -> confirm."""

    async def test_make_payment_workflow(self, registry, recorder, mock_wrapper):
        scenario = load_scenario(registry, "make_payment.json")

        responses = [
            make_tool_call("get_bill_details", {"bill_id": "BILL-001"}),
            make_tool_call("make_payment", {"bill_id": "BILL-001", "amount": 4500}),
            make_final_response(
                "Payment submitted for $4,500. Your payment is processing."
            ),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = inject_scenario_metadata({}, scenario)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Pay my bill BILL-001")]},
            config=config,
        )

        recorder.assert_called("get_bill_details")
        recorder.assert_called("make_payment")
        recorder.assert_call_order("get_bill_details", "make_payment")
        assert "4,500" in result["messages"][-1].content


class TestMakePaymentDynamicPlaceholders:
    """Verify that dynamic placeholders ({{uuid}}, {{now}}, {{input.*}}) resolve."""

    async def test_placeholders_resolve(self, registry, recorder, mock_wrapper):
        scenario = load_scenario(registry, "make_payment.json")

        responses = [
            make_tool_call("get_bill_details", {"bill_id": "BILL-042"}),
            make_tool_call("make_payment", {"bill_id": "BILL-042", "amount": 5000}),
            make_final_response("Payment confirmed."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = inject_scenario_metadata({}, scenario)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Pay bill BILL-042")]},
            config=config,
        )

        # Find the make_payment tool response in the message history
        payment_responses = [
            m
            for m in result["messages"]
            if hasattr(m, "name") and m.name == "make_payment"
        ]
        assert len(payment_responses) == 1

        # Parse the tool response content (it's JSON-stringified)
        content = payment_responses[0].content
        if isinstance(content, str):
            payload = json.loads(content)
        else:
            payload = content

        # payment_id should be a resolved UUID, not the literal placeholder
        assert payload["payment_id"] != "{{uuid}}"
        assert len(payload["payment_id"]) > 8  # UUID-like string

        # processed_at should be a resolved timestamp, not the literal placeholder
        assert payload["processed_at"] != "{{now}}"

        # amount should echo the input value, not the literal placeholder
        assert payload["amount"] == 5000 or str(payload["amount"]) == "5000"


class TestMakePaymentInputMatching:
    """Input matching: high-value ($gte 1000) vs low-value payments."""

    async def test_high_value_payment(self, registry, recorder, mock_wrapper):
        scenario = load_scenario(registry, "make_payment.json")

        responses = [
            make_tool_call("make_payment", {"bill_id": "BILL-001", "amount": 5000}),
            make_final_response("High-value payment processing."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = inject_scenario_metadata({}, scenario)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Pay $5000")]},
            config=config,
        )

        payment_responses = [
            m
            for m in result["messages"]
            if hasattr(m, "name") and m.name == "make_payment"
        ]
        payload = json.loads(payment_responses[0].content)
        # High-value payments ($gte 1000) should have status "processing"
        assert payload["status"] == "processing"

    async def test_low_value_payment(self, registry, recorder, mock_wrapper):
        scenario = load_scenario(registry, "make_payment.json")

        responses = [
            make_tool_call("make_payment", {"bill_id": "BILL-002", "amount": 50}),
            make_final_response("Payment completed."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = inject_scenario_metadata({}, scenario)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Pay $50")]},
            config=config,
        )

        payment_responses = [
            m
            for m in result["messages"]
            if hasattr(m, "name") and m.name == "make_payment"
        ]
        payload = json.loads(payment_responses[0].content)
        # Low-value payments (< 1000) fall through to second case: "completed"
        assert payload["status"] == "completed"


class TestMakePaymentCallRecording:
    """Call recording: verify calls and arguments."""

    async def test_call_recording(self, registry, recorder, mock_wrapper):
        scenario = load_scenario(registry, "make_payment.json")

        responses = [
            make_tool_call("get_bill_details", {"bill_id": "BILL-007"}),
            make_tool_call("make_payment", {"bill_id": "BILL-007", "amount": 2000}),
            make_final_response("Done."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = inject_scenario_metadata({}, scenario)

        await graph.ainvoke(
            {"messages": [HumanMessage(content="Pay bill BILL-007")]},
            config=config,
        )

        recorder.assert_called("make_payment")
        recorder.assert_called_once("make_payment")
        recorder.assert_called_once("get_bill_details")
        recorder.assert_called_with("make_payment", bill_id="BILL-007", amount=2000)
        recorder.assert_called_with("get_bill_details", bill_id="BILL-007")
        recorder.assert_not_called("get_customer")
        assert len(recorder.calls) == 2
