"""Update Profile workflow tests — demonstrates mixed mocking patterns."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from stuntdouble import inject_scenario_metadata, register_data_driven
from tests.conftest import (
    build_test_graph,
    make_final_response,
    make_tool_call,
    mock_config,
)

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def load_scenario(registry, filename):
    """Load a JSON scenario file and register all mocks as data-driven."""
    scenario = json.loads((SCENARIOS_DIR / filename).read_text())
    for tool_name in scenario["mocks"]:
        register_data_driven(registry, tool_name)
    return scenario


class TestUpdateProfileHappyPath:
    """Mixed pattern: data-driven JSON for get_profile, builder for update_profile."""

    async def test_mixed_data_driven_and_builder(
        self, registry, recorder, mock_wrapper
    ):
        # Load scenario but only register get_profile as data-driven
        scenario = json.loads((SCENARIOS_DIR / "update_profile.json").read_text())
        register_data_driven(registry, "get_profile")

        # Use builder for update_profile
        registry.mock("update_profile").returns(
            {"status": "updated", "fields_changed": ["email"]}
        )

        responses = [
            make_tool_call("get_profile", {"customer_id": "CUST-001"}),
            make_tool_call(
                "update_profile", {"customer_id": "CUST-001", "email": "new@acme.com"}
            ),
            make_final_response("Profile updated — email changed to new@acme.com."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = inject_scenario_metadata({}, scenario)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Update my email to new@acme.com")]},
            config=config,
        )

        recorder.assert_called("get_profile")
        recorder.assert_called("update_profile")
        recorder.assert_call_order("get_profile", "update_profile")

        # Verify the update_profile response came from the builder
        update_responses = [
            m
            for m in result["messages"]
            if hasattr(m, "name") and m.name == "update_profile"
        ]
        assert len(update_responses) == 1
        payload = json.loads(update_responses[0].content)
        assert payload["status"] == "updated"
        assert "email" in payload["fields_changed"]


class TestUpdateProfileErrorSimulation:
    """Error simulation: builder returns an error response."""

    async def test_permission_denied_error(self, registry, recorder, mock_wrapper):
        registry.mock("get_profile").returns(
            {"customer_id": "CUST-001", "email": "old@acme.com", "phone": "555-0100"}
        )
        registry.mock("update_profile").returns(
            {"error": "permission_denied", "message": "Cannot update profile"}
        )

        responses = [
            make_tool_call("get_profile", {"customer_id": "CUST-001"}),
            make_tool_call(
                "update_profile", {"customer_id": "CUST-001", "email": "new@acme.com"}
            ),
            make_final_response(
                "Sorry, you don't have permission to update the profile."
            ),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = mock_config({"scenario_id": "update-profile-error"})

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Update email")]},
            config=config,
        )

        recorder.assert_called("update_profile")

        # Verify the error response
        update_responses = [
            m
            for m in result["messages"]
            if hasattr(m, "name") and m.name == "update_profile"
        ]
        payload = json.loads(update_responses[0].content)
        assert payload["error"] == "permission_denied"
        assert payload["message"] == "Cannot update profile"


class TestUpdateProfileScenarioPredicates:
    """Two-phase mock factory: branches on scenario_metadata readonly flag."""

    async def test_readonly_mode_blocks_update(self, registry, recorder, mock_wrapper):
        registry.mock("get_profile").returns(
            {"customer_id": "CUST-001", "email": "old@acme.com", "phone": "555-0100"}
        )

        def readonly_aware_mock(scenario_metadata: dict, config: dict = None):
            readonly = scenario_metadata.get("readonly", False)
            if readonly:
                return lambda **kwargs: {
                    "error": "readonly_mode",
                    "message": "Updates disabled",
                }
            return lambda **kwargs: {"status": "updated", "fields_changed": ["email"]}

        registry.register("update_profile", mock_fn=readonly_aware_mock)

        responses = [
            make_tool_call("get_profile", {"customer_id": "CUST-001"}),
            make_tool_call(
                "update_profile", {"customer_id": "CUST-001", "email": "new@acme.com"}
            ),
            make_final_response("Updates are disabled in readonly mode."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = mock_config({"scenario_id": "readonly-test", "readonly": True})

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Update my email")]},
            config=config,
        )

        recorder.assert_called("update_profile")

        update_responses = [
            m
            for m in result["messages"]
            if hasattr(m, "name") and m.name == "update_profile"
        ]
        payload = json.loads(update_responses[0].content)
        assert payload["error"] == "readonly_mode"
        assert payload["message"] == "Updates disabled"

    async def test_writable_mode_allows_update(self, registry, recorder, mock_wrapper):
        registry.mock("get_profile").returns(
            {"customer_id": "CUST-001", "email": "old@acme.com", "phone": "555-0100"}
        )

        def readonly_aware_mock(scenario_metadata: dict, config: dict = None):
            readonly = scenario_metadata.get("readonly", False)
            if readonly:
                return lambda **kwargs: {
                    "error": "readonly_mode",
                    "message": "Updates disabled",
                }
            return lambda **kwargs: {"status": "updated", "fields_changed": ["email"]}

        registry.register("update_profile", mock_fn=readonly_aware_mock)

        responses = [
            make_tool_call("get_profile", {"customer_id": "CUST-001"}),
            make_tool_call(
                "update_profile", {"customer_id": "CUST-001", "email": "new@acme.com"}
            ),
            make_final_response("Profile updated successfully."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = mock_config({"scenario_id": "writable-test", "readonly": False})

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Update my email")]},
            config=config,
        )

        recorder.assert_called("update_profile")

        update_responses = [
            m
            for m in result["messages"]
            if hasattr(m, "name") and m.name == "update_profile"
        ]
        payload = json.loads(update_responses[0].content)
        assert payload["status"] == "updated"
        assert "email" in payload["fields_changed"]


class TestUpdateProfileFullScenario:
    """Pure data-driven: load update_profile.json for ALL tools."""

    async def test_full_data_driven_workflow(self, registry, recorder, mock_wrapper):
        scenario = load_scenario(registry, "update_profile.json")

        responses = [
            make_tool_call("get_profile", {"customer_id": "CUST-001"}),
            make_tool_call(
                "update_profile", {"customer_id": "CUST-001", "email": "new@acme.com"}
            ),
            make_final_response("Profile updated — email changed."),
        ]
        graph = build_test_graph(responses, mock_wrapper)
        config = inject_scenario_metadata({}, scenario)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Update my email")]},
            config=config,
        )

        recorder.assert_called("get_profile")
        recorder.assert_called("update_profile")
        recorder.assert_call_order("get_profile", "update_profile")
        assert len(recorder.calls) == 2

        # Verify the update response has resolved placeholders
        update_responses = [
            m
            for m in result["messages"]
            if hasattr(m, "name") and m.name == "update_profile"
        ]
        payload = json.loads(update_responses[0].content)
        assert payload["status"] == "updated"
        assert payload["updated_at"] != "{{now}}"
        assert "email" in payload["fields_changed"]
