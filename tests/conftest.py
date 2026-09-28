"""Shared test fixtures for StuntDouble sample agent tests."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from stuntdouble import (
    CallRecorder,
    MockToolsRegistry,
    create_mockable_tool_wrapper,
    inject_scenario_metadata,
)
from support_agent.agent import build_graph


class FakeChatWithTools(GenericFakeChatModel):
    """GenericFakeChatModel subclass that accepts bind_tools (no-op).

    The real bind_tools injects tool schemas into the prompt; for testing
    we already craft AIMessage objects with the correct tool_calls, so
    the bind is a harmless pass-through.
    """

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def registry():
    """Fresh MockToolsRegistry for each test."""
    return MockToolsRegistry()


@pytest.fixture
def recorder():
    """Fresh CallRecorder for each test."""
    return CallRecorder()


@pytest.fixture
def mock_wrapper(registry, recorder):
    """StuntDouble wrapper wired to the test's registry and recorder.

    No tools=/signature validation here: fluent ``registry.mock()`` builders
    do not mirror tool parameter signatures. Use explicit factories + tools=
    in tests that need validation (see TestCheckBillsPythonMockFactories).
    """
    return create_mockable_tool_wrapper(registry, recorder=recorder)


def make_tool_call(name: str, args: dict, call_id: str = None) -> AIMessage:
    """Helper to construct an AIMessage with a single tool call."""
    call_id = call_id or f"call_{name}"
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def make_final_response(text: str) -> AIMessage:
    """Helper to construct a terminal AIMessage (no tool calls)."""
    return AIMessage(content=text)


def build_test_graph(responses, mock_wrapper):
    """Build graph with FakeChatWithTools and StuntDouble wrapper."""
    llm = FakeChatWithTools(messages=iter(responses))
    return build_graph(llm, wrapper=mock_wrapper)


def mock_config(scenario_metadata: dict) -> dict:
    """Create a LangGraph config with injected scenario metadata."""
    return inject_scenario_metadata({}, scenario_metadata)
