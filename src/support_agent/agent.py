"""LangGraph agent with injectable tool wrapper for StuntDouble testing."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from support_agent.prompts import SYSTEM_PROMPT
from support_agent.tools import (
    get_bill_details,
    get_customer,
    get_profile,
    list_bills,
    make_payment,
    update_profile,
)

ALL_TOOLS = [
    get_customer,
    list_bills,
    get_bill_details,
    make_payment,
    get_profile,
    update_profile,
]


def build_graph(llm: BaseChatModel, tools=None, wrapper=None):
    """Build the agent graph.

    Args:
        llm: The chat model (real or fake).
        tools: List of tools. Defaults to ALL_TOOLS.
        wrapper: Optional awrap_tool_call wrapper (e.g., from StuntDouble).
    """
    tools = tools or ALL_TOOLS
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesState, config: RunnableConfig):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        ai = await llm_with_tools.ainvoke(messages, config=config)
        return {"messages": [ai]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)

    tool_node_kwargs = {"tools": tools}
    if wrapper is not None:
        tool_node_kwargs["awrap_tool_call"] = wrapper

    builder.add_node("tools", ToolNode(**tool_node_kwargs))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()
