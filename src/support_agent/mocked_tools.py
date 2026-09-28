"""
Mock factories for per-invocation tool mocking with StuntDouble.

Each factory takes scenario_metadata and returns a callable whose signature
matches the real tool. Scenario JSON under mocks.<tool_name> drives outputs
via InputMatcher and resolve_output.
"""

from __future__ import annotations

from typing import Any, Callable

from stuntdouble import InputMatcher, resolve_output

_input_matcher = InputMatcher()


def _extract_cases(scenario_metadata: dict[str, Any], tool_name: str) -> list[dict]:
    mocks = scenario_metadata.get("mocks", {})
    cases = mocks.get(tool_name, [])
    if not isinstance(cases, list):
        cases = [{"output": cases}] if cases else []
    return cases


def _match_and_resolve(cases: list[dict], kwargs: dict[str, Any]) -> dict:
    for case in cases:
        case_input = case.get("input")
        if _input_matcher.matches(case_input, kwargs):
            output = case.get("output", case)
            return resolve_output(output, input_data=kwargs)
    if cases:
        output = cases[0].get("output", cases[0])
        return resolve_output(output, input_data=kwargs)
    return {"error": "No matching mock case"}


def get_customer_mock(scenario_metadata: dict[str, Any]) -> Callable[..., dict]:
    cases = _extract_cases(scenario_metadata, "get_customer")

    def mock_fn(customer_id: str) -> dict:
        return _match_and_resolve(cases, {"customer_id": customer_id})

    return mock_fn


def list_bills_mock(scenario_metadata: dict[str, Any]) -> Callable[..., dict]:
    cases = _extract_cases(scenario_metadata, "list_bills")

    def mock_fn(customer_id: str = "", status: str = "") -> dict:
        return _match_and_resolve(cases, {"customer_id": customer_id, "status": status})

    return mock_fn


def get_bill_details_mock(scenario_metadata: dict[str, Any]) -> Callable[..., dict]:
    cases = _extract_cases(scenario_metadata, "get_bill_details")

    def mock_fn(bill_id: str) -> dict:
        return _match_and_resolve(cases, {"bill_id": bill_id})

    return mock_fn


def make_payment_mock(scenario_metadata: dict[str, Any]) -> Callable[..., dict]:
    cases = _extract_cases(scenario_metadata, "make_payment")

    def mock_fn(bill_id: str, amount: float) -> dict:
        return _match_and_resolve(cases, {"bill_id": bill_id, "amount": amount})

    return mock_fn


def get_profile_mock(scenario_metadata: dict[str, Any]) -> Callable[..., dict]:
    cases = _extract_cases(scenario_metadata, "get_profile")

    def mock_fn(customer_id: str = "CUST-001") -> dict:
        return _match_and_resolve(cases, {"customer_id": customer_id})

    return mock_fn


def update_profile_mock(scenario_metadata: dict[str, Any]) -> Callable[..., dict]:
    cases = _extract_cases(scenario_metadata, "update_profile")

    def mock_fn(
        customer_id: str = "CUST-001",
        email: str = "",
        phone: str = "",
        address: str = "",
    ) -> dict:
        return _match_and_resolve(
            cases,
            {
                "customer_id": customer_id,
                "email": email,
                "phone": phone,
                "address": address,
            },
        )

    return mock_fn


MOCK_FACTORIES: list[tuple[str, Callable[[dict[str, Any]], Callable[..., dict]]]] = [
    ("get_customer", get_customer_mock),
    ("list_bills", list_bills_mock),
    ("get_bill_details", get_bill_details_mock),
    ("make_payment", make_payment_mock),
    ("get_profile", get_profile_mock),
    ("update_profile", update_profile_mock),
]


def register_support_agent_mocks(registry, tools: list) -> None:
    """Register all support-agent tool mocks with signature validation."""
    by_name = {t.name: t for t in tools}
    for tool_name, factory in MOCK_FACTORIES:
        tool = by_name.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool name for mock registration: {tool_name}")
        registry.register(tool_name, mock_fn=factory, tool=tool)
