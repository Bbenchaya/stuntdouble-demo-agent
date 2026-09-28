"""Customer support tools — stubs for schema definition.

These tools define the interface (name, args, description) that LangGraph
uses for tool routing. The actual implementations raise NotImplementedError
because StuntDouble intercepts all calls during testing.
"""

from langchain_core.tools import tool


@tool
def get_customer(customer_id: str) -> dict:
    """Look up a customer by their ID."""
    raise NotImplementedError("Connect to real API")


@tool
def list_bills(customer_id: str = "", status: str = "") -> dict:
    """List bills, optionally filtered by customer and/or status."""
    raise NotImplementedError("Connect to real API")


@tool
def get_bill_details(bill_id: str) -> dict:
    """Get detailed information about a specific bill."""
    raise NotImplementedError("Connect to real API")


@tool
def make_payment(bill_id: str, amount: float) -> dict:
    """Submit a payment for a bill."""
    raise NotImplementedError("Connect to real API")


@tool
def get_profile(customer_id: str = "CUST-001") -> dict:
    """Get customer profile information."""
    raise NotImplementedError("Connect to real API")


@tool
def update_profile(
    customer_id: str = "CUST-001", email: str = "", phone: str = "", address: str = ""
) -> dict:
    """Update customer profile fields."""
    raise NotImplementedError("Connect to real API")
