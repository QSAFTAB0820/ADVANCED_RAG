from langchain_core.tools import tool

@tool
def create_support_ticket(details: str) -> str:
    """Creates a customer support ticket given the issues details."""
    print(f"\n--- [Tool Execute] Creating ticket with details: {details} ---")
    ticket_id = "TKT-8989"
    return f"Successfully created support ticket #{ticket_id} with details: {details}"

@tool
def fetch_order_status(order_id: str) -> str:
    """Retrieves order-related information for a given order ID."""
    print(f"\n--- [Tool Execute] Fetching status for order: {order_id} ---")
    orders = {
        "12345": "Shipped - expected delivery on Friday.",
        "99999": "Processing - waiting for inventory.",
    }
    status = orders.get(order_id, "Order not found. Please try another order ID.")
    return f"Order {order_id} status: {status}"

@tool
def schedule_call(date_time: str, topic: str) -> str:
    """Schedules a customer support call at the specified date/time for the given topic."""
    print(f"\n--- [Tool Execute] Scheduling call on {date_time} for '{topic}' ---")
    return f"A support call has been scheduled for {date_time} regarding '{topic}'."
