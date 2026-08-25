from pathlib import Path

import pytest

from app.tools.order_lookup import OrderLookup


ORDERS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "orders.json"
)


def make_lookup():
    return OrderLookup(ORDERS_PATH)


def test_valid_order_lookup():
    lookup = make_lookup()

    result = lookup.lookup("ORD-1007")

    assert result["success"] is True
    assert result["order"]["order_id"] == "ORD-1007"
    assert result["order"]["status"] == "shipped"
    assert result["order"]["estimated_delivery"] == "2026-08-22"


@pytest.mark.parametrize(
    "raw_order_id",
    [
        "ORD-1005",
        "ord-1005",
        "  ord-1005 ",
        "ORD - 1005",
        "ord - 1005",
        "ORD- 1005",
        "ORD -1005",
        "ORD 1005",
        "ORD_1005",
        "   ORD - 1005   ",
    ],
)
def test_common_order_id_variations_are_normalized(raw_order_id):
    lookup = make_lookup()

    result = lookup.lookup(raw_order_id)

    assert result["success"] is True
    assert result["order"]["order_id"] == "ORD-1005"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Where is ORD-1005?", "ORD-1005"),
        ("Where is ord - 1005?", "ORD-1005"),
        ("Please check ORD 1005 for me", "ORD-1005"),
        ("Track ORD_1005", "ORD-1005"),
        ("Can you check ORD-1005?", "ORD-1005"),
    ],
)
def test_order_id_is_extracted_from_natural_text(message, expected):
    assert OrderLookup.extract_order_id(message) == expected


@pytest.mark.parametrize(
    "invalid_order_id",
    [
        "HELLO-123",
        "ORD-10O5",
        "ORD - 10O5",
        "ORD-10055",
        "ORD-ABC",
        "ORD1005",
        "1005",
    ],
)
def test_malformed_order_ids_are_not_guessed(invalid_order_id):
    lookup = make_lookup()

    result = lookup.lookup(invalid_order_id)

    assert result["success"] is False
    assert result["error"] == "invalid_order_id"


def test_ambiguous_order_id_is_not_extracted_from_text():
    assert OrderLookup.extract_order_id("Where is ORD - 10O5?") is None
    assert OrderLookup.extract_order_id("Where is ORD-10055?") is None
    assert OrderLookup.extract_order_id("Where is ORD1005?") is None


def test_order_id_without_separator_is_rejected():
    lookup = make_lookup()

    result = lookup.lookup("ORD1007")

    assert result["success"] is False
    assert result["error"] == "invalid_order_id"


def test_missing_order_id():
    lookup = make_lookup()

    result = lookup.lookup("")

    assert result["success"] is False
    assert result["error"] == "missing_order_id"


def test_unknown_order():
    lookup = make_lookup()

    result = lookup.lookup("ORD-9999")

    assert result["success"] is False
    assert result["error"] == "order_not_found"


def test_internal_fields_are_not_exposed():
    lookup = make_lookup()

    result = lookup.lookup("ORD-1007")

    assert result["success"] is True

    order = result["order"]

    assert "customer" not in order
    assert "internal" not in order
    assert "risk_score" not in order
    assert "warehouse_note" not in order
    assert order["membership_tier"] == "standard"


def test_cancelled_order_does_not_expose_stale_shipping_data():
    lookup = make_lookup()

    result = lookup.lookup("ORD-1004")

    assert result["success"] is True

    order = result["order"]

    assert order["status"] == "cancelled"
    assert order["carrier"] is None
    assert order["tracking_number"] is None
    assert order["estimated_delivery"] is None


def test_returned_order_does_not_expose_stale_eta():
    lookup = make_lookup()

    result = lookup.lookup("ORD-1008")

    assert result["success"] is True

    order = result["order"]

    assert order["status"] == "returned"
    assert order["estimated_delivery"] is None


def test_missing_delivery_estimate_is_not_invented():
    lookup = make_lookup()

    result = lookup.lookup("ORD-1011")

    assert result["success"] is True
    assert result["order"]["status"] == "shipped"
    assert result["order"]["estimated_delivery"] is None


def test_exception_order_is_returned_safely():
    lookup = make_lookup()

    result = lookup.lookup("ORD-1010")

    assert result["success"] is True
    assert result["order"]["status"] == "exception"
    assert result["order"]["estimated_delivery"] is None