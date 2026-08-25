import json
import re
from pathlib import Path
from typing import Any


class OrderLookup:
    """Safe lookup over the supplied mock order snapshot."""
    
    # Canonical format stored in the order dataset.
    ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4}$")
    
    # Accepted human-entered variants (e.g. ORD-1005, ord - 1005,
    # ORD 1005, and ORD_1005). A separator is required so malformed
    # values such as ORD1005 are never silently guessed.
    ORDER_ID_FLEX_PATTERN = re.compile(
        r"^\s*ORD(?:\s*[-_]\s*|\s+)(\d{4})\s*$",
        re.IGNORECASE,
    )

    # Accepted variants when searching inside natural-language text.
    ORDER_ID_SEARCH_PATTERN = re.compile(
        r"\bORD(?:\s*[-_]\s*|\s+)(\d{4})\b",
        re.IGNORECASE,
    )

    NON_SHIPPING_STATUSES = {
        "cancelled",
        "returned",
    }

    def __init__(self, orders_path: str | Path):
        self.orders_path = Path(orders_path)
        self.snapshot_at: str | None = None
        self.orders = self._load_orders()

    def _load_orders(self) -> dict[str, dict[str, Any]]:
        with self.orders_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.snapshot_at = data.get("snapshot_at")
        orders = data.get("orders", [])

        return {
            self.normalize_order_id(order["order_id"]): order
            for order in orders
            if isinstance(order, dict) and order.get("order_id")
        }

    @classmethod
    def normalize_order_id(cls, order_id: str) -> str:
        """
        Normalize common, unambiguous human formatting variations to the
        canonical ORD-1234 representation.
        """
        if not isinstance(order_id, str):
            return ""

        value = order_id.strip()
        if not value:
            return ""

        match = cls.ORDER_ID_FLEX_PATTERN.fullmatch(value)
        if match:
            return f"ORD-{match.group(1)}"

        return value.upper()

    @classmethod
    def extract_order_id(cls, text: str) -> str | None:
        """
        Extract an order ID from natural-language text and return the canonical
        ORD-1234 representation.
        """
        if not isinstance(text, str):
            return None

        match = cls.ORDER_ID_SEARCH_PATTERN.search(text)
        if not match:
            return None

        return f"ORD-{match.group(1)}"

    @classmethod
    def is_valid_order_id(cls, order_id: str) -> bool:
        return bool(
            cls.ORDER_ID_PATTERN.fullmatch(
                cls.normalize_order_id(order_id)
            )
        )

    def lookup(self, order_id: str) -> dict[str, Any]:
        normalized_id = self.normalize_order_id(order_id)

        if not normalized_id:
            return {
                "success": False,
                "error": "missing_order_id",
                "message": "Please provide your order ID.",
            }

        if not self.is_valid_order_id(normalized_id):
            return {
                "success": False,
                "error": "invalid_order_id",
                "message": (
                    "The order ID format is invalid. "
                    "Please provide an ID such as ORD-1007."
                ),
            }

        order = self.orders.get(normalized_id)

        if order is None:
            return {
                "success": False,
                "error": "order_not_found",
                "message": (
                    f"I couldn't find order {normalized_id}. "
                    "Please check the order ID or contact support."
                ),
            }

        return {
            "success": True,
            "order": self._sanitize_order(order),
        }

    def get_snapshot_at(self) -> str | None:
        """Return evaluation snapshot time for deterministic time checks."""
        return self.snapshot_at

    def _sanitize_order(
        self,
        order: dict[str, Any],
    ) -> dict[str, Any]:
        """Whitelist only fields explicitly customer-safe in the dictionary."""
        safe_order = {
            "order_id": order.get("order_id"),
            "membership_tier": order.get("membership_tier"),
            "placed_at": order.get("placed_at"),
            "status": order.get("status"),
            "status_updated_at": order.get("status_updated_at"),
            "shipped_at": order.get("shipped_at"),
            "delivered_at": order.get("delivered_at"),
            "carrier": order.get("carrier"),
            "tracking_number": order.get("tracking_number"),
            "estimated_delivery": order.get("estimated_delivery"),
            "customer_safe_message": order.get("customer_safe_message"),
            "items": [
                {
                    "name": item.get("name"),
                    "quantity": item.get("quantity"),
                    "final_sale": item.get("final_sale"),
                }
                for item in order.get("items", [])
                if isinstance(item, dict)
            ],
        }

        status = str(order.get("status", "")).lower()

        # Status is authoritative. Remove stale logistics for terminal states.
        if status in self.NON_SHIPPING_STATUSES:
            safe_order["carrier"] = None
            safe_order["tracking_number"] = None
            safe_order["estimated_delivery"] = None

        if not order.get("estimated_delivery"):
            safe_order["estimated_delivery"] = None

        return safe_order
