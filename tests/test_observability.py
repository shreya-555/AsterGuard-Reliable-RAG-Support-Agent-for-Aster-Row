import json

from app.observability import TraceLogger


def test_trace_logger_redacts_sensitive_nested_fields(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = TraceLogger(enabled=True, path=path)
    tracer.log(
        "tool_result",
        result={
            "order_id": "ORD-1007",
            "email": "secret@example.test",
            "nested": {"risk_score": 82, "status": "shipped"},
        },
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["result"]["email"] == "[REDACTED]"
    assert record["result"]["nested"]["risk_score"] == "[REDACTED]"
    assert record["result"]["nested"]["status"] == "shipped"


def test_trace_logger_redacts_secrets_inside_user_text(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = TraceLogger(enabled=True, path=path)
    tracer.log(
        "request",
        user_message=(
            "Email me at person@example.com; API key sk-abcdefghijk; "
            "gift card code ABCD-1234-EFGH"
        ),
    )
    text = path.read_text(encoding="utf-8")
    assert "person@example.com" not in text
    assert "sk-abcdefghijk" not in text
    assert "ABCD-1234-EFGH" not in text
