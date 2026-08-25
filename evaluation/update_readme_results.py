import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BASELINE = ROOT / "evaluation" / "baseline-results.json"
FINAL = ROOT / "evaluation" / "final-results.json"
START = "<!-- EVAL_RESULTS_START -->"
END = "<!-- EVAL_RESULTS_END -->"


def load(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def score(data, group):
    if not data:
        return "Not run"
    if group == "overall":
        return f"{data['passed']}/{data['total']}"
    item = data.get("category_summary", {}).get(group)
    return f"{item['passed']}/{item['total']}" if item else "—"


def main():
    baseline = load(BASELINE)
    final = load(FINAL)

    table = [
        START,
        "| Category | Baseline | Final |",
        "|---|---:|---:|",
    ]
    for group in ["retrieval", "groundedness", "tool_use", "privacy", "multi_turn", "overall"]:
        table.append(
            f"| {group.replace('_', ' ').title()} | {score(baseline, group)} | {score(final, group)} |"
        )
    table.append(END)
    replacement = "\n".join(table)

    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit("README evaluation markers not found.")

    before = text.split(START, 1)[0]
    after = text.split(END, 1)[1]
    README.write_text(before + replacement + after, encoding="utf-8")
    print("README evaluation table updated.")


if __name__ == "__main__":
    main()
