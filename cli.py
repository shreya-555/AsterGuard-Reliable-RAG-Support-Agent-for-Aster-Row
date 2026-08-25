import argparse

from app.agent.state import AgentState
from app.bootstrap import create_agent


def print_sources(sources):
    if not sources:
        return

    print("\nSources:")
    for source in sources:
        print(f"- {source['filename']} — {source['heading']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Aster & Row support agent")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print sanitized structured trace events and write logs/agent.jsonl.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Aster & Row Support Agent")
    print("Type 'exit' to quit")
    if args.debug:
        print("Debug trace enabled (sanitized).")
    print("=" * 60)

    agent = create_agent(debug=args.debug)
    state = AgentState()

    while True:
        try:
            message = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response = agent.handle_message(message, state)

        print(f"\nAgent: {response.answer}")
        print_sources(response.sources)

        print(
            "\nHuman handoff:",
            "YES" if response.handoff else "NO",
        )

        if response.tool_used:
            print("Tool:", response.tool_used)
            if response.tool_arguments:
                print("Tool arguments:", response.tool_arguments)


if __name__ == "__main__":
    main()
