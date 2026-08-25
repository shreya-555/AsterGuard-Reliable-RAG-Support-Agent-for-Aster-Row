from uuid import uuid4

import streamlit as st

from app.agent.state import AgentState
from app.bootstrap import create_agent


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Aster & Row Support",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =========================================================
# SMALL UI STYLING
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .app-subtitle {
        color: #666;
        margin-top: -10px;
        margin-bottom: 24px;
    }

    .handoff-box {
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        margin-top: 12px;
        border-radius: 5px;
        background: rgba(245, 158, 11, 0.08);
    }

    .tool-box {
        border-left: 4px solid #3b82f6;
        padding: 8px 12px;
        margin-top: 10px;
        border-radius: 5px;
        background: rgba(59, 130, 246, 0.07);
    }

    .history-note {
        color: #777;
        font-size: 0.78rem;
        line-height: 1.25rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD AGENT ONCE
# =========================================================

@st.cache_resource(show_spinner=False)
def load_agent():
    """
    Build expensive embedding/index/LLM components only once.

    Conversation memory is deliberately not cached here. Each saved
    chat in this browser session owns a separate AgentState.
    """
    return create_agent(debug=False)


try:
    with st.spinner("Loading support assistant..."):
        agent = load_agent()
except Exception as exc:
    st.error("The support assistant could not be started.")
    st.code(str(exc))
    st.info(
        "Check your .env configuration, model settings, "
        "and whether the FAISS index has been built."
    )
    st.stop()


# =========================================================
# CHAT HISTORY STATE
# =========================================================

MAX_CHAT_HISTORY = 10


def _new_chat_record(title: str = "New conversation") -> dict:
    """Create one isolated conversation with its own agent memory."""
    return {
        "id": uuid4().hex,
        "title": title,
        "messages": [],
        "agent_state": AgentState(),
    }


def _make_chat_title(prompt: str, max_length: int = 34) -> str:
    """Create a deterministic sidebar title from the first user message."""
    clean = " ".join(prompt.split()).strip()
    if not clean:
        return "New conversation"
    if len(clean) <= max_length:
        return clean
    return clean[: max_length - 1].rstrip() + "…"


# Migrate the older single-chat session state when Streamlit hot-reloads
# after this upgrade. This avoids unexpectedly losing the current chat.
if "conversations" not in st.session_state:
    legacy_messages = st.session_state.get("messages", [])
    legacy_agent_state = st.session_state.get("agent_state", AgentState())

    initial_chat = _new_chat_record()
    initial_chat["messages"] = legacy_messages
    initial_chat["agent_state"] = legacy_agent_state

    if legacy_messages:
        first_user = next(
            (
                item.get("content", "")
                for item in legacy_messages
                if item.get("role") == "user"
            ),
            "",
        )
        initial_chat["title"] = _make_chat_title(first_user)

    st.session_state.conversations = {
        initial_chat["id"]: initial_chat
    }
    st.session_state.conversation_order = [initial_chat["id"]]
    st.session_state.current_conversation_id = initial_chat["id"]

# Defensive recovery in case session state becomes incomplete during a
# development hot reload.
if "conversation_order" not in st.session_state:
    st.session_state.conversation_order = list(
        st.session_state.conversations.keys()
    )

if (
    "current_conversation_id" not in st.session_state
    or st.session_state.current_conversation_id
    not in st.session_state.conversations
):
    if not st.session_state.conversation_order:
        chat = _new_chat_record()
        st.session_state.conversations[chat["id"]] = chat
        st.session_state.conversation_order = [chat["id"]]

    st.session_state.current_conversation_id = (
        st.session_state.conversation_order[0]
    )

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Remove obsolete single-chat keys after migration so there is only one
# source of truth for conversation memory.
st.session_state.pop("messages", None)
st.session_state.pop("agent_state", None)


# =========================================================
# CHAT HISTORY HELPERS
# =========================================================


def current_chat() -> dict:
    return st.session_state.conversations[
        st.session_state.current_conversation_id
    ]


def touch_chat(chat_id: str) -> None:
    """Move a used conversation to the top of the history list."""
    order = st.session_state.conversation_order
    if chat_id in order:
        order.remove(chat_id)
    order.insert(0, chat_id)


def create_new_conversation() -> None:
    chat = _new_chat_record()
    st.session_state.conversations[chat["id"]] = chat
    st.session_state.conversation_order.insert(0, chat["id"])
    st.session_state.current_conversation_id = chat["id"]
    st.session_state.pending_prompt = None

    # Keep the in-memory sidebar bounded. Prefer dropping the oldest empty
    # or least-recently-used chats rather than allowing unbounded growth.
    while len(st.session_state.conversation_order) > MAX_CHAT_HISTORY:
        oldest_id = st.session_state.conversation_order.pop()
        st.session_state.conversations.pop(oldest_id, None)


def switch_conversation(chat_id: str) -> None:
    if chat_id in st.session_state.conversations:
        st.session_state.current_conversation_id = chat_id
        st.session_state.pending_prompt = None
        touch_chat(chat_id)


def delete_conversation(chat_id: str) -> None:
    """Delete one saved in-session chat and safely select another."""
    st.session_state.conversations.pop(chat_id, None)

    if chat_id in st.session_state.conversation_order:
        st.session_state.conversation_order.remove(chat_id)

    if not st.session_state.conversation_order:
        replacement = _new_chat_record()
        st.session_state.conversations[replacement["id"]] = replacement
        st.session_state.conversation_order = [replacement["id"]]

    if st.session_state.current_conversation_id == chat_id:
        st.session_state.current_conversation_id = (
            st.session_state.conversation_order[0]
        )

    st.session_state.pending_prompt = None


def clear_all_conversations() -> None:
    """Clear in-session history without touching the cached model/index."""
    st.session_state.conversations = {}
    st.session_state.conversation_order = []
    create_new_conversation()


# =========================================================
# RENDERING HELPERS
# =========================================================


def render_sources(sources):
    """Render structured citations grouped by source document."""
    if not sources:
        return

    grouped = {}

    for source in sources:
        filename = source.get("filename", "Unknown source")
        heading = source.get("heading", "Unknown section")
        document_id = source.get("document_id")

        if filename not in grouped:
            grouped[filename] = {
                "document_id": document_id,
                "headings": [],
            }

        if heading not in grouped[filename]["headings"]:
            grouped[filename]["headings"].append(heading)

    with st.expander(
        f"Sources ({len(grouped)})",
        expanded=False,
    ):
        for filename, data in grouped.items():
            st.markdown(f"**📄 {filename}**")

            for heading in data["headings"]:
                st.markdown(f"- {heading}")

            if data["document_id"]:
                st.caption(f"Document ID: {data['document_id']}")


def render_response_metadata(message):
    """Render handoff/tool status separately from the answer text."""
    if message.get("handoff"):
        st.markdown(
            """
            <div class="handoff-box">
            <strong>Human support recommended</strong><br>
            This request needs confirmation or action from a
            human support specialist.
            </div>
            """,
            unsafe_allow_html=True,
        )

    tool_used = message.get("tool_used")

    if tool_used:
        friendly_tool = (
            "Order status verified"
            if tool_used == "order_lookup"
            else f"Verified using: {tool_used}"
        )
        st.markdown(
            f"""
            <div class="tool-box">
            <strong>{friendly_tool}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_message(message):
    role = message["role"]
    avatar = "🧑" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

        if role == "assistant":
            render_sources(message.get("sources", []))
            render_response_metadata(message)


def submit_example(prompt):
    st.session_state.pending_prompt = prompt


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("Aster & Row")
    st.caption("Reliable customer support assistant")

    if st.button(
        "＋ New conversation",
        use_container_width=True,
        type="primary",
    ):
        create_new_conversation()
        st.rerun()

    st.markdown("### Chat history")

    # Copy the list because switching/deleting can reorder it.
    for chat_id in list(st.session_state.conversation_order):
        chat = st.session_state.conversations.get(chat_id)
        if not chat:
            continue

        active = chat_id == st.session_state.current_conversation_id
        label = f"● {chat['title']}" if active else chat["title"]

        title_col, delete_col = st.columns([0.84, 0.16])

        with title_col:
            if st.button(
                label,
                key=f"open_chat_{chat_id}",
                use_container_width=True,
                disabled=active,
            ):
                switch_conversation(chat_id)
                st.rerun()

        with delete_col:
            if st.button(
                "×",
                key=f"delete_chat_{chat_id}",
                help="Delete this conversation",
                use_container_width=True,
            ):
                delete_conversation(chat_id)
                st.rerun()

    st.markdown(
        """
        <div class="history-note">
        Chat history is kept only for this browser session and is not
        written to the order dataset or a database.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(st.session_state.conversation_order) > 1:
        if st.button(
            "Clear chat history",
            use_container_width=True,
        ):
            clear_all_conversations()
            st.rerun()

    st.divider()
    st.markdown("### Try an example")

    if st.button("↩️ Return policy", use_container_width=True):
        submit_example("What is the standard return window?")

    if st.button("📦 Track an order", use_container_width=True):
        submit_example("Where is ORD-1007 and when should it arrive?")

    if st.button("⚠️ Policy conflict", use_container_width=True):
        submit_example(
            "Can I put the entire Breeze Tumbler in the dishwasher?"
        )

    if st.button("✋ Cancel an order", use_container_width=True):
        submit_example("Please cancel my order ORD-1001.")

    if st.button("🛠️ Damaged item", use_container_width=True):
        submit_example(
            "A final-sale bag arrived with a broken zipper yesterday. "
            "Am I completely out of luck?"
        )


# =========================================================
# CURRENT CHAT
# =========================================================

chat = current_chat()

st.title("✦ Aster & Row Support")
st.markdown(
    """
    <div class="app-subtitle">
    Ask about orders, shipping, returns, product care,
    warranties, damaged items, or store policies.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# EMPTY-STATE WELCOME
# =========================================================

if not chat["messages"]:
    st.info(
        "Ask a question below or choose one of the "
        "example prompts from the sidebar."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Policy questions**")
        st.caption("Returns, shipping, warranty and TrailPlus.")

    with col2:
        st.markdown("**Order status**")
        st.caption("Check customer-safe status using an order ID.")

    with col3:
        st.markdown("**Human handoff**")
        st.caption("Conflicts and unsupported actions are escalated.")


# =========================================================
# EXISTING MESSAGES IN THE SELECTED CHAT
# =========================================================

for stored_message in chat["messages"]:
    render_message(stored_message)


# =========================================================
# INPUT
# =========================================================

typed_prompt = st.chat_input(
    "Ask Aster & Row support...",
    max_chars=2000,
)

prompt = None

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif typed_prompt:
    prompt = typed_prompt


# =========================================================
# PROCESS MESSAGE
# =========================================================

if prompt:
    # Re-fetch after sidebar interactions to guarantee we write to the
    # currently selected conversation.
    chat = current_chat()
    chat_id = chat["id"]

    if not chat["messages"]:
        chat["title"] = _make_chat_title(prompt)

    user_message = {
        "role": "user",
        "content": prompt,
    }
    chat["messages"].append(user_message)
    touch_chat(chat_id)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("Checking Aster & Row information..."):
                response = agent.handle_message(
                    prompt,
                    chat["agent_state"],
                )

            st.markdown(response.answer)
            render_sources(response.sources)

            assistant_message = {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "handoff": response.handoff,
                "tool_used": response.tool_used,
            }

            render_response_metadata(assistant_message)
            chat["messages"].append(assistant_message)

        except Exception:
            safe_error = (
                "I couldn't process that request right now. "
                "Please try again or contact a human support specialist."
            )

            st.error(safe_error)

            chat["messages"].append(
                {
                    "role": "assistant",
                    "content": safe_error,
                    "sources": [],
                    "handoff": True,
                    "tool_used": None,
                }
            )

    # Refresh once so a newly generated chat title and recent-chat ordering
    # appear in the sidebar immediately. The prompt is already consumed, so
    # this rerun does not call the agent a second time.
    st.rerun()
