"""Chat interface for the RagFlowProPlus Streamlit app.

Shows the answer plus an expandable trace of the agent's reasoning steps
(retrieve, grade, rewrite, generate, self-check), so the self-correction is
visible rather than hidden.
"""

import streamlit as st

from frontend import api_utils

_STEP_LABEL = {
    "retrieve": "Retrieve",
    "grade": "Grade documents",
    "rewrite": "Rewrite query",
    "web_search": "Web fallback",
    "generate": "Generate answer",
    "self_check": "Self-check",
}


def _render_steps(steps, confidence, used_web) -> None:
    if not steps:
        return
    path = ", ".join(_STEP_LABEL.get(s.get("node"), s.get("node", "")) for s in steps)
    header = f"Agent reasoning: {len(steps)} steps"
    if confidence is not None:
        header += f", confidence {confidence:.2f}"
    if used_web:
        header += ", used web fallback"
    with st.expander(header, expanded=False):
        st.caption(path)
        for index, step in enumerate(steps, 1):
            node = _STEP_LABEL.get(step.get("node"), step.get("node", ""))
            detail = {key: value for key, value in step.items() if key != "node"}
            st.markdown(f"**{index}. {node}**")
            if detail:
                st.json(detail, expanded=False)


def display_chat_interface() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("steps"):
                _render_steps(
                    message["steps"], message.get("confidence"), message.get("used_web")
                )

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        model = st.session_state.get("model", "gpt-4o-mini")
        with st.spinner("Agent is reasoning..."):
            try:
                result = api_utils.chat(prompt, st.session_state.session_id, model)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return
        st.session_state.session_id = result.get("session_id")
        answer = result.get("answer", "")
        st.markdown(answer)
        _render_steps(
            result.get("steps", []),
            result.get("confidence"),
            result.get("used_web", False),
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "steps": result.get("steps", []),
            "confidence": result.get("confidence"),
            "used_web": result.get("used_web", False),
        }
    )
