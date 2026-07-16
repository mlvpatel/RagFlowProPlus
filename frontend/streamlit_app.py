"""rag-agentic-2025 Streamlit application entry point."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from frontend.chat_interface import display_chat_interface  # noqa: E402
from frontend.sidebar import display_sidebar  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="rag-agentic-2025", layout="wide")
    st.title("rag-agentic-2025")
    st.caption("Agentic RAG, 2025. Self correcting retrieval with confidence grading.")
    display_sidebar()
    display_chat_interface()


if __name__ == "__main__":
    main()
