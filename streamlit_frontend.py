# streamlit_frontend.py
import os
import re

import streamlit as st
from dotenv import load_dotenv, find_dotenv

from backend import CBTChatbot
from langchain.callbacks.base import BaseCallbackHandler

# ─── 0) Page & env ────────────────────────────────────────────────────────────
st.set_page_config(page_title="CBT Chatbot", page_icon="💬")
load_dotenv(find_dotenv())

# ─── 1) Streaming callback ─────────────────────────────────────────────────────
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.buffer = ""
    def on_llm_new_token(self, token: str, **kwargs):
        self.buffer += token
        # overwrite each token
        self.placeholder.write(self.buffer)

# ─── 2) Init bot & history ─────────────────────────────────────────────────────
if "cbt_bot" not in st.session_state:
    st.session_state.cbt_bot = CBTChatbot()
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)

# ─── 3) Render chat ────────────────────────────────────────────────────────────
st.title("💬 CBT Chatbot")
for role, txt in st.session_state.history:
    with st.chat_message(role):
        st.markdown(txt)

# ─── 4) Input ───────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type here…")
if user_input:
    # record user
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # stream bot reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        handler = StreamlitCallbackHandler(placeholder)
        # call .stream() with your handler
        reply = st.session_state.cbt_bot.stream(
            user_input,
            callbacks=[handler]
        )
        # if streaming never fired, show full reply
        if not handler.buffer:
            placeholder.write(reply)
    # append final text (either buffered or full reply)
    final = handler.buffer or reply
    st.session_state.history.append(("assistant", final))
