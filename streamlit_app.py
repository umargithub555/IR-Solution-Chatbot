import requests
import streamlit as st

API_URL = "http://localhost:8000/chat"




st.title("📄 IR Solution ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the document..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        response = requests.post(API_URL, json={
            "question": prompt,
            "messages": st.session_state.messages
        })

        full_response = response.json()["answer"]
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
