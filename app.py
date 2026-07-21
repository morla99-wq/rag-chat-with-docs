# app.py
import streamlit as st
from generate import answer_question

st.set_page_config(page_title="Chat With Your Docs", page_icon="📄")
st.title("📄 Chat With Your Notes")
st.caption("A RAG app that answers questions from your notes.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your documents...")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander("Sources"):
            for i, hit in enumerate(entry["hits"], start=1):
                st.markdown(f"**[{i}] {hit['source']}** (chunk {hit['chunk_index']})")
                st.text(hit["text"][:300] + "...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Searching documents and generating answer..."):
        answer, hits = answer_question(question)

    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Sources"):
            for i, hit in enumerate(hits, start=1):
                st.markdown(f"**[{i}] {hit['source']}** (chunk {hit['chunk_index']})")
                st.text(hit["text"][:300] + "...")

    st.session_state.history.append({"question": question, "answer": answer, "hits": hits})