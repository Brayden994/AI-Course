import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import uuid

chatContext = []
chatContextStr = ""

st.title("Talk to an AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

db = chromadb.PersistentClient(path="../chroma_db")
memories = db.get_or_create_collection("my_facts")

def add_memory(information):
    memories.add(
        documents=[information],
        ids=[str(uuid.uuid4())]
    )

if memories.count() == 0:
    add_memory("I like to play the piano")

if "chatContext" not in st.session_state:
    st.session_state.chatContext = []

load_dotenv()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):

        if message["role"] == "assistant" and "thinking" in message:
            thinking = st.expander("Thinking...")
            thinking.write(message["thinking"])

        st.write(message["content"])

question = st.chat_input("Ask anything...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    results = memories.query(
        query_texts=[question],
        n_results=min(5, memories.count())
    )

    notes = "\n".join(results["documents"][0])

    chatContextStr = " ".join(st.session_state.chatContext)

    prompt = (
        f"Using these notes about the user: {notes} "
        f"and this conversation context (if any): {chatContextStr} "
        f"answer the user's prompt: {question}. "
        f"Do not mention the notes or that you have access to them. "
        f"If the user provides new information that should be added to your notes, "
        f"append exactly what should be added at the end of your response, "
        f"after a single '#' symbol. "
        f"Only use '#' for storing memories. "
        f"Memories must be short, simple facts about the user, no longer than one sentence. "
        f"After that memory, add another '#' and include any important conversation context "
        f"that should be saved. "
        f"If there is nothing to add for either section, still include the '#' symbols."
    )

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GITHUB_TOKEN"),
    )

    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    with st.chat_message("assistant"):
        thinking = st.expander("Thinking...")
        thinkingArea = thinking.empty()

        fullThinking = ""
        fullText = ""

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta

                reasoning = getattr(delta, "reasoning", None)
                content = getattr(delta, "content", None)

                if reasoning:
                    fullThinking += reasoning
                    thinkingArea.write(fullThinking)

                if content:
                    fullText += content

        parts = fullText.split("#")

        while len(parts) < 3:
            parts.append("")

        response = parts[0].strip()
        new_memory = parts[1].strip()
        new_context = parts[2].strip()

        if new_memory:
            add_memory(new_memory)

        if new_context:
            st.session_state.chatContext.append(new_context)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "thinking": fullThinking
        })

        st.write(response)

    st.rerun()