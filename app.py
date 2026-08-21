import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import uuid
from pypdf import PdfReader
from doc_helper import read_file
import tempfile, os

st.set_page_config(page_title="Startup Co-Pilot", page_icon="🐨")

st.html("""
<style>
  [data-testid="stChatMessage"] {
    background-color: transparent;
    border-radius: 18px;
    padding: 10px 16px;
  }
</style>
""")

st.title("Startup Co-Pilot")
st.sidebar.header("Settings")

if "messages" not in st.session_state:
    st.session_state.messages = []
dbPath = os.path.join(tempfile.gettempdir(), "chroma_db")
db = chromadb.PersistentClient(path=dbPath)
memories = db.get_or_create_collection("my_facts")
brain = db.get_or_create_collection("docs")
def chunkIt(text, size=400):
    bits = text.split()
    chunks = []
    current = ""
    for bit in bits:
        if len(current) + len(bit) + 1 > size:
            if current.strip():
                chunks.append(current.strip())
            current = bit + " "
        else:
            current += bit + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks

def storeDocument(file):
    text = read_file(file)
    chunks = chunkIt(text)
    prefix = file.name.replace(" ", "_")
    brain.upsert(
        documents=chunks,
        ids=[
            prefix + "_" + str(uuid.uuid4())
            for _ in range(len(chunks))
        ],
    )
    return len(chunks)

def add_memory(information):
    memories.add(
        documents=[information],
        ids=[str(uuid.uuid4())]
    )

def storeConversation(prompt, answer):
    text = f"Q: {prompt} A: {answer}"
    chunks = chunkIt(text)
    for chunk in chunks:
        add_memory(chunk)


load_dotenv()
avatars = {"assistant": "🐨", "user": "🧑"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatars[message["role"]]):
        if message["role"] == "assistant" and "thinking" in message:
            thinking = st.expander("Thinking...")
            thinking.write(message["thinking"])
        st.write(message["content"])



question = st.chat_input("Ask anything...", accept_file = True, file_type=["pdf", "txt"])

if question:
    prompt = question.text
    if question.files:
        n = storeDocument(question.files[0])
        st.success(f"Stored {question.files[0].name} as {n} chunks")

with st.sidebar:
    creativity = st.slider("Creativity", 0, 10, 5)
    numOfMemories = st.slider("Number of Memories", 0, 5, 2)
    mood = st.slider("Mood", 0.0, 1.0, 0.5)
    messageHistoryNum = st.slider("Message History Number", 0, 10, 5)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Clear Documents"):
        db.delete_collection("docs")
        st.rerun()
    if st.button("Clear Memories"):
        db.delete_collection("my_facts")
        st.rerun()
    st.caption("Current Model: openai/gpt-oss-120b")
    st.caption(f"Number of memories: {memories.count()}")
    st.caption(f"Number of document chunks: {brain.count()}")


if question and prompt:

    st.session_state.messages.append({"role":"user", "content":prompt})

    if memories.count() > 0:
        results = memories.query(
            query_texts=[question.text],
            n_results = max(1, numOfMemories)
        )
        notes = "\n".join(results["documents"][0])
    else:
        notes = ""

    relatedDocument = brain.query(
        query_texts = [prompt],
        n_results = 1
    )
    document = relatedDocument["documents"][0]

    prompt = (
        f"You are KOALA (Knowledgeable Online Assistant for Logic and Analysis), "
        f"an AI assistant whose goal is to help the user start a business. "
        f"Only reveal your name or what it stands for if the user asks.\n\n"

        f"## Context\n"
        f"User notes: {notes}\n"
        f"Related document (if applicable): {document}\n\n"

        f"## Personality settings\n"
        f"Creativity level (0 = strictly practical/conventional, "
        f"10 = highly novel/unconventional ideas): {creativity}/10\n\n"

        f"## Behavior rules\n"
        f"- If the user asks a business-related question (ideas, logistics, "
        f"budgeting, etc.), help them to the best of your ability.\n"
        f"- If the conversation goes off-topic, gently redirect and remind them "
        f"of your purpose. Normal greetings and small talk are always fine.\n"
        f"- Keep answers concise.\n"
        f"- Once the user asks for step-by-step help, walk them through it one "
        f"step at a time. Only give ONE step per response, and confirm the user "
        f"is ready before moving to the next step. Do not start this step-by-step "
        f"mode until the user explicitly asks for help getting started.\n\n"

        f"## User's question\n"
        f"{question.text}"
    )
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GITHUB_TOKEN") or st.secrets("GITHUB_TOKEN"),
    )

    apiHistory = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=apiHistory[-(messageHistoryNum+1):-1] + [{"role": "user", "content": prompt}],
        temperature=mood,
        stream=True
    )
    with st.chat_message("user", avatar="🧑"):
        st.write(question.text)

    with st.chat_message("assistant", avatar="🐨"):
        thinking = st.expander("Thinking...")
        thinkingArea = thinking.empty()
        responseArea = st.empty()
        fullThinking = ""
        fullText = ""
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning", None)
                content = getattr(delta, "content", None)
                if reasoning:
                    fullThinking += reasoning
                if content:
                    fullText += content

        storeConversation(question.text, fullText)

        st.session_state.messages.append({
            "role": "assistant",
            "content": fullText,
            "thinking": fullThinking
        })
        responseArea.write(fullText)
    st.rerun()