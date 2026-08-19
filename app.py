import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import uuid
from pypdf import PdfReader
from doc_helper import read_file

st.title("Talk to an AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

db = chromadb.PersistentClient(path="../chroma_db")
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

load_dotenv()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "thinking" in message:
            thinking = st.expander("Thinking...")
            thinking.write(message["thinking"])
        st.write(message["content"])

with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Clear Documents"):
        db.delete_collection("docs")
        st.rerun()
    if st.button("Clear Memories"):
        db.delete_collection("my_facts")
        st.rerun()

question = st.chat_input("Ask anything...", accept_file = True, file_type=["pdf", "txt"])

if question:
    prompt = question.text
    if question.files:
        n = storeDocument(question.files[0])
        st.success(f"Stored {question.files[0].name} as {n} chunks")

if question and prompt:

    st.session_state.messages.append({"role":"user", "content":prompt})

    if memories.count() > 0:
        results = memories.query(
            query_texts=[question.text],
            n_results=min(5, memories.count())
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
        f"Using these notes about the user: {notes}, "
        f"and if applicable, this related document: {document} "
        f"answer the user's prompt: {prompt}. "
        f"If the user provides new information that should be added to your notes, "
        f"append exactly what should be added at the end of your response, "
        f"after a single '#' symbol. "
        f"Only use '#' for storing memories. "
        f"Memories must be short, simple facts about the user, no longer than one sentence. but they mut be important, "
        f"please use memories sparingly only store important personal details, no chat information."
        f"If there is nothing to add for the memories, still include the '#' symbol. "
    )
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GITHUB_TOKEN"),
    )

    apiHistory = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=apiHistory[:-1] + [{"role": "user", "content": prompt}],
        stream=True
    )

    with st.chat_message("assistant"):
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
        parts = fullText.rsplit("#", 1)

        while len(parts) < 2:
            parts.append("")

        response = parts[0].strip()
        new_memory = parts[1].strip()

        if new_memory:
            add_memory(new_memory)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "thinking": fullThinking
        })
        responseArea.write(response)

    st.rerun()