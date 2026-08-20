import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import uuid
from pypdf import PdfReader
from doc_helper import read_file

st.html("""
<style>
  [data-testid="stChatMessage"] {
    background-color: transparent;
    border-radius: 18px;
    padding: 10px 16px;
  }
</style>
""")

st.title("Start a Buisness")
st.sidebar.header("Settings")

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

def storeConversation(prompt, answer):
    text = f"Q: {prompt} A: {answer}"
    chunks = chunkIt(text)
    for chunk in chunks:
        add_memory(chunk)


load_dotenv()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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
        f"Using these notes about the user: {notes}, "
        f"and if applicable, this related document: {document} "
        f"answer the user's prompt: {question.text}. "
        f"keep in mind these personality settings,"
        f"your creativity is on a scale from 0 to 10: {creativity}"
        
        f"Your goal is to help the user start a buissness"
        f"if they ask questions related to that, please help "
        f"them to the best of your ability, you can give them ideas"
        f"answer logistics questions, help them with budgeting, ect."
        f"try to keep answers on the short side, go though with the user"
        f"step by step, making sure they are good before moving on."
        f"this is very important; only one step per response"
        f"and only start the step by step guide once they ask for help."
    )
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GITHUB_TOKEN") or st.secrects("GITHUB_TOKEN"),
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

        storeConversation(question.text, fullText)

        st.session_state.messages.append({
            "role": "assistant",
            "content": fullText,
            "thinking": fullThinking
        })
        responseArea.write(fullText)
    st.rerun()