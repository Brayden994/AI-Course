import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import numpy as np
import chromadb.utils.embedding_functions as ef
chatContext = []
chatContextStr = ""
st.title("Talk to an AI")

db = chromadb.PersistentClient(path="../chroma_db")
memories = db.get_or_create_collection("my_facts")
def add_memory(memories, information):
    memories.upsert(
        documents=[
            information
        ],
        ids = [f"Fact{memories.count()+1}"]
    )

add_memory(memories, "I like to play the piano")
load_dotenv()
question = st.chat_input("Ask anything...")
if question:

    all_memories = memories.get(include=["documents"])
    notes = "\n".join(all_memories["documents"])

    chatContextStr = " ".join(chatContext)
    prompt = (
        f"Using these notes about the user: {notes} "
        f"and this conversation context (if any): {chatContextStr} "
        f"answer the user's prompt: {question}. "
        f"Do not mention the notes or that you have access to them. "
        f"If the user provides new information that should be added to your notes, "
        f"append exactly what should be added at the end of your response, "
        f"after a single '#' symbol. "
        f"Only use '#' for storing memories. "
        f"Memories must be short, simple facts about the user, no longer than one sentence "
        f"(e.g., 'user can play the piano'). "
        f"After that memory, add another '#' and include any important conversation context "
        f"that should be saved. This context can be more detailed but must not duplicate of"
        f"existing information. Please do your best to find something to add to context, even if it's just what the user asked."
        f"If there is nothing to add for either section, still include the '#' symbols "
        f"with a blank space after each."
    )
    client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GITHUB_TOKEN"),
    )
    r = client.chat.completions.create(
    model="openai/gpt-oss-120b",

    messages=[{"role": "user", "content": prompt}]
    )
    # print(r) # uncomment to see the whole messy responsse
    answer = r.choices[0].message.content
    parts = answer.split("#")
    while len(parts) < 3:
        parts.append("")

    response = parts[0].strip()
    new_memory = parts[1].strip()
    new_context = parts[2].strip()

    if new_memory:
        add_memory(memories, new_memory)

    if new_context:
        chatContext.append(new_context)

    with st.chat_message("You: "):
        st.write(question)

    with st.chat_message("AI: "):
        st.write(f"{response} {chatContextStr}")

