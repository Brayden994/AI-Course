import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import numpy as np
import chromadb.utils.embedding_functions as ef

st.title("Talk to an AI")

db = chromadb.PersistentClient(path="./chroma_db")
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
count = 1
while True:
    question = st.text_input("How may I help you?", key = f"key{count}")
    results = memories.query(query_texts=[question], n_results=1)
    notes = "\n".join(results["documents"][0])
    prompt = f"Using these notes about the user: {notes} answer this prompt: {question} the notes may not always be relavent, dont mention that you have notes to the user, and if the user says something that you think should be added you your notes, add exactly what to add to the end of your response separated by a #, but note, you cannot use this symbol outside of storing memories, also please make sure to also include important context in your notes so that you can carry on the conversation"
    client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GITHUB_TOKEN"),
    )
    r = client.chat.completions.create(
    model="llama-3.3-70b-versatile",

    messages=[{"role": "user", "content": prompt}]
)
# print(r) # uncomment to see the whole messy response
    answer = r.choices[0].message.content
    for letter in answer:
        if letter == "#":
            answer = answer.split("#")


    add_memory(memories, answer[1])

    st.write(answer[0])
    count +=1

