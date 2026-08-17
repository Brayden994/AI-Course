import chromadb
import numpy as np
import chromadb.utils.embedding_functions as ef

db = chromadb.PersistentClient(path="../chroma_db")
memories = db.get_or_create_collection("my_facts")
memories.upsert(
    documents=[
        "I like to code",
        "I can play the piano",
        "I am in the 9th grade"
    ],
    ids = ["Fact1", "Fact2", "Fact3"],
)


print("\nstored: ", memories.count(), "my_facts")

question = "What can i play?"

results = memories.query(query_texts=[question], n_results=1)
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(doc, dist)