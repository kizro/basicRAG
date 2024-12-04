import os
from openai import OpenAI
import pickle

client = OpenAI(
    api_key = ''
)

files = ['f1.txt', 'f2.txt', 'f3.txt']

documents = ""

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        documents += text

def split_into_chunks(text, num_chunks):
    words = text.split()
    avg_chunk_size = len(words) // num_chunks
    chunks = [" ".join(words[i * avg_chunk_size:(i + 1) * avg_chunk_size]) for i in range(num_chunks - 1)]
    chunks.append(" ".join(words[(num_chunks - 1) * avg_chunk_size:])) 
    return chunks

chunks = split_into_chunks(documents, 100)

def get_embeddings(text, model="text-embedding-3-large"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response

embeddings = get_embeddings(chunks)
    
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
    
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)