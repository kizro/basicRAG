import pickle
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(
    api_key = ''
)

with open('embeddings.pkl', 'rb') as f:
    embeddings_responses = pickle.load(f)

with open('chunks.pkl', 'rb') as f:
    documents = pickle.load(f)

embeddings = [item.embedding for item in embeddings_responses.data]

def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return np.array(response.data[0].embedding)

k = 5

while True:
    query = input("Enter your query: ")
    query_embedding = get_embedding(query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    retrieved_texts = [documents[i] for i in top_k_indices]
    prompt = f"""You are an expert assistant with access to the following information:

    1. {retrieved_texts[0]}
    2. {retrieved_texts[1]}
    3. {retrieved_texts[2]}
    4. {retrieved_texts[3]}
    5. {retrieved_texts[4]}

    Using the information above, please answer the following question:

    {query}

    Answer:"""
    print(prompt)
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                    {"role": "system", "content": "You are ChatGPT, a large language model."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
        )
    
    print(completion.choices[0].message)