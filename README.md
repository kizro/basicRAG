# basicRAG

## Intro
This repo contains the implementation of a basic RAG pipeline with a knowledgebase consisting of "Universal and Transferable Adversarial Attacks on Aligned Language Models", "Jailbreaking Black Box Large Language Models in Twenty Queries", and "Representation Engineering: A Top-Down Approach to AI Transparency".

## Flow
Running createEmbeddings.py will aggregate the three papers detailed above, split them into 100 chunks, generate embeddings, and pkl the embeddings as well as word chunks. Running rag.py will allow you to perform RAG on the knowledgebase and you can tweak k manually.

## Time
Watching tutorials and understanding RAG: 30 min
Installations/setup: 10 min
Development: 1 hour

