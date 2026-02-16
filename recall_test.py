import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cuda"

documents = [
    "You can pay fines online using the government portal.",
    "Fines can be paid through online banking services.",
    "Taxes must be declared before March 31.",
    "Айыппұлды онлайн түрде мемлекеттік портал арқылы төлеуге болады.",
    "Салық декларациясы 31 наурызға дейін тапсырылады.",
    "Жасанды интеллект бизнес процестерін өзгертуде."
]

queries = [
    ("How can I pay a fine online?", [0, 1, 3]),
    ("Как оплатить штраф онлайн?", [0, 1, 3]),
    ("Айыппұлды қалай төлеуге болады?", [0, 1, 3]),
    ("Когда нужно сдавать налоговую декларацию?", [2, 4]),
]

model = SentenceTransformer(MODEL_NAME, device=DEVICE)

doc_embeddings = model.encode(documents, convert_to_numpy=True)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

hits = 0

for query, relevant_docs in queries:
    q_emb = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(q_emb, 5)

    if any(i in indices[0] for i in relevant_docs):
        hits += 1

recall_at_5 = hits / len(queries)
print(f"Recall@5: {recall_at_5:.2f}")