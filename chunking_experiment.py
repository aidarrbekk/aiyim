import nltk
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('punkt_tab')

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(MODEL_NAME, device=DEVICE)

documents = [
    """Artificial intelligence is transforming modern business by automating processes and improving efficiency.
    Companies use AI to analyze data and improve decision making.""",

    """Taxes must be declared annually before the end of March.
    Failure to declare taxes may result in financial penalties.""",

    """Айыппұлды онлайн мемлекеттік портал арқылы төлеуге болады.
    Салық декларациясы 31 наурызға дейін тапсырылады."""
]

queries = [
    ("How does AI help companies?", [0]),
    ("When must taxes be declared?", [1]),
    ("Как оплатить штраф онлайн?", [2])
]

def fixed_chunking(docs, size=512):
    chunks = []
    for doc in docs:
        for i in range(0, len(doc), size):
            chunks.append(doc[i:i+size])
    return chunks

def sentence_chunking(docs, overlap=1):
    chunks = []
    for doc in docs:
        sentences = nltk.sent_tokenize(doc)
        for i in range(len(sentences)):
            chunk = sentences[i]
            if i+overlap < len(sentences):
                chunk += " " + sentences[i+overlap]
            chunks.append(chunk)
    return chunks

def semantic_chunking(docs):
    chunks = []
    for doc in docs:
        sentences = nltk.sent_tokenize(doc)
        buffer = sentences[0]
        for sent in sentences[1:]:
            emb = model.encode([buffer, sent])
            sim = np.dot(emb[0], emb[1]) / (
                np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])
            )
            if sim > 0.6:
                buffer += " " + sent
            else:
                chunks.append(buffer)
                buffer = sent
        chunks.append(buffer)
    return chunks

def recall_at_5(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    hits = 0
    for q, relevant_doc_ids in queries:
        q_emb = model.encode([q], convert_to_numpy=True)
        _, idx = index.search(q_emb, 5)
        if any(i in idx[0] for i in relevant_doc_ids):
            hits += 1

    return hits / len(queries)

strategies = {
    "Fixed 512": fixed_chunking(documents),
    "Sentence + Overlap": sentence_chunking(documents),
    "Semantic": semantic_chunking(documents)
}

print("\nCHUNKING RESULTS\n")

for name, chunks in strategies.items():
    avg_size = np.mean([len(c) for c in chunks])
    recall = recall_at_5(chunks)
    print(f"{name}")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Average chunk size: {avg_size:.2f}")
    print(f"  Recall@5: {recall:.3f}\n")