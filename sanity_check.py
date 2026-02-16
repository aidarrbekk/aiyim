from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

def cos_sim(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]

tests = {
    "SIMILAR": [
        ("Artificial intelligence improves business efficiency.",
         "AI helps companies optimize business processes."),
        ("ИИ повышает эффективность бизнеса.",
         "Жасанды интеллект бизнес тиімділігін арттырады.")
    ],
    "DISSIMILAR": [
        ("Artificial intelligence improves business efficiency.",
         "Quantum entanglement explains particle correlations."),
        ("How to pay taxes online?",
         "The recipe for baking chocolate cake.")
    ],
    "MULTILINGUAL": [
        ("Artificial intelligence improves business efficiency.",
         "ИИ повышает эффективность бизнеса."),
        ("ИИ повышает эффективность бизнеса.",
         "Жасанды интеллект бизнес тиімділігін арттырады.")
    ]
}

results = {}

for category, pairs in tests.items():
    scores = []
    print(f"\n{category}")
    for a, b in pairs:
        emb = model.encode([a, b], convert_to_numpy=True)
        score = cos_sim(emb[0], emb[1])
        scores.append(score)
        print(f"Cosine: {score:.4f}")
    results[category] = np.mean(scores)

print("\nSUMMARY")
for k, v in results.items():
    print(f"{k}: {v:.4f}")