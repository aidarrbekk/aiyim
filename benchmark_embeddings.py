import time
import torch
import gc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODELS = [
    "intfloat/e5-large-v2",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-base-en-v1.5",
    "jinaai/jina-embeddings-v2-base-en"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIMILAR_PAIRS = [
    ("Artificial intelligence is transforming business.",
     "AI is changing business processes."),

    ("ИИ меняет бизнес.",
     "Искусственный интеллект трансформирует компании."),

    ("Жасанды интеллект бизнеске әсер етеді.",
     "Жасанды интеллект компанияларды өзгертуде.")
]

DISSIMILAR_PAIRS = [
    ("Artificial intelligence is transforming business.",
     "The cat sits on the mat."),

    ("ИИ меняет бизнес.",
     "Кошка сидит на коврике."),

    ("Жасанды интеллект бизнеске әсер етеді.",
     "Мысық кілемнің үстінде отыр.")
]

CROSS_LINGUAL_PAIRS = [
    ("Artificial intelligence is transforming business.",
     "ИИ меняет бизнес."),

    ("Artificial intelligence is transforming business.",
     "Жасанды интеллект бизнеске әсер етеді."),

    ("ИИ меняет бизнес.",
     "Жасанды интеллект бизнеске әсер етеді.")
]

SPEED_TEXTS = [
    "Artificial intelligence is transforming business.",
    "ИИ меняет бизнес-процессы в компаниях.",
    "Жасанды интеллект компанияларды өзгертуде.",
    "Machine learning models require data.",
    "Кошка сидит на коврике.",
    "Мысық кілемнің үстінде отыр."
]

def cosine(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]

def get_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3

def benchmark_model(model_name):
    print("\n" + "=" * 90)
    print(f"MODEL: {model_name}")
    print("=" * 90)

    torch.cuda.empty_cache()
    gc.collect()

    vram_before = get_vram_gb()
    load_start = time.time()

    model = SentenceTransformer(model_name, device=DEVICE)

    load_time = time.time() - load_start
    vram_after_load = get_vram_gb()

    print(f"Load time: {load_time:.2f} sec")
    print(f"VRAM used: {vram_after_load - vram_before:.2f} GB")

    torch.cuda.empty_cache()
    speed_start = time.time()

    embeddings = model.encode(
        SPEED_TEXTS,
        convert_to_tensor=True,
        batch_size=6,
        show_progress_bar=False
    )

    speed_time = time.time() - speed_start
    print(f"Inference time (batch {len(SPEED_TEXTS)}): {speed_time:.3f} sec")

    def run_pairs(pairs, label):
        print(f"\n{label}")
        scores = []
        for a, b in pairs:
            emb = model.encode([a, b], convert_to_tensor=True)
            score = cosine(
                emb[0].cpu().numpy(),
                emb[1].cpu().numpy()
            )
            scores.append(score)
            print(f"  Cosine = {score:.3f}")
        return sum(scores) / len(scores)

    avg_similar = run_pairs(SIMILAR_PAIRS, "SIMILAR PAIRS (EN/RU/KZ)")
    avg_dissimilar = run_pairs(DISSIMILAR_PAIRS, "DISSIMILAR PAIRS")
    avg_cross = run_pairs(CROSS_LINGUAL_PAIRS, "CROSS-LINGUAL PAIRS")

    print("\nSUMMARY")
    print(f"  Avg similar cosine      : {avg_similar:.3f}")
    print(f"  Avg dissimilar cosine   : {avg_dissimilar:.3f}")
    print(f"  Avg cross-lingual cosine: {avg_cross:.3f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for model_name in MODELS:
        benchmark_model(model_name)