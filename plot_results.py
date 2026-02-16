import matplotlib.pyplot as plt

models = [
    "e5-large-v2",
    "multilingual-e5-instruct",
    "bge-base-en",
    "jina-base-en"
]

similar = [0.906, 0.947, 0.845, 0.384]
cross = [0.818, 0.901, 0.587, 0.349]
time = [0.408, 0.133, 0.020, 0.017]
vram = [1.25, 1.04, 0.41, 0.43]

plt.figure()
plt.bar(models, cross)
plt.title("Cross-lingual Cosine Similarity (EN/RU/KZ)")
plt.ylabel("Cosine similarity")
plt.show()

plt.figure()
plt.bar(models, time)
plt.title("Inference Time (lower is better)")
plt.ylabel("Seconds per batch")
plt.show()