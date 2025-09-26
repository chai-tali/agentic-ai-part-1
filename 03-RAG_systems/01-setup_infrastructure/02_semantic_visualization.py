from __future__ import annotations

import math
import os
from typing import List, Sequence, Tuple

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =============================================================================
# ENV SETUP AND UTILITIES
# =============================================================================
def ensure_env() -> None:
    """Load .env and verify required environment variables.

    Required:
      - GEMINI_API_KEY
    Optional:
      - GEMINI_MODEL_NAME (default: text-embedding-001)
    """
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("GEMINI_API_KEY is not set. Provide it in .env or export it.")


def build_embedder() -> GoogleGenerativeAIEmbeddings:
    """Create a Gemini embeddings client.

    - Defaults to text-embedding-001
    - Prefixes model id with "models/" as required by Google APIs
    """
    model_name = os.getenv("GEMINI_MODEL_NAME", "text-embedding-001")
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=os.getenv("GEMINI_API_KEY"))


def embed_texts(embedder: GoogleGenerativeAIEmbeddings, texts: Sequence[str]) -> List[List[float]]:
    """Return embeddings for multiple texts."""
    return embedder.embed_documents(list(texts))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# =============================================================================
# DATASET
# =============================================================================
def build_dataset() -> Tuple[List[str], List[str]]:
    """Return (sentences, labels) for 12 short sentences across 3 themes.

    Themes: sports (blue), food (green), technology (red)
    """
    sports = [
        "I love playing football.",
        "Cricket is a popular game in India.",
        "Basketball is played on a court.",
        "Tennis requires speed and precision.",
    ]
    food = [
        "I enjoy eating pizza.",
        "Ice cream is my favorite dessert.",
        "Pasta is made with wheat.",
        "Fresh salads are very healthy.",
    ]
    tech = [
        "Artificial Intelligence is changing the world.",
        "I use my laptop for coding.",
        "Mobile phones connect people globally.",
        "Cloud computing scales applications efficiently.",
    ]

    sentences = sports + food + tech
    labels = ["sports"] * len(sports) + ["food"] * len(food) + ["tech"] * len(tech)
    return sentences, labels


# =============================================================================
# VISUALIZATION
# =============================================================================
def reduce_to_2d(vectors: List[List[float]]) -> List[List[float]]:
    """Project high-dimensional vectors to 2D using PCA."""
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(vectors).tolist()


def plot_dataset(points_2d: List[List[float]], sentences: Sequence[str], labels: Sequence[str]) -> None:
    """Plot the 2D points with color by label."""
    color_map = {"sports": "tab:blue", "food": "tab:green", "tech": "tab:red"}
    xs = [p[0] for p in points_2d]
    ys = [p[1] for p in points_2d]
    plt.figure(figsize=(9, 7))
    for x, y, s, lab in zip(xs, ys, sentences, labels):
        plt.scatter(x, y, c=color_map.get(lab, "gray"), s=80, edgecolors="black")
        plt.text(x + 0.02, y + 0.02, s, fontsize=8)
    plt.title("2D PCA of Sentence Embeddings by Topic")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_query_and_arrows(
    dataset_points: List[List[float]],
    query_point: List[float],
    neighbor_indices: Sequence[int],
    sentences: Sequence[str],
):
    """Overlay the query point and draw arrows to its nearest neighbors."""
    qx, qy = query_point
    plt.scatter(qx, qy, c="gold", s=140, marker="*", edgecolors="black", label="query")
    for idx in neighbor_indices:
        dx, dy = dataset_points[idx]
        plt.annotate(
            "",
            xy=(dx, dy),
            xytext=(qx, qy),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.8),
        )
        # Label neighbors slightly offset
        plt.text(dx + 0.02, dy + 0.02, f"NN: {sentences[idx]}", fontsize=8, color="black")
    plt.legend(loc="best")


# =============================================================================
# MAIN FLOW
# =============================================================================
def main() -> None:
    # 1) Environment and embedder
    ensure_env()
    embedder = build_embedder()

    # 2) Dataset of 12 sentences across 3 topics
    sentences, labels = build_dataset()

    # 3) Generate embeddings and verify dimension
    vectors = embed_texts(embedder, sentences)
    if not vectors:
        raise RuntimeError("Got empty embeddings list")
    print(f"Vector dimension: {len(vectors[0])}")

    # 4) Reduce to 2D and plot clusters
    dataset_2d = reduce_to_2d(vectors)
    plot_dataset(dataset_2d, sentences, labels)

    # 5) Nearest Neighbor search for a query
    query = "I like tennis."
    query_vec = embed_texts(embedder, [query])[0]
    # Find top-2 most similar sentences via cosine similarity
    sims = [cosine_similarity(query_vec, v) for v in vectors]
    top2 = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:2]
    print("Top-2 nearest neighbors:")
    for i in top2:
        print(f"- {sentences[i]} (similarity={sims[i]:.4f})")

    # 6) Project query to 2D with the same PCA transformation
    # Refit PCA on combined (dataset + query) to keep it simple for learning
    combined = vectors + [query_vec]
    combined_2d = reduce_to_2d(combined)
    query_2d = combined_2d[-1]
    dataset_only_2d = combined_2d[:-1]

    # Overlay query and neighbor arrows
    plot_query_and_arrows(dataset_only_2d, query_2d, top2, sentences)

    # Save plot to file (better for non-interactive environments)
    plt.savefig("semantic_clusters.png", dpi=150, bbox_inches="tight")
    print("Plot saved as 'semantic_clusters.png'")
    plt.close()  # Clean up memory


if __name__ == "__main__":
    main()


