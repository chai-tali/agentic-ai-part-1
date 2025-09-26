from __future__ import annotations

import math
import os
from typing import Iterable, List, Sequence, Tuple

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# =============================================================================
# ENVIRONMENT SETUP
# - Loads variables from .env and verifies the OpenAI key exists.
# - Keeping this separate makes the script reusable across exercises.
# =============================================================================
def load_env_or_raise() -> None:
    """Load .env and validate that OPENAI_API_KEY is present."""
    load_dotenv()
    # For this exercise we use Gemini native embeddings
    missing: list[str] = []
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    # GEMINI_MODEL_NAME is optional; default set below
    if missing:
        raise EnvironmentError(
            "Missing Gemini env vars: " + ", ".join(missing) + ".\n"
            "Ensure your .env defines GEMINI_API_KEY; optionally GEMINI_MODEL_NAME."
        )


def build_embedder() -> GoogleGenerativeAIEmbeddings:
    """Return a configured embeddings client targeting Gemini native embeddings.

    Why separate this?
    - Single-responsibility: construction/configuration isolated here
    - Reusability: later exercises can import and reuse this factory
    """
    model_name = os.getenv("GEMINI_MODEL_NAME", "text-embedding-001")
    # Google Generative AI expects fully-qualified model IDs like "models/text-embedding-001"
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=os.getenv("GEMINI_API_KEY"))


def embed_sentences(embedder: GoogleGenerativeAIEmbeddings, sentences: Sequence[str]) -> List[List[float]]:
    """Embed a sequence of sentences into numeric vectors.

    Implementation details
    - We call embed_documents (batch) for consistency and efficiency
    - Returns a list[List[float]] where each inner list is a high-dimensional vector
    """
    return embedder.embed_documents(list(sentences))


def vector_summary(vector: Sequence[float]) -> Tuple[int, float, float, float]:
    """Return (dimensions, min, max, mean) for a vector.

    These basic stats help you sanity-check embeddings:
    - Dimensions: model-dependent size (e.g., 1536)
    - Min/Max: value range
    - Mean: overall distribution center
    """
    dims = len(vector)
    vmin = min(vector) if dims else 0.0
    vmax = max(vector) if dims else 0.0
    vmean = (sum(vector) / dims) if dims else 0.0
    return dims, vmin, vmax, vmean


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Concept
    - Measures angle between vectors, ignoring magnitude
    - Range: [-1, 1]; higher means more semantically similar
    """
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def print_overview(sentences: Sequence[str], embeddings: Sequence[Sequence[float]]) -> None:
    """Print an overview of each sentence and its embedding stats.

    We also show a short "head" of the vector so you can see real numbers.
    """
    print("Generated embeddings:\n")
    for i, (s, v) in enumerate(zip(sentences, embeddings), start=1):
        dims, vmin, vmax, vmean = vector_summary(v)
        head = list(v[:6])
        print(f"{i}. {s}")
        print(f"   - dims: {dims}, min/max: {vmin:.6f}/{vmax:.6f}, mean: {vmean:.6f}")
        print(f"   - head: {head}\n")


def print_pairwise_similarities(sentences: Sequence[str], embeddings: Sequence[Sequence[float]]) -> None:
    """Print pairwise cosine similarity for all sentence pairs.

    Expectation
    - Related sentences (e.g., RAG and vector DB topics) should score higher
    - Unrelated pairs (e.g., cooking vs ML) should score lower
    """
    print("Pairwise cosine similarities (higher = more semantically similar):\n")
    n = len(sentences)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"({i+1}, {j+1}) {sim:.4f}")
    print()


def default_sentences() -> List[str]:
    """Provide a small, mixed set of sentences to show semantic grouping.

    We include 2 tech/RAG-focused and 2 cooking-focused sentences to
    demonstrate clustering by topic, plus an additional ML sentence.
    """
    return [
        "RAG retrieves relevant context to improve LLM answers.",
        "Vector databases enable efficient similarity search over embeddings.",
        "A sourdough starter needs regular feeding for best results.",
        "Neural networks learn complex patterns from data.",
        "Roasting vegetables brings out natural sweetness.",
    ]


def main() -> None:
    # 1) Ensure environment is configured
    load_env_or_raise()

    # 2) Choose inputs and prepare the embedding client
    sentences = default_sentences()
    embedder = build_embedder()

    # 3) Generate embeddings and inspect vector properties
    vectors = embed_sentences(embedder, sentences)
    print_overview(sentences, vectors)

    # 4) Compare sentence meanings via cosine similarity
    print_pairwise_similarities(sentences, vectors)


if __name__ == "__main__":
    main()

# =============================================================================
# LEARNING SUMMARY
# - Embeddings map text to numbers so we can compare meanings with math.
# - Cosine similarity quantifies semantic closeness between sentences.
# - Modular functions enable reuse in later RAG pipeline steps.
# =============================================================================


