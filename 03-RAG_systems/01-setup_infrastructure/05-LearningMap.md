## 01_basic_embedding.py

### What I will build?
- A small Python script that generates embeddings for a list of sentences, prints vector statistics (dimensions, min/max/mean, head preview), and computes pairwise cosine similarities to visualize semantic closeness.

### What concepts gets applied in this exercise?
- Text embeddings (mapping text to high-dimensional vectors)
- Semantic similarity and cosine similarity
- Inspecting vector properties for quick sanity checks

### What skills I will acquire post completion of this exercise?
- Generate embeddings via `langchain_openai.OpenAIEmbeddings`
- Compute and interpret cosine similarity between vectors
- Structure small utilities for reusability in a RAG pipeline

### What tools, libraries, frameworks I will use for this exercise?
- Python 3.10/3.11
- `langchain-google-genai`, `langchain-core`
- `python-dotenv` for environment loading
- Gemini native embeddings (`GoogleGenerativeAIEmbeddings`)

### How to run this example?
```bash
 1) Ensure these are present in your .env at repo root (or export them):
    GEMINI_API_KEY=...
    GEMINI_MODEL_NAME=text-embedding-001   # optional; defaults to this if omitted

# 2) Run the script
uv run python 03-RAG_systems/01-setup_infrastructure/01_basic_embedding.py
```

Expected behavior:
- Prints each sentence with its embedding stats and a short head of the vector
- Prints pairwise cosine similarity scores ([-1, 1], higher = more similar)


## 02_semantic_visualization.py

### What I will build?
- A Python script that visualizes semantic clusters of 12 sentences across 3 themes (sports, food, technology) in 2D space using PCA, and demonstrates nearest neighbor search with a query sentence.

### What concepts gets applied in this exercise?
- Text embeddings and semantic similarity
- Dimensionality reduction with PCA (Principal Component Analysis)
- Nearest neighbor search using cosine similarity
- Data visualization with matplotlib

### What skills I will acquire post completion of this exercise?
- Generate embeddings for multiple texts using Gemini native embeddings
- Reduce high-dimensional vectors to 2D for visualization
- Find and visualize nearest neighbors for semantic search
- Create scatter plots with color-coded clusters and arrows

### What tools, libraries, frameworks I will use for this exercise?
- Python 3.10/3.11
- `langchain-google-genai`, `langchain-core`
- `matplotlib`, `scikit-learn`
- `python-dotenv` for environment loading

### How to run this example?
```bash
# 1) Ensure these are present in your .env at repo root (or export them):
   GEMINI_API_KEY=...
   GEMINI_MODEL_NAME=text-embedding-001   # optional; defaults to this if omitted

# 2) Run the script
uv run python 03-RAG_systems/01-setup_infrastructure/02_semantic_visualization.py
```

Expected behavior:
- Prints vector dimension (e.g., 3072 for Gemini embeddings)
- Prints top-2 nearest neighbors for query "I like tennis." with similarity scores
- Saves a plot as "semantic_clusters.png" showing:
  - Color-coded clusters by topic (sports=blue, food=green, tech=red)
  - Query point as a gold star
  - Arrows pointing from query to its 2 nearest neighbors


