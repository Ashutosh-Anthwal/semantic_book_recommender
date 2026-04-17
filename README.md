# 📚 Semantic Book Recommender

An AI-powered book recommendation system that understands the **meaning** behind your query — not just keywords. Describe a mood, a theme, or a plot, and it finds your next read using semantic search, emotion analysis, and category filtering.

---

## ✨ Features

- **Semantic Search** — Natural language queries like *"a spy thriller set in Cold War Berlin"*
- **Category Filtering** — Fiction, Nonfiction, Children's Fiction, Children's Nonfiction
- **Emotional Tone Sorting** — Sort results by Joy, Sadness, Fear, Anger, Disgust, Surprise, or Neutral
- **Interactive Dashboard** — Book covers, ratings, page count, and description in a clean Gradio UI
- **Persistent Vector DB** — ChromaDB index built once, reloaded on every subsequent run

---

## 🗂️ Project Structure

```
semantic_book_recommender/
│
├── data/
│   └── processed/
│       ├── books_cleaned.csv               # Cleaned book dataset
│       ├── books_with_categories.csv       # After category mapping
│       ├── books_with_emotions.csv         # Final dataset with emotion scores
│       └── tagged_description.txt          # ISBN-tagged descriptions for vector search
│
├── notebooks/
│   ├── data-exploration.ipynb              # Data loading, cleaning, EDA
│   ├── vector-search.ipynb                 # ChromaDB vector store + semantic search
│   ├── text-classification.ipynb           # Fiction/Nonfiction zero-shot classification
│   └── sentiment-analysis.ipynb            # Per-sentence emotion scoring
│
├── src/
│   └── gradio_dashboard.py                 # Gradio web app
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

**1. Data Cleaning** — `data-exploration.ipynb`

Downloads the [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset via `kagglehub`. Removes books with missing descriptions, ratings, page counts, or published year. Analyses word count distribution across descriptions to filter out low-quality entries.

**2. Vector Search** — `vector-search.ipynb`

Book descriptions are tagged with their ISBN and embedded using `all-MiniLM-L6-v2` via HuggingFace. The embeddings are stored in a ChromaDB vector database. At query time, your input is embedded and the top 50 most semantically similar books are retrieved using cosine similarity.

**3. Text Classification** — `text-classification.ipynb`

Uses `facebook/bart-large-mnli` via a zero-shot classification pipeline to map raw Kaggle categories (e.g. "Juvenile Fiction", "Biography & Autobiography") into four simplified labels: Fiction, Nonfiction, Children's Fiction, and Children's Nonfiction.

**4. Sentiment Analysis** — `sentiment-analysis.ipynb`

Uses `j-hartmann/emotion-english-distilroberta-base` and spaCy sentence segmentation to score each book's description sentence-by-sentence across 7 emotions: anger, disgust, fear, joy, sadness, surprise, neutral. The maximum score per emotion is stored as the book's emotional fingerprint.

**5. Gradio Dashboard** — `gradio_dashboard.py`

Loads the final `books_with_emotions.csv` and the ChromaDB index, then runs semantic search + filtering + ranking on every query. Results are shown as a clickable gallery — clicking a cover reveals full metadata and description.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Ashutosh-Anthwal/semantic_book_recommender.git
cd semantic_book_recommender
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 5. Run the notebooks in order
Open Jupyter and run these in sequence — each one produces output files that the next depends on:

```
1. data-exploration.ipynb       → books_cleaned.csv
2. vector-search.ipynb          → tagged_description.txt + chroma_db/
3. text-classification.ipynb    → books_with_categories.csv
4. sentiment-analysis.ipynb     → books_with_emotions.csv
```

### 6. Launch the dashboard
```bash
python src/gradio_dashboard.py
```

Open your browser at `http://127.0.0.1:7860`

> On first launch, ChromaDB will build and persist the vector index automatically. Subsequent launches load it from disk instantly.

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Dataset | `kagglehub` — Dylan Castillo's 7k Books dataset |
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Vector Database | `chromadb` via `langchain-chroma` |
| Text Classification | `transformers` · `facebook/bart-large-mnli` |
| Emotion Analysis | `transformers` · `j-hartmann/emotion-english-distilroberta-base` |
| Sentence Splitting | `spacy` · `en_core_web_sm` |
| LLM Orchestration | `langchain`, `langchain-community`, `langchain-huggingface` |
| Data Processing | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Web UI | `gradio` |

---

## 🔮 Future Improvements

- Deploy on Hugging Face Spaces
- Add author and publisher filters
- Support user bookmarking of favourite recommendations
- Swap `all-MiniLM-L6-v2` for a larger model for better retrieval accuracy

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
