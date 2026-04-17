from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


BASE_DIR        = Path(__file__).resolve().parent
BOOKS_CSV       = BASE_DIR / "books_with_emotions.csv"
TAGGED_TXT      = BASE_DIR / "tagged_description.txt"
CHROMA_DIR      = BASE_DIR / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


EMOTION_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

CATEGORY_OPTIONS = [
    "All",
    "Fiction",
    "Nonfiction",
    "Children's Fiction",
    "Children's Nonfiction",
]

EMOTION_OPTIONS = [
    "All", "Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", "Neutral",
]

PLACEHOLDER_COVER = "https://placehold.co/128x192?text=No+Cover"


def load_books(path: Path = BOOKS_CSV) -> pd.DataFrame:
    log.info("Loading books from %s ...", path)
    df = pd.read_csv(path)

    df["title"]             = df["title"].fillna("Unknown Title")
    df["authors"]           = df["authors"].fillna("Unknown Author")
    df["description"]       = df["description"].fillna("No description available.")
    df["thumbnail"]         = df["thumbnail"].fillna(PLACEHOLDER_COVER)
    df["simple_categories"] = df["simple_categories"].fillna("Unknown")
    df["average_rating"]    = pd.to_numeric(df.get("average_rating"), errors="coerce")
    df["num_pages"]         = pd.to_numeric(df.get("num_pages"),      errors="coerce")
    df["published_year"]    = pd.to_numeric(df.get("published_year"), errors="coerce")

    for col in EMOTION_COLS:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)

    df["isbn13"] = pd.to_numeric(df["isbn13"], errors="coerce")
    df = df.dropna(subset=["isbn13"])
    df["isbn13"] = df["isbn13"].astype(int)

    log.info("Loaded %d books.", len(df))
    return df


def build_or_load_db(
    txt_path:   Path = TAGGED_TXT,
    db_path:    Path = CHROMA_DIR,
    model_name: str  = EMBEDDING_MODEL,
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    if db_path.exists() and any(db_path.iterdir()):
        log.info("Loading existing Chroma DB from %s ...", db_path)
        return Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings,
        )

    log.info("Building Chroma DB — first-run only, please wait ...")
    raw  = TextLoader(str(txt_path), encoding="utf-8").load()
    docs = CharacterTextSplitter(
        chunk_size=6000, chunk_overlap=0, separator="\n"
    ).split_documents(raw)
    log.info("Split into %d chunks.", len(docs))

    db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=str(db_path),
    )
    log.info("Chroma DB saved to %s.", db_path)
    return db


def semantic_search(query: str, top_k: int = 50) -> list[int]:
    docs  = _DB.similarity_search(query, k=top_k)
    isbns: list[int] = []
    for doc in docs:
        try:
            isbn = int(doc.page_content.strip('"').split()[0])
            isbns.append(isbn)
        except (ValueError, IndexError):
            continue
    return isbns


def filter_and_rank(
    isbns:    list[int],
    category: str,
    emotion:  str,
    top_n:    int = 16,
) -> pd.DataFrame:
    result = _BOOKS_DF[_BOOKS_DF["isbn13"].isin(isbns)].copy()

    if result.empty:
        return result

    if category != "All":
        result = result[result["simple_categories"] == category]

    if result.empty:
        return result

    emotion_col = emotion.lower()
    if emotion_col in EMOTION_COLS:
        result = result.sort_values(emotion_col, ascending=False)

    return result.head(top_n)


def to_gallery(df: pd.DataFrame) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for row in df.itertuples(index=False):
        thumb  = row.thumbnail or PLACEHOLDER_COVER
        rating = row.average_rating
        stars  = f" · ★ {rating:.1f}" if (rating and not np.isnan(float(rating))) else ""
        caption = f"{row.title}\n{row.authors}{stars}"
        output.append((thumb, caption))
    return output


_last_results: pd.DataFrame = pd.DataFrame()


def recommend(query: str, category: str, emotion: str):
    global _last_results
    query = (query or "").strip()
    if not query:
        gr.Warning("Please enter a description of the book you're looking for.")
        return [], "", "", "", gr.update(visible=False)

    isbns    = semantic_search(query)
    filtered = filter_and_rank(isbns, category or "All", emotion or "All")

    if filtered.empty:
        gr.Warning("No books matched. Try different keywords or relax the filters.")
        _last_results = pd.DataFrame()
        return [], "", "", "", gr.update(visible=False)

    _last_results = filtered.reset_index(drop=True)
    gallery = to_gallery(filtered)
    return gallery, "", "", "", gr.update(visible=False)


def show_detail(evt: gr.SelectData):
    """Fires when the user clicks a book cover in the gallery."""
    idx = evt.index
    if _last_results.empty or idx >= len(_last_results):
        return "", "", "", gr.update(visible=False)

    row = _last_results.iloc[idx]

    title   = str(row.get("title",   "Unknown Title"))
    authors = str(row.get("authors", "Unknown Author"))
    rating  = row.get("average_rating", None)
    pages   = row.get("num_pages",      None)
    year    = row.get("published_year", None)
    cat     = str(row.get("simple_categories", ""))
    desc    = str(row.get("description", "No description available."))

    stars = f"★ {float(rating):.1f} / 5" if (rating and not np.isnan(float(rating))) else "No rating"
    pg    = f"{int(pages)} pages"        if (pages  and not np.isnan(float(pages)))  else ""
    yr    = f"Published {int(year)}"     if (year   and not np.isnan(float(year)))   else ""

    meta = "  ·  ".join(p for p in [stars, pg, yr, cat] if p)

    header = f"### {title}\n**{authors}**"
    return header, meta, desc, gr.update(visible=True)


CSS = """
#app-header { text-align: center; padding: 2rem 1rem 1rem; }
#app-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
#app-header p  { color: #6b7280; margin-top: 0.4rem; font-size: 1.05rem; }
#detail-box { border-radius: 12px; padding: 1.2rem 1.4rem; }
footer { display: none !important; }
"""

EXAMPLES = [
    ["A gripping thriller set in a futuristic city",             "Fiction",              "Fear"   ],
    ["Heartwarming story about family and forgiveness",          "Fiction",              "Joy"    ],
    ["True stories of scientific discovery",                     "Nonfiction",           "Joy"    ],
    ["A bedtime story about friendly forest animals",            "Children's Fiction",   "Joy"    ],
    ["History of World War II from multiple perspectives",       "Nonfiction",           "Sadness"],
    ["A mystery where the detective questions their own sanity", "Fiction",              "Fear"   ],
    ["Motivational book about overcoming adversity",             "Nonfiction",           "All"    ],
    ["Funny adventure for kids involving dragons",               "Children's Fiction",   "Surprise"],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Semantic Book Recommender") as demo:

        gr.HTML("""
            <div id="app-header">
                <h1>📚 Semantic Book Recommender</h1>
                <p>Describe the kind of book you're in the mood for — AI will find your next read.</p>
            </div>
        """)

        with gr.Row():
            query_box   = gr.Textbox(
                label="Describe your ideal book",
                placeholder='e.g. "A spy thriller set in Cold War Berlin with a female protagonist" ...',
                lines=2,
                scale=5,
            )
            category_dd = gr.Dropdown(
                choices=CATEGORY_OPTIONS,
                value="All",
                label="Category",
                scale=1,
            )
            emotion_dd  = gr.Dropdown(
                choices=EMOTION_OPTIONS,
                value="All",
                label="Emotional Tone",
                scale=1,
            )

        search_btn = gr.Button("🔍  Find Books", variant="primary", size="lg")

        gr.Markdown("---")

        gallery = gr.Gallery(
            label="Your Recommendations  —  click any cover to see details",
            show_label=True,
            columns=4,
            rows=4,
            height="auto",
            object_fit="contain",
        )

        # Detail panel — hidden until a book is clicked
        with gr.Group(visible=False, elem_id="detail-box") as detail_panel:
            gr.Markdown("---")
            detail_header = gr.Markdown("")
            detail_meta   = gr.Markdown("")
            detail_desc   = gr.Textbox(
                label="Description",
                lines=6,
                interactive=False,
            )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[query_box, category_dd, emotion_dd],
            label="💡 Try one of these",
            examples_per_page=8,
        )

        gr.Markdown(
            "<center><sub>Built with LangChain · HuggingFace · ChromaDB · Gradio"
            " &nbsp;|&nbsp; Embeddings: all-MiniLM-L6-v2</sub></center>"
        )

        search_outputs = [gallery, detail_header, detail_meta, detail_desc, detail_panel]

        search_btn.click(fn=recommend, inputs=[query_box, category_dd, emotion_dd], outputs=search_outputs)
        query_box.submit(fn=recommend, inputs=[query_box, category_dd, emotion_dd], outputs=search_outputs)

        gallery.select(
            fn=show_detail,
            outputs=[detail_header, detail_meta, detail_desc, detail_panel],
        )

    return demo


log.info("=== Semantic Book Recommender — starting up ===")
_BOOKS_DF: pd.DataFrame = load_books()
_DB:       Chroma       = build_or_load_db()
log.info("=== Ready. ===")

if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css=CSS,
    )