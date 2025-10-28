import os, math
from collections import defaultdict
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient   # ✅ use PersistentClient

load_dotenv()

# ---------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------
VSTORE_DIR = "vectorstore"
COLLECTION_NAME = "python_course_docs"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------
# INIT CHROMA (Persistent)
# ---------------------------------------------------------------------
def init_chroma():
    """Initialize persistent Chroma client and load existing collection"""
    if not os.path.exists(VSTORE_DIR):
        raise FileNotFoundError(f"Vectorstore directory not found: {VSTORE_DIR}")

    clientdb = PersistentClient(path=VSTORE_DIR)
    collections = [c.name for c in clientdb.list_collections()]
    if COLLECTION_NAME not in collections:
        raise ValueError(f"Collection [{COLLECTION_NAME}] does not exist. Available: {collections}")

    col = clientdb.get_collection(COLLECTION_NAME)
    return col

# ---------------------------------------------------------------------
# CORE RETRIEVAL
# ---------------------------------------------------------------------
def dense_retrieval(col, query, k=8):
    """Semantic vector search (dense retrieval)"""
    results = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return []  # ✅ handle empty retrievals gracefully

    docs = [
        {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1.0 - results["distances"][0][i]  # cosine sim proxy
        }
        for i in range(len(results["documents"][0]))
    ]
    return docs


def bm25_retrieval(corpus_docs, query, k=8):
    """Lexical retrieval using BM25 (based on word overlap)"""
    if not corpus_docs:
        return []
    tokenized = [d["text"].split() for d in corpus_docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    ranked = sorted(
        [(i, s) for i, s in enumerate(scores)],
        key=lambda x: x[1],
        reverse=True
    )[:k]
    items = []
    for i, s in ranked:
        d = corpus_docs[i].copy()
        d["bm25"] = float(s)
        items.append(d)
    return items


def reciprocal_rank_fusion(runs, k=20):
    """Fuse multiple ranked lists using Reciprocal Rank Fusion"""
    scores = defaultdict(float)
    for run in runs:
        for rank, doc_id in enumerate(run, start=1):
            scores[doc_id] += 1.0 / (60.0 + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [doc_id for doc_id, _ in fused]


def fuse_dense_bm25(col, query, k=6):
    """Hybrid retrieval combining dense and BM25 signals"""
    dense = dense_retrieval(col, query, k=12)
    if not dense:
        return []  # ✅ early exit if no dense results

    pool = dense
    bm25 = bm25_retrieval(pool, query, k=12)

    def did(doc):
        return f"{doc['metadata']['source']}::{hash(doc['text'][:200])}"

    run1 = [did(d) for d in dense]
    run2 = [did(d) for d in bm25]

    fused_ids = reciprocal_rank_fusion([run1, run2], k=max(k, 6))
    id2doc = {did(d): d for d in (dense + bm25)}

    dedup, seen = [], set()
    for fid in fused_ids:
        if fid in id2doc and fid not in seen:
            dedup.append(id2doc[fid])
            seen.add(fid)

    return dedup[:k]

# ---------------------------------------------------------------------
# QUERY REWRITE + RERANK
# ---------------------------------------------------------------------
QUERY_REWRITE_SYSTEM = "You rewrite student questions into SHORT, focused search queries for Python course materials."

def rewrite_query(user_query: str) -> str:
    msg = [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {
            "role": "user",
            "content": f"Student question:\n{user_query}\n\nReturn ONE line of keywords (topic + concept).",
        },
    ]
    out = client.chat.completions.create(
        model=os.getenv("RAG_REWRITE_MODEL", "gpt-4o-mini"),
        messages=msg,
        temperature=0.2,
    )
    return out.choices[0].message.content.strip()


RERANK_SYSTEM = "You are selecting the most educationally useful chunks for answering the student's exact question."

def rerank_chunks(user_query, chunks, topn=3):
    """Ask LLM to select best snippets for relevance/quality."""
    if not chunks:
        return []
    joined = "\n\n".join(
        [f"[{i}] {c['metadata']['source']} :: {c['text'][:550]}" for i, c in enumerate(chunks)]
    )
    prompt = f"""Question: {user_query}

Below are candidate snippets. Pick up to {topn} indices that best answer the question and come from authoritative material. 
Return a comma-separated list of indices only (e.g., 0,2,3).

CANDIDATES:
{joined}
"""
    msg = [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    out = client.chat.completions.create(
        model=os.getenv("RAG_RERANK_MODEL", "gpt-4o-mini"),
        messages=msg,
        temperature=0,
    )
    txt = out.choices[0].message.content.strip()
    try:
        picks = [int(x) for x in txt.split(",") if x.strip().isdigit()]
        picks = [p for p in picks if 0 <= p < len(chunks)]
    except Exception:
        picks = list(range(min(topn, len(chunks))))
    return [chunks[i] for i in picks[:topn]]

# ---------------------------------------------------------------------
# PROMPT ASSEMBLY + FINAL ANSWER
# ---------------------------------------------------------------------
BASE_SYSTEM = """You are a Python Teaching Assistant for an undergraduate course.
Strict rules:
- Only use the provided context from course documents.
- If the question is off-topic (not Python fundamentals, NumPy, or Pandas), say so and redirect.
- If the user asks for an assignment solution, DO NOT provide a full solution. Explain concepts, give hints, but not final answers.
- Cite sources at the end as (source: filename, module).
- If no relevant context found, say: "I couldn't find this in our course materials." and suggest related topics.
"""

def build_context_block(chunks: list[dict]) -> str:
    """Combine retrieved chunks into a readable context block."""
    out = []
    for c in chunks:
        src = c["metadata"]["source"]
        mod = c["metadata"].get("module", "?")
        out.append(f"[{src} | module {mod}]\n{c['text']}")
    return "\n\n---\n\n".join(out)


def answer_with_context(user_query: str, context_chunks: list[dict]) -> str:
    """Generate final answer from retrieved context"""
    if not context_chunks:
        return (
            "I couldn’t find relevant material for your question in our course documents. "
            "Try rephrasing your question or ask about a different Python concept."
        )
    context = build_context_block(context_chunks)
    msg = [
        {"role": "system", "content": BASE_SYSTEM},
        {
            "role": "user",
            "content": f"Question:\n{user_query}\n\nContext:\n{context}\n\nAnswer in a friendly, step-by-step way for undergrads, and add 1–2 short code examples if useful.",
        },
    ]
    resp = client.chat.completions.create(
        model=os.getenv("RAG_ANSWER_MODEL", "gpt-4o-mini"),
        messages=msg,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------
# HIGH-LEVEL ENTRYPOINT
# ---------------------------------------------------------------------
def rag_answer(user_query: str, advanced: bool = True):
    """Convenience entry for debugging / testing"""
    col = init_chroma()
    q = rewrite_query(user_query) if advanced else user_query
    candidates = fuse_dense_bm25(col, q, k=8)
    return candidates
