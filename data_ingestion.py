import os, json, uuid
from pathlib import Path
from pypdf import PdfReader
import tiktoken
from chromadb import PersistentClient
from dotenv import load_dotenv
from utils.text import clean_text, chunk_text
from openai import OpenAI

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
load_dotenv()

# Load environment variables again (redundant but harmless)
load_dotenv()
print("✅ Loaded EMBED_MODEL =", os.getenv("EMBED_MODEL"))

# --- DIMENSION FIX ---
# ChromaDB complained it got 384, but expected 1536. 
# We will explicitly configure OpenAI to use 384, as it's a common, smaller size.
EMBEDDING_DIMENSION = 384
# ---------------------

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
VSTORE_DIR = Path("vectorstore")
COLLECTION_NAME = "python_course_docs"

OPENAI_MODEL_EMB = os.getenv("EMBED_MODEL", "text-embedding-3-small")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using OpenAI model (batched)."""
    # 🌟 FIX: Add the 'dimensions' parameter to match the expected 384 dimensions.
    resp = client.embeddings.create(
        model=OPENAI_MODEL_EMB, 
        input=texts,
        dimensions=EMBEDDING_DIMENSION # Explicitly setting dimension to 384
    )
    return [d.embedding for d in resp.data]

def read_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

# ---------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------
def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # 1️⃣ 	Read and clean PDFs
    payloads = []
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        raw = read_pdf(pdf)
        text = clean_text(raw)
        payloads.append({"filename": pdf.name, "text": text})

    # 2️⃣ 	Save processed text for transparency
    with open(PROC_DIR / "processed.jsonl", "w", encoding="utf-8") as f:
        for p in payloads:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 3️⃣ 	Chunk documents
    encoding = tiktoken.get_encoding("cl100k_base")
    docs = []
    for p in payloads:
        # Note: chunk_tokens=700 is a good size for 384-dim embeddings
        chunks = chunk_text(p["text"], encoding=encoding, chunk_tokens=700, overlap_tokens=80) 
        for ch in chunks:
            doc_id = str(uuid.uuid4())
            docs.append({
                "id": doc_id,
                "text": ch,
                "metadata": {
                    "source": p["filename"],
                    "doc_type": "assignment" if "assignment" in p["filename"].lower() else "concept",
                    "module": p["filename"].split("_")[0]
                }
            })

    # 4️⃣ 	Create persistent Chroma collection
    clientdb = PersistentClient(path=str(VSTORE_DIR))

    # Recreate collection fresh each time
    # This step is CRITICAL to ensure Chroma infers the new 384-dimension correctly
    existing = [c.name for c in clientdb.list_collections()]
    if COLLECTION_NAME in existing:
        clientdb.delete_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' deleted to reset dimensions.") # Added print for clarity
        
    col = clientdb.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # 5️⃣ 	Embed & upsert in batches
    BATCH = 64
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i + BATCH]
        
        # This calls the corrected embed function which now returns 384-dimensional vectors
        embeddings = embed([d["text"] for d in batch]) 
        
        col.upsert(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
            embeddings=embeddings
        )

    print(f"Ingested {len(docs)} chunks of {EMBEDDING_DIMENSION} dimensions into collection '{COLLECTION_NAME}' ✓")
    print(f"Persistent vector store saved to: {VSTORE_DIR.resolve()}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()