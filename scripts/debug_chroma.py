"""Debug ChromaDB state and retrieval for pencil products."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DIR  = ROOT / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
try:
    col = client.get_collection("products")
    total = col.count()
    print(f"Total docs in ChromaDB: {total}")
except Exception as e:
    print(f"ERROR getting collection: {e}")
    sys.exit(1)

# Show 10 samples
results = col.get(limit=10, include=["documents", "metadatas"])
print("\n--- Sample docs in ChromaDB ---")
for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    first_line = doc.split("\n")[0]
    avail = meta.get("availability")
    sku = meta.get("sku")
    print(f"[{i+1}] avail={avail} | sku={sku} | {first_line}")

# Check availability distribution
print("\n--- Checking availability counts ---")
all_data = col.get(include=["metadatas"])
avail_1 = sum(1 for m in all_data["metadatas"] if m.get("availability") == 1)
avail_0 = sum(1 for m in all_data["metadatas"] if m.get("availability") == 0)
print(f"  availability=1 (con hang): {avail_1}")
print(f"  availability=0 (het hang): {avail_0}")

# Check specific SKUs from user's list
target_skus = ["3532", "3552", "3550", "3400", "3404", "3507", "3506", "3551", "3520"]
print(f"\n--- Searching for SKUs: {target_skus} ---")
for m in all_data["metadatas"]:
    if m.get("sku") in target_skus:
        print(f"  FOUND: sku={m.get('sku')} | avail={m.get('availability')} | brand={m.get('brand')}")

# Now test retrieval with embedding
print("\n--- Retrieval test ---")
embed_model = SentenceTransformer(EMBED_MODEL)

test_queries = [
    "bút chì gỗ 2B",
    "but chi go",
    "bút chì gỗ",
    "bút chì",
    "bút chì Hồng Hà",
    "bút chì 2B có tẩy",
]

for query in test_queries:
    qe = embed_model.encode(query, normalize_embeddings=True).tolist()
    # Without availability filter
    res = col.query(
        query_embeddings=[qe],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )
    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    print(f"\nQuery: '{query}'")
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1 - dist
        first_line = doc.split("\n")[0]
        print(f"  score={score:.3f} | sku={meta.get('sku')} | avail={meta.get('availability')} | {first_line[:60]}")
