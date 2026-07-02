"""Test script: kiem tra retrieval va LLM response."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # rag-mini/ (project root)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---- Setup ----
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
col = client.get_collection("products")

chat_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "glm-5.2")

TEST_QUERIES = [
    "bút gel màu tím giá rẻ",
    "sổ bìa da cho văn phòng",
    "bút chì gỗ 2B có tẩy cho học sinh",
]

for query in TEST_QUERIES:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print("="*60)

    qe = embed_model.encode(query, normalize_embeddings=True).tolist()
    res = col.query(
        query_embeddings=[qe],
        n_results=3,
        where={"availability": {"$eq": 1}},
        include=["documents", "metadatas", "distances"],
    )

    docs   = res["documents"][0]
    metas  = res["metadatas"][0]
    dists  = res["distances"][0]

    print(f"\nTop {len(docs)} san pham tim duoc:")
    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        score = 1 - dist
        name_line = doc.split("\n")[0]
        print(f"  [{i}] score={score:.3f} | {name_line} | gia={meta['price']:,.0f} VND")
        context_parts.append(f"[San pham {i} - do lien quan: {score*100:.0f}%]\n{doc}")

    context = "\n\n" + ("-"*40 + "\n\n").join(context_parts)

    prompt = (
        "[NGU CANH - Thong tin san pham tu kho du lieu Hong Ha]\n"
        f"{context}\n\n"
        f"[CAU HOI CUA KHACH]\n{query}\n\n"
        "[TRA LOI]"
    )
    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": (
                "Ban la tro ly tu van ban hang cua cua hang van phong pham Hong Ha. "
                "Tra loi TRUC TIEP bang tieng Viet, KHONG hien thi buoc suy nghi hay phan tich. "
                "Chi tu van dua tren thong tin san pham duoc cung cap. Ngan gon, than thien."
            )},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    choice = resp.choices[0]
    content = choice.message.content
    if not content:
        content = getattr(choice.message, 'reasoning_content', None) or ""
    answer = content.strip() if content else "(Khong co noi dung tra ve)"
    print(f"\nCHATBOT:\n{answer}\n")
