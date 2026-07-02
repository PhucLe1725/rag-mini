"""
RAG Mini — Chatbot tư vấn sản phẩm Hồng Hà
============================================
Pipeline: PostgreSQL → ChromaDB → LLM (OpenAI-compatible)

Cách dùng:
  1. Điền đủ biến trong .env (OPENAI_*, PG_*)
  2. pip install -r requirements.txt
  3. python scripts/build_product_rag.py   # build index lần đầu (hoặc khi DB thay đổi)
  4. python main.py                        # chạy chatbot

Embedding: sentence-transformers/all-MiniLM-L6-v2 (local, ~80MB, miễn phí)
Chat:      OpenAI-compatible API (GLM, OpenAI, v.v.)
Vector DB: ChromaDB (local, persistent)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()

BASE_URL   = os.getenv("OPENAI_BASE_URL")
API_KEY    = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "glm-5.2")
TOP_K      = 3    # 3 san pham la du cho tu van; giam context, tang toc LLM

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME       = "products"   # phải khớp với build_product_rag.py

BASE_DIR   = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"

if not API_KEY:
    print("Thieu OPENAI_API_KEY. Xem .env.example de biet cach cau hinh.")
    sys.exit(1)
if not BASE_URL:
    print("Thieu OPENAI_BASE_URL. Xem .env.example de biet cach cau hinh.")
    sys.exit(1)

# Chat client
chat_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Embedding model (chay local)
print("Dang tai embedding model local...")
embed_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
print("Embedding model san sang.\n")

# ChromaDB
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Lay collection products. Bao loi ro rang neu chua build index."""
    try:
        col = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )
        return col
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K, only_available: bool = True) -> list[dict]:
    """
    Tim k san pham lien quan nhat den query.
    Tra ve list dict chua ca document text va metadata.
    """
    col = get_collection()
    if col is None:
        return []

    query_embedding = embed_model.encode(
        query, normalize_embeddings=True
    ).tolist()

    where = {"availability": {"$eq": 1}} if only_available else None

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    try:
        results = col.query(**kwargs)
    except Exception:
        kwargs.pop("where", None)
        results = col.query(**kwargs)

    docs      = results["documents"][0] if results["documents"] else []
    metas     = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    return [
        {"text": doc, "meta": meta, "score": 1 - dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Ban la tro ly tu van ban hang cua cua hang van phong pham Hong Ha.\n"
    "Nhiem vu: Tu van san pham, giai dap thac mac, ho tro khach chon dung san pham.\n\n"
    "QUAN TRONG - Nguyen tac tra loi:\n"
    "- Tra loi TRUC TIEP bang tieng Viet, KHONG hien thi buoc suy nghi hay phan tich trung gian\n"
    "- Chi tu van dua tren thong tin san pham trong [NGU CANH]\n"
    "- Khong bia dat gia, tinh nang khong co trong ngu canh\n"
    "- Khi gioi thieu san pham: neu ten, gia, diem noi bat (2-3 dong moi san pham)\n"
    "- Neu khong tim thay san pham phu hop, noi thang va goi y lien he nhan vien\n"
    "- Giu giong than thien, ngan gon, thuc te"
)


def _build_context(retrieved: list[dict]) -> str:
    """
    Tao context ngan gon cho LLM:
    - Chi dung short_description (khong phai full_description)
    - Giup giam ~60% so token, tang toc LLM ro ret
    """
    if not retrieved:
        return "Khong tim thay san pham lien quan."

    parts = []
    for i, item in enumerate(retrieved, 1):
        meta = item["meta"]
        # Lay dong dau (ten san pham) va cac dong ngan (bo full_description)
        lines = item["text"].split("\n")
        # Giu: ten, sku, danh muc, gia, tinh trang, mo ta ngan (bo 'Mo ta chi tiet')
        short_lines = []
        skip = False
        for line in lines:
            if line.startswith("Mo ta chi tiet:"):
                skip = True   # bo phan full_description
            if not skip:
                short_lines.append(line)
        short_text = "\n".join(l for l in short_lines if l.strip())
        parts.append(f"[San pham {i} - SKU:{meta.get('sku','')} - score:{item['score']*100:.0f}%]\n{short_text}")

    return "\n\n".join(parts)


def generate_answer(query: str, retrieved: list[dict]) -> str:
    """Goi LLM voi context ngan gon, streaming de hien thi nhanh."""
    context = _build_context(retrieved)

    prompt = (
        f"[NGU CANH - San pham Hong Ha]\n"
        f"{context}\n\n"
        f"[CAU HOI]\n{query}\n\n"
        f"[TRA LOI]"
    )

    stream = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,   # du cho tu van ngan gon; giam thoi gian reasoning
        stream=True,
    )

    full_text = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        token = delta.content
        if not token:
            token = getattr(delta, "reasoning_content", None) or ""
        if token:
            print(token, end="", flush=True)
            full_text.append(token)

    print()  # newline sau khi stream xong
    return "".join(full_text).strip() or "Xin loi, khong the tao phan hoi."


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
def chat_loop():
    col = get_collection()
    if col is None:
        print("\nChua co RAG index. Hay chay lenh sau truoc:")
        print("   python scripts/build_product_rag.py\n")
        sys.exit(1)

    n_products = col.count()
    print("=" * 60)
    print(f"  Hong Ha - Chatbot Tu Van San Pham")
    print(f"  Dang co {n_products:,} san pham trong kho du lieu")
    print(f"  Model chat: {CHAT_MODEL}")
    print("=" * 60)
    print("  Go 'quit' hoac 'exit' de thoat")
    print("  Go 'rebuild' de build lai RAG index tu DB\n")

    while True:
        try:
            query = input("Ban: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTam biet!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Tam biet!")
            break
        if query.lower() == "rebuild":
            print("Dang build lai RAG index...")
            import subprocess
            subprocess.run(
                [sys.executable,
                 str(BASE_DIR / "scripts" / "build_product_rag.py")],
                check=True,
            )
            print("Build xong! Tiep tuc chat...\n")
            continue

        print("Dang tim san pham lien quan...")
        retrieved = retrieve(query, k=TOP_K)

        if not retrieved:
            print("Khong tim thay san pham nao. Thu tu khoa khac hoac chay 'rebuild'.\n")
            continue

        names = [r["meta"].get("sku", "?") for r in retrieved]
        print(f"   Tim thay {len(retrieved)} san pham (SKU: {', '.join(names)})")
        print("Dang tra loi...\n")

        answer = generate_answer(query, retrieved)
        print()
        print("-" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("RAG Mini - Khoi dong chatbot tu van san pham Hong Ha")
    chat_loop()


if __name__ == "__main__":
    main()
