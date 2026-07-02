"""
rag_core.py — Logic dung chung giua CLI (main.py) va API server (server.py)

Khoi tao 1 lan khi import:
  - Load embedding model (sentence-transformers)
  - Ket noi ChromaDB
  - Khoi tao OpenAI-compatible chat client
"""

import io
import os
import sys

# Force UTF-8 tren Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL   = os.getenv("OPENAI_BASE_URL", "")
API_KEY    = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "glm-5.2")

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME       = "products"
CHROMA_DIR            = BASE_DIR / "chroma_db"
TOP_K                 = 3

# ---------------------------------------------------------------------------
# Khoi tao (chay 1 lan khi module duoc import)
# ---------------------------------------------------------------------------
print("Dang tai embedding model...", flush=True)
embed_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
print("Embedding model san sang.", flush=True)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
chat_client   = OpenAI(base_url=BASE_URL, api_key=API_KEY) if API_KEY else None


def get_collection():
    """Lay collection 'products'. Tra None neu chua build index."""
    try:
        return chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K, only_available: bool = True) -> list[dict]:
    """
    Tim k san pham lien quan nhat voi query.
    Tra ve list[{text, meta, score}].
    """
    col = get_collection()
    if col is None:
        return []

    qe = embed_model.encode(query, normalize_embeddings=True).tolist()

    kwargs: dict = dict(
        query_embeddings=[qe],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    if only_available:
        kwargs["where"] = {"availability": {"$eq": 1}}

    try:
        res = col.query(**kwargs)
    except Exception:
        kwargs.pop("where", None)
        res = col.query(**kwargs)

    docs  = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res["distances"] else []

    return [
        {"text": doc, "meta": meta, "score": 1 - dist}
        for doc, meta, dist in zip(docs, metas, dists)
    ]


# ---------------------------------------------------------------------------
# Build context (ngan gon, bo full_description)
# ---------------------------------------------------------------------------
def build_context(retrieved: list[dict]) -> str:
    """Tao context ngan gon: ten + gia + short_description (bo full_description)."""
    if not retrieved:
        return "Khong tim thay san pham lien quan."

    parts = []
    for i, item in enumerate(retrieved, 1):
        meta = item["meta"]
        lines, skip = [], False
        for line in item["text"].split("\n"):
            if line.startswith("Mo ta chi tiet:"):
                skip = True
            if not skip:
                lines.append(line)
        short = "\n".join(l for l in lines if l.strip())
        parts.append(
            f"[San pham {i} - SKU:{meta.get('sku','')} score:{item['score']*100:.0f}%]\n{short}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Ban la tro ly tu van ban hang cua cua hang van phong pham Hong Ha.\n"
    "Nhiem vu: Tu van san pham, giai dap thac mac, ho tro khach chon dung san pham.\n\n"
    "QUAN TRONG - Nguyen tac tra loi:\n"
    "- Tra loi TRUC TIEP bang tieng Viet, KHONG hien thi buoc suy nghi hay phan tich\n"
    "- Chi tu van dua tren thong tin san pham trong [NGU CANH]\n"
    "- Khong bia dat gia, tinh nang khong co trong ngu canh\n"
    "- Khi gioi thieu san pham: neu ten, gia, diem noi bat (2-3 dong moi san pham)\n"
    "- Neu khong tim thay san pham phu hop, noi thang va goi y lien he nhan vien\n"
    "- Giu giong than thien, ngan gon, thuc te"
)


def _make_prompt(query: str, retrieved: list[dict]) -> str:
    return (
        f"[NGU CANH - San pham Hong Ha]\n"
        f"{build_context(retrieved)}\n\n"
        f"[CAU HOI]\n{query}\n\n"
        f"[TRA LOI]"
    )


# ---------------------------------------------------------------------------
# Generate — non-streaming (cho API server)
# ---------------------------------------------------------------------------
def generate_answer(query: str, retrieved: list[dict], max_tokens: int = 600) -> str:
    """Goi LLM, tra ve chuoi ket qua (khong streaming). Dung cho REST API."""
    if not chat_client:
        return "Chua cau hinh OPENAI_API_KEY."

    prompt = _make_prompt(query, retrieved)
    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        stream=False,
    )
    choice = resp.choices[0]
    content = choice.message.content
    if not content:
        content = getattr(choice.message, "reasoning_content", None) or ""
    return content.strip() or "Xin loi, khong the tao phan hoi."


# ---------------------------------------------------------------------------
# Generate — streaming (cho CLI main.py)
# ---------------------------------------------------------------------------
def generate_answer_stream(query: str, retrieved: list[dict], max_tokens: int = 600) -> str:
    """Goi LLM voi streaming, in token ra stdout, tra ve full text. Dung cho CLI."""
    if not chat_client:
        print("Chua cau hinh OPENAI_API_KEY.")
        return ""

    prompt = _make_prompt(query, retrieved)
    stream = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        stream=True,
    )

    full_text = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        token = delta.content or getattr(delta, "reasoning_content", None) or ""
        if token:
            print(token, end="", flush=True)
            full_text.append(token)

    print()
    return "".join(full_text).strip() or "Xin loi, khong the tao phan hoi."
