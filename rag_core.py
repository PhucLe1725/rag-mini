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
import time

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

LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME       = "products"
CHROMA_DIR            = BASE_DIR / "chroma_db"
TOP_K                 = 8
MIN_SCORE             = 0.30   # Nguong toi thieu — ket qua duoi nguong nay bi loai bo

# ---------------------------------------------------------------------------
# Query normalization — map tu khoa khong dau sang co dau (tang recall)
# ---------------------------------------------------------------------------
_QUERY_EXPAND = {
    # --- but chi ---
    "but chi":      "bút chì",
    "but chi go":   "bút chì gỗ",
    "but chi kim":  "bút chì kim",
    "but chi mau":  "bút chì màu",
    "but chi 2b":   "bút chì 2B",
    "but chi hb":   "bút chì HB",
    "but chi 4b":   "bút chì 4B",
    "but chi co tay": "bút chì có tẩy",
    "but chi hong ha": "bút chì Hồng Hà",
    # --- but khac ---
    "but bi":       "bút bi",
    "but gel":      "bút gel",
    "but long":     "bút lông",
    "but da quang": "bút dạ quang",
    "but ky":       "bút ký",
    "but xoa":      "bút xóa",
    # --- van phong pham ---
    "so ghi chep":  "sổ ghi chép",
    "so tay":       "sổ tay",
    "so bia da":    "sổ bìa da",
    "tap viet":     "tập viết",
    "tay chi":      "tẩy chì",
    "got chi":      "gọt chì",
    "compa chi":    "compa chì",
    "thuoc ke":     "thước kẻ",
    "balo hoc sinh": "balo học sinh",
    "hop but":      "hộp bút",
    "muc in":       "mực in",
    "giay in":      "giấy in",
}

# Map token don (khong dau → co dau), dung khi khong khop duoc voi _QUERY_EXPAND
_TOKEN_MAP = {
    "but":   "bút",
    "chi":   "chì",
    "go":    "gỗ",
    "mau":   "màu",
    "kim":   "kim",
    "tay":   "tẩy",
    "bi":    "bi",
    "gel":   "gel",
    "long":  "lông",
    "da":    "dạ",
    "so":    "sổ",
    "tap":   "tập",
    "viet":  "viết",
    "thuoc": "thước",
    "ke":    "kẻ",
    "got":   "gọt",
    "muc":   "mực",
    "giay":  "giấy",
}


def _normalize_query(query: str) -> str:
    """Neu query giong viet tat/khong dau, tra ve phien ban co dau tuong duong."""
    q_lower = query.lower().strip()
    # 1. Uu tien khop chinh xac voi dictionary
    if q_lower in _QUERY_EXPAND:
        return _QUERY_EXPAND[q_lower]
    # 2. Kiem tra co chua substring nao khong (dai nhat truoc)
    for key in sorted(_QUERY_EXPAND.keys(), key=len, reverse=True):
        if key in q_lower:
            return q_lower.replace(key, _QUERY_EXPAND[key])
    # 3. Fallback: map tung token mot
    tokens = q_lower.split()
    mapped = [_TOKEN_MAP.get(t, t) for t in tokens]
    result = " ".join(mapped)
    return result if result != q_lower else query

# ---------------------------------------------------------------------------
# Khoi tao (chay 1 lan khi module duoc import)
# ---------------------------------------------------------------------------
print("Dang tai embedding model...", flush=True)
embed_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
print("Embedding model san sang.", flush=True)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Timeout: 30s ket noi, 120s cho phan hoi (LLM co the mat vai chuc giay)
import httpx
_HTTP_TIMEOUT = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
chat_client = (
    OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=_HTTP_TIMEOUT)
    if API_KEY else None
)


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
    Su dung dual-query (query goc + query normalized) de tang recall cho tieng Viet khong dau.
    Fetch nhieu hon (k*4) roi loc theo MIN_SCORE, tra ve top-k ket qua tot nhat.
    Tra ve list[{text, meta, score}].
    """
    col = get_collection()
    if col is None:
        return []

    total = col.count()
    fetch_k = min(k * 6, total)   # fetch nhieu hon (6x) de tang recall truoc khi loc

    normalized = _normalize_query(query)
    if normalized == query:
        # Query da co dau hoac khong map duoc → dung chinh query do
        queries_to_run = [query]
    else:
        # Query khong dau da duoc normalize thanh cong → chi dung ban co dau
        # (tranh query khong dau tao ra embedding nhieu, keo score xuong thap)
        queries_to_run = [normalized]

    kwargs_base: dict = dict(
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    if only_available:
        kwargs_base["where"] = {"availability": {"$eq": 1}}

    seen_skus: set = set()
    merged: list[dict] = []

    for q in queries_to_run:
        qe = embed_model.encode(q, normalize_embeddings=True).tolist()
        kwargs = {**kwargs_base, "query_embeddings": [qe]}
        try:
            res = col.query(**kwargs)
        except Exception:
            kw2 = {k2: v for k2, v in kwargs.items() if k2 != "where"}
            res = col.query(**kw2)

        docs  = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []

        for doc, meta, dist in zip(docs, metas, dists):
            sku = meta.get("sku", doc[:30])
            if sku not in seen_skus:
                seen_skus.add(sku)
                merged.append({"text": doc, "meta": meta, "score": 1 - dist})

    # Sap xep theo score giam dan, loc theo MIN_SCORE, giu lai top-k
    merged.sort(key=lambda x: x["score"], reverse=True)
    filtered = [c for c in merged if c["score"] >= MIN_SCORE]
    return filtered[:k]


# ---------------------------------------------------------------------------
# Build context (ngan gon, bo full_description)
# ---------------------------------------------------------------------------
def build_context(retrieved: list[dict]) -> str:
    """Tao context ngan gon: ten + danh muc + gia + short_description (bo full_description)."""
    if not retrieved:
        return "Khong tim thay san pham phu hop voi yeu cau. Hay hoi khach hang them thong tin chi tiet."

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
        cat_path = meta.get("category_path", "")
        cat_info = f" | Danh muc: {cat_path}" if cat_path else ""
        parts.append(
            f"[San pham {i} - SKU:{meta.get('sku','')} | Do phu hop: {item['score']*100:.0f}%{cat_info}]\n{short}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Ban la tro ly tu van ban hang cua cua hang van phong pham Quang Minh.\n"
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
    """Goi LLM, tra ve chuoi ket qua (khong streaming). Dung cho REST API.
    Tu dong thu lai toi da 3 lan neu gap loi timeout/ket noi.
    """
    if not chat_client:
        return "Chua cau hinh OPENAI_API_KEY."

    prompt = _make_prompt(query, retrieved)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    MAX_RETRIES = 3
    RETRY_DELAYS = [2, 4, 8]  # giay, tang dan theo so lan thu

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
                stream=False,
            )
            choice = resp.choices[0]
            content = choice.message.content
            if not content:
                content = getattr(choice.message, "reasoning_content", None) or ""
            return content.strip() or "Xin loi, khong the tao phan hoi."

        except Exception as e:
            err_name = type(e).__name__
            is_timeout = "Timeout" in err_name or "timeout" in str(e).lower()
            is_conn    = "Connect" in err_name or "connect" in str(e).lower()
            if (is_timeout or is_conn) and attempt < MAX_RETRIES:
                wait = RETRY_DELAYS[attempt - 1]
                print(f"  [Warn] {err_name} — thu lai lan {attempt}/{MAX_RETRIES} sau {wait}s...", flush=True)
                time.sleep(wait)
                continue
            # Het luot thu hoac loi khac
            print(f"  [Error] Goi LLM that bai sau {attempt} lan thu: {err_name}", flush=True)
            return "Xin loi, hien tai khong the ket noi den he thong. Vui long thu lai sau."


# ---------------------------------------------------------------------------
# Generate — streaming (cho CLI main.py)
# ---------------------------------------------------------------------------
def generate_answer_stream(query: str, retrieved: list[dict], max_tokens: int = 600) -> str:
    """Goi LLM voi streaming, in token ra stdout, tra ve full text. Dung cho CLI.
    Tu dong thu lai toi da 3 lan neu gap loi timeout/ket noi.
    """
    if not chat_client:
        print("Chua cau hinh OPENAI_API_KEY.")
        return ""

    prompt = _make_prompt(query, retrieved)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    MAX_RETRIES = 3
    RETRY_DELAYS = [2, 4, 8]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stream = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
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

        except Exception as e:
            err_name = type(e).__name__
            is_timeout = "Timeout" in err_name or "timeout" in str(e).lower()
            is_conn    = "Connect" in err_name or "connect" in str(e).lower()
            if (is_timeout or is_conn) and attempt < MAX_RETRIES:
                wait = RETRY_DELAYS[attempt - 1]
                print(f"\n  [Warn] {err_name} — thu lai lan {attempt}/{MAX_RETRIES} sau {wait}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"\n  [Error] Goi LLM that bai sau {attempt} lan thu: {err_name}", flush=True)
            print("Xin loi, hien tai khong the ket noi den he thong. Vui long thu lai sau.")
            return ""
