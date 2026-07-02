"""
server.py — FastAPI HTTP server cho RAG-Mini
============================================
Expose cac endpoint de frontend (QM-Bookstore) goi vao.

Chay server:
  pip install fastapi uvicorn[standard]
  python server.py
  # hoac: uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /health        → kiem tra server con song
  POST /ask           → tu van san pham
"""

import sys
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import toan bo logic tu rag_core (model duoc load 1 lan o day)
from rag_core import retrieve, generate_answer, get_collection, TOP_K

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG-Mini API — Hong Ha",
    description="Chatbot tu van san pham van phong pham Hong Ha",
    version="1.0.0",
)

# CORS: cho phep frontend QM-Bookstore goi vao
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

SERVER_START = time.time()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K
    history: Optional[list] = []    # multi-turn (hien tai chua dung, de mo rong)


class ProductSource(BaseModel):
    sku: str
    name: str
    price: float
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[ProductSource]
    confidence: Optional[float]
    latency_ms: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Kiem tra server va RAG index co san sang khong."""
    col = get_collection()
    if col is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index chua duoc build. Chay: python scripts/build_product_rag.py"
        )
    return {
        "status": "ok",
        "collection": "products",
        "n_products": col.count(),
        "uptime_s": round(time.time() - SERVER_START, 1),
    }


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    """
    Tu van san pham dua tren cau hoi cua nguoi dung.

    Request:
        { "query": "but gel mau tim gia re", "top_k": 3 }

    Response:
        {
          "answer": "Chao ban, ...",
          "sources": [{"sku": "GP01", "name": "But gel GP01", "price": 6800, "score": 0.85}],
          "confidence": 0.85,
          "latency_ms": 1200
        }
    """
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query khong duoc de trong")

    t0 = time.time()

    # 1. Retrieve
    k = max(1, min(body.top_k or TOP_K, 10))   # gioi han 1-10
    retrieved = retrieve(body.query.strip(), k=k)

    if not retrieved:
        return AskResponse(
            answer="Xin loi, hien tai khong tim thay san pham phu hop. Ban co the lien he nhan vien de duoc ho tro.",
            sources=[],
            confidence=None,
            latency_ms=int((time.time() - t0) * 1000),
        )

    # 2. Generate (non-streaming)
    answer = generate_answer(body.query.strip(), retrieved)

    # 3. Build sources (thong tin san pham tra ve frontend)
    sources = []
    for item in retrieved:
        meta = item["meta"]
        # Lay ten san pham tu dong dau cua document text
        name_line = item["text"].split("\n")[0]   # "San pham: But gel GP01 - 2752"
        name = name_line.replace("San pham:", "").strip() if "San pham:" in name_line else name_line
        sources.append(ProductSource(
            sku=str(meta.get("sku", "")),
            name=name,
            price=float(meta.get("price", 0)),
            score=round(float(item["score"]), 3),
        ))

    top_score = sources[0].score if sources else None

    return AskResponse(
        answer=answer,
        sources=sources,
        confidence=top_score,
        latency_ms=int((time.time() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Dev run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"RAG-Mini API server: http://localhost:{port}")
    print(f"Docs: http://localhost:{port}/docs")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
