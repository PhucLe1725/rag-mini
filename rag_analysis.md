# RAG-Mini — Phân tích luồng & Đề xuất cải tiến

## Cấu trúc thư mục

```
rag-mini/
├── .env / .env.example        # Cấu hình API key, DB, model
├── main.py                    # Entrypoint CLI chatbot
├── server.py                  # FastAPI HTTP server (cho frontend)
├── rag_core.py                # Logic dùng chung: embed, retrieve, generate
├── requirements.txt
├── data/
│   ├── categories_*.sql       # Dump SQL categories (backup)
│   └── products_*.sql         # Dump SQL products (backup)
└── scripts/
    ├── build_product_rag.py   # Build ChromaDB index từ PostgreSQL
    ├── check_sql.py           # Kiểm tra SQL dump
    ├── check_stock.py         # Kiểm tra tồn kho
    ├── debug_chroma.py        # Debug ChromaDB
    ├── test_chatbot.py        # Test chatbot
    └── verify_stock.py        # Xác nhận stock
```

---

## Luồng hoạt động hiện tại

### Giai đoạn 1 — Xây dựng index (offline, chạy 1 lần)

```
PostgreSQL (Aiven)
  └─► fetch_category_map()       → cây danh mục (id → path đầy đủ)
  └─► fetch_products()           → list sản phẩm (name, sku, price, desc...)
        │
        ▼
build_embed_text()               → text ngắn gọn cho embedding:
  • Tên sản phẩm
  • Category path đầy đủ
  • Thương hiệu
  • Short description
  • Keyword aliases (2B, HB, gỗ, kim...)

build_document()                 → full text lưu vào Chroma (cho LLM đọc):
  • Tên, SKU, danh mục, thương hiệu, giá, tình trạng
  • Mô tả ngắn + Mô tả chi tiết
        │
        ▼
SentenceTransformer.encode()     → vector embedding (multilingual-MiniLM-L12)
        │
        ▼
ChromaDB (cosine similarity)     → lưu collection "products"
  • embeddings: từ embed_text ngắn
  • documents: full document (cho LLM)
  • metadatas: sku, price, availability, category_path...
```

### Giai đoạn 2 — Trả lời câu hỏi (online, mỗi request)

```
User query
  └─► _normalize_query()         → map không dấu → có dấu
         (e.g. "but chi" → "bút chì")
        │
        ▼
embed_model.encode(query)        → query vector
        │
        ▼
ChromaDB.query()                 → lấy top fetch_k (k×6) kết quả
  • Filter: availability = 1 (chỉ hàng còn)
  • De-dup theo SKU
  • Score = 1 - cosine_distance
  • Lọc score >= MIN_SCORE (0.30)
  • Trả về top-k (mặc định 8)
        │
        ▼
build_context()                  → cắt bỏ full_description,
                                   giữ phần ngắn gọn cho LLM
        │
        ▼
_make_prompt()                   → ghép [NGU CANH] + [CAU HOI]
        │
        ▼
OpenAI-compatible LLM (GLM)      → generate answer
  • temperature=0.3, max_tokens=600
  • Retry tối đa 3 lần (timeout/connection error)
        │
        ▼
Response → CLI (streaming) hoặc REST API (non-streaming)
```

---

## Điểm mạnh hiện tại

| Điểm mạnh | Chi tiết |
|---|---|
| Tách biệt embed text vs display text | `embed_text` ngắn gọn tránh semantic dilution, `document` đầy đủ cho LLM |
| Query normalization | Xử lý tiếng Việt không dấu → có dấu |
| Keyword aliases | Boost recall cho các biến thể tên sản phẩm |
| Dual-path: CLI + API | Cùng logic `rag_core.py`, hai interface khác nhau |
| Retry mechanism | 3 lần retry khi LLM timeout/connection error |
| MIN_SCORE threshold | Lọc kết quả không liên quan |

---

## Hạn chế & Đề xuất cải tiến

### 🔴 Vấn đề nghiêm trọng

#### 1. Query normalization dùng hardcode dictionary — dễ miss
**Hiện tại**: `_QUERY_EXPAND` và `_TOKEN_MAP` chỉ có ~30 từ hardcode.

**Vấn đề**: Người dùng gõ "but da quang mau vang" → chỉ map được "but da quang" nhưng "mau vang" bị bỏ. Mọi từ tiếng Việt mới cần thêm tay.

**Cải tiến**:
```python
# Dùng thư viện bỏ dấu đúng cách
from unidecode import unidecode
import unicodedata

def remove_diacritics(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

# So sánh similarity giữa query không dấu và tên sản phẩm không dấu
# → tự động xử lý mà không cần hardcode
```

#### 2. Không có conversation history — mất ngữ cảnh hội thoại
**Hiện tại**: `history: Optional[list] = []` trong schema nhưng **không được dùng** trong generate.

**Vấn đề**: 
- User hỏi: "Cho tôi xem bút chì HB"
- User hỏi tiếp: "Cái nào có tẩy?" → chatbot không biết đang nói về bút chì

**Cải tiến**:
```python
# Thêm history vào messages
def generate_answer(query, retrieved, history=None, max_tokens=600):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Inject lịch sử hội thoại (tối đa 3 lượt gần nhất)
    for turn in (history or [])[-3:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    
    messages.append({"role": "user", "content": _make_prompt(query, retrieved)})
```

#### 3. MIN_SCORE = 0.30 quá thấp, có thể trả kết quả không liên quan
**Vấn đề**: Với câu hỏi chung như "hàng còn không?" → có thể retrieve sản phẩm ngẫu nhiên với score 0.31.

**Cải tiến**: Thêm adaptive threshold hoặc dùng reranker.

---

### 🟡 Cải tiến độ chính xác (accuracy)

#### 4. Thêm BM25 hybrid search (keyword + semantic)
**Hiện tại**: Chỉ dùng semantic search (vector similarity).

**Vấn đề**: Người dùng tìm đúng tên SKU "CH-HH02" → semantic search kém hiệu quả với exact keyword.

**Cải tiến**:
```python
# Thêm BM25 keyword search song song với vector search
from rank_bm25 import BM25Okapi

# Kết hợp score: final_score = alpha * semantic_score + (1-alpha) * bm25_score
# Điều chỉnh alpha theo độ dài query (query ngắn → tăng BM25 weight)
```

#### 5. Cross-encoder reranker sau khi retrieve
**Hiện tại**: Kết quả từ vector search được sắp xếp thuần theo cosine score.

**Cải tiến**: Thêm bước rerank với cross-encoder model.
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates, top_n=5):
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]
```

#### 6. Query expansion với LLM (HyDE — Hypothetical Document Embedding)
**Cải tiến**: Dùng LLM sinh ra một "hypothetical answer" rồi embed câu trả lời đó thay vì embed query.
```python
def hyde_query(query: str) -> str:
    """Tạo hypothetical document embedding để cải thiện recall."""
    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": f"Mô tả ngắn một sản phẩm văn phòng phẩm phù hợp với yêu cầu: {query}"}],
        max_tokens=100,
    )
    return resp.choices[0].message.content
```

#### 7. Category-aware filtering
**Hiện tại**: Filter chỉ theo `availability`.

**Cải tiến**: Detect intent category từ query rồi filter thêm theo `category_path`.
```python
def detect_category(query: str) -> Optional[str]:
    """Phát hiện danh mục sản phẩm từ query."""
    CATEGORY_KEYWORDS = {
        "bút": ["bút bi", "bút gel", "bút chì"],
        "sổ": ["sổ tay", "sổ ghi chép"],
        # ...
    }
    # Nếu detect được → thêm where filter theo category_path
```

---

### 🟢 Cải tiến kỹ thuật / vận hành

#### 8. Embed text thiếu giá và tình trạng
**Hiện tại**: `build_embed_text()` không chứa thông tin giá.

**Vấn đề**: Query "bút gel dưới 10.000 đồng" → không retrieve được chính xác.

**Cải tiến**:
```python
# Thêm price range vào embed text
if p["price"] < 10000:
    parts.append("giá rẻ dưới 10 nghìn")
elif p["price"] < 30000:
    parts.append("giá trung bình")
else:
    parts.append("giá cao cấp")
```

#### 9. Logging & monitoring còn thiếu
**Hiện tại**: Chỉ có `print()` statements, không có structured logging.

**Cải tiến**:
```python
import logging
import json

logger = logging.getLogger("rag_mini")

# Log mỗi request: query, retrieved SKUs, score, latency
logger.info(json.dumps({
    "query": query,
    "retrieved": [r["meta"]["sku"] for r in retrieved],
    "top_score": retrieved[0]["score"] if retrieved else 0,
    "latency_ms": latency,
}))
```

#### 10. ChromaDB query fallback không log warning
**Hiện tại** (rag_core.py L186):
```python
except Exception:
    kw2 = {k2: v for k2, v in kwargs.items() if k2 != "where"}
    res = col.query(**kw2)
```
Khi filter `availability` fail, silently bỏ filter → có thể trả hàng hết hàng.

**Cải tiến**: Thêm warning log khi fallback xảy ra.

---

## Tóm tắt ưu tiên cải tiến

| Mức độ | Cải tiến | Effort | Impact |
|---|---|---|---|
| 🔴 Cao | Conversation history (multi-turn) | Thấp | Cao |
| 🔴 Cao | Sửa query normalize dùng `unicodedata` | Thấp | Cao |
| 🟡 Trung bình | BM25 hybrid search | Trung bình | Cao |
| 🟡 Trung bình | Cross-encoder reranker | Trung bình | Cao |
| 🟡 Trung bình | Price-aware embedding | Thấp | Trung bình |
| 🟡 Trung bình | Category-aware filtering | Trung bình | Trung bình |
| 🟢 Thấp | Structured logging | Thấp | Thấp |
| 🟢 Thấp | HyDE query expansion | Cao | Trung bình |
