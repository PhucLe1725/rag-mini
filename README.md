# RAG Mini — Chatbot Tư Vấn Sản Phẩm Hồng Hà

Chatbot tư vấn sản phẩm văn phòng phẩm Hồng Hà sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)**.

## Kiến trúc

```
PostgreSQL (Aiven)  →  ChromaDB (local)  →  LLM (OpenAI-compatible)
  350 sản phẩm          Vector store          Streaming response
  62 categories         cosine similarity      Tiếng Việt
```

## Cài đặt

### 1. Tạo môi trường ảo

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 2. Cấu hình `.env`

```bash
cp .env.example .env
```

Điền đầy đủ các biến trong `.env`:

```env
# LLM API (OpenAI-compatible)
OPENAI_BASE_URL=https://api.xxx.net/v1
OPENAI_API_KEY=sk-...
CHAT_MODEL=glm-5.2

# PostgreSQL - Aiven
PG_HOST=your-host.aivencloud.com
PG_PORT=12345
PG_USER=avnadmin
PG_PASSWORD=your-password
PG_DBNAME=defaultdb
PG_SSLMODE=require
```

### 3. Build RAG index từ PostgreSQL

```bash
python scripts/build_product_rag.py
```

> Chạy lại lệnh này bất cứ khi nào dữ liệu sản phẩm trong DB thay đổi.

### 4. Chạy chatbot

```bash
# Windows (cần UTF-8 cho tiếng Việt)
$env:PYTHONIOENCODING='utf-8'; python main.py

# Linux/macOS
PYTHONIOENCODING=utf-8 python main.py
```

## Lệnh trong chatbot

| Lệnh | Chức năng |
|---|---|
| Câu hỏi bất kỳ | Tư vấn sản phẩm |
| `rebuild` | Cập nhật index từ PostgreSQL |
| `quit` / `exit` | Thoát |

## Hiệu năng (đo thực tế)

| Bước | Thời gian | Ghi chú |
|---|---|---|
| Load embedding model | ~11s | Chỉ 1 lần khi start |
| Encode query | ~0.01s | Rất nhanh sau lần đầu |
| Vector search | ~0.004s | ChromaDB cosine similarity |
| LLM response | ~5-17s | Phụ thuộc model/network |

**Bottleneck chính:** LLM API call (network latency + model inference).  
Dùng **streaming** để hiển thị từng token ngay khi có → UX tốt hơn.

## Cấu trúc project

```
rag-mini/
├── .env.example              # Template cấu hình
├── main.py                   # Chatbot chính
├── requirements.txt
├── scripts/
│   ├── build_product_rag.py  # Build vector index từ PostgreSQL
│   └── test_chatbot.py       # Test retrieval + LLM
└── chroma_db/                # Vector store (gitignored, tự tạo)
```

## Phụ thuộc chính

- `sentence-transformers` — Embedding model local (`all-MiniLM-L6-v2`)
- `chromadb` — Vector database local
- `psycopg2-binary` — Kết nối PostgreSQL
- `openai` — OpenAI-compatible API client
