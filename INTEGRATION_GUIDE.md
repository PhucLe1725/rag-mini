# Hướng dẫn tích hợp RAG-Mini vào QM-Bookstore

> Tài liệu này mô tả **cách tích hợp nhẹ nhàng nhất** — chỉ cần thêm 1 file mới và sửa tối thiểu vào `ChatContext.jsx`.

---

## Đánh giá tính khả thi

| Điểm đề xuất | Tình trạng | Ghi chú |
|---|---|---|
| REST API `/ask` và `/health` | ✅ **Đã triển khai** — `server.py` (FastAPI) | Mới tạo |
| Response format `{answer, sources, confidence}` | ✅ **Khớp** | `server.py` trả đúng format |
| `ragService.js` gọi `fetch` | ✅ **Dùng được ngay** | Giữ nguyên code trong RAG_INTEGRATION.md |
| CORS cho `localhost:5173` | ✅ **Đã cấu hình** | Trong `server.py` |
| Sửa `ChatContext.jsx` | ✅ **Dùng được ngay** | Code trong RAG_INTEGRATION.md hợp lệ |
| `history` (multi-turn) | ⚠️ Server nhận nhưng chưa xử lý | Dễ bỏ qua, không ảnh hưởng MVP |

**Kết luận: Khả thi 100%**, chỉ cần thêm `server.py` (đã tạo). Phần frontend dùng nguyên code trong `RAG_INTEGRATION.md`.

---

## Kiến trúc sau tích hợp

```
QM-Bookstore (React)          RAG-Mini (Python)
─────────────────────         ─────────────────────────────
Chatbot.jsx                   server.py  (FastAPI, port 8000)
  │                                │
  ▼                                ├─ GET  /health
ChatContext.jsx                    └─ POST /ask
  │                                        │
  ├─ import ragService.js                  ├─ rag_core.retrieve()    → ChromaDB
  └─ ragService.ask(query)  ──────────►   └─ rag_core.generate()    → LLM API
                              HTTP/JSON
```

---

## Cấu trúc project RAG-Mini sau refactor

```
rag-mini/
├── rag_core.py              ← [MỚI] Shared logic: embed model, retrieve, generate
├── server.py                ← [MỚI] FastAPI HTTP server
├── main.py                  ← [SỬA NHẸ] CLI chatbot, import từ rag_core
├── scripts/
│   └── build_product_rag.py ← Build ChromaDB index từ PostgreSQL
└── chroma_db/               ← Vector store (350 sản phẩm)
```

---

## Bước 1 — Chạy RAG-Mini API server

```powershell
# Terminal 1 — trong thư mục rag-mini/
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\activate
python server.py
```

Kết quả mong đợi:
```
Dang tai embedding model...
Embedding model san sang.
RAG-Mini API server: http://localhost:8000
Docs: http://localhost:8000/docs
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test nhanh:**
```powershell
# Health check
curl http://localhost:8000/health

# Test /ask
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"query": "Co but gel mau tim gia re khong?", "top_k": 3}'
```

Response mẫu:
```json
{
  "answer": "Chào bạn, cửa hàng có bút gel GP01...",
  "sources": [
    {"sku": "2752X", "name": "Bút gel GP01 - 2752", "price": 6800.0, "score": 0.446}
  ],
  "confidence": 0.446,
  "latency_ms": 1240
}
```

---

## Bước 2 — Phía frontend QM-Bookstore

### 2.1 Tạo `ragService.js` (copy nguyên từ RAG_INTEGRATION.md)

```
frontend/src/services/ragService.js   ← TẠO MỚI
```

Dùng **nguyên file** trong `RAG_INTEGRATION.md` Phần 2.1, **không cần sửa gì**.

---

### 2.2 Thêm biến môi trường

Trong `frontend/.env` (hoặc `frontend/.env.local`):

```env
VITE_RAG_API_URL=http://localhost:8000
```

---

### 2.3 Sửa `ChatContext.jsx` — tối thiểu 2 thay đổi

**Thay đổi 1:** Thêm import ở đầu file (1 dòng):
```js
import { ragService } from '../services/ragService'
```

**Thay đổi 2:** Thay toàn bộ hàm `sendChatbotMessage` (dùng nguyên code trong `RAG_INTEGRATION.md` Phần 2.2).

> **Lưu ý:** Xóa hoặc giữ `generateChatbotResponse` tùy bạn — nếu muốn dùng làm fallback thì giữ lại và gọi nó trong `catch` block.

---

### 2.4 (Tuỳ chọn) Hiển thị tên sản phẩm nguồn

Dùng nguyên code trong `RAG_INTEGRATION.md` Phần 2.4 cho `Chatbot.jsx`.

---

## Bước 3 — Chạy cùng lúc

```powershell
# Terminal 1: RAG-Mini API
cd rag-mini
python server.py

# Terminal 2: QM-Bookstore frontend
cd qm-bookstore/frontend
npm run dev
```

---

## Xử lý sự cố thường gặp

### ❌ CORS error trong browser console

```python
# Trong server.py, thêm origin của frontend vào CORS_ORIGINS
# Hoặc set biến môi trường:
$env:CORS_ORIGINS="http://localhost:5173,http://localhost:3000"
python server.py
```

### ❌ "RAG request timed out after 15 seconds"

Model GLM đôi khi chậm (~17s). Có 2 cách xử lý:

**Cách 1:** Tăng timeout trong `ragService.js`:
```js
const timeoutId = setTimeout(() => controller.abort(), 30000) // 30s
```

**Cách 2:** Dùng streaming SSE (nâng cao — xem Phần mở rộng bên dưới).

### ❌ "RAG index chua duoc build"

```powershell
cd rag-mini
python scripts/build_product_rag.py
```

### ❌ Collection không tìm thấy sau khi rebuild

```powershell
# Xoa chroma_db cu, build lai
Remove-Item -Recurse -Force chroma_db
python scripts/build_product_rag.py
```

---

## Deploy production

Khi deploy lên server thật (VPS, Railway, Render...):

1. **RAG-Mini** chạy trên `https://rag.yourdomain.com`
2. **Frontend** set `VITE_RAG_API_URL=https://rag.yourdomain.com`
3. **CORS** trong `server.py`: thêm domain thật vào `CORS_ORIGINS`

```python
# Trong server.py hoặc biến môi trường:
CORS_ORIGINS=https://bookstore.yourdomain.com
```

Hoặc dùng **nginx reverse proxy** để mount RAG-Mini dưới cùng domain:
```nginx
location /rag/ {
    proxy_pass http://localhost:8000/;
}
```
Khi đó `VITE_RAG_API_URL=/rag` (không cần CORS vì cùng origin).

---

## Mở rộng: Streaming SSE (cải thiện UX)

Nếu muốn chatbot hiển thị từng chữ như ChatGPT thay vì chờ toàn bộ response:

**Phía server** — thêm endpoint stream vào `server.py`:
```python
from fastapi.responses import StreamingResponse

@app.post("/ask/stream")
async def ask_stream(body: AskRequest):
    retrieved = retrieve(body.query, k=body.top_k or TOP_K)
    # ... dùng stream=True và yield từng chunk
```

**Phía frontend** — dùng `EventSource` hoặc fetch với `ReadableStream`.

> Đây là bước nâng cao, không cần thiết cho MVP.
