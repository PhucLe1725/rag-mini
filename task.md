# Task: Triển khai 10 cải tiến RAG-Mini

- [x] Đọc và phân tích toàn bộ code hiện tại
- [ ] Cập nhật requirements.txt (thêm rank-bm25, unidecode)
- [ ] Rewrite rag_core.py với #1 #2 #3 #4 #5 #6 #7 #9 #10
  - [ ] #1 — Query normalize dùng unicodedata
  - [ ] #2 — Conversation history
  - [ ] #3 — Adaptive MIN_SCORE (0.45)
  - [ ] #4 — BM25 hybrid search
  - [ ] #5 — Cross-encoder reranker (lazy load, opt-in)
  - [ ] #6 — HyDE query expansion (opt-in)
  - [ ] #7 — Category-aware score boost
  - [ ] #9 — Structured JSON logging
  - [ ] #10 — ChromaDB fallback warning
- [ ] Cập nhật server.py (#2 history, #9 request log)
- [ ] Cập nhật build_product_rag.py (#8 price-aware embed)
- [ ] Cài đặt dependencies mới
- [ ] Chạy smoke test import
