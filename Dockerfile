# ─────────────────────────────────────────────────────────────
# Stage 1: Builder — cài dependencies vào /install
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Cài các build-deps cần thiết (psycopg2-binary, sentence-transformers...)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install/packages --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime — image cuối nhỏ gọn
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Biến môi trường
ENV PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # HuggingFace cache — mount volume vào đây để model không tải lại mỗi lần restart
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # ChromaDB lưu vector store vào đây — mount volume vào đây
    CHROMA_DB_PATH=/app/chroma_db

WORKDIR /app

# Copy packages đã cài từ builder
COPY --from=builder /install/packages /usr/local

# Copy toàn bộ source code (xem .dockerignore để biết file nào bị loại bỏ)
COPY . .

# Tạo thư mục cache và chroma_db (sẽ bị ghi đè bởi volume mount)
RUN mkdir -p /app/.cache/huggingface /app/chroma_db

# Expose cổng FastAPI
EXPOSE 8000

# Health check — gọi endpoint /health mỗi 30s
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Khởi động server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
