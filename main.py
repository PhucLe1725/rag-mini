"""
RAG Mini — Chatbot tư vấn sản phẩm Hồng Hà (CLI)
==================================================
Pipeline: PostgreSQL → ChromaDB → LLM (OpenAI-compatible)

Cách dùng:
  1. Điền đủ biến trong .env (OPENAI_*, PG_*)
  2. pip install -r requirements.txt
  3. python scripts/build_product_rag.py   # build index lần đầu
  4. python main.py                        # chạy chatbot CLI

Để chạy dưới dạng HTTP API (cho frontend tích hợp):
  python server.py
"""

import sys
from pathlib import Path

# Import toan bo core logic (model duoc load 1 lan trong rag_core)
from rag_core import (
    retrieve,
    generate_answer_stream,
    get_collection,
    TOP_K,
    BASE_DIR,
    API_KEY,
    BASE_URL,
    CHAT_MODEL,
)

if not API_KEY:
    print("Thieu OPENAI_API_KEY. Xem .env.example de biet cach cau hinh.")
    sys.exit(1)
if not BASE_URL:
    print("Thieu OPENAI_BASE_URL. Xem .env.example de biet cach cau hinh.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Chat loop (CLI)
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
                [sys.executable, str(BASE_DIR / "scripts" / "build_product_rag.py")],
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
        print("Tro ly: ", end="", flush=True)

        generate_answer_stream(query, retrieved)
        print("-" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("RAG Mini - Khoi dong chatbot tu van san pham Hong Ha")
    chat_loop()


if __name__ == "__main__":
    main()
