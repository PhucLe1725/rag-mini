"""Check stock_quantity values from ChromaDB and SQL dump."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import chromadb

client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
col = client.get_collection("products")

# Get all metadata
all_data = col.get(include=["metadatas", "documents"])
stocks = [(m.get("sku"), m.get("stock"), m.get("availability")) for m in all_data["metadatas"]]

# Thong ke
unique_stocks = sorted(set(s[1] for s in stocks))
print(f"Cac gia tri stock_quantity khac nhau: {unique_stocks[:30]}")
print(f"Tong san pham: {len(stocks)}")

# Count distribution
from collections import Counter
counter = Counter(s[1] for s in stocks)
print("\nPhan phoi stock:")
for val, count in sorted(counter.items()):
    print(f"  stock={val}: {count} san pham")

# Show a few samples with their doc stock line
print("\n--- 5 mau document (dong Ton kho) ---")
for doc, meta in zip(all_data["documents"][:10], all_data["metadatas"][:10]):
    for line in doc.split("\n"):
        if "Ton kho" in line or "stock" in line.lower():
            print(f"  SKU={meta.get('sku')} | meta.stock={meta.get('stock')} | doc: {line}")
            break
