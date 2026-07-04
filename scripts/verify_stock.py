"""Verify stock display in rebuilt ChromaDB documents."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
import chromadb

client = chromadb.PersistentClient(path=str(Path('chroma_db')))
col = client.get_collection('products')
data = col.get(include=['documents','metadatas'])

print("--- Kiem tra dong Tinh trang / Ton kho (15 mau) ---")
shown = 0
for doc, meta in zip(data['documents'], data['metadatas']):
    for line in doc.split('\n'):
        if 'Tinh trang' in line:
            if shown < 15:
                sku = meta['sku']
                stock = meta['stock']
                print(f"  SKU={sku:<12} | stock_meta={stock:4d} | {line}")
                shown += 1
            break

print()
print("--- San pham co ton kho thuc (khac 0 va 100) ---")
for doc, meta in zip(data['documents'], data['metadatas']):
    s = meta.get('stock', 0)
    if s not in (0, 100):
        for line in doc.split('\n'):
            if 'Tinh trang' in line:
                print(f"  SKU={meta['sku']:<12} | stock={s:4d} | {line}")
                break
