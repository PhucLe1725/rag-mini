"""
build_product_rag.py
--------------------
Ket noi PostgreSQL (Aiven), lay bang products + categories,
tong hop thanh document text, embed va luu vao ChromaDB.

Cach dung:
  1. Dien thong tin PG_ vao .env
  2. pip install psycopg2-binary chromadb sentence-transformers python-dotenv
  3. python scripts/build_product_rag.py

Sau khi chay xong, ChromaDB collection 'products' da san sang.
"""
import sys
import io

# Force UTF-8 output (fix UnicodeEncodeError tren Windows)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import psycopg2
import psycopg2.extras
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Cau hinh
# ---------------------------------------------------------------------------
PG_HOST     = os.getenv("PG_HOST", "")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_USER     = os.getenv("PG_USER", "")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DBNAME   = os.getenv("PG_DBNAME", "")
PG_SSLMODE  = os.getenv("PG_SSLMODE", "require")

CHROMA_DIR      = ROOT / "chroma_db"
COLLECTION_NAME = "products"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------
def get_pg_connection():
    if not all([PG_HOST, PG_USER, PG_PASSWORD, PG_DBNAME]):
        print("Thieu thong tin ket noi PostgreSQL. Dien PG_HOST/PORT/USER/PASSWORD/DBNAME vao .env")
        sys.exit(1)
    print(f"Ket noi PostgreSQL: {PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DBNAME} ...")
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DBNAME,
        sslmode=PG_SSLMODE, connect_timeout=15,
    )
    print("Ket noi thanh cong!")
    return conn


def fetch_category_map(conn) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, name, parent_id FROM categories WHERE status = true ORDER BY id")
        rows = cur.fetchall()

    cat_map = {row["id"]: {"name": row["name"], "parent_id": row["parent_id"]} for row in rows}

    def get_path(cid, visited=None):
        if visited is None:
            visited = set()
        if cid not in cat_map or cid in visited:
            return cat_map.get(cid, {}).get("name", "")
        visited.add(cid)
        cat = cat_map[cid]
        if cat["parent_id"] and cat["parent_id"] in cat_map:
            return f"{get_path(cat['parent_id'], visited)} > {cat['name']}"
        return cat["name"]

    for cid in cat_map:
        cat_map[cid]["path"] = get_path(cid)

    print(f"  Da tai {len(cat_map)} categories.")
    return cat_map


def fetch_products(conn) -> list:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, category_id, name, sku, price, short_description,
                   full_description, brand, availability, stock_quantity
            FROM products
            ORDER BY id
        """)
        rows = cur.fetchall()
    print(f"  Da tai {len(rows)} san pham.")
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Keyword aliases de boost recall (tu dong them vao embed_text)
# ---------------------------------------------------------------------------
_KEYWORD_ALIASES = [
    # (pattern_in_name,           alias_to_append)
    ("2B",              "bút chì 2B ngòi đậm"),
    ("HB",              "bút chì HB ngòi vừa"),
    ("4B",              "bút chì 4B ngòi rất đậm"),
    ("có tẩy",          "bút chì kèm tẩy gôm"),
    ("gỗ",              "bút chì gỗ lục giác thân gỗ"),
    ("kim",             "bút chì kim mechanical pencil ngòi 0.5mm"),
    ("màu",             "bút chì màu color pencil"),
    ("Oringa",          "bút chì gỗ bạch dương Oringa Hồng Hà"),
    ("Justice League",  "bút chì gỗ Justice League Hồng Hà"),
    ("Siliver",         "bút chì gỗ 2B bạc Siliver"),
    ("ABC",             "bút chì ABC Hồng Hà học sinh"),
]


def build_embed_text(p: dict, cat_map: dict) -> str:
    """
    Tao text DANH CHO EMBEDDING, bao gom:
      - Ten san pham (nguyen ven)
      - Full category path (khong chi leaf) de tang tinh chinh xac danh muc
      - Thuong hieu
      - Toan bo short_description (thong so + dac diem)
      - Keyword aliases tu ten san pham (2B, HB, co tay, go...) de tang recall
    KHONG bao gom full_description de tranh semantic dilution.
    """
    cat_path = ""
    if p["category_id"] and p["category_id"] in cat_map:
        cat_path = cat_map[p["category_id"]]["path"]

    product_name = p["name"] or ""
    short_desc   = (p.get("short_description") or "").strip()
    brand        = (p.get("brand") or "").strip()

    parts = [product_name]

    # Full category path (khong chi leaf)
    if cat_path:
        parts.append(cat_path)

    # Thuong hieu (neu chua co trong ten)
    if brand and brand.upper() not in product_name.upper():
        parts.append(brand)

    # Short description day du (khong cat bot)
    if short_desc:
        parts.append(short_desc)

    # Keyword aliases dua tren ten san pham
    name_upper = product_name.upper()
    aliases = []
    for keyword, alias in _KEYWORD_ALIASES:
        if keyword.upper() in name_upper:
            aliases.append(alias)
    if aliases:
        parts.append(" | ".join(aliases))

    return "\n".join(parts)


def build_document(p: dict, cat_map: dict) -> str:
    """Tao full document de luu vao ChromaDB (hien thi cho LLM)."""
    cat_path = ""
    if p["category_id"] and p["category_id"] in cat_map:
        cat_path = cat_map[p["category_id"]]["path"]

    price_str = f"{int(p['price']):,}".replace(",", ".") + " VND"
    status = "Con hang" if p["availability"] else "Het hang"

    # Ton kho: gia tri 100 la mac dinh cua he thong, khong phan anh thuc te
    # Chi hien so khi la gia tri thuc (khac 100 va > 0)
    stock_qty = int(p["stock_quantity"] or 0)
    if stock_qty == 0:
        stock_str = "Het hang"
    elif stock_qty == 100:
        stock_str = "Con hang"          # an gia tri placeholder 100
    else:
        stock_str = f"{stock_qty} san pham"

    product_name = p['name'] or ""

    lines = [
        f"San pham: {product_name}",
        f"Danh muc: {cat_path}" if cat_path else "",
        f"Ma SKU: {p['sku']}",
        f"Thuong hieu: {p['brand']}" if p.get("brand") else "",
        f"Gia: {price_str}",
        f"Tinh trang: {status} | Ton kho: {stock_str}",
        "",
        "Mo ta ngan:",
        (p.get("short_description") or "").strip(),
        "",
        "Mo ta chi tiet:",
        (p.get("full_description") or "").strip(),
    ]
    return "\n".join(l for l in lines if l is not None)


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------
def build_rag_index(products: list, cat_map: dict, embed_model, chroma_client):
    print(f"\nBat dau embed {len(products)} san pham...")

    # ── Thong ke availability ──────────────────────────────────────────────
    available_count = sum(1 for p in products if p["availability"])
    unavailable_count = len(products) - available_count
    print(f"  Ket qua: {available_count} san pham con hang, {unavailable_count} het hang.")

    documents, embed_texts, metadatas, ids = [], [], [], []

    for p in products:
        cat_path = ""
        if p["category_id"] and p["category_id"] in cat_map:
            cat_path = cat_map[p["category_id"]]["path"]

        # Full document luu vao ChromaDB (hien thi cho LLM)
        documents.append(build_document(p, cat_map))

        # ── FIX: Chi embed text ngan gon (ten + danh muc + brand + short_desc)
        # Tranh semantic dilution tu full_description dai ──────────────────────
        embed_text = build_embed_text(p, cat_map)
        embed_texts.append(embed_text)

        metadatas.append({
            "product_id":    int(p["id"]),
            "category_id":   int(p["category_id"]) if p["category_id"] else -1,
            "category_path": cat_path,
            "sku":           str(p["sku"] or ""),
            "price":         float(p["price"]),
            "brand":         str(p["brand"] or ""),
            "availability":  1 if p["availability"] else 0,
            "stock":         int(p["stock_quantity"] or 0),
            "doc_type":      "product",
        })
        ids.append(f"product_{p['id']}")

    # In mau embed_text de kiem tra
    print("\n  [Preview] 3 embed_text dau tien:")
    for et in embed_texts[:3]:
        print(f"    ---\n    {et[:200]}")

    print("\n  Encoding embeddings (tu embed_text ngan gon)...")
    embeddings = embed_model.encode(
        embed_texts,          # ← EMBED BANG TEXT NGAN GON, KHONG PHAI FULL DOCUMENT
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    # Xoa collection cu neu ton tai
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"  Da xoa collection cu '{COLLECTION_NAME}'.")
    except Exception:
        pass

    # Tao collection moi voi cosine similarity
    col = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Da tao collection '{COLLECTION_NAME}' (cosine similarity).")

    # Upsert theo batch
    BATCH = 100
    for start in range(0, len(documents), BATCH):
        end = min(start + BATCH, len(documents))
        col.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],   # luu full doc cho LLM
            metadatas=metadatas[start:end],
        )
        print(f"  Luu batch {start+1}-{end} / {len(documents)}")

    count = col.count()
    print(f"\nHoan tat! Da luu {count} san pham vao ChromaDB collection '{COLLECTION_NAME}'.")
    return col


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Build Product RAG Index - Hong Ha")
    print("=" * 60)

    conn = get_pg_connection()

    print("\nDang tai du lieu tu database...")
    cat_map  = fetch_category_map(conn)
    products = fetch_products(conn)
    conn.close()
    print("Da dong ket noi PostgreSQL.")

    if not products:
        print("Khong co san pham nao trong database.")
        sys.exit(1)

    print(f"\nDang tai embedding model '{EMBED_MODEL_NAME}'...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print("Embedding model san sang.")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    build_rag_index(products, cat_map, embed_model, chroma_client)

    print(f"\nRAG index luu tai: {CHROMA_DIR}")
    print("Ban co the chay: python main.py")


if __name__ == "__main__":
    main()
