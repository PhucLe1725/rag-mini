"""Check short_description in SQL dump for pencil SKUs."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

target_skus = {'3532','3552','3550','3400','3404','3507','3506','3551','3520'}

with open('data/products_202606291416.sql', encoding='utf-8', errors='replace') as f:
    content = f.read()

lines = content.split('\n')
for line in lines:
    for sku in target_skus:
        if f"'{sku}'" in line:
            # Print first 400 chars of the line to see short_description
            print(f"SKU {sku}:")
            print(line[:500])
            print("---")
            break
