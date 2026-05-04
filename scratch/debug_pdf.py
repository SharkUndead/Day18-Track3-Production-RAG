import os, sys, fitz
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
from config import DATA_DIR

for fp in [os.path.join(DATA_DIR, "BCTC.pdf"), os.path.join(DATA_DIR, "Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf")]:
    if not os.path.exists(fp):
        print(f"File not found: {fp}")
        continue
    doc = fitz.open(fp)
    print(f"File: {os.path.basename(fp)}, Pages: {len(doc)}")
    text = "\n".join([page.get_text() for page in doc])
    print(f"  Extracted text length: {len(text)}")
    print(f"  First 100 chars: {repr(text[:100])}")
