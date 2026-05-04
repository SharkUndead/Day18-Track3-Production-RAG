import os, sys, glob
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
from src.m1_chunking import load_documents
from config import DATA_DIR

print(f"DATA_DIR: {DATA_DIR}")
print(f"Files in DATA_DIR: {os.listdir(DATA_DIR)}")
docs = load_documents(DATA_DIR)
print(f"Loaded {len(docs)} documents")
for doc in docs:
    print(f"  Source: {doc['metadata']['source']}, Length: {len(doc['text'])}")
