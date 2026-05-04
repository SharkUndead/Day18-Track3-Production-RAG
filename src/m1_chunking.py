"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    # Load Markdown files
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
            
    # Load PDF files
    try:
        import fitz  # PyMuPDF
        for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
            doc = fitz.open(fp)
            text = "\n".join([page.get_text() for page in doc])
            if text.strip():
                docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
    except ImportError:
        print("Thu vien 'pymupdf' chua duoc cai dat. Khong the doc file PDF. Chay: pip install pymupdf")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    import re
    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm

    # 1. Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences (sử dụng model nhẹ để chạy nhanh)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    def cosine_sim(a, b):
        if norm(a) == 0 or norm(b) == 0: return 0.0
        return dot(a, b) / (norm(a) * norm(b))

    chunks = []
    current_group = [sentences[0]]

    # 3 & 4. Compare and group sentences
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group), 
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []
        current_group.append(sentences[i])

    # 5. Add the last group
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group), 
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))
        
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents, children = [], []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return parents, children

    # 1. Split text into parents
    current_parent_text = ""
    for p in paragraphs:
        if len(current_parent_text) + len(p) > parent_size and current_parent_text:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=current_parent_text.strip(), 
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))
            current_parent_text = ""
        current_parent_text += p + "\n\n"
        
    if current_parent_text.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=current_parent_text.strip(), 
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))

    # 2. Split each parent into children
    for parent in parents:
        pid = parent.metadata["parent_id"]
        p_text = parent.text
        start = 0
        while start < len(p_text):
            end = min(start + child_size, len(p_text))
            child_text = p_text[start:end].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text, 
                    metadata={**metadata, "chunk_type": "child"}, 
                    parent_id=pid
                ))
            start += child_size

    # 3. Return (parents_list, children_list)
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    import re
    # 1. Split by markdown headers
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    
    chunks = []
    current_header = ""
    current_content = ""
    
    # 2. Pair headers with their content
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip() or current_header.strip():
                if current_content.strip():
                    chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={**metadata, "section": current_header, "strategy": "structure"}
                    ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
            
    if current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "section": current_header, "strategy": "structure"}
        ))
        
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {"basic": {}, "semantic": {}, "hierarchical": {}, "structure": {}}
    
    all_basic, all_semantic, all_parents, all_children, all_structure = [], [], [], [], []
    
    # 1. For each doc, run all strategies
    for doc in documents:
        text = doc["text"]
        meta = doc["metadata"]
        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))
        p, c = chunk_hierarchical(text, metadata=meta)
        all_parents.extend(p)
        all_children.extend(c)
        all_structure.extend(chunk_structure_aware(text, metadata=meta))
        
    # 2. Collect stats function
    def calc_stats(chunks_list):
        if not chunks_list:
            return {"chunks": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        lengths = [len(c.text) for c in chunks_list]
        return {
            "chunks": len(chunks_list),
            "avg_len": int(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths)
        }
        
    results["basic"] = calc_stats(all_basic)
    results["semantic"] = calc_stats(all_semantic)
    
    p_stats = calc_stats(all_parents)
    c_stats = calc_stats(all_children)
    results["hierarchical"] = {
        "chunks": f"{p_stats['chunks']}p/{c_stats['chunks']}c",
        "avg_len": c_stats["avg_len"],
        "min_len": c_stats["min_len"],
        "max_len": p_stats["max_len"]
    }
    
    results["structure"] = calc_stats(all_structure)
    
    # 3. Print comparison table
    print(f"\n{'Strategy':<15} | {'Chunks':<8} | {'Avg Len':<8} | {'Min':<5} | {'Max':<5}")
    print("-" * 50)
    for k, v in results.items():
        chunks_str = str(v['chunks'])
        print(f"{k:<15} | {chunks_str:<8} | {v['avg_len']:<8} | {v['min_len']:<5} | {v['max_len']:<5}")
        
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
