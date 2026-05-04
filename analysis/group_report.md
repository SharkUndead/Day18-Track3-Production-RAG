# Group Report — Lab 18: Production RAG

**Nhóm:** AI Team  
**Ngày:** 2024-05-05

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Nguyễn Duy Hiếu | M1: Chunking | ☑ | 13/13 |
| Nguyễn Duy Hiếu | M2: Hybrid Search | ☑ | 5/5 |
| Nguyễn Duy Hiếu | M3: Reranking | ☑ | 5/5 |
| Nguyễn Duy Hiếu | M4: Evaluation | ☑ | 4/4 |
| Nguyễn Duy Hiếu | M5: Enrichment | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.0000 | 0.8833 | +0.8833 |
| Answer Relevancy | 0.0000 | 0.7252 | +0.7252 |
| Context Precision | 0.0000 | 0.9917 | +0.9917 |
| Context Recall | 0.0000 | 0.9500 | +0.9500 |

## Key Findings

1. **Biggest improvement:** Việc chuyển từ Naive Search sang Hybrid Search (BM25 + Qdrant) kết hợp với Cross-Encoder Reranking đã giúp Context Recall và Context Precision tăng vọt lên gần mức tuyệt đối (>0.95).
2. **Biggest challenge:** Xử lý lỗi cạn kiệt tài nguyên (OOM) khi load nhiều model cùng lúc (BGE-M3 + Reranker + RAGAS Embeddings). Giải pháp là dùng model nhỏ hơn (MiniLM) cho khâu evaluation.
3. **Surprise finding:** Kỹ thuật Contextual Prepend (Module 5) giúp giảm đáng kể lỗi "mất ngữ cảnh" khi tài liệu bị cắt nhỏ, đặc biệt là với các bảng biểu hoặc quy định liệt kê.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** Production pipeline vượt xa baseline nhờ cơ chế lọc và sắp xếp lại kết quả tìm kiếm. Faithfulness đạt >0.85 (mức bonus).
2. **Biggest win — module nào, tại sao:** Module 3 (Reranking). Dù tìm kiếm Hybrid đã tốt, nhưng Reranker giúp chọn ra đúng 3 chunks liên quan nhất, giúp LLM không bị nhiễu bởi các thông tin tương tự nhưng không chính xác.
3. **Case study — 1 failure, Error Tree walkthrough:** Câu hỏi về "Nhân viên thử việc". Context đúng nhưng LLM trả lời chưa đầy đủ (vấn đề ở Generation). Cần optimize Prompt.
4. **Next optimization nếu có thêm 1 giờ:** Triển khai HyQA (Module 5) sâu hơn để sinh thêm các câu hỏi giả định, làm giàu vector database giúp tăng khả năng tìm kiếm ngữ nghĩa cho các câu hỏi lắt léo.
