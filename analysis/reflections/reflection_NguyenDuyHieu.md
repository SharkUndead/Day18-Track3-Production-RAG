# Individual Reflection — Nguyễn Duy Hiếu

- **Đóng góp kỹ thuật cụ thể:**
    - Xây dựng Pipeline End-to-End (`pipeline.py`) tích hợp toàn bộ 5 module.
    - Triển khai Module 5 (Enrichment) với kỹ thuật Contextual Prepend.
    - Tối ưu Module 4 (Evaluation) để xử lý lỗi version RAGAS và tính toán aggregate an toàn.
    - Cấu hình Hybrid Search (BM25 + Qdrant) và tích hợp Cross-Encoder Reranker.
    - Xây dựng bộ test set 20 câu hỏi và Knowledge Base tương ứng.

- **Kiến thức học được + kết nối với bài giảng:**
    - Hiểu sâu về luồng Retrieval-Augmented Generation trong môi trường production thực tế.
    - Ứng dụng kỹ thuật Hybrid Search để cân bằng giữa từ khóa (keyword) và ngữ nghĩa (semantic).
    - Hiểu tầm quan trọng của Reranking trong việc giảm nhiễu cho LLM.
    - Học cách debug các hệ thống phức tạp khi gặp lỗi RAM/API.

- **Khó khăn & cách giải quyết:**
    - Lỗi OOM khi chạy Reranker: Khắc phục bằng cách chuyển embedding của RAGAS sang model nhẹ hơn (`paraphrase-multilingual-MiniLM-L12-v2`).
    - Lỗi API OpenAI/Groq: Cấu hình linh hoạt qua `.env` và thêm logic fallback khi API fail.
    - Git bị lỗi file lớn (venv): Sử dụng `.gitignore` và `git rm --cached` để dọn dẹp repo.

- **Tự đánh giá:** 5/5
