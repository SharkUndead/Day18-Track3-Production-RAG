# Failure Analysis — Lab 18: Production RAG

**Nhóm:** AI Team  
**Thành viên:** Nguyễn Duy Hiếu

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.0000 | 0.8833 | +0.8833 |
| Answer Relevancy | 0.0000 | 0.7252 | +0.7252 |
| Context Precision | 0.0000 | 0.9917 | +0.9917 |
| Context Recall | 0.0000 | 0.9500 | +0.9500 |

## Bottom-5 Failures

### #1
- **Question:** Yêu cầu cấp mới thiết bị làm việc (chuột, bàn phím) như thế nào?
- **Expected:** Nhân viên tạo ticket trên hệ thống quản lý tài sản hoặc gửi email cho bộ phận Hành chính - IT.
- **Got:** (Hallucination or incomplete answer)
- **Worst metric:** faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng? (Yes) → Query OK? (Yes)
- **Root cause:** LLM không trích xuất được chi tiết về "email" hoặc "ticket" dù context có đủ. Có thể do prompt template quá dài hoặc instruction không đủ chặt chẽ.
- **Suggested fix:** Tighten prompt, hạ temperature xuống 0.

### #2
- **Question:** Nhân viên thử việc có được hưởng ngày nghỉ phép năm không?
- **Expected:** Thời gian thử việc không được tính nghỉ phép ngay, nhưng sau khi ký hợp đồng chính thức, thời gian thử việc sẽ được tính vào thâm niên nghỉ phép năm.
- **Got:** (Answer was partially relevant but missed the nuance)
- **Worst metric:** answer_relevancy (0.40)
- **Error Tree:** Output chưa sát → Context đúng? (Yes)
- **Root cause:** LLM tập trung quá nhiều vào việc "không được nghỉ ngay" mà quên mất phần "được tính thâm niên sau này".
- **Suggested fix:** Cải thiện prompt để yêu cầu trả lời đầy đủ các vế của câu hỏi.

### #3
- **Question:** Ông nội, bà nội mất thì được nghỉ bao nhiêu ngày?
- **Expected:** Ông nội, bà nội mất thì được nghỉ 1 ngày không hưởng lương.
- **Got:** (Faithfulness issue)
- **Worst metric:** faithfulness (0.50)
- **Root cause:** Context có nhắc đến "1 ngày không hưởng lương" nhưng LLM có thể bị lẫn với các quy định 3 ngày của tứ thân phụ mẫu.

### #4
- **Question:** Con kết hôn thì nhân viên được nghỉ bao nhiêu ngày?
- **Expected:** Con kết hôn thì nhân viên được nghỉ 1 ngày hưởng nguyên lương.
- **Worst metric:** faithfulness (0.50)
- **Root cause:** Tương tự như trên, LLM dễ bị nhầm lẫn giữa các con số 1 ngày và 3 ngày trong cùng một đoạn văn bản policy.

### #5
- **Question:** Quy định về trang phục khi đi làm?
- **Expected:** Nhân viên cần mặc trang phục lịch sự, chỉn chu và chuyên nghiệp khi đến văn phòng làm việc.
- **Worst metric:** faithfulness (0.67)
- **Root cause:** Câu trả lời có thể bị thêm thắt các chi tiết không có trong context (ví dụ: "com-lê", "cà vạt").

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên thử việc có được hưởng ngày nghỉ phép năm không?"

**Error Tree walkthrough:**
1. Output đúng? → Một nửa (Partial). LLM trả lời là không được nghỉ ngay, đúng nhưng chưa đủ.
2. Context đúng? → Đúng. Chunk trích xuất có đủ thông tin về việc cộng dồn thâm niên sau thử việc.
3. Query rewrite OK? → OK. Hybrid search đã tìm được đúng đoạn văn bản.
4. Fix ở bước: Bước Generation (LLM). Cần prompt yêu cầu "trả lời toàn diện dựa trên thông tin được cung cấp".

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tối ưu Prompt Engineering bằng kỹ thuật Few-shot để LLM trả lời sát với Ground Truth hơn.
- Thử nghiệm với các model Reranker mạnh hơn như BGE-Reranker-v2-Gemma để đảm bảo Context Precision tuyệt đối 1.0.
