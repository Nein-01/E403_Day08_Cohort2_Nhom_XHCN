# RAG Evaluation Results

**Dataset:** 18 test cases  
**Model:** gpt-4o-mini (judge + generation)  
**Framework:** LLM-as-Judge (custom)

---

## Overall Scores

| Metric | Config A (Dense-only) | Config B (Hybrid+RRF) | Delta |
|--------|----------------------|----------------------|-------|
| Faithfulness    | 0.583 | 0.311 | -0.272 |
| Answer Relevance | 0.889 | 0.889 | +0.000 |
| Context Recall  | 0.639 | 0.556 | -0.083 |
| Context Precision| 0.683 | 0.656 | -0.027 |
| **Average**     | **0.699** | **0.603** | **-0.096** |

---

## A/B Comparison Analysis

- **Config A** (Dense-only): Chỉ dùng semantic search với OpenAI embeddings, không reranking.
- **Config B** (Hybrid+RRF): Kết hợp semantic search + BM25 lexical search, merge bằng RRF.

**Nhận xét:**
- RRF fusion cho phép kết quả từ cả 2 retriever, giảm miss rate.

---

## Worst Performers (Config B, 5 câu thấp nhất)

| # | Câu hỏi | Avg Score | Vấn đề |
|---|---------|-----------|--------|
| 1 | Ca sĩ Long Nhật có liên quan gì đến ma túy? | 0.000 | Thiếu context chi tiết |
| 2 | Rapper Bình Gold đã cướp tài sản gì và vào thời điểm nào? | 0.250 | Thiếu context chi tiết |
| 3 | Rapper Bình Gold là ai và có những thành tích âm nhạc gì? | 0.250 | Thiếu context chi tiết |
| 4 | Ca sĩ Chu Bin nổi tiếng với những bài hát nào? | 0.250 | Thiếu context chi tiết |
| 5 | Công an đã bắt Rapper Bình Gold ở đâu và sau hành vi gì? | 0.250 | Thiếu context chi tiết |

---

## Detailed Results (Config B)

| Câu hỏi | Faith. | Relev. | Recall | Prec. | Trạng thái |
|---------|--------|--------|--------|-------|------------|
| Rapper Bình Gold bị bắt tạm giam vì tội danh gì? | 0.00 | 1.00 | 0.00 | 1.00 | Xem lai |
| Ca sĩ Chu Bin bị bắt vì lý do gì? | 1.00 | 1.00 | 1.00 | 1.00 | Dat |
| Ca sĩ Miu Lê bị bắt với cáo buộc gì? | 1.00 | 1.00 | 1.00 | 1.00 | Dat |
| Nhà thiết kế Nguyễn Công Trí bị bắt vì lý do gì? | 1.00 | 1.00 | 1.00 | 1.00 | Dat |
| Ca sĩ Chi Dân bị truy tố cùng với những ai? | 0.00 | 1.00 | 1.00 | 1.00 | Dat |
| Trong vòng chưa đầy 1 tháng, có bao nhiêu nghệ sĩ ... | 0.00 | 1.00 | 1.00 | 1.00 | Dat |
| Ca sĩ Long Nhật có liên quan gì đến ma túy? | 0.00 | 0.00 | 0.00 | 0.00 | Xem lai |
| Rapper Bình Gold đã cướp tài sản gì và vào thời đi... | 0.00 | 1.00 | 0.00 | 0.00 | Xem lai |
| Những nghệ sĩ Việt Nam nào bị bắt vì liên quan đến... | 0.00 | 1.00 | 0.00 | 1.00 | Xem lai |
| Ca sĩ Miu Lê tên thật là gì? | 0.00 | 1.00 | 1.00 | 1.00 | Dat |
| Đường dây ma túy liên quan đến ca sĩ Chi Dân bao g... | 0.00 | 1.00 | 1.00 | 1.00 | Dat |
| Rapper Bình Gold là ai và có những thành tích âm n... | 1.00 | 0.00 | 0.00 | 0.00 | Xem lai |
| Tình trạng của Rapper Bình Gold khi thực hiện vụ c... | 0.00 | 1.00 | 0.00 | 0.50 | Xem lai |
| Ca sĩ Chu Bin nổi tiếng với những bài hát nào? | 0.00 | 1.00 | 0.00 | 0.00 | Xem lai |
| Việc nhiều nghệ sĩ bị bắt vì ma túy trong thời gia... | 0.80 | 1.00 | 1.00 | 0.80 | Dat |
| Công an đã bắt Rapper Bình Gold ở đâu và sau hành ... | 0.00 | 1.00 | 0.00 | 0.00 | Xem lai |
| Long Nhật có cuộc đời tư như thế nào trước khi bị ... | 0.00 | 1.00 | 1.00 | 0.50 | Xem lai |
| Nguyễn Công Trí là ai trong làng thời trang Việt N... | 0.80 | 1.00 | 1.00 | 1.00 | Dat |

---

## Recommendations

1. **Them data phap luat**: Hien tai chi co bai bao tin tuc, can them van ban phap luat de tra loi cau hoi ve dieu luat.
2. **Cai thien chunking**: Loc bo cac chunk navigation/menu, chi giu noi dung chinh.
3. **Cross-encoder reranking**: Dung Jina Reranker de cai thien Context Precision.
4. **Conversation memory**: Them follow-up question support cho chatbot.