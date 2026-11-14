# Báo cáo cá nhân — Day 6: Data Ingestion Engineering

**Học viên:** Võ Thị Phương Nhung — **MSSV:** 2A202603918 — **Lớp:** L3B
**Repo:** [K5-L3B-Day06-VoThiPhuongNhung-2A202603918-DataIngestionEngineering](https://github.com/nhungvo631-stack/K5-L3B-Day06-VoThiPhuongNhung-2A202603918-DataIngestionEngineering)

## 1. Dữ liệu và pipeline

- **Corpus:** *Pass It On: Stories, Volume Two*, do The Foundation for a Better Life xuất bản. PDF có 58 trang vật lý; manifest chọn 32 trang chứa nội dung truyện, loại bìa, phần giới thiệu/mục lục, trang chỉ có tiêu đề và phần cuối không phải nội dung truyện. Các số trang trong gold và trace là số trang vật lý của PDF.
- **Nguồn và quyền:** PDF từ [passiton.com](https://cdn.passiton.com/ebooks/FBL_PassItOn_Stories_Volume_2.pdf). Terms of Use được ghi trong manifest cho phép sử dụng/tái phân phối có giới hạn cho mục đích giáo dục hoặc cá nhân phi thương mại, với điều kiện giữ thông báo bản quyền/thương hiệu; đây không phải giấy phép dữ liệu mở hoặc quyền sử dụng thương mại. Manifest ghi ngày truy cập 2026-11-14 và SHA-256 của snapshot.
- **Đọc và chuẩn hóa:** `pypdf` trích xuất theo trang; chuẩn hóa Unicode NFC, xuống dòng và khoảng trắng, vẫn giữ ranh giới đoạn. Lỗi trích xuất được log rồi báo lỗi; trang không có text được ghi cảnh báo. Pipeline đối chiếu trang, source ID và evidence quote trong câu hỏi trước khi lập chỉ mục.
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, chạy local trên CPU. Mã dùng mean pooling có attention mask và chuẩn hóa L2; cùng model mã hóa query và document. Version/revision cụ thể của model, PyTorch và Transformers chưa được lưu trong run metadata.
- **Vector store:** Chroma Persistent 1.5.9, metric cosine, HNSW approximate; dữ liệu lưu tại `starter/results/chroma`. Chi phí API ghi nhận: **$0** (embedding chạy local). Tổng thời gian ingest/index của lần chạy: **30.394 giây**.
- **Metadata:** `document_id`, `source`, `page`, `start_char`, `end_char`; bổ sung `story_id` và `story_title` từ manifest để lọc theo truyện. `story_id` lấy từ số truyện trong manifest, không suy đoán từ nội dung truy vấn.

## 2. Cấu hình và benchmark

| Cấu hình | Chunking | Metadata / truy xuất |
|---|---|---|
| A | Fixed-size 500 ký tự, overlap 50 | Giữ metadata truy vết; tìm kiếm không filter |
| B | Theo đoạn, tối đa 700 ký tự | Cùng metadata; tìm kiếm không filter |
| C | Giống B | Chỉ áp dụng filter `story_id` khi câu hỏi cung cấp filter |

Benchmark có **20 câu**: 15 dev và 5 test. Có hai câu filter ở dev (`q03`, `q12`) và một câu filter ở test (`q18`). Với mỗi câu, file câu hỏi lưu đáp án kỳ vọng, source ID, trang gold và evidence quote; code xác minh quote tồn tại ở trang đã chọn trước khi chạy. Cả ba cấu hình dùng chung câu hỏi, model và top-3. Filter không chứa gold source hay đáp án; B–C chỉ khác việc áp dụng metadata filter. Quy tắc chọn đã lưu trong metadata: chọn theo Evidence Hit@3 trên dev, dùng grounding và tài nguyên làm tiêu chí phụ. Không dùng kết quả test dưới đây để điều chỉnh cấu hình.

## 3. Kết quả truy xuất tự động

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Số chunk | Ingest/index (giây) | Chi phí |
|---|---|---:|---:|---:|---:|---:|
| A | Dev | 12/15 (80%) | 15/15 (100%) | 174 | 13.903 | $0 |
| A | Test | 3/5 (60%) | 5/5 (100%) | 174 | 13.903 | $0 |
| B | Dev | 9/15 (60%) | 13/15 (86.7%) | 122 | 14.336 | $0 |
| B | Test | 3/5 (60%) | 5/5 (100%) | 122 | 14.336 | $0 |
| C | Dev | 9/15 (60%) | 13/15 (86.7%) | 122 | 1.094 | $0 |
| C | Test | 3/5 (60%) | 5/5 (100%) | 122 | 1.094 | $0 |

| Strategy | Coherence trung bình /2 (n) | Metadata thủ công đủ/đúng | Filter tuân thủ | Grounding thủ công |
|---|---|---|---|---|
| A | 1.0/2 (10) | 15/15 | N/A — không áp dụng filter | 0:0, 1:0, 2:5; trung bình 2.0/2 (5) |
| B | 1.1/2 (10) | 15/15 | N/A — không áp dụng filter | 0:0, 1:0, 2:5; trung bình 2.0/2 (5) |
| C | 1.1/2 (10) | 15/15 | 1/1 câu filter test; 3/3 cả tập | 0:0, 1:0, 2:5; trung bình 2.0/2 (5) |

Audit thủ công có AI hỗ trợ, đối chiếu chunk với trang PDF, manifest và rubric; cần tác giả rà soát nhãn trước khi nộp. Nhãn và lý do từng mẫu được lưu trong [manual_audit_template.json](../../starter/results/manual_audit_template.json). Coherence: A được 1/2 ở cả 10 vị trí vì đoạn ý chính vẫn hiểu được nhưng bị cắt giữa từ/câu. B và C giống nhau ở các vị trí mẫu; được 1/2 ở 9 vị trí và 2/2 ở trang 15, nơi đoạn hoàn chỉnh và tự hiểu được. Metadata: kiểm tra 15 kết quả top-3 trên test cho mỗi cấu hình; `source`, `document_id`, `page`, `story_id`, `story_title` và offset đầy đủ, đúng với PDF/manifest trong cả 45/45 kết quả. Grounding: cả 5 câu test ở mỗi cấu hình đều có bằng chứng đủ trong top-3; các câu `q17` và `q20` có bằng chứng ở hạng thấp hơn top-1. Filter C được áp dụng đúng trên `story_id` ở cả ba câu filter (dev/test); với test là 1/1. A và B không áp dụng filter nên ghi N/A.

Trace top-3: [traces.jsonl](../../starter/results/traces.jsonl). Tổng hợp retrieval: [retrieval_summary.json](../../starter/results/retrieval_summary.json). Tham số, thời gian và model: [run_metadata.json](../../starter/results/run_metadata.json). Cách chọn mẫu coherence được mã cố định ở 10 trang 7, 9, 11, 13, 15, 17, 19, 21, 23, 25; mỗi trang lấy chunk bao phủ offset ký tự 150.

## 4. Phân tích và kết luận

- **So sánh A–B và B–C; hai ví dụ sai có ID câu và nguồn:** Trên dev, A hơn B 3/15 Hit@1 và 2/15 Evidence Hit@3, nhưng dùng 174 chunk so với 122; audit coherence A 1.0/2, B 1.1/2. B–C có cùng chunk, kết quả retrieval tổng hợp bằng nhau; C lọc đúng `story_id` cho 3/3 câu filter, còn B không áp dụng filter. `q17` — nguồn [Pass It On: Stories, Volume Two, trang PDF 39](https://cdn.passiton.com/ebooks/FBL_PassItOn_Stories_Volume_2.pdf): top-1 trả về trang 47 thay vì evidence ở trang 39 (evidence xuất hiện ở hạng 3). `q20` — cùng nguồn, trang PDF 45: top-1 không chứa evidence; evidence xuất hiện ở hạng 3 với A và hạng 2 với B/C. Đây là lỗi xếp hạng top-1, không phải lỗi thiếu bằng chứng trong top-3.
- **Chiến lược chọn, đánh đổi, điều kiện áp dụng:** Chọn A theo quy tắc đã chốt vì Evidence Hit@3 dev cao nhất (15/15 so với 13/15 của B/C); test chỉ cho thấy cả ba cùng 5/5 Evidence Hit@3 và không chứng minh A tổng quát tốt hơn. A phù hợp khi ưu tiên retrieval top-3 trên corpus/bộ câu hỏi này; B/C có ít chunk hơn và coherence mẫu nhỉnh hơn. Chọn C khi ứng dụng cần lọc theo `story_id` đáng tin được cung cấp trước truy xuất; không dùng A/B nếu filter là yêu cầu bắt buộc.
- **Hạn chế dữ liệu, nhãn, test, thời gian/chi phí:** Corpus là một tuyển tập 32 trang nội dung, test chỉ 5 câu và chỉ 1 câu filter; nhãn coherence/grounding chủ quan và audit có AI hỗ trợ, cần tác giả rà soát. Hit@3 đo truy xuất evidence, không đo độ đúng câu trả lời LLM. Chi phí $0 là chi phí API; không tính tài nguyên máy. Thời gian C (1.094 giây) không so sánh trực tiếp được với B vì embedding được tái sử dụng; kết quả không phản ánh cold run. HNSW là approximate search, và B/C có chunk ngắn nhất 6 ký tự.
- **Nếu làm lại hoặc mở rộng:** Tăng số câu test và câu filter trên nhiều `story_id`, thêm câu cần điều kiện/ngoại lệ; chấm nhãn thủ công độc lập hoặc đối chiếu giữa hai người; chạy lặp cold/warm để báo median và độ lệch thời gian, đo rõ tài nguyên; ghi version/revision của model và dependencies; kiểm tra, lọc hoặc gộp chunk cực ngắn rồi benchmark lại trên cùng bộ câu hỏi.
