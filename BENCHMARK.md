# Benchmark — năm tiêu chí đánh giá

Mục tiêu là so sánh **ít nhất ba chiến lược dữ liệu** trên cùng corpus, model và bộ câu hỏi để chọn chiến lược phù hợp, kèm bằng chứng và lý do. Benchmark gồm bộ câu hỏi có gold, kết quả truy xuất, mẫu chấm thủ công và phân tích theo **năm tiêu chí** dưới đây.

| Tiêu chí | Cần đo hoặc kiểm tra | Minh chứng |
|---|---|---|
| **1. Độ chính xác truy xuất** | Hit@1 và Evidence Hit@3 trên cùng bộ câu hỏi | Số câu đúng/tổng câu, top-3 và gold theo từng chiến lược |
| **2. Tính mạch lạc của chunk** | Chunk có giữ trọn ý và đủ ngữ cảnh để hiểu không? | Chấm thủ công cùng 10 vị trí nguồn cho mỗi chiến lược, thang 0–2, kèm lý do |
| **3. Tính hữu dụng của metadata** | Metadata có đúng, đủ để truy vết và giúp lọc phạm vi không? | Kiểm tra `source`, `page` và trường lọc trong top-3; tỷ lệ tuân thủ filter khi có yêu cầu |
| **4. Chất lượng thông tin nền (Grounding Quality)** | Top-3 có đủ bằng chứng và điều kiện để trả lời, có truy được nguồn không? | Chấm thủ công 5 câu test mỗi chiến lược, thang 0–2, kèm lý do |
| **5. Tác động của chiến lược dữ liệu** | Các thay đổi về chunking/metadata ảnh hưởng kết quả, chất lượng và tài nguyên ra sao? | So sánh A–B và B–C, số chunk, thời gian, chi phí, ví dụ lỗi và đánh đổi |

Định nghĩa chi tiết thang điểm, mẫu và công thức xem [GUIDE.md](GUIDE.md). **Grounding ở đây đánh giá độ đủ của bằng chứng truy xuất**, không phải độ chính xác của câu trả lời LLM. Hit@3 không thay cho chấm thủ công coherence hoặc grounding. Điểm bài làm theo [RUBRIC.md](RUBRIC.md); năm tiêu chí trên là các khía cạnh cần báo cáo, không phải năm loại câu hỏi.

## Thiết lập benchmark hiện tại

Benchmark đang dùng corpus *Pass It On: Stories, Volume Two* (32 trang nội dung được chọn từ 58 trang PDF), cùng model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, chạy CPU. Vector được lưu bền vững bằng Chroma 1.5.9, cosine, HNSW approximate; mọi cấu hình truy xuất top-3.

| Cấu hình | Chunking | Filter |
|---|---|---|
| A | Fixed-size 500 ký tự, overlap 50 | Không |
| B | Theo đoạn, tối đa 700 ký tự | Không |
| C | Giống B | Lọc `story_id` khi câu hỏi có filter |

Tập câu hỏi hiện có 20 câu (15 dev / 5 test), gồm 3 câu filter: `q03` và `q12` ở dev, `q18` ở test. `story_id` lấy từ manifest; filter được áp dụng chỉ trong C và không dùng gold source hoặc đáp án. Mã kiểm tra evidence quote trên trang nguồn trước khi chạy. Tiêu chí chọn cấu hình đã ghi trong metadata: tối đa hóa Evidence Hit@3 trên dev, rồi dùng grounding và tài nguyên làm tiêu chí phụ; không chọn theo test.

Kết quả truy xuất hiện lưu tại `starter/results/retrieval_summary.json`, trace top-3 tại `starter/results/traces.jsonl`, và thông tin lần chạy tại `starter/results/run_metadata.json`. Kết quả tổng hợp: A đạt Evidence Hit@3 15/15 dev và 5/5 test; B/C đạt 13/15 dev và 5/5 test. Hit@1 là A 12/15 dev, B/C 9/15 dev; cả ba đạt 3/5 trên test. Đây là số liệu truy xuất tự động, **không thay thế** nhãn thủ công coherence, metadata hoặc grounding. Các nhãn thủ công vẫn cần hoàn tất trong `starter/results/manual_audit_template.json`; không được xem giá trị trống là điểm 0 hoặc điểm đạt.

## Bộ câu hỏi và gold

Benchmark hiện có **20 câu** bằng tiếng Anh (ngôn ngữ corpus): 15 câu dev và 5 câu test. Gold source của tất cả câu là `fbl-passiton-stories-volume-2`; trang là số trang vật lý trong PDF. `evidence_quote` được lưu nguyên câu/đoạn nguồn (sau chuẩn hóa khoảng trắng); `expected_answer` là đáp án cần truy xuất, không được thêm vào query. Bộ câu hỏi đầy đủ dạng JSON có tại [starter/data/questions.json](starter/data/questions.json).

### Dev (15 câu)

| ID / loại | Câu hỏi | Đáp án kỳ vọng | Gold page | Evidence quote | Filter |
|---|---|---|---:|---|---|
| q01 · direct | Which sport is Simone Biles known for? | Gymnastics. | 7 | “one of the greatest gymnasts in the sport’s history” | — |
| q02 · direct | Which group did Graham Nash co-found with his school friend Allan Clarke? | The Hollies. | 9 | “Graham Nash co- founded The Hollies with his school friend, Allan Clarke” | — |
| q03 · metadata_filter | In story 3, how many pets did Captain Dusty Mascaro save during the Hillsborough County flood? | More than 100 pets. | 11 | “saved more than 100 pets in Hillsborough County, Florida” | `story_id=3` |
| q04 · direct | What group of organisms is Anna Devolld passionate about helping? | Pollinators. | 13 | “Anna is passionate about pollinators” | — |
| q05 · direct | What caused Amy Purdy to lose her legs at age 19? | Bacterial meningitis. | 15 | “Amy Purdy lost her legs to bacterial meningitis at the age of 19” | — |
| q06 · paraphrase | What subject does Arthur Brooks teach at Harvard? | Happiness. | 17 | “His class on happiness has the longest waiting list in Harvard’s catalog” | — |
| q07 · direct | What did the trucker community do to help save a man's life? | They carried out an impromptu maneuver. | 19 | “pulled off an impromptu maneuver that saved a man’s life” | — |
| q08 · direct | After being asked to leave a restaurant because of his facial disfigurement, what did Oliver Bromley ask people to seek? | Understanding. | 21 | “He only asks that we all seek understanding” | — |
| q09 · paraphrase | What community project does Austin Picinich use to help Seattle's salmon? | He paints community murals. | 23 | “saving the salmon of Seattle by painting community murals” | — |
| q10 · direct | In which country did Ta’u Pupu’a grow up? | Tonga. | 25 | “Ta’u Pupu’a grew up in Tonga” | — |
| q11 · direct | What about Karissa Trinchera's birth increased the odds that she would develop more slowly than typical children? | She was born prematurely. | 27 | “Karissa Trinchera was born prematurely” | — |
| q12 · metadata_filter | In story 12, where did Preston Pollard move after high school to pursue professional skateboarding? | Los Angeles. | 29 | “He moved to Los Angeles after high school and became a professional skateboarder” | `story_id=12` |
| q13 · paraphrase | According to the Tolstoy quote in the story, what does an intelligent person discover in others? | Kindness. | 31 | “the more he discovers kindness in others” | — |
| q14 · direct | In what year was James Croll born? | 1821. | 33 | “James Croll was born into abject poverty in 1821” | — |
| q15 · direct | What endurance feat did Lawrence complete? | 50 iron-distance triathlons in 50 days in 50 states. | 35 | “50 iron-distance triathlons in 50 days in 50 states” | — |

### Test (5 câu)

| ID / loại | Câu hỏi | Đáp án kỳ vọng | Gold page | Evidence quote | Filter |
|---|---|---|---:|---|---|
| q16 · direct | When his family could afford either shoes or school fees, what did Mustapha choose? | School fees; he went barefoot. | 37 | “Mustapha chose school, and every day he walked two miles to class, barefoot” | — |
| q17 · direct | During which major conflict did Major League Baseball face a possible shutdown? | World War II. | 39 | “Major League Baseball faced a shutdown during World War II” | — |
| q18 · metadata_filter | In story 18, how did Dr. Taryn Marie respond to her failing grades as a child? | She used them as motivation to do better. | 41 | “she allowed them to motivate her to do better” | `story_id=18` |
| q19 · direct | What activity did JuJu love? | Dancing. | 43 | “JuJu loved to dance” | — |
| q20 · direct | What happened to Julie and Ron Lynam's dream home, prompting them to rebuild? | It was lost in a forest fire. | 45 | “Julie and Ron Lynam lost their dream home in a forest fire” | — |

### Kiểm chứng và sử dụng gold

Manifest xác định trang, truyện và số ký tự nội dung: [starter/data/manifest.json](starter/data/manifest.json). Trước khi lập chỉ mục, `starter/src/run.py` kiểm tra đủ 20 câu, split, ID duy nhất, tối thiểu 3 câu filter (gồm test), trang gold tồn tại trong corpus, source ID trùng khớp, evidence quote xuất hiện trên trang sau chuẩn hóa NFC/whitespace, và `story_id` của filter khớp truyện trên trang gold. Vì vậy, bảng này là nhãn gold của bộ câu hỏi đã chạy, không phải nhãn do điểm retrieval suy ra.

Các câu `q03`, `q12`, `q18` cung cấp filter theo `story_id` trước khi truy xuất; mã chỉ bật filter trong cấu hình C. A và B không lọc. Không câu nào dùng gold page, source ID hoặc đáp án làm query/filter. Với câu không có filter, C cũng truy xuất không filter. Evidence quote là đoạn hỗ trợ trực tiếp cho đáp án; nếu thay đổi câu hỏi hoặc gold, cần xác minh lại quote với PDF và chạy lại benchmark.
