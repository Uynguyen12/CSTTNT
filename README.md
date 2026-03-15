CSTTNT/
├── algorithms/                              # Chứa toàn bộ logic thuật toán (Classical & Nature-Inspired)
│   ├── __init__.py                          # Khai báo package gốc, export các class dùng chung.
│   ├── base_optimizer.py                    # [Lõi Tối ưu] Chứa BaseOptimizer (vòng lặp chung, early-stop) & OptimizationResults.
│   ├── classical/                           # Nhóm thuật toán Tìm kiếm Đồ thị (Graph Search)
│   │   ├── __init__.py                      # Export thuật toán & các hàm khởi tạo nhanh (factory functions).
│   │   ├── base_search.py                   # [Lõi Tìm kiếm] Chứa BaseSearchAlgorithm (đo thời gian, lưu vết đường đi, tính cost).
│   │   ├── informed.py                      # Tìm kiếm có thông tin / Dùng Heuristic (A*, Greedy Best-First).
│   │   ├── local_search.py                  # Tìm kiếm cục bộ trên đồ thị (Hill Climbing, Simulated Annealing).
│   │   └── uninformed.py                    # Tìm kiếm mù (BFS, DFS, Uniform Cost Search).
│   └── nature_inspired/                     # Nhóm thuật toán Tối ưu hóa Cảm hứng Thiên nhiên
│       ├── __init__.py                      # Export thuật toán & factory functions phân theo nhóm.
│       ├── evolution_based.py               # Nhóm Tiến hóa sinh học (GA, DE) - Xử lý lai ghép, đột biến.
│       ├── human_based.py                   # Nhóm Hành vi con người (TLBO) - Thuật toán nâng cao không cần chỉnh tham số.
│       ├── physics_based.py                 # Nhóm Quy luật Vật lý (GSA, HS) - Mô phỏng trọng lực, dò âm điệu.
│       └── swarm_based.py                   # Nhóm Bầy đàn sinh học (PSO, ACO, ABC, FA, CS) - Mô phỏng chim, kiến, ong...
│
├── examples/                                # Chứa các file script chạy thử (.py) cho từng nhóm
│   ├── 01_graph_search_examples.py          # Chạy thuật toán tìm kiếm đồ thị cổ điển (BFS, DFS, UCS, A*, Greedy) và in bảng so sánh.
│   ├── 02_evolution_based_examples.py       # Tối ưu hàm liên tục bằng GA và DE trên các hàm benchmark.
│   ├── 03_swarm_based_examples.py           # Chạy thử và so sánh kết quả nhóm swarm (chủ yếu PSO và ACO).
│   ├── 04_physics_human_based_examples.py   # Chạy thử Simulated Annealing, Hill Climbing, Harmony Search.
│   ├── 05_complete_benchmark.py             # Benchmark tổng hợp toàn bộ thuật toán trên hàm liên tục và đồ thị, in thống kê cuối cùng.
│   └── README.md                            # Hướng dẫn chi tiết cách chạy ví dụ, giải thích output và tham số cấu hình.
│
├── plots/                                   # Nơi lưu các biểu đồ xuất ra tự động (.png)
│                                            # Bao gồm: benchmark functions, graph scalability, tsp routes, knapsack comparison,
│                                            # convergence curves, robustness boxplot, hypothesis testing heatmap, exploration vs exploitation...
│                                            # Dùng để chèn trực tiếp vào báo cáo PDF và Slides.
│
├── results/                                 # Nơi lưu các file số liệu thống kê kết quả (.csv)
│                                            # Lưu dữ liệu số, thống kê thô (file .csv) sau khi chạy benchmark.
│
├── test_cases/                              # Chứa toàn bộ dữ liệu đầu vào (Graph, TSP, Knapsack...)
│   ├── continuous/                          # Cấu hình test cho các hàm liên tục (Continuous Functions)
│   │   └── continuous_testcases.txt
│   ├── graph/                               # Dữ liệu test thuật toán Tìm kiếm Đồ thị (Độ khó tăng dần)
│   │   ├── tc01_trivial_chain.txt
│   │   ├── tc02_trivial_twopaths.txt
│   │   ├── tc03_easy_tree.txt
│   │   ├── tc04_easy_deadend.txt
│   │   ├── tc05_easy_classic.txt
│   │   ├── tc06_medium_multipaths.txt
│   │   ├── tc07_medium_dense.txt
│   │   ├── tc08_medium_maze.txt
│   │   ├── tc09_medhard_city.txt
│   │   ├── tc10_hard_sparse.txt
│   │   ├── tc11_hard_large.txt
│   │   └── tc12_expert_100nodes.txt
│   ├── knapsack/                            # Dữ liệu test Bài toán Cái túi (Tăng dần theo số lượng đồ vật)
│   │   ├── tc01_trivial_5items.txt
│   │   ├── tc02_easy_10items.txt
│   │   ├── tc03_easymed_15items.txt
│   │   ├── tc04_medium_20items.txt
│   │   ├── tc05_medhard_30items.txt
│   │   ├── tc06_hard_50items.txt
│   │   └── tc07_expert_100items.txt
│   ├── tsp/                                 # Dữ liệu test Bài toán Người chào hàng (Tăng dần theo số thành phố)
│   │   ├── tc01_trivial_5cities.txt
│   │   ├── tc02_easy_clustered.txt
│   │   ├── tc03_easy_10cities.txt
│   │   ├── tc04_medium_15cities.txt
│   │   ├── tc05_medium_20cities.txt
│   │   ├── tc06_medhard_30cities.txt
│   │   ├── tc07_hard_50cities.txt
│   │   └── tc08_expert_100cities.txt
│   └── README.txt                           # Ghi chú hướng dẫn đọc file dữ liệu
│
├── utils/                                   # Chứa các hàm/lớp tiện ích dùng chung (graph.py)
│   ├── __init__.py                          # Biến thư mục utils thành một Python package.
│   └── graph.py                             # Chứa cấu trúc dữ liệu đồ thị, hàm tải dữ liệu và tính toán Heuristic (Euclidean, Manhattan).
│
├── .gitignore                               # Cấu hình bỏ qua file cho Git
├── benchmark_experiments.ipynb              # File Notebook kịch bản thực thi và phân tích chính
└── requirements.txt                         # Danh sách thư viện môi trường cần cài đặt


















