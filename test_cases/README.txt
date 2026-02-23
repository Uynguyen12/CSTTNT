# TEST CASES — CSC14003 Search & Nature-Inspired Algorithms
# Đặt folder `test_cases/` vào thư mục gốc CSTTNT/ cùng với notebook

==============================================================
  CẤU TRÚC THƯ MỤC
==============================================================

test_cases/
├── graph/          ← Shortest Path (BFS, DFS, UCS, Greedy, A*, HC, SA)
├── tsp/            ← Traveling Salesman (GA, DE, PSO, ACO, ABC, FA, CS, SA)
├── knapsack/       ← 0/1 Knapsack (GA, ABC, PSO, DE, FA, CS, TLBO)
└── continuous/     ← Continuous Optimization (tất cả nature-inspired)

==============================================================
  GRAPH TEST CASES (12 files, từ TRIVIAL → EXPERT)
==============================================================

File                        | Nodes | Edges | Level        | Mục đích
----------------------------|-------|-------|--------------|----------------------------------
tc01_trivial_chain.txt      |   3   |   2   | TRIVIAL      | Linear, warmup
tc02_trivial_twopaths.txt   |   4   |   4   | TRIVIAL      | BFS vs UCS tiebreak
tc03_easy_tree.txt          |   5   |   6   | EASY         | Tree structure, no cycles
tc04_easy_deadend.txt       |   6   |   7   | EASY         | Dead-end, DFS behavior
tc05_easy_classic.txt       |   8   |  11   | EASY         | Sách giáo khoa chuẩn
tc06_medium_multipaths.txt  |  10   |  14   | MEDIUM       | Multiple optimal paths
tc07_medium_dense.txt       |  12   |  19   | MEDIUM       | Dense, heuristic misleading
tc08_medium_maze.txt        |  15   |  21   | MEDIUM       | Maze-like, obstacle
tc09_medhard_city.txt       |  20   |  44   | MEDIUM-HARD  | City road network
tc10_hard_sparse.txt        |  30   |  53   | HARD         | Sparse, nhiều dead ends
tc11_hard_large.txt         |  50   |  80   | HARD         | Large scale
tc12_expert_100nodes.txt    | 100   | 100+  | EXPERT       | Stress test, BFS vs A*

FORMAT:
  N                    ← số nodes
  x y                  ← tọa độ mỗi node (N dòng)
  u v weight           ← mỗi cạnh (undirected)
  START GOAL           ← dòng cuối

==============================================================
  TSP TEST CASES (8 files, từ TRIVIAL → EXPERT)
==============================================================

File                        | Cities | Level        | Đặc điểm
----------------------------|--------|--------------|----------------------------------
tc01_trivial_5cities.txt    |   5    | TRIVIAL      | Square + center, tour rõ ràng
tc02_easy_clustered.txt     |   8    | EASY         | 2 clusters, inter-cluster gap lớn
tc03_easy_10cities.txt      |  10    | EASY         | Random, benchmark chuẩn
tc04_medium_15cities.txt    |  15    | MEDIUM       | Diagonal structure ẩn
tc05_medium_20cities.txt    |  20    | MEDIUM       | Random, phân biệt algo tốt/xấu
tc06_medhard_30cities.txt   |  30    | MEDIUM-HARD  | Large area, ACO bắt đầu shine
tc07_hard_50cities.txt      |  50    | HARD         | Stress test convergence
tc08_expert_100cities.txt   | 100    | EXPERT       | Cần pop>=100, iter>=1000

FORMAT:
  N                    ← số thành phố
  id x y               ← tọa độ mỗi thành phố (N dòng)

==============================================================
  KNAPSACK TEST CASES (7 files, từ TRIVIAL → EXPERT)
==============================================================

File                        | Items | Capacity | Level        | Đặc điểm
----------------------------|-------|----------|--------------|---------------------------
tc01_trivial_5items.txt     |   5   |    10    | TRIVIAL      | Dễ tính tay
tc02_easy_10items.txt       |  10   |    15    | EASY         | Benchmark chuẩn
tc03_easymed_15items.txt    |  15   |    25    | EASY-MEDIUM  | Ratio cao/thấp xen kẽ
tc04_medium_20items.txt     |  20   |    40    | MEDIUM       | High-value/heavy groups
tc05_medhard_30items.txt    |  30   |    60    | MEDIUM-HARD  | Weight gần nhau, value khác
tc06_hard_50items.txt       |  50   |   150    | HARD         | Redundant items
tc07_expert_100items.txt    | 100   |   300    | EXPERT       | Maximum stress test

FORMAT:
  N capacity           ← số items và sức chứa
  value weight         ← mỗi item (N dòng)

==============================================================
  CONTINUOUS TEST CASES (24 configs trong 1 file)
==============================================================

File: continuous/continuous_testcases.txt

TC01–TC02   | D=2   | TRIVIAL      | Sphere, Ackley
TC03–TC05   | D=5   | EASY         | Sphere, Rosenbrock, Griewank
TC06–TC10   | D=10  | MEDIUM       | Tất cả 5 hàm
TC11–TC14   | D=20  | MEDIUM-HARD  | Sphere, Rastrigin, Ackley, Rosenbrock
TC15–TC19   | D=30  | HARD         | Tất cả 5 hàm
TC20–TC22   | D=50  | EXPERT       | Sphere, Rastrigin, Ackley
TC23–TC24   | D=100 | EXPERT       | Sphere, Rastrigin (ultimate scale)

FORMAT mỗi dòng:
  TC_ID  function  dim  lb  ub  optimum  description

ALGORITHMS phù hợp cho continuous:
  GA, DE, PSO, ACO, ABC, FA, CS, GSA, HS, SA, HC, TLBO

==============================================================
  ALGORITHM ↔ TEST CASE MAPPING
==============================================================

Algorithm | Graph | TSP | Knapsack | Continuous
----------|-------|-----|----------|----------
BFS       |  ✓   |  -  |    -     |    -
DFS       |  ✓   |  -  |    -     |    -
UCS       |  ✓   |  -  |    -     |    -
Greedy    |  ✓   |  -  |    -     |    -
A*        |  ✓   |  -  |    -     |    -
HC        |  ✓   |  -  |    -     |    ✓
SA        |  ✓   |  ✓  |    -     |    ✓
GA        |  -   |  ✓  |    ✓     |    ✓
DE        |  -   |  ✓  |    ✓     |    ✓
PSO       |  -   |  ✓  |    ✓     |    ✓
ACO       |  -   |  ✓  |    -     |    ✓
ABC       |  -   |  -  |    ✓     |    ✓
FA        |  -   |  -  |    ✓     |    ✓
CS        |  -   |  ✓  |    ✓     |    ✓
GSA       |  -   |  -  |    -     |    ✓
HS        |  -   |  -  |    -     |    ✓
TLBO      |  -   |  -  |    ✓     |    ✓

==============================================================
  CÁCH ĐỌC FILE TRONG NOTEBOOK
==============================================================

# Graph:
from utils.graph import Graph
g, start, goal = build_graph_from_file('test_cases/graph/tc05_easy_classic.txt')

# TSP:
cities = load_tsp_file('test_cases/tsp/tc05_medium_20cities.txt')

# Knapsack:
weights, values, capacity = load_knapsack_file('test_cases/knapsack/tc04_medium_20items.txt')

# Continuous:
# Đọc config file và chạy theo từng dòng trong notebook
