# Quick Start Guide - README for Examples
Hướng dẫn nhanh để chạy các ví dụ

## Các File Ví Dụ Có Sẵn

### 1. `01_graph_search_examples.py`
**Thuật toán tìm kiếm trên đồ thị (Classical Search)**

Chạy:
```bash
python examples/01_graph_search_examples.py
```

Giới thiệu:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Uniform Cost Search (UCS)
- A* Search
- Greedy Best-First Search

Đặc điểm:
- BFS: Complete, Optimal, nhưng tốn memory
- DFS: Memory efficient nhưng không tối ưu
- UCS: Tối ưu với chi phí nhưng chậm
- A*: Kết hợp tốt UCS + heuristic, tối ưu
- Greedy: Nhanh nhưng không tối ưu

---

### 2. `02_evolution_based_examples.py`
**Thuật toán dựa trên Evolution (Nature-Inspired)**

Chạy:
```bash
python examples/02_evolution_based_examples.py
```

Thuật toán:
- Genetic Algorithm (GA) - Inspired by Darwinian Evolution
- Differential Evolution (DE) - Direct search method

Test Functions:
- Sphere Function (đơn giản)
- Rastrigin Function (multimodal)
- Rosenbrock Function (valley-shaped)
- Ackley Function (nhiều local minima)

Khi nào dùng:
- GA: Combinatorial optimization, Feature selection, Scheduling
- DE: Continuous optimization, Nếu không có gradient

---

### 3. `03_swarm_based_examples.py`
**Thuật toán dựa trên Swarm (Nature-Inspired)**

Chạy:
```bash
python examples/03_swarm_based_examples.py
```

Thuật toán:
- Particle Swarm Optimization (PSO) - Inspired by bird flocking
- Ant Colony Optimization (ACO) - Inspired by pheromone trails

Ứng dụng:
- PSO: Continuous optimization, multi-dimensional problems
- ACO: Traveling Salesman Problem (TSP), Combinatorial problems

Ưu điểm:
- PSO: Nhanh hội tụ, intuitive parameters
- ACO: Tốt cho discrete problems, natural inspiration

---

### 4. `04_physics_human_based_examples.py`
**Thuật toán Physics-Based & Human-Based (Nature-Inspired)**

Chạy:
```bash
python examples/04_physics_human_based_examples.py
```

Thuật toán:
- Simulated Annealing (SA) - Physics: cooling process
- Hill Climbing (HC) - Greedy local search
- Harmony Search (HS) - Human: musicians creating harmony

Đặc điểm:
- SA: Có thể escape local minima, tốn nhiều iterations
- HC: Rất nhanh nhưng hay bị stuck
- HS: Cân bằng exploration/exploitation

---

### 5. `05_complete_benchmark.py`
**So sánh tất cả thuật toán (COMPREHENSIVE BENCHMARK)**

Chạy:
```bash
python examples/05_complete_benchmark.py
```

Chạy:
- Tất cả thuật toán continuous optimization trên 4 test functions
- Graph search algorithms
- Thống kê so sánh

Output:
- Fitness values cho mỗi function
- Execution time
- Số iterations
- Xác định algorithm tốt nhất cho từng problem

---

## Cách Chạy

### Option 1: Chạy từ trong Visual Studio Code
```
Ctrl+Shift+P → Python: Run Python File in Terminal
```

### Option 2: Chạy từ Terminal
```bash
cd d:\Code\CSTTNT\Team_Project_01\CSTTNT
python examples/01_graph_search_examples.py
python examples/02_evolution_based_examples.py
python examples/03_swarm_based_examples.py
python examples/04_physics_human_based_examples.py
python examples/05_complete_benchmark.py
```

### Option 3: Chạy tất cả
```bash
for /r examples %%f in (*.py) do python "%%f"
```

---

## Cấu Trúc Dữ Liệu Kết Quả

### Cho Graph Search (Classical)
```python
result.path_found      # Boolean
result.path            # List[int] - các nodes trong path
result.cost            # float - tổng chi phí
result.nodes_explored  # int - số nodes explored
result.execution_time  # float - thời gian (seconds)
```

### Cho Optimization (Nature-Inspired)
```python
result.fitness                # float - best value found
result.solution               # array - best solution
result.iterations             # int - iterations used
result.evaluations            # int - function evaluations
result.execution_time         # float - thời gian (seconds)
result.convergence_curve      # List[float] - history
result.success                # boolean - found solution?
```

---

## Lựa Chọn Algorithm

### Nếu bạn có:
1. **Graph/Network Problem**
   - Không có heuristic → BFS, DFS, UCS
   - Có heuristic tốt → A*, Greedy

2. **Continuous Optimization Problem**
   - Unimodal → Hill Climbing, DE
   - Multimodal → PSO, GA, SA, HS
   - Cần tối ưu → A*, DE, GA

3. **Combinatorial Problem (TSP, Scheduling)**
   - GA, ACO, PSO

4. **Không biết loại gì**
   - Thử `05_complete_benchmark.py` để xem cái nào tốt nhất

---

## Parameters Quan Trọng

### GA (Genetic Algorithm)
- `population_size`: 30-100 (lớn hơn = explore hơn)
- `mutation_rate`: 0.01-0.1
- `crossover_rate`: 0.6-0.9

### DE (Differential Evolution)
- `population_size`: 30-100
- `mutation_factor`: 0.5-1.0
- `crossover_probability`: 0.7-0.9

### PSO (Particle Swarm)
- `num_particles`: 20-50
- `inertia_weight`: 0.4-0.9
- `cognitive_coeff`: 1-2
- `social_coeff`: 1-2

### SA (Simulated Annealing)
- `initial_temperature`: Cao (100-1000)
- `cooling_rate`: 0.9-0.99
- `temperature_min`: Thấp (0.001-0.1)

### HS (Harmony Search)
- `harmony_memory_size`: 10-50
- `harmony_memory_accept_rate`: 0.7-0.95
- `pitch_adjustment_rate`: 0.1-0.5

---

## Performance Tips

1. **Mở rộng iterations** nếu không hội tụ
2. **Điều chỉnh population/particle size** (lớn hơn = tìm kiếm tốt hơn)
3. **Dùng multiple runs** để tìm average performance
4. **Normalize input** nếu ranges khác nhau lớn
5. **Seed cho reproducibility** (khi so sánh)

---

## Các Hàm Test Phổ Biến

| Function | Tính chất | Độ khó | Optimal |
|----------|----------|--------|---------|
| Sphere | Unimodal, smooth | Dễ | 0 |
| Rastrigin | Multimodal | Khó | 0 |
| Ackley | Multimodal | Trung bình | 0 |
| Schwefel | Deceptive | Rất khó | -418.98*n |
| Rosenbrock | Valley-shaped | Trung bình | 0 |

---

## Ghi Chú

- Tất cả thuật toán đều có `verbose` option để xem chi tiết
- Results có `convergence_curve` - lịch sử best fitness
- Có thể lấy `metadata` để lưu thêm thông tin
- Dùng `seed` para reproducibility

---

## Troubleshooting

**ImportError**: Đảm bảo chạy từ đúng directory
```bash
cd d:\Code\CSTTNT\Team_Project_01\CSTTNT
```

**Module not found**: Cài đặt requirements
```bash
pip install -r requirements.txt
```

**Quá chậm**: Giảm iterations hoặc population size

**Kết quả xấu**: Điều chỉnh parameters, tăng iterations

---

Happy Optimizing! 🚀
