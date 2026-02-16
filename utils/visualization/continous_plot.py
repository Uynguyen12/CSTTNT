import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Bắt buộc để vẽ 3D

def plot_convergence(histories, labels, title="Convergence Speed", filename=None):
    """
    So sánh tốc độ hội tụ (Metric của Sphere, Rastrigin...)
    histories: list các mảng fitness history của từng thuật toán
    labels: list tên thuật toán (VD: ['PSO', 'DE', 'GA'])
    filename: Tên file để lưu ảnh (VD: 'convergence.png')
    """
    plt.figure(figsize=(10, 6))
    
    # Dùng cycle màu để các đường khác biệt rõ ràng
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for hist, label, color in zip(histories, labels, colors):
        plt.plot(hist, label=label, color=color, linewidth=2, alpha=0.8)
        
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Fitness (Log Scale)", fontsize=12)
    plt.yscale("log") # Dùng log scale để nhìn rõ sự hội tụ về 0
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ hội tụ tại: {filename}")
    
    plt.show()

def plot_landscape_3d(objective_func, bounds, title="Function Landscape", resolution=100, filename=None):
    """
    Vẽ bề mặt 3D và đường đồng mức cho hàm mục tiêu.
    
    Args:
        objective_func: Hàm cần vẽ (nhận vào array 1D [x, y], trả về float)
        bounds: List tuple giới hạn [(min_x, max_x), (min_y, max_y)]
        title: Tiêu đề biểu đồ
        resolution: Độ phân giải lưới (mặc định 100x100 điểm)
        filename: Tên file để lưu ảnh (nếu cần xuất báo cáo)
    """
    # 1. Tạo lưới dữ liệu
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 2. Tính giá trị Z (Fitness) tại từng điểm lưới
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            # Gọi hàm mục tiêu với vector [x, y]
            Z[i, j] = objective_func(np.array([X[i, j], Y[i, j]]))

    # 3. Khởi tạo đồ thị 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 4. Vẽ bề mặt (Surface)
    # cmap='viridis': Màu xanh-vàng rất chuyên nghiệp cho báo cáo khoa học
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                           edgecolor='none', alpha=0.85, 
                           rstride=1, cstride=1, antialiased=True)
    
    # 5. Vẽ đường đồng mức (Contour) ở đáy để thấy rõ local optima
    z_min = np.min(Z)
    z_max = np.max(Z)
    offset = z_min - (z_max - z_min) * 0.1 # Đẩy contour xuống dưới thấp hơn giá trị nhỏ nhất
    
    ax.contour(X, Y, Z, zdir='z', offset=offset, cmap='viridis', alpha=0.5)

    # 6. Trang trí
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('X Axis', fontsize=11)
    ax.set_ylabel('Y Axis', fontsize=11)
    ax.set_zlabel('Fitness Value', fontsize=11)
    ax.set_zlim(offset, z_max) # Giới hạn trục Z để thấy contour đáy

    # Thêm thanh màu (Colorbar)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Fitness Value', rotation=270, labelpad=15)

    # Chỉnh góc nhìn đẹp nhất
    ax.view_init(elev=35, azim=45)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ 3D tại: {filename}")
        
    plt.show()