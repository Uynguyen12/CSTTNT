import matplotlib.pyplot as plt

def plot_tsp_route(cities, route, title="TSP Route"):
    """
    cities: array tọa độ [[x1, y1], [x2, y2]...]
    route: danh sách index thứ tự đi [0, 4, 2, 1, 3, 0]
    """
    plt.figure(figsize=(8, 8))
    # Vẽ điểm thành phố
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=50)
    
    # Vẽ đường nối
    path_coords = cities[route]
    plt.plot(path_coords[:, 0], path_coords[:, 1], 'b-')
    
    plt.title(title)
    plt.show()