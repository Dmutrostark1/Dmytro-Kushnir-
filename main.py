import requests
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 4. Обчислення кумулятивної відстані
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ==========================================
# 7. Метод прогонки (TDMA) для тридіагональної матриці
# ==========================================
def tdma(a, b, c, d):
    n = len(d)
    P = np.zeros(n)
    Q = np.zeros(n)
    x = np.zeros(n)
    
    # Прямий хід
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n):
        denominator = b[i] + a[i] * P[i-1]
        if i < n - 1:
            P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i] * Q[i-1]) / denominator
        
    # Зворотний хід
    x[-1] = Q[-1]
    for i in range(n-2, -1, -1):
        x[i] = P[i] * x[i+1] + Q[i]
    return x

# ==========================================
# 6-9. Обчислення коефіцієнтів кубічного сплайна
# ==========================================
def get_spline_coeffs(x, y, print_details=False):
    n = len(x) - 1
    h = np.diff(x)
    
    # Формування системи для c_i
    A_diag = np.zeros(n - 1)
    A_lower = np.zeros(n - 1)
    A_upper = np.zeros(n - 1)
    B = np.zeros(n - 1)
    
    for i in range(1, n):
        A_diag[i-1] = 2 * (h[i-1] + h[i])
        if i > 1:
            A_lower[i-1] = h[i-1]
        if i < n - 1:
            A_upper[i-1] = h[i]
        B[i-1] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        
    if print_details:
        print("\n--- 6-7. Система лінійних алгебраїчних рівнянь (Трьохдіагональна матриця) ---")
        print("Нижня діагональ (alpha):", np.round(A_lower, 4))
        print("Головна діагональ (beta):", np.round(A_diag, 4))
        print("Верхня діагональ (gamma):", np.round(A_upper, 4))
        print("Вектор вільних членів (delta):", np.round(B, 4))

    # 8. Знаходження коефіцієнтів c
    c = np.zeros(n + 1)
    c[1:n] = tdma(A_lower, A_diag, A_upper, B)
    
    # 9. Знаходження коефіцієнтів a, b, d
    a = y[:-1]
    b = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
        
    if print_details:
        print("\n--- 8-9. Коефіцієнти кубічних сплайнів ---")
        for i in range(n):
            print(f"Сплайн {i+1}: a={a[i]:.4f}, b={b[i]:.4f}, c={c[i]:.4f}, d={d[i]:.4f}")
            
    return a, b, c[:-1], d

def evaluate_spline(x_val, x_nodes, a, b, c, d):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i+1]:
            dx = x_val - x_nodes[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return a[-1]

def main():
    # ==========================================
    # 1. Запит до Open-Elevation API (з резервним варіантом)
    # ==========================================
    print("Виконуємо запит до API...")
    url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        results = data["results"]
    except Exception as e:
        print("\n[!] Сервер open-elevation не відповідає. Використовуємо резервні дані маршруту...")
        results = [
            {"latitude": 48.164214, "longitude": 24.536044, "elevation": 1330.0},
            {"latitude": 48.164983, "longitude": 24.534836, "elevation": 1370.0},
            {"latitude": 48.165605, "longitude": 24.534068, "elevation": 1410.0},
            {"latitude": 48.166228, "longitude": 24.532915, "elevation": 1450.0},
            {"latitude": 48.166777, "longitude": 24.531927, "elevation": 1490.0},
            {"latitude": 48.167326, "longitude": 24.530884, "elevation": 1530.0},
            {"latitude": 48.167011, "longitude": 24.530061, "elevation": 1580.0},
            {"latitude": 48.166053, "longitude": 24.528039, "elevation": 1640.0},
            {"latitude": 48.166655, "longitude": 24.526064, "elevation": 1690.0},
            {"latitude": 48.166497, "longitude": 24.523574, "elevation": 1730.0},
            {"latitude": 48.166128, "longitude": 24.520214, "elevation": 1780.0},
            {"latitude": 48.165416, "longitude": 24.517170, "elevation": 1820.0},
            {"latitude": 48.164546, "longitude": 24.514640, "elevation": 1860.0},
            {"latitude": 48.163412, "longitude": 24.512980, "elevation": 1900.0},
            {"latitude": 48.162331, "longitude": 24.511715, "elevation": 1930.0},
            {"latitude": 48.162015, "longitude": 24.509462, "elevation": 1960.0},
            {"latitude": 48.162147, "longitude": 24.506932, "elevation": 1990.0},
            {"latitude": 48.161751, "longitude": 24.504244, "elevation": 2010.0},
            {"latitude": 48.161197, "longitude": 24.501793, "elevation": 2030.0},
            {"latitude": 48.160580, "longitude": 24.500537, "elevation": 2050.0},
            {"latitude": 48.160250, "longitude": 24.500106, "elevation": 2061.0}
        ]
        
    n_points = len(results)

    # ==========================================
    # 2-3. Табуляція вузлів та запис у текстовий файл
    # ==========================================
    print(f"Кількість вузлів: {n_points}")
    print("\nТабуляція вузлів:")
    
    # Створюємо текстовий файл і записуємо туди дані
    with open("tabulation_results.txt", "w", encoding="utf-8") as file:
        header = "№  | Latitude  | Longitude | Elevation (m)\n"
        print(header.strip())
        file.write("Результати табуляції маршруту Заросляк - Говерла\n")
        file.write(header)
        for i, point in enumerate(results):
            line = f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}"
            print(line)
            file.write(line + "\n")
    print("-> Дані табуляції успішно збережено у файл 'tabulation_results.txt'")

    # ==========================================
    # 4. Обчислення кумулятивної відстані
    # ==========================================
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = [p["elevation"] for p in results]
    
    distances = [0]
    for i in range(1, n_points):
        d = haversine(*coords[i-1], *coords[i])
        distances.append(distances[-1] + d)

    distances = np.array(distances)
    elevations = np.array(elevations)

    # ==========================================
    # 6-9. Знаходження коефіцієнтів та вивід у консоль
    # ==========================================
    # Викликаємо функцію з прапорцем print_details=True, щоб вивести матрицю і коефіцієнти для повного набору точок
    a, b, c, d = get_spline_coeffs(distances, elevations, print_details=True)

    # ==========================================
    # 10-12. Графіки та аналіз точності
    # ==========================================
    node_counts = [10, 15, 20]
    fig, axes = plt.subplots(3, 2, figsize=(16, 12)) # 3 рядки, 2 колонки (сплайн та похибка)
    
    x_smooth = np.linspace(distances[0], distances[-1], 200)
    # Створюємо еталонний сплайн по всіх точках для розрахунку похибки
    y_true = np.array([evaluate_spline(x, distances, a, b, c, d) for x in x_smooth])
    
    for idx, n_nodes in enumerate(node_counts):
        indices = np.linspace(0, n_points - 1, n_nodes, dtype=int)
        x_nodes = distances[indices]
        y_nodes = elevations[indices]
        
        a_n, b_n, c_n, d_n = get_spline_coeffs(x_nodes, y_nodes, print_details=False)
        y_spline = np.array([evaluate_spline(x, x_nodes, a_n, b_n, c_n, d_n) for x in x_smooth])
        
        # Обчислення похибки (12 пункт)
        error = np.abs(y_true - y_spline)
        
        # Графік сплайну
        ax1 = axes[idx, 0]
        ax1.plot(distances, elevations, 'o-', label="Всі дані", color="black", alpha=0.3)
        ax1.plot(x_smooth, y_spline, '-', label=f"Сплайн", color="red")
        ax1.plot(x_nodes, y_nodes, 'ro', label="Вузли")
        ax1.set_title(f"Апроксимація ({n_nodes} вузлів)")
        ax1.set_xlabel("Відстань (м)")
        ax1.set_ylabel("Висота (м)")
        ax1.legend()
        ax1.grid(True)
        
        # Графік похибки
        ax2 = axes[idx, 1]
        ax2.plot(x_smooth, error, '-', color="purple")
        ax2.set_title(f"Похибка $\epsilon$ ({n_nodes} вузлів)")
        ax2.set_xlabel("Відстань (м)")
        ax2.set_ylabel("Похибка (м)")
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig("spline_approximation.png")
    print("\n-> Графіки збережено як 'spline_approximation.png'")

    # ==========================================
    # Додаткове завдання
    # ==========================================
    print("\n--- ХАРАКТЕРИСТИКИ МАРШРУТУ ---")
    print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")
    
    total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n_points))
    print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
    
    total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n_points))
    print(f"Сумарний спуск (м): {total_descent:.2f}")

    grad_full = np.gradient(elevations, distances) * 100
    print(f"\nМаксимальний підйом (%): {np.max(grad_full):.2f}")
    print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
    print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

    mass = 80
    g = 9.81
    energy = mass * g * total_ascent
    print(f"\nМеханічна робота (Дж): {energy:.2f}")
    print(f"Механічна робота (кДж): {energy/1000:.2f}")
    print(f"Енергія (ккал): {energy / 4184:.2f}")

if __name__ == "__main__":
    main()