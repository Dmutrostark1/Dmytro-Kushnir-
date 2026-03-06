import csv
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. Зчитування даних з CSV [cite: 171-181]
# ==========================================
def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['RPS']))
            y.append(float(row['CPU']))
    return np.array(x), np.array(y)

# ==========================================
# 2. Метод Ньютона (Розділені різниці) [cite: 6-12, 54-55]
# ==========================================
def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef[0, :] # Повертаємо розділені різниці f(x0, ... xk)

def newton_polynomial(x_data, y_data, x):
    coef = divided_differences(x_data, y_data)
    n = len(x_data)
    
    # N_n(x) = f_0 + sum( f(x0..xk) * w_k(x) ) [cite: 158, 159]
    result = coef[0]
    for k in range(1, n):
        term = coef[k]
        for i in range(k):
            term *= (x - x_data[i])
        result += term
    return result

# ==========================================
# 3. Факторіальні многочлени [cite: 79, 131-150]
# ==========================================
# Увага: Факторіальні многочлени вимагають рівновіддалених вузлів.
# Оскільки наші дані RPS (50, 100, 200, 400, 800) не є рівномірними, 
# ми згенеруємо рівномірну сітку за допомогою полінома Ньютона для демонстрації методу.
def finite_differences(y):
    n = len(y)
    diffs = np.zeros([n, n])
    diffs[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            diffs[i][j] = diffs[i+1][j-1] - diffs[i][j-1] # Delta f(t) = f(t+1) - f(t) [cite: 85]
    return diffs[0, :]

def factorial_polynomial(y_uniform, x_uniform, x):
    n = len(y_uniform)
    h = x_uniform[1] - x_uniform[0]
    t = (x - x_uniform[0]) / h # t = (x - x_0) / h [cite: 81]
    
    diffs = finite_differences(y_uniform)
    result = diffs[0]
    
    for k in range(1, n):
        term = diffs[k] / math.factorial(k)
        # t^(k) = t*(t-1)*...*(t-k+1) [cite: 134]
        t_fact = 1
        for i in range(k):
            t_fact *= (t - i)
        result += term * t_fact
        
    return result

# ==========================================
# 4. Основна логіка та побудова графіків
# ==========================================
def main():
    # Зчитування
    x_data, y_data = read_data('data.csv')
    
    # Прогноз для 600 RPS
    target_x = 600
    pred_newton = newton_polynomial(x_data, y_data, target_x)
    
    # Створення рівномірної сітки для факторіального полінома
    x_uniform = np.linspace(x_data[0], x_data[-1], len(x_data))
    y_uniform = [newton_polynomial(x_data, y_data, xi) for xi in x_uniform]
    pred_factorial = factorial_polynomial(y_uniform, x_uniform, target_x)
    
    print(f"--- Прогноз для RPS = {target_x} ---")
    print(f"Метод Ньютона: CPU = {pred_newton:.2f}%")
    print(f"Факторіальні многочлени: CPU = {pred_factorial:.2f}%")
    
    # Побудова графіка інтерполяції
    x_plot = np.linspace(min(x_data), max(x_data), 200)
    y_plot_newton = [newton_polynomial(x_data, y_data, xi) for xi in x_plot]
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, y_plot_newton, label="Інтерполяція Ньютона", color='blue')
    plt.scatter(x_data, y_data, color='red', s=50, zorder=5, label="Експериментальні дані (Вузли)")
    plt.scatter([target_x], [pred_newton], color='green', s=100, marker='*', zorder=6, label=f"Прогноз (600 RPS)")
    
    plt.title("Залежність навантаження CPU від RPS")
    plt.xlabel("RPS (Запити за секунду)")
    plt.ylabel("CPU (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ==========================================
    # 5. Дослідження: Ефект Рунге та різна кількість вузлів [cite: 236, 303]
    # ==========================================
    # Для демонстрації ефекту використаємо функцію Рунге f(x) = 1 / (1 + 25x^2) на відрізку [-1, 1]
    def runge_func(x):
        return 1 / (1 + 25 * x**2)
    
    plt.figure(figsize=(10, 5))
    x_runge_plot = np.linspace(-1, 1, 500)
    plt.plot(x_runge_plot, runge_func(x_runge_plot), label="Справжня функція Рунге", color='black', linewidth=2)
    
    nodes_list = [5, 10, 20] # Дослідження для 5, 10 та 20 вузлів [cite: 236]
    for n in nodes_list:
        x_nodes = np.linspace(-1, 1, n)
        y_nodes = runge_func(x_nodes)
        
        y_interp = [newton_polynomial(x_nodes, y_nodes, xi) for xi in x_runge_plot]
        plt.plot(x_runge_plot, y_interp, label=f"Інтерполяція (n={n})", linestyle='--')
        
    plt.title("Дослідження похибки: Ефект Рунге")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-0.5, 1.5)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
