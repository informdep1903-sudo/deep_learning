import numpy as np
import matplotlib.pyplot as plt

# Генерация синтетического ряда
t = np.linspace(0, 365 * 2, 730)  # 2 года данных
trend = 0.01 * t
seasonal = 10 * np.sin(2 * np.pi * t / 30)  # 30-дневный цикл
noise = np.random.normal(0, 1, size=t.shape)
data = trend + seasonal + noise

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(t, data)
plt.title("Синтетический временной ряд: тренд + сезонность + шум")
plt.show()
