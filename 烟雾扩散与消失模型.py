import numpy as np
import matplotlib.pyplot as plt

# 定义常数
k = 1.0  # 扩散系数
alpha = 1.0  # 参数alpha
Q = 1.0  # 炮弹释放的烟雾总量
mu = 0.01  # 仪器灵敏度
pi = np.pi

# 定义不透光区域边界半径 r(t)
def r(t):
    return np.sqrt(4 * k * t * np.log(alpha * Q / (4 * pi * k * mu * t)))

# 计算最大边界半径和相关时间点
t1 = alpha * Q / (4 * pi * k * mu * np.e)
r_max = np.sqrt(Q / (pi * mu * np.e))
t2 = alpha * Q / (4 * pi * k * mu)

# 时间范围
t = np.linspace(0.1, t2 * 1.5, 500)  # 从小于 t2 的时间开始

# 绘制不透光区域边界随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(t, r(t), label=r'$r(t) = \sqrt{4 kt \ln \frac{\alpha Q}{4 \pi k \mu t}}$')
plt.axvline(t1, color='r', linestyle='--', label=r'$t_1 = \frac{\alpha Q}{4 \pi k \mu e}$')
plt.axhline(r_max, color='g', linestyle='--', label=r'$r_m = \sqrt{\frac{Q}{\pi \mu e}}$')
plt.axvline(t2, color='b', linestyle='--', label=r'$t_2 = \frac{\alpha Q}{4 \pi k \mu}$')
plt.xlabel('Time t')
plt.ylabel('Boundary Radius r(t)')
plt.title('Smoke Diffusion and Dissipation Model')
plt.legend()
plt.grid(True)
plt.show()

# 输出最大边界半径和相关时间点
print(f"最大不透光区域边界半径 r_max = {r_max:.4f}")
print(f"边界达到最大值的时间 t1 = {t1:.4f}")
print(f"烟雾完全消失的时间 t2 = {t2:.4f}")
