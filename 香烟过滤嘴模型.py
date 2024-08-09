import numpy as np
import matplotlib.pyplot as plt

# 参数设置
M = 1.0  # 毒物总量 (任意单位)
a = 0.5  # 毒物进入空气和穿行的比例 (a = 0.5, a' = 0.5)
b = 0.1  # 烟草的吸收系数
beta = 0.8  # 过滤嘴的吸收系数, 增大以突出非线性效果
v = 1.0  # 烟雾穿行速度
l1 = 1.0  # 烟草长度
l2_values = np.linspace(0, 10, 100)  # 过滤嘴长度的变化范围, 增大以观察非线性效果

# 计算 Q1 和 Q2
def calculate_Q(l1, l2, a, M, b, beta, v):
    a_prime = 1 - a
    r = a_prime * b * l1 / v
    Q1 = (a * M * v / (a_prime * b)) * np.exp(-beta * l2 / v) * (1 - np.exp(-r))
    Q2 = (a * M * v / (a_prime * b)) * (1 - np.exp(-r))
    return Q1, Q2

Q1_values = []
Q2_values = []
for l2 in l2_values:
    Q1, Q2 = calculate_Q(l1, l2, a, M, b, beta, v)
    Q1_values.append(Q1)
    Q2_values.append(Q2)

# 比较有过滤嘴和无过滤嘴的毒物量
Q1_values = np.array(Q1_values)
Q2_values = np.array(Q2_values)
Q_ratio = Q1_values / Q2_values

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(l2_values, Q1_values, label='Q1 (With Filter)')
plt.plot(l2_values, Q2_values, label='Q2 (Without Filter)')
plt.plot(l2_values, Q_ratio, label='Q1/Q2 Ratio', linestyle='--')
plt.xlabel('Filter Length l2')
plt.ylabel('Toxin Amount Q')
plt.title('Effect of Filter Length and Material on Toxin Inhalation')
plt.legend()
plt.grid(True)
plt.show()
