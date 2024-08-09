import numpy as np

# 已知条件
T0 = 60  # 初始温度
T1 = 50  # 3分钟后的温度
T_room = 18  # 环境温度
time1 = 3  # 时间为3分钟

# 求解冷却常数k
k = -np.log((T1 - T_room) / (T0 - T_room)) / time1

# 求解降到30度所需的时间
T_target = 30
time_to_target = -np.log((T_target - T_room) / (T0 - T_room)) / k

# 计算10分钟后的温度
time2 = 10
T_after_10_minutes = T_room + (T0 - T_room) * np.exp(-k * time2)

# 输出结果
print(f"物体温度降到30°C所需的时间为: {time_to_target:.2f} 分钟")
print(f"10分钟后物体的温度为: {T_after_10_minutes:.2f} °C")
