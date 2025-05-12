import numpy as np
import scipy.io

# 模拟生成目标的时间戳和位置数据
timestamps = np.array([1, 2, 3, 4, 5])  # 假设时间戳
positions = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]])  # 假设位置数据，二维数组

# 将数据保存为.mat文件
data_to_save = {
    "timestamps": timestamps,
    "positions": positions
}
scipy.io.savemat("./data/simulated_mouse_trajectory.mat", data_to_save)

# 模拟读取.mat文件并处理数据
targetsInfo = []
trajTemp = scipy.io.loadmat("./data/simulated_mouse_trajectory.mat")
targetsInfo.append(dict(rsc=1, times=trajTemp["timestamps"].ravel() * 10, pos=trajTemp["positions"] * 10))

# 打印处理后的数据
print("targetsInfo:")
for target in targetsInfo:
    print("rsc:", target["rsc"])
    print("times:", target["times"])
    print("pos:", target["pos"])