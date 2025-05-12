import numpy as np
import scipy.interpolate

# 示例数据
target = {
    "times": np.array([0, 1, 2, 3, 4]),
    "pos": np.array([0, 1, 2, 3,4])
}

# 创建插值函数
interp_pos = scipy.interpolate.interp1d(
    target["times"], 
    target["pos"], 
    axis=0, 
    kind="quadratic", 
    bounds_error=False, 
    fill_value=(target["pos"][0], target["pos"][-1])
)

# 插值计算
new_times = np.array([0.5, 2.5, 5])
new_pos = interp_pos(new_times)
print(new_pos)