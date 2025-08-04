import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体，确保中文显示正常
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# 解决负号显示问题
matplotlib.rcParams['axes.unicode_minus'] = False

# 卡尔曼滤波参数配置类
# 在参数配置类中添加对应映射
class KalmanFilterParams:
    def __init__(self):
        # 从配置参数映射（这里使用你提供的默认值）
        velocity_noise_coef = 5    # 对应 config.tracker_cfg.velocity_noise_coef
        sigma_phi = 0.1              # 对应 config.tracker_cfg.sigma_phi
        sigma_r = 0.2                # 对应 config.tracker_cfg.sigma_r
        sigma_r_dot = 0.05           # 对应 config.tracker_cfg.sigma_r_dot
        
        # 状态转移矩阵 (距离, 速度, 角度)
        self.A = np.array([[1, 1, 0],
                           [0, 1, 0],
                           [0, 0, 1]], dtype=np.float64)
        
        # 测量矩阵（距离和角度有直接测量）
        self.H = np.array([[1, 0, 0],
                           [0, 0, 1]], dtype=np.float64)
        
        # 过程噪声协方差 Q
        # 对应关系：[距离过程噪声, 速度过程噪声（结合velocity_noise_coef和sigma_r_dot）, 角度过程噪声]
        self.Q = np.diag([
            velocity_noise_coef,  # 距离过程噪声（与速度噪声系数关联）
            sigma_r_dot * velocity_noise_coef,  # 速度过程噪声（核心参数）
            velocity_noise_coef   # 角度过程噪声（与速度噪声系数关联）
        ]).astype(np.float64)
        
        # 测量噪声协方差 R（方差 = 标准差的平方）
        # 对应关系：[距离测量噪声方差(sigma_r²), 角度测量噪声方差(sigma_phi²)]
        self.R = np.diag([
            sigma_r **2,    # 距离测量噪声（sigma_r的平方）
            sigma_phi** 2   # 角度测量噪声（sigma_phi的平方）
        ]).astype(np.float64)
        
        # 初始状态和协方差（保持不变）
        self.x = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 1.0]).astype(np.float64)


# 卡尔曼滤波类
class KalmanFilter:
    def __init__(self, params):
        self.A = params.A
        self.H = params.H
        self.Q = params.Q
        self.R = params.R
        self.x = params.x.copy()  # 确保是float类型
        self.P = params.P.copy()

    def predict(self):
        # 预测步骤
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x.copy()

    def update(self, z):
        # 确保测量值是float64类型
        z = z.astype(np.float64)
        
        # 更新步骤
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x += np.dot(K, y)
        self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, self.H)), self.P)
        return self.x.copy()


# 生成模拟的测量数据（距离、角度）
def generate_measurements(steps):
    np.random.seed(0)
    # 生成真实的距离（累积的随机步长）
    true_distances = np.cumsum(np.random.uniform(0.5, 1.5, steps)).astype(np.float64)
    # 生成真实的速度（距离的一阶差分）
    true_velocities = np.diff(true_distances, prepend=0)
    # 生成真实的角度（累积的随机角度变化）
    true_angles = np.cumsum(np.random.uniform(-0.1, 0.1, steps)).astype(np.float64)
    
    # 添加测量噪声
    measured_distances = true_distances + np.random.normal(0, 1, steps)
    measured_angles = true_angles + np.random.normal(0, 0.1, steps)
    
    return true_distances, true_velocities, true_angles, measured_distances, measured_angles


# 主流程
if __name__ == "__main__":
    # 参数配置
    kf_params = KalmanFilterParams()
    
    # 可在这里调节参数，例如：
    # kf_params.Q = np.diag([0.2, 0.02, 0.02]).astype(np.float64)  # 改变过程噪声
    # kf_params.R = np.diag([2, 0.2]).astype(np.float64)  # 改变测量噪声

    kf = KalmanFilter(kf_params)

    steps = 50  # 时间步数
    true_distances, true_velocities, true_angles, measured_distances, measured_angles = generate_measurements(steps)

    # 存储结果
    prior_estimates = []  # [距离, 速度, 角度]
    posterior_estimates = []
    true_states = []

    for i in range(steps):
        # 预测
        prior_x = kf.predict()
        prior_estimates.append(prior_x.flatten())

        # 构造测量值 [距离, 角度]
        z = np.array([[measured_distances[i]], [measured_angles[i]]])
        # 更新
        posterior_x = kf.update(z)
        posterior_estimates.append(posterior_x.flatten())

        true_states.append([true_distances[i], true_velocities[i], true_angles[i]])

    # 转换为 numpy 数组
    prior_estimates = np.array(prior_estimates)
    posterior_estimates = np.array(posterior_estimates)
    true_states = np.array(true_states)

    # 创建一个包含3个子图的图形
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('卡尔曼滤波行人运动状态估计', fontsize=16)

    # 1. 距离对比图
    ax1.plot(true_states[:, 0], label='实际距离', color='blue')
    ax1.plot(measured_distances, label='测量距离', color='orange', alpha=0.7)
    ax1.plot(prior_estimates[:, 0], label='先验估计距离', linestyle='--', color='gray')
    ax1.plot(posterior_estimates[:, 0], label='后验估计距离', linestyle='-', color='red')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('距离 (米)')
    ax1.set_title('距离估计对比')
    ax1.legend()
    ax1.grid(True)

    # 2. 速度对比图
    ax2.plot(true_states[:, 1], label='实际速度', color='blue')
    ax2.plot(prior_estimates[:, 1], label='先验估计速度', linestyle='--', color='gray')
    ax2.plot(posterior_estimates[:, 1], label='后验估计速度', linestyle='-', color='red')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('速度 (米/步)')
    ax2.set_title('速度估计对比')
    ax2.legend()
    ax2.grid(True)

    # 3. 角度对比图
    ax3.plot(true_states[:, 2], label='实际角度', color='blue')
    ax3.plot(measured_angles, label='测量角度', color='orange', alpha=0.7)
    ax3.plot(prior_estimates[:, 2], label='先验估计角度', linestyle='--', color='gray')
    ax3.plot(posterior_estimates[:, 2], label='后验估计角度', linestyle='-', color='red')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('角度 (弧度)')
    ax3.set_title('角度估计对比')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    