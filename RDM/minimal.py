# %%
import numpy as np
import matplotlib

import matplotlib.pyplot as plt


# 设置全局字体为支持中文的字体，比如 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 兼容中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ==========================================================
# --- 1. 参数定义 ---
# ==========================================================
c = 3e8
f_start = 24e9
bandwidth = 250e6
T_chirp = 40e-6
num_samples = 256
num_chirps = 64
f_c = f_start 
lambda_c = c / f_c
sweep_slope = bandwidth / T_chirp
range_res = c / (2 * bandwidth)
# max_range_m=(c/2*bandwidth)*(num_samples//2)
# %%
# ==========================================================
# --- 2. 数据生成 ---
# ==========================================================
targets_truth = [[50, -10, 3], [20, 15, 8]]
t_grid, chirp_idx_grid = np.meshgrid(
    np.linspace(0, T_chirp, num_samples),
    np.arange(num_chirps)
)
raw_data = np.zeros((num_chirps, num_samples), dtype=np.complex64)

print("正在生成多目标、无噪声数据...")
for i, (r, v, snr) in enumerate(targets_truth):
    f_beat = sweep_slope * (2 * r / c)
    f_doppler = 2 * v / lambda_c
    print(f"[目标{i+1}] r={r} m, v={v} m/s, f_beat={f_beat:.2f} Hz, f_doppler={f_doppler:.2f} Hz")
    # phase_grid = 2j * np.pi * (f_beat * t_grid + f_doppler * T_chirp * chirp_idx_grid)
    phase_grid = 2j * np.pi * (f_beat * t_grid - f_doppler * T_chirp * chirp_idx_grid)

    raw_data += snr * np.exp(phase_grid)
print("数据生成完毕。")


# # --- 添加复高斯白噪声 ---
# noise_power = 1  # 控制噪声功率（越大噪声越强）
# noise = (np.random.randn(*raw_data.shape) + 1j * np.random.randn(*raw_data.shape)) * np.sqrt(noise_power / 2)
# raw_data += noise


# %%
# ==========================================================
# --- 3. 距离维FFT ---
# ==========================================================
print("第一步: 正在执行距离维FFT...")
window_range = np.hanning(num_samples)
range_fft_data = np.fft.fft(raw_data * window_range, axis=1)
# range_fft_data = np.fft.fft(raw_data, axis=1)
# --- 诊断慢时间图 ---
range_bin_index = np.argmax(np.abs(range_fft_data[0, :]))
slow_time_signal = range_fft_data[:, range_bin_index]

print("正在生成“慢时间信号”诊断图...")
plt.figure(figsize=(12, 6))
plt.plot(np.real(slow_time_signal), label='实部 (I)')
plt.plot(np.imag(slow_time_signal), label='虚部 (Q)')
plt.title('诊断图：最大能量的距离单元的慢时间信号')
plt.xlabel('Chirp 序号')
plt.ylabel('信号幅度')
plt.legend()
plt.grid(True)
plt.show()
#%%
# --- 距离维度可视化 ---
range_map = 20 * np.log10(range_fft_data + 1e-6)  # dB 显示
range_fft_magnitude = 20 * np.log10(np.abs(range_fft_data[:, :num_samples//2]) + 1e-6)  # 取前一半 & 转为 dB

# 构造距离轴（只用正频率）
range_axis = np.arange(num_samples) * range_res/2

plt.figure(figsize=(6, 5))
plt.imshow(
    range_fft_magnitude,
    aspect='auto',
    cmap='jet',
    extent=[range_axis[0], range_axis[-1], 0, num_chirps]
)
plt.colorbar(label='幅值 / dB')
plt.xlabel('距离 / m')
plt.ylabel('帧编号（Frame）')
plt.title('距离 - FFT（Range-Time Map）')
plt.tight_layout()
plt.show()
# %%
# ==========================================================
# --- 4. 速度维FFT ---
# ==========================================================
print("第二步: 正在执行速度维FFT...")
window_doppler = np.hanning(num_chirps)
doppler_fft_data = np.fft.fft(range_fft_data * window_doppler[:, np.newaxis], axis=0)
# doppler_fft_data = np.fft.fft(range_fft_data, axis=0)
rdm = np.fft.fftshift(doppler_fft_data, axes=0)

# %%
# ==========================================================
# --- 5. RDM 可视化 ---
# ==========================================================
print("第三步: 正在生成RDM图像...")
rdm_db = 20 * np.log10(np.abs(rdm) + 1e-6)
rdm_db[np.isneginf(rdm_db)] = 0

doppler_res = lambda_c / (2 * num_chirps * T_chirp)

range_axis = np.arange(num_samples) * range_res
velocity_axis = np.arange(-num_chirps // 2, num_chirps // 2) * doppler_res

rdm_plot = rdm_db  # 不截断显示全部

plt.figure(figsize=(12, 8))
plt.imshow(
    rdm_plot,
    aspect='auto',
    extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
    cmap='jet'
)
plt.title('最终RDM图（多目标，有噪声）')
plt.xlabel('距离 (m)')
plt.ylabel('速度 (m/s)')
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# 多目标红色标注
for i, (r, v, _) in enumerate(targets_truth):
    plt.plot(r, v, 'rx', markersize=10, markeredgewidth=2, label=f'目标{i+1}: ({r}m, {v}m/s)')

plt.legend(loc='upper right')
plt.colorbar(label='信号强度 (dB)')
plt.show()

print("处理完成。")

# %%
