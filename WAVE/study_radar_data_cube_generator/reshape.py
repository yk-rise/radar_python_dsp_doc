import numpy as np

# 假设的参数
numTx = 2
numRx = 3
numFrame = 4
numChrip = 5
numSampling = 6

# 初始化二维数组
signal = np.zeros((numTx * numRx, numSampling * numChrip * numFrame), dtype=np.complex128)
print("初始形状:", signal.shape)

# 进行 reshape
signal = signal.reshape(numTx * numRx, numFrame, numChrip, numSampling)
print("reshape 后的形状:", signal.shape)

# 进行 swapaxes
signal = signal.swapaxes(0, 1)
print("swapaxes 后的形状:", signal.shape)