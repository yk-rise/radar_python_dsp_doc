import numpy as np

# 假设的参数
numTx = 2  # 发射天线数量
numRx = 3  # 接收天线数量
posTx = np.array([[0, 0, 0], [1, 0, 0]])  # 发射天线位置
posRx = np.array([[0, 1, 0], [0, -1, 0], [1, 1, 0]])  # 接收天线位置
# 假设目标位置（这里简单模拟几个点）
posNew = np.array([[2, 2, 0], [3, 3, 0], [4, 4, 0]])
axisTime = np.array([0.1, 0.2, 0.3])  # 简单的时间轴
scipy_constants_c = 3e8  # 模拟光速

# 循环计算
for i in range(numTx):
    for j in range(numRx):
        index = i * numRx + j
        tmpTx = posTx[i]
        tmpRx = posRx[j]
        disTx = np.linalg.norm(posNew - tmpTx, axis=1)
        disRx = np.linalg.norm(posNew - tmpRx, axis=1)
        txTimeStamp = axisTime - (disTx + disRx) / scipy_constants_c

        print(f"发射天线索引: {i}, 接收天线索引: {j}")
        print(f"目标到发射天线的距离 disTx: {disTx}")
        print(f"目标到接收天线的距离 disRx: {disRx}")
        print(f"考虑延迟后的时间轴 txTimeStamp: {txTimeStamp}")
        print("-" * 50)
        import numpy as np

