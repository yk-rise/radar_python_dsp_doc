import numpy as np
 
# 假设的参数
fc  = 0 
bandwidth = 5
timeChrip = 8   #每个chrip的持续时间
slope = bandwidth / timeChrip
numSampling = 5  #每一个chrip采集五个点
numChrip = 3    #每一个帧有两个chrip个数
numFrame = 2    #帧数
freqSampling = 1 #每秒采一个点
timeIdle = 2     #两个chrip之间的空闲时间
timeNop  = 100   #一帧数据的结尾时间

# 生成时间轴
axisT = (
        #详细见study/tile.py 生成一个采样点对应的时间轴
        np.tile(np.linspace(0, numSampling / freqSampling, numSampling, endpoint=False), numChrip * numFrame)
        + np.repeat(np.linspace(0, numChrip * numFrame * (timeIdle + timeChrip), numChrip * numFrame, endpoint=False), numSampling)
        + np.repeat(np.linspace(0, numFrame * timeNop, numFrame, endpoint=False), numSampling * numChrip)
    ) 

#将时间定格在每一个chrip的第一个点
t = (axisT % ((timeChrip + timeIdle) * numChrip + timeNop)) % (timeChrip + timeIdle)

phase = 2 * np.pi * t * (fc + 0.5 * slope * t)

# 打印前 10 个时间点和对应的相位值
print("前 10 个时间点和对应的相位值：")
for i in range(10):
    print(f"时间点: {axisT[i]:.6f} 秒, 相位: {phase[i]:.6f} 弧度")