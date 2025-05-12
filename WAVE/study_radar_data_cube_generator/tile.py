import numpy as np

# 假设的参数
numSampling = 5  #每一个chrip采集五个点
numChrip = 3    #每一个帧有两个chrip个数
numFrame = 2    #帧数
freqSampling = 1 #每秒采一个点
timeIdle = 2     #两个chrip之间的空闲时间
timeChrip = 8   #每个chrip的持续时间
timeNop  = 100   #一帧数据的结尾时间
unusedChrip = timeChrip - (numSampling - 1) / freqSampling  #计算未使用的啁啾时间 减一是因为其间隔为1/freqSamplingm 占据了numSampling - 1个间隔
# 生成单个啁啾内的采样时间数组
single_chirp_time = np.linspace(0, numSampling /freqSampling , numSampling, endpoint=False)
single_chirps_time = np.linspace(0, numChrip * numFrame * (timeIdle + timeChrip), numChrip * numFrame, endpoint=False)
chirps_time = np.linspace(0, numFrame * timeNop, numFrame, endpoint=False)

print("单个啁啾内的采样时间数组:")
print(single_chirp_time)
print("啁啾的采样时间数组:")
print(single_chirps_time)
print("啁啾的采样时间数组:")
print(chirps_time)

# 使用 np.tile 进行重复
repeated_time = np.tile(single_chirp_time, numChrip * numFrame)
repeated_times = np.repeat(single_chirps_time,numSampling)
repeated_timess = np.repeat(chirps_time, numSampling * numChrip)
print("\n重复后的时间数组:")
print(repeated_time)
print("\n重复后的时间数组:")
print(repeated_times)
print("\n重复后的时间数组:")
print(repeated_timess)

repeated_times_reshape = repeated_time + repeated_times + repeated_timess + unusedChrip / 2
print("\n重复后的时间数组:")
print(repeated_times_reshape)
