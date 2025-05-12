# %%

import numpy as np
import scipy.interpolate
import scipy.constants
import joblib


def generateRadarDataCube(frequency, bandwidth, timeChrip, timeIdle, timeNop, freqSampling, numSampling, numChrip, numFrame, posTx, posRx, targetsInfo):
    """
    生成LFMCW雷达数据立方体

    Parameters
    ----------
    frequency : float
        载波频率
    bandwidth : float
        带宽
    timeChrip : float
        一个chrip的持续时间
    timeIdle : float
        两个chrip之间的时间间隔
    timeNop : float
        一帧数据的结尾时间
    freqSampling : float
        采样频率
    numSampling : int
        采样点数
    numChrip : int
        chrip个数
    numFrame : int
        帧数
    posTx : array_like
        发射天线的位置，shape=(N,3)，N为发射天线的个数，每一行表示一个天线的坐标
    posRx : array_like
        接收天线的位置，格式posTx
    targetsInfo : list
        目标的信息

    Returns
    -------
    signal : array_like
        LFMCW雷达数据立方体，4-D数组，四个维度分别为(帧序号,通道序号,chrip序号,采样点序号)

    Examples
    --------

    """
    numTx = len(posTx)
    numRx = len(posRx)

    unusedChrip = timeChrip - (numSampling - 1) / freqSampling     #计算啁啾信号中未使用的时间部分，即啁啾总时长减去实际采样时间（考虑到采样点数和采样频率）
    axisTime = (
        #详细见study/tile.py 生成一个采样点对应的时间轴
        np.tile(np.linspace(0, numSampling / freqSampling, numSampling, endpoint=False), numChrip * numFrame)
        + np.repeat(np.linspace(0, numChrip * numFrame * (timeIdle + timeChrip), numChrip * numFrame, endpoint=False), numSampling)
        + np.repeat(np.linspace(0, numFrame * timeNop, numFrame, endpoint=False), numSampling * numChrip)
    ) + unusedChrip / 2
        #用于计算给定时间轴上的相位值，详见study/clacPhase.py
    def clacPhase(axisT, fc, timeChrip, timeIdle, timeNop, numChrip, slope):
        t = (axisT % ((timeChrip + timeIdle) * numChrip + timeNop)) % (timeChrip + timeIdle)
        phase = 2 * np.pi * t * (fc + 0.5 * slope * t)
        return phase
     #定义一个并行计算版本的 clacPhase 函数 clacPhase_parallel，
    def clacPhase_parallel(axisT, fc, timeChrip, timeIdle, timeNop, numChrip, slope):
        #joblib.cpu_count()：调用 joblib 库的 cpu_count 函数，该函数会返回当前系统可用的 CPU 核心数
        num_slices = joblib.cpu_count() * 8
        #使用 numpy 库的 array_split 函数将 axisT 数组分割成 num_slices 个切片。array_split 会尽量平均地分割数组，
        axisT_slices = np.array_split(axisT, num_slices)
        #joblib.Parallel(n_jobs=-1)：创建一个并行计算的上下文，n_jobs=-1 表示使用所有可用的 CPU 核心进行并行计算。
        #joblib.delayed(clacPhase)：delayed 是 joblib 库中的一个函数，用于将 clacPhase 函数封装成一个可延迟执行的对象。
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(clacPhase)(slice_, fc, timeChrip, timeIdle, timeNop, numChrip, slope) for slice_ in axisT_slices)
        return np.concatenate(results)

    phaseTx = clacPhase_parallel(axisTime, frequency, timeChrip, timeIdle, timeNop, numChrip, bandwidth / timeChrip)

    signal = np.zeros((numTx * numRx, numSampling * numChrip * numFrame), dtype=np.complex128)  #

    tragetsPos = []
    for target in targetsInfo:
        # 1. 补全目标的位置信息，对于目标轨迹时间轴之外的时间点，使用外插法补全
        # 使用原始时间轴的边界值进行填充
        interp_pos = scipy.interpolate.interp1d(
            #kind="quadratic"：指定插值的类型。"quadratic" 表示使用二次插值，也就是用二次多项式来拟合数据点，进而得到插值函数。除了 "quadratic"，还能选择其他类型，例如 "linear"（线性插值）、"cubic"（三次插值）等。
            #该参数用于控制当插值点超出已知数据点范围时是否抛出错误。bounds_error=False 表示不抛出错误，而是使用 fill_value 指定的值进行填充。
            #在插值点小于最小时间点时，使用 target["pos"] 的第一个元素进行填充；在插值点大于最大时间点时，使用 target["pos"] 的最后一个元素进行填充。
            #详见
            target["times"], target["pos"], axis=0, kind="quadratic", bounds_error=False, fill_value=(target["pos"][0], target["pos"][-1])
        )

        posNew = interp_pos(axisTime)
        tragetsPos.append(posNew)
        # 2. 计算目标在不同时刻时的 电磁波从发射然后经过反射最后被天线接收的时间延迟，减去延迟和得到新的时间轴
        #详见targetsPos.py
        for i in range(numTx):
            for j in range(numRx):
                index = i * numRx + j
                tmpTx = posTx[i]   #tmpTx 和 tmpRx 是当前发射天线和接收天线的三维坐标。
                tmpRx = posRx[j]
                disTx = np.linalg.norm(posNew - tmpTx, axis=1)
                disRx = np.linalg.norm(posNew - tmpRx, axis=1)
                txTimeStamp = axisTime - (disTx + disRx) / scipy.constants.c

                signal[index, :] += (
                    target["rsc"]
                    / (disTx**2)
                    / (disRx**2)
                    * np.exp(1j * (phaseTx - clacPhase_parallel(txTimeStamp, frequency, timeChrip, timeIdle, timeNop, numChrip, bandwidth / timeChrip)))
                )
    tragetsPos = np.array(tragetsPos)
    signal = signal.reshape(numTx * numRx, numFrame, numChrip, numSampling).swapaxes(0, 1)  #这里交换了第一维和第二维，

    return signal, tragetsPos


# %% 测试

def trajGen_line(posStart: np.ndarray, velocity: np.ndarray, freqSampling, numSamples):
    # 初始化数组来存储时间和位置
    times = np.arange(0, numSamples) / freqSampling
    positions = np.matmul(times.reshape((-1, 1)), velocity.reshape((1, 3))) + posStart.reshape((1, 3))

    return times, positions

if __name__ == "__main__":
    import cProfile
    import drawhelp.draw as dh 
    import plotly.graph_objects as go
    from scipy.fft import fft, fftshift, ifft, fft2

    frequency = 24e9
    bandwidth = 1000e6
    timeChrip = 150e-6
    timeIdle = 200e-6
    timeNop = 2000e-6  #
    freqSampling = 1e6

    numSampling = 128
    numChrip = 32
    numFrame = 10

    posTx = np.array([[0, 0, 0]])
    posRx = np.array([[0, -0.25 * scipy.constants.c / frequency, 0], [0, 0.25 * scipy.constants.c / frequency, 0]])

    # targetsInfo包含多个目标的轨迹，每个目标用一个字典储存。

    times, positions = trajGen_line(np.array([10, 5, 0]), np.array([-1, 0, 0]), freqSampling, freqSampling * 10)

    target0 = dict(rsc=1, times=times, pos=positions)

    times, positions = trajGen_line(np.array([5, 0, 0]), np.array([1, 0, 0]), freqSampling, freqSampling * 10)
    target1 = dict(rsc=1, times=times, pos=positions)

    targetsInfo = [target0, target1]

    signal, _ = generateRadarDataCube(
        frequency, bandwidth, timeChrip, timeIdle, timeNop, freqSampling, numSampling, numChrip, numFrame, posTx, posRx, targetsInfo
    )

    frame0 = signal[0, 0, :]
    frame1 = signal[0, 1, :]

    # 观察RDM
    resRange = scipy.constants.c / (2 * bandwidth * (numSampling / freqSampling) / timeChrip)
    axis_x = np.arange(0, 128) * resRange
    axis_y = np.arange(-16, 16) * scipy.constants.c / (2 * frequency * (timeIdle + timeChrip) * 31)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(frame0), axes=0))
    # dh.draw_spectrum(spec, x=axis_x, y=axis_y)
    dh.draw_spectrum(spec)

    # 相位法测角

    pos0 = (14, 64)
    pos1 = (18, 29)

    rdm0 = fftshift(fft2(frame0), axes=0)
    rdm1 = fftshift(fft2(frame1), axes=0)

    cplx0 = rdm0[pos0[0], pos0[1]]
    cplx1 = rdm1[pos0[0], pos0[1]]
    phsaeDelta = np.arctan2(cplx0.imag, cplx0.real) - np.arctan2(cplx1.imag, cplx1.real)
    if phsaeDelta < -np.pi:
        phsaeDelta += 2 * np.pi
    elif phsaeDelta > np.pi:
        phsaeDelta -= 2 * np.pi
    theta = np.arcsin(phsaeDelta / np.pi)
    print(pos0[1] * resRange * np.cos(theta), pos0[1] * resRange * np.sin(theta))


# %%
