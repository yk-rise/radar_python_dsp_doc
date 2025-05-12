[[雷达]]
## 什么是CFAR
CFAR即恒虚警率检测器，能够在复杂且不断变化的环境中，动态调整检测门限，从而能够保持较低虚警率 **（恒虚警率）** 的同时，尽可能准确的发现真实目标。
CFAR的核心思想是：动态地适应局部环境，它不使用全局统一的阈值，而是为地图上的**每一个点**，单独计算它周围区域的**平均噪声水平**，然后基于这个**局部**噪声水平来判断这个点是否“足够突出”，从而成为一个潜在的目标。也就是说，一个信号是否为目标，不应该由一个固定的绝对强度来判断，而应该看它相对于其局部背景噪声有多么突出。
通过这种方式，如果背景噪声整体升高，CFAR估计出的噪声水平也会相应升高，检测门限随之提高，从而避免将增强的噪声误判为目标。反之，如果背景噪声降低，门限也会降低，使得雷达能检测到更微弱的目标。它追求的是在各种环境下，将噪声误判为目标的概率保持在一个相对稳定的低水平。
## 常用CFAR种类
### CA-CFAR
 **CA-CFAR**：通过滑动窗口估计背景噪声功率。窗口中心为待检测单元（CUT），两侧为保护单元（避免信号能量干扰）和训练单元（用于噪声估计）。噪声功率取训练单元的平均值，结合虚警概率（P_fa）计算动态阈值。
 CA-CFAR噪声估计： $$ P_n = \frac{1}{N} \sum_{i=1}^{N} X_i $$ 其中，$$N$$为训练单元数量，$$X_i$$
 适用均匀噪声环境，无邻近目标干扰。
### GO-CFAR
**GO-CFAR**：取前/后训练单元平均值的较大者，抑制杂波边缘影响。 
GO-CFAR噪声估计： $$ P_n = \max \left( \frac{1}{N/2} \sum_{\text{前}} X_i, \frac{1}{N/2} \sum_{\text{后}} X_i \right) $$
适用杂波边缘环境。（杂波边缘是不同杂波特性区域的分界线）
### SO-CFAR
 **SO-CFAR**：取前/后训练单元平均值的较小者，提升密集目标检测能力。
 SO-CFAR噪声估计： $$ P_n = \min \left( \frac{1}{N/2} \sum_{\text{前}} X_i, \frac{1}{N/2} \sum_{\text{后}} X_i \right) $$
 适用密集目标环境。
### OS-CFAR
**OS-CFAR**:对训练单元信号按强度排序，选择中间值作为噪声估计，避免异常值（如邻近目标）的影响，提高非均匀环境下的鲁棒性。
排序后噪声估计： $$ P_n = X_{(k)} \quad $$ 其中，$$X_{(k)}$$为排序后的第$$k$$个值。
适用存在邻近目标或杂波边缘的非均匀环境。
### CMLD-CFAR
**CMLD-CFAR**:截断训练单元中高能量样本（如剔除前m个最大值），计算剩余样本均值，避免强干扰目标抬升阈值。 
截断后噪声估计： $$ P_n = \frac{1}{N - m} \sum_{i=1}^{N - m} X_{(i)} $$ 其中，$$X_{(i)}$$为排序后保留的样本.
适用存在少量强干扰目标的非均匀环境。

## CFAR的python仿真
整个雷达信号处理是通过`LFMCWRadarProcessor` 类来实现。而CFAR是通过其中的`searchPeak_in_AmpSpec`方法执行。
~~~python
  def searchPeak_in_AmpSpec(self, rdm, numTrain, numGuard, thCFAR=1.5, thAMP=0.1, type="GOCA"):

        ampSpec2D = np.abs(rdm[0]) + np.abs(rdm[1])

        (_, noiselevel) = cfar_2d(ampSpec2D, numTrain, numGuard, thCFAR, type)

        snr = ampSpec2D / noiselevel

  

        # 从幅度谱中查找目标，同时满足信噪比和幅度谱两个条件。indices是有序的，第一个维度优先级高，第二个维度优先级低。C语言实现时也要满足

        indices = np.argwhere(np.logical_and(snr > thCFAR, ampSpec2D > thAMP))

        # print(indices)

  

        # 在同一个距离单元中，将连续的一个检测出来的点合并成一个从而减少速度弥散带来的影响

        bools = np.ones(shape=(len(indices)), dtype=bool)

        amp = ampSpec2D[indices[:, 0], indices[:, 1]]

  

        a = (indices[:-1, 0] == indices[1:, 0]) & (indices[:-1, 1] == indices[1:, 1] - 1)

        b = amp[:-1] < amp[1:]

        bools[:-1] &= ~(a & b)

        bools[1:] &= ~(a & ~b)

        indices = indices[bools]

  

        points = []

        for index in indices:

            point = RadarPointCloud(

                radius=index[0] * self.param.resRange,

                radialVelocity=(index[1] - self.param.numChirp / 2) * self.param.resVelocity,

                amplitude=ampSpec2D[tuple(index)],

                theta=angleDualCh(rdm[0, index[0], index[1]], rdm[1, index[0], index[1]]),

            )

            points.append(point)

  

        # dh.draw_spectrum(ampSpec2D)

        # snr[ampSpec2D < thAMP] = 0

        # dh.draw_spectrum(snr)

        # print(f"平均幅度:{np.mean(ampSpec2D)}")

        return points
~~~
这个函数
1. 首先调用`myRadar.cfar` 模块中的 `cfar_2d` 函数，输入参数：
- `ampSpec2D`：当前帧的二维幅度谱数据（通常是各个通道幅度谱的叠加）。
- `self.config.numTrain`：训练单元数量配置。
- `self.config.numGuard`：保护单元数量配置。
- `self.config.cfarThreshold`：门限因子。
- `type="GOCA"`：指定使用“取大单元平均”的CFAR

2. 之后`cfar_2d`函数返回两个结果，我们主要用`noiselevel`，`noiselevel` 是一个与 `ampSpec2D` 同样大小的二维数组，其中每个单元的值代表了对应位置的局部背景噪声功率估计。

3. 计算信噪比，`snr = ampSpec2D / noiselevel`。这表示每个单元的信号强度是其局部噪声强度的多少倍。

4. 进行判决：`np.logical_and(snr > self.config.cfarThreshold, ampSpec2D > self.config.ampThreshold)`
- `snr > self.config.cfarThreshold`：这是CFAR的核心判决。只有当一个单元的信噪比大于设定的门限因子时，才认为它通过了CFAR检测。
- `ampSpec2D > self.config.ampThreshold`：这是一个额外的绝对幅度门限，用于滤除那些虽然相对背景噪声显著，但其绝对信号强度过低的点。这些点可能不是我们关心的真实目标。

5. `np.argwhere` 用来找出所有满足条件的索引，表示初步检测到的目标。

### GOCA-CFAR
GOCA (Greatest Of Cell-Averaging) CFAR，即“取大单元平均CFAR”，是CA-CFAR (Cell-Averaging CFAR，单元平均CFAR) 的一种改进。
- **CA-CFAR**：简单地将CUT周围所有训练单元的功率平均起来作为噪声估计。
- **GOCA-CFAR**：它通常会将训练单元分成几个部分（例如，在二维情况下，分成距离维度的前导/后随训练窗，和速度维度的前导/后随训练窗，共四个“分支”或“臂”）。分别计算每个分支的平均功率，然后取这些平均功率中的 **最大值** 作为最终的局部噪声估计。
GOCA-CFAR对于处理非均匀噪声环境（杂波边缘）或者存在多个密集目标时（共性都是背景噪声并不是单纯的均匀噪声），只选择噪声最大的那个分支作为噪声评估水平。
==这里其实就是分了四块的GO-CFAR。==
#### 十字形CFAR
十字形CFAR在博客、网页上没有找到解释。
通过查找相关论文，得知十字形CFAR通常用来减少目标的回波副峰和旁瓣对检测的影响[1]  [2]。
旁瓣对目标检测的影响：
1. 增加虚警率，将非目标的强回波误判为目标
2. 旁瓣可能在RDM中形成虚假峰值。（可能和鬼影有关）
3. 干扰多目标分辨能力，多个目标距离较近的时候，其中一个目标的旁瓣可能覆盖另一个真实目标。
而之前从仿真数据得到的RDM图中也可以看到目标的旁瓣是沿着距离维和速度维延申的，呈现十字形。而用十字形CFAR可以避免将旁瓣检测成目标。
![[Pasted image 20250510143447.png]]

参考文献：
[1] 邹俊杰, 程丰, 万显荣. 外源雷达空时联合恒虚警检测分析与实验[J]. 雷达科学与技术, 2022, 20(4): 415-420.
[2] 牛蕾. SAR 图像舰船目标去旁瓣处理[J]. 雷达科学与技术, 2018, 16(2): 197-200.

## CFAR的C实现
输入数据仍然是RDM矩阵，存储在 `radar_handle->basic.magSpec2D`。
参数配置：
~~~c
// 文件: Include/radar_cfar.h

typedef struct {
    uint16_t numTrain[2]; ///< 训练单元大小 [距离维, 速度维]
    uint16_t numGuard[2]; ///< 保护单元大小 [距离维, 速度维]
    float thSNR;          ///< 信噪比(SNR)阈值 (对应门限因子)
    int32_t thAmp;        ///< 幅度绝对阈值
} cfar2d_cfg_t;
~~~
- `numTrain`: 一个包含两个元素的数组，分别指定在距离维度和速度维度上，单侧训练窗口的大小。第一位是距离维度，第二位是速度维度。例如 `numTrain[0]=5` 表示在距离方向上，向上和向下各取 5 个单元作为训练单元。
- `numGuard`: 类似地，指定保护窗口的大小。
- `thSNR`: 信噪比阈值。这是一个浮点数，表示检测到的信号强度需要比估计的噪声水平高多少倍才被认为是目标。例如，`thSNR = 10.0` (对应约 10dB) 意味着信号需要比噪声强 10 倍。
- `thAmp`: 绝对幅度阈值。信号强度必须超过这个值才可能被考虑，用于过滤掉非常微弱的信号。

执行函数`radar_cfar2d_goca` 
- 所有通过检测的点（满足阈值条件）会被添加到 `cfar_result->point` 数组中，并且 `cfar_result->numPoint` 会被更新为检测到的总点数。
- 如果成功，函数返回 0。
- 输出结果会存放在一个 `cfar2d_result_t` 结构体中。
~~~c
/**
 * @brief CFAR检测点
 */
typedef struct {
    uint16_t idx0; ///< 距离维度坐标
    uint16_t idx1; ///< 速度维度坐标
    int32_t amp;   ///< 幅度
    int32_t snr;   ///< 信噪比，Q23.8
} cfar2d_point_t;

/**
 * @brief 2D-CFAR检测结果
 */
typedef struct {
    cfar2d_point_t *point;           ///< 需要外部分配内存然后赋值
    size_t capacity;                 ///< 容量
    size_t numPoint;                 ///< 当前点的个数
    uint8_t is_point_need_free  : 1; ///< point是否需要外部释放
    uint8_t is_struct_need_free : 1; ///< struct是否需要外部释放
    uint8_t reserved            : 6; ///< 保留
} cfar2d_result_t;
~~~
- `point`: 一个指向 `cfar2d_point_t` 数组的指针。每个 `cfar2d_point_t` 存储了一个检测到的潜在目标点的信息：它在幅度谱矩阵中的二维索引 (`idx0`, `idx1`)、该点的幅度值 (`amp`) 以及计算出的信噪比 (`snr`)。
- `capacity`: 表示 `point` 数组预分配的大小。
- `numPoint`: 表示 CFAR 检测后，实际找到并存入 `point` 数组的点数。
- `is_..._need_free`: 这两个标志位用于内存管理，是否需要外部释放内存。
