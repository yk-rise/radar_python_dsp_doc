# 目标关联
主要用于多目标跟踪系统中实现测量值与已有目标之间的匹配关联。
Associator 类负责将传感器的测量值与系统中已有的目标（假设）进行匹配，是多目标跟踪系统中的关键组件，解决 "哪个测量值属于哪个目标" 的问题。

# 工作流程：
输入已有的目标假设和新的测量值
计算每个假设与每个测量值之间的关联代价
使用某种匹配算法（如匈牙利算法、最近邻算法等）找到最优匹配
将匹配结果保存到假设中
未匹配的测量值会被移动到容器头部，用于后续的新目标初始化
提供了两个版本，一个使用类成员的missed_distance，另一个允许临时指定
# 状态更新函数
功能：根据关联结果更新目标状态
内部会调用卡尔曼更新器（KalmanUpdater），使用匹配到的测量值更新目标的状态估计
将关联后的假设结果应用到实际目标上
```c++
#pragma once

#include <radar/ot/track_kalman.hh>
#include <radar/ot/track_target.hh>
#include <vector>

class Associator
{
private:
    KalmanPredictor &predictor;
    KalmanUpdater &updater;

public:
    rd_float_t missed_distance = 5; ///< 无法关联的最大距离
    rd_float_t wr = 1;              ///< 距离权重
    rd_float_t wv = 1;              ///< 速度权重

    Associator(KalmanPredictor &predictor, KalmanUpdater &updater, rd_float_t missed_distance)
        : predictor(predictor)
        , updater(updater)
        , missed_distance(missed_distance) { };


    Associator(KalmanPredictor &predictor, KalmanUpdater &updater, rd_float_t missed_distance, rd_float_t wr, rd_float_t wv)
        : predictor(predictor)
        , updater(updater)
        , missed_distance(missed_distance)
        , wr(wr)
        , wv(wv) { };
    ~Associator() { };

     //功能：计算假设（目标预测状态）与测量值之间的 "距离"（实际是关联代价
     //返回值：距离 / 代价越小，表示测量值与目标匹配度越高
    rd_float_t distance(Hypothesis &hypothesis, Vector3r &measurement, rd_float_t wr, rd_float_t wv);

    //功能：为已有的目标创建或初始化假设（Hypothesis）
    //hypotheses_new 直接创建新的假设集合，hypotheses_init 则是在已有集合上初始化
    void hypotheses_init(std::vector<Hypothesis> &hypotheses, TrackedTargets &targets);

    std::vector<Hypothesis> hypotheses_new(TrackedTargets &targets);


    /**
     * @brief 数据关联
     *
     * @details 将假设中的先验状态和测量值进行关联，匹配结果保存到对应假设中；未使用的测量值保存在measurements头部
     *
     * @param[in,out]   hypotheses          假设；输入时假设中需要包含先验状态， 关联后将测量值和预测值保存在假设中
     * @param[in,out]   measurements        测量值；未关联的测量值会移动到measurements头部
     * @param           timestamp_ms        时间戳
     * @param           missed_distance     关联成功的最大距离
     */
    void associate(std::vector<Hypothesis> &hypotheses, std::vector<Vector3r> &measurements, uint32_t timestamp_ms, rd_float_t missed_distance);

    void associate(std::vector<Hypothesis> &hypotheses, std::vector<Vector3r> &measurements, uint32_t timestamp_ms)
    {
        associate(hypotheses, measurements, timestamp_ms, this->missed_distance);
    }


    /**
     * @brief 状态更新
     *
     * @details 调用卡尔曼更新器，处理每一个目标
     *
     * @param targets
     * @param hypotheses
     */
    void update(TrackedTargets &targets, std::vector<Hypothesis> &hypotheses);
};

```