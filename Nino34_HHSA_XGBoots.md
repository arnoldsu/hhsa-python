# Niño3.4 HHSA–XGBoost：思路演化、实现、测试结果与下一步

## 1. 研究目标

本项目检验以下假设：

> Niño3.4 的年际振荡可以表示为一个或多个归一化 FM carrier 与其 AM
> envelope 的乘积；10–25 年低频 AM 状态调制 ENSO 事件的幅度，并可能为
> 3、6、12 个月预测提供信息。

这里必须区分两种“非线性”：

1. XGBoost 的函数拟合能力是非线性的；
2. HHSA 所描述的是具有明确结构的乘性非线性：

   [
   IMF_k(t)=A_k(t)F_k(t),
   ]

   其中 (A_k(t)) 是 AM envelope，(F_k(t)) 是归一化 FM carrier。

仅把 HHSA 数组展开成表格交给 XGBoost，并不等于让模型理解这个乘法结构。
本项目因此经历了四轮模型设计，最终将 envelope–carrier 乘法直接写进预测
架构。

## 2. 数据与 HHSA 输出

### 2.1 Niño3.4 数据

- 本地输入：`data/nino34_monthly.npz`
- 变量：月平均 Niño3.4 SST anomaly，单位 °C
- 时间：1948-01 至 2026-07
- 样本：943 个月
- 采样率：12 samples/year

### 2.2 完整 HHSA 特征

完整离线分解保存在：

`outputs/nino34_hhsa_features.npz`

主要变量如下：

| 变量 | 形状 | 含义 |
|---|---:|---|
| `date` | `(943,)` | 月时间坐标 |
| `nino34_anomaly_c` | `(943,)` | 原始 Niño3.4 |
| `IMF` | `(943, 8)` | 第一层 carrier IMF |
| `fm` | `(943, 8)` | 第一层瞬时频率 |
| `am` | `(943, 8)` | 第一层 carrier envelope |
| `IMF2` | `(943, 6, 8)` | envelope 的第二层 EMD 分量 |
| `FM` | `(943, 6, 8)` | 第二层瞬时调制频率 |
| `AM` | `(943, 6, 8)` | 第二层分量自身的瞬时振幅 |

要注意三个不同对象：

- `am[:, k]`：第 (k) 个 carrier 的完整 envelope；
- `IMF2[:, j, k]`：完整 envelope 内带正负相位的第 (j) 个调制分量；
- `AM[:, j, k]`：该第二层调制分量自身的正振幅。

预测 carrier magnitude 时的直接目标是第一层 `am`。10–25 年
`IMF2` 状态用于帮助预测这个未来 envelope。

### 2.3 当前实现不是随机 EEMD

当前代码使用从原始 MATLAB HHSA 流程移植的 masking EMD 和 direct
quadrature，而不是 PyEMD 的加噪 ensemble EEMD。论文中应称为
Masking-EMD/HHSA，除非以后另行实现并验证 EEMD 版本。

## 3. HHSA 数学逻辑

### 3.1 Envelope 与归一化 carrier

对一个满足 IMF 条件的信号，上下包络分别为 \(U_k(t)\) 和 \(L_k(t)\)：

$$
M_k(t)=\frac{U_k(t)+L_k(t)}{2},
\qquad
A_k(t)=\frac{U_k(t)-L_k(t)}{2}.
$$

由于 IMF 的局部均值近似为零：

$$
F_k(t)\simeq\frac{IMF_k(t)}{A_k(t)+\epsilon},
\qquad
IMF_k(t)=A_k(t)F_k(t).
$$

`F_k(t)` 通常具有近似 `[-1, 1]` 的局部幅度。单位余弦的标准差是
\(1/\sqrt 2\)，因此“局部最大/最小为 ±1”与“标准差为 1”不是同一归一化
定义。如果把 carrier 强制标准化为标准差 1，必须对 envelope 做反向尺度
补偿才能保持乘积不变。

Niño3.4 的 ENSO 部分表示为：

$$
N_{ENSO}(t)=A_3(t)F_3(t)+A_4(t)F_4(t).
$$

完整预测还包含其他 IMF 与趋势组成的残差：

$$
N(t)=A_3(t)F_3(t)+A_4(t)F_4(t)+R(t).
$$

### 3.2 第二层 AM

第一层 envelope 可以进一步分解：

$$
A_k(t)=\sum_j IMF2_{j,k}(t)+R_{A,k}(t).
$$

这使 carrier frequency 和 AM frequency 可以同时表达。HHSA 的原始理论
工作明确提出用嵌套 EMD/HHT 表示 additive 与 multiplicative、AM 与 FM
过程；它也强调理想 AM 是 FM carrier 的 envelope，而普通 Hilbert transform
并不总能为任意信号生成良好 envelope
（[Huang et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792412/)）。

## 4. 主 ENSO carrier 与低频 AM 的识别

为了减小边界影响，诊断时排除首尾各 24 个月。每个 IMF 的结果如下：

| 零基编号 | 常用编号 | 振幅加权平均周期 | 2–7 年频带能量占比 | 与 Niño3.4 相关 |
|---:|---:|---:|---:|---:|
| 0 | IMF1 | 0.294 年 | 0.0% | 0.140 |
| 1 | IMF2 | 0.555 年 | 0.4% | 0.158 |
| 2 | IMF3 | 1.242 年 | 2.7% | 0.372 |
| **3** | **IMF4** | **2.313 年** | **97.0%** | **0.607** |
| **4** | **IMF5** | **4.308 年** | **98.7%** | **0.663** |
| 5 | IMF6 | 9.989 年 | 0.3% | 0.389 |
| 6 | IMF7 | 14.461 年 | 0.0% | 0.277 |
| 7 | residual/slow | 41.507 年 | 0.7% | 0.275 |

因此项目不把 ENSO 固定为单一 IMF，而使用两个 carrier：

$$
N_{ENSO}(t)=IMF_3(t)+IMF_4(t).
$$

两个 carrier 合计约占原始 Niño3.4 方差的一半。

识别出的主要第二层调制为：

| Carrier | 第二层分量 | AM 周期 | 该 carrier 第二层方差占比 |
|---|---|---:|---:|
| 2.31 年 `IMF_3` | `IMF2[:,1,3]` | 9.66 年 | 59.2% |
| 2.31 年 `IMF_3` | `IMF2[:,2,3]` | 18.03 年 | 9.0% |
| 4.31 年 `IMF_4` | `IMF2[:,0,4]` | 11.71 年 | 58.8% |
| 4.31 年 `IMF_4` | `IMF2[:,1,4]` | 25.33 年 | 27.2% |

约 32–42 年分量由于 78.6 年记录只能覆盖约 2 个周期，暂不作为主要
ML 输入。

## 5. 统一实验设置

所有原型使用：

- 提前期：3、6、12 个月；
- 训练目标按 target date 对齐；
- 训练集：截至 2013-12；
- 验证集：2014-01 至 2020-12，每个 lead 84 个目标月；
- 测试集：2021-01 至 2026-07，每个 lead 67 个目标月；
- 指标：RMSE、ACC/Pearson correlation、相对 persistence 的 MSE skill；
- XGBoost 使用固定 seed 和正则化浅树；
- 所有实验保持严格时间顺序，不随机切分月份。

Skill 定义为：

$$
Skill=1-\frac{MSE_{model}}{MSE_{persistence}}.
$$

## 6. 第一轮：全部 HHSA 特征作为普通表格

### 6.1 逻辑

`src/hhsa/ml.py` 最初把全部第一层和第二层数组按多个 lag 展开：

`Raw → HHSA without AM → HHSA with all AM → XGBoost`

这能测试“HHSA 数组是否包含预测信息”，但不能测试明确的乘性 AM 机制。

### 6.2 测试集结果

| Lead | XGB Raw RMSE/ACC | HHSA No-AM RMSE/ACC | HHSA All-AM RMSE/ACC |
|---:|---:|---:|---:|
| 3 | 0.477 / 0.821 | **0.291 / 0.957** | 0.361 / 0.944 |
| 6 | 0.678 / 0.588 | **0.475 / 0.878** | 0.572 / 0.875 |
| 12 | 0.842 / 0.300 | **0.598 / 0.753** | 0.663 / 0.689 |

加入全部第二层 AM 反而降低技巧。该模型约有 687 个输入，而训练月数不到
800，包含大量非 ENSO、高频、长周期、零值及残差特征，过拟合风险很高。

结果：`outputs/ml/`

## 7. 第二轮：物理筛选 carrier 与 AM

### 7.1 逻辑

将特征压缩为：

- Raw：15 个；
- Raw + 两个 ENSO carrier：23 个；
- Carrier + 第一层 envelope/frequency：39 个；
- 再加四个精选 LF-AM 的 `IMF2/FM/AM`：87 个。

这一轮仍是普通特征回归，但清除了大量无关模式。

### 7.2 测试集结果

| Lead | XGB Raw | ENSO carrier | Carrier + envelope | Selected AM |
|---:|---:|---:|---:|---:|
| 3 RMSE | 0.477 | 0.468 | **0.468** | 0.495 |
| 3 ACC | 0.821 | 0.858 | **0.860** | 0.859 |
| 6 RMSE | 0.678 | **0.606** | 0.637 | 0.644 |
| 6 ACC | 0.588 | **0.739** | 0.713 | 0.671 |
| 12 RMSE | 0.842 | **0.582** | 0.632 | 0.658 |
| 12 ACC | 0.300 | **0.749** | 0.721 | 0.613 |

精选 carrier 对 6–12 个月预测有价值，但把 AM 作为额外表格列仍没有改善。

结果：`outputs/ml_selected_am/`

## 8. 第三轮：learned multiplicative gate

### 8.1 逻辑

`src/hhsa/structured_ml.py` 分别预测基础振幅和相位，再让 AM 模型学习
对数振幅修正：

$$
\hat A_k
=
\exp(\log\hat A_{base,k}+\log\hat G_{AM,k})
=
\hat A_{base,k}\hat G_{AM,k}.
$$

随后重构：

$$
\hat N(t+h)
=\sum_{k=3,4}\hat A_k(t+h)\cos\hat\phi_k(t+h)+\hat R(t+h).
$$

Gate 的训练目标来自 forward-only out-of-fold 基础振幅残差，避免使用
训练集内残差制造虚假 AM 技巧。实验包含 `gate=1`、shuffled-AM 和
real-AM。

### 8.2 测试集结果

| Lead | Gate=1 RMSE/ACC | Shuffled AM RMSE/ACC | Real AM RMSE/ACC |
|---:|---:|---:|---:|
| 3 | **0.555 / 0.878** | 0.556 / 0.882 | 0.583 / 0.859 |
| 6 | **0.624 / 0.810** | 0.636 / 0.791 | 0.674 / 0.758 |
| 12 | **0.631 / 0.769** | 0.653 / 0.744 | 0.651 / 0.733 |

虽然乘法被显式写入，模型仍然让 XGBoost“猜一个 gate”，没有直接预测
HHSA 已定义的完整 envelope。真实 AM 没有超过控制组。

结果：`outputs/ml_structured_hhsa/`

## 9. 第四轮：直接预测 envelope × normalized FM carrier

### 9.1 当前正确实现

当前主要模型位于 `src/hhsa/envelope_ml.py`。

首先使用已提取的第一层 envelope：

$$
A_k(t)=am[:,k].
$$

由于 cubic-spline envelope 偶尔产生很小的负过冲，代码只将这些理论上
不应为负的值抬升到由数据决定的小正数 floor。然后严格定义：

$$
F_k(t)=\frac{IMF_k(t)}{A_k(t)}.
$$

单元测试确认：

$$
A_k(t)F_k(t)=IMF_k(t)
$$

达到数值精度。

### 9.2 ML 如何工作

模型不直接预测最终 Niño3.4，而是分成三类目标。

#### A. FM carrier forecast

每个 carrier 使用过去 25 个月的 normalized carrier、过去频率和季节相位，
直接预测：

$$
\hat F_k(t+h).
$$

预测结果限制在 `[-1.25, 1.25]`，防止低 envelope 附近的数值异常产生
不合理 carrier。

#### B. AM envelope forecast

基础模型使用过去 60 个月完整 envelope 及变化率预测：

$$
\log\hat A_k(t+h).
$$

LF-AM 模型在相同基础输入上增加对应 9.7–25.3 年的
`IMF2/FM/AM` 历史状态。由于目标是 `log(A)`，反变换后的 envelope
保持为正。

三组 envelope 对照是：

1. `hhsa_no_lf_am`：只使用完整 envelope 自身历史；
2. `hhsa_shuffled_lf_am`：训练时打乱 LF-AM 与时间的对应；
3. `hhsa_real_lf_am`：使用真实 LF-AM 时间关系。

#### C. 强制乘法和重构

$$
\widehat{IMF}_k(t+h)
=
\hat A_k(t+h)\hat F_k(t+h),
$$

$$
\widehat{NINO3.4}(t+h)
=
\widehat{IMF}_3(t+h)
+
\widehat{IMF}_4(t+h)
+
\hat R(t+h).
$$

乘法由 HHSA 恒等式强制执行，不交给树模型自行发现。

### 9.3 Oracle 诊断

Oracle 只用于定位误差，不是合法预测：

- `oracle_envelope`：提供真实未来 envelope，只预测 carrier；
- `oracle_carrier`：提供真实未来 carrier，只预测 envelope。

如果 oracle envelope 改善更大，瓶颈是 envelope forecast；反之则是 carrier
相位/FM forecast。

### 9.4 验证集结果

| Lead | No LF-AM RMSE/ACC | Shuffled LF-AM | Real LF-AM | Oracle envelope | Oracle carrier |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.392 / 0.853 | 0.391 / 0.854 | **0.390 / 0.857** | 0.385 / 0.860 | 0.381 / 0.858 |
| 6 | **0.475 / 0.788** | 0.475 / 0.788 | 0.477 / 0.787 | 0.463 / 0.805 | 0.449 / 0.806 |
| 12 | 0.443 / 0.812 | **0.442 / 0.813** | 0.454 / 0.825 | 0.422 / 0.844 | 0.424 / 0.847 |

### 9.5 测试集结果

| Lead | No LF-AM RMSE/ACC | Shuffled LF-AM | Real LF-AM | Oracle envelope | Oracle carrier |
|---:|---:|---:|---:|---:|---:|
| 3 | **0.393 / 0.900** | 0.394 / 0.900 | 0.403 / 0.896 | 0.370 / 0.911 | 0.396 / 0.897 |
| 6 | 0.514 / 0.840 | 0.515 / 0.841 | **0.507 / 0.852** | 0.455 / 0.877 | 0.483 / 0.853 |
| 12 | 0.564 / 0.842 | 0.567 / 0.840 | **0.545 / 0.848** | 0.501 / 0.855 | 0.500 / 0.848 |

相对 persistence 的测试技巧：

| Lead | Real LF-AM skill |
|---:|---:|
| 3 | 0.454 |
| 6 | 0.664 |
| 12 | 0.777 |

第四轮明显优于第三轮 learned gate，并且 real LF-AM 在测试集的 6、12
个月提前期超过 no-LF 与 shuffled-LF。不过该增益没有在验证集的 6、12
个月重复，因此只能称为初步证据，不能称为稳健证明。

结果：`outputs/ml_envelope_multiply/`

## 10. 2026 事件诊断

2026-07 实际 Niño3.4 为 1.73°C：

| Lead | No LF-AM | Real LF-AM | Oracle envelope | Oracle carrier |
|---:|---:|---:|---:|---:|
| 3 | 0.713 | 0.587 | 0.915 | 0.631 |
| 6 | 0.167 | 0.095 | 0.540 | 0.070 |
| 12 | -0.021 | 0.001 | 0.540 | -0.052 |

模型仍严重低估 2026 的快速增强。3 个月 lead 中，提供真实未来 envelope
将结果从 0.587 提高到 0.915，而提供真实 carrier 只提高到 0.631。因此
主要瓶颈是 envelope forecast，而不是 envelope–carrier 乘法关系。

这也说明逐月总体 RMSE 的改善不等于强事件峰值已经预测成功。后续需要增加：

- peak-intensity loss；
- event-weighted loss；
- 未来 6/12 个月最大 envelope 的辅助目标；
- leave-one-strong-event-out 验证。

## 11. 代码、输出与复现

### 11.1 代码

| 阶段 | 文件 |
|---|---|
| 保存完整 HHSA | `examples/nino34_example.py` |
| 普通表格/精选特征模型 | `src/hhsa/ml.py` |
| learned gate 模型 | `src/hhsa/structured_ml.py` |
| envelope × carrier 模型 | `src/hhsa/envelope_ml.py` |
| 当前运行入口 | `examples/nino34_envelope_ml.py` |

### 11.2 当前模型复现

```bash
PYTHONPATH=src python examples/nino34_envelope_ml.py
```

输出：

```text
outputs/ml_envelope_multiply/metrics.csv
outputs/ml_envelope_multiply/predictions.csv
outputs/ml_envelope_multiply/test_predictions.png
```

### 11.3 自动测试

```bash
PYTHONPATH=src python -m pytest -q
```

当前结果：

```text
7 passed
```

测试包括：

- EMD 重构；
- HHSA 数组形状和非负谱能量；
- ML lead/lag 对齐；
- chronological metric；
- gate forward-only out-of-fold；
- envelope × normalized carrier 的精确 IMF 重构。

## 12. 方法是否新颖

### 12.1 已有工作

以下组成部分分别已有研究基础：

1. HHSA 的 nested EMD、carrier–AM frequency 表示及乘性 AM/FM 理论已经由
   [Huang et al. (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792412/)
   建立。
2. 对气候非平稳振荡的各 IMF 分别建模预测至少可追溯到
   [Lee and Ouarda (2011)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JD015142)。
3. 已有研究把 EMD 与 ConvLSTM encoder–decoder 结合用于 El Niño 预测
   （[Ali et al., 2021](https://doi.org/10.1016/j.cageo.2021.104695)）。
4. EMD/HHT 特征与 tree ensemble、SVM、LSTM 等 ML 的组合在其他时间序列
   领域并不罕见。
5. ENSO 深度学习预测、模拟数据预训练和 transfer learning 已由
   [Ham et al. (2019)](https://www.nature.com/articles/s41586-019-1559-7)
   等工作展示。
6. 近期 ENSO ML 越来越强调来源消融、物理可解释性和因果路径，例如
   [Colfescu et al. (2024)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL105194)
   与
   [Cui et al. (2026)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL118701)。

因此不能声称“EEMD/HHSA + ML”这个宽泛组合本身是新的。

### 12.2 本项目可能的新意

截至当前针对性检索，尚未找到与以下完整组合直接同构的已发表 ENSO 方法：

1. 用两层 HHSA 从 Niño3.4 显式识别 carrier frequency × AM frequency；
2. 自动选定约 2.3/4.3 年 ENSO carriers 和约 9.7–25.3 年 modulation；
3. 把每个 ENSO IMF 严格写为
   (Envelope\times Normalized\ FM\ carrier)；
4. 分别预测未来 envelope 与 carrier，而不是直接预测 Niño3.4；
5. 用 XGBoost 只学习两个隐状态的演化；
6. 通过 HHSA 方程强制乘法重构；
7. 使用 no-LF、shuffled-LF、real-LF 和双 oracle 消融判断 AM 的真实贡献。


![Here is a detailed diagram that breaks down how XGBoost operates, moving from the basic concept of iterative boosting to the specific optimizations that give it its speed and accuracy.](docs/images/Gemini_Generated_Image_henegnhenegnhene.png)


这可以被描述为：

> 一个具有潜在方法创新性的、HHSA 结构约束的 ENSO 预测框架。

现阶段不应写成“首个”或“世界首创”。要支持正式 novelty claim，还需要：

- 更系统的 Scopus/Web of Science/Google Scholar 检索；
- 比较 EMD–ConvLSTM 论文的具体重构方式；
- 检索专利、会议论文和非英文文献；
- 明确本方法相对于 decomposition–ensemble forecasting 的数学差异。

## 13. 当前最重要的限制

### 13.1 全记录分解造成未来信息泄漏

当前 HHSA 对 1948–2026 整段一次性分解。历史 IMF/envelope 会受到未来
样本影响，2026 又处于 EMD 最不可靠的右边界。因此所有 ML 指标必须标为
`exploratory/offline`，不能作为正式 forecast skill。

正式版本必须使用 expanding-window：

```text
每个起报月只使用该月及以前的数据
→ 重新 HHSA
→ 按周期匹配 carrier 和 AM
→ 提取起报时可用的最后状态
```

### 13.2 IMF 重构与 envelope 数值质量

当前 `upsample_level=1` 的第一层 IMF 最大重构误差约 0.446°C。第一层
envelope 也存在很小的 cubic-spline 负过冲。当前 envelope 模型修正了负
过冲，但正式分析必须：

- 查清 upsample 后丢弃模式导致的重构误差；
- 对比 `upsample_level=0`；
- 量化端点 padding/cropping 敏感性；
- 确认每个窗口中的模式匹配稳定。

### 13.3 长周期的有效自由度

完整 AM 序列确实覆盖全部 943 个月，因此可作为逐月状态输入；但对统计
泛化而言，10、18、25 年调制只覆盖约 8、4、3 个完整周期。两者并不矛盾：

- 数据表示层面：有完整长度的月度 AM signal；
- 独立验证层面：可重复的长周期 realization 很少。

因此必须使用 surrogate、block bootstrap、事件留一和多资料/模式集合验证。

## 14. 下一步优先级

1. 修复/解释 `upsample_level=1` 的 0.446°C 重构误差。
2. 生成 causal expanding-window HHSA 特征。
3. 在每个窗口按 2–7 年频带匹配 carriers，避免固定 IMF 编号漂移。
4. 直接保存 normalized FM carrier 和 envelope，不再事后重复归一化。
5. 对 envelope forecast 增加 phase、slope、curvature 和 multi-horizon loss。
6. 增加 peak envelope/peak Niño3.4 的事件目标。
7. 做 real/shuffled/no-LF、red-noise surrogate 和 leave-one-event-out 检验。
8. 引入 WWV、热跃层、风应力等变量，检验它们是否改善 envelope 的未来
   演化预测。

## 15. 当前结论

1. Niño3.4 的主要 ENSO carriers 是约 2.31 和 4.31 年的两个 IMF。
2. 对应 envelope 中存在约 9.66、11.71、18.03 和 25.33 年的主要调制。
3. 把 HHSA 当普通表格特征或 learned correction gate，均未得到稳定 AM
   增益。
4. 直接预测真实 envelope 和 normalized FM carrier，再强制相乘，显著改善
   了预测表现。
5. Real LF-AM 在测试集 6、12 月 lead 有小幅增益，但验证集没有一致重复。
6. 2026 峰值仍被显著低估；oracle 诊断表明 envelope forecast 是主要瓶颈。
7. 该完整结构可能具有方法创新性，但在 causal HHSA、显著性、重构误差和
   独立数据验证完成前，不能声称已经证明 AM 驱动 2026，也不能声称首创。
