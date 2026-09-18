# Spatial HHSA–ConvLSTM/U-Net: Short Note

## 中文摘要

目前的 Niño3.4 是单点时间序列，因此继续使用 XGBoost 作为标量基线。
如果以后使用二维 SST、风场或热含量场，就应使用 ConvLSTM/U-Net。

正确的 HHSA 空间模型是：

~~~text
空间场 → CNN/U-Net → ConvLSTM
       → 分别预测 AM envelope 和 FM carrier
       → 强制执行 envelope × carrier
       → 未来 SSTA 场 → Niño3.4
~~~

第一版建议先做 EOF/PCA，把空间场压缩成 10–20 个 ENSO 相关系数，再对这些
系数做 HHSA。不要把单点 Niño3.4 人为变成二维图像，也不要直接照搬
FourCastNet/Pangu-Weather 的模型规模。

最小实施顺序是：

1. 保留当前 HHSA envelope × normalized carrier XGBoost；
2. 加入空间 SST、WWV、温跃层、风应力和热收支；
3. EOF/PCA 压缩空间场；
4. 对 EOF 系数做 HHSA；
5. 建立小型 ConvLSTM-U-Net；
6. 比较 raw、no-AM、shuffled-AM 和 real-AM。

## Main conclusion

Current Niño3.4 is a single time series, so XGBoost is still a useful baseline.
For gridded SST, wind, heat-content, or SSTA fields, use ConvLSTM/U-Net.

Do not reshape a scalar Niño3.4 series into a fake image.

## Which model for which data?

| Model | Best use |
|---|---|
| XGBoost | Single-index Niño3.4 and a few scalar variables |
| ConvLSTM | Time-evolving spatial fields |
| U-Net | Multi-scale spatial patterns |
| ConvLSTM-U-Net | Spatial structure plus temporal evolution |

## Recommended architecture

~~~text
Past 24–48 months of SST, WWV, thermocline, wind, and heat budget
                         ↓
                    CNN / U-Net
                         ↓
                      ConvLSTM
                         ↓
          ┌─────────────────────────────┐
          │                             │
      AM envelope head             FM carrier head
      positive A                   phase / normalized F
          └──────────────┬──────────────┘
                         ↓
                 A × F reconstruction
                         ↓
                  Future SSTA field
                         ↓
                    Niño3.4
~~~

The physical signal structure is:

$$
SSTA_k(t,x,y)=A_k(t,x,y)F_k(t,x,y).
$$

The network should predict A and F separately and force their product. It should
not be asked to discover this multiplication from ordinary feature columns.

## How to add HHSA

### Recommended first version

~~~text
Spatial SSTA field
      ↓
EOF/PCA (10–20 ENSO-related coefficients)
      ↓
HHSA on the coefficients
      ↓
Predict AM and FM latent states
      ↓
ConvLSTM or a small temporal model
~~~

This is computationally manageable and makes IMF tracking easier.

Applying HHSA independently to every grid point is a later, much larger
experiment because mode matching, land masks, missing values, and endpoint
effects become difficult.

## Network outputs

1. Envelope head: use softplus so the predicted envelope is positive.
2. Carrier head: predict sine and cosine of phase instead of raw phase.
3. Residual head: predict the part not represented by selected carriers.

Reconstruction:

~~~python
SSTA_future = A_future * F_future + residual_future
~~~

## Minimum experiment plan

1. Keep the current scalar HHSA envelope × normalized-carrier XGBoost as a
   baseline.
2. Add monthly gridded Pacific SSTA, WWV/heat content, thermocline depth, wind
   stress, and heat-budget variables.
3. Compress the fields using EOF/PCA.
4. Apply HHSA to the selected EOF coefficients.
5. Train a small ConvLSTM-U-Net with separate A and F heads.
6. Compare raw, no-AM, shuffled-AM, and real-AM versions.

Required comparisons:

~~~text
A. Raw spatial field → Niño3.4
B. Raw spatial field → future SSTA field
C. HHSA states → Niño3.4
D. HHSA envelope × carrier → future SSTA field
~~~

Only a stable improvement of D over A–C supports a useful HHSA contribution.

## What not to do

- Do not use ConvLSTM on scalar Niño3.4 by inventing spatial dimensions.
- Do not copy the scale of FourCastNet or Pangu-Weather with 943 monthly
  observations.
- Do not flatten every HHSA variable and call that learned AM dynamics.
- Do not use full-record offline HHSA as a formal causal forecast.
- Do not judge strong-event prediction only by average RMSE.

## Final recommendation

Use:

~~~text
Current stage: XGBoost + HHSA envelope × normalized FM carrier
Next stage: EOF/PCA + HHSA latent states + small ConvLSTM-U-Net
~~~

The second stage is the correct route for spatial climate AI; the first stage
remains the necessary scalar benchmark.
