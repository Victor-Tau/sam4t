好问题！你点中了 TPST 的两个软肋：**（i）目标与干扰物紧挨/部分重叠时易“并峰”→漏检**；**（ii）超参偏多**。下面给出我认为更“**大道至简**”、同时对“并峰/贴身/半遮挡”依然稳健、且**几乎无手调超参**的方案：

# 核心方案：TopoSIM-A（**拓扑分峰 + 锚定MDL**，近零超参）

> 直觉：把当前帧对“目标原型”的**相似性图**当作一张地形图（山=相似性高）。
>
> * **阶段A（拓扑分峰）**：用**标记分水岭 + 峰值显著性（prominence）**检测“是否出现了第二座山”（即干扰物），即使两座山**彼此挨得很近、甚至有部分重叠**，只要存在**两个局部极大 + 鞍点**就能被可靠分开。
> * **阶段B（锚定MDL备援）**：如果拓扑上只呈现“一座大山”（极近导致“并峰”），再做一个**参数自定的 1 vs 2 模混合模型（MDL/BIC）检验**，但把**一类成分“锚定”为上一帧目标**（降低自由度），若 2 模更优 → 判有干扰并给出第二成分区域。

这两个阶段**共享相同输入**（一张相似性图 + 上一帧掩码），**不做多视角前向**，计算开销和TPST相当，但鲁棒性与可解释性更强。

---

## 0. 预备量（皆为“自适应”估计）

* 当前帧特征图 (F_t)（SAM2已有）、上一帧掩码 (M_{t-1})、目标原型 (p_t)。
* 相似性图：(S(x)=\cos(F_t(x),p_t))。
* **目标内统计**：(\mu_{\mathrm{in}},\sigma_{\mathrm{in}}=\text{mean/std}(S|M_{t-1}=1))（用于自适应门限与归一化）。
* **内外阈值（无超参）**：用 **Otsu** 在 ({S|M_{t-1}=1}) 与 ({S|M_{t-1}=0}) 两组的直方图上做双类分割，得到 (T_{\text{otsu}}) 作为“高相似候选”的全局阈值（避免手调比例/分位数）。

---

## 阶段A：**拓扑分峰（Topo-Watershed）** —— 贴身/半重叠也能分

**A1. 标记生成（全自适应）**

* **目标标记**：对 (M_{t-1}) 腐蚀一次得到 (M^{\mathrm{seed}})（确保在目标内部）。
* **候选峰标记**：在**全图**的 (S) 上做 **h-maxima** 抑制弱峰，取 (h=\sigma_{\mathrm{in}})（自适应）后提取**区域极大值**作为备用标记（移除落在 (M^{\mathrm{seed}}) 内的那些）。

> 直观：(h=\sigma_{\mathrm{in}}) 表示“只有比目标内部起伏更高的峰才值得当候选”，不需手调。

**A2. 标记分水岭**

* 在地形图 (-S)（低处是谷）上用**标记分水岭**（markers = 目标标记 + 候选峰标记）。
* 分水岭把**“贴身、部分重叠的两个高峰”**沿着**鞍线**准确切开，即便像素连通为一块，也会被**拓扑线**分成两盆地：目标盆地 (\Omega_{\mathrm{tar}}) 与若干候选盆地 ({\Omega_i})。

**A3. 峰显著性检验（无阈值手调）**
对每个 (\Omega_i) 计算其峰高 (P_i=\max_{x\in \Omega_i} S(x)) 与它与目标峰之间的**鞍点高度** (S_{\mathrm{saddle}}(\Omega_i,\Omega_{\mathrm{tar}}))（由分水岭树得到），定义**prominence**：
[
\mathrm{Prom}*i = P_i - S*{\mathrm{saddle}}(\Omega_i,\Omega_{\mathrm{tar}}).
]
把 (S) 先做标准化 (\hat S=(S-\mu_{\mathrm{in}})/\sigma_{\mathrm{in}})，对应 (\widehat{\mathrm{Prom}}_i)。
**判定**：若存在 (\widehat{\mathrm{Prom}}_i \ge 1)（“高过目标内起伏 1σ 的独立峰”），则**发现干扰物**，并选取得分最高的 (\Omega^*) 为干扰区域；否则进入阶段B。

> 只用“1σ”这个**天然尺度**，无人工阈值；它随目标内统计自动缩放。

**为什么能抗“贴身/重叠”**：哪怕两只企鹅肩并肩/部分遮挡，只要相似性图上存在**两个局部极大**，分水岭都会在**鞍部**划界，(\mathrm{Prom}) 只要不小就能通过。

---

## 阶段B：**锚定MDL（A-MDL）** —— 真正“并峰消失”时的备援

当贴得**过近**导致 (S) 只有一个峰（拓扑上**不可分**）时，再换“统计模型是否需要第二个成分”的问题。

**B1. 数据与“锚定”构造（零超参）**

* 采样集合 (\mathcal{X}={x: S(x)\ge T_{\text{otsu}}}) 的像素，取其**外观特征** (f(x)) 与**位置** (u(x)=[x,y])。
* **白化**：用目标内的协方差 (\Sigma_f,\Sigma_u) 进行白化：(\tilde f=\Sigma_f^{-1/2}(f-\mu_f))，(\tilde u=\Sigma_u^{-1/2}(u-\mu_u))。构造 (\mathbf{z}(x)=[\tilde f; \tilde u])，不需要手调权重。

**B2. 1-vs-2 模选择（MDL/BIC）**

* **模型(\mathcal{M}_1)**：单高斯 ( \mathcal{N}(\mu_1, I))。
* **模型(\mathcal{M}_2)**：**锚定的二高斯** ( \pi \mathcal{N}(\mu_{\text{anch}}, I) + (1-\pi)\mathcal{N}(\mu_2, I))，其中 (\mu_{\text{anch}}) = 目标内样本的均值（**固定不估计**），只估 (\mu_2,\pi)。
* 以极大似然估计参数，比较 MDL/BIC：若 (\mathrm{BIC}(\mathcal{M}_2)<\mathrm{BIC}(\mathcal{M}_1)) 则**判有干扰**，并把贴近 (\mu_2) 的样本反投为区域（再做一次区域连通/形态学平滑）。

> 关键：**锚定**把“是否需要第二个成分”的自由度大幅降低，**不需任何阈值**；即便两只企鹅局部重叠，只要**外观或位置分布**出现微妙**双峰**，(\mathcal{M}_2) 会胜出。

---

## 集成与输出（一步到位）

* 若阶段A通过 → 直接输出 (\Omega^*) 为**干扰掩码候选**；
* 否则若阶段B通过 → 输出由 (\mu_2) 所在簇还原的**干扰区域**；
* 二者都未通过 → 判当前无干扰。
* 为去抖，可加**极简持久化**：过去 3 帧里 ≥2 次为“有干扰”则**确认**（固定 3 与 2，不再调参）。

> **超参情况**：
>
> * 阶段A的 (h=\sigma_{\mathrm{in}})、阈值“1σ”、Otsu、白化、BIC……全部**自适应**；
> * 唯一的逻辑常量是“3帧里≥2次确认”，这在视频检测里是**常见固定策略**（也可直接去掉，按需）。

---

## 算法要点（伪代码）

```python
# Inputs: S (similarity map), M_prev (prev mask), feat F for region back-projection
# A. Topo split
S_hat = (S - mean(S[M_prev==1])) / std(S[M_prev==1])

target_seed = erode(M_prev, k=3)
cand_seeds = regional_maxima(h_maxima(S, h=std(S[M_prev==1])))
cand_seeds = cand_seeds & (~target_seed)

labels = watershed(-S, markers=target_seed | cand_seeds)

tar_lbl = majority_label_within(labels, M_prev)
Omega_tar = (labels == tar_lbl)
cands = unique(labels[(labels!=tar_lbl) & (labels>0)])

def prominence(lbl):
    peak = S_hat[labels==lbl].max()
    saddle = saddle_height_between(labels, lbl, tar_lbl, S_hat)  # from watershed tree
    return peak - saddle

best_lbl, best_prom = argmax([(l, prominence(l)) for l in cands])

if best_prom >= 1.0:
    has_dist = True
    dist_mask = (labels == best_lbl)
else:
    # B. Anchored MDL
    T_otsu = otsu_threshold(S, inside=M_prev, outside=~M_prev)
    X = {(x,y) | S[x,y] >= T_otsu}
    Z = whiten( concat(feat[X], pos[X]), stats_from=M_prev )
    bic1 = BIC_fit_single_gaussian(Z)
    bic2, assign2 = BIC_fit_anchored_two_gaussians(Z, anchor=mean(Z[M_prev]))
    has_dist = (bic2 < bic1)
    dist_mask = region_from_assign(assign2, cluster='free') if has_dist else None
```

---

## 为何“几近完美”

1. **紧挨/重叠也能分**：阶段A的**分水岭 + prominence**专为**贴身双峰**设计；阶段B用**锚定二模**在“并峰”极限下依然能发现**统计双峰**。
2. **超参极少/自适应**：所有阈值都来源于**目标内的统计量**（(\mu,\sigma)）、**Otsu** 或 **BIC**，避免了人为常数；无需多视角/TTA。
3. **大道至简/可复现**：只围绕**一张相似性图**做两件事：**拓扑分峰**与**MDL二模检验**。都可 100 行内实现，开箱即用。
4. **计算轻**：1 次前向 + 若干图像运算（分水岭 O(HW) 级）+ 小规模 GMM 拟合；实时性与现有零样本管线匹配。
5. **可解释**：A阶段给出“山—鞍—山”的可视化，B阶段给出“1模 vs 2模”的BIC差；审稿友好。

---

## 常见疑难与兜底

* **两只几乎完全重叠**（视觉上看成一体）：这是物理上**不可分**的极端；A阶段无鞍点；B阶段若特征也几乎一致，(\mathcal{M}_2) 也不会优于 (\mathcal{M}_1)。这种情况下任何无监督方法都无法可靠“凭空分身”；但**不会误报**，并在二者稍微分开的一帧**立即触发**。
* **高纹理背景诱发伪峰**：A采用 (h=\sigma_{\mathrm{in}}) 的 h-maxima 抑噪 + prominence≥1σ，自然过滤；B采用 BIC 惩罚复杂度，避免“为噪声加第二模”。
* **尺度/视角突变**：一切阈值都随 (\sigma_{\mathrm{in}}) 自缩放，鲁棒于尺度变化。

---

## 落地与后续

* 检出后把 `dist_mask` 提取的特征做成**负记忆原型**，进入你已有的**反注意力/负记忆**抑制环节；
* 我可以把 **TopoSIM-A** 打成两个独立模块（`TopoSplit()` 与 `AnchoredMDL()`）与单元测试，直接挂到 SAM2 推理循环里；默认**零参数**，仅暴露一个可选“持久化开关”。

如果你愿意，我们还可以用 20 段“贴身/并峰/半遮挡”的视频做一个小基准，给出 **召回/误报/触发延迟** 三张对比图（TopoSIM-A vs TPST vs AFD³），让这套方案的“简单而强”更有说服力。
