### SAMURAI Memory Bank 多样性选择（基于格拉姆行列式/DPP）的技术方案

#### 1) 背景与目标

- **目标**：将 `docs/thor.md` 中“基于格拉姆行列式的模板多样性最大化”思想引入 SAMURAI，在从 `valid_indices` 选择 Memory Bank 帧时，不再使用“最近 K 帧”，而是选择一个同时满足高质量与高多样性的子集，提高遮挡、形变和视角变化下的鲁棒性。
- **不变项**：
  - Memory Bank 总容量维持 `num_maskmem = 7`（1 个 cond + 6 个 non-cond）。
  - 质量门控维持现有三重阈值：`memory_bank_iou_threshold`、`memory_bank_obj_score_threshold`、`memory_bank_kf_score_threshold`。
  - 与原始 SAM 2 完全兼容，提供回退为“最近帧”的开关。

参考现状（待替换的“最近 K 帧”策略位于 `sam2/sam2/modeling/sam2_base.py::_prepare_memory_conditioned_features()` 的 SAMURAI 分支）：

```python
# 现状概要：从 valid_indices 末尾取最近 6 帧
for t_pos in range(1, self.num_maskmem):
    idx = t_pos - self.num_maskmem   # -6, ..., -1
    if idx < -len(valid_indices):
        continue
    out = output_dict["non_cond_frame_outputs"].get(valid_indices[idx], None)
    if out is None:
        out = unselected_cond_outputs.get(valid_indices[idx], None)
    t_pos_and_prevs.append((t_pos, out))
```

#### 2) 设计概述：质量×多样性联合最优

- **候选特征 f_i**：优先选用 `obj_ptr`（目标指针，1×D 向量，轻量稳定）；备选为对 `maskmem_features` 在目标掩码内做全局池化的向量。
- **相似度 s_ij**：对特征 L2 归一化后使用余弦相似度 `s_ij = ⟨f_i, f_j⟩`。
- **质量权重 q_i**：由 `IoU / obj_score / KF` 归一化融合，示例：`q_i = sigmoid(a·IoU_i + b·Obj_i + c·KF_i)` 或乘积归一。
- **核矩阵 K（推荐，DPP 风格）**：`K_ij = q_i · s_ij · q_j`。最大化 `det(K_S)` 在几何上鼓励“高质量 × 低冗余”。
- 轻量替代（Gram）：最大化 `det(F_S^T F_S)`，其中 `F` 为单位范数特征堆叠矩阵。
- **时间连续性保障**：强制包含最近一帧（`frame_idx-1`）作为锚点。

#### 3) 算法细节（贪心 logdet 近似）

输入：候选集 `C = valid_indices`（≤15），目标数量 `K = num_maskmem-1 = 6`。

步骤：
1. 特征构造：`f_i = L2_normalize(feat(i))`，相似度 `S_ij = ⟨f_i, f_j⟩`。
2. 质量融合：对 `IoU/Obj/KF` 做缩放/归一，得到 `q_i`。
3. 组合核：`K = diag(q) · S · diag(q)`。
4. 初始化：`S = { last_idx }`（如存在）。
5. 迭代：在 `C \ S` 中选择能最大化 `Δ = log det(K_{S∪{c}} + εI) - log det(K_S + εI)` 的 `c`，直至 |S|=K。
6. 输出：将 `S` 按时间升序映射到 `t_pos=1..K`。

复杂度：在 |C|≤15、K=6 下，直接 `logdet` 足够快；可用 Cholesky/Schur 增量更新降复杂度。近似方案可用 Gram-Schmidt 残差范数作为增益代理。

伪代码（接口与回退）：

```python
def select_diverse_frames(valid_indices, output_dict, K,
                          feature_source="obj_ptr",
                          quality_weighting=True,
                          ensure_last=True, eps=1e-6):
    feats, quals, ids = [], [], []
    for i in valid_indices:
        out = output_dict["non_cond_frame_outputs"].get(i)
        if out is None:
            continue
        f = out.get("obj_ptr") if feature_source == "obj_ptr" else pooled_maskmem(out)
        if f is None:
            continue
        feats.append(l2_normalize(f))
        quals.append(quality_score(out) if quality_weighting else 1.0)
        ids.append(i)

    # 回退：候选不足或缺字段，保持现状（最近 K 帧）
    if len(ids) <= K:
        return ids[-K:]

    S = cosine_similarity_matrix(feats)
    Q = diag(quals)
    Kmat = Q @ S @ Q

    selected = []
    if ensure_last and ids[-1] not in selected:
        selected.append(ids[-1])

    while len(selected) < K:
        best_idx, best_gain = None, -float("inf")
        for idx in ids:
            if idx in selected:
                continue
            cand = selected + [idx]
            gain = logdet(Kmat[cand][:, cand] + eps * I) - logdet(Kmat[selected][:, selected] + eps * I)
            if gain > best_gain:
                best_gain, best_idx = gain, idx
        selected.append(best_idx)

    return sorted(selected)
```

#### 4) 工程集成点（最小侵入改动）

- 位置：`sam2/sam2/modeling/sam2_base.py::_prepare_memory_conditioned_features()` 的 SAMURAI 分支中，构建完 `valid_indices` 后将“最近 K 帧”替换为“多样性选择”。
- 仅新增一个私有选择函数（或 utils 函数），其输入为 `valid_indices/output_dict/K`，输出 `selected_indices`。其余逻辑（`t_pos_and_prevs` 构造、非 cond/cond 对齐等）保持不变。
- 始终包含最近一帧（若存在），保持时间连续性与恢复能力。
- 缺特征时回退：保持稳定行为与向后兼容。

#### 5) 新增配置（需在 `configs/samurai/*.yaml` 同步）

- `memory_bank_selection_mode: {recent, diverse}`（默认：`diverse` 当 `samurai_mode=true`，否则 `recent`）
- `memory_bank_feature_source: {obj_ptr, maskmem_avgpool}`（默认：`obj_ptr`）
- `memory_bank_diversity_metric: {gram_det, residual}`（默认：`gram_det`）
- `memory_bank_quality_weighting: true|false`（默认：`true`）
- `memory_bank_anchor_last: true|false`（默认：`true`）
- `memory_bank_diversity_eps: 1e-6`

保持现有：`num_maskmem`、`memory_bank_*_threshold`、`max_obj_ptrs_in_encoder`。

#### 6) 数值与性能注意事项

- 特征 L2 归一化；`K+εI` 防奇异；质量归一化避免单源支配。
- 首选 `obj_ptr` 作为特征（小向量，无需搬运大张量，最省时省显存）。
- 候选规模固定 ≤15，贪心 logdet 的时间开销微小；如需极致优化可换残差近似。

#### 7) 调试与评测计划

- 日志：打印 `valid_indices`、(IoU/Obj/KF)、相似度均值/方差、被选子集、增量 `logdet`。
- 评测：DAVIS/YouTube-VOS + 小目标/快速运动/长序列子集；指标含 J&F、重捕获时延、漂移率、平均帧时延。
- 消融：`feature_source`、`quality_weighting`、`anchor_last`、`metric={gram_det,residual}`。

#### 8) 风险与回退

- 风险：外观稳定时收益有限；错误高质量帧可能影响多样性选择。
- 缓解：保留质量门控、强制包含最近帧、随时回退 `selection_mode=recent`。

#### 9) 变更清单与工期

- 代码：在 `sam2_base.py` 添加选择函数及调用（约 50–120 行）。
- 配置：为 `samurai` 配置族新增 5–6 个轻量超参（半天内完成同步）。
- 文档：已新增本方案；更新 `MEMORY_BANK_ARCHITECTURE.md` 的流程图对比。
- 预算工期：开发 ~0.5 天；小规模评测与消融 ~0.5–1 天。



