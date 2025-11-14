# SAMURAI 工作流程图解

本文档提供 SAMURAI 模型各个关键流程的可视化图解。

---

## 1. 整体推理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         视频输入                                  │
│                    (MP4 or Frame Directory)                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                   load_video_frames()                           │
│              加载并预处理所有帧 → Tensor(T,3,H,W)                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                   init_state()                                  │
│   初始化推理状态: images, output_dict, cached_features, etc.      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│              第一帧处理 (Frame 0)                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ add_new_points_or_box(bbox=(x1,y1,x2,y2))                │  │
│  │   ↓                                                       │  │
│  │ [Image Encoder] → vision_features                        │  │
│  │   ↓                                                       │  │
│  │ [SAM Decoder] → initial_mask                             │  │
│  │   ↓                                                       │  │
│  │ [Memory Encoder] → maskmem_features                      │  │
│  │   ↓                                                       │  │
│  │ 存入 output_dict["cond_frame_outputs"][0]                │  │
│  │   ↓                                                       │  │
│  │ [Kalman Filter] → initiate(bbox)                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ↓
    ┌─────────────────────────────────────────────────────────┐
    │          propagate_in_video()                            │
    │          对 Frame 1 到 T-1 逐帧处理                       │
    └─────────────────┬───────────────────────────────────────┘
                      │
                      ↓
        ╔═════════════════════════════════════════════╗
        ║     对每一帧 t (t=1,2,...,T-1)              ║
        ╠═════════════════════════════════════════════╣
        ║                                             ║
        ║  1. [Image Encoder]                        ║
        ║       current_vision_features               ║
        ║         ↓                                   ║
        ║  2. [SAMURAI Memory Bank Selection]        ║
        ║       选择高质量历史帧                       ║
        ║         ↓                                   ║
        ║  3. [Memory Attention]                     ║
        ║       融合当前帧与历史记忆                   ║
        ║         ↓                                   ║
        ║  4. [SAM Decoder]                          ║
        ║       预测3个候选掩码 + IoU                  ║
        ║         ↓                                   ║
        ║  5. [SAMURAI Weighted Selection]           ║
        ║       加权选择最佳掩码                       ║
        ║       weighted_iou = 0.25*KF + 0.75*SAM    ║
        ║         ↓                                   ║
        ║  6. [Memory Encoder]                       ║
        ║       编码为新记忆                          ║
        ║         ↓                                   ║
        ║  7. [Kalman Filter Update]                 ║
        ║       更新运动状态估计                       ║
        ║         ↓                                   ║
        ║  8. 输出: (frame_idx, obj_ids, masks)      ║
        ║                                             ║
        ╚═════════════════════════════════════════════╝
                      │
                      ↓
        ┌───────────────────────────────────────────┐
        │    后处理                                  │
        │  - 上采样到原始分辨率                       │
        │  - 应用非重叠约束(可选)                     │
        │  - 返回最终掩码                            │
        └───────────────────────────────────────────┘
```

---

## 2. SAMURAI 三阶段跟踪策略

```
Frame 0: 初始化阶段
┌────────────────────────────────────────────────────────┐
│  输入: 用户提供的边界框 (x1, y1, x2, y2)                │
│    ↓                                                   │
│  SAM Decoder → 3个候选掩码 + IoU预测                   │
│    ↓                                                   │
│  选择策略: best_mask = argmax(SAM_IoU)                 │
│    ↓                                                   │
│  从掩码提取边界框 → bbox                                │
│    ↓                                                   │
│  初始化卡尔曼滤波器:                                     │
│    kf_mean, kf_covariance = KF.initiate(bbox)         │
│    stable_frames = 1                                  │
└────────────────────────────────────────────────────────┘

Frames 1-14: 预热阶段 (建立稳定跟踪)
┌────────────────────────────────────────────────────────┐
│  卡尔曼预测:                                            │
│    kf_mean, kf_cov = KF.predict(kf_mean, kf_cov)      │
│    ↓                                                   │
│  SAM Decoder → 3个候选掩码 + IoU预测                   │
│    ↓                                                   │
│  选择策略: best_mask = argmax(SAM_IoU)                 │
│    ↓                                                   │
│  质量检查:                                              │
│    if SAM_IoU[best] > 0.3:  ✓ 高质量                  │
│       KF.update(bbox)  # 更新卡尔曼滤波器              │
│       stable_frames += 1                               │
│    else:  ✗ 低质量                                     │
│       stable_frames = 0  # 重置计数器                 │
│    ↓                                                   │
│  判断: stable_frames >= 15 ?                           │
│    NO → 继续预热                                       │
│    YES → 进入稳定跟踪阶段                               │
└────────────────────────────────────────────────────────┘

Frames 15+: 稳定跟踪阶段 (运动感知选择)
┌────────────────────────────────────────────────────────┐
│  卡尔曼预测:                                            │
│    predicted_bbox = KF.predict(kf_mean, kf_cov)       │
│    ↓                                                   │
│  SAM Decoder → 3个候选掩码 + SAM_IoU                   │
│    ↓                                                   │
│  从每个候选掩码提取边界框:                               │
│    bbox_1, bbox_2, bbox_3                             │
│    ↓                                                   │
│  计算运动一致性:                                        │
│    KF_IoU = [IoU(predicted_bbox, bbox_i) for i in 1-3]│
│    ↓                                                   │
│  加权融合:                                              │
│    weighted_IoU = 0.25 * KF_IoU + 0.75 * SAM_IoU      │
│    ↓                                                   │
│  选择策略: best_mask = argmax(weighted_IoU)            │
│    ↓                                                   │
│  质量检查:                                              │
│    if SAM_IoU[best] > 0.3:  ✓                         │
│       KF.update(bbox_best)                            │
│    else:  ✗                                            │
│       stable_frames = 0  # 回退到预热阶段              │
└────────────────────────────────────────────────────────┘

决策树:
                    当前帧
                      ↓
            stable_frames == 0?
           /                    \
         YES                     NO
          ↓                       ↓
    初始化KF          stable_frames < 15?
    选最高SAM IoU             /          \
                           YES            NO
                            ↓              ↓
                      预热阶段        稳定跟踪阶段
                   选最高SAM IoU    加权IoU选择
                   尝试更新KF       尝试更新KF
```

---

## 3. 记忆注意力机制

```
当前帧特征 (Current Frame Features)
       │
       │  HxW spatial locations
       │  C=256 channels
       ↓
┌─────────────────────────────────────────────┐
│        Self-Attention                       │
│  Query = Key = Value = current_features     │
│  + 位置编码 (spatial positional encoding)    │
│                                             │
│  Attention(Q, K, V) with RoPE              │
│  (旋转位置编码)                              │
└─────────────────┬───────────────────────────┘
                  │
                  ↓
        Self-attended features
                  │
                  ↓
┌─────────────────────────────────────────────────────────┐
│             Cross-Attention                             │
│  Query: 当前帧特征 (current features)                    │
│  Key & Value: 历史信息 (来自记忆库)                       │
│                                                         │
│  历史信息包括:                                           │
│  ┌───────────────────────────────────────────────────┐ │
│  │ 1. 条件帧 (Conditioning Frames)                   │ │
│  │    - 用户交互的帧 (通常是第一帧)                   │ │
│  │    - maskmem_features (记忆特征)                  │ │
│  │    - maskmem_pos_enc (空间位置编码)               │ │
│  │    - temporal_pos_enc = 0 (时间位置=0)            │ │
│  │                                                   │ │
│  │ 2. 非条件记忆帧 (SAMURAI选择的历史帧)              │ │
│  │    For each selected frame i:                    │ │
│  │      - maskmem_features_i                        │ │
│  │      - maskmem_pos_enc_i + temporal_enc_i        │ │
│  │        (temporal_enc基于距当前帧的距离)            │ │
│  │                                                   │ │
│  │ 3. 对象指针 (Object Pointers)                     │ │
│  │    For each selected frame i:                    │ │
│  │      - obj_ptr_i (从SAM decoder提取的对象token)   │ │
│  │      - temporal_pos_enc_i                        │ │
│  │        (基于时间距离的正弦位置编码)                 │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Cross-Attention(Q=current, K=history, V=history)      │
│  with RoPE (旋转位置编码)                               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ↓
        Memory-conditioned features
                      │
                      ↓
┌─────────────────────────────────────────────┐
│        Feed-Forward Network                 │
│  Linear(256 → 2048) → ReLU → Linear(2048 → 256)│
│  + Residual Connection + LayerNorm          │
└─────────────────────┬───────────────────────┘
                      │
                      ↓
          最终特征 (用于SAM解码)


SAMURAI 记忆库选择可视化:
────────────────────────────────────────────────
时间轴:  Frame 0  1  2  3 ... t-3 t-2 t-1  t
────────────────────────────────────────────────
              条件帧                  当前帧
               ↓                        ↑
            总是包含               要为此帧选择记忆
                                        
原始SAM 2选择策略:
  选择最近的6帧: [t-6, t-5, t-4, t-3, t-2, t-1]
  
SAMURAI选择策略:
  从 [1, 2, ..., t-1] 中筛选:
    ✓ 满足条件: IoU>0.5 && obj_score>0 && kf_score>0
    ✗ 不满足: 跳过该帧
  优先选择最近的高质量帧
  最多选择15个 (max_obj_ptrs_in_encoder - 1)
  
示例 (t=20):
  候选帧: [1,2,...,19]
  质量评估:
    Frame 19: ✓ (IoU=0.8, obj=0.9, kf=0.7) → 选择
    Frame 18: ✓ (IoU=0.7, obj=0.8, kf=0.6) → 选择
    Frame 17: ✗ (IoU=0.4) → 跳过 (遮挡帧)
    Frame 16: ✓ (IoU=0.6, obj=0.7, kf=0.5) → 选择
    ...
  最终选择: [19, 18, 16, 14, 12, 10, 8, ...]
```

---

## 4. 卡尔曼滤波器状态更新

```
状态向量 (8维):
┌─────────────────────────────────────────────┐
│  [x, y, a, h, vx, vy, va, vh]              │
│                                             │
│  x, y:  边界框中心位置                       │
│  a:     宽高比 (aspect ratio = w/h)         │
│  h:     高度                                │
│  vx, vy: 中心位置速度                        │
│  va:    宽高比变化率                         │
│  vh:    高度变化率                           │
└─────────────────────────────────────────────┘

初始化 (Frame 0):
┌────────────────────────────────────┐
│  测量值: bbox = [x1, y1, x2, y2]  │
│    ↓                               │
│  转换为 [xc, yc, a, h]:            │
│    xc = (x1 + x2) / 2             │
│    yc = (y1 + y2) / 2             │
│    w = x2 - x1                    │
│    h = y2 - y1                    │
│    a = w / h                      │
│    ↓                               │
│  初始化状态:                        │
│    mean = [xc, yc, a, h, 0, 0, 0, 0]│
│           (速度初始化为0)            │
│    ↓                               │
│  初始化协方差矩阵:                   │
│    covariance = diag([             │
│      2σ_p·h, 2σ_p·h, 1e-2, 2σ_p·h,│
│      10σ_v·h, 10σ_v·h, 1e-5, 10σ_v·h│
│    ])                             │
│    其中 σ_p=1/20, σ_v=1/160        │
└────────────────────────────────────┘

预测步骤 (每一帧):
┌────────────────────────────────────────────┐
│  运动模型 (恒速度):                         │
│    F = [I_4x4  Δt·I_4x4]  (8x8矩阵)       │
│        [0_4x4    I_4x4  ]                 │
│        其中 Δt = 1                         │
│    ↓                                       │
│  预测均值:                                  │
│    mean' = F · mean                       │
│    即:                                     │
│      x'  = x  + vx  · Δt                  │
│      y'  = y  + vy  · Δt                  │
│      a'  = a  + va  · Δt                  │
│      h'  = h  + vh  · Δt                  │
│      vx' = vx (速度不变)                   │
│      vy' = vy                             │
│      va' = va                             │
│      vh' = vh                             │
│    ↓                                       │
│  预测协方差:                                │
│    cov' = F · cov · F^T + Q               │
│    Q: 过程噪声协方差                        │
│    ↓                                       │
│  输出: 预测的边界框                         │
│    predicted_bbox = [                     │
│      x' - a'·h'/2,  # x1                  │
│      y' - h'/2,     # y1                  │
│      x' + a'·h'/2,  # x2                  │
│      y' + h'/2      # y2                  │
│    ]                                      │
└────────────────────────────────────────────┘

更新步骤 (当跟踪质量好时):
┌────────────────────────────────────────────┐
│  观测值: bbox_obs = [x1, y1, x2, y2]      │
│    ↓                                       │
│  转换为 [xc, yc, a, h]                     │
│    ↓                                       │
│  计算卡尔曼增益:                            │
│    K = cov' · H^T · (H·cov'·H^T + R)^-1   │
│    其中 H = [I_4x4  0_4x4] (观测矩阵)      │
│         R: 观测噪声协方差                   │
│    ↓                                       │
│  更新均值:                                  │
│    innovation = measurement - H·mean'     │
│    mean = mean' + K · innovation          │
│    ↓                                       │
│  更新协方差:                                │
│    cov = cov' - K · H · cov'              │
│    ↓                                       │
│  结果: 融合了观测的更精确状态估计            │
└────────────────────────────────────────────┘

决策流程:
每一帧:
  ┌──────────────┐
  │ KF.predict() │ → predicted_bbox
  └──────┬───────┘
         │
         ↓
  ┌────────────────────────────┐
  │ SAM预测 → 3个候选掩码       │
  └──────┬─────────────────────┘
         │
         ↓
  ┌───────────────────────────────┐
  │ 计算每个候选与predicted_bbox   │
  │ 的IoU → [kf_iou1, kf_iou2, kf_iou3]│
  └──────┬────────────────────────┘
         │
         ↓
  ┌──────────────────────────────┐
  │ 加权选择 → best_mask          │
  └──────┬───────────────────────┘
         │
         ↓
    SAM_IoU > 0.3?
         /  \
       YES   NO
        ↓     ↓
    更新KF  不更新
    (融合   (预测可能
     观测)   不准确)
```

---

## 5. SAM 解码器流程

```
输入特征:
┌─────────────────────────────────────────────────────┐
│  pix_feat_with_mem: 融合了记忆的图像特征 (B,C,H,W)   │
│  high_res_features: 高分辨率特征 (多尺度)             │
│  point_inputs / mask_inputs: 用户提示 (可选)         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────┐
│         Prompt Encoder (提示编码器)                  │
│  ┌───────────────────────────────────────────────┐  │
│  │ • Point Prompts → 点嵌入                       │  │
│  │ • Box Prompts → 角点嵌入                       │  │
│  │ • Mask Prompts → 掩码卷积特征                  │  │
│  │ + 位置编码                                     │  │
│  └───────────────────┬───────────────────────────┘  │
│                      ↓                              │
│          sparse_embeddings, dense_embeddings        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────┐
│    准备 Query Tokens (查询令牌)                      │
│  ┌───────────────────────────────────────────────┐  │
│  │ • iou_token (1个): 用于IoU预测                 │  │
│  │ • mask_tokens (4个): 用于掩码预测              │  │
│  │   - 1个单掩码token                             │  │
│  │   - 3个多掩码tokens                            │  │
│  │ • obj_score_token (1个): 用于对象分数预测      │  │
│  └───────────────────┬───────────────────────────┘  │
│                      ↓                              │
│          output_tokens: (B, 6, C)                   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────┐
│      Two-Way Transformer (双向Transformer)          │
│  ┌───────────────────────────────────────────────┐  │
│  │  重复 N=2 层:                                  │  │
│  │    1. Self-Attention on Queries               │  │
│  │    2. Cross-Attention (Q=queries, KV=image)   │  │
│  │    3. MLP                                     │  │
│  │    4. Cross-Attention (Q=image, KV=queries)   │  │
│  │    5. Self-Attention on Image                 │  │
│  │    6. MLP                                     │  │
│  └───────────────────┬───────────────────────────┘  │
│                      ↓                              │
│        refined_tokens: (B, 6, C)                    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────┐
│           解码输出                                   │
│  ┌───────────────────────────────────────────────┐  │
│  │ 1. 掩码预测:                                   │  │
│  │    for each mask_token:                       │  │
│  │      mlp_output = MLP(mask_token)             │  │
│  │      upscaled_features = UpSample(pix_feat)   │  │
│  │      + 高分辨率特征 (如果启用)                  │  │
│  │      mask_logits = mlp_output @ upscaled      │  │
│  │    → low_res_masks: (B, 4, 64, 64)            │  │
│  │    → high_res_masks: (B, 4, 256, 256)         │  │
│  │                                               │  │
│  │ 2. IoU预测:                                    │  │
│  │    iou_scores = MLP(iou_token)                │  │
│  │    → ious: (B, 4)                             │  │
│  │                                               │  │
│  │ 3. 对象分数预测:                                │  │
│  │    obj_score = Linear(obj_score_token)        │  │
│  │    → object_score_logits: (B, 1)              │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────┐
│        SAMURAI 多掩码选择                            │
│  如果 multimask_output=True 且 samurai_mode=True:   │
│  ┌───────────────────────────────────────────────┐  │
│  │ 从4个掩码中选择3个 (多掩码: indices 0,1,2)      │  │
│  │                                               │  │
│  │ 稳定跟踪阶段:                                   │  │
│  │   1. 从每个多掩码提取bbox                      │  │
│  │   2. 计算与KF预测的IoU → kf_ious              │  │
│  │   3. 加权: w = 0.25*kf + 0.75*sam             │  │
│  │   4. 选择 best = argmax(w)                    │  │
│  │                                               │  │
│  │ 预热/初始化阶段:                                │  │
│  │   best = argmax(sam_ious)                     │  │
│  └───────────────────┬───────────────────────────┘  │
│                      ↓                              │
│        final_mask: (B, 1, H, W)                     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────┐
│         提取对象指针 (Object Pointer)                 │
│  obj_ptr = Linear(sam_output_token[best])           │
│  + 对象出现性调制:                                    │
│    if obj_score < threshold:                        │
│      obj_ptr = λ * obj_ptr + (1-λ) * no_obj_ptr     │
│    ↓                                                │
│  obj_ptr: (B, C=256)                                │
└─────────────────────────────────────────────────────┘

输出:
  • low_res_masks: (B, 1, 64, 64) - 低分辨率掩码
  • high_res_masks: (B, 1, 256, 256) - 高分辨率掩码
  • ious: (B,) - 选中掩码的IoU预测
  • obj_ptr: (B, 256) - 对象指针 (用于后续帧)
  • object_score_logits: (B, 1) - 对象出现性分数
  • kf_ious: (B,) - 卡尔曼IoU (仅稳定阶段)
```

---

## 6. 记忆编码流程

```
输入:
┌─────────────────────────────────────────────────┐
│  high_res_mask: (B, 1, 256, 256) - 预测的掩码   │
│  current_vision_feats: (B, C, H, W) - 当前帧特征│
│  object_score_logits: (B, 1) - 对象分数         │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│      掩码下采样 (Mask Downsampler)               │
│  Conv2d(kernel=3, stride=2, padding=1)          │
│  256x256 → 128x128 → 64x64                      │
│    ↓                                            │
│  downsampled_mask: (B, 1, H/4, W/4)             │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│       融合器 (Fuser)                             │
│  将掩码与图像特征融合                             │
│  ┌───────────────────────────────────────────┐  │
│  │ 输入:                                      │  │
│  │   • current_vision_feats: (B, 256, H, W) │  │
│  │   • downsampled_mask: (B, 1, H, W)       │  │
│  │   ↓                                       │  │
│  │ CXBlock × 2 层:                            │  │
│  │   每层包含:                                │  │
│  │   - DepthWise Conv (7x7)                  │  │
│  │   - LayerNorm                             │  │
│  │   - Linear (256 → 1024)                   │  │
│  │   - GELU                                  │  │
│  │   - Linear (1024 → 256)                   │  │
│  │   - Layer Scale                           │  │
│  │   - Residual Connection                   │  │
│  │   ↓                                       │  │
│  │ 掩码调制:                                  │  │
│  │   fused = features * sigmoid(mask)        │  │
│  └───────────────┬───────────────────────────┘  │
│                  ↓                              │
│  fused_features: (B, 256, H, W)                 │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│       通道压缩 (Channel Compression)             │
│  Linear projection: 256 → 64                    │
│    ↓                                            │
│  compressed_features: (B, 64, H, W)             │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│       添加位置编码                                │
│  spatial_pos_enc = PositionEmbeddingSine(64)    │
│    ↓                                            │
│  maskmem_features: (B, 64, H, W)                │
│  maskmem_pos_enc: list of (B, 64, H_i, W_i)     │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│    对象出现性处理                                 │
│  if object_score_logits < threshold:            │
│    # 对象可能不存在，添加 no_obj_embedding       │
│    maskmem_features += no_obj_embed_spatial     │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│       存入输出字典                                │
│  output_dict[frame_idx] = {                     │
│    "maskmem_features": maskmem_features,        │
│    "maskmem_pos_enc": maskmem_pos_enc,          │
│    "pred_masks": low_res_masks,                 │
│    "obj_ptr": obj_ptr,                          │
│    "object_score_logits": object_score_logits,  │
│    "best_iou_score": best_iou,                  │
│    "kf_score": kf_iou  # SAMURAI添加            │
│  }                                              │
└─────────────────────────────────────────────────┘

记忆特征的使用:
  后续帧会通过 Memory Attention 模块查询这些编码的记忆:
    • maskmem_features 作为 Key 和 Value
    • maskmem_pos_enc 提供空间位置信息
    • 添加时间位置编码表示相对时间距离
```

---

## 7. 加权 IoU 计算详解

```
场景: 稳定跟踪阶段，SAM输出3个候选掩码

步骤1: 卡尔曼预测
┌────────────────────────────────────┐
│  kf_mean = [xc, yc, a, h, vx, vy, va, vh]│
│    ↓                                │
│  predicted_bbox = xyah_to_xyxy(     │
│    kf_mean[0:4]                     │
│  )                                  │
│  = [x_min, y_min, x_max, y_max]     │
└────────────────────────────────────┘

步骤2: 从SAM掩码提取边界框
┌────────────────────────────────────┐
│  对于每个候选掩码 i (i=0,1,2):      │
│    mask_i = high_res_masks[i]      │
│    ↓                                │
│    non_zero = argwhere(mask_i > 0) │
│    ↓                                │
│    if len(non_zero) > 0:           │
│      y_min = min(non_zero[:, 0])   │
│      x_min = min(non_zero[:, 1])   │
│      y_max = max(non_zero[:, 0])   │
│      x_max = max(non_zero[:, 1])   │
│      bbox_i = [x_min, y_min,       │
│                x_max, y_max]       │
│    else:                           │
│      bbox_i = [0, 0, 0, 0]         │
│    ↓                                │
│  multi_bboxes = [bbox_0, bbox_1, bbox_2]│
└────────────────────────────────────┘

步骤3: 计算运动IoU
┌────────────────────────────────────┐
│  def compute_iou(bbox1, bbox2):    │
│    x1_min, y1_min, x1_max, y1_max = bbox1│
│    x2_min, y2_min, x2_max, y2_max = bbox2│
│    ↓                                │
│    # 交集                           │
│    inter_x_min = max(x1_min, x2_min)│
│    inter_y_min = max(y1_min, y2_min)│
│    inter_x_max = min(x1_max, x2_max)│
│    inter_y_max = min(y1_max, y2_max)│
│    ↓                                │
│    inter_area = max(0, inter_x_max - inter_x_min) * \│
│                max(0, inter_y_max - inter_y_min)│
│    ↓                                │
│    # 并集                           │
│    area1 = (x1_max - x1_min) * (y1_max - y1_min)│
│    area2 = (x2_max - x2_min) * (y2_max - y2_min)│
│    union_area = area1 + area2 - inter_area│
│    ↓                                │
│    iou = inter_area / union_area   │
│    return iou                      │
│  ↓                                  │
│  kf_ious = [                        │
│    compute_iou(predicted_bbox, bbox_0),│
│    compute_iou(predicted_bbox, bbox_1),│
│    compute_iou(predicted_bbox, bbox_2) │
│  ]                                  │
└────────────────────────────────────┘

步骤4: 加权融合
┌────────────────────────────────────┐
│  sam_ious = [iou_0, iou_1, iou_2]  │
│    (来自SAM的IoU预测头)             │
│  ↓                                  │
│  w = kf_score_weight = 0.25        │
│  ↓                                  │
│  weighted_ious = [                 │
│    w * kf_ious[0] + (1-w) * sam_ious[0],│
│    w * kf_ious[1] + (1-w) * sam_ious[1],│
│    w * kf_ious[2] + (1-w) * sam_ious[2] │
│  ]                                  │
└────────────────┬───────────────────┘
                 │
                 ↓
┌────────────────────────────────────┐
│  best_idx = argmax(weighted_ious)  │
│  ↓                                  │
│  final_mask = multi_masks[best_idx]│
└────────────────────────────────────┘

示例数值:
  SAM IoUs:     [0.85, 0.60, 0.72]
  KF IoUs:      [0.65, 0.90, 0.55]
  Weighted:     [0.80, 0.68, 0.67]
                  ↑ 最高
  选择: mask_0

分析:
  • SAM IoU 认为 mask_0 最好 (0.85)
  • 但 KF IoU 认为 mask_1 更符合运动 (0.90)
  • 加权后 mask_0 仍然获胜 (0.80)
  • 因为 SAM IoU 权重更大 (75%)
  
当运动一致性很重要时 (快速运动):
  增大 kf_score_weight 至 0.4-0.5
  这会让 KF IoU 有更大影响力
```

---

## 总结

本文档提供了 SAMURAI 各个关键流程的可视化图解，包括：

1. **整体推理流程**: 从视频输入到最终输出的完整数据流
2. **三阶段跟踪策略**: 初始化、预热、稳定跟踪的决策逻辑
3. **记忆注意力机制**: 如何融合当前帧与历史信息
4. **卡尔曼滤波器**: 状态预测和更新的数学流程
5. **SAM解码器**: 从特征到掩码的解码过程
6. **记忆编码**: 如何将掩码编码为记忆特征
7. **加权IoU计算**: SAMURAI核心的多掩码选择算法

这些图解可以帮助理解代码实现，并指导模型的修改和调试。

**配合使用:**
- `ARCHITECTURE.md`: 详细的文本描述和代码位置
- `QUICK_REFERENCE.md`: 快速查阅参数和代码片段
- `WORKFLOW_DIAGRAMS.md` (本文档): 可视化流程图

