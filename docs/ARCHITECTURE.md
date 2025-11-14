# SAMURAI 目标跟踪模型架构文档

## 概述

SAMURAI（Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory）是基于 SAM 2（Segment Anything Model 2）的零样本视觉目标跟踪模型。该模型通过引入运动感知记忆机制（Motion-Aware Memory）和卡尔曼滤波器来增强 SAM 2 的视频目标跟踪能力。

**核心特性：**
- 零样本跟踪（无需额外训练）
- 基于 SAM 2.1 预训练权重
- 运动感知的记忆库选择机制
- 卡尔曼滤波器辅助的多掩码选择

---

## 整体架构

### 1. 系统层次结构

```
SAMURAI
├── 推理接口层
│   ├── SAM2VideoPredictor (视频预测器)
│   └── Demo/Evaluation Scripts (演示/评估脚本)
├── 核心模型层 
│   ├── SAM2Base (基础模型)
│   ├── Image Encoder (图像编码器)
│   ├── Memory Attention (记忆注意力)
│   ├── Memory Encoder (记忆编码器)
│   └── SAM Decoder (SAM 解码器)
├── SAMURAI 增强层
│   ├── Kalman Filter (卡尔曼滤波器)
│   ├── Motion-Aware Memory Bank Selection (运动感知记忆库选择)
│   └── Weighted Multi-mask Selection (加权多掩码选择)
└── 数据处理层
    ├── Video Frame Loading (视频帧加载)
    └── Inference State Management (推理状态管理)
```

---

## 核心模块详解

### 2.1 SAM2Base 基础模型 (`sam2/sam2/modeling/sam2_base.py`)

**类定义：** `SAM2Base(torch.nn.Module)`

**主要职责：**
- 定义整体模型架构
- 管理图像编码、记忆注意力、掩码解码等核心组件
- 实现 SAMURAI 模式的运动感知跟踪逻辑

**关键参数：**
```python
# 基础参数
num_maskmem: int = 7              # 记忆帧数量（1个输入帧 + 6个历史帧）
image_size: int = 512/1024        # 输入图像尺寸
use_high_res_features_in_sam: bool  # 是否使用高分辨率特征

# SAMURAI 特有参数
samurai_mode: bool = True/False   # 是否启用 SAMURAI 模式
stable_frames_threshold: int = 15  # 稳定帧阈值
stable_ious_threshold: float = 0.3 # 稳定 IoU 阈值
kf_score_weight: float = 0.25     # 卡尔曼滤波器分数权重
memory_bank_iou_threshold: float = 0.5      # 记忆库 IoU 阈值
memory_bank_obj_score_threshold: float = 0.0 # 记忆库对象分数阈值
memory_bank_kf_score_threshold: float = 0.0  # 记忆库卡尔曼分数阈值
```

**核心方法：**

1. **`track_step()`** - 单帧跟踪
   - 输入：当前帧特征、点/框提示
   - 输出：预测掩码、对象指针、置信度分数
   - 流程：特征提取 → 记忆融合 → SAM 解码 → 记忆编码

2. **`_track_step()`** - 内部跟踪实现
   - 准备记忆条件特征
   - 调用 SAM 头部进行掩码预测
   - 处理 SAMURAI 模式下的多掩码选择

3. **`_forward_sam_heads()`** - SAM 解码头前向传播
   - **SAMURAI 核心逻辑所在**
   - 实现基于卡尔曼滤波器的多掩码选择（第 420-534 行）
   - 三阶段处理：
     - **初始化阶段**（`kf_mean is None`）：初始化卡尔曼滤波器
     - **预热阶段**（`stable_frames < stable_frames_threshold`）：建立稳定跟踪
     - **稳定跟踪阶段**：使用加权 IoU 进行掩码选择

4. **`_prepare_memory_conditioned_features()`** - 准备记忆条件特征
   - **SAMURAI 记忆库选择逻辑**（第 663-687 行）
   - 基于 IoU、对象分数、卡尔曼分数过滤历史帧
   - 选择高质量记忆帧用于注意力机制

---

### 2.2 图像编码器 (Image Encoder)

**路径：** `sam2/sam2/modeling/backbones/`

**组件结构：**
```
ImageEncoder
├── trunk: Hiera (分层视觉 Transformer)
│   ├── embed_dim: 112 (base_plus), 144 (large)
│   └── num_heads: 2/4
└── neck: FpnNeck (特征金字塔网络)
    ├── position_encoding (位置编码)
    ├── d_model: 256
    └── fpn_top_down_levels: [2, 3]
```

**功能：**
- 多尺度特征提取
- Hiera backbone 进行高效图像编码
- FPN 融合多尺度特征

**输出：**
```python
{
    "vision_features": src,        # 主特征图
    "vision_pos_enc": pos,         # 位置编码
    "backbone_fpn": features,      # 多尺度特征列表
}
```

---

### 2.3 记忆注意力 (Memory Attention)

**路径：** `sam2/sam2/modeling/memory_attention.py`

**类：** `MemoryAttention`, `MemoryAttentionLayer`

**架构：**
```
MemoryAttentionLayer
├── Self-Attention (RoPEAttention)
│   └── 旋转位置编码注意力
├── Cross-Attention (RoPEAttention)
│   └── 当前帧与记忆帧的交叉注意力
└── Feed-Forward Network
    └── MLP (d_model → dim_feedforward → d_model)
```

**关键机制：**
- **RoPE（旋转位置编码）**：处理空间和时间位置信息
- **交叉注意力**：当前帧特征查询历史记忆
- **对象指针（Object Pointers）**：从其他帧传递对象信息

**参数配置：**
```yaml
d_model: 256
num_layers: 4
dim_feedforward: 2048
num_heads: 1
```

---

### 2.4 记忆编码器 (Memory Encoder)

**路径：** `sam2/sam2/modeling/memory_encoder.py`

**功能：**
- 将预测的掩码编码为记忆特征
- 压缩通道维度（256 → 64）
- 添加位置编码和时间编码

**组件：**
```
MemoryEncoder
├── mask_downsampler: MaskDownSampler (掩码下采样)
└── fuser: Fuser
    └── CXBlock layers (卷积块)
        ├── kernel_size: 7
        ├── padding: 3
        └── num_layers: 2
```

**输出：**
- `maskmem_features`: 记忆特征 (B, C=64, H, W)
- `maskmem_pos_enc`: 记忆位置编码

---

### 2.5 SAM 解码器 (Mask Decoder)

**路径：** `sam2/sam2/modeling/sam/mask_decoder.py`

**类：** `MaskDecoder`

**功能：**
- 预测目标掩码
- 多掩码输出（3个候选掩码）
- IoU 质量预测
- 对象出现性预测（Object Score）

**核心组件：**
```
MaskDecoder
├── transformer: TwoWayTransformer (双向 Transformer)
├── iou_prediction_head: MLP (IoU 预测头)
├── pred_obj_score_head: Linear (对象分数预测头)
├── mask_tokens: Embedding (掩码 token)
└── output_upscaling: ConvTranspose2d (上采样网络)
```

**输出：**
```python
low_res_masks:      # 低分辨率掩码 (B, 3, H/4, W/4)
high_res_masks:     # 高分辨率掩码 (B, 3, H, W)
ious:               # IoU 预测分数 (B, 3)
object_score_logits: # 对象出现性分数 (B, 1)
```

---

### 2.6 卡尔曼滤波器 (Kalman Filter)

**路径：** `sam2/sam2/utils/kalman_filter.py`

**类：** `KalmanFilter`

**状态空间：** 8 维
```
[x, y, a, h, vx, vy, va, vh]
x, y:    边界框中心位置
a:       宽高比
h:       高度
vx, vy:  速度分量
va, vh:  宽高比和高度的变化速率
```

**核心方法：**

1. **`initiate(measurement)`** - 初始化跟踪
   - 输入：初始边界框 `[x, y, a, h]`
   - 输出：均值向量和协方差矩阵

2. **`predict(mean, covariance)`** - 预测下一状态
   - 基于恒速度模型
   - 更新均值和协方差

3. **`update(mean, covariance, measurement)`** - 更新状态
   - 融合观测值
   - 卡尔曼增益计算

4. **`compute_iou(pred_bbox, bboxes)`** - 计算 IoU
   - 预测边界框与候选框的 IoU
   - 用于 SAMURAI 的加权选择

**坐标转换：**
- `xyxy_to_xyah()`: (x1, y1, x2, y2) → (xc, yc, a, h)
- `xyah_to_xyxy()`: (xc, yc, a, h) → (x1, y1, x2, y2)

---

## SAMURAI 核心机制

### 3.1 运动感知的多掩码选择

**位置：** `sam2_base.py` 第 420-534 行

**算法流程：**

```python
if multimask_output and samurai_mode:
    if 初始化阶段 (kf_mean is None or stable_frames == 0):
        # 1. 选择最高 IoU 的掩码
        best_mask = argmax(ious)
        # 2. 从掩码提取边界框
        bbox = extract_bbox_from_mask(best_mask)
        # 3. 初始化卡尔曼滤波器
        kf_mean, kf_covariance = kf.initiate(bbox)
        stable_frames += 1
    
    elif 预热阶段 (stable_frames < stable_frames_threshold):
        # 1. 预测当前帧状态
        kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)
        # 2. 选择最高 IoU 的掩码
        best_mask = argmax(ious)
        # 3. 如果 IoU > stable_ious_threshold，更新卡尔曼滤波器
        if iou > stable_ious_threshold:
            kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, bbox)
            stable_frames += 1
        else:
            stable_frames = 0  # 重置稳定帧计数
    
    else:  # 稳定跟踪阶段
        # 1. 预测当前帧状态
        kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)
        # 2. 提取所有候选掩码的边界框
        multi_bboxes = [extract_bbox(mask) for mask in multi_masks]
        # 3. 计算卡尔曼预测框与候选框的 IoU
        kf_ious = compute_iou(kf_predicted_bbox, multi_bboxes)
        # 4. 加权融合 SAM IoU 和卡尔曼 IoU
        weighted_ious = kf_score_weight * kf_ious + (1 - kf_score_weight) * ious
        # 5. 选择加权 IoU 最高的掩码
        best_mask = argmax(weighted_ious)
        # 6. 如果 SAM IoU > threshold，更新卡尔曼滤波器
        if sam_iou > stable_ious_threshold:
            kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, best_bbox)
        else:
            stable_frames = 0  # 跟踪不稳定，重置
```

**关键参数：**
- `stable_frames_threshold = 15`: 预热期长度
- `stable_ious_threshold = 0.3`: 判断跟踪稳定的 IoU 阈值
- `kf_score_weight = 0.25`: 卡尔曼 IoU 的权重

---

### 3.2 运动感知的记忆库选择

**位置：** `sam2_base.py` 第 663-687 行

**目的：** 从历史帧中选择高质量的记忆帧用于注意力机制

**算法流程：**

```python
if samurai_mode:
    valid_indices = []
    # 从最近的帧向前遍历
    for i in range(frame_idx - 1, 1, -1):
        # 获取该帧的质量指标
        iou_score = output_dict[i]["best_iou_score"]      # 掩码质量
        obj_score = output_dict[i]["object_score_logits"] # 对象置信度
        kf_score = output_dict[i]["kf_score"]             # 运动一致性
        
        # 检查是否满足质量要求
        if (iou_score > memory_bank_iou_threshold and 
            obj_score > memory_bank_obj_score_threshold and
            (kf_score is None or kf_score > memory_bank_kf_score_threshold)):
            valid_indices.insert(0, i)
        
        # 最多选择 max_obj_ptrs_in_encoder - 1 个历史帧
        if len(valid_indices) >= max_obj_ptrs_in_encoder - 1:
            break
    
    # 确保包含最近一帧
    if frame_idx - 1 not in valid_indices:
        valid_indices.append(frame_idx - 1)
    
    # 使用选择的帧构建记忆库
    for t_pos in range(1, num_maskmem):
        idx = t_pos - num_maskmem
        if idx < -len(valid_indices):
            continue
        out = output_dict["non_cond_frame_outputs"][valid_indices[idx]]
        memory_bank.append((t_pos, out))
```

**质量过滤标准：**
```python
memory_bank_iou_threshold = 0.5         # IoU > 0.5
memory_bank_obj_score_threshold = 0.0   # 对象存在
memory_bank_kf_score_threshold = 0.0    # 运动一致性
```

**优势：**
- 避免使用低质量帧污染记忆
- 保持时间连续性（优先选择最近的高质量帧）
- 自适应处理遮挡和外观变化

---

## 推理接口层

### 4.1 SAM2VideoPredictor

**路径：** `sam2/sam2/sam2_video_predictor.py`

**类：** `SAM2VideoPredictor(SAM2Base)`

**主要方法：**

#### 4.1.1 `init_state(video_path, ...)`
初始化推理状态

**输入：**
- `video_path`: 视频路径或帧目录
- `offload_video_to_cpu`: 是否将视频帧卸载到 CPU
- `offload_state_to_cpu`: 是否将推理状态卸载到 CPU

**输出：**
```python
inference_state = {
    "images": List[Tensor],              # 视频帧
    "num_frames": int,                   # 总帧数
    "video_height": int,                 # 原始高度
    "video_width": int,                  # 原始宽度
    "obj_id_to_idx": OrderedDict,        # 对象 ID 映射
    "output_dict": {                     # 输出字典
        "cond_frame_outputs": {},        # 条件帧输出
        "non_cond_frame_outputs": {},    # 非条件帧输出
    },
    "cached_features": {},               # 缓存的特征
    ...
}
```

#### 4.1.2 `add_new_points_or_box(state, box=None, frame_idx=0, obj_id=0)`
添加初始边界框或点提示

**输入：**
- `state`: 推理状态
- `box`: 边界框 `(x1, y1, x2, y2)`
- `frame_idx`: 帧索引
- `obj_id`: 对象 ID

**输出：**
- `frame_idx`: 处理的帧索引
- `object_ids`: 对象 ID 列表
- `masks`: 预测的掩码

#### 4.1.3 `propagate_in_video(state, start_frame_idx=None, reverse=False)`
在视频中传播跟踪

**输入：**
- `state`: 推理状态
- `start_frame_idx`: 起始帧（默认从第一个条件帧开始）
- `reverse`: 是否反向跟踪

**输出：** 生成器，逐帧产生
```python
for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    # frame_idx: 当前帧索引
    # object_ids: 跟踪的对象 ID
    # masks: 预测的掩码 (N, 1, H, W)
```

**工作流程：**
1. 确定处理顺序（前向/反向）
2. 对每一帧：
   - 提取图像特征
   - 融合记忆信息
   - 预测掩码
   - 编码为新记忆
   - 更新推理状态
3. 应用非重叠约束（如果启用）
4. 返回原始分辨率的掩码

---

### 4.2 推理脚本

#### 4.2.1 主推理脚本 (`scripts/main_inference.py`)

**用途：** 在 LaSOT 等数据集上批量评估

**核心流程：**
```python
# 1. 加载模型
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 2. 初始化推理状态
state = predictor.init_state(frame_folder, 
                             offload_video_to_cpu=True,
                             offload_state_to_cpu=True)

# 3. 添加第一帧的边界框提示
bbox, _ = load_lasot_gt(gt_path)[0]  # (x, y, x+w, y+h)
predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

# 4. 传播跟踪
for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    # 从掩码提取边界框
    bbox = extract_bbox_from_mask(masks[0])
    predictions.append(bbox)

# 5. 保存结果
save_predictions(predictions, output_path)
```

#### 4.2.2 演示脚本 (`scripts/demo.py`)

**用途：** 在自定义视频上进行跟踪演示

**支持输入：**
- 视频文件 (`.mp4`)
- 帧目录（JPG 图像）

**使用方法：**
```bash
# 视频文件输入
python scripts/demo.py \
    --video_path video.mp4 \
    --txt_path bbox.txt \
    --model_path sam2/checkpoints/sam2.1_hiera_base_plus.pt \
    --video_output_path output.mp4

# 帧目录输入
python scripts/demo.py \
    --video_path frames_dir/ \
    --txt_path bbox.txt
```

**边界框格式：** `bbox.txt` 包含第一帧的边界框，格式为 `x,y,w,h`

---

## 配置文件

### 5.1 SAMURAI 配置 (`sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml`)

**模型配置：**
```yaml
model:
  _target_: sam2.modeling.sam2_base.SAM2Base
  
  # 图像编码器
  image_encoder:
    trunk:
      _target_: sam2.modeling.backbones.hieradet.Hiera
      embed_dim: 112
      num_heads: 2
    neck:
      d_model: 256
      backbone_channel_list: [896, 448, 224, 112]
  
  # 记忆注意力
  memory_attention:
    d_model: 256
    num_layers: 4
    layer:
      dim_feedforward: 2048
      self_attention: RoPEAttention
      cross_attention: RoPEAttention
  
  # 记忆编码器
  memory_encoder:
    out_dim: 64
    fuser:
      num_layers: 2
  
  # 记忆和掩码参数
  num_maskmem: 7
  image_size: 1024
  use_high_res_features_in_sam: true
  multimask_output_in_sam: true
  multimask_output_for_tracking: true
  
  # SAMURAI 参数
  samurai_mode: true
  stable_frames_threshold: 15
  stable_ious_threshold: 0.3
  min_obj_score_logits: -1
  kf_score_weight: 0.25
  memory_bank_iou_threshold: 0.5
  memory_bank_obj_score_threshold: 0.0
  memory_bank_kf_score_threshold: 0.0
```

**不同模型尺寸：**
- **Tiny (T)**: `embed_dim=96`, 最快，精度稍低
- **Small (S)**: `embed_dim=96`, 平衡
- **Base+ (B+)**: `embed_dim=112`, 推荐，性能最优
- **Large (L)**: `embed_dim=144`, 最准确，较慢

---

## 数据流

### 6.1 完整推理流程

```
输入视频
    ↓
[视频帧加载] → images (T, 3, H, W)
    ↓
[初始化推理状态] → inference_state
    ↓
[第一帧] ← 边界框提示 (x1, y1, x2, y2)
    ↓
[图像编码器] → vision_features, vision_pos_enc
    ↓
[SAM 解码器] → 初始掩码
    ↓
[记忆编码器] → maskmem_features (存入 output_dict)
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
对于每一帧 (t = 1, 2, ..., T):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
[图像编码器] → current_vision_features
    ↓
[运动感知记忆库选择] → 选择高质量历史帧
    ↓
[记忆注意力] → 融合当前帧与历史记忆
    ↓         (cross-attention + object pointers)
current_features_with_memory
    ↓
[SAM 解码器] → 预测多个候选掩码 (3个)
    ↓         + IoU 预测
    ↓         + 对象分数预测
    ↓
[SAMURAI 多掩码选择]
    ├─ 如果处于初始化/预热阶段：
    │    └─ 选择最高 IoU 掩码
    └─ 如果处于稳定跟踪阶段：
         ├─ 卡尔曼滤波器预测边界框
         ├─ 计算卡尔曼 IoU
         ├─ 加权融合: weighted_iou = 0.25 * kf_iou + 0.75 * sam_iou
         └─ 选择最高加权 IoU 掩码
    ↓
final_mask (选定的掩码)
    ↓
[记忆编码器] → 编码为新记忆
    ↓
[更新推理状态] → 存入 output_dict["non_cond_frame_outputs"][t]
    ↓
[卡尔曼滤波器更新] → 更新状态估计 (如果 IoU > 阈值)
    ↓
输出: (frame_idx, object_ids, masks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 6.2 记忆注意力数据流

```
当前帧特征 (current_vision_feats)
    +
记忆帧特征 (maskmem_features from selected frames)
    +
对象指针 (obj_ptr from selected frames)
    ↓
[Self-Attention] → 特征自注意力
    ↓
[Cross-Attention] → 查询历史记忆
    ↓         (query: current frame, key/value: memory + obj_ptrs)
    ↓
[FFN] → 前馈网络
    ↓
记忆条件特征 (pix_feat_with_mem)
```

---

## 关键接口总结

### 7.1 外部调用接口

| 接口 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `build_sam2_video_predictor()` | 配置文件, 权重路径 | `SAM2VideoPredictor` | 构建预测器 |
| `predictor.init_state()` | 视频路径 | `inference_state` | 初始化推理 |
| `predictor.add_new_points_or_box()` | 边界框/点, 帧索引 | 初始掩码 | 添加提示 |
| `predictor.propagate_in_video()` | `inference_state` | 生成器: (frame_idx, obj_ids, masks) | 视频跟踪 |

### 7.2 内部核心接口

| 模块 | 方法 | 功能 |
|------|------|------|
| `SAM2Base` | `track_step()` | 单帧跟踪主函数 |
| `SAM2Base` | `_forward_sam_heads()` | SAM 解码 + SAMURAI 掩码选择 |
| `SAM2Base` | `_prepare_memory_conditioned_features()` | 记忆融合 + SAMURAI 记忆库选择 |
| `KalmanFilter` | `predict()` | 预测下一状态 |
| `KalmanFilter` | `update()` | 融合观测更新 |
| `KalmanFilter` | `compute_iou()` | 计算 IoU |
| `ImageEncoder` | `forward()` | 图像特征提取 |
| `MemoryAttention` | `forward()` | 记忆注意力 |
| `MemoryEncoder` | `forward()` | 掩码编码为记忆 |
| `MaskDecoder` | `forward()` | 掩码解码 |

---

## 修改模型指南

### 8.1 调整 SAMURAI 参数

**修改配置文件：** `sam2/sam2/configs/samurai/sam2.1_hiera_*.yaml`

```yaml
# 调整卡尔曼滤波器影响
kf_score_weight: 0.25  # 增大 → 更依赖运动预测
                       # 减小 → 更依赖 SAM IoU

# 调整稳定性要求
stable_frames_threshold: 15  # 增大 → 延长预热期
stable_ious_threshold: 0.3   # 增大 → 更严格的稳定判断

# 调整记忆库质量过滤
memory_bank_iou_threshold: 0.5      # 增大 → 更严格的帧选择
memory_bank_obj_score_threshold: 0.0
memory_bank_kf_score_threshold: 0.0
```

### 8.2 修改卡尔曼滤波器

**位置：** `sam2/sam2/utils/kalman_filter.py`

**可调参数：**
```python
# 调整过程噪声
self._std_weight_position = 1. / 20  # 位置不确定性
self._std_weight_velocity = 1. / 160 # 速度不确定性

# 调整状态转移
self._motion_mat  # 运动模型矩阵（恒速度模型）
```

### 8.3 修改多掩码选择逻辑

**位置：** `sam2/sam2/modeling/sam2_base.py` 第 420-534 行

**示例：添加额外的掩码选择策略**

```python
# 在 _forward_sam_heads() 方法中
if multimask_output and self.samurai_mode:
    # ... 现有代码 ...
    
    # 新增：结合掩码面积作为额外约束
    mask_areas = [mask.sum() for mask in high_res_multimasks]
    area_scores = normalize(mask_areas)
    
    # 三重加权
    weighted_ious = (
        self.kf_score_weight * kf_ious + 
        (1 - self.kf_score_weight - self.area_weight) * ious +
        self.area_weight * area_scores
    )
```

### 8.4 修改记忆库选择策略

**位置：** `sam2/sam2/modeling/sam2_base.py` 第 663-687 行

**示例：添加时间衰减权重**

```python
if self.samurai_mode:
    valid_indices = []
    frame_weights = []  # 新增权重
    
    for i in range(frame_idx - 1, 1, -1):
        # ... 现有质量检查 ...
        if passes_quality_check:
            valid_indices.insert(0, i)
            # 计算时间衰减权重
            time_diff = frame_idx - i
            weight = np.exp(-time_diff / self.temporal_decay_rate)
            frame_weights.insert(0, weight)
        
        if len(valid_indices) >= self.max_obj_ptrs_in_encoder - 1:
            break
    
    # 使用权重调整注意力
    # ...
```

### 8.5 添加新的质量指标

**步骤：**

1. **在 `track_step()` 中计算新指标**
   ```python
   # 在 sam2_base.py 的 track_step() 方法中
   current_out["custom_quality_score"] = compute_custom_quality(masks)
   ```

2. **在记忆库选择中使用新指标**
   ```python
   if self.samurai_mode:
       for i in range(frame_idx - 1, 1, -1):
           custom_score = output_dict[i]["custom_quality_score"]
           if custom_score > self.custom_threshold:
               valid_indices.insert(0, i)
   ```

3. **在配置文件中添加新阈值**
   ```yaml
   custom_threshold: 0.8
   ```

### 8.6 切换到原始 SAM 2 模式

**方法 1：修改配置文件**
```yaml
samurai_mode: false  # 禁用 SAMURAI 功能
```

**方法 2：使用官方 SAM 2 配置**
```python
# 使用 sam2/sam2/configs/sam2.1/ 下的配置
predictor = build_sam2_video_predictor(
    "configs/sam2.1/sam2.1_hiera_b+.yaml",  # 而非 samurai/
    checkpoint_path
)
```

---

## 性能优化建议

### 9.1 内存优化

```python
# 启用 CPU 卸载
state = predictor.init_state(
    video_path,
    offload_video_to_cpu=True,      # 将视频帧卸载到 CPU
    offload_state_to_cpu=True,      # 将推理状态卸载到 CPU
    async_loading_frames=True,      # 异步加载帧
)
```

### 9.2 速度优化

**选择更小的模型：**
- Tiny/Small 模型速度更快，精度略有下降
- Base+ 是速度和精度的最佳平衡

**调整图像尺寸：**
```yaml
image_size: 512  # 从 1024 降低到 512 可显著提速
```

**减少记忆帧数量：**
```yaml
num_maskmem: 4  # 从 7 减少到 4
```

### 9.3 精度优化

**使用高分辨率特征：**
```yaml
use_high_res_features_in_sam: true
```

**启用多掩码跟踪：**
```yaml
multimask_output_for_tracking: true
```

**调整 SAMURAI 参数：**
```yaml
kf_score_weight: 0.3  # 增大以更依赖运动模型
stable_frames_threshold: 20  # 延长稳定期
```

---

## 常见问题

### Q1: SAMURAI 与 SAM 2 的主要区别是什么？

**A:** SAMURAI 在 SAM 2 基础上添加了：
1. **卡尔曼滤波器**：跟踪目标的运动状态
2. **运动感知的多掩码选择**：结合运动预测和外观特征选择最佳掩码
3. **运动感知的记忆库选择**：基于质量指标过滤历史帧
4. **零样本跟踪**：无需额外训练，直接使用 SAM 2.1 权重

### Q2: 如何判断当前是否处于稳定跟踪阶段？

**A:** 检查 `self.stable_frames` 计数器：
```python
if self.stable_frames >= self.stable_frames_threshold:
    # 稳定跟踪阶段，使用加权 IoU 选择
else:
    # 预热阶段，使用简单的最大 IoU 选择
```

### Q3: 记忆库选择如何影响跟踪性能？

**A:** 高质量的记忆库可以：
- 减少误差累积
- 提高对遮挡的鲁棒性
- 避免使用失败帧污染记忆
- 保持外观和运动的时间一致性

### Q4: 如何处理长视频的内存问题？

**A:**
1. 启用 `offload_video_to_cpu=True` 和 `offload_state_to_cpu=True`
2. 使用分块推理（参见 `scripts/main_inference_chunk.py`）
3. 降低图像分辨率或使用更小的模型

### Q5: 卡尔曼滤波器的状态何时会重置？

**A:** 当 `stable_frames` 归零时（`iou < stable_ious_threshold`），表示跟踪不稳定，但卡尔曼滤波器状态本身不会完全重置，会继续进行预测。

---

## 引用与许可

**SAMURAI 论文：**
```bibtex
@misc{yang2024samurai,
  title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory}, 
  author={Cheng-Yen Yang and Hsiang-Wei Huang and Wenhao Chai and Zhongyu Jiang and Jenq-Neng Hwang},
  year={2024},
  eprint={2411.11922},
  archivePrefix={arXiv},
}
```

**SAM 2 论文：**
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

**许可证：** Apache 2.0 License

---

## 附录

### A. 目录结构说明

```
samurai/
├── sam2/                          # SAM 2 核心代码
│   ├── sam2/                      # SAM 2 Python 包
│   │   ├── modeling/              # 模型定义
│   │   │   ├── sam2_base.py       # ★ SAMURAI 核心逻辑
│   │   │   ├── backbones/         # 图像编码器
│   │   │   ├── memory_attention.py # 记忆注意力
│   │   │   ├── memory_encoder.py  # 记忆编码器
│   │   │   └── sam/               # SAM 解码器
│   │   ├── utils/
│   │   │   └── kalman_filter.py   # ★ 卡尔曼滤波器
│   │   ├── configs/
│   │   │   └── samurai/           # ★ SAMURAI 配置
│   │   ├── sam2_video_predictor.py # ★ 视频预测器
│   │   └── build_sam.py           # 模型构建
│   └── checkpoints/               # 预训练权重
├── scripts/
│   ├── main_inference.py          # ★ 主推理脚本
│   ├── demo.py                    # ★ 演示脚本
│   └── main_inference_chunk.py    # 分块推理
├── lib/                           # 训练和评估工具
│   ├── test/                      # 测试工具
│   └── train/                     # 训练工具
└── README.md                      # 项目说明

★ 标记为核心文件
```

### B. 配置参数完整列表

**图像编码器参数：**
- `embed_dim`: 嵌入维度
- `num_heads`: 注意力头数
- `backbone_channel_list`: 各层通道数

**记忆参数：**
- `num_maskmem`: 记忆帧数量
- `memory_temporal_stride_for_eval`: 记忆帧采样步长
- `max_cond_frames_in_attn`: 最大条件帧数
- `max_obj_ptrs_in_encoder`: 最大对象指针数

**SAM 解码器参数：**
- `use_high_res_features_in_sam`: 使用高分辨率特征
- `multimask_output_in_sam`: 多掩码输出
- `multimask_output_for_tracking`: 跟踪时使用多掩码
- `pred_obj_scores`: 预测对象分数
- `use_obj_ptrs_in_encoder`: 使用对象指针

**SAMURAI 参数：**
- `samurai_mode`: 启用 SAMURAI
- `stable_frames_threshold`: 稳定帧阈值
- `stable_ious_threshold`: 稳定 IoU 阈值
- `kf_score_weight`: 卡尔曼分数权重
- `memory_bank_iou_threshold`: 记忆库 IoU 阈值
- `memory_bank_obj_score_threshold`: 记忆库对象分数阈值
- `memory_bank_kf_score_threshold`: 记忆库卡尔曼分数阈值

---

**文档版本：** v1.0  
**最后更新：** 2025-10-21  
**作者：** AI Assistant  
**适用模型版本：** SAMURAI (based on SAM 2.1)

