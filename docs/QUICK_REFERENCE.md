# SAMURAI 快速参考手册

## 核心概念速查

### 模型架构（3 层）
```
1. 编码层: Hiera Backbone → FPN → 多尺度特征
2. 记忆层: Memory Attention + Memory Encoder → 时序信息融合
3. 解码层: SAM Decoder → 多掩码预测 + IoU估计
```

### SAMURAI 三大创新

**1. 卡尔曼滤波器辅助选择**
```python
# 状态: [x, y, a, h, vx, vy, va, vh]
预测下一帧 → 计算运动IoU → 加权融合: 0.25*KF_IoU + 0.75*SAM_IoU
```

**2. 三阶段跟踪策略**
- 初始化 (帧0): 初始化KF，选最高IoU掩码
- 预热 (帧1-15): 建立稳定跟踪，IoU>0.3时更新KF
- 稳定跟踪 (帧15+): 使用加权IoU选择

**3. 智能记忆库选择**
```python
过滤条件:
- IoU > 0.5 (掩码质量)
- obj_score > 0.0 (对象存在)
- kf_score > 0.0 (运动一致)
优先选择: 最近的高质量帧
```

---

## 关键文件速查

| 文件 | 核心功能 | 关键行数 |
|------|---------|---------|
| `sam2_base.py` | SAMURAI主逻辑 | 420-534 (多掩码), 663-687 (记忆库) |
| `kalman_filter.py` | 运动模型 | 全文 |
| `sam2_video_predictor.py` | 推理接口 | 44-100 (init), 663-703 (propagate) |
| `sam2.1_hiera_b+.yaml` | 配置文件 | 117-126 (SAMURAI参数) |

---

## 修改指南速查表

### 场景1: 调整运动模型影响
```yaml
# 增大 → 更依赖运动预测，适合快速运动
# 减小 → 更依赖外观特征，适合慢速运动
kf_score_weight: 0.25  # 范围: 0.0-0.5
```

### 场景2: 处理遮挡问题
```yaml
# 增大 → 更严格的稳定性要求，抗遮挡能力强
stable_frames_threshold: 15    # 推荐: 10-25
stable_ious_threshold: 0.3     # 推荐: 0.2-0.5
```

### 场景3: 提升记忆库质量
```yaml
# 增大 → 更严格的历史帧筛选
memory_bank_iou_threshold: 0.5      # 推荐: 0.4-0.7
memory_bank_obj_score_threshold: 0.0 # 推荐: -1.0-1.0
```

### 场景4: 速度/精度权衡
```yaml
# 速度优先
image_size: 512
num_maskmem: 4
model: sam2.1_hiera_t.yaml

# 精度优先  
image_size: 1024
num_maskmem: 7
model: sam2.1_hiera_l.yaml
```

---

## 常用代码片段

### 基础推理
```python
from sam2.build_sam import build_sam2_video_predictor

# 1. 构建模型
predictor = build_sam2_video_predictor(
    "configs/samurai/sam2.1_hiera_b+.yaml",
    "checkpoints/sam2.1_hiera_base_plus.pt"
)

# 2. 初始化
state = predictor.init_state(
    "video.mp4", 
    offload_video_to_cpu=True
)

# 3. 添加第一帧提示
bbox = (x1, y1, x2, y2)  # 边界框
predictor.add_new_points_or_box(
    state, box=bbox, frame_idx=0, obj_id=0
)

# 4. 跟踪
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    # 处理每帧的掩码
    mask = masks[0]  # shape: (1, H, W)
```

### 从掩码提取边界框
```python
import numpy as np

mask = masks[0].cpu().numpy() > 0.0
non_zero = np.argwhere(mask)
if len(non_zero) > 0:
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
```

### 禁用SAMURAI（使用原始SAM 2）
```python
# 方法1: 使用SAM 2配置
predictor = build_sam2_video_predictor(
    "configs/sam2.1/sam2.1_hiera_b+.yaml",  # 非samurai目录
    checkpoint
)

# 方法2: 修改配置文件
# 在 yaml 中设置: samurai_mode: false
```

---

## 调试技巧

### 查看卡尔曼状态
```python
# 在 sam2_base.py 中添加打印
print(f"KF mean: {self.kf_mean}")
print(f"Stable frames: {self.stable_frames}")
print(f"Predicted bbox: {self.kf.xyah_to_xyxy(self.kf_mean[:4])}")
```

### 可视化多掩码选择
```python
# 在 _forward_sam_heads() 中
print(f"SAM IoUs: {ious}")
print(f"KF IoUs: {kf_ious}")
print(f"Weighted IoUs: {weighted_ious}")
print(f"Selected: {best_iou_inds}")
```

### 检查记忆库质量
```python
# 在 _prepare_memory_conditioned_features() 中
for idx in valid_indices:
    out = output_dict["non_cond_frame_outputs"][idx]
    print(f"Frame {idx}: IoU={out['best_iou_score']:.3f}, "
          f"Obj={out['object_score_logits']:.3f}")
```

---

## 性能基准

### 不同模型对比 (LaSOT数据集)

| 模型 | AUC | 速度 (fps) | GPU内存 |
|------|-----|-----------|---------|
| SAMURAI-T | ~68% | ~35 | 6GB |
| SAMURAI-S | ~70% | ~30 | 8GB |
| SAMURAI-B+ | ~73% | ~25 | 10GB |
| SAMURAI-L | ~74% | ~20 | 14GB |

### 优化建议

**内存受限 (<8GB):**
```yaml
model: sam2.1_hiera_t.yaml
image_size: 512
offload_video_to_cpu: true
offload_state_to_cpu: true
```

**速度优先 (>25fps):**
```yaml
model: sam2.1_hiera_s.yaml
num_maskmem: 4
memory_temporal_stride_for_eval: 2
```

**精度优先 (SOTA):**
```yaml
model: sam2.1_hiera_l.yaml
image_size: 1024
num_maskmem: 7
use_high_res_features_in_sam: true
```

---

## 错误排查

### 问题1: CUDA OOM
**解决方案:**
```python
state = predictor.init_state(
    video_path,
    offload_video_to_cpu=True,
    offload_state_to_cpu=True
)
# 或使用更小的模型 / 降低 image_size
```

### 问题2: 跟踪漂移
**原因:** `kf_score_weight` 过大或过小  
**解决方案:**
```yaml
# 快速运动 → 增大 (0.3-0.4)
# 慢速运动 → 减小 (0.1-0.2)
kf_score_weight: 0.25
```

### 问题3: 遮挡后无法恢复
**解决方案:**
```yaml
# 降低稳定性要求
stable_ious_threshold: 0.2  # 从 0.3 降低
# 提高记忆库质量过滤
memory_bank_iou_threshold: 0.6  # 从 0.5 提高
```

### 问题4: 速度太慢
**检查项:**
- [ ] 是否使用了异步加载? `async_loading_frames=True`
- [ ] 是否使用了合适的模型? (推荐 Base+ 或 Small)
- [ ] 是否启用了 GPU? `device="cuda:0"`
- [ ] 是否降低了分辨率? `image_size=512`

---

## 参数速查卡

### SAMURAI核心参数

| 参数 | 默认值 | 范围 | 作用 |
|------|--------|------|------|
| `samurai_mode` | true | bool | 启用/禁用SAMURAI |
| `kf_score_weight` | 0.25 | 0.0-0.5 | 运动预测权重 |
| `stable_frames_threshold` | 15 | 5-30 | 预热期长度 |
| `stable_ious_threshold` | 0.3 | 0.2-0.5 | 稳定判断阈值 |
| `memory_bank_iou_threshold` | 0.5 | 0.3-0.7 | 记忆库IoU过滤 |

### 模型规模参数

| 参数 | Tiny | Small | Base+ | Large |
|------|------|-------|-------|-------|
| `embed_dim` | 96 | 96 | 112 | 144 |
| `num_heads` | 2 | 2 | 2 | 4 |
| 参数量 | ~38M | ~46M | ~80M | ~224M |

### 记忆参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `num_maskmem` | 7 | 记忆帧数量 |
| `max_obj_ptrs_in_encoder` | 16 | 最大对象指针数 |
| `max_cond_frames_in_attn` | -1 | 最大条件帧数(-1=无限制) |
| `memory_temporal_stride_for_eval` | 1 | 记忆帧采样步长 |

---

## 命令行工具

### 批量推理
```bash
# LaSOT数据集
python scripts/main_inference.py

# 自定义视频
python scripts/demo.py \
    --video_path video.mp4 \
    --txt_path bbox.txt \
    --model_path sam2/checkpoints/sam2.1_hiera_base_plus.pt \
    --video_output_path output.mp4
```

### 多GPU推理
```bash
# 见 main_inference_chunk.py
python scripts/main_inference_chunk.py \
    --num_gpus 4 \
    --model base_plus
```

---

## 关键算法伪代码

### SAMURAI多掩码选择
```
输入: multi_masks (3个候选), ious (3个SAM IoU)
输出: best_mask

if 初始化或预热:
    return argmax(ious)  # 选最高IoU
else (稳定跟踪):
    kf_bbox = kalman_predict()
    kf_ious = [iou(kf_bbox, mask_bbox) for mask in multi_masks]
    weighted = 0.25 * kf_ious + 0.75 * ious
    best_idx = argmax(weighted)
    
    if ious[best_idx] > 0.3:
        kalman_update(multi_masks[best_idx])
    
    return multi_masks[best_idx]
```

### 记忆库选择
```
输入: 历史帧 [1, 2, ..., t-1]
输出: valid_frames (最多16个)

valid_frames = []
for frame in reversed([1..t-1]):
    if frame.iou > 0.5 and frame.obj_score > 0 and frame.kf_score > 0:
        valid_frames.append(frame)
    if len(valid_frames) >= 15:  # max_obj_ptrs - 1
        break

if (t-1) not in valid_frames:
    valid_frames.append(t-1)  # 确保包含最近一帧

return valid_frames
```

---

## 版本信息

- **SAMURAI版本:** 基于 SAM 2.1
- **支持的数据集:** LaSOT, LaSOT-ext, GOT-10k, UAV123, TrackingNet, OTB100
- **依赖:** PyTorch ≥2.3.1, torchvision ≥0.18.1, Python ≥3.10
- **许可证:** Apache 2.0

---

**最后更新:** 2025-10-21  
**对应详细文档:** ARCHITECTURE.md

