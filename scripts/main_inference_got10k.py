import cv2
import gc
import glob
import numpy as np
import os
import os.path as osp
import torch
import time
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm


def load_got10k_gt(gt_path):
    """Load GOT-10k groundtruth file"""
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    
    # bbox in first frame are prompts
    prompts = {}
    fid = 0
    for line in gt:
        # GOT-10k format: x,y,w,h (can be float)
        x, y, w, h = map(float, line.strip().split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x+w, y+h), 0)  # Convert to xyxy format
        fid += 1

    return prompts


color = [
    (255, 0, 0),
]

# GOT-10k dataset path
video_folder = "/home/tau/datasets/test"

exp_name = "samurai"
model_name = "base_plus"

checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
if model_name == "base_plus":
    model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
else:
    model_cfg = f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"

pred_folder = f"results/GOT10k/{exp_name}/{exp_name}_{model_name}"

save_to_video = False
if save_to_video:
    vis_folder = f"visualization/GOT10k/{exp_name}/{model_name}"
    os.makedirs(vis_folder, exist_ok=True)

# Get all test videos using glob
test_videos = sorted(glob.glob(f"{video_folder}/*"))
print(f"Found {len(test_videos)} videos in {video_folder}")

# Build predictor once and reuse it for all videos
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")
print("Model loaded successfully")
start_time = time.time()

for vid, video_path in enumerate(test_videos):
    video_basename = osp.basename(video_path)
    frame_folder = video_path  # GOT-10k frames are directly in video folder
    
    # Get all jpg files in the folder
    frames = sorted(glob.glob(f"{frame_folder}/*.jpg"))
    if len(frames) == 0:
        print(f"Warning: No frames found in {frame_folder}, skipping...")
        continue
    
    num_frames = len(frames)

    print(f"\033[91mRunning video [{vid+1}/{len(test_videos)}]: {video_basename} with {num_frames} frames\033[0m")

    # Read first frame to get dimensions
    first_frame_path = frames[0]
    height, width = cv2.imread(first_frame_path).shape[:2]

    predictions = []

    if save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(osp.join(vis_folder, f'{video_basename}.mp4'), fourcc, 30, (width, height))

    # Start processing frames
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frame_folder, offload_video_to_cpu=False, offload_state_to_cpu=False, async_loading_frames=True)

        # Load groundtruth
        gt_path = osp.join(video_path, "groundtruth.txt")
        if not osp.exists(gt_path):
            print(f"Warning: {gt_path} not found, skipping...")
            continue
            
        prompts = load_got10k_gt(gt_path)

        # Add first frame prompt
        bbox, track_label = prompts[0]
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            assert len(masks) == 1 and len(object_ids) == 1, "Only one object is supported right now"
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if save_to_video:
                # GOT-10k frame naming: 00000001.jpg, 00000002.jpg, ...
                img = cv2.imread(f'{frame_folder}/{frame_idx+1:08d}.jpg') 
                if img is None:
                    break
                
                # Draw mask overlay
                for obj_id in mask_to_vis.keys():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask_to_vis[obj_id]] = color[(obj_id+1)%len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.75, 0)
                
                # Draw predicted bbox (red)
                for obj_id in bbox_to_vis.keys():
                    cv2.rectangle(img, (bbox_to_vis[obj_id][0], bbox_to_vis[obj_id][1]), 
                                  (bbox_to_vis[obj_id][0]+bbox_to_vis[obj_id][2], 
                                   bbox_to_vis[obj_id][1]+bbox_to_vis[obj_id][3]), 
                                  color[(obj_id)%len(color)], 2)
                
                # Draw groundtruth bbox (green)
                if frame_idx in prompts:
                    x1, y1, x2, y2 = prompts[frame_idx][0]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                out.write(img)

            predictions.append(bbox_to_vis)        
        
    # Save predictions
    os.makedirs(pred_folder, exist_ok=True)
    with open(osp.join(pred_folder, f'{video_basename}.txt'), 'w') as f:
        for pred in predictions:
            x, y, w, h = pred[0]
            f.write(f"{x},{y},{w},{h}\n")

    if save_to_video:
        out.release() 

    # Clean up state for this video
    del state
    gc.collect()
    torch.cuda.empty_cache()

print("\033[92m✓ Inference completed!\033[0m")
end_time = time.time()
total_time = end_time - start_time
print(f"\u63a8\u7406 GOT-10k \u6570\u636e\u96c6\u603b\u8017\u65f6: {total_time:.2f} \u79d2")
print(f"Results saved to: {pred_folder}")
if save_to_video:
    print(f"Visualizations saved to: {vis_folder}")
