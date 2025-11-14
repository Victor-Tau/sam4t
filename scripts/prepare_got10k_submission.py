"""
GOT-10k Submission Preparation Script

This script converts SAMURAI inference results to GOT-10k submission format:
- Creates individual folder for each sequence
- Renames result file to {sequence_name}_001.txt
- Packages everything into submit.zip

Usage:
    python scripts/prepare_got10k_submission.py --exp_name samurai --model_name base_plus
"""

import argparse
import numpy as np
import os
import os.path as osp
import shutil


def prepare_got10k_submission(exp_name, model_name):
    """
    Transform GOT-10k tracking results to official submission format.
    
    Args:
        exp_name: Experiment name (e.g., 'samurai')
        model_name: Model name (e.g., 'base_plus')
    """
    # Source directory with inference results
    src_dir = f"results/GOT10k/{exp_name}/{exp_name}_{model_name}"
    
    # Destination directory for submission format
    dest_dir = f"results/GOT10k/{exp_name}/{exp_name}_{model_name}_submit"
    
    if not osp.exists(src_dir):
        print(f"\033[91mError: Source directory not found: {src_dir}\033[0m")
        print("Please run main_inference_got10k.py first to generate results.")
        return
    
    # Create destination directory
    if osp.exists(dest_dir):
        print(f"Removing existing submission folder: {dest_dir}")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    
    # Get all result files
    result_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.txt')])
    
    if len(result_files) == 0:
        print(f"\033[91mError: No result files found in {src_dir}\033[0m")
        return
    
    print(f"\033[94mProcessing {len(result_files)} sequences...\033[0m")
    
    for idx, result_file in enumerate(result_files):
        # Get sequence name (remove .txt extension)
        seq_name = result_file.replace('.txt', '')
        
        # Create sequence folder
        seq_dir = osp.join(dest_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        
        # Source and destination paths
        src_path = osp.join(src_dir, result_file)
        dest_file = f"{seq_name}_001.txt"
        dest_path = osp.join(seq_dir, dest_file)
        
        # Load bbox array and save in GOT-10k format
        # Our format is already x,y,w,h with comma delimiter
        try:
            bbox_arr = np.loadtxt(src_path, dtype=np.float32, delimiter=',')
            # GOT-10k requires integer format
            bbox_arr = bbox_arr.astype(np.int32)
            np.savetxt(dest_path, bbox_arr, fmt='%d', delimiter=',')
            
            # Generate time file (required by GOT-10k)
            time_file = f"{seq_name}_time.txt"
            time_path = osp.join(seq_dir, time_file)
            num_frames = len(bbox_arr)
            # Set all frame times to 0.1 seconds
            time_arr = np.ones(num_frames, dtype=np.float32) * 0.1
            np.savetxt(time_path, time_arr, fmt='%.6f')
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(result_files)} sequences...")
        except Exception as e:
            print(f"\033[91mError processing {result_file}: {e}\033[0m")
            continue
    
    print(f"\033[92m✓ All sequences processed!\033[0m")
    
    # Create zip archive
    print(f"\033[94mCreating zip archive...\033[0m")
    zip_path = osp.join(".", "submit")  # Will create submit.zip
    shutil.make_archive(zip_path, "zip", dest_dir)
    
    print(f"\033[92m✓ Submission package created: submit.zip\033[0m")
    print(f"\nSubmission structure:")
    print(f"  submit.zip")
    print(f"  ├── {result_files[0].replace('.txt', '')}/")
    print(f"  │   ├── {result_files[0].replace('.txt', '')}_001.txt")
    print(f"  │   └── {result_files[0].replace('.txt', '')}_time.txt")
    print(f"  ├── ...")
    print(f"  └── {result_files[-1].replace('.txt', '')}/")
    print(f"      ├── {result_files[-1].replace('.txt', '')}_001.txt")
    print(f"      └── {result_files[-1].replace('.txt', '')}_time.txt")
    print(f"\nTotal sequences: {len(result_files)}")
    print(f"Each sequence includes tracking results and timing file (0.1s per frame)")
    print(f"\nYou can now submit 'submit.zip' to GOT-10k benchmark!")


def main():
    parser = argparse.ArgumentParser(description='Prepare GOT-10k submission package')
    parser.add_argument('--exp_name', type=str, default='camera',
                        help='Experiment name (default: samurai)')
    parser.add_argument('--model_name', type=str, default='base_plus',
                        help='Model name (default: base_plus)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GOT-10k Submission Preparation")
    print("=" * 70)
    print(f"Experiment: {args.exp_name}")
    print(f"Model: {args.model_name}")
    print("=" * 70)
    
    prepare_got10k_submission(args.exp_name, args.model_name)


if __name__ == "__main__":
    main()

