import os

os.environ['HF_HOME'] = '/home/new_storage/sherlock/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/new_storage/sherlock/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/new_storage/sherlock/hf_cache'
import cv2
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import re

def parse_answer(response):
    match = re.search(r'<answer>(\d+)</answer>', response)
    if match:
        return int(match.group(1))
    return None  # or -1 if no answer found

def extract_frame_number(filepath):
    """Extract frame number from filepath."""
    try:
        filename = os.path.basename(filepath)
        number_part = filename.split("_")[-1].split(".")[0]
        clean_number = ''.join(c for c in number_part if c.isdigit())
        return int(clean_number)
    except (ValueError, IndexError):
        return -1

def analyze_frames(root_dir, model, processor, tr_ref,
                  samples_per_seq=8,
                  seq_prefix='TR',
                  seq_range=None,
                  file_extension='.jpg',
                  output_path='results.csv',
                  save_interval=50,
                  device='cuda',
                  ):
    
    samples_per_seq = samples_per_seq * tr_ref
    # Get all TR directories
    seq_dirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d)) 
               and d.startswith(seq_prefix)] 
    seq_dirs.sort(key=lambda x: int(x[len(seq_prefix):]))
    
    # Apply TR range (If specified in arguments)
    if seq_range:
        start, end = seq_range
        seq_dirs = [d for d in seq_dirs 
                   if start <= int(d[len(seq_prefix):]) <= end]
    
    results = []
    probe_similarities_list = []
    
    i = 0
    while i <= len(seq_dirs) - tr_ref:
        
        group_dirs = seq_dirs[i:i+tr_ref]
        group_nums = [int(d[len(seq_prefix):]) for d in group_dirs]
        group_label = f"{group_nums[0]:04d}_{group_nums[-1]:04d}"
        print(f"Processing group: {group_label}")
        
        # Get and sort frame paths
        frame_paths = []
        for seq_dir in group_dirs:
            seq_path = os.path.join(root_dir, seq_dir)
            frames = [os.path.join(seq_path, f) for f in os.listdir(seq_path)
                    if f.endswith(file_extension)]
            frames.sort(key=extract_frame_number)
            frame_paths.extend(frames)
        
        # Sample frames evenly
        total_frames = len(frame_paths)
        if total_frames == 0:
            print(f"Warning: No frames found for group {group_label}")
            i += tr_ref
            continue
            
        indices = [
            i * (total_frames - 1) // (samples_per_seq - 1) 
            for i in range(samples_per_seq)
        ]
        sampled_frames = [frame_paths[i] for i in indices]
        
        samples_processed = len(sampled_frames)
        
        # Process frames in batches
        group_scores = []
        BATCHSIZE = 1
        for batch_start in range(0, len(sampled_frames), BATCHSIZE):
            batch_paths = sampled_frames[batch_start:batch_start + BATCHSIZE]
            images = [Image.open(path).convert("RGB") for path in batch_paths]

            ## images are loaded in batches 
            ## define prompts 
            SYSTEM_PROMPT = (
                "A conversation between User and Assistant. The user asks a question, and the Assistant "
                "solves it. The assistant first thinks about the reasoning process in the mind and then "
                "provides the user with the answer. The reasoning process and answer are enclosed within "
                "<think> </think> and <answer> </answer> tags."
            )

            texts = []
            all_image_inputs = []
            for img in images:
                msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What's the shot size of this shot? Please select the most likely answer from the options: 5 = Extreme Close-Up, 4 = Close-Up, 3 = Medium Shot, 2 = Long Shot, 1 = Extreme Long Shot."},
                    #{"type": "text", "text": "What is the narrative purpose of this shot?? Please select the most likely answer from the options: 5 =  Showing critical detail, 4 = Emphasizing character emotion, 3 = Normal conversation/interaction, 2 = Establishing location/context, 1 = Transition moment / Landscape ."},
                    ],
                },
                ]

                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                texts.append(text)
                image_inputs, _ = process_vision_info(msgs)
                all_image_inputs.extend(image_inputs)
            inputs = processor(
                    text=texts,
                    images=all_image_inputs,
                    padding=True,
                    return_tensors="pt",
                    ).to(device)
            
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=640)
            trimmed =  [o[len(i):] for i, o in zip(inputs.input_ids, outputs)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            print(decoded)

            match = re.search(r'<answer>(\d+)</answer>', decoded)
            score = int(match.group(1)) if match else None
            if score is not None:
                    print(f"Extracted shot narrative score: {score}")
                    group_scores.append(score)
        avg_score = np.mean([s for s in group_scores if s is not None])
        results.append({
            'group_label': group_label,
            'avg_shot_narrative_score': avg_score,
            'num_samples': samples_processed,
        })
            

        i += tr_ref
        # Save intermediate results
        if (i // tr_ref) % save_interval == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{output_path}_shot_narrative_scores.csv", index=False)
            print(f"Intermediate results saved to {output_path}")
    # Save final results
    results_df = pd.DataFrame(results)
    zoom = np.array([r['avg_shot_narrative_score'] for r in results])
    np.save(f"{output_path}_shot_narrative_scores.npy", zoom)
    
    


    


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract CLIP embeddings from video sequences')
    parser.add_argument('--TR_root', type=str, default="/home/new_storage/sherlock/data/frames", 
                       help='Root directory containing TR sequences')
    parser.add_argument('--output_path', type=str, 
                       default="/home/new_storage/sherlock/STS_sherlock/projects data/annotations/",
                       help='Path to save results CSV')
    parser.add_argument('--start_seq', type=int, default=0, help='Starting sequence number')
    parser.add_argument('--end_seq', type=int, default=1976, help='Ending sequence number')
    parser.add_argument('--samples_per_seq', type=int, default=8, help='Number of frames to sample per sequence')
    parser.add_argument('--tr_ref', type=int, default=1, help='How big is the reference TR')
    parser.add_argument('--save_interval', type=int, default=1, help='Save intermediate results every N sequences')
    parser.add_argument('--model_name', type=str, default="openai/clip-vit-base-patch32", 
                       help='CLIP model to use')
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    

    print(f"Loading model: {args.model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        "Vchitect/ShotVL-3B",  # 3B instead of 7B
        device_map="balanced",
        torch_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(
        "Vchitect/ShotVL-3B",
        use_fast=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()


    # Run analysis
    results_df = analyze_frames(
        root_dir=args.TR_root,
        model=model,
        processor=processor,
        tr_ref=args.tr_ref,
        seq_range=(args.start_seq, args.end_seq),
        output_path=args.output_path,
        samples_per_seq=args.samples_per_seq,
        save_interval=args.save_interval,
        device=device,
    )
    
