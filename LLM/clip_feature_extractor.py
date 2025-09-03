import os

os.environ['HF_HOME'] = '/home/new_storage/sherlock/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/new_storage/sherlock/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/new_storage/sherlock/hf_cache'

import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

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
    
    i = 0
    while i <= len(seq_dirs) - tr_ref:
        clip_embeddings = []
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
        BATCHSIZE = 8
        for batch_start in range(0, len(sampled_frames), BATCHSIZE):
            batch_paths = sampled_frames[batch_start:batch_start + BATCHSIZE]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            
            # Process images with CLIP
            image_tensors = torch.stack([preprocess(image) for image in images]).to(device)

            with torch.no_grad():
                # Get image embeddings from CLIP
                image_features = model.encode_image(image_tensors)

                # Normalize features (CLIP typically uses normalized embeddings)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                print(f"[DEBUG] CLIP batch shape: {image_features.shape}")
                print(f"[DEBUG] CLIP batch std: {image_features.std().item():.4f}")
                
                # Store embeddings
                for embedding in image_features:
                    clip_embeddings.append(embedding.cpu().numpy())
        
        if clip_embeddings:
            # Average all embeddings for this TR group
            avg_embedding = np.mean(np.stack(clip_embeddings), axis=0)
            print(f"[DEBUG] Saving CLIP embedding for group {group_label}, mean: {avg_embedding.mean():.4f}, std: {avg_embedding.std():.4f}")
            
            # Create output directory if it doesn't exist
            output_dir = "/home/new_storage/sherlock/STS_sherlock/projects data/clip_embeddings"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the averaged embedding
            np.save(f"{output_dir}/{group_label}_clip.npy", avg_embedding)
            
            # Store result for tracking
            results.append([group_label, samples_processed, avg_embedding.shape[0]])
            
            # Save intermediate results
            if len(results) % save_interval == 0:
                results_df = pd.DataFrame(results, columns=['TR', 'samples_processed', 'embedding_dim'])
                #results_df.to_csv(output_path, index=False)
                print(f"\nIntermediate results saved to {output_path}")
        else:
            print(f"Warning: No embeddings generated for group {group_label}")
        
        i += tr_ref  # No overlap between groups

    # Save final results
    results_df = pd.DataFrame(results, columns=['TR', 'samples_processed', 'embedding_dim'])
    #results_df.to_csv(output_path, index=False)
    print(f"\nFinal results saved to {output_path}")
    
    return results_df

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract CLIP embeddings from video sequences')
    parser.add_argument('--TR_root', type=str, default="/home/new_storage/sherlock/data/frames", 
                       help='Root directory containing TR sequences')
    parser.add_argument('--output_path', type=str, 
                       default="/home/new_storage/sherlock/STS_sherlock/projects data/annotations/clip_embeddingscsv",
                       help='Path to save results CSV')
    parser.add_argument('--start_seq', type=int, default=0, help='Starting sequence number')
    parser.add_argument('--end_seq', type=int, default=919, help='Ending sequence number')
    parser.add_argument('--samples_per_seq', type=int, default=8, help='Number of frames to sample per sequence')
    parser.add_argument('--tr_ref', type=int, default=1, help='How big is the reference TR')
    parser.add_argument('--save_interval', type=int, default=50, help='Save intermediate results every N sequences')
    parser.add_argument('--model_name', type=str, default="openai/clip-vit-base-patch32", 
                       help='CLIP model to use')
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # Load CLIP model and processor
    print(f"Loading CLIP model: {args.model_name}")
    model, preprocess = clip.load('ViT-L/14@336px', device=device)
    model.eval()
    
    # Run analysis
    results_df = analyze_frames(
        root_dir=args.TR_root,
        model=model,
        processor=preprocess,
        tr_ref=args.tr_ref, 
        seq_range=(args.start_seq, args.end_seq),
        output_path=args.output_path,
        samples_per_seq=args.samples_per_seq,
        save_interval=args.save_interval,
    )
    
    print(f"Processing complete! Generated {len(results_df)} CLIP embeddings.")
    print(f"Embeddings saved to: /home/new_storage/sherlock/STS_sherlock/projects data/clip_embeddings/")