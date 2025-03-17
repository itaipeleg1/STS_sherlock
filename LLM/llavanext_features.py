
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForCausalLM
import os
from PIL import Image
import pandas as pd
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

def analyze_frames(root_dir, pipe, 
                  samples_per_seq=13,
                  seq_prefix='TR',
                  seq_range=None,  
                  file_extension='.jpg',
                  output_path='results.csv',
                  save_interval=50,
                  prompts=None):
    
    # Default prompts If none is provided in Main function
    if prompts is None:
        prompts = [
            ('social', "USER: <image>\nDoes this image contain social interaction between people? Answer in 1 word - yes or no..\nASSISTANT:"),
            ('speak', "USER: <image>\nIs there a person speaking in this image(Lips moving)? Answer in 1 word - yes or no.\nASSISTANT:"),
            ('gaze', "USER: <image>\nIs the person's gaze directed towards someone off-screen? Answer in 1 word - yes or no.\nASSISTANT:")
        ]

    # Get all sequence directories
    seq_dirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d)) 
               and d.startswith(seq_prefix)]
    seq_dirs.sort(key=lambda x: int(x[len(seq_prefix):]))
    
    # Apply sequence range filter if specified
    if seq_range:
        start, end = seq_range
        seq_dirs = [d for d in seq_dirs 
                   if start <= int(d[len(seq_prefix):]) <= end]
    
    results = []
    
    for seq_dir in tqdm(seq_dirs, desc="Processing sequences"):
        seq_path = os.path.join(root_dir, seq_dir)
        seq_num = int(seq_dir[len(seq_prefix):])
        
        # Get and sort frame paths
        frame_paths = [
            os.path.join(seq_path, f) for f in os.listdir(seq_path)
            if os.path.isfile(os.path.join(seq_path, f)) 
            and f.endswith(file_extension)
        ]
        frame_paths.sort(key=extract_frame_number)
        
        # Sample frames evenly
        total_frames = len(frame_paths)
        indices = [
            i * (total_frames - 1) // (samples_per_seq - 1) 
            for i in range(samples_per_seq)
        ]
        sampled_frames = [frame_paths[i] for i in indices]
        
        # Initialize counters for each prompt
        feature_counts = {name: 0 for name, _ in prompts}
        samples_processed = len(sampled_frames)
        
        # Process each frame
        for path in sampled_frames:
            image = Image.open(path)
            
            # Run each prompt on the image
            for name, prompt in prompts:
                outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
                response = outputs[0]["generated_text"].split("ASSISTANT:")[-1].strip().lower()
                if "yes" in response:
                    feature_counts[name] += 1
        
        # Calculate binary decisions based on majority vote
        binary_results = {
            name: 1 if count > samples_processed/2 else 0
            for name, count in feature_counts.items()
        }
        
        # Store results
        result_row = {
            'sequence': seq_num,
            'samples_processed': samples_processed,
            **binary_results
        }
        results.append(result_row)
        
        print(f"{seq_prefix}{seq_num:04d}: Results: {binary_results}")
        
        # Save intermediate results
        if seq_num % save_interval == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            print(f"\nIntermediate results saved to {output_path}")
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nFinal results saved to {output_path}")
    return results_df

# Example usage:
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze frames from video sequences using LLaVA model')
    parser.add_argument('--TR_root', type=str, required=True, help='Root directory containing TR sequences')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results CSV')
    parser.add_argument('--start_seq', type=int, default=0, help='Starting sequence number')
    parser.add_argument('--end_seq', type=int, default=1000, help='Ending sequence number')
    parser.add_argument('--samples_per_seq', type=int, default=13, help='Number of frames to sample per sequence')
    parser.add_argument('--save_interval', type=int, default=50, help='Save intermediate results every N sequences')
    
    args = parser.parse_args()

    # Configure model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model_id = "llava-hf/llava-1.5-7b-hf"
    pipe = pipeline("image-to-text", 
                   model=model_id, 
                   model_kwargs={"quantization_config": quantization_config})

    # Optional: Define custom prompts
    custom_prompts = []
    
    # Run analysis
    results_df = analyze_frames(
        root_dir=args.TR_root,
        pipe=pipe,  
        seq_range=(args.start_seq, args.end_seq),
        output_path=args.output_path,
        samples_per_seq=args.samples_per_seq,
        save_interval=args.save_interval,
        prompts=custom_prompts
    )