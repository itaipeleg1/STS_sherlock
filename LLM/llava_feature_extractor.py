import os
## Set environment variables for Hugging Face cache
## This is to avoid running out of space in the default cache location
os.environ['HF_HOME'] = '/home/new_storage/sherlock/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/new_storage/sherlock/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/new_storage/sherlock/hf_cache'
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForVision2Seq
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
    
    #if prompts is None:
     #   prompts = [
      #      ('social', "USER: <image>\nDoes this image contain social interaction between people? Answer in 1 word - yes or no..\nASSISTANT:"),
       #     ('speak', "USER: <image>\nIs there a person speaking in this image(Lips moving)? Answer in 1 word - yes or no.\nASSISTANT:"),
        #    ('gaze', "USER: <image>\nIs the person's gaze directed towards someone off-screen? Answer in 1 word - yes or no.\nASSISTANT:")
        #]

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
    
    for seq_dir in tqdm(seq_dirs, desc="Processing TR"):
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
        social_count = 0
        gaze_count = 0
        speak_count= 0
        samples_processed = len(sampled_frames)
        
        # Figure out a general use of the prompts
        for path in sampled_frames:
            image = Image.open(path)
            
            prompt1 = "USER: <image>\nDoes this image contain social interaction between people? Answer in 1 word - yes or no..\nASSISTANT:"
            outputs1 = pipe(image, prompt=prompt1, generate_kwargs={"max_new_tokens": 200})
            response1 = outputs1[0]["generated_text"].split("ASSISTANT:")[-1].strip().lower()
            prompt2 = "USER: <image>\nIs there a person speaking in this image(Lips moving)? Answear in 1 word - yes or no.\nASSISTANT:"
            outputs2 = pipe(image, prompt=prompt2, generate_kwargs={"max_new_tokens": 200})
            response2 = outputs2[0]["generated_text"].split("ASSISTANT:")[-1].strip().lower()
            prompt3 = "USER: <image>\nIs the angle of this image is 'Over the shoulder'? Answer in 1 word - yes or no.\nASSISTANT:"
            outputs3 = pipe(image, prompt=prompt3, generate_kwargs={"max_new_tokens": 200})
            response3 = outputs3[0]["generated_text"].split("ASSISTANT:")[-1].strip().lower()
            if "yes" in response1:
               social_count += 1
            if "yes" in response2:
               speak_count += 1
            if "yes" in response3:
               gaze_count += 1

        # Binary decision based on majority vote
        gaze = 1 if gaze_count > samples_processed/2 else 0
        social= 1 if social_count > samples_processed/2 else 0
        speak = 1 if speak_count > samples_processed/2 else 0


        results.append([seq_num, social,  speak, gaze, samples_processed])
        print(f"TR{seq_num:04d}: {social} ({social_count}/{samples_processed} sampled frames)")
        
        # Save intermediate results
        if seq_num % save_interval == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            print(f"\nIntermediate results saved to {output_path}")

    results_df = pd.DataFrame(results, columns=['TR', 'social',"speak", "gaze" ,'samples_processed']) ##for this case
    results_df.to_csv(output_path, index=False)
    print(f"\nFinal results saved to {output_path}")
    return results_df


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
    pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
    custom_prompts = []  ## Should be available in the future
    
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