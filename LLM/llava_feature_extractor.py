
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

def analyze_frames(root_dir,model,processor,tr_ref, 
                  samples_per_seq=8,
                  seq_prefix='TR',
                  seq_range=None,  
                  file_extension='.jpg',
                  output_path='results.csv',
                  save_interval=50,
                    ):
    

    samples_per_seq = samples_per_seq*tr_ref
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
    while i <= len(seq_dirs)-tr_ref:
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
        indices = [
            i * (total_frames - 1) // (samples_per_seq - 1) 
            for i in range(samples_per_seq)
        ]
        sampled_frames = [frame_paths[i] for i in indices]
        
        # Initialize counters for each prompt
        social_count = 0
        gaze_count = 0
        speak_count= 0
        object_count = 0
        samples_processed = len(sampled_frames)
        final=0
        ## I need to figure out how to use prompts in a general way
        BATCHSIZE = 8
        prompt1 = f"USER: <image>\nIs there people in this image that appear to be engaging with each other socially (e.g., talking, making eye contact, or interacting)? Answer 'yes' or 'no'.\nASSISTANT:"
        prompt2 = f"USER: <image>\nIs there a person whose gaze is directed towards someone off-screen in this image? Answer 'yes' or 'no'.\nASSISTANT:"
        prompt3 = f"USER: <image>\nIs there a person in this image who appears to be speaking or making a gesture that suggests communication? Answer 'yes' or 'no'.\nASSISTANT:"
        for batch_start in range(0, len(sampled_frames), BATCHSIZE):
            batch_paths = sampled_frames[batch_start:batch_start + BATCHSIZE]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            prompts1 = [prompt1] * len(images)  # Repeat prompt for each image in batch
            prompts2 = [prompt2] * len(images)  # Repeat prompt for each image in batch
            prompts3 = [prompt3] * len(images)  # Repeat prompt for each image
            inputs1 = processor(images=images, text=prompts1, return_tensors="pt").to(model.device)
            inputs2 = processor(images=images, text=prompts2, return_tensors="pt").to(model.device)
            inputs3 = processor(images=images, text=prompts3, return_tensors="pt").to(model.device)
            # Generate with logits returned
            with torch.no_grad():
                outputs1 = model.generate(
                    **inputs1,
                    max_new_tokens=1,
                )
                outputs2 = model.generate(
                    **inputs2,
                    max_new_tokens=1,
                )
                outputs3 = model.generate(
                    **inputs3,
                    max_new_tokens=1,
                )

            generated_text1 = processor.batch_decode(outputs1, skip_special_tokens=True)
            generated_text2 = processor.batch_decode(outputs2, skip_special_tokens=True)
            generated_text3 = processor.batch_decode(outputs3, skip_special_tokens=True)
            generated_text1 = [t.split("ASSISTANT:")[-1].strip() for t in generated_text1]
            generated_text2 = [t.split("ASSISTANT:")[-1].strip() for t in generated_text2]
            generated_text3 = [t.split("ASSISTANT:")[-1].strip() for t in generated_text3]
            print(generated_text1)
            print(generated_text2)
            print(generated_text3)
            for text1, text2, text3 in zip(generated_text1, generated_text2, generated_text3):
                # Process the generated text to determine the label
                if "yes" in text1.lower():
                    social_count += 1
                if "yes" in text2.lower():
                    gaze_count += 1
                if "yes" in text3.lower():
                    speak_count += 1
                
        print(f"TR{group_label}:  Social: {social_count}, Gaze: {gaze_count}, Speak: {speak_count}")

        threshold = 0.5
        social= 1 if social_count > threshold*samples_processed else 0
        speak = 1 if speak_count > threshold*samples_processed else 0
        gaze = 1 if gaze_count > threshold*samples_processed else 0
        if social == 1:
            final = 1
        elif speak == 1 and gaze == 1:
            final = 1
        else:
            final = 0

        results.append([group_label, social,speak,gaze,final, samples_processed])

        
        # Save intermediate results
        if len(results) % save_interval == 0:
            
            results_df = pd.DataFrame(results,columns=['TR', 'social', 'speak', 'gaze', 'final', 'samples_processed'])
            results_df.to_csv(output_path, index=False)
            print(f"\nIntermediate results saved to {output_path}")
        
        i += tr_ref #  no overlap between groups

    results_df = pd.DataFrame(results,columns=['TR', 'social', 'speak', 'gaze', 'final', 'samples_processed'])

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze frames from video sequences using LLaVA model')
    parser.add_argument('--TR_root', type=str,  help='Root directory containing TR sequences')
    parser.add_argument('--output_path', type=str, help='Path to save results CSV')
    parser.add_argument('--start_seq', type=int, default=0, help='Starting sequence number')
    parser.add_argument('--end_seq', type=int, default=5469, help='Ending sequence number')
    parser.add_argument('--samples_per_seq', type=int, default=8, help='Number of frames to sample per sequence')
    parser.add_argument('--tr_ref', type=int,default=1, help='How big is the reference TR')
    parser.add_argument('--save_interval', type=int, default=50, help='Save intermediate results every N sequences')
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    # Configure model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
        )
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    model.eval()
    custom_prompts = []  ## Should be available in the future


        # Run analysis
    output = f"/home/new_storage/sherlock/STS_sherlock/500days/data/annotations_from_models/llava_social_500days{1}TR.csv"
    results_df = analyze_frames(
            root_dir="/home/new_storage/sherlock/STS_sherlock/500days/data/frames",
            model=model,
            processor=processor,
            tr_ref=1, 
            seq_range=(args.start_seq, args.end_seq),
            output_path=output,
            samples_per_seq=args.samples_per_seq,
            save_interval=args.save_interval,
        )