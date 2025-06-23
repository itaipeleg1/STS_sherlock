
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
                  samples_per_seq=13,
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
        BATCHSIZE = 13
        prompt1 = f"USER: <image>\nIs there a human face visible in this frame? Respond only with 'yes' or 'no'.\nASSISTANT:"
        for batch_start in range(0, len(sampled_frames), BATCHSIZE):
            batch_paths = sampled_frames[batch_start:batch_start + BATCHSIZE]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            prompts = [prompt1] * len(images)  # Repeat prompt for each image in batch

            inputs = processor(images=images, text=prompts, return_tensors="pt").to(model.device)

            # Generate with logits returned
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True, temperature=0
                )
            tokenizer = processor.tokenizer
            yes_prob_agg = 0.0
            no_prob_agg = 0.0
            logits_tensor = outputs.scores[0] ## size of  logits is (batch_size, vocab_size)
            generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            for l, path in enumerate(batch_paths):


                response1 = generated_text[l].split("ASSISTANT:")[-1].strip().lower()
            # Get logits of first generated token
                
                probs = torch.softmax(logits_tensor[l], dim=-1)

                # Get correct token id
                    # Score both 'yes' and 'Yes'
                yes_token_lower = tokenizer.tokenize("yes")[0]
                yes_token_upper = tokenizer.tokenize("Yes")[0]
                no_token_lower = tokenizer.tokenize("no")[0]
                no_token_upper = tokenizer.tokenize("No")[0]
                yes_id_lower = tokenizer.convert_tokens_to_ids(yes_token_lower)
                yes_id_upper = tokenizer.convert_tokens_to_ids(yes_token_upper)
                no_id_lower = tokenizer.convert_tokens_to_ids(no_token_lower)
                no_id_upper = tokenizer.convert_tokens_to_ids(no_token_upper)

                yes_prob = probs[yes_id_lower].item() + probs[yes_id_upper].item()
                no_prob = probs[no_id_lower].item() + probs[no_id_upper].item()
                print(f"Prob for 'Yes': {yes_prob:.4f} image: {path} | Response: {response1}")
                print(f"Response: {response1}")
                yes_prob_agg += yes_prob
                no_prob_agg += no_prob
                #if  yes_prob > 0.4:
                 #   social_count += 1
        threshold = 0.5
        # Binary decision based on majority vote
       # gaze = 1 if gaze_count > threshold*samples_processed else 0
        social= 1 if social_count > threshold*samples_processed else 0
       # speak = 1 if speak_count > threshold*samples_processed else 0
       # object_c= 1 if object_count > threshold*samples_processed else 0
        if social ==1: 
            final = 1
        #elif speak==1 and gaze==1:
        
         #   final = 1
        else:
            final = 0
        ## Fixing for probs aggregation
        yes_prob_agg /= samples_processed
        no_prob_agg /= samples_processed
        results.append([group_label, yes_prob_agg, yes_prob_agg - no_prob_agg, samples_processed])
       # results.append([group_label, social,  final,samples_processed])
        print(f"TR{group_label}: {social} ({social_count}/{samples_processed} sampled frames)")
        
        # Save intermediate results
        if len(results) % save_interval == 0:
            
            results_df = pd.DataFrame(results,columns=['TR', 'Probabillity for yes',"Probabillity for yes minus no" ,'samples_processed'])
            results_df.to_csv(output_path, index=False)
            print(f"\nIntermediate results saved to {output_path}")
        
        i += tr_ref #  no overlap between groups

    results_df = pd.DataFrame(results, columns=['TR', 'Probabillity for yes',"Probabillity for yes minus no" ,'samples_processed']) ##for this case
    #annotation = results_df["final"]
    #annotation = np.array(annotation)
    ## duplicate the annotation
    #annotation = np.repeat(annotation, tr_ref)

    ## add the aniamtion "let's all go to the movies" and save the annotation array
    ## 27 TR sequences
    #orig = np.load("/home/new_storage/sherlock/STS_sherlock/projects data/annotations/social_nonsocial.npy")
    #orig = orig.flatten()
    #anima = orig[:27]
    #annotation = np.concatenate((anima,annotation), axis=0)
    #t1 = annotation[:946]
    #t2 = annotation[946:]
    #annotation = np.concatenate([t1,anima,t2])
    #annotation = annotation[:1976]
    #annotation = np.reshape(annotation, (-1, 1))
    #np.save(os.path.join(root_dir, f'llava_pics_face(TR{tr_ref}).npy'), annotation)
    #results_df.to_csv(output_path, index=False)
    #print(f"\nFinal results saved to {output_path}")
    #print(f"\nNumpy of annotation is saved to {os.path.join(output_path, f'llava_pics_face(TR{tr_ref})2.npy')} with shape {annotation.shape}")
    #return results_df


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze frames from video sequences using LLaVA model')
    parser.add_argument('--TR_root', type=str, required=True, help='Root directory containing TR sequences')
    parser.add_argument('--output_path', type=str, help='Path to save results CSV')
    parser.add_argument('--start_seq', type=int, default=0, help='Starting sequence number')
    parser.add_argument('--end_seq', type=int, default=1950, help='Ending sequence number')
    parser.add_argument('--samples_per_seq', type=int, default=13, help='Number of frames to sample per sequence')
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
    output = f"/home/new_storage/sherlock/data/annotations_from_models/llava_face_pics_{1}TR_Onlylogits.csv"
    results_df = analyze_frames(
            root_dir="/home/new_storage/sherlock/data/frames",
            model=model,
            processor=processor,
            tr_ref=1, 
            seq_range=(args.start_seq, args.end_seq),
            output_path=output,
            samples_per_seq=args.samples_per_seq,
            save_interval=args.save_interval,
        )