
import os
import sys
## Set environment variables for Hugging Face cache
## This is to avoid running out of space in the default cache location
os.environ['HF_HOME'] = '/home/new_storage/sherlock/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/new_storage/sherlock/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/new_storage/sherlock/hf_cache'

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
sys.path.append(src_path)
sys.path.append('/home/new_storage/sherlock/llava-interp')

from src.HookedLVLM import HookedLVLM
import torch
import torch.nn.functional as F
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

def analyze_frames(root_dir,model,tokenizer,
                   tr_ref, 
                    layer,
                  samples_per_seq=8,
                  seq_prefix='TR',
                  seq_range=None,  
                  file_extension='.jpg',
                  output_path='results.csv',
                  save_interval=50,
                    ):

    tokenizer = tokenizer
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
        language_latent = []
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
            idx * (total_frames - 1) // (samples_per_seq - 1)
            for idx in range(samples_per_seq)
        ]
        sampled_frames = [frame_paths[idx] for idx in indices]
        
        # Get model components and configure

        norm = model.model.language_model.model.norm
        lm_head = model.model.language_model.lm_head
        embedding_layer = model.model.language_model.model.embed_tokens
        samples_processed = len(sampled_frames)
        prompt = "USER: <image>\nIs there social interaction in this image?.\nASSISTANT:"
        keyword = "social"

        # Process images one at a time (no batching)
        for frame_path in sampled_frames:
            image = Image.open(frame_path).convert("RGB")

            # Forward pass to get hidden states
            outputs = model.forward(image, prompt, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            ## Extract hidden states at layer 25
            layer_hidden = hidden_states[layer]  # shape (1, seq_len, hidden_dim)
            print(f"Layer {layer} hidden states shape: {layer_hidden.shape}")

            ## Encode the keyword
            print(f"\n=== Analyzing similarity to keyword: '{keyword}' ===")
            keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)

            ## Get key word embedding
            keyword_tensor = torch.tensor(keyword_tokens).to(model.model.device)
            keyword_embeddings = embedding_layer(keyword_tensor)
            keyword_embeddings = F.normalize(keyword_embeddings, dim=-1)

            ## For each IMAGE token position, compute what it predicts and its similarity to keyword
            num_image_tokens = 576  # Assuming 576 image tokens for 336x336
            all_similarities = []
            all_predicted_tokens = []

            for pos in range(num_image_tokens):
                token_hidden = layer_hidden[0, pos, :].unsqueeze(0)  # shape (1, hidden_dim)

                # Compute logits for this token position
                logits = norm(token_hidden)
                logits = lm_head(logits)  # shape (1, vocab_size)

                # Get predicted token
                predicted_token_id = logits.argmax(dim=-1).item()
                predicted_token = tokenizer.decode([predicted_token_id])

                # Get embedding of predicted token
                predicted_token_embedding = embedding_layer(torch.tensor([predicted_token_id]).to(model.model.device))

                # Normalize predicted token embedding
                predicted_token_embedding_norm = F.normalize(predicted_token_embedding, dim=-1)

                # Compute cosine similarity between predicted token and keyword
                similarity = torch.matmul(predicted_token_embedding_norm, keyword_embeddings.T)

                # Average across keyword tokens if multiple
                if similarity.dim() > 1:
                    similarity = similarity.mean(dim=-1)

                all_similarities.append(similarity.item())
                all_predicted_tokens.append(predicted_token)

            # Convert to tensor
            all_similarities = torch.tensor(all_similarities)

            # Get top 10 IMAGE token positions by similarity to keyword
            top_10_values, top_10_indices = torch.topk(all_similarities, k=min(10, num_image_tokens))

            # Collect embeddings of top tokens
            top_embeddings = []
            top_words = []
            for pos_idx in top_10_indices:
                pos = pos_idx.item()
                token_hidden = layer_hidden[0, pos, :]  # shape (hidden_dim,)
                top_embeddings.append(token_hidden.cpu().numpy())
                top_words.append(all_predicted_tokens[pos])

            # Average the top 10 token embeddings
            if top_embeddings:
                top_embeddings_stack = np.stack(top_embeddings)  # shape (10, hidden_dim)
                mean_embedding = np.mean(top_embeddings_stack, axis=0)  # shape (hidden_dim,)
                language_latent.append(mean_embedding)

                ## print the top 10 words
                print(f"\n=== Top 10 predicted words most similar to '{keyword}' ===")
                for idx, (word, score) in enumerate(zip(top_words, top_10_values)):
                    print(f"{idx+1}. '{word}' (similarity: {score.item():.4f})")

        if language_latent:
            # Stack all frame embeddings into a matrix (num_frames, hidden_dim)
            embeddings_matrix = np.stack(language_latent, axis=0)

            # Compute average across all frames
            avg = np.mean(embeddings_matrix, axis=0)

            print(f"[DEBUG] Saving latent for group {group_label}")
            print(f"[DEBUG] Matrix shape: {embeddings_matrix.shape}, mean: {avg.mean():.4f}, std: {avg.std():.4f}")

            # Save the full matrix of embeddings (one row per frame)
           

            # Save the averaged vector
            np.save(f"/home/new_storage/sherlock/STS_sherlock/projects data/CLS_social_layer25/{group_label}_latent.npy", avg)
        
        i += tr_ref #  no overlap between groups



if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze frames from video sequences using LLaVA model')
    parser.add_argument('--TR_root', type=str,  help='Root directory containing TR sequences')
    parser.add_argument('--output_path', type=str, help='Path to save results CSV')
    parser.add_argument('--start_seq', type=int, default=222, help='Starting sequence number')
    parser.add_argument('--end_seq', type=int, default=919, help='Ending sequence number')
    parser.add_argument('--samples_per_seq', type=int, default=8, help='Number of frames to sample per sequence')
    parser.add_argument('--tr_ref', type=int,default=1, help='How big is the reference TR')
    parser.add_argument('--save_interval', type=int, default=50, help='Save intermediate results every N sequences')
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    model = HookedLVLM(device=device, quantize=True, quantize_type="4bit")
    tokenizer = model.processor.tokenizer


        # Run analysis
    output = f"/home/new_storage/sherlock/STS_sherlock/projects data/annotations/llava_social_Sherlock{1}TR.csv"
    results_df = analyze_frames(
            root_dir="/home/new_storage/sherlock/data/frames",
            model=model,
            tokenizer=tokenizer,
            tr_ref=1,layer=25, 
            seq_range=(args.start_seq, args.end_seq),
            output_path=output,
            samples_per_seq=args.samples_per_seq,
            save_interval=args.save_interval,
        )