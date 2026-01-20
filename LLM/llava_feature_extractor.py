
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

def compute_attention_rollout(attentions, start_layer=0, end_layer=None):
    """
    Compute attention rollout from start_layer to end_layer.

    Uses a simpler, more stable approach: just use the last layer's attention
    instead of rolling through all layers (which can be numerically unstable).

    Args:
        attentions: List of attention tensors from the model
                   Each tensor shape: (batch, num_heads, seq_len, seq_len)
        start_layer: Which layer to start from (default 0)
        end_layer: Which layer to end at (default None = all layers)

    Returns:
        Attention matrix: (batch, seq_len, seq_len)
    """
    if end_layer is None:
        end_layer = len(attentions)

    # Simply use the attention from the target layer (more stable!)
    # Average over attention heads
    target_attention = attentions[end_layer - 1]  # Last layer in range
    attention_heads_fused = target_attention.mean(dim=1)  # (batch, seq_len, seq_len)

    return attention_heads_fused

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
        prompt = "USER: <image>\nIs there an object in this image? Answer with Yes or No.\nASSISTANT:"

        # Process images one at a time (no batching)
        for frame_path in sampled_frames:
            image = Image.open(frame_path).convert("RGB")

            # Generate answer with attention tracking
            # Need to call underlying model directly to get attentions
            print("Generating answer...")

            # Prepare inputs
            inputs = model.processor(text=prompt, images=image, return_tensors="pt")
            inputs.to(model.model.device)

            # Call generate on the underlying model directly
            with torch.no_grad():
                generation_outputs = model.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    do_sample=False
                )

            # Decode the response
            response_str = model.processor.batch_decode(
                generation_outputs.sequences,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            print(f"Model response: {response_str}")

            # Extract attentions from generation
            # generation_outputs.attentions is a tuple of tuples:
            # - Outer tuple: generation steps (one per generated token)
            # - Inner tuple: layers (32 layers for LLaVA)
            gen_attentions = generation_outputs.attentions

            # We want attention from the FIRST generated token (Yes/No)
            # That's generation step 0
            if len(gen_attentions) == 0:
                print("Warning: No attentions in generation, skipping frame")
                continue

            first_token_attentions = gen_attentions[0]  # Tuple of 32 layer attentions

            # Also get hidden states from the last generation step at our target layer
            # hidden_states is tuple of tuples: (generation_step, layer)
            gen_hidden_states = generation_outputs.hidden_states

            # Get hidden states from first generation step, at our target layer
            if len(gen_hidden_states) == 0:
                print("Warning: No hidden states in generation, skipping frame")
                continue

            first_token_hidden_states = gen_hidden_states[0]  # First generation step
            layer_hidden = first_token_hidden_states[layer]  # Our target layer (e.g., 25)

            print(f"Layer {layer} hidden states shape: {layer_hidden.shape}")

            ## Compute attention rollout up to layer 25 for the first generated token
            print(f"\n=== Computing attention rollout up to layer {layer} for generated token ===")
            rollout = compute_attention_rollout(first_token_attentions, start_layer=0, end_layer=layer)
            # rollout shape: (1, seq_len, seq_len)

            # Get attention from the GENERATED TOKEN position to IMAGE tokens
            # Image tokens: positions 0-575
            # Generated token position: last position in the sequence
            num_image_tokens = 576
            seq_len = rollout.shape[-1]
            generated_token_position = seq_len - 1  # Last position (the Yes/No token)

            print(f"Looking at attention from generated token position {generated_token_position} to image tokens")

            # Get attention from generated token to all image tokens
            # This shows: "Which image patches did the model attend to when generating Yes/No?"
            image_attention_scores = rollout[0, generated_token_position, :num_image_tokens]  # (576,)

            # Filter out register tokens (first 5 positions seem to be artifacts)
            # These consistently show high attention but predict nonsense
            num_register_tokens = 1
            print(f"Filtering out first {num_register_tokens} positions (likely register tokens)")

            # Zero out attention to register positions
            image_attention_scores[:num_register_tokens] = 1

            print(f"Image attention scores shape: {image_attention_scores.shape}")
            print(f"Attention score range (after filtering): [{image_attention_scores.min().item():.4f}, {image_attention_scores.max().item():.4f}]")
            print(f"Sum of attention (after filtering): {image_attention_scores.sum().item():.4f}")

            # Get top 30 image token positions based on attention scores (excluding position 0)
            # Create mask to exclude register tokens
            valid_positions = torch.arange(num_register_tokens, num_image_tokens, device=image_attention_scores.device)
            valid_attention_scores = image_attention_scores[num_register_tokens:]

            # Get top 30 positions by attention score
            top_30_values, top_30_relative_indices = torch.topk(valid_attention_scores, k=30, largest=True)
            top_30_indices = valid_positions[top_30_relative_indices]

            # Collect embeddings and apply logit lens to top tokens
            top_embeddings = []
            top_words = []

            print(f"\n=== Top 30 image tokens by attention (with logit lens) ===")
            for idx, (pos_idx, attn_score) in enumerate(zip(top_30_indices, top_30_values)):
                pos = pos_idx.item()

                # Get hidden state for this position
                token_hidden = layer_hidden[0, pos, :].unsqueeze(0)  # shape (1, hidden_dim)

                # Apply logit lens to get predicted word
                logits = lm_head(norm(token_hidden))  # shape (1, vocab_size)
                predicted_token_id = logits.argmax(dim=-1).item()
                predicted_word = tokenizer.decode([predicted_token_id])

                # Store embedding and word
                top_embeddings.append(layer_hidden[0, pos, :].cpu().numpy())
                top_words.append(predicted_word)

                print(f"{idx+1}. Position {pos} predicts '{predicted_word}' (attention: {attn_score.item():.4f})")

            # Average the top 30 token embeddings
            if top_embeddings:
                top_embeddings_stack = np.stack(top_embeddings)  # shape (30, hidden_dim)
                mean_embedding = np.mean(top_embeddings_stack, axis=0)  # shape (hidden_dim,)
                language_latent.append(mean_embedding)

        if language_latent:
            # Stack all frame embeddings into a matrix (num_frames, hidden_dim)
            embeddings_matrix = np.stack(language_latent, axis=0)

            # Compute average across all frames
            avg = np.mean(embeddings_matrix, axis=0)

            print(f"[DEBUG] Saving latent for group {group_label}")
            print(f"[DEBUG] Matrix shape: {embeddings_matrix.shape}, mean: {avg.mean():.4f}, std: {avg.std():.4f}")

            # Save the full matrix of embeddings (one row per frame)
           

            # Save the averaged vector
            np.save(f"/home/new_storage/sherlock/STS_sherlock/projects data/CLS_object_layer16_top30/{group_label}_latent.npy", avg)
        
        i += tr_ref #  no overlap between groups



if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze frames from video sequences using LLaVA model')
    parser.add_argument('--TR_root', type=str,  help='Root directory containing TR sequences')
    parser.add_argument('--output_path', type=str, help='Path to save results CSV')
    parser.add_argument('--start_seq', type=int, default=0, help='Starting sequence number')
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
            tr_ref=1,layer=16, 
            seq_range=(args.start_seq, args.end_seq),
            output_path=output,
            samples_per_seq=args.samples_per_seq,
            save_interval=args.save_interval,
        )