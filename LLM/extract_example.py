import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
sys.path.append(src_path)
sys.path.append('/home/new_storage/sherlock/llava-interp')

from src.HookedLVLM import HookedLVLM


def analyze_layer_with_keyword(image_path, prompt, layer_idx, keyword, device="cuda:0", quantize_type="fp16"):
    """
    Analyzes a specific layer's output and finds tokens most similar to a keyword.

    Args:
        image_path: Path to the image file
        prompt: Text prompt for the model
        layer_idx: Which layer to analyze (0-indexed)
        keyword: Keyword to compare against vocabulary
        device: Device to run on
        quantize_type: Quantization type for the model
    """
    # Load model
    print(f"Loading model on {device}...")
    model = HookedLVLM(device=device, quantize=True, quantize_type=quantize_type)

    # Get model components
    norm = model.model.language_model.model.norm
    lm_head = model.model.language_model.lm_head
    tokenizer = model.processor.tokenizer
    embedding_layer = model.model.language_model.model.embed_tokens

    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)

    # Forward pass to get hidden states
    print(f"Running forward pass...")
    outputs = model.forward(image, prompt, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    num_layers = len(hidden_states)
    print(f"Model has {num_layers} layers")

    if layer_idx >= num_layers or layer_idx < 0:
        raise ValueError(f"Layer index {layer_idx} out of range. Model has {num_layers} layers (0-{num_layers-1})")

    # Extract hidden states at specified layer
    # Shape: (batch_size, sequence_length, hidden_dim)
    layer_hidden = hidden_states[layer_idx]
    print(f"Layer {layer_idx} hidden states shape: {layer_hidden.shape}")

    sequence_length = layer_hidden.shape[1]
    print(f"Sequence has {sequence_length} token positions")

    # Figure out how many image tokens there are
    # For LLaVA with 336x336 images and 14x14 patches: (336/14)^2 = 576 tokens
    image_size = 336
    patch_size = 14
    num_image_tokens = (image_size // patch_size) ** 2
    print(f"Number of image tokens: {num_image_tokens}")
    print(f"Image tokens are at positions 0-{num_image_tokens-1}")
    print(f"Text tokens start at position {num_image_tokens}")

    # Encode the keyword
    print(f"\n=== Analyzing similarity to keyword: '{keyword}' ===")
    keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
    print(f"Keyword tokenized as: {keyword_tokens} -> {[tokenizer.decode([t]) for t in keyword_tokens]}")

    # Get keyword embedding (average if multiple tokens)
    keyword_token_ids = torch.tensor(keyword_tokens).to(device)
    keyword_embeddings = embedding_layer(keyword_token_ids)
    keyword_embedding = keyword_embeddings.mean(dim=0)  # Average if multiple tokens
    print(f"Keyword embedding shape: {keyword_embedding.shape}")

    # Normalize keyword embedding for cosine similarity
    keyword_embedding_norm = F.normalize(keyword_embedding.unsqueeze(0), dim=-1)

    # For each IMAGE token position, compute what it predicts and its similarity to keyword
    print(f"\n=== Computing per-token predictions and similarities (IMAGE TOKENS ONLY) ===")

    token_predictions = []
    token_similarities = []

    for pos in range(num_image_tokens):
        # Get hidden state for this token position
        token_hidden = layer_hidden[0, pos, :].unsqueeze(0)  # Shape: (1, hidden_dim)

        # Apply logit lens to get predicted token
        normalized = norm(token_hidden)
        logits = lm_head(normalized)
        predicted_token_id = logits.argmax(dim=-1).item()
        predicted_token = tokenizer.decode([predicted_token_id])

        # Get the embedding of the predicted token
        predicted_token_tensor = torch.tensor([predicted_token_id]).to(device)
        predicted_token_embedding = embedding_layer(predicted_token_tensor)

        # Normalize predicted token embedding for cosine similarity
        predicted_token_embedding_norm = F.normalize(predicted_token_embedding, dim=-1)

        # Compute cosine similarity between predicted token and keyword
        similarity = torch.matmul(predicted_token_embedding_norm, keyword_embedding_norm.T).squeeze().item()

        token_predictions.append(predicted_token)
        token_similarities.append(similarity)

    # Get top 30 IMAGE token positions by similarity to keyword
    similarities_tensor = torch.tensor(token_similarities)
    top_30_values, top_30_indices = torch.topk(similarities_tensor, k=min(30, num_image_tokens))

    print(f"\n=== Top 30 IMAGE token positions by cosine similarity to '{keyword}' ===")
    for i, (pos_idx, similarity) in enumerate(zip(top_30_indices, top_30_values)):
        pos = pos_idx.item()
        predicted_token = token_predictions[pos]
        print(f"{i+1}. Position {pos} (predicts '{predicted_token}'): similarity {similarity.item():.4f}")

    print("\n=== Analysis complete ===")


def main():
    image ="/home/new_storage/sherlock/data/frames/TR0402/frame_015102.jpg"

    analyze_layer_with_keyword(
        image_path=image,
        prompt="USER: <image>\nIs there social interaction in this image? ASSISTANT:",
        layer_idx=25,
        keyword="face",
        device="cuda:0",
        quantize_type="4bit"
    )


if __name__ == "__main__":
    main()
