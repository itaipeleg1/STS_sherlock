"""
Token Probability Experiment for LLaVA
Explores how token meaning changes with the same image but different prompts.

For each layer: ΔP(w) = Σ_{vlz∈I} Σ_{ω∈heads} P(w|Prompt1) - P(w|Prompt2)
"""
import os
import sys
## Set environment variables for Hugging Face cache
## This is to avoid running out of space in the default cache location
os.environ['HF_HOME'] = '/home/new_storage/sherlock/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/new_storage/sherlock/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/new_storage/sherlock/hf_cache'
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
import torch.nn.functional as F
from tqdm import tqdm


class TokenProbabilityAnalyzer:
    """Analyzes token probability changes across prompts in LLaVA model."""

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        """
        Initialize the analyzer with LLaVA model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device
        print(f"Loading model: {model_name}")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Get the language model and its layers
        self.language_model = self.model.language_model
        self.lm_head = self.language_model.lm_head
        self.num_layers = len(self.language_model.model.layers)

        print(f"Model loaded. Number of layers: {self.num_layers}")
    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the token embedding matrix from LLaVA's language model."""
        return self.language_model.model.embed_tokens.weight

    def find_similar_words(self, keywords: List[str], top_k: int = 50) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find top-k similar words to each keyword using cosine similarity in LLaVA's embedding space.

        Args:
            keywords: List of seed keywords
            top_k: Number of similar words to retrieve per keyword

        Returns:
            Dictionary mapping each keyword to list of (word, similarity_score) tuples
        """
        print(f"Finding similar words for {len(keywords)} keywords...")
        embedding_matrix = self.get_embedding_matrix()  # [vocab_size, embedding_dim]

        similar_words = {}

        for keyword in keywords:
            # Tokenize keyword (handle multi-token keywords by taking first token)
            token_ids = self.processor.tokenizer(keyword, add_special_tokens=False)["input_ids"]
            if len(token_ids) == 0:
                print(f"Warning: Could not tokenize '{keyword}', skipping.")
                continue

            keyword_id = token_ids[0]
            keyword_embedding = embedding_matrix[keyword_id].unsqueeze(0)  # [1, embedding_dim]

            # Compute cosine similarity with all embeddings
            similarities = F.cosine_similarity(
                keyword_embedding,
                embedding_matrix,
                dim=1
            )  # [vocab_size]

            # Get top-k similar tokens
            top_similarities, top_indices = torch.topk(similarities, k=top_k + 1)  # +1 to exclude the keyword itself

            # Convert to words
            similar_word_list = []
            for idx, sim in zip(top_indices.detach().cpu().numpy(), top_similarities.detach().cpu().numpy()):
                word = self.processor.tokenizer.decode([idx])
                if word != keyword:  # Exclude the keyword itself
                    similar_word_list.append((word, float(sim)))
                if len(similar_word_list) >= top_k:
                    break

            similar_words[keyword] = similar_word_list
            print(f"  {keyword}: Found {len(similar_word_list)} similar words")

        return similar_words

    def get_layer_probabilities(
        self,
        image_path: str,
        prompt: str,
        target_tokens: List[str]
    ) -> Dict[int, torch.Tensor]:
        """
        Extract probability distributions at each layer for target tokens.

        Args:
            image_path: Path to input image
            prompt: Text prompt to use
            target_tokens: List of token strings to extract probabilities for

        Returns:
            Dictionary mapping layer_idx to probability tensor [num_target_tokens]
        """
        # Prepare inputs
        image = Image.open(image_path).convert("RGB")

        # Format prompt for LLaVA
        if prompt:
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        else:
            full_prompt = "USER: <image>\nASSISTANT:"

        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # Get token IDs for target tokens
        target_token_ids = []
        for token in target_tokens:
            token_ids = self.processor.tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(token_ids) > 0:
                target_token_ids.append(token_ids[0])
            else:
                print(f"Warning: Could not tokenize '{token}', using padding token.")
                target_token_ids.append(0)

        target_token_ids = torch.tensor(target_token_ids).to(self.device)

        layer_probs = {}

        # Hook to capture hidden states at each layer
        hidden_states_dict = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output[0] is the hidden state
                hidden_states_dict[layer_idx] = output[0].detach()
            return hook

        # Register hooks for all layers
        hooks = []
        for layer_idx, layer in enumerate(self.language_model.model.layers):
            hook = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process hidden states at each layer
        for layer_idx in range(self.num_layers):
            hidden_state = hidden_states_dict[layer_idx]  # [batch_size, seq_len, hidden_dim]

            # Extract only visual tokens (first 576 tokens in LLaVA)
            num_visual_tokens = 576
            visual_hidden_states = hidden_state[:, :num_visual_tokens, :]  # [batch_size, 576, hidden_dim]

            # Apply LM head to EACH visual token separately
            # This gives P(w) for each of the 576 visual tokens
            logits_all = self.lm_head(visual_hidden_states)  # [batch_size, 576, vocab_size]

            # Convert to probabilities for each visual token
            probs_all = F.softmax(logits_all, dim=-1)  # [batch_size, 576, vocab_size]

            # Sum probabilities over all visual tokens
            # This implements Σ_{vlz∈I} P(w) from your formula
            probs_summed = probs_all.sum(dim=1)  # [batch_size, vocab_size]

            # Extract probabilities for target tokens
            target_probs = probs_summed[0, target_token_ids]  # [num_target_tokens]

            layer_probs[layer_idx] = target_probs.cpu()

        return layer_probs

    def compute_probability_delta(
        self,
        image_path: str,
        prompt1: str,
        prompt2: str,
        keywords: List[str],
        top_k: int = 50
    ) -> Tuple[Dict[int, np.ndarray], List[str]]:
        """
        Compute ΔP for each layer comparing two prompts.

        Args:
            image_path: Path to input image
            prompt1: First prompt
            prompt2: Second prompt
            keywords: List of seed keywords
            top_k: Number of similar words per keyword

        Returns:
            Tuple of (layer_deltas, token_labels) where:
                - layer_deltas: Dict mapping layer_idx to ΔP array
                - token_labels: List of token strings in same order as ΔP arrays
        """
        # Build vocabulary (keywords + similar words, or entire vocabulary if keywords is empty)
        if keywords is None or len(keywords) == 0:
            # Use entire vocabulary
            print("\nNo keywords provided - using entire vocabulary")
            vocab_size = self.processor.tokenizer.vocab_size
            token_list = []
            for token_id in range(vocab_size):
                token = self.processor.tokenizer.decode([token_id])
                # Filter out special tokens and empty strings
                if token and len(token.strip()) > 0:
                    token_list.append(token)
            print(f"Total vocabulary size: {len(token_list)} tokens (entire vocabulary)")
        else:
            # Use keywords + similar words
            similar_words_dict = self.find_similar_words(keywords, top_k)

            # Collect all unique tokens
            all_tokens = set(keywords)
            for similar_list in similar_words_dict.values():
                for word, _ in similar_list:
                    all_tokens.add(word)

            token_list = sorted(list(all_tokens))
            print(f"\nTotal unique tokens to analyze: {len(token_list)} (from {len(keywords)} keywords)")

        # Get probabilities for both prompts
        print(f"\nProcessing prompt 1: '{prompt1}'")
        probs1 = self.get_layer_probabilities(image_path, prompt1, token_list)

        print(f"\nProcessing prompt 2: '{prompt2}'")
        probs2 = self.get_layer_probabilities(image_path, prompt2, token_list)

        # Compute delta for each layer
        layer_deltas = {}
        for layer_idx in range(self.num_layers):
            delta = (probs1[layer_idx] - probs2[layer_idx]).detach().numpy()
            layer_deltas[layer_idx] = delta

        return layer_deltas, token_list

    def visualize_and_save(
        self,
        layer_deltas: Dict[int, np.ndarray],
        token_labels: List[str],
        output_dir: str,
        top_n_display: int = 30
    ):
        """
        Create and save bar plots for each layer showing ΔP.

        Args:
            layer_deltas: Dictionary mapping layer_idx to ΔP arrays
            token_labels: List of token labels
            output_dir: Directory to save plots
            top_n_display: Number of top tokens to display per plot
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating plots for {len(layer_deltas)} layers...")

        for layer_idx in tqdm(range(len(layer_deltas))):
            delta = layer_deltas[layer_idx]

            # Sort all tokens by absolute delta (descending)
            sorted_indices = np.argsort(np.abs(delta))[::-1]

            # Filter out single characters and collect top_n_display tokens
            filtered_indices = []
            filtered_deltas = []
            filtered_labels = []

            for idx in sorted_indices:
                label = token_labels[idx]
                # Skip single characters (numbers or letters)
                if len(label.strip()) <= 1:
                    continue

                filtered_indices.append(idx)
                filtered_deltas.append(delta[idx])
                filtered_labels.append(label)

                # Stop once we have enough tokens
                if len(filtered_indices) >= top_n_display:
                    break

            top_deltas = np.array(filtered_deltas)
            top_labels = filtered_labels

            # Debug: print actual number of tokens displayed
            if layer_idx == 0:  # Only print for first layer to avoid spam
                print(f"  Layer 0: Displaying {len(top_labels)} tokens (requested {top_n_display})")
                if len(top_labels) < top_n_display:
                    print(f"  Note: Only {len(token_labels)} total tokens available after filtering")

            # Dynamic figure height based on number of tokens (at least 0.15 inches per token)
            fig_height = max(8, len(top_labels) * 0.15)
            fig, ax = plt.subplots(figsize=(14, fig_height))

            colors = ['red' if d < 0 else 'green' for d in top_deltas]
            bars = ax.barh(range(len(top_deltas)), top_deltas, color=colors, alpha=0.7)

            # Adjust font size based on number of tokens
            label_fontsize = max(4, min(8, 800 // len(top_labels)))
            ax.set_yticks(range(len(top_deltas)))
            ax.set_yticklabels(top_labels, fontsize=label_fontsize)
            ax.set_xlabel('ΔP (Prompt1 - Prompt2)', fontsize=12)
            ax.set_title(f'Layer {layer_idx}: Token Probability Differences', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()

            # Save figure
            save_path = output_path / f"layer_{layer_idx:02d}_delta_probs.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        print(f"✓ Plots saved to {output_dir}")

    def compute_delta_statistics(
        self,
        layer_deltas: Dict[int, np.ndarray],
        token_labels: List[str],
        output_dir: str
    ):
        """
        Compute and save statistics about probability deltas across layers.

        Args:
            layer_deltas: Dictionary mapping layer_idx to ΔP arrays
            token_labels: List of token labels
            output_dir: Directory to save statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("DELTA STATISTICS")
        print("=" * 80)

        # Compute statistics for each layer
        stats_per_layer = []

        for layer_idx in range(len(layer_deltas)):
            delta = layer_deltas[layer_idx]

            stats = {
                'layer': layer_idx,
                'mean_delta': float(np.mean(delta)),
                'std_delta': float(np.std(delta)),
                'mean_abs_delta': float(np.mean(np.abs(delta))),
                'max_positive_delta': float(np.max(delta)),
                'max_negative_delta': float(np.min(delta)),
                'total_variation': float(np.sum(np.abs(delta))),
                'num_positive': int(np.sum(delta > 0)),
                'num_negative': int(np.sum(delta < 0)),
                'num_tokens': len(delta)
            }

            stats_per_layer.append(stats)

        # Print summary statistics
        print(f"\nOverall Statistics Across All Layers:")
        print(f"  Mean Absolute Delta: {np.mean([s['mean_abs_delta'] for s in stats_per_layer]):.8f}")
        print(f"  Std of Mean Absolute Delta: {np.std([s['mean_abs_delta'] for s in stats_per_layer]):.8f}")
        print(f"  Max Total Variation: {np.max([s['total_variation'] for s in stats_per_layer]):.8f} (Layer {np.argmax([s['total_variation'] for s in stats_per_layer])})")
        print(f"  Min Total Variation: {np.min([s['total_variation'] for s in stats_per_layer]):.8f} (Layer {np.argmin([s['total_variation'] for s in stats_per_layer])})")

        # Save to CSV
        import csv
        csv_path = output_path / "delta_statistics.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats_per_layer[0].keys())
            writer.writeheader()
            writer.writerows(stats_per_layer)

        print(f"\n✓ Statistics saved to {csv_path}")

        # Create a plot showing statistics across layers
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        layers = [s['layer'] for s in stats_per_layer]

        # Plot 1: Mean Absolute Delta
        axes[0, 0].plot(layers, [s['mean_abs_delta'] for s in stats_per_layer], marker='o', linewidth=2, color='blue')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Absolute ΔP')
        axes[0, 0].set_title('Mean Absolute Delta per Layer')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Total Variation
        axes[0, 1].plot(layers, [s['total_variation'] for s in stats_per_layer], marker='o', linewidth=2, color='green')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Total Variation')
        axes[0, 1].set_title('Total Variation (Sum of |ΔP|) per Layer')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Max Positive vs Negative Delta
        axes[1, 0].plot(layers, [s['max_positive_delta'] for s in stats_per_layer], marker='o', linewidth=2, color='green', label='Max Positive')
        axes[1, 0].plot(layers, [s['max_negative_delta'] for s in stats_per_layer], marker='o', linewidth=2, color='red', label='Max Negative')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('ΔP')
        axes[1, 0].set_title('Max Positive/Negative Delta per Layer')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)

        # Plot 4: Standard Deviation
        axes[1, 1].plot(layers, [s['std_delta'] for s in stats_per_layer], marker='o', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Std(ΔP)')
        axes[1, 1].set_title('Standard Deviation of Delta per Layer')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_path / "delta_statistics_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Statistics plots saved to {plot_path}")


def extract_frame_number(filepath: str) -> int:
    """Extract frame number from filepath."""
    try:
        filename = os.path.basename(filepath)
        number_part = filename.split("_")[-1].split(".")[0]
        clean_number = ''.join(c for c in number_part if c.isdigit())
        return int(clean_number)
    except (ValueError, IndexError):
        return -1


def run_multi_image_experiment(
    tr_root: str,
    prompt1: str,
    prompt2: str,
    keywords: List[str],
    output_dir: str = "./experiment_results_multi",
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    top_k: int = 50,
    top_n_display: int = 100,
    samples_per_seq: int = 8,
    tr_ref: int = 1,
    seq_range: tuple = None,
    tr_list: List[int] = None,
    seq_prefix: str = "TR",
    file_extension: str = ".jpg"
):
    """
    Run experiment over multiple images, summing deltas across images for robustness.

    Args:
        tr_root: Root directory containing TR sequence folders
        prompt1: First prompt
        prompt2: Second prompt
        keywords: List of seed keywords for vocabulary selection
        output_dir: Directory to save results
        model_name: HuggingFace model identifier
        top_k: Number of similar words per keyword
        top_n_display: Number of tokens to show in each plot
        samples_per_seq: Number of frames to sample per TR sequence
        tr_ref: Number of consecutive TRs to group together
        seq_range: Tuple (start, end) for TR range, e.g., (0, 100). Ignored if tr_list is provided.
        tr_list: List of specific TR indices to process, e.g., [5, 10, 15, 402]. Takes priority over seq_range.
        seq_prefix: Prefix for sequence directories (default "TR")
        file_extension: Image file extension (default ".jpg")
    """
    print("=" * 80)
    print("MULTI-IMAGE TOKEN PROBABILITY EXPERIMENT")
    print("=" * 80)

    # Initialize analyzer
    analyzer = TokenProbabilityAnalyzer(model_name=model_name)

    # Get vocabulary (keywords + similar words, or entire vocabulary if keywords is empty)
    print("\nBuilding vocabulary...")

    if keywords is None or len(keywords) == 0:
        # Use entire vocabulary
        print("No keywords provided - using entire vocabulary")
        vocab_size = analyzer.processor.tokenizer.vocab_size
        token_list = []
        for token_id in range(vocab_size):
            token = analyzer.processor.tokenizer.decode([token_id])
            # Filter out special tokens and empty strings
            if token and len(token.strip()) > 0:
                token_list.append(token)
        print(f"Total vocabulary size: {len(token_list)} tokens (entire vocabulary)")
    else:
        # Use keywords + similar words
        similar_words_dict = analyzer.find_similar_words(keywords, top_k)

        all_tokens = set(keywords)
        for similar_list in similar_words_dict.values():
            for word, _ in similar_list:
                all_tokens.add(word)

        token_list = sorted(list(all_tokens))
        print(f"Total vocabulary size: {len(token_list)} tokens (from {len(keywords)} keywords)")

    # Get all TR directories
    seq_dirs = [d for d in os.listdir(tr_root)
               if os.path.isdir(os.path.join(tr_root, d))
               and d.startswith(seq_prefix)]
    seq_dirs.sort(key=lambda x: int(x[len(seq_prefix):]))

    # Apply TR filtering: tr_list takes priority over seq_range
    if tr_list is not None:
        # Filter to only specific TR indices
        seq_dirs = [d for d in seq_dirs
                   if int(d[len(seq_prefix):]) in tr_list]
        print(f"Filtering to specific TRs: {tr_list}")
    elif seq_range:
        # Apply TR range
        start, end = seq_range
        seq_dirs = [d for d in seq_dirs
                   if start <= int(d[len(seq_prefix):]) <= end]
        print(f"Filtering to TR range: {seq_range}")

    print(f"Found {len(seq_dirs)} TR directories to process")

    # Collect frame paths
    all_frame_paths = []
    samples_per_group = samples_per_seq * tr_ref

    i = 0
    while i <= len(seq_dirs) - tr_ref:
        group_dirs = seq_dirs[i:i+tr_ref]

        # Get and sort frame paths for this group
        frame_paths = []
        for seq_dir in group_dirs:
            seq_path = os.path.join(tr_root, seq_dir)
            frames = [os.path.join(seq_path, f) for f in os.listdir(seq_path)
                    if f.endswith(file_extension)]
            frames.sort(key=extract_frame_number)
            frame_paths.extend(frames)

        # Sample frames evenly
        total_frames = len(frame_paths)
        if total_frames > 0:
            indices = [
                idx * (total_frames - 1) // (samples_per_group - 1)
                for idx in range(samples_per_group)
            ]
            sampled_frames = [frame_paths[idx] for idx in indices]
            all_frame_paths.extend(sampled_frames)

        i += tr_ref

    print(f"Total frames to process: {len(all_frame_paths)}")

    # Initialize accumulators for deltas across all images
    # layer_deltas_sum[layer_idx] will accumulate sum of deltas across images
    layer_deltas_sum = {}
    for layer_idx in range(analyzer.num_layers):
        layer_deltas_sum[layer_idx] = np.zeros(len(token_list))

    # Process each image
    for img_idx, frame_path in enumerate(tqdm(all_frame_paths, desc="Processing images")):
        print(f"\nProcessing image {img_idx + 1}/{len(all_frame_paths)}: {os.path.basename(frame_path)}")

        # Get probabilities for both prompts
        probs1 = analyzer.get_layer_probabilities(frame_path, prompt1, token_list)
        probs2 = analyzer.get_layer_probabilities(frame_path, prompt2, token_list)

        # Accumulate deltas for each layer
        for layer_idx in range(analyzer.num_layers):
            delta = (probs1[layer_idx] - probs2[layer_idx]).detach().numpy()
            layer_deltas_sum[layer_idx] += delta

    print(f"\n✓ Processed {len(all_frame_paths)} images")
    print(f"Computing average deltas...")

    # Average the deltas (or keep as sum, depending on preference)
    # For now, let's keep it as sum to show total effect
    layer_deltas_aggregated = layer_deltas_sum

    # Visualize aggregated results
    analyzer.visualize_and_save(
        layer_deltas=layer_deltas_aggregated,
        token_labels=token_list,
        output_dir=output_dir,
        top_n_display=top_n_display
    )

    # Compute statistics
    analyzer.compute_delta_statistics(
        layer_deltas=layer_deltas_aggregated,
        token_labels=token_list,
        output_dir=output_dir
    )

    print("\n" + "=" * 80)
    print("MULTI-IMAGE EXPERIMENT COMPLETE")
    print("=" * 80)



def run_experiment(
    image_path: str,
    prompt1: str,
    prompt2: str,
    keywords: List[str],
    output_dir: str = "./experiment_results",
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    top_k: int = 50,
    top_n_display: int = 30
):
    """
    Run the full experiment.

    Args:
        image_path: Path to input image
        prompt1: First prompt
        prompt2: Second prompt
        keywords: List of seed keywords for vocabulary selection
        output_dir: Directory to save results
        model_name: HuggingFace model identifier
        top_k: Number of similar words per keyword
        top_n_display: Number of tokens to show in each plot
    """
    print("=" * 80)
    print("TOKEN PROBABILITY EXPERIMENT")
    print("=" * 80)

    # Initialize analyzer
    analyzer = TokenProbabilityAnalyzer(model_name=model_name)

    # Compute probability deltas
    layer_deltas, token_labels = analyzer.compute_probability_delta(
        image_path=image_path,
        prompt1=prompt1,
        prompt2=prompt2,
        keywords=keywords,
        top_k=top_k
    )

    # Visualize and save
    analyzer.visualize_and_save(
        layer_deltas=layer_deltas,
        token_labels=token_labels,
        output_dir=output_dir,
        top_n_display=top_n_display
    )

    # Compute and save statistics
    analyzer.compute_delta_statistics(
        layer_deltas=layer_deltas,
        token_labels=token_labels,
        output_dir=output_dir
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # ========================================================================
    # CHOOSE EXPERIMENT TYPE: Single-image or Multi-image
    # ========================================================================

    EXPERIMENT_TYPE = "multi"  # "single" or "multi"

    # ========================================================================
    # SHARED PARAMETERS
    # ========================================================================

    # Prompts to compare
    PROMPT_1 = ""
    PROMPT_2 = "Is there social interaction in this image?"

    # Keywords for vocabulary selection (ADJUST THIS ARRAY)
    # Set to [] or None to use entire vocabulary (~32,000 tokens)
    KEYWORDS = ["social","face","people","group","interaction","conversation",
                "gathering","smile","looking","talking","gaze","eye","mouth","object",
                "hand","body","scene","background","activity","event","meeting",]

    # ========================================================================
    # EXPERIMENT-SPECIFIC PARAMETERS
    # ========================================================================

    if EXPERIMENT_TYPE == "single":
        # Single-image experiment
        IMAGE_PATH = "/home/new_storage/sherlock/data/frames/TR0402/frame_015111.jpg"
        OUTPUT_DIR = "/home/new_storage/sherlock/STS_sherlock/projects data/experiment_results_single"

        run_experiment(
            image_path=IMAGE_PATH,
            prompt1=PROMPT_1,
            prompt2=PROMPT_2,
            keywords=KEYWORDS,
            output_dir=OUTPUT_DIR,
            top_k=500,
            top_n_display=100
        )

    elif EXPERIMENT_TYPE == "multi":
        # Multi-image experiment (more robust)
        TR_ROOT = "/home/new_storage/sherlock/data/frames"
        OUTPUT_DIR = "/home/new_storage/sherlock/STS_sherlock/projects data/experiment_results_multi"

        # Choose ONE of the following filtering options:
        # Option 1: Specify a list of specific TRs
        ## load annotations to get specific TRs
        annotation_path = "/home/new_storage/sherlock/STS_sherlock/projects data/annotations/social_nonsocial.npy"
        annotation = np.load(annotation_path)
        annotation = annotation[26:]
        annotation = annotation[:919]
        TR_LIST = np.where(annotation==1)[0].flatten()

        # Option 2: Specify a range (start, end)
        # TR_RANGE = (0, 919)  # Process TR0000 to TR0919

        run_multi_image_experiment(
            tr_root=TR_ROOT,
            prompt1=PROMPT_1,
            prompt2=PROMPT_2,
            keywords=KEYWORDS,
            output_dir=OUTPUT_DIR,
            top_k=300,  # Top 300 similar words per keyword
            top_n_display=150,  # Show top 100 tokens in each plot
            samples_per_seq=8,  # Sample 8 frames per TR
            tr_ref=1,  # Process 1 TR at a time
            tr_list=TR_LIST,  # Use specific TRs (uncomment to use)
            #seq_range=(0, 919),  # Use range (comment out if using tr_list)
            seq_prefix="TR",
            file_extension=".jpg"
        )

    else:
        raise ValueError(f"Unknown experiment type: {EXPERIMENT_TYPE}. Must be 'single' or 'multi'")
