"""
Interactive t-SNE Viewer with Clickable Frame Images
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import base64
from PIL import Image
import io
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = 'clap_full'
OUTPUT_DIR = f'/home/new_storage/sherlock/STS_sherlock/pca_results/clap/{MODEL_NAME}_interactive_tsne'

# Weights paths
WEIGHTS_PPA = '/home/new_storage/sherlock/STS_sherlock/projects data/results/clap_full_ppa/clap_full_all_subjects_weights.npy'
WEIGHTS_STS = '/home/new_storage/sherlock/STS_sherlock/projects data/results/clap_full_sts/clap_full_all_subjects_weights.npy'
WEIGHTS_FFA = '/home/new_storage/sherlock/STS_sherlock/projects data/results/clap_full_ffa/clap_full_all_subjects_weights.npy'

# Embeddings directory
EMBEDDINGS_DIR = "/home/new_storage/sherlock/project_data/audio_embeddings"

# Annotation paths
ANNOTATION_PATHS = {
    'social': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/social_nonsocial_truncated.npy',
    'face': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/face_truncated.npy',
    'speaking': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/speaking_truncated.npy',
    'music': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/music_truncated.npy',
    'text': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/text_truncated.npy',
    'indoor': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/indoor_truncated.npy',
    'arousal': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/arousal_truncated.npy',
    'valence': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/valence_truncated.npy',
    'mentalization': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/mentalization_new.npy',
    'voice affecto dominance': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/dominance_scores.npy',
}

# Frames directory
FRAMES_DIR = "/home/new_storage/sherlock/data/frames"

# Parameters
N_PCS_FOR_TSNE = 10  # Number of PCs to use as input to t-SNE
N_IMAGES_TO_SHOW = 6  # Number of images to show per frame
IMAGE_MAX_WIDTH = 400  # Max width for images in pixels
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
RANDOM_SEED = 42


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_frame_html_file(frame_idx, output_dir, n_images=6, max_width=400):
    """
    Create an HTML file for a given frame showing randomly selected images.
    """
    tr_folder = f"TR{frame_idx:04d}"
    folder_path = os.path.join(FRAMES_DIR, tr_folder)

    html_filename = f"frame_{frame_idx:04d}.html"
    html_path = os.path.join(output_dir, html_filename)

    if not os.path.exists(folder_path):
        with open(html_path, 'w') as f:
            f.write(f"<html><body><h2>Frame {frame_idx}: Images not found</h2></body></html>")
        return html_filename

    # Get all available images in this TR folder
    available_images = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    if len(available_images) == 0:
        with open(html_path, 'w') as f:
            f.write(f"<html><body><h2>Frame {frame_idx}: No images found</h2></body></html>")
        return html_filename

    # Randomly select n_images from available
    n_to_select = min(n_images, len(available_images))
    selected_images = random.sample(available_images, n_to_select)

    # Build HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Frame {frame_idx}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .image-container {{
                background: white;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
            }}
            .image-label {{
                margin-top: 5px;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <h1>Frame {frame_idx} ({tr_folder})</h1>
        <p>Showing {n_to_select} randomly selected images from {len(available_images)} available</p>
        <div class="image-grid">
    """

    for img_filename in selected_images:
        img_path = os.path.join(folder_path, img_filename)
        try:
            # Load and encode image
            img = Image.open(img_path)
            # Resize to max_width while maintaining aspect ratio
            img.thumbnail((max_width, max_width * 2), Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()

            html_content += f"""
            <div class="image-container">
                <img src="data:image/jpeg;base64,{img_str}" alt="{img_filename}">
                <div class="image-label">{img_filename}</div>
            </div>
            """
        except Exception as e:
            html_content += f"""
            <div class="image-container">
                <p>Error loading {img_filename}: {str(e)}</p>
            </div>
            """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)

    return html_filename


# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # ------------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------------
    print("="*80)
    print("Loading data...")
    print("="*80)

    # Load weights
    print("Loading weights...")
    weights_ppa = np.load(WEIGHTS_PPA)
    weights_sts = np.load(WEIGHTS_STS)
    weights_ffa = np.load(WEIGHTS_FFA)
    weights = np.hstack([weights_ppa, weights_sts, weights_ffa])
    print(f"Weights shape: {weights.shape}")

    # Load embeddings
    print("Loading embeddings...")
    files = sorted([f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.npy')])
    embeddings = np.array([np.load(os.path.join(EMBEDDINGS_DIR, f)) for f in files])
    print(f"Embeddings shape: {embeddings.shape}")

    # Load annotations
    print("Loading annotations...")
    annotations = {}
    for name, path in ANNOTATION_PATHS.items():
        ann = np.load(path)
        annotations[name] = np.asarray(ann, dtype=float).flatten()
        print(f"  {name}: {len(annotations[name])}")

    # Add derived annotation
    if 'speaking' in annotations and 'social' in annotations:
        min_len_derived = min(len(annotations['speaking']), len(annotations['social']))
        speaking_trimmed = annotations['speaking'][:min_len_derived]
        social_trimmed = annotations['social'][:min_len_derived]
        annotations['speaking_no_social'] = np.where(
            (speaking_trimmed==1) & (social_trimmed==0), 1.0, 0.0
        )
        print(f"  speaking_no_social: {len(annotations['speaking_no_social'])} (derived)")

    # ------------------------------------------------------------------------
    # 2. Run PCA
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("Running PCA...")
    print("="*80)

    W_centered = weights - weights.mean(axis=1, keepdims=True)
    pca = PCA()
    pca.fit_transform(W_centered.T)
    pcs = pca.components_

    # PC activations over time
    pc_activations = embeddings @ pcs.T
    print(f"PC activations shape: {pc_activations.shape}")

    # ------------------------------------------------------------------------
    # 3. Run t-SNE
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("Running t-SNE...")
    print("="*80)

    print(f"Using first {N_PCS_FOR_TSNE} PCs as input to t-SNE")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER)
    pc_tsne = tsne.fit_transform(pc_activations[:, :N_PCS_FOR_TSNE])
    print(f"t-SNE embedding shape: {pc_tsne.shape}")

    # ------------------------------------------------------------------------
    # 4. Create interactive plots
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("Creating interactive t-SNE plots...")
    print("="*80)

    annotation_names = list(annotations.keys())

    for ann_name in annotation_names:
        print(f"\n  Processing {ann_name}...")

        ann_data = annotations[ann_name]

        # Trim to minimum length
        min_len = min(len(pc_tsne), len(ann_data))

        # Create subdirectory for frame HTML files
        frames_subdir = os.path.join(OUTPUT_DIR, f'frames_{ann_name}')
        os.makedirs(frames_subdir, exist_ok=True)

        # Generate HTML files for each frame
        print(f"    Generating frame HTML files...")
        random.seed(RANDOM_SEED)
        frame_files = []
        for i in range(min_len):
            if i % 100 == 0:
                print(f"      Processing frame {i}/{min_len}...")
            frame_file = create_frame_html_file(i, frames_subdir,
                                               n_images=N_IMAGES_TO_SHOW,
                                               max_width=IMAGE_MAX_WIDTH)
            frame_files.append(frame_file)

        # Create the plotly figure
        print(f"    Creating interactive plot...")
        fig = go.Figure(data=go.Scatter(
            x=pc_tsne[:min_len, 0],
            y=pc_tsne[:min_len, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=ann_data[:min_len],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=ann_name),
                opacity=0.7
            ),
            text=[f'Frame: {i}<br>{ann_name}: {ann_data[i]:.2f}' for i in range(min_len)],
            hovertemplate='<b>%{text}</b><br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<br><i>Click to view images</i><extra></extra>',
            customdata=[[i] for i in range(min_len)]
        ))

        fig.update_layout(
            title=f'{MODEL_NAME} - PC activations colored by {ann_name} (Click points to view images)',
            xaxis_title='t-SNE dimension 1',
            yaxis_title='t-SNE dimension 2',
            width=1200,
            height=900,
            hovermode='closest'
        )

        html_path = os.path.join(OUTPUT_DIR, f'tsne_{ann_name}_interactive.html')

        # Save using plotly's built-in method first
        fig.write_html(html_path)

        # Now read it back and add click handler
        with open(html_path, 'r') as f:
            html_content = f.read()

        # Add click handler JavaScript before closing body tag
        click_script = f'''
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var plotDiv = document.querySelector('.plotly-graph-div');
            if (plotDiv) {{
                plotDiv.on('plotly_click', function(eventData) {{
                    var pointIndex = eventData.points[0].pointIndex;
                    var frameUrl = 'frames_{ann_name}/frame_' + String(pointIndex).padStart(4, '0') + '.html';
                    window.open(frameUrl, '_blank');
                }});
            }}
        }});
        </script>
        '''

        html_content = html_content.replace('</body>', click_script + '\n</body>')

        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"    Saved: {html_path}")
        print(f"    Saved {len(frame_files)} frame HTML files in: {frames_subdir}")

    # ------------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("INTERACTIVE T-SNE VIEWER COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}\n")
    print("Files created:")
    for ann_name in annotation_names:
        print(f"  - tsne_{ann_name}_interactive.html (click points to view images)")
        print(f"  - frames_{ann_name}/ (directory with frame HTML files)")
    print()
