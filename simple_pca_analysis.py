"""
Simple PCA Analysis Script
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import nibabel as nib
import pandas as pd




MODEL_NAME = 'clip_full'
OUTPUT_DIR = f'/home/new_storage/sherlock/STS_sherlock/pca_results/clip/{MODEL_NAME}_analysis'

# Weights paths
WEIGHTS_PPA = '/home/new_storage/sherlock/STS_sherlock/projects data/results/clip_full_ppa/clip_full_all_subjects_weights.npy'
WEIGHTS_STS = '/home/new_storage/sherlock/STS_sherlock/projects data/results/clip_full_sts/clip_full_all_subjects_weights.npy'
WEIGHTS_FFA = '/home/new_storage/sherlock/STS_sherlock/projects data/results/clip_full_ffa/clip_full_all_subjects_weights.npy'

# Embeddings directory
EMBEDDINGS_DIR = "/home/new_storage/sherlock/STS_sherlock/projects data/clip_embeddings"

# Annotation paths (UPDATE THESE!)
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
    'close up': '/home/new_storage/sherlock/STS_sherlock/projects data/annotations/close_up.npy',
}

# Results directories for brain maps
RESULTS_DIR_PPA = '/home/new_storage/sherlock/STS_sherlock/projects data/results/clip_full_ppa'
RESULTS_DIR_STS = '/home/new_storage/sherlock/STS_sherlock/projects data/results/clip_full_sts'
RESULTS_DIR_FFA = '/home/new_storage/sherlock/STS_sherlock/projects data/results/clip_full_ffa'

# Analysis parameters
N_PCS_TO_ANALYZE = 5
N_BRAIN_MAPS = 2
N_SUBJECTS = 17
N_VOXELS_PER_SUBJECT = 1000


# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}\n")

# ----------------------------------------------------------------------------
# 1. Load weights
# ----------------------------------------------------------------------------
print("Loading weights...")
weights_ppa = np.load(WEIGHTS_PPA)
weights_sts = np.load(WEIGHTS_STS)
weights_ffa = np.load(WEIGHTS_FFA)
weights = np.hstack([weights_ppa, weights_sts, weights_ffa])
print(f"Weights shape: {weights.shape}")

# ----------------------------------------------------------------------------
# 2. Load embeddings
# ----------------------------------------------------------------------------
print("\nLoading embeddings...")
files = sorted([f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.npy')])
embeddings = np.array([np.load(os.path.join(EMBEDDINGS_DIR, f)) for f in files])
print(f"Embeddings shape: {embeddings.shape}")

# ----------------------------------------------------------------------------
# 3. Load annotations and align
# ----------------------------------------------------------------------------
print("\nLoading annotations...")
annotations = {}
for name, path in ANNOTATION_PATHS.items():
    ann = np.load(path)
    # Ensure numeric dtype
    annotations[name] = np.asarray(ann, dtype=float)
    print(f"  {name}: {len(ann)}")

# Add derived annotation (face_no_social)
if 'face' in annotations and 'social' in annotations:
    # Trim face and social to same length first for derived annotation
    min_len_derived = min(len(annotations['face']), len(annotations['social']))
    face_trimmed = annotations['face'][:min_len_derived]
    social_trimmed = annotations['social'][:min_len_derived]
    annotations['face_no_social'] = np.where(
        (face_trimmed==1) & (social_trimmed==0), 1.0, 0.0
    )
    print(f"  face_no_social: {len(annotations['face_no_social'])} (derived)")

print(f"\nEmbeddings shape: {embeddings.shape}")
print("Note: Annotations will be trimmed individually during correlation calculation")

# ----------------------------------------------------------------------------
# 4. Run PCA
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("Running PCA...")
print("="*80)

# Center weights
W_centered = weights - weights.mean(axis=1, keepdims=True)

# PCA - voxels as observations
pca = PCA()
voxel_projections = pca.fit_transform(W_centered.T)
pcs = pca.components_
explained_var = pca.explained_variance_ratio_

print(f"PC1: {explained_var[0]*100:.1f}%")
print(f"PC2: {explained_var[1]*100:.1f}%")
print(f"PC3: {explained_var[2]*100:.1f}%")
print(f"Top 10 PCs: {explained_var[:10].sum()*100:.1f}%")

# PC activations over time
pc_activations = embeddings @ pcs.T
print(f"PC activations shape: {pc_activations.shape}")

# ----------------------------------------------------------------------------
# 5. Compute correlations
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("Computing correlations...")
print("="*80)

annotation_names = list(annotations.keys())
correlations = np.zeros((N_PCS_TO_ANALYZE, len(annotation_names)))

for i in range(N_PCS_TO_ANALYZE):
    for j, ann_name in enumerate(annotation_names):
        # Trim to minimum length for this specific annotation
        min_len = min(len(pc_activations), len(annotations[ann_name]))

        pc_data = np.asarray(pc_activations[:min_len, i], dtype=np.float64)
        ann_data = np.asarray(annotations[ann_name][:min_len], dtype=np.float64)
        r, p = pearsonr(pc_data, ann_data.flatten())
        correlations[i, j] = r

# Print correlation table
print(f"\n{'PC':<6}", end="")
for name in annotation_names:
    print(f"{name:<12}", end="")
print()

for i in range(N_PCS_TO_ANALYZE):
    print(f"PC{i+1:<4}", end="")
    for j in range(len(annotation_names)):
        print(f"{correlations[i,j]:>11.3f} ", end="")
    print()

# Save to CSV
df_corr = pd.DataFrame(
    correlations,
    columns=annotation_names,
    index=[f'PC{i+1}' for i in range(N_PCS_TO_ANALYZE)]
)
corr_csv_path = os.path.join(OUTPUT_DIR, 'pc_annotation_correlations.csv')
df_corr.to_csv(corr_csv_path)
print(f"\nSaved: {corr_csv_path}")

# ----------------------------------------------------------------------------
# 6. Plot correlation heatmap
# ----------------------------------------------------------------------------
print("\nCreating correlation heatmap...")
plt.figure(figsize=(12, 6))
im = plt.imshow(correlations, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, label='Correlation')
plt.xticks(range(len(annotation_names)), annotation_names, rotation=45, ha='right')
plt.yticks(range(N_PCS_TO_ANALYZE), [f'PC{i+1}' for i in range(N_PCS_TO_ANALYZE)])
plt.xlabel('Annotations', fontsize=12)
plt.ylabel('Principal Components', fontsize=12)
plt.title(f'{MODEL_NAME} - PC vs Annotation Correlations', fontsize=14)
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {heatmap_path}")

# ----------------------------------------------------------------------------
# 7. Compare ROI projections
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("ROI Comparison...")
print("="*80)

n_voxels_per_roi = N_VOXELS_PER_SUBJECT * N_SUBJECTS

proj_ppa = voxel_projections[:n_voxels_per_roi, :]
proj_sts = voxel_projections[n_voxels_per_roi:2*n_voxels_per_roi, :]
proj_ffa = voxel_projections[2*n_voxels_per_roi:3*n_voxels_per_roi, :]

print(f"\n{'PC':<6} {'scene-responsive voxels':>25} {'STS':>10} {'face sensitive voxels':>10}")
print("-" * 55)

roi_results = []
for i in range(20):
    ppa_mean = proj_ppa[:, i].mean()
    sts_mean = proj_sts[:, i].mean()
    ffa_mean = proj_ffa[:, i].mean()
    print(f"PC{i+1:<4} {ppa_mean:>10.3f} {sts_mean:>10.3f} {ffa_mean:>10.3f}")
    roi_results.append({'PC': f'PC{i+1}', 'scene-responsive voxels': ppa_mean, 'STS': sts_mean, 'face sensitive voxels': ffa_mean})

# Save to CSV
df_roi = pd.DataFrame(roi_results)
roi_csv_path = os.path.join(OUTPUT_DIR, 'roi_mean_projections.csv')
df_roi.to_csv(roi_csv_path, index=False)
print(f"\nSaved: {roi_csv_path}")

# ----------------------------------------------------------------------------
# 8. Plot ROI scatter
# ----------------------------------------------------------------------------
print("\nCreating ROI scatter plot...")
plt.figure(figsize=(10, 8))
plt.scatter(proj_ppa[:, 0], proj_ppa[:, 1], alpha=0.4, s=4, label='scene-responsive voxels', c='blue')
plt.scatter(proj_sts[:, 0], proj_sts[:, 1], alpha=0.4, s=4, label='STS', c='red')
plt.scatter(proj_ffa[:, 0], proj_ffa[:, 1], alpha=0.4, s=4, label='face sensitive voxels', c='green')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('PC1', fontsize=16)
plt.ylabel('PC2', fontsize=16)
plt.title(f'{MODEL_NAME} - ROI Projections', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, 'roi_scatter_PC1_PC2.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {scatter_path}")

# ----------------------------------------------------------------------------
# 9. t-SNE visualization colored by annotations
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("Creating t-SNE visualizations...")
print("="*80)

# Run t-SNE on PC activations (timepoints in PC space)
print("Running t-SNE on PC activations...")
n_components_for_tsne = min(10, pc_activations.shape[1])  # Use first 10 PCs for t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
pc_tsne = tsne.fit_transform(pc_activations[:, :n_components_for_tsne])

print(f"t-SNE embedding shape: {pc_tsne.shape}")

# For each annotation, create a scatter plot
for ann_name in annotation_names:
    print(f"  Creating t-SNE plot for {ann_name}...")

    ann_data = np.asarray(annotations[ann_name], dtype=np.float64).flatten()

    # Trim to minimum length
    min_len = min(len(pc_tsne), len(ann_data))

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pc_tsne[:min_len, 0], pc_tsne[:min_len, 1],
                         c=ann_data[:min_len],
                         cmap='viridis',
                         alpha=0.6,
                         s=20)
    plt.colorbar(scatter, label=ann_name)
    plt.xlabel('t-SNE dimension 1', fontsize=14)
    plt.ylabel('t-SNE dimension 2', fontsize=14)
    plt.title(f'{MODEL_NAME} - PC activations colored by {ann_name}', fontsize=16)
    plt.tight_layout()

    tsne_path = os.path.join(OUTPUT_DIR, f'tsne_{ann_name}.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {tsne_path}")

# ----------------------------------------------------------------------------
# 10. Create brain maps
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("Creating brain maps...")
print("="*80)

original_shape = (61, 73, 61)

# Load template for affine
template_path = f"{RESULTS_DIR_PPA}/{MODEL_NAME}/trial_1/subject1/{MODEL_NAME}_r_sub1.nii"
template_nii = nib.load(template_path)
affine = template_nii.affine

for pc_idx in range(N_BRAIN_MAPS):
    print(f"\nCreating PC{pc_idx+1} brain map...")
    group_map = np.zeros(original_shape)
    count_map = np.zeros(original_shape)

    # Process PPA
    for subj in range(1, N_SUBJECTS + 1):
        brain_indices = np.load(
            f"{RESULTS_DIR_PPA}/{MODEL_NAME}/trial_1/subject{subj}/{MODEL_NAME}_top1000_brain_indices_sub{subj}.npy",
            allow_pickle=True
        )
        start_idx = (subj - 1) * N_VOXELS_PER_SUBJECT
        end_idx = subj * N_VOXELS_PER_SUBJECT
        subj_projections = voxel_projections[start_idx:end_idx, pc_idx]

        group_map[brain_indices[0], brain_indices[1], brain_indices[2]] += subj_projections
        count_map[brain_indices[0], brain_indices[1], brain_indices[2]] += 1

    # Process STS
    for subj in range(1, N_SUBJECTS + 1):
        brain_indices = np.load(
            f"{RESULTS_DIR_STS}/{MODEL_NAME}/trial_1/subject{subj}/{MODEL_NAME}_top1000_brain_indices_sub{subj}.npy",
            allow_pickle=True
        )
        start_idx = n_voxels_per_roi + (subj - 1) * N_VOXELS_PER_SUBJECT
        end_idx = n_voxels_per_roi + subj * N_VOXELS_PER_SUBJECT
        subj_projections = voxel_projections[start_idx:end_idx, pc_idx]

        group_map[brain_indices[0], brain_indices[1], brain_indices[2]] += subj_projections
        count_map[brain_indices[0], brain_indices[1], brain_indices[2]] += 1

    # Process FFA
    for subj in range(1, N_SUBJECTS + 1):
        brain_indices = np.load(
            f"{RESULTS_DIR_FFA}/{MODEL_NAME}/trial_1/subject{subj}/{MODEL_NAME}_top1000_brain_indices_sub{subj}.npy",
            allow_pickle=True
        )
        start_idx = 2 * n_voxels_per_roi + (subj - 1) * N_VOXELS_PER_SUBJECT
        end_idx = 2 * n_voxels_per_roi + subj * N_VOXELS_PER_SUBJECT
        subj_projections = voxel_projections[start_idx:end_idx, pc_idx]

        group_map[brain_indices[0], brain_indices[1], brain_indices[2]] += subj_projections
        count_map[brain_indices[0], brain_indices[1], brain_indices[2]] += 1

    # Average
    mask = count_map > 0
    group_map[mask] /= count_map[mask]

    # Save as NIfTI
    pc_img = nib.Nifti1Image(group_map, affine)
    brain_map_path = os.path.join(OUTPUT_DIR, f'PC{pc_idx+1}_brain_map.nii.gz')
    pc_img.to_filename(brain_map_path)

    print(f"  Range: [{group_map[group_map != 0].min():.2f}, {group_map[group_map != 0].max():.2f}]")
    print(f"  Saved: {brain_map_path}")

# ----------------------------------------------------------------------------
# Done!
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}\n")
print("Files created:")
print("  - pc_annotation_correlations.csv")
print("  - correlation_heatmap.png")
print("  - roi_mean_projections.csv")
print("  - roi_scatter_PC1_PC2.png")
for ann_name in annotation_names:
    print(f"  - tsne_{ann_name}.png")
for i in range(N_BRAIN_MAPS):
    print(f"  - PC{i+1}_brain_map.nii.gz")
print()
