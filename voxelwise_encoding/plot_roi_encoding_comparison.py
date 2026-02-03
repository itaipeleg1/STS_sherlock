#!/usr/bin/env python3
"""
Plot encoding model performance comparison across ROIs and models.

This script analyzes the top-performing voxels in different brain regions
and compares how well different embedding models predict neural responses.
"""

import argparse
import nibabel as nib
import numpy as np
from nilearn import image
import os
import matplotlib.pyplot as plt


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare encoding model performance across ROIs'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['clap_full', 'clip_full', 'social'],
        help='List of models to compare (default: clap_full clip_full Leyla_social_model)'
    )
    parser.add_argument(
        '--regions',
        nargs='+',
        default=['sts_anterior', 'sts_posterior', 'ffa', 'ppa'],
        help='List of brain regions to analyze (default: sts_anterior sts_posterior ffa ppa)'
    )
    parser.add_argument(
        '--voxel-percentage',
        type=float,
        default=20.0,
        help='Percentage of top voxels to use (default: 10.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/new_storage/sherlock/STS_sherlock/encoding_models_comparison.png',
        help='Output path for the figure (default: encoding_models_comparison.png)'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='/home/new_storage/sherlock/STS_sherlock/projects data/results',
        help='Base path for results data'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figure (default: 300)'
    )

    return parser.parse_args()


def collect_results(models, regions, base_path, voxel_percentage):
    """
    Collect encoding model results for all models, regions, and subjects.
    Calculates the top percentage of voxels dynamically for each subject.

    Parameters
    ----------
    models : list
        List of model names
    regions : list
        List of brain regions
    base_path : str
        Base path to results directory
    voxel_percentage : float
        Percentage of top voxels to use per subject

    Returns
    -------
    dict
        Nested dictionary containing results for each model/region/hemisphere
    """
    # Get subject list
    sample_model = models[0]
    sample_region = regions[0]
    sample_path = os.path.join(base_path, f"{sample_model}_{sample_region}/{sample_model}/trial_1")
    subject_list = [s for s in os.listdir(sample_path) if s != "group"]

    # Initialize storage
    results = {m: {r: {'left': [], 'right': []} for r in regions} for m in models}

    # Loop through everything
    for model in models:
        for region in regions:
            for subj in subject_list:
                fmri_path = os.path.join(
                    base_path,
                    f"{model}_{region}/{model}/trial_1",
                    f"subject{subj[7:]}",
                    f"{model}_r_sub{subj[7:]}.nii"
                )

                if not os.path.exists(fmri_path):
                    print(f"Missing: {fmri_path}")
                    continue

                img = nib.load(fmri_path)
                data = img.get_fdata()

                # Get x coordinates for hemisphere split
                coords = image.coord_transform(
                    *np.where(np.ones(img.shape[:3])),
                    img.affine
                )
                x_coords_3d = coords[0].reshape(img.shape[:3])

                left_mask = x_coords_3d < 0
                right_mask = x_coords_3d > 0

                # Get data for each hemisphere
                left_data = data[left_mask]
                right_data = data[right_mask]

                # Clean NaNs
                left_clean = left_data[~np.isnan(left_data) & (left_data > 0)]
                right_clean = right_data[~np.isnan(right_data) & (right_data > 0)]

                # Calculate top N dynamically for this subject's mask
                n_left = max(1, int(len(left_clean) * voxel_percentage / 100))
                n_right = max(1, int(len(right_clean) * voxel_percentage / 100))

                top_left = np.sort(left_clean)[-n_left:]
                top_right = np.sort(right_clean)[-n_right:]

                # Store mean R² (square r to get R²)
                results[model][region]['left'].append(np.mean(top_left ** 2))
                results[model][region]['right'].append(np.mean(top_right ** 2))

    return results


def plot_results(results, models, regions, voxel_percentage, output_path, dpi):
    """
    Create and save the comparison plot.

    Parameters
    ----------
    results : dict
        Results dictionary from collect_results
    models : list
        List of model names
    regions : list
        List of brain regions
    voxel_percentage : float
        Percentage of voxels used
    output_path : str
        Path to save the figure
    dpi : int
        DPI for saved figure
    """
    # Model display configuration
    model_display = {
        'clap_full': {'label': 'CLAP Audio', 'color': '#bebada'},
        'clip_full': {'label': 'CLIP Visual', 'color': '#8dd3c7'},
        'social': {'label': 'Social Model', 'color': '#fb8072'},
        'social_only': {'label': 'Social Model', 'color': '#fb8072'},
    }

    # Get labels and colors for the models in order
    model_labels = []
    model_colors = []
    for model in models:
        if model in model_display:
            model_labels.append(model_display[model]['label'])
            model_colors.append(model_display[model]['color'])
        else:
            # Default for unknown models
            model_labels.append(model.replace('_', ' ').title())
            model_colors.append('#999999')

    # Custom ROI label mapping
    roi_label_mapping = {
        'ppa': 'scene-responsive voxels',
        'ffa': 'face sensitive voxels'
    }

    # Build ROI keys and labels
    roi_keys = []
    roi_labels = []
    for r in regions:
        roi_keys.append((r, 'right'))
        roi_keys.append((r, 'left'))

        # Use custom label if available, otherwise use default
        if r.lower() in roi_label_mapping:
            name = roi_label_mapping[r.lower()]
            roi_labels.append(f"{name} Right")
            roi_labels.append(f"{name} Left")
        else:
            name = r.replace('_', ' ').title()
            roi_labels.append(f"{name} Right")
            roi_labels.append(f"{name} Left")

    # Create plot
    _, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(roi_keys))
    width = 0.25

    for i, (model, label, color) in enumerate(zip(models, model_labels, model_colors)):
        means = []
        all_subjects = []

        for (region, hemi) in roi_keys:
            subj_vals = results[model][region][hemi]
            means.append(np.mean(subj_vals))
            all_subjects.append(subj_vals)

        offset = (i - 1) * width
        ax.bar(x + offset, means, width, label=label, color=color,
               alpha=0.7, edgecolor='black')

        # Plot individual subject dots
        for j, subj_vals in enumerate(all_subjects):
            x_jitter = x[j] + offset + np.random.uniform(-width/5, width/5, len(subj_vals))
            ax.scatter(x_jitter, subj_vals, color='black', s=5, alpha=0.4, zorder=3)

    ax.set_ylabel(f'Mean R² (Top {voxel_percentage:.1f}% Voxels)',
                  fontsize=12, fontweight='bold')
    ax.set_xlabel('Region of Interest', fontsize=12, fontweight='bold')
    ax.set_title('Neural Response Prediction by Embedding Model Across ROIs',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels, fontsize=10, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 60)
    print("ROI Encoding Model Comparison")
    print("=" * 60)
    print(f"Models: {', '.join(args.models)}")
    print(f"Regions: {', '.join(args.regions)}")
    print(f"Voxel percentage: {args.voxel_percentage}%")
    print(f"Output path: {args.output}")
    print("=" * 60)

    # Collect results (percentage calculated per subject)
    print("\nCollecting results...")
    results = collect_results(args.models, args.regions, args.base_path,
                            args.voxel_percentage)

    # Plot and save
    print("\nGenerating plot...")
    plot_results(results, args.models, args.regions, args.voxel_percentage,
                args.output, args.dpi)

    print("\nDone!")


if __name__ == '__main__':
    main()