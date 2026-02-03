import numpy as np
import nibabel as nib
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import argparse
import sys
from nilearn import plotting


def load_r2_maps(model_path, model_name, subjects):
    """
    Load R² maps for all subjects from a given model folder.

    Parameters:
    -----------
    model_path : Path
        Base path to model results
    model_name : str
        Name of the model (e.g., 'social', 'clip_full', 'clap_full')
    subjects : list
        List of subject numbers

    Returns:
    --------
    r2_data : np.ndarray
        Array of shape (n_subjects, x, y, z) containing R² values
    affine : np.ndarray
        Affine transformation matrix from first subject
    """
    model_path = Path(model_path)
    r2_data = []
    affine = None

    for subj_num in subjects:
        # Construct path to correlation map
        r_file = model_path / f"subject{subj_num}" / f"{model_name}_r_sub{subj_num}.nii"

        if not r_file.exists():
            print(f"Warning: File not found: {r_file}")
            continue

        img = nib.load(r_file)
        r_values = img.get_fdata()

        # Convert correlation to R²
        r2_values = r_values ** 2
        r2_data.append(r2_values)

        if affine is None:
            affine = img.affine

        print(f"Loaded subject {subj_num}: {r_file.name}")

    return np.array(r2_data), affine


def compute_unique_variance(joint_r2, other_model_r2):
    """
    Compute unique variance for a model.

    Parameters:
    -----------
    joint_r2 : np.ndarray
        R² from joint model (model1 + model2 together)
    other_model_r2 : np.ndarray
        R² from the other model alone

    Returns:
    --------
    unique_var : np.ndarray
        Unique variance (can be negative)
    """
    return joint_r2 - other_model_r2


def test_unique_variance_greater_than_zero(unique_var, model_name):
    """
    Stage 1: Test if unique variance is significantly > 0 at each voxel.

    Parameters:
    -----------
    unique_var : np.ndarray
        Unique variance values, shape (n_subjects, x, y, z)
    model_name : str
        Name of the model

    Returns:
    --------
    sig_mask : np.ndarray
        Boolean mask where unique variance is significantly > 0
    """
    print(f"  Testing {model_name}: unique variance > 0...")

    # One-sample t-test against 0
    _, p_values = stats.ttest_1samp(unique_var, 0, axis=0, alternative='greater')

    # Flatten for FDR correction
    p_flat = p_values.flatten()
    valid_mask = ~np.isnan(p_flat)

    if np.sum(valid_mask) > 0:
        _, p_corrected_flat, _, _ = multipletests(
            p_flat[valid_mask],
            alpha=0.05,
            method='fdr_bh'
        )

        p_corrected = np.full_like(p_flat, 1.0)
        p_corrected[valid_mask] = p_corrected_flat
        p_corrected = p_corrected.reshape(p_values.shape)
    else:
        p_corrected = np.ones_like(p_values)

    sig_mask = p_corrected < 0.05
    n_sig = np.sum(sig_mask)
    mean_unique = np.mean(unique_var, axis=0)
    mean_sig = np.mean(mean_unique[sig_mask]) if n_sig > 0 else 0

    print(f"    {n_sig:,} voxels with unique variance > 0 (mean={mean_sig:.4f})")

    return sig_mask


def test_pairwise_preference(unique_var1, unique_var2, model1_name, model2_name):
    """
    Two-stage approach:
    Stage 1: Test if each model's unique variance > 0
    Stage 2: Among voxels where at least one model has unique variance > 0,
             test which model has significantly greater unique variance

    Parameters:
    -----------
    unique_var1 : np.ndarray
        Unique variance for model 1, shape (n_subjects, x, y, z)
    unique_var2 : np.ndarray
        Unique variance for model 2, shape (n_subjects, x, y, z)
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2

    Returns:
    --------
    preference_map : np.ndarray
        3D map with values: 0 (no sig difference), 1 (model1 wins), 2 (model2 wins)
    unique_var_map : np.ndarray
        3D map with unique variance magnitude of winning model
    final_sig_mask : np.ndarray
        Boolean mask of voxels with significant preference
    """
    n_subjects = unique_var1.shape[0]
    shape_3d = unique_var1.shape[1:]
    total_voxels = np.prod(shape_3d)

    print(f"\n{'='*60}")
    print(f"COMPARING: {model1_name.upper()} vs {model2_name.upper()}")
    print(f"{'='*60}")

    # Stage 1: Test each model's unique variance > 0
    print("\nStage 1: Testing if unique variance > 0 for each model...")
    sig_mask_model1 = test_unique_variance_greater_than_zero(unique_var1, model1_name)
    sig_mask_model2 = test_unique_variance_greater_than_zero(unique_var2, model2_name)

    # Combine masks: test voxels where at least one model has unique variance > 0
    any_sig_mask = np.logical_or(sig_mask_model1, sig_mask_model2)
    n_test_voxels = np.sum(any_sig_mask)
    print(f"\nVoxels with at least one model having unique variance > 0: {n_test_voxels:,} / {total_voxels:,}")

    # Stage 2: Among significant voxels, test which model has greater unique variance
    print(f"\nStage 2: Testing which model has greater unique variance...")

    # Compute mean unique variance for each model
    mean_unique_var1 = np.mean(unique_var1, axis=0)
    mean_unique_var2 = np.mean(unique_var2, axis=0)

    # Initialize outputs
    preference_map = np.zeros(shape_3d)
    unique_var_map = np.zeros(shape_3d)
    final_sig_mask = np.zeros(shape_3d, dtype=bool)

    # Flatten for easier processing
    unique_var1_flat = unique_var1.reshape(n_subjects, -1)
    unique_var2_flat = unique_var2.reshape(n_subjects, -1)
    any_sig_flat = any_sig_mask.flatten()

    # Get indices of voxels to test
    test_indices = np.where(any_sig_flat)[0]

    # Run pairwise t-tests only on voxels with at least one model unique variance > 0
    p_values = np.ones(total_voxels)

    for voxel_idx in test_indices:
        # Paired t-test: unique_var1 vs unique_var2
        _, p_val = stats.ttest_rel(
            unique_var1_flat[:, voxel_idx],
            unique_var2_flat[:, voxel_idx],
            alternative='two-sided'
        )
        p_values[voxel_idx] = p_val

    # FDR correction only on tested voxels
    valid_mask = np.isfinite(p_values) & any_sig_flat

    if np.sum(valid_mask) > 0:
        _, p_corrected_flat, _, _ = multipletests(
            p_values[valid_mask],
            alpha=0.05,
            method='fdr_bh'
        )

        p_corrected = np.ones_like(p_values)
        p_corrected[valid_mask] = p_corrected_flat
    else:
        p_corrected = np.ones_like(p_values)

    # Reshape back to 3D
    p_corrected = p_corrected.reshape(shape_3d)

    # Create preference map
    final_sig_mask = p_corrected < 0.05

    # Assign preference based on which model has higher mean unique variance
    model1_wins = final_sig_mask & (mean_unique_var1 > mean_unique_var2)
    model2_wins = final_sig_mask & (mean_unique_var2 > mean_unique_var1)

    preference_map[model1_wins] = 1
    preference_map[model2_wins] = 2

    # Unique variance map: use winner's unique variance
    unique_var_map[model1_wins] = mean_unique_var1[model1_wins]
    unique_var_map[model2_wins] = mean_unique_var2[model2_wins]

    # Print statistics
    n_model1 = np.sum(preference_map == 1)
    n_model2 = np.sum(preference_map == 2)
    n_no_pref = np.sum(any_sig_mask & ~final_sig_mask)

    print(f"\nResults:")
    if n_model1 > 0:
        print(f"  {model1_name} wins: {n_model1:,} voxels (mean unique var: {np.mean(mean_unique_var1[preference_map == 1]):.4f} max: {np.max(mean_unique_var1[preference_map == 1]):.4f})")
    else:
        print(f"  {model1_name} wins: 0 voxels")

    if n_model2 > 0:
        print(f"  {model2_name} wins: {n_model2:,} voxels (mean unique var: {np.mean(mean_unique_var2[preference_map == 2]):.4f} max: {np.max(mean_unique_var2[preference_map == 2]):.4f})")
    else:
        print(f"  {model2_name} wins: 0 voxels")

    print(f"  No significant difference: {n_no_pref:,} voxels")
    print(f"  Total with preference: {n_model1 + n_model2:,} voxels")

    return preference_map, unique_var_map, final_sig_mask


def create_visualization(preference_map, unique_var_map, affine,
                         model1_name, model2_name, output_path):
    """
    Create visualizations for pairwise comparison.

    Parameters:
    -----------
    preference_map : np.ndarray
        3D map with model preferences (0, 1, 2)
    unique_var_map : np.ndarray
        3D map with unique variance magnitudes
    affine : np.ndarray
        Affine transformation matrix
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2
    output_path : str
        Path to save outputs
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create separate maps for each model
    model1_map = np.where(preference_map == 1, unique_var_map, 0)
    model2_map = np.where(preference_map == 2, unique_var_map, 0)

    # Save model 1 map
    if np.sum(preference_map == 1) > 0:
        img1 = nib.Nifti1Image(model1_map, affine)
        nii_path1 = output_path.parent / f"{output_path.stem}_{model1_name}_wins.nii.gz"
        img1.to_filename(nii_path1)
        print(f"Saved {model1_name} map: {nii_path1}")

        vmax1 = np.max(model1_map)
        mean_unique_1 = np.mean(model1_map[model1_map > 0])
        n_voxels_1 = np.sum(preference_map == 1)
        title1 = f'{model1_name.upper()} > {model2_name.upper()} (n={n_voxels_1:,}, mean unique var={mean_unique_1:.4f}, max={vmax1:.4f})'

        html_path1 = output_path.parent / f"{output_path.stem}_{model1_name}_wins.html"
        plotting.view_img_on_surf(
            img1,
            surf_mesh='fsaverage',
            title=title1,
            symmetric_cmap=False,
            cmap='hot',
            colorbar=True,
            vmax=vmax1,
            threshold=0.0001
        ).save_as_html(html_path1)
        print(f"Saved {model1_name} HTML: {html_path1}")
    else:
        print(f"No voxels where {model1_name} > {model2_name}, skipping visualization")

    # Save model 2 map
    if np.sum(preference_map == 2) > 0:
        img2 = nib.Nifti1Image(model2_map, affine)
        nii_path2 = output_path.parent / f"{output_path.stem}_{model2_name}_wins.nii.gz"
        img2.to_filename(nii_path2)
        print(f"Saved {model2_name} map: {nii_path2}")

        vmax2 = np.max(model2_map)
        mean_unique_2 = np.mean(model2_map[model2_map > 0])
        n_voxels_2 = np.sum(preference_map == 2)
        title2 = f'{model2_name.upper()} > {model1_name.upper()} (n={n_voxels_2:,}, mean unique var={mean_unique_2:.4f}, max={vmax2:.4f})'

        html_path2 = output_path.parent / f"{output_path.stem}_{model2_name}_wins.html"
        plotting.view_img_on_surf(
            img2,
            surf_mesh='fsaverage',
            title=title2,
            symmetric_cmap=False,
            cmap='hot',
            colorbar=True,
            vmax=vmax2,
            threshold=0.0001
        ).save_as_html(html_path2)
        print(f"Saved {model2_name} HTML: {html_path2}")
    else:
        print(f"No voxels where {model2_name} > {model1_name}, skipping visualization")

    # Create composite map (model1=positive, model2=negative)
    composite_map = np.zeros_like(unique_var_map)
    composite_map[preference_map == 1] = unique_var_map[preference_map == 1]
    composite_map[preference_map == 2] = -unique_var_map[preference_map == 2]

    if np.sum(preference_map > 0) > 0:
        composite_img = nib.Nifti1Image(composite_map, affine)
        composite_nii_path = output_path.parent / f"{output_path.stem}_composite.nii.gz"
        composite_img.to_filename(composite_nii_path)
        print(f"Saved composite map: {composite_nii_path}")

        vmax_composite = max(np.abs(np.min(composite_map)), np.max(composite_map))
        title_composite = f'{model1_name.upper()} (red) vs {model2_name.upper()} (blue)'
        html_composite_path = output_path.parent / f"{output_path.stem}_composite.html"

        plotting.view_img_on_surf(
            composite_img,
            surf_mesh='fsaverage',
            title=title_composite,
            symmetric_cmap=True,
            cmap='cold_hot',
            colorbar=True,
            vmax=vmax_composite,
            threshold=0.0001
        ).save_as_html(html_composite_path)
        print(f"Saved composite HTML: {html_composite_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create preference map comparing unique variance of two models'
    )

    # Paths to the 3 required models
    parser.add_argument('--joint', type=str, required=True,
                        help='Path to joint model (model1+model2) results folder')
    parser.add_argument('--model1', type=str, required=True,
                        help='Path to model1 alone results folder')
    parser.add_argument('--model2', type=str, required=True,
                        help='Path to model2 alone results folder')

    # Model names (for file loading)
    parser.add_argument('--joint_name', type=str, required=True,
                        help='Model name for joint model files (e.g., "social_clip")')
    parser.add_argument('--model1_name', type=str, required=True,
                        help='Model name for model1 files (e.g., "social")')
    parser.add_argument('--model2_name', type=str, required=True,
                        help='Model name for model2 files (e.g., "clip_full")')

    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for HTML visualization (e.g., /path/to/model1_vs_model2.html)')

    # Subject info
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=list(range(1, 18)),
                        help='List of subject numbers (default: 1-17)')

    parser.add_argument('--trial', type=int, default=1,
                        help='Trial number (default: 1)')

    args = parser.parse_args()

    # Append trial folder to paths
    joint_path = Path(args.joint)
    if not joint_path.name == args.joint_name:
        joint_path = joint_path / args.joint_name
    joint_path = joint_path / f"trial_{args.trial}"

    model1_path = Path(args.model1)
    if not model1_path.name == args.model1_name:
        model1_path = model1_path / args.model1_name
    model1_path = model1_path / f"trial_{args.trial}"

    model2_path = Path(args.model2)
    if not model2_path.name == args.model2_name:
        model2_path = model2_path / args.model2_name
    model2_path = model2_path / f"trial_{args.trial}"

    print("="*80)
    print("UNIQUE VARIANCE PREFERENCE MAPPING")
    print("="*80)
    print(f"\nLoading models:")
    print(f"  Joint: {args.joint_name} from {joint_path}")
    print(f"  Model 1: {args.model1_name} from {model1_path}")
    print(f"  Model 2: {args.model2_name} from {model2_path}")
    print(f"\nSubjects: {args.subjects}")
    print(f"Output: {args.output}\n")

    # Load R² maps for all 3 models
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    print(f"\nLoading joint model ({args.joint_name})...")
    joint_r2, affine = load_r2_maps(joint_path, args.joint_name, args.subjects)

    print(f"\nLoading {args.model1_name} model...")
    model1_r2, _ = load_r2_maps(model1_path, args.model1_name, args.subjects)

    print(f"\nLoading {args.model2_name} model...")
    model2_r2, _ = load_r2_maps(model2_path, args.model2_name, args.subjects)

    # Compute unique variance for each model
    print("\n" + "="*80)
    print("COMPUTING UNIQUE VARIANCE")
    print("="*80)

    print(f"\n{args.model1_name} unique = Joint - {args.model2_name}")
    unique_var1 = compute_unique_variance(joint_r2, model2_r2)
    print(f"  Mean: {np.nanmean(unique_var1):.4f}")
    print(f"  Max: {np.nanmax(unique_var1):.4f}")
    print(f"  Min: {np.nanmin(unique_var1):.4f}")

    print(f"\n{args.model2_name} unique = Joint - {args.model1_name}")
    unique_var2 = compute_unique_variance(joint_r2, model1_r2)
    print(f"  Mean: {np.nanmean(unique_var2):.4f}")
    print(f"  Max: {np.nanmax(unique_var2):.4f}")
    print(f"  Min: {np.nanmin(unique_var2):.4f}")

    # Test preference
    print("\n" + "="*80)
    print("STATISTICAL TESTING")
    print("="*80)

    preference_map, unique_var_map, _ = test_pairwise_preference(
        unique_var1, unique_var2, args.model1_name, args.model2_name
    )

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    create_visualization(
        preference_map, unique_var_map, affine,
        args.model1_name, args.model2_name, args.output
    )

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        joint_path = "/home/new_storage/sherlock/STS_sherlock/projects data/results/social_clip_clap_whole"
        model1_path = "/home/new_storage/sherlock/STS_sherlock/projects data/results/clip_clap_whole"
        model2_path = "/home/new_storage/sherlock/STS_sherlock/projects data/results/social/social"
        joint_name = "social_clip_clap"
        model1_name = "clip_clap"
        model2_name = "social"
        output = "/home/new_storage/sherlock/STS_sherlock/projects data/preference_maps/clip_clap_vs_social_unique.html"
        subjects = list(range(1, 18))

        sys.argv = [
            sys.argv[0],
            "--joint", joint_path,
            "--model1", model1_path,
            "--model2", model2_path,
            "--joint_name", joint_name,
            "--model1_name", model1_name,
            "--model2_name", model2_name,
            "--output", output,
            "--subjects"
        ] + [str(s) for s in subjects] + ["--trial", "1"]

    main()
