import numpy as np
import nibabel as nib
from scipy import stats
from pathlib import Path
from utils import apply_mask
from statsmodels.stats.multitest import multipletests


def compute_unique_variance_maps(model1_base_folder, model2_base_folder,model1_name,model2_name,mask=None, output_folder="/home/new_storage/sherlock/STS_sherlock/projects data/results"):
    """
    Simple function to compute unique variance maps for CLIP vs LLaVA+CLIP.
    
    Parameters:
    -----------
    model1_base_folder : str
        Base folder containing subject folders with clip_r_subX.nii files
    model2_base_folder : str  
        Base folder containing subject folders with clip_llava_r_subX.nii files
    output_folder : str
        Where to save results
    """
    
    model1_path = Path(model1_base_folder)
    model2_path = Path(model2_base_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Find all subject folders
    model1_subjects = sorted([d for d in model1_path.iterdir() if d.is_dir() and d.name.startswith('subject')])
    model2_subjects = sorted([d for d in model2_path.iterdir() if d.is_dir() and d.name.startswith('subject')])
    
    print(f"Found {len(model1_subjects)} CLIP subjects and {len(model2_subjects)} LLaVA subjects")
    
    # Match subjects by number
    model1_dict = {int(d.name.replace('subject', '')): d for d in model1_subjects}
    model2_dict = {int(d.name.replace('subject', '')): d for d in model2_subjects}
    
    common_subjects = sorted(set(model1_dict.keys()) & set(model2_dict.keys()))
    print(f"Common subjects: {common_subjects}")
    
    # Load first file to get dimensions
    first_model1_file = model1_dict[common_subjects[0]] / f"{model1_name}_r_sub{common_subjects[0]}.nii"
    first_img = nib.load(first_model1_file)
    affine = first_img.affine
    shape = first_img.shape
    
    # Store all data
    n_subjects = len(common_subjects)
    model1_data = np.zeros((n_subjects,) + shape)
    model2_data = np.zeros((n_subjects,) + shape)
    
    # Load all subjects
    for i, subj_num in enumerate(common_subjects):
        # Load CLIP correlation map
        model1_file = model1_dict[subj_num] / f"{model1_name}_r_sub{subj_num}.nii"
        clip_img = nib.load(model1_file)
        model1_data[i] = clip_img.get_fdata()
        
        # Load LLaVA+CLIP correlation map  
        model2_file = model2_dict[subj_num] / f"{model2_name}_r_sub{subj_num}.nii"
        llava_img = nib.load(model2_file)
        model2_data[i] = llava_img.get_fdata()
        
        print(f"Loaded subject {subj_num}")
    
    # Convert correlations to r-squared (variance explained)
    model1_r2 = model1_data**2
    model2_r2 = model2_data**2
    
    # Compute unique variance
    # LLaVA unique = what LLaVA+CLIP explains beyond CLIP alone
    unique_raw = model2_r2 - model1_r2  # Keep negative values for now

    # Test if unique variance is significantly > 0
    t_stats, p_values = stats.ttest_1samp(unique_raw, 0, axis=0, alternative='two-sided')
    
    # Apply FDR correction for multiple comparisons
    p_flat = p_values.flatten()
    valid_mask = ~np.isnan(p_flat)
    
    if np.sum(valid_mask) > 0:
        # FDR correction only on valid p-values
        _, p_corrected_flat, _, _ = multipletests(p_flat[valid_mask], alpha=0.05, method='fdr_bh')
        
        # Reconstruct corrected p-values
        p_corrected = np.full_like(p_flat, 1.0)
        p_corrected[valid_mask] = p_corrected_flat
        p_corrected = p_corrected.reshape(p_values.shape)
    else:
        p_corrected = np.ones_like(p_values)
    
    # Create significance mask (FDR corrected p < 0.05)
    sig_mask = p_corrected < 0.05
    
    # Now apply positive constraint only to significant voxels
    unique =  unique_raw
    
    # For individual subjects, save their unique variance maps
    for i, subj_num in enumerate(common_subjects):
        unique_img = nib.Nifti1Image(unique[i], affine)
        #unique_img.to_filename(output_path / f"llava_unique_sub{subj_num}.nii.gz")
    
    # Group analysis: mean 
    group_mean = np.mean(unique, axis=0)
    
    # Group map with only significant voxels
    group_sig = group_mean * sig_mask
    
    # Apply mask if provided
    if mask is not None:
        print(f"Applying mask: {mask}")
        # Apply mask to get masked versions
        group_sig_masked, mask_indices = apply_mask(group_sig, mask)
        group_mean_masked, _ = apply_mask(group_mean, mask)
        
        # Create masked 3D volumes (zero outside mask)
        group_sig_3d = np.zeros_like(group_sig)
        group_mean_3d = np.zeros_like(group_mean)
        
        # Put masked data back into 3D space
        group_sig_3d[mask_indices] = group_sig_masked
        group_mean_3d[mask_indices] = group_mean_masked
        
        # Save masked versions
        group_mean_img = nib.Nifti1Image(group_mean_3d, affine)
        group_mean_img.to_filename(output_path / f"{model2_name}-{model1_name}_unique_group_mean_sts_masked.nii.gz")
        
        group_sig_img = nib.Nifti1Image(group_sig_3d, affine)
        group_sig_img.to_filename(output_path / f"{model2_name}-{model1_name}_unique_group_significant_sts_masked.nii.gz")

        print(f"Voxels in mask: {len(group_sig_masked):,}")
    

    group_mean_img = nib.Nifti1Image(group_mean, affine)
    group_mean_img.to_filename(output_path / f"{model2_name}-{model1_name}_unique_group_mean_sts.nii.gz")

    group_sig_img = nib.Nifti1Image(group_sig, affine)
    group_sig_img.to_filename(output_path / f"{model2_name}-{model1_name}_unique_group_significant_sts.nii.gz")

    sig_mask_img = nib.Nifti1Image(sig_mask.astype(float), affine)
    sig_mask_img.to_filename(output_path / "significance_mask.nii.gz")
    
    print(f"\nResults saved to: {output_path}")
    print(f"Significant voxels: {np.sum(sig_mask):,} out of {np.prod(shape):,}")
    print(f"Mean unique variance: {np.nanmean(group_mean):.4f}")
    print(f"Max unique variance: {np.nanmax(group_mean):.4f}")
    print(f"Min unique variance: {np.nanmin(group_mean):.4f}")
    print(f"NaN voxels in group_mean: {np.sum(np.isnan(group_mean)):,}")
    print(f"Mean significant unique variance: {np.nanmean(group_sig):.4f}")
    
    return {
        'llava_unique': unique,
        'group_mean': group_mean, 
        'group_significant': group_sig,
        'sig_mask': sig_mask,
        'subjects': common_subjects
    }


if __name__ == "__main__":
    model1_folder = "/home/new_storage/sherlock/STS_sherlock/projects data/results/llava_social_whole/cls_social/trial_1"
    model2_folder = "/home/new_storage/sherlock/STS_sherlock/projects data/results/llava_social_uniquevar_whole/unique_variance_social/trial_1"

    results = compute_unique_variance_maps(model1_folder, model2_folder, model1_name="cls_social", model2_name="unique_variance_social", mask=None)