import nibabel as nib
import numpy as np
from nilearn import plotting, surface,datasets
import cortex
import matplotlib.pyplot as plt
import os
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import copy




def plot_voxelwise_encoding_results_on_surface(results_file_path: str,
                                                model: str,
                                                feature: str,
                                                output_path: str = None,
                                                vmax: float = None,
                                                vmin: float = None,):
    """
    Plot using nilearn on fsaverage surface (original method)

    Parameters:
    -----------
    results_file_path : str
        Path to NIfTI file with voxelwise results
    model : str
        Model name for title/filename
    feature : str
        Feature name for title/filename
    apply_fdr : bool
        If True, apply FDR correction (requires n_test_samples)
    n_test_samples : int, optional
        Number of test samples used in encoding model (required if apply_fdr=True)
    fdr_alpha : float
        FDR significance level (default: 0.05)
    output_path : str, optional
        Path to save HTML output. If None, uses default location
    vmax : float, optional
        Maximum value for colormap. If None, uses data max
    """
    nii = nib.load(results_file_path)
    data = nii.get_fdata()
    # replace nan with 0, as nans are voxels that are outside the ISC mask
    print("min and max before nan to num:", np.nanmin(data), np.nanmax(data))   
    #data = np.nan_to_num(data, nan=0)


    model = model
    #data = apply_fdr_correction(data, n_test_samples, alpha=fdr_alpha)

    # for visualization purposes, only plot positive correlations
    #threshold = np.percentile(data[data > 0], 90)

    # Set vmax if not specified
    if vmax is None:
        vmax = np.max(data)

    # Set output path if not specified
    if output_path is None:
        output_path = f"/home/new_storage/sherlock/STS_sherlock/projects data/results/{model}.html"

    img = nib.Nifti1Image(data, affine=nii.affine)
    title = f'{model} - {feature}, max r: {np.max(data):.4f},min r {np.min(data):.4f}, avg top 100 r: {np.mean(np.sort(data[data > 0])[-100:]):.4f}' if np.sum(data > 0) >= 100 else f'{model} - {feature}, max r: {np.max(data):.4f}'


    #plotting.view_img_on_surf(img, surf_mesh='fsaverage', title=title,
          #                  symmetric_cmap=False, cmap=cmap, colorbar=True, 
           #                 vmax=vmax, vmin=vmin).save_as_html(output_path)
    plotting.view_img_on_surf(img, surf_mesh='fsaverage', title=title,
                              symmetric_cmap=False, cmap='cold_hot',colorbar=True, vmax=vmax,bg_on_data=True).save_as_html(output_path)

    print(f"Saved Nilearn surface plot to: {output_path}")





if __name__ == '__main__':
    # Example usage

    path = '/home/new_storage/sherlock/STS_sherlock/projects data/preference_maps/social_vs_clap_unique_composite.html'


    # Method 1: Original nilearn method (fsaverage surface) - WORKS WITH MNI DATA
    plot_voxelwise_encoding_results_on_surface(path, model=f'Social vs Clap composite ', feature='')