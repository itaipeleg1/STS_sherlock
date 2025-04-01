import nibabel as nib
import numpy as np
from nilearn import plotting


def plot_voxelwise_encoding_results_on_surface(results_file_path: str, model:str, feature:str):
    nii = nib.load(results_file_path)
    data = nii.get_fdata()
    # replace nan with 0, as nans are voxels that are outside the ISC mask
    data = np.nan_to_num(data, nan=0.0)
    # for visualization purposes, only plot positive correlations
    data[data < 0] = 0
    img = nib.Nifti1Image(data, affine=nii.affine)

    plotting.view_img_on_surf(img, surf_mesh='fsaverage', title=f'{model} - {feature}, max r: {np.max(data)}',
                              symmetric_cmap=False, cmap=plotting.cm.black_red, vmax=np.max(data)).open_in_browser()


path = r"C:\uni\Msc Brain\Lab work\STS_sherlock\projects data\results\exp_sts_social_video6s_afteraveraging\group\llava_video_6s\trial_1\llava_6s_video_results_primitives.nii"
plot_voxelwise_encoding_results_on_surface(path, model='face', feature='llava')

