import nibabel as nib
import numpy as np
from nilearn import plotting


def plot_voxelwise_encoding_results_on_surface(results_file_path: str, model:str, feature:str):
    nii = nib.load(results_file_path)
    data = nii.get_fdata()
    # replace nan with 0, as nans are voxels that are outside the ISC mask
    data[data < 0] = 0  
    data = np.nan_to_num(data, nan=0.0)
    # for visualization purposes, only plot positive correlations
    #threshold = np.percentile(data[data > 0], 90)
    img = nib.Nifti1Image(data, affine=nii.affine)
    plotting.view_img_on_surf(img, surf_mesh='fsaverage', title=f'{model} - {feature}, max r: {np.max(data)}, avg top 100 r: {np.mean(np.sort(data[data > 0])[-100:])}',
                              symmetric_cmap=False, cmap=plotting.cm.black_red, vmax=np.max(data)).save_as_html(r"/home/new_storage/sherlock/STS_sherlock/projects data/results/cls_social_layer25_40tokens.html")



path = r"/home/new_storage/sherlock/STS_sherlock/projects data/results/llava_social_layer25_40tokens_whole/cls_social_layer25_40tokens/trial_1/subject3/cls_social_layer25_40tokens_r_sub3.nii"

plot_voxelwise_encoding_results_on_surface(path, model='CLS Social Layer 25 40 Tokens', feature='')

