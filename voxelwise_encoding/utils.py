import os
import numpy as np
import nibabel as nib
from nilearn import image


def clean_image(fmri_path, subj, mask, results_dir):
    """
    Read fMRI data, mask the image with intersubject correlation map (r>0.25),
    clean the data (remove NaN, Inf, columns with constant values), save the nii file.
    """
    subj_results_dir = os.path.join(results_dir, f'sub{subj}')
    os.makedirs(subj_results_dir, exist_ok=True)
    npy_path = os.path.join(subj_results_dir, f'fmri_s{subj}.npy')
    masked_indices_path = os.path.join(subj_results_dir, f'masked_indices_s{subj}.npy')
    original_data_shape_path = os.path.join(subj_results_dir, f'original_data_shape_s{subj}.npy')
    affine_path = os.path.join(subj_results_dir, f'affine_s{subj}.npy')

    if os.path.isfile(npy_path) and os.path.isfile(masked_indices_path):
        print(f'all files exist for the subject {subj}.\nskip the operation.')
        data_clean = np.load(npy_path, allow_pickle=True)
        mask_indices = np.load(masked_indices_path, allow_pickle=True)
        original_data_shape = np.load(original_data_shape_path, allow_pickle=True)
        img_affine = np.load(affine_path, allow_pickle=True)
    else:
        original_data, img_affine = load_fmri_data(fmri_path)
        original_data_shape = original_data.shape[:3]
        masked_data, mask_indices = apply_mask(original_data, mask)
        print(f'original data shape: {original_data_shape}, masked data shape: {masked_data.shape}')
        # data_clean = remove_useless_data(masked_data)
        data_clean = masked_data
        data_clean = data_clean.T
        np.save(npy_path, data_clean)
        np.save(masked_indices_path, mask_indices)
        np.save(original_data_shape_path, original_data_shape)
        np.save(affine_path, img_affine)
        print(f'saving npy files for the subject {subj}.')

    return data_clean, mask_indices, original_data_shape, img_affine


def load_fmri_data(filepath):
    img = image.load_img(filepath)
    data = img.get_fdata()
    return data, img.affine


def apply_mask(data, mask):
    if mask is not None:
        mask_data = nib.load(mask).get_fdata()
        new_mask = mask_data > 0.25
        # get the indices of the mask
        mask_indices = np.where(new_mask)
        new_data = data[mask_indices]
        return new_data, mask_indices
    else:
        mask_indices = np.where(np.ones(data.shape[:3])) 
        new_data = data[mask_indices]  
        return new_data, mask_indices


def save_as_nii(model, subj, r_mean, weight_mean, savedir, name_feature, img_affine):
    """
    Save data to NIfTI format.
    """
    # Update data_clean.samples with r_mean and save to NIfTI
    save_nifti(r_mean, os.path.join(savedir, f"{model}_r_sub{subj}.nii"), affine=img_affine)

    # Save each feature's weights to a separate NIfTI file
    for i, feature_name in enumerate(name_feature):
        encoding_weights = weight_mean[i, :]
        save_nifti(encoding_weights, os.path.join(savedir, f"{feature_name}_sub{subj}.nii"), affine=img_affine)

    return


def save_group_nii(model, r_mean, weight_mean, savedir, name_feature, img_affine):
    """
    Save data to NIfTI format.
    """
    # Update data_clean.samples with r_mean and save to NIfTI
    save_nifti(r_mean, os.path.join(savedir, f"{model}_r.nii"), affine=img_affine)

    # Save each feature's weights to a separate NIfTI file
    for i, feature_name in enumerate(name_feature):
        encoding_weights = weight_mean[i, :]
        save_nifti(encoding_weights, os.path.join(savedir, f"{feature_name}.nii"), affine=img_affine)

    return


def save_nifti(data, filename, affine=None):
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, filename)