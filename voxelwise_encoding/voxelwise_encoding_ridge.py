import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import time
import argparse
import logging
import sys


from utils import clean_image, save_group_nii, save_as_nii
from models_config import models_config_dict
import os
import torch
import numpy as np

def load_cls_matrix(embedding_folder, sort_by_index=True, return_indices=False):
    """
    Loads CLS embeddings saved as .pt files into a NumPy matrix.

    Args:
        embedding_folder (str or Path): Path to the folder with .pt embeddings.
        sort_by_index (bool): Whether to sort files by numeric filename.
        return_indices (bool): If True, also return the sorted list of indices.

    Returns:
        np.ndarray: Matrix of shape (n_segments, embed_dim)
        list (optional): List of segment/frame indices
    """
    embedding_folder = Path(embedding_folder)
    files = [f for f in embedding_folder.glob("*.pt")]

    if sort_by_index:
        files = sorted(files, key=lambda x: int(x.stem))

    embeddings = []
    indices = []

    for f in files:
        tensor = torch.load(f)
        if tensor.dim() > 1:
            tensor = tensor.view(-1)
        embeddings.append(tensor.cpu().numpy())
        indices.append(int(f.stem))

    matrix = np.stack(embeddings)

    if return_indices:
        return matrix, indices
    return matrix


def concat_features(features_list, single_features_dir):
    processed_annotations = [np.load(os.path.join(single_features_dir, f'{item}.npy'),allow_pickle=True) for item in features_list]
    return np.concatenate(processed_annotations, axis=1)

def main(data_path, annotations_path, mask_path, model, results_dir, original_data_shape, num_subjects, alphas, trials):
    feature_names = models_config_dict[model]
    features = concat_features(feature_names, annotations_path)
    center = False
    before = False  
    ## If both false then future only

    
    if trials > 1:
        window = np.ones(trials) / trials  # Define the averaging window

        if center:  
            features = np.convolve(features.flatten(), window, mode='same').reshape(-1, 1)  # Centered averaging

        elif before:
            features = np.convolve(features.flatten(), window, mode='full')[:len(features)].reshape(-1, 1)  # Past-only averaging

        else:  
            features = np.convolve(features.flatten(), window, mode='full')[trials-1:len(features)+trials-1].reshape(-1, 1)  # Future-only averaging


    
    num_features = len(feature_names)
    X = normalize(features, axis=0).astype(np.float32)


    ##Hemodynamic lag correction delete in case of sherlock (corrected in the data)
    #hrf_shift = 4
    #X = np.roll(X, hrf_shift, axis=0)
    #X[:hrf_shift, :] = 0


    r_nifti_group = np.zeros([num_subjects, *original_data_shape])
    r_per_feature_nifti_group = np.zeros([num_subjects, num_features, *original_data_shape])
    
    for subj in range(1, num_subjects + 1):
        print(f'Processing subject: {subj}')
        save_dir = os.path.join(results_dir, f'sub{subj}/{model}/trial_{trials}/')
        os.makedirs(save_dir, exist_ok=True)
        
        fmri_path = os.path.join(data_path, f'sub{subj}/derivatives', f'sherlock_movie_s{subj}.nii')
        #fmri_path = os.path.join(data_path, f'sub21/derivatives', f'sub-21_task-citizenfour_bold_blur_no_censor_ica.nii.gz')
        mask = mask_path if mask_path else None

        data_clean, masked_indices, original_data_shape, img_affine = clean_image(fmri_path, subj, mask, results_dir)
        data_clean = data_clean.reshape(data_clean.shape[0], -1)
        data_clean = data_clean[:len(X)] ##making sure same dims
        X_train, X_test, y_train, y_test = train_test_split(X, data_clean.astype(np.float32), test_size=0.2, random_state=42)
        
        # Fit ridge regression
        logging.info('Fitting ridge regression')
        ridge_results = RidgeCV(alphas=alphas)
        ridge_results.fit(X_train, y_train)
        ridge_coef = ridge_results.coef_
        
        # Predict and calculate correlations
        logging.info('Predicting and calculating correlation per voxel')
        y_pred = ridge_results.predict(X_test)
        r = np.array([np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1] for i in range(y_test.shape[1])])
        
        # Compute feature-wise weights
        logging.info('Calculating feature-wise weights')
        r_per_feature = np.zeros((num_features, y_test.shape[1]))
        for i in range(num_features):
            feature_coef = ridge_coef.copy()
            feature_coef[:, np.arange(num_features) != i] = 0  # Zero out all other features
            y_pred = np.dot(X_test, feature_coef.T)
            r_per_feature[i, :] = [np.corrcoef(y_test[:, v], y_pred[:, v])[0, 1] for v in range(y_test.shape[1])]
        
        # Map results to original 3D space
        r_nifti = np.zeros(original_data_shape)
        r_per_feature_nifti = np.zeros([num_features, *original_data_shape])
        
        r_nifti[masked_indices[0], masked_indices[1], masked_indices[2]] = r
        r_per_feature_nifti[:, masked_indices[0], masked_indices[1], masked_indices[2]] = r_per_feature
        
        if trials <= 1:
            save_as_nii(model, subj, r_nifti, r_per_feature_nifti, save_dir, feature_names, img_affine)
        
        r_nifti_group[subj - 1] = r_nifti
        r_per_feature_nifti_group[subj - 1] = r_per_feature_nifti
        
        print(f'Subject {subj} done. Max r: {np.max(r_nifti[~np.isnan(r_nifti)])}')
    
    # Save group results
    group_dir = os.path.join(results_dir, f'group/{model}/trial_{trials}/')
    os.makedirs(group_dir, exist_ok=True)
    r_mean = np.mean(r_nifti_group, axis=0)
    weight_mean = np.mean(r_per_feature_nifti_group, axis=0)
    save_group_nii(model, r_mean, weight_mean, group_dir, feature_names, img_affine)
    print(f'Group results saved. Max r: {np.max(r_mean[~np.isnan(r_mean)])}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fmri_data_path', type=str, required=True)
    parser.add_argument('--annotations_path', type=str, required=True)
    parser.add_argument('--isc_mask_path', type=str, required=False)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='full',
                        help='Model options: full, social, social_plus_llava, llava_features, llava_only_social')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials for moving average')

    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([
        "--model", 'llava_1TR_video',
        '--fmri_data_path', r"/home/new_storage/sherlock/STS_sherlock/projects data/fmri_data",
        '--annotations_path', r'/home/new_storage/sherlock/STS_sherlock/projects data/annotations',
        '--results_dir', r'/home/new_storage/sherlock/STS_sherlock/projects data/results/llava_video_TRrange_onlysocial_FFA',
        "--isc_mask_path", r"/home/new_storage/sherlock/STS_sherlock/projects data/masks/ffa_mask.nii",
        "--trials", "1"
    ])
    
    start_time = time.time()
    print(f'Model type: {args.model}')
    
    alphas = np.logspace(1, 4, 10)
    original_data_shape = [61, 73, 61]
    #original_data_shape = [64, 76, 64]
    num_subjects = 17
    for trial in range(1, args.trials+1):
        main(args.fmri_data_path, args.annotations_path, args.isc_mask_path, args.model, args.results_dir, 
             original_data_shape, num_subjects, alphas, trial)
    
    for i in range (1,10):
        model = f'llava_{i}TR_video'
        main(args.fmri_data_path, args.annotations_path, args.isc_mask_path, model, args.results_dir, 
             original_data_shape, num_subjects, alphas, i)
    
    duration = round((time.time() - start_time) / 60)
    print(f'Duration: {duration} mins')