import os
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.stats import zscore
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import time
import argparse
import logging
import sys
from pathlib import Path
from nilearn.glm.first_level import glover_hrf
from utils import clean_image, save_group_nii, save_as_nii, compute_group_significant_map
from models_config import models_config_dict
import os
import torch
import numpy as np


def create_lagged_features(X, lags=[0, 1, 2, 3, 4]):
    """
    X: (n_samples, n_features)
    lags: list of lags in number of TRs
    """
    lagged_X = []
    for lag in lags:
        if lag == 0:
            lagged_X.append(X)
        else:
            lagged = np.roll(X, shift=lag, axis=0)
            lagged[:lag, :] = 0  # Zero padding the beginning
            lagged_X.append(lagged)
    return np.hstack(lagged_X)


def concat_features(features_list, single_features_dir):
    processed_annotations = [np.load(os.path.join(single_features_dir, f'{item}.npy'),allow_pickle=True) for item in features_list]
    return np.concatenate(processed_annotations, axis=1)

def main(data_path, annotations_path, mask_path , model, results_dir, original_data_shape, num_subjects, alphas, trials):
    feature_names = models_config_dict[model]

    features = concat_features(feature_names, annotations_path)



    
    # Shuffle features across timepoints in blocks of size `trials`
    if trials > 1:
        num_samples, num_features = features.shape
        block_size = trials

        # Truncate to a multiple of block_size
        num_full_blocks = num_samples // block_size
        truncated_len = num_full_blocks * block_size
        features_truncated = features[:truncated_len]

        # Reshape to (num_blocks, block_size, 40)
        blocks = features_truncated.reshape(num_full_blocks, block_size, num_features)

        # Shuffle block order (each block is a sequence of timepoints)
        np.random.seed(42)  # Optional: reproducibility
        np.random.shuffle(blocks)

        # Flatten back to (truncated_len, 40)
        shuffled_features = blocks.reshape(-1, num_features)

        # Optionally add leftover rows that didn't fit into full blocks
        if truncated_len < num_samples:
            leftovers = features[truncated_len:]
            features = np.vstack([shuffled_features, leftovers])
        else:
            features = shuffled_features

        # Sanity check
        assert features.shape == (num_samples, num_features), "Shape mismatch after shuffling"

    
    
    X = normalize(features, axis=0).astype(np.float32)
    face_indices = np.load("/home/new_storage/sherlock/STS_sherlock/projects data/annotations/face_mask.npy")
    face_mask = np.zeros(X.shape[0], dtype=bool)
    face_mask[face_indices] = True

    ### Talk with Idan about this
    ## To account for hrf
    lags = [0]  # Define lags in TRs
    X_lagged = create_lagged_features(X, lags=lags)
    num_features = X.shape[1]

    # Normalize again if needed3
    X = normalize(X_lagged, axis=0).astype(np.float32)
    #X = X[face_mask]


    r_nifti_group = np.zeros([num_subjects, *original_data_shape])
    r_per_feature_nifti_group = np.zeros([num_subjects, num_features, *original_data_shape])
   # weights_save_dir = os.path.join(results_dir,"weights")
    #os.makedirs(weights_save_dir, exist_ok=True)
    all_subject_weights = []
    for subj in range(1, num_subjects + 1):
        print(f'Processing subject: {subj}')
        save_dir = os.path.join(results_dir, model, f"trial_{trials}", f"subject{subj}")
        os.makedirs(save_dir, exist_ok=True)
        fmri_path = os.path.join(data_path, f'sub{subj}/derivatives', f'sherlock_movie_s{subj}.nii')
        
        mask = mask_path if mask_path else None

        data_clean, masked_indices, original_data_shape, img_affine = clean_image(fmri_path, subj, mask, results_dir)
        data_clean = data_clean.reshape(data_clean.shape[0], -1)
        data_clean = data_clean[26:]
        data_clean = data_clean[:len(X)]  # Ensure data_clean matches the length of X
       # data_clean = data_clean[face_mask]

        print(f'X shape: {X.shape}, data_clean shape: {data_clean.shape}')


        X_train, X_test, y_train, y_test = train_test_split(X, data_clean.astype(np.float32), test_size=0.2, random_state=42)
        
        # Fit ridge regression
        logging.info('Fitting ridge regression')
        ridge_results = RidgeCV(alphas=alphas)
        ridge_results.fit(X_train, y_train)
        ridge_coef = ridge_results.coef_
        
        ## Individual weights matrix
        #subject_weights = ridge_coef.T  # Shape: (num_features, num_voxels)
       # all_subject_weights.append(subject_weights)


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
        print("Final r_nifti max:", np.max(r_nifti))
        
        save_as_nii(model, subj, r_nifti, r_per_feature_nifti, save_dir, feature_names, img_affine)
        
        r_nifti_group[subj - 1] = r_nifti
        r_per_feature_nifti_group[subj - 1] = r_per_feature_nifti
        
        print(f'Subject {subj} done. Max r: {np.max(r_nifti[~np.isnan(r_nifti)])} avg top 50 r: {np.mean(np.sort(r[~np.isnan(r)])[-50:])}')
    
    # Save group results
    group_dir = os.path.join(results_dir, model, f"trial_{trials}", "group")
    rmap_paths = [
     os.path.join(results_dir, model, f"trial_{trials}", f"subject{subj}", f"{model}_r_sub{subj}.nii")
     for subj in range(1, num_subjects + 1)
     ]
    
    os.makedirs(group_dir, exist_ok=True)
    r_mean = np.mean(r_nifti_group, axis=0)
    weight_mean = np.mean(r_per_feature_nifti_group, axis=0)
    save_group_nii(model, r_mean, weight_mean, group_dir, feature_names, img_affine)
    print(f'Group results saved. Max r: {np.max(r_mean[~np.isnan(r_mean)])}')

    ## group weights
    #concat_weights = np.vstack(all_subject_weights)  # Shape: (num_subjects * num_features, num_voxels)
    #np.save(os.path.join(weights_save_dir, f"{model}_all_subjects_weights.npy"), concat_weights)

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
        "--model",  "cls_inside_pca1", 
        '--fmri_data_path', r"/home/new_storage/sherlock/STS_sherlock/projects data/fmri_data",
        '--annotations_path', r'/home/new_storage/sherlock/STS_sherlock/projects data/annotations',
        '--results_dir', r'/home/new_storage/sherlock/STS_sherlock/projects data/results/cls_inside_pca1',
       #'--isc_mask_path', r'/home/new_storage/sherlock/STS_sherlock/projects data/masks/sts_mask.nii',
        "--trials", "1"
    ])
    
    start_time = time.time()
    print(f'Model type: {args.model}')
    
    alphas = np.logspace(1, 4, 10)
    original_data_shape = [61, 73, 61]
    #original_data_shape = [64, 76, 64]
    num_subjects = 1
    means = []
    stds = []
    trials  = range(1,2) 
    for trial in trials:
        main(args.fmri_data_path, args.annotations_path, args.isc_mask_path ,args.model, args.results_dir, 
             original_data_shape, num_subjects, alphas, trial)

    #for i in range (1,21):
        #model = f'llava_{i}TR_onlysocial'
        #main(args.fmri_data_path, args.annotations_path, args.isc_mask_path, model, args.results_dir, 
        #     original_data_shape, num_subjects, alphas, i)

    duration = round((time.time() - start_time) / 60)
    print(f'Duration: {duration} mins')