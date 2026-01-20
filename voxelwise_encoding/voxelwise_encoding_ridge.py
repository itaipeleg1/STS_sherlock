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



def concat_features(features_list, single_features_dir):


    max_length = 1924 # Maximum length to truncate/pad features
    processed_annotations = [np.load(os.path.join(single_features_dir, f'{item}.npy'),allow_pickle=True) for item in features_list]
    for i in range(len(processed_annotations)):
        print(f"Truncating feature {features_list[i]} from length {processed_annotations[i].shape[0]} to {max_length}")
        if processed_annotations[i].shape[0] >= max_length:   
            processed_annotations[i] = processed_annotations[i][:max_length]
    return np.concatenate(processed_annotations, axis=1)

def main(data_path, annotations_path, mask_path , model, results_dir, original_data_shape, num_subjects, alphas, trials):
    feature_names = models_config_dict[model]
    if len(feature_names) >1:
        features = concat_features(feature_names, annotations_path)
    else:
        features = np.load(os.path.join(annotations_path, f'{feature_names[0]}.npy'), allow_pickle=True)
        ## truncate to max length
        features = features[:1924]


    



    
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
    ## This is used to mask for face if needed later
    #face_mask[face_indices] = True

    num_features = X.shape[1]
    print(f'Initial X shape: {X.shape}')


    print(f'Final X shape after masking: {X.shape}')

    r_nifti_group = np.zeros([num_subjects, *original_data_shape])
    r_per_feature_nifti_group = np.zeros([num_subjects, num_features, *original_data_shape])
    all_subjects_weights = []
    for subj in range(1, num_subjects + 1):
        print(f'Processing subject: {subj}')
        save_dir = os.path.join(results_dir, model, f"trial_{trials}", f"subject{subj}")
        os.makedirs(save_dir, exist_ok=True)
        fmri_path = os.path.join(data_path, f'sub{subj}/derivatives', f'sherlock_movie_s{subj}.nii')
        
        mask = mask_path if mask_path else None

        data_clean, masked_indices, original_data_shape, img_affine = clean_image(fmri_path, subj, mask, results_dir)
        data_clean = data_clean.reshape(data_clean.shape[0], -1)
        ## Remove first 26 TR and
        data_clean = data_clean[26:]
        t1 = data_clean[:946]
        t2 = data_clean[946+26:]
        data_clean = np.vstack((t1, t2))
        data_clean = data_clean[:len(X)]  
         # Apply the same mask as features
        
        print(f'X shape: {X.shape}, data_clean shape: {data_clean.shape}')


        X_train, X_test, y_train, y_test = train_test_split(X, data_clean.astype(np.float32), test_size=0.2, random_state=42)
        
        # Fit ridge regression
        logging.info('Fitting ridge regression')
        ridge_results = RidgeCV(alphas=alphas)
        ridge_results.fit(X_train, y_train)
        ridge_coef = ridge_results.coef_
        # Predict and calculate correlations FIRST
        logging.info('Predicting and calculating correlation per voxel')
        y_pred = ridge_results.predict(X_test)
        r = np.array([np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1] for i in range(y_test.shape[1])])

        
        ## Individual weights matrix
        subject_weights = ridge_coef.T  # shape (features, voxels)
        print(f'Subject {subj} weights shape: {subject_weights.shape}')

        # Select top 1000 voxels by correlation
        top_k = 1000
        r_clean = np.nan_to_num(r, nan=-1)  # Handle NaNs
        top_voxel_indices = np.argsort(r_clean)[-top_k:]  # Indices of top 1000

        # Extract weights for top voxels only
        subject_weights_top = subject_weights[:, top_voxel_indices]  # shape (features, 1000)
        r_top = r[top_voxel_indices]  # shape (1000,)

        print(f'Subject {subj} top {top_k} voxels - weights shape: {subject_weights_top.shape}, r range: [{r_top.min():.3f}, {r_top.max():.3f}]')

        # Save top weights and indices
        np.save(os.path.join(save_dir, f"{model}_weights_top{top_k}_sub{subj}.npy"), subject_weights_top)
        np.save(os.path.join(save_dir, f"{model}_top{top_k}_indices_sub{subj}.npy"), top_voxel_indices)
        np.save(os.path.join(save_dir, f"{model}_top{top_k}_r_sub{subj}.npy"), r_top)
        top_voxel_brain_indices = (
        masked_indices[0][top_voxel_indices],
        masked_indices[1][top_voxel_indices],
        masked_indices[2][top_voxel_indices]
        )
        np.save(os.path.join(save_dir, f"{model}_top{top_k}_brain_indices_sub{subj}.npy"), top_voxel_brain_indices)

        all_subjects_weights.append(subject_weights_top)  # Now appending only top 1000
       

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
    concat_weights = np.hstack(all_subjects_weights) # shape: (num_features,num_voxels*num_subjects)
    print(f'All subjects weights shape: {concat_weights.shape}')
    np.save(os.path.join(results_dir, f"{model}_all_subjects_weights.npy"), concat_weights)

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
        "--model",  'social', 
        '--fmri_data_path', r"/home/new_storage/sherlock/STS_sherlock/projects data/fmri_data",
        '--annotations_path', r'/home/new_storage/sherlock/STS_sherlock/projects data/annotations',
        '--results_dir', r'/home/new_storage/sherlock/STS_sherlock/projects data/results/leyla_whole',
       # '--isc_mask_path', r"/home/new_storage/sherlock/STS_sherlock/projects data/masks/ppa_mask.nii",
        "--trials", "1"
    ])
    
    start_time = time.time()
    print(f'Model type: {args.model}')
    
    alphas = np.logspace(1, 4, 10)
    original_data_shape = [61, 73, 61]
    #original_data_shape = [64, 76, 64]
    num_subjects = 17
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