import os
import numpy as np
import cupy as cp
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import time
import argparse
import logging
import sys
from pathlib import Path
from nilearn.glm.first_level import glover_hrf
from utils import clean_image, save_group_nii, save_as_nii, compute_group_significant_map
from models_config import models_config_dict


def normalize_gpu(X):
    """Normalize features on GPU using CuPy"""
    X_gpu = cp.asarray(X)
    norms = cp.linalg.norm(X_gpu, axis=0)
    norms[norms == 0] = 1  # Avoid division by zero
    return X_gpu / norms


def corrcoef_gpu(x, y):
    """Compute correlation coefficient on GPU"""
    x = cp.asarray(x)
    y = cp.asarray(y)
    x = x - cp.mean(x)
    y = y - cp.mean(y)
    return cp.sum(x * y) / cp.sqrt(cp.sum(x**2) * cp.sum(y**2))


def corrcoef_gpu_vectorized(y_true, y_pred):
    """
    Vectorized correlation coefficient on GPU
    y_true: (n_samples, n_voxels) on GPU
    y_pred: (n_samples, n_voxels) on GPU
    Returns: (n_voxels,) array of correlations
    """
    # Center the data
    y_true_centered = y_true - cp.mean(y_true, axis=0, keepdims=True)
    y_pred_centered = y_pred - cp.mean(y_pred, axis=0, keepdims=True)

    # Compute correlations vectorized
    numerator = cp.sum(y_true_centered * y_pred_centered, axis=0)
    denominator = cp.sqrt(cp.sum(y_true_centered**2, axis=0) * cp.sum(y_pred_centered**2, axis=0))

    # Avoid division by zero
    denominator = cp.where(denominator == 0, 1, denominator)

    return numerator / denominator


def ridge_cv_gpu(X_train, y_train, alphas):
    """Manual Ridge CV with sklearn Ridge and GPU-accelerated scoring"""
    from sklearn.model_selection import KFold

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_alpha = None
    best_score = -np.inf

    print(f"Running {n_splits}-fold cross-validation for {len(alphas)} alpha values...")

    for alpha in alphas:
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            # Split data (on CPU for sklearn)
            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]

            # Fit Ridge on CPU using sklearn
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_tr, y_tr)

            # Score
            score = ridge.score(X_val, y_val)
            scores.append(float(score))

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    print(f"Best alpha: {best_alpha} with score: {best_score:.4f}")

    # Fit final model with best alpha
    ridge = Ridge(alpha=best_alpha, fit_intercept=True)
    ridge.fit(X_train, y_train)

    return ridge, best_alpha


def concat_features(features_list, single_features_dir):
    max_length = 1924  # Maximum length to truncate/pad features
    processed_annotations = [np.load(os.path.join(single_features_dir, f'{item}.npy'), allow_pickle=True) for item in features_list]
    for i in range(len(processed_annotations)):
        print(f"Truncating feature {features_list[i]} from length {processed_annotations[i].shape[0]} to {max_length}")
        if processed_annotations[i].shape[0] >= max_length:
            processed_annotations[i] = processed_annotations[i][:max_length]
    return np.concatenate(processed_annotations, axis=1)


def main(data_path, annotations_path, mask_path, model, results_dir, original_data_shape, num_subjects, alphas, trials):
    feature_names = models_config_dict[model]
    if len(feature_names) > 1:
        features = concat_features(feature_names, annotations_path)
    else:
        features = np.load(os.path.join(annotations_path, f'{feature_names[0]}.npy'), allow_pickle=True)
        features = features[:1924]

    # Shuffle features across timepoints in blocks of size `trials`
    if trials > 1:
        num_samples, num_features = features.shape
        block_size = trials

        num_full_blocks = num_samples // block_size
        truncated_len = num_full_blocks * block_size
        features_truncated = features[:truncated_len]

        blocks = features_truncated.reshape(num_full_blocks, block_size, num_features)

        np.random.seed(42)
        np.random.shuffle(blocks)

        shuffled_features = blocks.reshape(-1, num_features)

        if truncated_len < num_samples:
            leftovers = features[truncated_len:]
            features = np.vstack([shuffled_features, leftovers])
        else:
            features = shuffled_features

        assert features.shape == (num_samples, num_features), "Shape mismatch after shuffling"

    # Normalize on GPU
    print("Normalizing features on GPU...")
    X_gpu = normalize_gpu(features.astype(np.float32))
    X = cp.asnumpy(X_gpu)  # Keep a CPU copy for train_test_split

    num_features = X.shape[1]
    print(f'Initial X shape: {X.shape}')
    print(f'Final X shape after masking: {X.shape}')

    r_nifti_group = np.zeros([num_subjects, *original_data_shape])
    r_per_feature_nifti_group = np.zeros([num_subjects, num_features, *original_data_shape])
    all_subjects_weights = []

    for subj in range(1, num_subjects + 1):
        print(f'\n{"="*60}')
        print(f'Processing subject: {subj}')
        print(f'{"="*60}')

        save_dir = os.path.join(results_dir, model, f"trial_{trials}", f"subject{subj}")
        os.makedirs(save_dir, exist_ok=True)
        fmri_path = os.path.join(data_path, f'sub{subj}/derivatives', f'sherlock_movie_s{subj}.nii')

        mask = mask_path if mask_path else None

        data_clean, masked_indices, original_data_shape, img_affine = clean_image(fmri_path, subj, mask, results_dir)
        data_clean = data_clean.reshape(data_clean.shape[0], -1)

        # Remove first 26 TR
        data_clean = data_clean[26:]
        t1 = data_clean[:946]
        t2 = data_clean[946+26:]
        data_clean = np.vstack((t1, t2))
        data_clean = data_clean[:len(X)]

        print(f'X shape: {X.shape}, data_clean shape: {data_clean.shape}')

        # Train/test split (on CPU)
        X_train, X_test, y_train, y_test = train_test_split(
            X, data_clean.astype(np.float32), test_size=0.2, random_state=42
        )

        # Fit ridge regression with CV
        print('Fitting Ridge regression with cross-validation...')
        ridge_results, best_alpha = ridge_cv_gpu(X_train, y_train, alphas)

        # Get coefficients from sklearn model
        ridge_coef = ridge_results.coef_
        ridge_intercept = ridge_results.intercept_

        # Predict on GPU using manual calculation
        print('Predicting and calculating correlation per voxel on GPU...')
        X_test_gpu = cp.asarray(X_test)
        ridge_coef_gpu = cp.asarray(ridge_coef)
        ridge_intercept_gpu = cp.asarray(ridge_intercept)

        # Manual prediction: y = X @ coef.T + intercept
        y_pred_gpu = cp.dot(X_test_gpu, ridge_coef_gpu.T) + ridge_intercept_gpu
        y_pred = cp.asnumpy(y_pred_gpu)

        # Calculate correlations on GPU (vectorized for all voxels at once)
        y_test_gpu = cp.asarray(y_test)
        r_gpu = corrcoef_gpu_vectorized(y_test_gpu, y_pred_gpu)
        r = cp.asnumpy(r_gpu)

        # Individual weights matrix
        subject_weights = ridge_coef.T  # shape (features, voxels)
        print(f'Subject {subj} weights shape: {subject_weights.shape}')

        # Select top 1000 voxels by correlation
        top_k = 1000
        r_clean = np.nan_to_num(r, nan=-1)
        top_voxel_indices = np.argsort(r_clean)[-top_k:]

        subject_weights_top = subject_weights[:, top_voxel_indices]
        r_top = r[top_voxel_indices]

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

        all_subjects_weights.append(subject_weights_top)

        # Compute feature-wise weights on GPU (vectorized)
        print('Calculating feature-wise weights on GPU...')
        r_per_feature = np.zeros((num_features, y_test.shape[1]))
        ridge_coef_gpu = cp.asarray(ridge_coef)

        for i in tqdm(range(num_features), desc="Feature-wise correlations"):
            # Create coefficient matrix with only feature i
            feature_coef = cp.zeros_like(ridge_coef_gpu)
            feature_coef[:, i] = ridge_coef_gpu[:, i]

            # Predict using only this feature: (n_samples, n_voxels)
            y_pred_feature = cp.dot(X_test_gpu, feature_coef.T) + ridge_intercept_gpu

            # Compute correlations for all voxels at once (vectorized)
            r_feature_gpu = corrcoef_gpu_vectorized(y_test_gpu, y_pred_feature)
            r_per_feature[i, :] = cp.asnumpy(r_feature_gpu)

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

    # Group weights
    concat_weights = np.hstack(all_subjects_weights)
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
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use')


    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([
        "--model",  "social", 
        '--fmri_data_path', r"/home/new_storage/sherlock/STS_sherlock/projects data/fmri_data",
        '--annotations_path', r'/home/new_storage/sherlock/STS_sherlock/projects data/annotations',
        '--results_dir', r'/home/new_storage/sherlock/STS_sherlock/projects data/results/social_ffa',
        '--isc_mask_path', r"/home/new_storage/sherlock/STS_sherlock/projects data/masks/ffa_mask.nii",
        "--trials", "1"
    ])

    # Set GPU device
    cp.cuda.Device(args.gpu_id).use()
    print(f"Using GPU: {cp.cuda.Device(args.gpu_id).compute_capability}")

    start_time = time.time()
    print(f'Model type: {args.model}')

    alphas = np.logspace(1, 4, 10)
    original_data_shape = [61, 73, 61]
    num_subjects = 17
    means = []
    stds = []
    trials = range(1, 2)

    for trial in trials:
        main(args.fmri_data_path, args.annotations_path, args.isc_mask_path, args.model, args.results_dir,
             original_data_shape, num_subjects, alphas, trial)

    duration = round((time.time() - start_time) / 60)
    print(f'Duration: {duration} mins')
    print("GPU processing complete!")
