import sys
sys.path.append('/home/new_storage/sherlock/STS_sherlock')
from voxelwise_encoding import utils
import os
import numpy as np
import time
import argparse
import logging
import sys
from pathlib import Path
import os
from scipy.spatial.distance import pdist,squareform
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from voxelwise_encoding.models_config import models_config_dict
import utils_rsa
from sklearn.decomposition import PCA



def main(data_path, annotations_path, mask_path , model, results_dir, original_data_shape, num_subjects):
    feature_names = models_config_dict[model]
    features_names = feature_names[0]
    features = np.load(os.path.join(annotations_path, f'{features_names}.npy'), allow_pickle=True)
    X = zscore(features, axis=0)
    rdm_embedding = utils_rsa.create_rdm(X)

    for subj in range(1, num_subjects + 1):
        print(f'Processing subject: {subj}')
        fmri_path = os.path.join(data_path, f'sub{subj}/derivatives', f'sherlock_movie_s{subj}.nii')
        
        mask = mask_path if mask_path else None
        print(mask)
        data_clean, masked_indices, original_data_shape, img_affine = utils.clean_image(fmri_path, subj, mask, results_dir)
        data_clean = data_clean.reshape(data_clean.shape[0], -1)
        data_clean = data_clean[26:]
        data_clean = data_clean[:len(X)]  # Ensure data_clean matches the length of X   
        print(data_clean.shape)
        print(f'X shape: {X.shape}, data_clean shape: {data_clean.shape}')
        data_clean = np.load("/home/new_storage/sherlock/STS_sherlock/projects data/annotations/cls_face_pca.npy",allow_pickle=True)
        path_subj = os.path.join(results_dir,f'sub{subj}' ,f'sub{subj}_rdm_fmri.npy')
        if not os.path.exists(path_subj):
            ## Create RDM for RSA
            rdm_fmri = utils_rsa.create_rdm(data_clean)
            ## save rdm fmri
            np.save(path_subj, rdm_fmri)
        else:
            rdm_fmri = np.load(path_subj)
        ## Correlate the RDMs
        r,p,b = utils_rsa.correlate_rdm(rdm_embedding,rdm_fmri)

        utils_rsa.save_fig_result(rdm_embedding, rdm_fmri, r, p,b, model, subj, results_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fmri_data_path', type=str, required=True)
    parser.add_argument('--annotations_path', type=str, required=True)
    parser.add_argument('--isc_mask_path', type=str, required=False)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='full',
                        help='Model options: full, social, social_plus_llava, llava_features, llava_only_social')

    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([
        "--model",  "cls_social", 
        '--fmri_data_path', r"/home/new_storage/sherlock/STS_sherlock/projects data/fmri_data",
        '--annotations_path', r'/home/new_storage/sherlock/STS_sherlock/projects data/annotations',
        '--results_dir', r'/home/new_storage/sherlock/STS_sherlock/projects data/results/RDM_cls_face_social',
       # '--isc_mask_path', r'/home/new_storage/sherlock/STS_sherlock/projects data/masks/sts_mask.nii',
    ])
    
    start_time = time.time()
    print(f'Model type: {args.model}')
    original_data_shape = [61, 73, 61]
    #original_data_shape = [64, 76, 64]
    num_subjects = 1
    means = []
    stds = []

    main(args.fmri_data_path, args.annotations_path, args.isc_mask_path ,args.model, args.results_dir, 
             original_data_shape, num_subjects)
    duration = round((time.time() - start_time) / 60)
    print(f'Duration: {duration} mins')