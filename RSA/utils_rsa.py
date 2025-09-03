import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import pdist,squareform

def create_rdm(matrix, method='pearson'):
    """
    Create a Representational Dissimilarity Matrix (RDM) using 1 - Pearson correlation.
    
    Parameters:
    features (numpy.ndarray): A 2D array where each row is a timepoint and each column is a feature or a voxel.

    Returns:
    numpy.ndarray: A square RDM matrix of shape (n_timepoints, n_timepoints).
    """
    # Compute the pairwise Pearson correlation matrix
    rdm  =np.zeros((matrix.shape[0],matrix.shape[0]))
    if method == 'pearson':
        for i in tqdm(range(matrix.shape[0]),desc="Creating RDM"):
            for j in range(matrix.shape[0]):
                if i!=j:
                    corr,_ = pearsonr(matrix[i,:],matrix[j,:])
                    rdm[i,j] = 1-corr
                else:
                    rdm[i,j] = 0
    
    if method == 'euclidean':
        rdm = squareform(pdist(matrix, metric='euclidean'))
    return rdm


def correlate_rdm(mat1,mat2):
    """
    Compute the correlation between two RDMs.

    Parameters:
    mat1 (numpy.ndarray): The first RDM matrix.
    mat2 (numpy.ndarray): The second RDM matrix.

    Returns:
    float: The Pearson correlation coefficient between the two RDMs.
    """
    mask = np.triu(np.ones(mat1.shape), k=1).astype(bool)
    rdm1_upper = mat1[mask]
    rdm2_upper = mat2[mask]

    r,p = pearsonr(rdm1_upper, rdm2_upper)
    model = LinearRegression().fit(rdm1_upper.reshape(-1, 1), rdm2_upper)
    beta = model.coef_[0]
    return r,p,beta

def save_fig_result(mat1,mat2,r, p, beta, model, subj, results_dir):
    figx,axes = plt.subplots(1,3,figsize=(8,4))

    im1 = axes[0].imshow(mat1, cmap='viridis')
    axes[0].set_title('Feature RDM')
    axes[0].set_xlabel('Timepoints')
    axes[0].set_ylabel('Timepoints')

    im2 = axes[1].imshow(mat2, cmap='viridis')
    axes[1].set_title('fMRI RDM')
    axes[1].set_xlabel('Timepoints')
    axes[1].set_ylabel('Timepoints')

    axes[2].text(0.5, 0.5, f'RSA Correlation\nr = {r:.4f}\np = {p:.4f}\nbeta = {float(beta):.4f}',
               ha='center', va='center', fontsize=14, transform=axes[2].transAxes)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_title('RSA Result')
    axes[2].axis('off')
    plt.tight_layout()


    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'rsa_correlation_model_{model}_subj_{subj}.png'))
    plt.close()