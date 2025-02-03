#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of dimensionality reduction methods (PCA, t-SNE, UMAP, PHATE, T-PHATE, CEBRA)
on the Sherlock dataset. For PCA we use default settings; for the other methods we
perform grid searches on one randomly selected subject and ROI, and then apply the
best hyperparameters to all subjects.
"""

import os
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import phate
import tphate

# Import CEBRA (ensure: pip install cebra)
import cebra

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from itertools import product

# Import your data preprocess function (for Sherlock)
from data_preprocess import sherlock_para

##############################
# Fix random seed
##############################
my_seed = 42
np.random.seed(my_seed)
random.seed(my_seed)

##############################
# Embedding Functions
##############################
def embed_pca(data, n_components=2):
    """Perform PCA embedding with default parameters."""
    pca = PCA(n_components=n_components, random_state=my_seed)
    return pca.fit_transform(data)

def embed_tsne(data, n_components=2, perplexity=30, early_exaggeration=12):
    """Perform t-SNE embedding with specified hyperparameters."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                early_exaggeration=early_exaggeration, random_state=my_seed)
    return tsne.fit_transform(data)

def embed_umap(data, n_components=2, n_neighbors=15, min_dist=0.1):
    """Perform UMAP embedding with specified hyperparameters."""
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, random_state=my_seed)
    return umap_model.fit_transform(data)

def embed_phate(data, n_components=2, n_landmark=2000, t="auto", knn=5, decay=40):
    """Perform PHATE embedding with specified hyperparameters."""
    return phate.PHATE(
        n_jobs=-1,
        n_landmark=n_landmark,
        verbose=0,
        t=t,
        knn=knn,
        decay=decay,
        n_components=n_components,
        random_state=my_seed
    ).fit_transform(data)

def embed_tphate(data, n_components=2, n_landmark=2000, t="auto", knn=5, decay=40):
    """Perform T-PHATE embedding with specified hyperparameters."""
    return tphate.TPHATE(
        n_jobs=-1,
        n_landmark=n_landmark,
        verbose=0,
        t=t,
        knn=knn,
        decay=decay,
        n_components=n_components,
        random_state=my_seed
    ).fit_transform(data)

def embed_cebra(data, n_components=2, cebra_params=None):
    """
    Perform CEBRA embedding. If cebra_params is None, default parameters are used.
    cebra_params should be a dictionary with keys including:
      'model_architecture', 'batch_size', 'learning_rate', 'temperature',
      'output_dimension', 'max_iterations', 'distance', 'conditional', 'device',
      'verbose', 'time_offsets'
    """
    if cebra_params is None:
        cebra_params = {
            'model_architecture': 'offset10-model',
            'batch_size': 512,
            'learning_rate': 3e-4,
            'temperature': 1.12,
            'output_dimension': n_components,
            'max_iterations': 2000,
            'distance': 'cosine',
            'conditional': 'time',
            'device': 'cuda_if_available',
            'verbose': True,
            'time_offsets': 10
        }
    else:
        # Ensure output_dimension is set to the desired n_components
        cebra_params['output_dimension'] = n_components

    model = cebra.CEBRA(**cebra_params)
    model.fit(data)
    return model.transform(data)

##############################
# Evaluation Function
##############################
def evaluate_knn(embedding, labels, k_values=[3, 8, 30, 50, 100]):
    """
    Evaluate the quality of the embedding using KNN cross-validation.
    Returns the average CV score over all specified k values.
    """
    scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, embedding, labels, cv=5)
        scores.append(np.mean(cv_scores))
    return np.mean(scores)

##############################
# Generic Grid Search Function
##############################
def grid_search(embed_func, data, labels, param_grid, fixed_params=None):
    """
    Perform a grid search over the provided parameter grid.
    embed_func: function to compute embedding, must accept data and hyperparameters as keyword arguments.
    param_grid: dictionary where keys are parameter names and values are lists of candidate values.
    fixed_params: dictionary of parameters that remain fixed.
    
    Returns: (best_params, best_score)
    """
    best_score = -np.inf
    best_params = {}
    if fixed_params is None:
        fixed_params = {}
    keys = list(param_grid.keys())
    for values in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))
        params.update(fixed_params)
        embedding = embed_func(data, **params)
        score = evaluate_knn(embedding, labels)
        # Debug print (optional): print(f"Trying params: {params} --> score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params.copy()
    return best_params, best_score

##############################
# Parameter Grids for Each Method
##############################
# t-SNE: perplexity in {5, 10, 20, 30, 40, 50}, early_exaggeration in {12, 18, 24, 32}
tsne_param_grid = {
    'perplexity': [5, 10, 20, 30, 40, 50],
    'early_exaggeration': [12, 18, 24, 32]
}

# UMAP: n_neighbors in {5, 12, 24, 48, 100, 200}, min_dist in {0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.99}
umap_param_grid = {
    'n_neighbors': [5, 12, 24, 48, 100, 200],
    'min_dist': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.99]
}

# PHATE & T-PHATE:
# n_landmark in {500, 1000, 2000, N_temporal_dim}, t in {1, 3, 5, "auto"}, knn in {5, 10}, decay in {20, 40}
def get_phate_param_grid(n_temporal_dim):
    return {
        'n_landmark': [500, 1000, 2000, n_temporal_dim],
        't': [1, 3, 5, "auto"],
        'knn': [5, 10],
        'decay': [20, 40]
    }

# CEBRA: batch_size in {512, 1024, 2048, 4096}, learning_rate in {1e-4, 3e-4},
# temperature in {0.5, 1, 1.5, 2, 3}, distance in {'cosine', 'euclidean'}
cebra_param_grid = {
    'batch_size': [512, 1024, 2048, 4096],
    'learning_rate': [1e-4, 3e-4],
    'temperature': [0.5, 1, 1.5, 2, 3],
    'distance': ['cosine', 'euclidean']
}

##############################
# Data Reading for Sherlock
##############################
def load_sherlock_data(subject, roi, balance=4):
    """
    Load the Sherlock dataset (stored as a .npy file) and preprocess it to obtain X_train.
    Modify the file paths and parameters as needed.
    """
    # Define directory names and file names for each ROI
    savemodel_names = ["HV_models", "EA_models", "EV_models", "PMC_models"]
    indata_dirs1 = ["HV", "EA", "EV", "PMC"]
    indata_dirs2 = ["high_Visual_sherlock_movie.npy", "aud_early_sherlock_movie.npy",
                    "early_visual_sherlock_movie.npy", "pmc_nn_sherlock_movie.npy"]
    # Select the corresponding directory and file name
    indata_dir1 = indata_dirs1[roi]
    indata_dir2 = indata_dirs2[roi]
    # Assume the data is stored under data/{ROI}/
    data_path = os.path.join('data', indata_dir1, f'sub-{subject:02d}_{indata_dir2}')
    data = np.load(data_path)
    X_train, batch_size, n = sherlock_para(data, balance)
    return X_train

def load_sherlock_labels(n):
    """
    Load the Sherlock labels (assumed to be stored in a CSV, with labels in the 10th column).
    Adjust the label length to match the number of samples.
    """
    sheet = pd.read_csv(os.path.join('data', 'sherlock_labels_coded_expanded.csv'), encoding='utf-8')
    cell_data = np.array(sheet)
    labels = cell_data[:, 9].astype(int)
    return labels[:n]

##############################
# Plotting and Saving Utility
##############################
def save_embedding_and_plot(embedding, labels, prefix, method_name):
    """
    Save the embedding (as a .npy file) and generate a plot (as a .png file).
    """
    # Define a custom color map using a list of colors (reversed)
    colors = ['darkorange', 'deepskyblue', 'gold', 'hotpink', 'lime', 'k', 'darkviolet',
              'peru', 'mediumblue', 'olive', 'midnightblue', 'palevioletred', 'c',
              'y', 'b', 'tan', 'navy', 'plum', 'slategray', 'lightseagreen', 'purple',
              'lightcoral', 'red', 'skyblue', 'moccasin', 'darkorchid', 'indigo',
              'palegreen', 'crimson', 'm', 'steelblue', 'darkgoldenrod', 'burlywood',
              'fuchsia', 'dodgerblue', 'greenyellow', 'khaki', 'lavender', 'azure']
    col = matplotlib.colors.ListedColormap(colors[::-1])
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 28})
    h1 = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=col, marker='o', s=18)
    plt.xlabel(f"{method_name} 1")
    plt.ylabel(f"{method_name} 2")
    plt.tight_layout()
    plt.colorbar(h1)    
    plt.savefig(prefix + f"{method_name}.png")
    np.save(prefix + f"{method_name}.npy", embedding)
    plt.close()

##############################
# Main Experiment Function for Sherlock
##############################
def run_sherlock_comparison():
    """
    Perform comparison experiments on the Sherlock dataset:
      1. Select a single subject and ROI for grid search to tune hyperparameters.
      2. Output the best hyperparameters and KNN evaluation scores for each method.
      3. Apply the best hyperparameters to all subjects and save the embeddings and plots.
    """
    # Define the list of methods to compare
    methods = ['PCA', 't-SNE', 'UMAP', 'PHATE', 'T-PHATE', 'CEBRA']
    
    # Select the subject and ROI for hyperparameter tuning (e.g., subject=1, roi=0)
    search_subject = 1
    search_roi = 0
    print(f"Performing grid search on subject {search_subject}, ROI {search_roi} ...")
    
    # Load data and labels (using all data for tuning)
    X_search = load_sherlock_data(search_subject, search_roi)
    n_samples = X_search.shape[0]
    labels_search = load_sherlock_labels(n_samples)
    
    best_params = {}  # Store best hyperparameters for each method
    best_scores = {}  # Store KNN evaluation scores for each method

    # PCA: no tuning, use default parameters
    best_params['PCA'] = {}
    emb_pca = embed_pca(X_search, n_components=2)
    best_scores['PCA'] = evaluate_knn(emb_pca, labels_search)
    print(f"PCA default score: {best_scores['PCA']:.4f}")

    # t-SNE grid search
    tsne_best, tsne_score = grid_search(embed_tsne, X_search, labels_search, tsne_param_grid)
    best_params['t-SNE'] = tsne_best
    best_scores['t-SNE'] = tsne_score
    print(f"t-SNE best params: {tsne_best}, score: {tsne_score:.4f}")

    # UMAP grid search
    umap_best, umap_score = grid_search(embed_umap, X_search, labels_search, umap_param_grid)
    best_params['UMAP'] = umap_best
    best_scores['UMAP'] = umap_score
    print(f"UMAP best params: {umap_best}, score: {umap_score:.4f}")

    # PHATE grid search (using sample size as N_temporal_dim)
    phate_grid = get_phate_param_grid(n_temporal_dim=X_search.shape[0])
    phate_best, phate_score = grid_search(embed_phate, X_search, labels_search, phate_grid)
    best_params['PHATE'] = phate_best
    best_scores['PHATE'] = phate_score
    print(f"PHATE best params: {phate_best}, score: {phate_score:.4f}")

    # T-PHATE grid search
    tphate_grid = get_phate_param_grid(n_temporal_dim=X_search.shape[0])
    tphate_best, tphate_score = grid_search(embed_tphate, X_search, labels_search, tphate_grid)
    best_params['T-PHATE'] = tphate_best
    best_scores['T-PHATE'] = tphate_score
    print(f"T-PHATE best params: {tphate_best}, score: {tphate_score:.4f}")

    # CEBRA grid search (fixing output_dimension to 2)
    cebra_best, cebra_score = grid_search(embed_cebra, X_search, labels_search,
                                           cebra_param_grid,
                                           fixed_params={'output_dimension': 2})
    best_params['CEBRA'] = cebra_best
    best_scores['CEBRA'] = cebra_score
    print(f"CEBRA best params: {cebra_best}, score: {cebra_score:.4f}")

    # Save the best hyperparameters to a CSV file (optional)
    param_df = pd.DataFrame({'Method': list(best_params.keys()),
                             'BestParams': list(best_params.values()),
                             'BestScore': list(best_scores.values())})
    os.makedirs('results/compare_methods', exist_ok=True)
    param_df.to_csv('results/compare_methods/best_params_sherlock.csv', index=False)

    # Apply the best hyperparameters to all subjects and ROIs and save the embeddings and plots
    subjects = range(1, 17)
    ROI_list = [0, 1, 2, 3]
    for subj in subjects:
        for roi in ROI_list:
            X = load_sherlock_data(subj, roi)
            n = X.shape[0]
            labels = load_sherlock_labels(n)
            prefix = f'results/compare_methods/sherlock_sub{subj:02d}_roi{roi:02d}_'
            # PCA
            emb = embed_pca(X, n_components=2)
            save_embedding_and_plot(emb, labels, prefix, 'PCA')
            # t-SNE
            emb = embed_tsne(X, n_components=2, **best_params['t-SNE'])
            save_embedding_and_plot(emb, labels, prefix, 't-SNE')
            # UMAP
            emb = embed_umap(X, n_components=2, **best_params['UMAP'])
            save_embedding_and_plot(emb, labels, prefix, 'UMAP')
            # PHATE
            emb = embed_phate(X, n_components=2, **best_params['PHATE'])
            save_embedding_and_plot(emb, labels, prefix, 'PHATE')
            # T-PHATE
            emb = embed_tphate(X, n_components=2, **best_params['T-PHATE'])
            save_embedding_and_plot(emb, labels, prefix, 'T-PHATE')
            # CEBRA
            emb = embed_cebra(X, n_components=2, cebra_params=best_params['CEBRA'])
            save_embedding_and_plot(emb, labels, prefix, 'CEBRA')
    print("All embeddings computed and saved.")

if __name__ == "__main__":
    run_sherlock_comparison()
