"""
Evaluation metrics and visualization for Brain Network Embedding results.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import load_model
from recursiveBCN_utils import create_kl_divergence
from data_preprocess import (sherlock_para, rat_data_preprocess,
                                monkey_data_preprocess, spatiotemporal_projection)

import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)

def test_knn_for_ks(embedding: np.ndarray, label: np.ndarray,
                    k_list: List[int] = [1, 3, 5, 8, 10, 30]) -> Dict[int, float]:
    """
    Perform 10-fold cross validation using KNN for a list of k values on the given embedding and labels.

    Parameters
    ----------
    embedding : np.ndarray
        The feature embedding.
    label : np.ndarray
        The target labels.
    k_list : List[int], optional
        List of k values to evaluate (default is [1, 3, 5, 8, 10, 30]).

    Returns
    -------
    Dict[int, float]
        A dictionary mapping each k value to the average cross-validation accuracy.
    """
    results = {}
    for k in k_list:
        knn_cv = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_cv, embedding, label, cv=10)
        results[k] = np.mean(scores)
    return results


def get_stage_label(direction_array: np.ndarray, threshold: int = 12) -> np.ndarray:
    """
    Generate stage labels based on the given direction array.

    The original logic computes trip_count // 12; here the divisor (12) is replaced by a configurable threshold.

    Parameters
    ----------
    direction_array : np.ndarray
        A 1D array representing directions (e.g., [0, 0, 0, 1, 1, 1, 0, 0, ...]).
    threshold : int, optional
        The divisor threshold for stage segmentation (default is 12).

    Returns
    -------
    np.ndarray
        An array of stage labels.
    """
    Y_time = np.zeros_like(direction_array)
    current_state = direction_array[0]
    trip_count = 0
    for i in range(1, len(direction_array)):
        if direction_array[i] != current_state:
            # When the direction changes
            if i == 1 or direction_array[i] != direction_array[i - 2]:
                trip_count += 1
            current_state = direction_array[i]
        Y_time[i] = trip_count // threshold
    return Y_time


def calculate_trustworthiness_continuity(dist_orig: np.ndarray, dist_embed: np.ndarray,
                                         n_neighbors: int = 10) -> Tuple[float, float]:
    """
    Calculate the trustworthiness and continuity metrics.

    Parameters
    ----------
    dist_orig : np.ndarray
        Pairwise distance matrix in the original space.
    dist_embed : np.ndarray
        Pairwise distance matrix in the embedding space.
    n_neighbors : int, optional
        Number of neighbors to consider (default is 10).

    Returns
    -------
    Tuple[float, float]
        Trustworthiness and continuity values.
    """
    n = dist_orig.shape[0]
    # Compute rank matrices
    rank_orig = np.argsort(np.argsort(dist_orig, axis=1), axis=1)
    rank_embed = np.argsort(np.argsort(dist_embed, axis=1), axis=1)

    trustworthiness = 0
    continuity = 0

    for i in range(n):
        # Neighbors in the original space
        orig_neighbors = set(np.where(rank_orig[i] < n_neighbors)[0])
        # Neighbors in the embedded space
        embed_neighbors = set(np.where(rank_embed[i] < n_neighbors)[0])

        # Compute trustworthiness loss
        for j in (embed_neighbors - orig_neighbors):
            trustworthiness += (rank_orig[i, j] - n_neighbors)

        # Compute continuity loss
        for j in (orig_neighbors - embed_neighbors):
            continuity += (rank_embed[i, j] - n_neighbors)

    norm = 2 * (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))
    trustworthiness = 1 - trustworthiness / norm
    continuity = 1 - continuity / norm

    return trustworthiness, continuity


def evaluate_embedding_quality(pos_ref: np.ndarray, pred: np.ndarray,
                               n_neighbors: int = 15) -> Dict[str, float]:
    """
    Evaluate the overall quality of the embedding using several metrics.

    Metrics include:
      - Pearson correlation (Representational Similarity Analysis, RSA) between pairwise distances.
      - Trustworthiness and continuity.

    Parameters
    ----------
    pos_ref : np.ndarray
        Reference positions.
    pred : np.ndarray
        Embedding coordinates.
    n_neighbors : int, optional
        Number of neighbors to use for trustworthiness and continuity (default is 15).

    Returns
    -------
    Dict[str, float]
        A dictionary containing the evaluation metrics.
    """
    # Compute pairwise distance matrices
    ref_euclidean = squareform(pdist(pos_ref, metric='euclidean'))
    pred_euclidean = squareform(pdist(pred, metric='euclidean'))

    # Pearson correlation coefficient
    pearson_corr, pearson_pval = pearsonr(ref_euclidean.flatten(), pred_euclidean.flatten())

    # Trustworthiness and continuity
    trust_eucl, cont_eucl = calculate_trustworthiness_continuity(ref_euclidean, pred_euclidean)

    return {
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_pval,
        'trustworthiness_euclidean': trust_eucl,
        'continuity_euclidean': cont_eucl,
    }


def evaluate_sherlock(subject_num: int, ROI_number: int, balance: int,
                      round_seed: int) -> None:
    """
    Evaluate Sherlock fMRI embeddings.

    This function preprocesses the data, applies a spatiotemporal projection,
    loads pre-trained models, obtains embeddings, and evaluates them using KNN classification.

    Parameters
    ----------
    subject_num : int
        The subject number.
    roi_number : int
        The region-of-interest (ROI) index.
    balance : int
        A parameter used during data preprocessing.
    round_seed : int
        The round seed used in naming model files.
    results_dir : Path
        Directory where model files and the results CSV are stored.
    """
    roi_names = ["HV", "EA", "EV", "PMC"]
    col_row_num = [23, 31, 17, 21]
    savemodel_names = ["HV_models", "EA_models", "EV_models", "PMC_models"]
    indata_dirs1 = ["HV", "EA", "EV", "PMC"]
    indata_dirs2 = ["high_Visual_sherlock_movie.npy", "aud_early_sherlock_movie.npy", 
                    "early_visual_sherlock_movie.npy", "pmc_nn_sherlock_movie.npy"]
    
    # Load and process data
    savemodel_name = savemodel_names[ROI_number]    
    indata_dir1 = indata_dirs1[ROI_number]
    indata_dir2 = indata_dirs2[ROI_number]


    # Load data
    outdata_dir = f'data/{indata_dir1}/sub-{subject_num:02d}_{indata_dir2}'
    data = np.load(outdata_dir)

    # Preprocess data
    X_train, batch_size, n = sherlock_para(data, balance)
    X_train_proj = spatiotemporal_projection(X_train, col_row_num[ROI_number],
                                             col_row_num[ROI_number])

    # Load labels from CSV (assumes labels are in the 10th column, index 9)
    sheet = pd.read_csv("data/sherlock_labels_coded_expanded.csv", encoding="utf-8")
    labels = np.array(sheet)[:, 9].astype(int)
    # Adjust label length to match the number of samples
    labels = labels[: X_train_proj.shape[0]]

    # Load models and compute embeddings
    kl_loss = create_kl_divergence(batch_size, 2)
    embeddings = []
    for i in range(1, 5):
        model_path = f'results/sherlock/{round_seed:02d}/{savemodel_name}/m{i}_{subject_num:02d}.h5'
        model = load_model(model_path, custom_objects={'KLdivergence': kl_loss})
        embeddings.append(model.predict(X_train_proj))

    # Evaluate each embedding using KNN via test_knn_for_ks (using k=7)
    results = []
    for i, emb in enumerate(embeddings, start=1):
        knn_scores = test_knn_for_ks(emb, labels, k_list=[7])
        results.append({
            "round_seed": round_seed,
            "subject_number": subject_num,
            "ROI": roi_names[ROI_number],
            "model": f"m{i}",
            "knn_cv_mean": knn_scores[7],
            "cv_scores": knn_scores,  # Dictionary of k: average accuracy
        })

    # Save or append results to CSV
    results_df = pd.DataFrame(results)   
    results_dir = Path('results/sherlock')
    csv_path = results_dir / "evaluation_metrics.csv"
    if not csv_path.exists():
        results_df.to_csv(csv_path, index=False)
    else:
        results_df.to_csv(csv_path, mode="a", header=False, index=False)


def evaluate_rat(rat_number: int, balance: int, round_seed: int) -> None:
    """
    Evaluate rat hippocampal embeddings.

    This function loads preprocessed rat data, applies a spatiotemporal projection,
    loads pre-trained models to obtain embeddings, generates stage labels using
    get_stage_label, and evaluates each embedding using KNN classification and
    embedding quality metrics.

    Parameters
    ----------
    rat_number : int
        The index (0-based) of the rat.
    balance : int
        A parameter used during data preprocessing.
    round_seed : int
        The round seed used in naming model files.
    """
    # Define rat names and projection sizes
    rat_names = ["achilles", "buddy", "cicero", "gatsby"]
    proj_sizes = [10, 6, 7, 8]

    # Load data and apply spatiotemporal projection
    data_path = f"data/hippo/spikes_{rat_names[rat_number]}.npy"
    pos_path = f"data/hippo/position_{rat_names[rat_number]}.npy"
    X_train, labels, n, batch_size = rat_data_preprocess(data_path, pos_path, rat_number, balance)
    X_train_proj = spatiotemporal_projection(X_train, proj_sizes[rat_number], proj_sizes[rat_number])

    # Process labels for evaluation:
    # Assume that labels[:, 0] holds the positional reference and labels[:, 1] holds directional information.
    Y_dir = labels[:n]
    # Generate stage labels using get_stage_label (default threshold = 12)
    Y_time = get_stage_label(Y_dir[:, 1], threshold=12)

    # Load models and compute embeddings
    kl_loss = create_kl_divergence(batch_size, 2)
    embeddings = []
    for model_num in range(1, 5):
        model_path = f'results/rat/{round_seed:02d}/m{model_num}_{rat_number:02d}.h5'
        model = load_model(model_path, custom_objects={'KLdivergence': kl_loss})
        pred = model.predict(X_train_proj)
        embeddings.append(pred)

    # Use the first column of Y_dir as reference positions (for correlation evaluation)
    pos_ref = Y_dir[:, 0:1]

    # Evaluate each embedding
    results = []
    for i, emb in enumerate(embeddings, start=1):
        # Evaluate classification accuracy using KNN via test_knn_for_ks (using k=8)
        knn_scores = test_knn_for_ks(emb, Y_time, k_list=[8])
        # Evaluate embedding quality (Pearson correlation, trustworthiness, continuity)
        quality_metrics = evaluate_embedding_quality(pos_ref, emb)

        results.append({
            "rat_number": rat_number,
            "balance": balance,
            "model": f"m{i}",
            "knn_cv_accuracy": knn_scores[8],
            "pearson_corr_pos": quality_metrics["pearson_correlation"],
            "pearson_pval_pos": quality_metrics["pearson_pvalue"],
            "trustworthiness": quality_metrics["trustworthiness_euclidean"],
            "continuity": quality_metrics["continuity_euclidean"],
        })

    # Save or append results to CSV
    results_df = pd.DataFrame(results)
    results_dir = Path('results/rat')
    csv_path = results_dir / "evaluation_metrics.csv"
    if not csv_path.exists():
        results_df.to_csv(csv_path, index=False)
    else:
        results_df.to_csv(csv_path, mode="a", header=False, index=False)


def _evaluate_position_correlation(pos_ref: np.ndarray,
                                   embedding: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the Pearson correlation between pairwise distances of the reference positions and the embedding.

    Parameters
    ----------
    pos_ref : np.ndarray
        Reference positions.
    embedding : np.ndarray
        Embedding coordinates.

    Returns
    -------
    Tuple[float, float]
        Pearson correlation coefficient and p-value.
    """
    ref_dist = squareform(pdist(pos_ref, metric="euclidean"))
    emb_dist = squareform(pdist(embedding, metric="euclidean"))
    return pearsonr(ref_dist.flatten(), emb_dist.flatten())


def _evaluate_monkey_position_correlation(pos_ref: np.ndarray,
                                          embedding: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate angle-specific and global position correlations for monkey data.

    Parameters
    ----------
    pos_ref : np.ndarray
        Reference positions.
    embedding : np.ndarray
        Embedding coordinates.

    Returns
    -------
    Tuple[np.ndarray, float]
        An array of Pearson correlation coefficients for each angle and the global correlation.
    """
    correlations = []
    for i in range(8):
        pos = pos_ref[i * 600:(i + 1) * 600]
        emb = embedding[i * 600:(i + 1) * 600]
        corr, _ = _evaluate_position_correlation(pos, emb)
        correlations.append(corr)

    global_corr, _ = _evaluate_position_correlation(pos_ref, embedding)
    return np.array(correlations), global_corr


def _process_position_data(pos_data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Process position data for monkey evaluation by averaging trials for each angle.

    Parameters
    ----------
    pos_data : np.ndarray
        Raw position data.
    labels : np.ndarray
        Array of labels indicating the angle for each trial.

    Returns
    -------
    np.ndarray
        Processed position data with one averaged vector per angle.
    """
    averaged_vectors = []
    for i in range(8):
        mask = labels == i
        selected_trials = np.where(mask)[0]
        trial_data = np.array([pos_data[trial * 600:(trial + 1) * 600]
                               for trial in selected_trials])
        avg_vector = np.mean(trial_data, axis=0)
        averaged_vectors.append(avg_vector)
    return np.vstack(averaged_vectors)


def evaluate_monkey(mode_number: int, balance: int, round_seed: int) -> None:
    """
    Evaluate monkey neural recording embeddings.

    This function preprocesses monkey data, loads pre-trained models,
    obtains embeddings, and evaluates them using KNN classification for angle
    classification as well as position correlation metrics.

    Parameters
    ----------
    mode_number : int
        The index indicating the mode ('active' or 'passive').
    balance : int
        A parameter used during data preprocessing.
    round_seed : int
        The round seed used in naming model files.
    """
    modes = ["active", "passive"]

    # Load data
    spike_path = f"data/monkey/spike_{modes[mode_number]}.npy"
    label_path = f"data/monkey/label_{modes[mode_number]}_ang.npy"
    pos_path = f"data/monkey/label_{modes[mode_number]}_pos.npy"

    X_train_proj, labels, n = monkey_data_preprocess(spike_path, label_path)
    batch_size = n // balance

    # Load models and compute embeddings
    kl_loss = create_kl_divergence(batch_size, 2)
    embeddings = []
        
    for model_num in range(1, 5):
        model_path = f'results/monkey/{round_seed:02d}_{balance:02d}/m{model_num}_{mode_number:02d}.h5'
        model = load_model(model_path, custom_objects={'KLdivergence': kl_loss})
        embeddings.append(model.predict(X_train_proj))      
        

    # Load position data and process it for evaluation
    pos_data = np.load(pos_path)
    pos_ref = _process_position_data(pos_data, labels)

    # Evaluate each embedding
    results = []
    for i, emb in enumerate(embeddings, start=1):
        # Angle classification using KNN via test_knn_for_ks (using k=8)
        Y_angles = np.repeat(np.arange(8), 600)
        knn_scores = test_knn_for_ks(emb, Y_angles, k_list=[8])
        # Evaluate position correlation
        corr_all, _ = _evaluate_monkey_position_correlation(pos_ref, emb)

        results.append({
            "mode": modes[mode_number],
            "round_seed": round_seed,
            "model": f"m{i}",
            "average_correlation": np.mean(corr_all),
            "angles_correlation": corr_all,           
            "angle_classification": knn_scores[8],
            "mean_angle_classification": knn_scores[8],
        })

    # Save or append results to CSV
    results_df = pd.DataFrame(results)
    results_dir = Path('results/monkey')
    csv_path = results_dir / "evaluation_metrics.csv"
    if not csv_path.exists():
        results_df.to_csv(csv_path, index=False)
    else:
        results_df.to_csv(csv_path, mode="a", header=False, index=False)
