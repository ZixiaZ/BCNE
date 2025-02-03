# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:54:28 2025

@author: Zixia Zhou
"""



import os
import numpy as np
import pandas as pd
import random

import matplotlib
matplotlib.use('Agg')
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr

# BrainIAK
from brainiak.eventseg.event import EventSegment

##############################
# Fix random seeds (optional)
##############################
my_seed = 42
np.random.seed(my_seed)
random.seed(my_seed)

##############################
# Behavior-based evaluation
##############################
def expand_boundary_labels_from_behavior(Y):
    """Directly use Y as event labels for each timepoint."""
    return Y

def WvB_evaluation_from_labels(timeseries_data, event_labels):
    """Compute within-event vs. between-event correlation."""
    tpts = timeseries_data.shape[0]
    corrmat = 1 - squareform(pdist(timeseries_data, 'correlation'))
    W, B = [], []
    for t0 in range(tpts):
        for t1 in range(t0 + 1, tpts):
            r = corrmat[t0, t1]
            if event_labels[t0] == event_labels[t1]:
                W.append(r)
            else:
                B.append(r)
    return np.nanmean(W), np.nanmean(B)

def test_behavioral_event_boundaries(timeseries_data, Y):
    """Behavior-based event boundary test."""
    event_labels = expand_boundary_labels_from_behavior(Y)
    avg_within, avg_between = WvB_evaluation_from_labels(timeseries_data, event_labels)
    avg_diff = np.nan_to_num(avg_within - avg_between)
    return {
        'avg_within_event_corr': avg_within,
        'avg_between_event_corr': avg_between,
        'avg_difference': avg_diff
    }

##############################
# HMM-based evaluation
##############################
def expand_boundary_labels(boundary_TRs, total_TRs):
    """
    Assign event labels to each timepoint based on identified boundaries.
    """
    labels = []
    event = 0
    for i in range(1, len(boundary_TRs)):
        length = boundary_TRs[i] - boundary_TRs[i - 1]
        labels.extend([event] * length)
        event += 1
    while len(labels) < total_TRs:
        labels.append(event)
    return labels

def WvB_evaluation(timeseries_data, boundary_TRs):
    """
    Compute within vs. between event correlation for HMM boundaries.
    """
    tpts = timeseries_data.shape[0]
    corrmat = 1 - squareform(pdist(timeseries_data, 'correlation'))
    event_labels = expand_boundary_labels(boundary_TRs, tpts)

    W, B = [], []
    for t0 in range(tpts):
        for t1 in range(t0 + 1, tpts):
            r = corrmat[t0, t1]
            if event_labels[t0] == event_labels[t1]:
                W.append(r)
            else:
                B.append(r)
    return np.nanmean(W), np.nanmean(B)

def compute_event_boundaries_diff_temporally_balanced(timeseries_data, boundary_TRs):
    """
    Compute within-vs-between correlation differences symmetrically,
    scanning outward from each timepoint.
    """
    from scipy.spatial.distance import pdist, squareform

    total_TRs = timeseries_data.shape[0]
    if boundary_TRs[0] != 0:
        boundary_TRs = [0] + boundary_TRs
    if boundary_TRs[-1] != total_TRs:
        boundary_TRs.append(total_TRs)

    timepoint_corrmat = 1 - squareform(pdist(timeseries_data, 'correlation'))
    boundary_labels = expand_boundary_labels(boundary_TRs, total_TRs)
    max_distance = np.max(np.diff(boundary_TRs))

    comparisons_made = []
    between_event_correlations, within_event_correlations, diffs = [], [], []
    for anchor in range(total_TRs):
        for distance in range(1, max_distance + 1):
            backward_tpt, forward_tpt = anchor - distance, anchor + distance
            if backward_tpt < 0 or forward_tpt >= total_TRs:
                continue
            if (backward_tpt, forward_tpt) in comparisons_made:
                continue
            comparisons_made.append((backward_tpt, forward_tpt))

            back_event = boundary_labels[backward_tpt]
            anchor_event = boundary_labels[anchor]
            forward_event = boundary_labels[forward_tpt]

            # Only proceed if exactly one side matches anchor
            within_anchor = [(back_event == anchor_event), (forward_event == anchor_event)]
            if sum(within_anchor) != 1:
                continue

            backward_corr = timepoint_corrmat[backward_tpt, anchor]
            forward_corr = timepoint_corrmat[forward_tpt, anchor]
            if back_event == anchor_event:
                between_event_correlations.append(forward_corr)
                within_event_correlations.append(backward_corr)
                diffs.append(backward_corr - forward_corr)
            else:
                between_event_correlations.append(backward_corr)
                within_event_correlations.append(forward_corr)
                diffs.append(forward_corr - backward_corr)

    return diffs, within_event_correlations, between_event_correlations, comparisons_made, boundary_labels



##############################
# CSV writing
##############################
def append_results_to_csv(results_list, csv_path):
    df = pd.DataFrame(results_list)
    write_header = not os.path.isfile(csv_path)
    df.to_csv(csv_path, index=False, mode='a', header=write_header)

##############################
# Main function
##############################
def run_sherlock_eventseg_evaluation():
    """
    Reads existing 2D embeddings from:
      - PCA, t-SNE, UMAP, PHATE, T-PHATE
      - CEBRA
      - BCNE (m1–m4)
    Then computes HMM (avg_within, avg_between, avg_diff, neg_loglikelihood, AIC)
    and behavior-based (avg_within, avg_between, avg_diff) metrics,
    storing results in real time to a CSV after each (seed, subject, roi).
    """

    # CSV where results are appended
    out_csv = "/home/zixia/BCNE/results/sherlock_eventseg_evaluation_all_final.csv"

    # Sherlock label file
    label_file = "/home/zixia/BCNE/data/sherlock_labels_coded_expanded.csv"
    sheet = pd.read_csv(label_file, encoding='utf-8')
    cell_data = np.array(sheet)
    cds = cell_data[:, 9].astype(int)  # Behavioral labels

    random_seeds = [0]
    subjects = range(1, 17)
    ROI_list = [0, 1, 2, 3]
    indata_dirs1 = ["HV", "EA", "EV", "PMC"]


    # Define comparison method names (PCA, t-SNE, UMAP, PHATE, T-PHATE)
    standard_methods = ["PCA", "t-SNE", "UMAP", "PHATE", "T-PHATE"]

    # Helper function to compute HMM & behavior metrics
    def evaluate_embedding(embedding, method_name, rseed, subj, roi):
        n_pts = embedding.shape[0]
        Y_trunc = cds[:n_pts]

        beh_res = test_behavioral_event_boundaries(embedding, Y_trunc)

        row = {
            'seed': rseed,
            'subject': subj,
            'roi': roi,
            'method': method_name,
            'beh_avg_within': beh_res['avg_within_event_corr'],
            'beh_avg_between': beh_res['avg_between_event_corr'],
            'beh_avg_diff': beh_res['avg_difference'],
        }
        return row

    for rseed in random_seeds:
        for subj in subjects:
            for roi in ROI_list:
                rows_to_save = []
                indata_dir1 = indata_dirs1[roi]

                # ----------------------------------------------------
                # 1) Load standard 2D methods: PCA, t-SNE, UMAP, PHATE, T-PHATE
                #    Suppose they are stored under:
                #    /home/zixia/BCNE/results/sherlock/sherlock_compare_2D/
                #    rseed{rseed}_sub{subj}_roi{roi}_{method}_2D_sub{subj}_roi{roi}_2D.npy
                #    If your actual paths differ, please update accordingly.
                # ----------------------------------------------------
                for method in standard_methods:
                    method_path = (f"/home/zixia/BCNE/results/sherlock/sherlock_compare_2D/"
                                      f"rseed{rseed}_sub{subj}_roi{roi}_{method}_2D_sub{subj}_roi{roi}_2D.npy")
                    if os.path.exists(method_path):
                        emb_com = np.load(method_path)
                        row_com = evaluate_embedding(emb_com, f"{method}_2D", rseed, subj, roi)
                        rows_to_save.append(row_com)


                # ----------------------------------------------------
                # 2) CEBRA 
                #    /home/zixia/BCNE/results/compare_CEBRA/sherlock_compare/
                #    rseed{rseed}_sub{subj}_roi{roi}_CEBRA_sub{subj:02d}_roi{roi:02d}.npy
                # ----------------------------------------------------
                cebra_path = (f"/home/zixia/BCNE/results/compare_CEBRA/sherlock_compare/"
                                 f"rseed{rseed}_sub{subj}_roi{roi}_CEBRA_2D_2D_sub{subj:02d}_roi{roi:02d}.npy")
                if os.path.exists(cebra_path):
                    cebra_emb = np.load(cebra_path)
                    row_cebra = evaluate_embedding(cebra_emb, "CEBRA", rseed, subj, roi)
                    rows_to_save.append(row_cebra)

                # ----------------------------------------------------
                # 3) BCNE  (m1–m4)
                #    Example:
                #    /home/zixia/BCNE/results/sherlock/{rseed:02d}/m1_{subj:02d}_ROI{indata_dir1}.npy
                #    Adjust indexing logic as needed.
                # ----------------------------------------------------
                # We'll try four model files:
                bcne_paths = [
                    (f"results/sherlock/{rseed:02d}/m1_{subj:02d}_ROI{indata_dir1}.npy", "BCNE_m1"),
                    (f"results/sherlock/{rseed:02d}/m2_{subj:02d}_ROI{indata_dir1}.npy", "BCNE_m2"),
                    (f"results/sherlock/{rseed:02d}/m3_{subj:02d}_ROI{indata_dir1}.npy", "BCNE_m3"),
                    (f"results/sherlock/{rseed:02d}/m4_{subj:02d}_ROI{indata_dir1}.npy", "BCNE_m4"),
                ]
                for path_bcne, method_name in bcne_paths:
                    if os.path.exists(path_bcne):
                        bcne_emb = np.load(path_bcne)
                        row_bcne = evaluate_embedding(bcne_emb, method_name, rseed, subj, roi)
                        rows_to_save.append(row_bcne)


                # ----------------------------------------------------
                # 4) Write out the rows for this (rseed, subj, roi)
                #    to CSV in real time
                # ----------------------------------------------------
                if rows_to_save:
                    append_results_to_csv(rows_to_save, out_csv)

    print("Done. Results appended in real time for each subject & ROI.")


if __name__ == "__main__":
    run_sherlock_eventseg_evaluation()
