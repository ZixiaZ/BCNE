"""
Model testing functionality for Brain Network Embedding.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras.models import load_model
from recursiveBCN_utils import create_kl_divergence
from data_preprocess import (sherlock_para, rat_data_preprocess, 
                              monkey_data_preprocess, spatiotemporal_projection)
matplotlib.use('Agg')

tf.config.experimental_run_functions_eagerly(True)

def test_sherlock(subject_num, ROI_number, balance, round_seed):
    """
    Test Sherlock fMRI model.
    """
    # Model parameters and paths
    savemodel_names = ["HV_models", "EA_models", "EV_models", "PMC_models"]
    indata_dirs1 = ["HV", "EA", "EV", "PMC"]
    indata_dirs2 = ["high_Visual_sherlock_movie.npy", "aud_early_sherlock_movie.npy", 
                    "early_visual_sherlock_movie.npy", "pmc_nn_sherlock_movie.npy"]
    colrowNum = [23, 31, 17, 21]
    
    # Load and process data
    savemodel_name = savemodel_names[ROI_number]    
    indata_dir1 = indata_dirs1[ROI_number]
    indata_dir2 = indata_dirs2[ROI_number]
    colNum = colrowNum[ROI_number]
    rowNum = colrowNum[ROI_number]    
    
    outdata_dir = f'data/{indata_dir1}/sub-{subject_num:02d}_{indata_dir2}'
    data = np.load(outdata_dir)
   
    # Preprocess data
    X_train, batch_size, n = sherlock_para(data, balance)
    X_train_proj = spatiotemporal_projection(X_train, rowNum, colNum)

    # Load labels from CSV (assumes labels are in the 10th column, index 9)
    sheet = pd.read_csv("data/sherlock_labels_coded_expanded.csv", encoding="utf-8")
    labels = np.array(sheet)[:, 9].astype(int)
    # Adjust label length to match the number of samples
    labels = labels[: X_train_proj.shape[0]]
    
    # Test and visualize each model
    kl_divergence_loss = create_kl_divergence(batch_size, 2)
    colors = ['darkorange', 'deepskyblue', 'gold', 'hotpink', 'lime', 'k', 'darkviolet',
              'peru', 'mediumblue', 'olive', 'midnightblue', 'palevioletred', 'c',
              'y', 'b', 'tan', 'navy', 'plum', 'slategray', 'lightseagreen', 'purple',
              'lightcoral', 'red', 'skyblue', 'moccasin', 'darkorchid', 'indigo',
              'palegreen', 'crimson', 'm', 'steelblue', 'darkgoldenrod', 'burlywood',
              'fuchsia', 'dodgerblue', 'greenyellow', 'khaki', 'lavender', 'azure']
    col = matplotlib.colors.ListedColormap(colors[::-1])  

    # Process each model (m1-m4)
    for model_num in range(1, 5):
        model_path = f'results/sherlock/{round_seed:02d}/{savemodel_name}/m{model_num}_{subject_num:02d}.h5'
        model = load_model(model_path, custom_objects={'KLdivergence': kl_divergence_loss})
        pred = model.predict(X_train_proj)
        
        # Visualization
        plt.figure(figsize=(15, 10))
        plt.rcParams.update({'font.size': 28})
        h1=plt.scatter(pred[:, 0], pred[:, 1], c=labels, cmap=col, marker='o', s=18)
        plt.xlabel('BCNE1')
        plt.ylabel('BCNE2')
        plt.tight_layout()
        plt.colorbar(h1)
        
        # Save visualization
        out_name = f'results/sherlock/{round_seed:02d}/{savemodel_name}/m{model_num}_{subject_num:02d}_ROI{indata_dir1}.png'
        plt.savefig(out_name)
           
               
        

def test_rat(rat_number, balance, round_seed):
    """
    Test rat hippocampal model.
    """
    low_dim = 2
    rat_name = ['achilles', 'buddy', 'cicero', 'gatsby']
    proj_size = [10, 6, 7, 8]
    
    # Load data
    hippo_file_name1 = f'data/hippo/spikes_{rat_name[rat_number]}.npy'
    hippo_file_name2 = f'data/hippo/position_{rat_name[rat_number]}.npy'
    
    # Process data
    X_train, labels, n, batch_size = rat_data_preprocess(hippo_file_name1, hippo_file_name2,
                                                        rat_number, balance)
    X_train_proj = spatiotemporal_projection(X_train, proj_size[rat_number],
                                           proj_size[rat_number])
    
    # Load and test models
    kl_divergence_loss = create_kl_divergence(batch_size, low_dim)
    
    predictions = []
    for model_num in range(1, 5):
        model_path = f'results/rat/{round_seed:02d}/m{model_num}_{rat_number:02d}.h5'
        model = load_model(model_path, custom_objects={'KLdivergence': kl_divergence_loss})
        pred = model.predict(X_train_proj)
        predictions.append(pred)
    
    # Process labels for visualization
    Y_dire = labels[0:n]
    Y_time = np.zeros_like(Y_dire[:, 1])
    current_state = Y_dire[0, 1]
    trip_count = 0
    
    for i in range(1, len(Y_dire)):
        if Y_dire[i, 1] != current_state:
            if i == 1 or Y_dire[i, 1] != Y_dire[i-2, 1]:
                trip_count += 1
            current_state = Y_dire[i, 1]
        Y_time[i] = trip_count // 2
    
    # Create color schemes
    colors_left = [(1, 1, 0), (0, 1, 0)]
    colors_right = [(1, 0.75, 0.79), (0, 0, 1)]
    cmap_left = LinearSegmentedColormap.from_list("LeftWalk", colors_left, N=100)
    cmap_right = LinearSegmentedColormap.from_list("RightWalk", colors_right, N=100)
    
    # Create colors array
    colors = np.zeros((Y_dire.shape[0], 4))
    for i in range(Y_dire.shape[0]):
        if Y_dire[i, 2] == 1:
            colors[i] = cmap_left(Y_dire[i, 0] / 1.6)
        elif Y_dire[i, 1] == 1:
            colors[i] = cmap_right(Y_dire[i, 0] / 1.6)
    
    # Visualize each model's predictions
    for model_num, pred in enumerate(predictions, 1):
        # Color scheme 1
        plt.figure(figsize=(15, 10))
        plt.rcParams.update({'font.size': 28})
        plt.scatter(pred[:, 0], pred[:, 1], color=colors, marker='o', s=18)
        plt.xlabel('BCNE1')
        plt.ylabel('BCNE2')
        plt.tight_layout()
        saveplot_name = f'results/rat/m{model_num}c1_rat{rat_number:02d}_{round_seed:02d}.png'
        plt.savefig(saveplot_name)
        plt.close()
        
        # Color scheme 2
        plt.figure(figsize=(15, 10))
        plt.rcParams.update({'font.size': 28})
        plt.scatter(pred[:, 0], pred[:, 1], c=Y_time, cmap='magma', marker='o', s=18)
        plt.xlabel('BCNE1')
        plt.ylabel('BCNE2')
        plt.tight_layout()
        saveplot_name = f'results/rat/m{model_num}c2_rat{rat_number:02d}_{round_seed:02d}.png'
        plt.savefig(saveplot_name)
        plt.close()

def test_monkey(mode_number, balance, round_seed):
    """
    Test monkey neural recording model.
    """
    mode = ['active', 'passive']
    
    # Load and process data
    monkey_file_name1 = f'data/monkey/spike_{mode[mode_number]}.npy'
    monkey_file_name2 = f'data/monkey/label_{mode[mode_number]}_ang.npy'
    
    X_train_proj, labels, n = monkey_data_preprocess(monkey_file_name1, monkey_file_name2)
    batch_size = n // balance
    
    # Load and test models
    kl_divergence_loss = create_kl_divergence(batch_size, 2)
    
    # Setup visualization parameters
    sizes = np.ones_like(np.arange(4800)) * 10
    sizes[::600] = 80
    
    distinct_colors = [
        (0.05, 0.33, 0.7), (0.08, 0.6, 0.34), (0.64, 0.05, 0.06),
        (0.83, 0.68, 0.21), (0.47, 0.02, 0.61), (0.3, 0.3, 0.3),
        (0.0, 0.7, 0.7), (0.95, 0.35, 0.0)
    ]
    
    # Create color gradients
    colors = [np.linspace((0.8, 0.8, 0.8), color, 600) for color in distinct_colors]
    col = np.concatenate(colors)
    
    # Test each model
    for model_num in range(1, 5):
        model_path = f'results/monkey/{round_seed:02d}_{balance:02d}/m{model_num}_{mode_number:02d}.h5'
        model = load_model(model_path, custom_objects={'KLdivergence': kl_divergence_loss})
        pred = model.predict(X_train_proj)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.scatter(pred[:, 0], pred[:, 1], c=np.arange(4800),
                  cmap=mcolors.ListedColormap(col), s=sizes)
        plt.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, 4799),
                                     cmap=mcolors.ListedColormap(col)), ax=ax)
        plt.rcParams.update({'font.size': 28})
        plt.xlabel('BCNE1')
        plt.ylabel('BCNE2')
        plt.tight_layout()
        
        # Save results
        out_name_plot = f'results/monkey/{round_seed:02d}_{balance:02d}/m{model_num}_{mode_number:02d}.png'
        out_name_data = f'results/monkey/{round_seed:02d}_{balance:02d}/m{model_num}_{mode_number:02d}.npy'
        plt.savefig(out_name_plot)
        np.save(out_name_data, pred)
        plt.close()
