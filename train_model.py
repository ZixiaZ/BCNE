import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from manifold_loss_utils import x2p,x2p1
from tensorflow.keras import backend as K
from data_preprocess import rat_data_preprocess, monkey_data_preprocess,sherlock_para, spatiotemporal_projection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import os
from pathlib import Path
import tensorflow as tf
import random
from recursiveBCN_utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
my_seed = 0
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)
tf.config.experimental_run_functions_eagerly(True)





def main_sherlock(subject_num, n_components,ROI_number,recur,balance,train_mode,round_seed, model_design=None,model_num=None):
    
    """
    To search a best model, please use "train_model_with_patient", it will compare real-time model during training and save the best model
    "train_model_with_patient" need more training time because of the real-time comparison and real-time model saving.
    "train_model" use epoch number pre-achieved by "train_model_with_patient", here I apply a light-version "train_model" to test faster. 
    """
    if model_design is None:
        model_design = [5, [3, 16, 32, 64, 64], (1024, 512, 256,8), n_components] #3D
        # model_design = [4, [3, 16, 32, 64], (1024, 512, 256,8), n_components] #2D
    print(model_design[0])
    low_dim=n_components
    ## test subjece-04's HV ROI
    savemodel_names = ["HV_models", "EA_models", "EV_models", "PMC_models"]
    indata_dirs1 = ["HV", "EA", "EV", "PMC"]
    indata_dirs2 = ["high_Visual_sherlock_movie.npy", "aud_early_sherlock_movie.npy", "early_visual_sherlock_movie.npy", "pmc_nn_sherlock_movie.npy"]
    colrowNum = [23, 31, 17, 21]
    HD_type='sherlock'
    region_iter = ROI_number
    savemodel_name=savemodel_names[region_iter]    
    indata_dir1 = indata_dirs1[region_iter]
    indata_dir2 = indata_dirs2[region_iter]
    colNum = colrowNum[region_iter]
    rowNum = colrowNum[region_iter]

    # Load and preprocess data
    data_path = Path(f'data/{indata_dir1}')
    outdata_dir = data_path / f'sub-{subject_num:02d}_{indata_dir2}'
    data = np.load(outdata_dir)
   
    X_train, batch_size, n = sherlock_para(data,balance)
    X_train_proj = spatiotemporal_projection(X_train,rowNum,colNum)

    input_shape = (rowNum, colNum, 1)
   # model = create_model(input_shape)  # light-weieght version
   # model = create_model1(input_shape)

    model = create_model(
        input_shape=input_shape,
        num_conv_layers=model_design[0],
        filters_list=model_design[1],   # Must match num_conv_layers
        kernel_size=3,
        alpha=0.05,
        dense_units=model_design[2],  # Dense1, Dense2, Dense3
        final_units=model_design[3]
    )

    kl_divergence_loss = create_kl_divergence(batch_size, low_dim)
    model.compile(loss=kl_divergence_loss, optimizer=Adam(learning_rate=0.0005))
    
    
    # Create output directories
    os.makedirs(f'results/sherlock/{round_seed:02d}/{savemodel_name}', exist_ok=True)
    
    # Training paths
    out_paths = [
        f'results/sherlock/{round_seed:02d}/{savemodel_name}/m{i}_{subject_num:02d}.h5'
        for i in range(1, 5)
    ]    


    if train_mode==0:
        patient_num=20
        epochs =200
        ### train with patient strategy (save the best model, better result but slow) ### 
        ## First round of training (Recurion 1)
        model = train_model_with_patient(model, X_train_proj, out_paths[0], calculate_low_para_for_input, epochs, patient_num, n, batch_size,HD_type)
        
        if recur>1:
            ## Second round of training (Recurion 2)
            model = load_model(out_paths[0], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model_with_patient(model, X_train_proj, out_paths[1], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense1', n, batch_size,HD_type), epochs, patient_num, n, batch_size,HD_type)
        
        if recur>2:
            ## Third round of training (Recurion 3)
            model = load_model(out_paths[1], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model_with_patient(model, X_train_proj, out_paths[2], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense2', n, batch_size,HD_type), epochs, patient_num, n, batch_size,HD_type)
        
        if recur>3:
            ## Fourth round of training (Recurion 4)
            model = load_model(out_paths[2], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model_with_patient(model, X_train_proj, out_paths[3], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense3', n, batch_size,HD_type), epochs, patient_num, n, batch_size,HD_type)


    if train_mode==1:
        epochs=[150,150,100,50]
        #### train with predefined epoch number (faster)  

        ## First round of training (Recurion 1)
        model = train_model(model, X_train_proj, out_paths[0], calculate_low_para_for_input, epochs[0], n, batch_size,HD_type)
        
        if recur>1:
            ## Second round of training (Recurion 2)
            model = load_model(out_paths[0], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model(model, X_train_proj, out_paths[1], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense1', n, batch_size,HD_type), epochs[1], n, batch_size,HD_type)
        
        if recur>2:
            ## Third round of training (Recurion 3)
            model = load_model(out_paths[1], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model(model, X_train_proj, out_paths[2], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense2', n, batch_size,HD_type), epochs[2], n, batch_size,HD_type)
        
        if recur>3:
            ## Fourth round of training (Recurion 4)
            model = load_model(out_paths[2], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model(model, X_train_proj, out_paths[3], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense3', n, batch_size,HD_type), epochs[3], n, batch_size,HD_type)
        return model



def main_rat(rat_number,n_components, recur,balance,train_mode,round_seed,model_design=None):

    if model_design is None:
        model_design = [4, [3, 16, 32, 64], (1024, 512, 256,8), n_components]

    low_dim =n_components
    rat_name=['achilles', 'buddy', 'cicero', 'gatsby']
    # Load and preprocess data
    data_path = Path('data/hippo')
    hippo_file_name1 = data_path / f'spikes_{rat_name[rat_number]}.npy'
    hippo_file_name2 = data_path / f'position_{rat_name[rat_number]}.npy'
    
    X_train,labels,n,batch_size=rat_data_preprocess(hippo_file_name1,hippo_file_name2,rat_number,balance_degree=balance)
    HD_type='hippo'
    # proj_size = int(np.sqrt(n_HD))  where n, n_HD = data.shape, precalculated here
    proj_size=[10,6,7,8]
    colNum=proj_size[rat_number]
    rowNum=proj_size[rat_number]
    X_train_proj = spatiotemporal_projection(X_train,rowNum,colNum)
    print(batch_size)
    input_shape = (rowNum, colNum, 1)
    model = create_model(
        input_shape=input_shape,
        num_conv_layers=model_design[0],
        filters_list=model_design[1],  # Must match num_conv_layers
        kernel_size=3,
        alpha=0.05,
        dense_units=model_design[2],  # Dense1, Dense2, Dense3
        final_units=model_design[3]
    )
    kl_divergence_loss = create_kl_divergence(batch_size, low_dim)
    model.compile(loss=kl_divergence_loss, optimizer=Adam(learning_rate=0.0005))
    
    # Training paths
    out_paths = [
        f'results/rat/{round_seed:02d}/m{i}_{rat_number:02d}.h5'
        for i in range(1, 5)
    ]
        

    
    if train_mode==0:
        epochs = 400   
        patient_num=50
        
        ### train with patient strategy (save the best model, better result but slow) ### 
        ## First round of training (Recurion 1)
        model = train_model_with_patient(model, X_train_proj, out_paths[0], calculate_low_para_for_input, epochs, patient_num, n, batch_size,HD_type)
        
        if recur>1:
            ## Second round of training (Recurion 2)
            model = load_model(out_paths[0], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model_with_patient(model, X_train_proj, out_paths[1], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense1', n, batch_size,HD_type), epochs, patient_num, n, batch_size,HD_type)
        
        if recur>2:
            ## Third round of training (Recurion 3)
            model = load_model(out_paths[1], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model_with_patient(model, X_train_proj, out_paths[2], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense2', n, batch_size,HD_type), epochs, patient_num, n, batch_size,HD_type)
        
        if recur>3:
            ## Fourth round of training (Recurion 4)
            model = load_model(out_paths[2], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model_with_patient(model, X_train_proj, out_paths[3], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense3', n, batch_size,HD_type), epochs, patient_num, n, batch_size,HD_type)


    if train_mode==1:
        epochs=[300,200,150,100]
        #### train with predefined epoch number (faster)  

        ## First round of training (Recurion 1)
        model = train_model(model, X_train_proj, out_paths[0], calculate_low_para_for_input, epochs[0], n, batch_size,HD_type)
        
        if recur>1:
            ## Second round of training (Recurion 2)
            model = load_model(out_paths[0], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model(model, X_train_proj, out_paths[1], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense1', n, batch_size,HD_type), epochs[1], n, batch_size,HD_type)
        
        if recur>2:
            ## Third round of training (Recurion 3)
            model = load_model(out_paths[1], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model(model, X_train_proj, out_paths[2], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense2', n, batch_size,HD_type), epochs[2], n, batch_size,HD_type)
        
        if recur>3:
            ## Fourth round of training (Recurion 4)
            model = load_model(out_paths[2], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model(model, X_train_proj, out_paths[3], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense3', n, batch_size,HD_type), epochs[3], n, batch_size,HD_type)
            
            
    return model



def main_monkey(mode_number,n_components, recur,balance,train_mode,round_seed,model_design=None):

    if model_design is None:
        model_design = [4, [3, 16, 32, 64], (1024, 512, 256,8), n_components]
    
    low_dim =n_components
    mode=['active', 'passive']    
    HD_type='monkey'
    
    data_path = Path('data/monkey')
    monkey_file_name1 = data_path / f'spike_{mode[mode_number]}.npy'
    monkey_file_name2 = data_path / f'label_{mode[mode_number]}_ang.npy'

    
    X_train_proj, labels, n = monkey_data_preprocess(monkey_file_name1,monkey_file_name2)
    proj_size=8 # proj_size = int(np.sqrt(n_HD))  where n, n_HD = data.shape
    colNum=proj_size
    rowNum=proj_size
    batch_size=n//balance
    input_shape = (rowNum, colNum, 1)
    model = create_model(
        input_shape=input_shape,
        num_conv_layers=model_design[0],
        filters_list=model_design[1],  # Must match num_conv_layers
        kernel_size=3,
        alpha=0.05,
        dense_units=model_design[2],  # Dense1, Dense2, Dense3
        final_units=model_design[3]
    )
    kl_divergence_loss = create_kl_divergence(batch_size, low_dim)
    model.compile(loss=kl_divergence_loss, optimizer=Adam(learning_rate=0.0005))
    
    # Create output directories
    os.makedirs(f'results/monkey/{round_seed:02d}_{balance:02d}', exist_ok=True)
    
    # Training paths
    out_paths = [
        f'results/monkey/{round_seed:02d}_{balance:02d}/m{i}_{mode_number:02d}.h5'
        for i in range(1, 5)
    ]


    if train_mode==0:
        epochs = 200
        patient_num=20
        
        ### train with patient strategy (save the best model, better result but slow) ### 
        ## First round of training (Recurion 1)
        model = train_model_with_patient(model, X_train_proj, out_paths[0], calculate_low_para_for_input, epochs, patient_num, n, batch_size,HD_type)
        
        if recur>1:
            ## Second round of training (Recurion 2)
            model = load_model(out_paths[0], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model_with_patient(model, X_train_proj, out_paths[1], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense1', n, batch_size,HD_type), epochs, patient_num, n, batch_size,HD_type)
        
        if recur>2:
            ## Third round of training (Recurion 3)
            model = load_model(out_paths[1], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model_with_patient(model, X_train_proj, out_paths[2], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense2', n, batch_size,HD_type), epochs, patient_num, n, batch_size,HD_type)
        
        if recur>3:
            ## Fourth round of training (Recurion 4)
            model = load_model(out_paths[2], custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model_with_patient(model, X_train_proj, out_paths[3], lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense3', n, batch_size,HD_type), epochs, patient_num, n, batch_size,HD_type)


    if train_mode==1:
        epochs=[150,100,50,50]
        #### train with predefined epoch number (faster)  

        ## First round of training (Recurion 1)
        model = train_model(model, X_train_proj, out_model_name1, calculate_low_para_for_input, epochs[0], n, batch_size,HD_type)
        
        if recur>1:
            ## Second round of training (Recurion 2)
            model = load_model(out_model_name1, custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model(model, X_train_proj, out_model_name2, lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense1', n, batch_size,HD_type), epochs[1], n, batch_size,HD_type)
        
        if recur>2:
            ## Third round of training (Recurion 3)
            model = load_model(out_model_name2, custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model(model, X_train_proj, out_model_name3, lambda model, X, n, batch_size,HD_type: calculate_low_para_for_layer(model, X, 'Dense2', n, batch_size,HD_type), epochs[2], n, batch_size,HD_type)
        
        if recur>3:
            ## Fourth round of training (Recurion 4)
            model = load_model(out_model_name3, custom_objects={'KLdivergence': kl_divergence_loss})
            model = train_model(model, X_train_proj, out_model_name4, lambda model, X, n, batch_size: calculate_low_para_for_layer(model, X, 'Dense3', n, batch_size,HD_type), epochs[3], n, batch_size,HD_type)
    


    return model







