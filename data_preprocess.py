# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:20:10 2024

@author: Zixia Zhou
"""
import numpy as np
from temporal_pro import TimeCORR
from sklearn.feature_selection import VarianceThreshold
from spatial_pro import construct_neuromap

def sherlock_para(data,balance_degree):
    n_all, _ = data.shape
    batch_size = n_all // balance_degree
    n=batch_size*balance_degree
    X_train=data[0:n,:]
    return X_train, batch_size, n

def spatiotemporal_projection(X_train,rowNum,colNum):
    
    autocorr_map = TimeCORR (X=X_train, smooth_window=1)
    X_train_auto = np.matmul(autocorr_map, X_train)

    nump = rowNum * colNum
    if nump < X_train_auto.shape[1]:
        selector = VarianceThreshold()
        var_threshold = selector.fit(X_train_auto)
        top_n_indices = var_threshold.get_support(indices=True)
        X_train_auto = X_train_auto[:, top_n_indices[:nump]]
    NeuroMaps = construct_neuromap(X_train_auto, rowNum, colNum, epsilon=0.0, num_iter=200)
    return NeuroMaps    
    

def rat_data_preprocess(hippo_file_name1,hippo_file_name2,rat_number,balance_degree):
    if rat_number==0:
          data = np.load(hippo_file_name1)
          labels=np.load(hippo_file_name2)
          n_all, _ = data.shape
          batch_size = 2000 ## set 2000 as an upper limit for effecient calculation 
          n=batch_size*(n_all//batch_size)
          X_train=data[0:n,:]
          labels=labels[0:n,:]     
    else:
        if rat_number==2:
              data = np.load(hippo_file_name1)
              labels=np.load(hippo_file_name2)
              X_train=data[13747:33747,:]
              labels=labels[13747:33747,:]        
              n_all, _ = X_train.shape
              batch_size = n_all // balance_degree
              n=batch_size*balance_degree
              X_train=X_train[0:n,:]
              labels=labels[0:n,:] 
        else:
              data = np.load(hippo_file_name1)
              labels=np.load(hippo_file_name2)
              n_all, _ = data.shape
              batch_size = n_all // balance_degree
              n=batch_size*balance_degree
              X_train=data[0:n,:]     
              labels=labels[0:n,:] 
            
    return X_train,labels,n,batch_size

def monkey_data_preprocess_compare (monkey_file_name1,monkey_file_name2):
    data = np.load(monkey_file_name1)
    labels=np.load(monkey_file_name2)



    for i in range(8):
        # Find trails belonging to the current class
        mask = labels == i
        selected_trails = np.where(mask)[0]
        
        # Extract vectors for these trails and average them
        concatenated_selected_trails = np.vstack([data[trail*600: (trail+1)*600] for trail in selected_trails])
        concatenated_selected_trails = np.array([data[trail*600: (trail+1)*600] for trail in selected_trails])
        # concatenated_selected_trails = np.stack([data[trail*600: (trail+1)*600] for trail in selected_trails], axis=0)
    
       
        avg_vector = np.mean(concatenated_selected_trails, axis=0)
    
        if i==0:
            avg_vector_proj = avg_vector
        else:
            avg_vector_proj = np.concatenate((avg_vector, avg_vector_proj), axis=0)
    
#    data=avg_vector_proj   
     
    #batch_size=600  #each trail contains 600 time point, so we define batch_size as 600 here.   
    X_train=[]
    for i in range ((len(avg_vector_proj)//600)):       
        X_train1=avg_vector_proj[600*i:600*(i+1),:]
        X_train_min = np.min(X_train1)
        X_train_max = np.max(X_train1)
        X_train1=(X_train1 - X_train_min)/(X_train_max - X_train_min)  
        X_train.append(X_train1)    
    X_train_proj = np.concatenate(X_train)         
    n=X_train_proj.shape[0]

    return X_train_proj,labels, n



def monkey_data_preprocess (monkey_file_name1,monkey_file_name2):
    data = np.load(monkey_file_name1)
    labels=np.load(monkey_file_name2)
#    averaged_vectors = []

    # Process each class
    # for i in range(8):
    #     # Find trails belonging to the current class
    #     mask = labels == i
    #     selected_trails = np.where(mask)[0]
        
    #     # Extract vectors for these trails and average them
    #     concatenated_selected_trails = np.vstack([data[trail*600: (trail+1)*600] for trail in selected_trails])
    #     concatenated_selected_trails = np.array([data[trail*600: (trail+1)*600] for trail in selected_trails])
    #     # concatenated_selected_trails = np.stack([data[trail*600: (trail+1)*600] for trail in selected_trails], axis=0)
       
    #     avg_vector = np.mean(concatenated_selected_trails, axis=0)
    #     averaged_vectors.append(avg_vector)

    # # Concatenate all averaged vectors
    # data = np.vstack(averaged_vectors)

    proj_size=8 # proj_size = int(np.sqrt(n_HD))  where n, n_HD = data.shape
    colNum=proj_size
    rowNum=proj_size
    for i in range(8):
        # Find trails belonging to the current class
        mask = labels == i
        selected_trails = np.where(mask)[0]
        
        # Extract vectors for these trails and average them
        concatenated_selected_trails = np.vstack([data[trail*600: (trail+1)*600] for trail in selected_trails])
        concatenated_selected_trails = np.array([data[trail*600: (trail+1)*600] for trail in selected_trails])
        # concatenated_selected_trails = np.stack([data[trail*600: (trail+1)*600] for trail in selected_trails], axis=0)
    
       
        avg_vector = np.mean(concatenated_selected_trails, axis=0)
    
        avg_vector = spatiotemporal_projection(avg_vector,rowNum,colNum)
        if i==0:
            avg_vector_proj = avg_vector
        else:
            avg_vector_proj = np.concatenate((avg_vector, avg_vector_proj), axis=0)
    
#    data=avg_vector_proj   
     
    #batch_size=600  #each trail contains 600 time point, so we define batch_size as 600 here.   
    X_train=[]
    for i in range ((len(avg_vector_proj)//600)):       
        X_train1=avg_vector_proj[600*i:600*(i+1),:]
        X_train_min = np.min(X_train1)
        X_train_max = np.max(X_train1)
        X_train1=(X_train1 - X_train_min)/(X_train_max - X_train_min)  
        X_train.append(X_train1)    
    X_train_proj = np.concatenate(X_train)         
    n=X_train_proj.shape[0]
    return X_train_proj, labels, n


# def monkey_data_norm (data):    
#     n,_=data.shape 
#     batch_size=600  #each trail contains 600 time point, so we define batch_size as 600 here.   
#     X_train=[]
#     for i in range ((len(data)//600)):       
#         X_train1=data[600*i:600*(i+1),:]
#         X_train_min = np.min(X_train1)
#         X_train_max = np.max(X_train1)
#         X_train1=(X_train1 - X_train_min)/(X_train_max - X_train_min)  
#         X_train.append(X_train1)    
#     X_train = np.concatenate(X_train)    
        
#     return data, n, batch_size


def monkey_pos_average (filename3,labels):  
    data = np.load(filename3)
    proj_size=8 # proj_size = int(np.sqrt(n_HD))  where n, n_HD = data.shape
    colNum=proj_size
    rowNum=proj_size
    # Process each class
    averaged_vectors = []
    for i in range(8):
        # Find trails belonging to the current class
        mask = labels == i
        selected_trails = np.where(mask)[0]
        
        # Extract vectors for these trails and average them
        concatenated_selected_trails = np.vstack([data[trail*600: (trail+1)*600] for trail in selected_trails])
        concatenated_selected_trails = np.array([data[trail*600: (trail+1)*600] for trail in selected_trails])
        # concatenated_selected_trails = np.stack([data[trail*600: (trail+1)*600] for trail in selected_trails], axis=0)
    
       
        avg_vector = np.mean(concatenated_selected_trails, axis=0)
        averaged_vectors.append(avg_vector)

    pos_ref = np.vstack(averaged_vectors)
    return pos_ref
    




       
    

