# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 23:14:18 2024

@author: Zixia Zhou
"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, LeakyReLU
from manifold_loss_utils import x2p,x2p1,x2p2
from tensorflow.keras import backend as K
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


def create_model(
        input_shape,
        # Convolutional layer settings
        num_conv_layers=4,
        filters_list=None,
        kernel_size=3,
        alpha=0.05,  # LeakyReLU alpha
        # Dense layer settings
        dense_units=(1024, 512, 256,8),
        final_units=2
):
    """
    Create a CNN + Flatten + Dense architecture.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input, e.g. (rowNum, colNum, 1).

    num_conv_layers : int
        Number of convolutional layers. Must be >= 1.

    filters_list : list of int, optional
        Number of filters for each conv layer. If None, defaults to [3, 16, 32, 64] for 4 conv layers.
        If num_conv_layers differs from 4, make sure filters_list has matching length.

    kernel_size : int or tuple
        Kernel size for all Conv2D layers (assumed same for each layer here).

    alpha : float
        Negative slope coefficient for LeakyReLU activations.

    dense_units : tuple of int
        Units for Dense1, Dense2, Dense3. Must have at least three entries so we can name them "Dense1", "Dense2", "Dense3".

    final_units : int
        Number of output units, defaults to 2 (e.g., for a 2D embedding).

    Returns
    -------
    model : tf.keras.Model
        The constructed Keras model.
    """

    # Provide default filters_list if not supplied
    if filters_list is None:
        # For example, if num_conv_layers=4
        filters_list = [3, 16, 32, 64]

    # Sanity checks
    if num_conv_layers < 1:
        raise ValueError("num_conv_layers must be >= 1.")
    if len(filters_list) != num_conv_layers:
        raise ValueError("filters_list length must match num_conv_layers.")
    if len(dense_units) < 3:
        raise ValueError("dense_units must have at least three elements for Dense1, Dense2, Dense3.")

    # Input
    input_layer = Input(shape=input_shape, name='input_layer')

    # Convolutional layers
    x = input_layer
    for i in range(num_conv_layers):
        x = Conv2D(filters_list[i],
                   kernel_size=kernel_size,
                   padding='same',
                   name=f'Conv2D_{i + 1}')(x)
        x = LeakyReLU(alpha=alpha, name=f'LeakyReLU_{i + 1}')(x)

    # Flatten
    x = Flatten(name='flatten')(x)

    # At least three Dense layers named Dense1, Dense2, Dense3
    # If you want more, expand or adapt the logic here.
    # (We assume dense_units has at least 3 elements.)
    x = Dense(dense_units[0], activation='relu', name='Dense1')(x)
    x = Dense(dense_units[1], activation='relu', name='Dense2')(x)
    x = Dense(dense_units[2], activation='relu', name='Dense3')(x)

    # If you want to add more dense layers beyond the mandatory three, loop over the remainder:
    if len(dense_units) > 3:
        for i, units in enumerate(dense_units[3:], start=4):
            x = Dense(units, activation='relu', name=f'Dense{i}')(x)

    # Final output layer
    output = Dense(final_units, name='output')(x)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output, name='BCNE_Model')

    return model

def calculate_P(X, batch_size, HD_type):
    perplexity=30
    n = X.shape[0]
    P = np.zeros([n, batch_size])
    for i in range(0, n, batch_size):

        if HD_type == 'sherlock':
            P_batch = x2p(X[i:i + batch_size], perplexity)
        elif HD_type == 'hippo':
            P_batch = x2p1(X[i:i + batch_size], perplexity)
        elif HD_type == 'monkey':
            P_batch = x2p2(X[i:i + batch_size], perplexity)
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T
        P_batch = P_batch * 2  # Exaggerate
        P_batch = P_batch / P_batch.sum()
        P_batch = np.maximum(P_batch, 1e-12)
        P[i:i + batch_size] = P_batch

    return P

def evaluate_pos_monkey(pos_ref,pred):
    correlation_all=[]
    
    for i in range (0, 8):
        pos_ref1 =pos_ref[i*600:(i+1)*600]
        low_dim_data=pred[i*600:(i+1)*600]
        # pos_ref1 =pos_ref
        # low_dim_data=pred1
        ref_pos_distance_matrix = squareform(pdist(pos_ref1, metric='euclidean'))
    
        low_dim_distance_matrix = squareform(pdist(low_dim_data, metric='euclidean'))
        
        correlation, p_value = pearsonr(ref_pos_distance_matrix.flatten(), low_dim_distance_matrix.flatten())
    #    print(f"Position-Low's Pearson Correlation: {correlation}, P-value: {p_value}")
        correlation_all.append(correlation)
    correlation_all=np.array(correlation_all)

    ref_pos_distance_matrix = squareform(pdist(pos_ref, metric='euclidean'))
    
    low_dim_distance_matrix = squareform(pdist(pred, metric='euclidean'))
    
    correlation, p_value = pearsonr(ref_pos_distance_matrix.flatten(), low_dim_distance_matrix.flatten())
    
 #   print(f"Position-Low's Pearson Correlation: {correlation}, P-value: {p_value}")
    return correlation_all, correlation

def evaluate_pos_rat(pos_ref, pred):

    ref_pos_distance_matrix = squareform(pdist(pos_ref, metric='euclidean'))

    low_dim_distance_matrix = squareform(pdist(pred, metric='euclidean'))

    correlation, p_value = pearsonr(ref_pos_distance_matrix.flatten(), low_dim_distance_matrix.flatten())

    return correlation, p_value

from scipy.stats import spearmanr

def evaluate_pos_rat_spearman(pos_ref, pred):
    dist_pos = squareform(pdist(pos_ref, metric='euclidean'))
    dist_pred = squareform(pdist(pred, metric='euclidean'))

    rho, pval = spearmanr(dist_pos.flatten(), dist_pred.flatten())
    return rho, pval

import numpy as np
from scipy.spatial.distance import pdist, squareform

def distance_covariance(data1, data2):
    # data1, data2 are shape (n, d1) and (n, d2).
    # Implement distance correlation steps:
    A = squareform(pdist(data1, 'euclidean'))
    B = squareform(pdist(data2, 'euclidean'))

    A_mean = A.mean(axis=0, keepdims=True)
    B_mean = B.mean(axis=0, keepdims=True)
    A_centered = A - A_mean - A_mean.T + A.mean()
    B_centered = B - B_mean - B_mean.T + B.mean()

    dcov = np.sqrt(np.mean(A_centered * B_centered))
    return dcov

def distance_correlation(data1, data2):
    dcovXY = distance_covariance(data1, data2)
    dcovXX = distance_covariance(data1, data1)
    dcovYY = distance_covariance(data2, data2)
    if dcovXX < 1e-12 or dcovYY < 1e-12:
        return 0
    return dcovXY / np.sqrt(dcovXX * dcovYY)

def evaluate_pos_rat_distance_corr(pos_ref, pred):
    # pos_ref shape (n, 1) or (n, 2) if 2D position
    # pred shape (n, d), embedding
    return distance_correlation(pos_ref, pred)

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

def mantel_test(matrix1, matrix2, permutations=1000):
    # matrix1, matrix2 are NxN distance matrices (no need to flatten upfront).
    # 1) compute real correlation
    real_r, _ = pearsonr(matrix1.flatten(), matrix2.flatten())

    # 2) permutations
    n = matrix1.shape[0]
    perm_r = []
    for _ in range(permutations):
        idx = np.random.permutation(n)
        # shuffle rows + columns of matrix2
        mat2_perm = matrix2[idx][:, idx]
        r_perm, _ = pearsonr(matrix1.flatten(), mat2_perm.flatten())
        perm_r.append(r_perm)

    perm_r = np.array(perm_r)
    # p-value: fraction of permutations that exceed real correlation
    p_value = np.mean(perm_r >= real_r)

    return real_r, p_value, perm_r  # or just real_r, p_value

def evaluate_pos_rat_mantel(pos_ref, pred, permutations=1000):
    dist_pos = squareform(pdist(pos_ref, metric='euclidean'))
    dist_pred = squareform(pdist(pred, metric='euclidean'))
    real_r, p_value, distribution = mantel_test(dist_pos, dist_pred, permutations)
    return real_r, p_value




def create_kl_divergence(batch_size, low_dim):
    def KLdivergence(P, Y):
        alpha = low_dim - 1
        sum_Y = K.sum(K.square(Y), axis=1)
        eps = K.constant(1e-15)
        D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
        Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
        Q *= K.variable(1 - np.eye(batch_size))
        Q /= K.sum(Q)
        Q = K.maximum(Q, eps)
        C = K.log((P + eps) / (Q + eps))
        C = K.sum(P * C)
        return C
    return KLdivergence


def train_model_with_patient(model, X_train_auto, out_model_name, low_para_calculation_func, epochs, patience_threshold, n, batch_size,HD_type):
    loss_temp = 100
    patience = 0
    loss_record = []
    batch_num = int(n // batch_size)
    for epoch in range(epochs):
        if patience >= patience_threshold:
            print(f"Early stopping at epoch {epoch}")
            break
        if epoch == 0:
            low_para = low_para_calculation_func(model, X_train_auto, n, batch_size,HD_type)
        loss = 0
        for i in range(0, n, batch_size):
            low_para_temp1 = low_para[i // batch_size]
            loss += model.train_on_batch(X_train_auto[i:i + batch_size], low_para_temp1)
        if loss < loss_temp:
            loss_temp = loss
            patience = 0
            model.save(out_model_name)
        else:
            patience += 1
        loss_record.append(loss / batch_num)
        print(f"Epoch: {epoch + 1}/{epochs}, loss: {loss / batch_num}")
    
    return model


def train_model(model, X_train_auto, out_model_name, low_para_calculation_func, epochs, n, batch_size,HD_type):    
    batch_num = int(n // batch_size)
    for epoch in range(epochs):
       
        if epoch  == 0:
            low_para = low_para_calculation_func(model, X_train_auto, n, batch_size,HD_type)
        loss = 0
        for i in range(0, n, batch_size):
            low_para_temp1 = low_para[i // batch_size]
            loss += model.train_on_batch(X_train_auto[i:i + batch_size], low_para_temp1)      
        print(f"Epoch: {epoch + 1}/{epochs}, loss: {loss / batch_num}")
    model.save(out_model_name)
    return model

def calculate_low_para_for_input(model, X_train_auto, n, batch_size,HD_type):
    X1 = X_train_auto.reshape(-1, np.prod(X_train_auto.shape[1:]))
    low_para = [calculate_P(X1[i:i + batch_size], batch_size, HD_type) for i in range(0, n, batch_size)]
    return low_para

def calculate_low_para_for_layer(model, X_train_auto, layer_name, n, batch_size,HD_type):
    layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    layer_output = layer_model.predict(X_train_auto)
    low_para = [calculate_P(layer_output[i:i + batch_size], batch_size,HD_type) for i in range(0, n, batch_size)]
    return low_para