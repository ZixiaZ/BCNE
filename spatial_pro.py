

import numpy as np
import sklearn.metrics as mpd
from spatial_OPT import create_space_distributions
from spatial_OPT import gromov_wasserstein_adjusted_norm

def createMeshDistance(rowNum,colNum):
  
# If the row number is even
    if (rowNum % 2) == 0:
        Nx=rowNum/2
        x = np.linspace(-Nx, Nx-1, rowNum)
# If the row number is odd
    else:
        Nx=(rowNum-1)/2
        x = np.linspace(-Nx, Nx, rowNum)

# If the column number is even
    if (colNum % 2) == 0:
        Mx=colNum/2
        y = np.linspace(-Mx, Mx-1, colNum)
# If the column number is odd
    else:
       Mx=(colNum-1)/2
       y = np.linspace(-Mx, Mx, colNum)

# Create 2D mesh grid from 1D x and y grids
    xx, yy = np.meshgrid(x, y)
# Compute Euclidean distance between grid points
    zz = np.sqrt(xx**2 + yy**2)
# Make the 2D grid into a 1D vector and form the Euclidean distance matrix
    gridVec=zz.flatten()
    distMat=mpd.pairwise_distances(gridVec.reshape(-1,1))
    return distMat

def createInteractionMatrix(data, metric='correlation'):
    interactMat=mpd.pairwise_distances(data.T,metric=metric)
    return interactMat


def construct_neuromap(data,rowNum,colNum,epsilon=0,num_iter=1000):

    sizeData=data.shape
    numCell=sizeData[0]
    numGene=sizeData[1]
    # distance matrix of 2D genomap grid
    distMat = createMeshDistance(rowNum,colNum)
    # gene-gene interaction matrix 
    interactMat = createInteractionMatrix(data, metric='correlation')

    totalGridPoint=rowNum*colNum
    
    if (numGene<totalGridPoint):
        totalGridPointEff=numGene
    else:
        totalGridPointEff=totalGridPoint
    
    M = np.zeros((totalGridPointEff, totalGridPointEff))
    p, q = create_space_distributions(totalGridPointEff, totalGridPointEff)

   # Coupling matrix 
    T = gromov_wasserstein_adjusted_norm(
    M, interactMat, distMat[:totalGridPointEff,:totalGridPointEff], p, q, loss_fun='kl_loss', epsilon=epsilon,max_iter=num_iter)
 
    projMat = T*totalGridPoint
    # Data projected onto the couping matrix
    projM = np.matmul(data, projMat)

    neuromaps = np.zeros((numCell,rowNum, colNum, 1))

    px = np.asmatrix(projM)

    # Formation of neuromaps from the projected data
    for i in range(0, numCell-1):
        dx = px[i, :]
        fullVec = np.zeros((1,rowNum*colNum))
        fullVec[:dx.shape[0],:dx.shape[1]] = dx
        ex = np.reshape(fullVec, (rowNum, colNum), order='F').copy()
        neuromaps[i, :, :, 0] = ex
        
        
    return neuromaps



