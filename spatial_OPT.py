import numpy as np
from bregman_spatial import sinkhorn
import ot

def tensor_square_loss_adjusted(C1, C2, T):
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return (a**2) / 2

    def f2(b):
        return (b**2) / 2

    def h1(a):
        return a

    def h2(b):
        return b

    tens = -np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()
    return tens


def tensor_KL_loss_adjusted(C1, C2, T):

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return (a*np.log(a+1e-15)-a)

    def f2(b):
        return b

    def h1(a):
        return a

    def h2(b):
        return (np.log(b+1e-15))

    tens = -np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()

    return tens

def create_space_distributions(num_locations, num_cells):

    p_locations = ot.unif(num_locations)
    p_expression = ot.unif(num_cells)
    return p_locations, p_expression

def compute_random_coupling(p, q, epsilon):
    num_cells = len(p)
    num_locations = len(q)
    K = np.random.rand(num_cells, num_locations)
    C = -epsilon * np.log(K)
    return sinkhorn(p, q, C, epsilon,method='sinkhorn')

def gromov_wasserstein_adjusted_norm(cost_mat, C1, C2,p, q, loss_fun, epsilon,
                                     max_iter=1000, tol=1e-9, verbose=False, log=False, random_ini=False):
   
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    cost_mat = np.asarray(cost_mat, dtype=np.float64)

    T = compute_random_coupling(p, q, epsilon) if random_ini else np.outer(p, q)  # Initialization

    cpt = 0
    err = 1
     
    while (err > tol and cpt < max_iter):
            
            Tprev = T

            if loss_fun == 'square_loss':
                tens = tensor_square_loss_adjusted(C1, C2, T)
            if loss_fun == 'kl_loss':
                    tens = tensor_KL_loss_adjusted(C1, C2, T)

            
            if epsilon ==0:
                T= ot.lp.emd(p, q, tens)
            else:
                T = sinkhorn(p, q, tens, epsilon,numItermax=max_iter)
        
            if cpt % 10 == 0:
            # We can speed up the process by checking for the error only all
            # the 10th iterations
                err = np.linalg.norm(T - Tprev)

                if log:
                    log['err'].append(err)

                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}'.format(
                                'It.', 'Err') + '\n' + '-' * 19)
                        print('{:5d}|{:8e}|'.format(cpt, err))
            cpt += 1
    if log:
        return T, log
    else:
        return T

