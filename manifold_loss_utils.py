# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:10:39 2020

@author: Summer
"""
import numpy as np
import sklearn
import numpy as Math



def Hbeta(D = np.array([]), beta = 1.0):
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p2(X, tol=1e-5, perplexity=30.0):
    n, d = X.shape
    D = np.square(sklearn.metrics.pairwise_distances(X, metric='euclidean'))
    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])
    P = np.zeros((n, n))
    logU = np.log(perplexity)
    
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        beta = 1.0
        Di = D[i]
        (H, thisP) = Hbeta(Di, beta)
        
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
      #  while  tries < 50:
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            (H, thisP) = Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1

        P[i, idx[i]] = thisP
    
    return P

def x2p1(X = np.array([]), tol = 1e-5, perplexity = 30.0):
    (n, d) = X.shape
    D = np.square(sklearn.metrics.pairwise_distances(X, metric='euclidean'))
    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])
    P = np.zeros((n, n))
    logU = np.log(perplexity)
    
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        beta = 1.0
        Di = D[i]
        (H, thisP) = Hbeta(Di, beta)
        
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
      #  while tries < 50:
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            (H, thisP) = Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1

        P[i, idx[i]] = thisP
    
    return P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    (n, d) = X.shape
    D = np.square(sklearn.metrics.pairwise_distances(X, metric='euclidean'))
    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])
    P = np.zeros((n, n))
    logU = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        beta = 1.0
        Di = D[i]
        (H, thisP) = Hbeta(Di, beta)

        Hdiff = H - logU
        tries = 0
       # while np.abs(Hdiff) > tol and tries < 50:
        while tries < 50:
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            (H, thisP) = Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1

        P[i, idx[i]] = thisP

    return P