
import numpy as np

import statsmodels.api as sm


#class TimeCORR(BaseEstimator):


def TimeCORR(X, smooth_window=None):
            
     n_samples, n_features = X.shape
     # calculate and store the AC functions for each feature separately
     A_feat = np.empty((n_samples,n_features))
     for f in range(n_features):
         A_feat[:,f] = sm.tsa.acf(X[:,f], fft=False, nlags=n_samples-1, missing='drop')
     A_feat = np.nanmean(A_feat, axis=1) # average over features to get one function
     acf = np.convolve(A_feat, np.ones(smooth_window), 'same') / smooth_window # rolling average
     dropoff = np.where(acf  < 0)[0][0] # timepoint where rolling average drops off
    # self.dropoff = dropoff
     # Spread out the autocorr function
     M = np.zeros((n_samples, n_samples))
     for i in range(n_samples):
         for j in range(n_samples):
             if 0 < abs(i-j) < dropoff:
                 M[i,j]=acf[abs(i-j)]
                 M[j,i]=acf[abs(i-j)]
     # row normalize to turn to probabilities
     for row in M:
         if np.sum(row) == 0: # this should never be true
             continue
         row[:] /= np.sum(row)
     temp = M[0, 1]
     M[0, 1] = M[1, 0]
     M[1, 0] = temp
     return M 

 