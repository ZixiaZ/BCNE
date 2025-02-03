import numpy as np
import os
import random
import tensorflow as tf
from train_model import *
from test_model import *
#from test_pretrained_model import *
from evaluations import *

# GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental_run_functions_eagerly(True)

np.seterr(divide='ignore', invalid='ignore')

if __name__ == "__main__":
  

    ########################################################
    # Train, Test, Evaluate all experimental results with different random_seed
    ########################################################

    # ## Experiment 1
    for i in range (1,2):
       for j in range (0,1):
         main_sherlock(subject_num=i, n_components=2, ROI_number=j, recur=4, balance=4, train_mode=0,round_seed=0)
         test_sherlock(subject_num=i, ROI_number=j, balance=4,round_seed=0)
         evaluate_sherlock(subject_num=i, ROI_number=j, balance=4,round_seed=0)
    
    
    ## Experiment 2
    for i in range(0, 1):
          main_rat(rat_number=i, n_components=2, recur=4, balance=3, train_mode=0,round_seed=0)
          test_rat(rat_number=i, balance=3,round_seed=0)
          evaluate_rat(rat_number=i, balance=3,round_seed=0)
    
    
    ## Experiment 3

    main_monkey(mode_number=0, n_components=2, recur=4, balance=1, train_mode=0,round_seed=0)
    test_monkey(mode_number=0, balance=1,round_seed=0)
    evaluate_monkey(mode_number=0, balance=1,round_seed=0)
 
    main_monkey(mode_number=1, n_components=2, recur=4, balance=1, train_mode=0,round_seed=0)
    test_monkey(mode_number=1, balance=1,round_seed=0)
    evaluate_monkey(mode_number=1, balance=1,round_seed=0)
 