# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:22:41 2019

@author: wangtianyu6162
"""

# main_head.py is meant for importing modules
#import sys
#sys.path.append('method/')
##comment on because of no use


# function for data_processing
#from data_process import *
#from data_process.data_input import data_input
#from data_process.data_stat import *
# functions for statistics and clustering
from data_process.input import *
#from data_process.cluster import *
#from data_process.choose_random import *

#different optimization problems
## parameters
from CVaR_parameter import *


# method
## benchmark

### whole portfolio benchmark
from method.naive_policy import *
from method.vw_strategy import *
from method.Markowitz import *
from method.Markowitz_revised import *
from method.SAA_CVaR import *
##another type of benchmarks used 
from method.CVaR_HMM import *

from method.MVP import *
from method.strategy import *
#Markowitz, Markowitz_revised, SAA_CVaR, FCVaR_no_cluster
### include naive policy, Markowitz, Markowitz_revised, SAA_CVaR, Popescu_no_cluster

## methods we suggest
from method.FCVaR_no_cluster import *
from method.FCVaR_cluster import *
from method.FCVaR_cluster_bs import *
from method.FCVaR_side_cluster import *
from method.FCVaR_approximate import *
from method.FCVaR_framework import *
from method.FCVaR_wasserstein import *
from method.FCVaR_wasserstein2 import *

from method.FCVaR_HMM_wasserstein import *
from method.mean_cvar import *
from method.mean_fcvar import *
from method.support import *
## Popescu_cluster
## Popescu_cluster_bs

#outputting the result
from data_process.test_result import *