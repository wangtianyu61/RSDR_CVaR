1# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:23:13 2019

@author: wangtianyu6162
"""
#CVaR_parameter.py
#store some risk parameters
epsilon = 0.05
## epsilon is for fixing the parameters of problems regarding cvar
 
target_rate = 2.5
threshold = 0.5
## When we compare the robust form of fcvar_cluster with robustness optimization form
## threshold is a number we use in binary search.
## target_rate is the right-hand side of constraint 


risk_aversion = 1
## the gamma in mean_var model. 
## When gamma becomes infinity, the mean_var will tend to min-var model


tran_cost_p = 0.0
## we include the linear proportional transaction costs with MAD between two weights
## Applications: all target functions containing mean vector and return results with transaction costs   

rolling_day = 1
## for the whole rolling approach, the number of days we consider in each period

shortsale_sign = 0
## for optimization models whether we include the shortsale constraints

sharpe_ratio_open = False
# whether we take the risk-free rate into account
## true means we need to minus the risk-free rate from the dataset and false means not

cluster_type = "KMeans"
## choose which type to make the cluster in choosing the ambiguity set

## below two are parameters for bootstrap methods
resample_number = 1000
block_size = 5

#balance between mean and cvar  
trade_off = 0.0
#tradeoff list for between mean and cvar
tradeoff_list = [-0.2, -0.1, 0, 0.05, 0.1, 0.15, 0.2]
#fold number in cross validation
fold_number = 4
#theta list 
theta_param = [0.02, 0.04, 0.06, 0.08, 0.1]
##CSV header controller
head = "0315"

##unit control factor to place them for the sample mean excess returns over the risk-free assets
#-----------------------------#
#to make it more clearly, unit = 1 means the unit of portfolio dta is x%, unit = 100 means the unit of portfolio dta is x. 
unit = 1

#the minimum return constraint, which should also be tuned by the cross-validation
R_min = -5


#cv_type for cross validation
##type == -1 means no 
##type == 0 means gridsearch
##type == 1 means pre_cv_lp-based search
##type == 2 means using theta from '../factor model/fcvar_hmm_theta_industryindices.csv'
cv_type = -1

#approaches to select theta with time passing by
#theta_type = 0 represents fixed type
#theta_type = -1 represents abs_minimax
theta_type = -1


#
#test = list(range(10))
#select_test = test[0:5]
#se_test = list()
#for item in select_test:
#    print(item)
#    if item > -1:
#        se_test.append(item)
#        #select_test.remove(select_test[i])
#        
#        if len(se_test) == 5:
#            select_test += test[5:10]
#    print(select_test)

#adjust the parameter for the size of HMM-Wasserstein ambiguity set with time
ambiguity_param = 0.0

#minimum return
lower_percentile = 40