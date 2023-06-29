#basic module
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from scipy import stats
from hmmlearn import hmm

import time

from gurobipy import *

from main_head import *

method_list = list()
return_list = list()

#possible_num = [3, 9, 11, 11, 21, 24]
#DataType = ['MKT', 'Country', 'S&P_sectors', 'IndustryPortfolios', 'FF', 'FF']
#start_time_list = [19630701, 19700101, 19810101, 19630701, 19630701, 19630701]
#train_test_split_list = [19730601, 19791231, 19901231, 19730631, 19730631, 19730631]
#end_time_list = [200412, 200108, 200301, 200412, 200412, 200412]
#for (i, j, s1, s2, s3) in zip(possible_num, DataType, start_time_list, train_test_split_list, end_time_list):
#for (i, j) in zip(possible_num, DataType):
#choose the dataset and the file path
portfolio_number = 16
#different kinds of datasets (6*/10/17/30/48)
freq = "Monthly"
#Daily/Weekly/Monthly

value = ""

#dta type
data_type = "IndustryIndices"
#select part of the data
start_time = 20040101
end_time = 201905
train_test_split = 20061231
#    train_test_next = 200701
#------------------------------------------------#


#data input
Input_csv = Input(portfolio_number, freq, value, start_time, end_time, train_test_split, data_type)
[data_head, data_parameter, csv_name] = Input_csv.parameter_output()
[df_select, df_train] = Input_csv.data_load()

df_factor = Input_csv.three_factor_load()
rfr_data = 0
# output the head of all policies here
Output_csv = output(csv_name, data_head, data_parameter)

#--------------------------benchmark------------------#    
# EW Portfolio
naive = naive_strategy(df_select, df_train, rolling_day, portfolio_number, "EW portfolio")
naive.rolling()
[method_list, return_list] = naive.finish_flag(method_list, return_list)

Output_csv.head()
Output_csv.return_info(naive, rfr_data)


## Min-Var Portfolio
#mvp = MVP(df_select, df_train, rolling_day, portfolio_number, df_factor, "Min-Variance")
#mvp.rolling(0, 0, 1, 0)
#Output_csv.return_info(mvp, rfr_data)
#
#
## Mean-Var Portfolio
##risk_aversion = 0.02
#method_name = "Mean-Variance"
#markowitz = Markowitz(risk_aversion, df_select, df_train, rolling_day, portfolio_number, df_factor, method_name)
##the fourth parameter means the robust level, 1 indicates we consider the robust case.
#markowitz.rolling(0, 0, 1, 0) 
#Output_csv.return_info(markowitz, rfr_data)
#
# Mean-CVaR model
meancvar = mean_CVaR(df_select, df_train, rolling_day, portfolio_number, 'Stochastic')
meancvar.rolling()
Output_csv.return_info(meancvar, rfr_data)
[method_list, return_list] = meancvar.finish_flag(method_list, return_list)


# mean-FCVaR model (Moment-based)
meanfcvar = mean_FCVaR(df_select, df_train, rolling_day, portfolio_number, 'DR (Moment)')
meanfcvar.rolling()
Output_csv.return_info(meanfcvar, rfr_data)
[method_list, return_list] = meanfcvar.finish_flag(method_list, return_list)

mean_constr = False
# mean-FCVaR model (Wasserstein)
method_name = "DR (Wasserstein)"
mean_fcvar_wasserstein = FCVaR_wasserstein2(df_select, df_train, rolling_day, portfolio_number, 1, method_name, cv_type, mean_constr)
mean_fcvar_wasserstein.rolling()
[method_list, return_list] = mean_fcvar_wasserstein.finish_flag(method_list, return_list)
Output_csv.return_info(mean_fcvar_wasserstein, rfr_data)
    

# Our Model
# Kang et al. (2020) setup
### Homogeneous Case: factor
#method_name = 'HoSDR Mean-CVaR (Moment, factor-based)'
#HSDR_factor = mean_FCVaR(df_select, df_train, rolling_day, portfolio_number, method_name, df_factor, False, 1, 4, mean_constr)
#HSDR_factor.rolling()
#Output_csv.return_info(HSDR_factor, rfr_data)
#
### Homogeneous Case: naive
#method_name = 'HoSDR Mean-CVaR (Moment, naive-based)'
#HSDR_naive = mean_FCVaR(df_select, df_train, rolling_day, portfolio_number, method_name, df_factor, False, 0, 4, mean_constr)
#HSDR_naive.rolling()
#Output_csv.return_info(HSDR_naive, rfr_data)

## #General Case
## ## Homogeneous Case: naive
## method_name = "HoSDR Mean-CVaR (Moment, naive-based)"
## HSDR_naive = FCluster_framework(df_select, df_train, rolling_day, portfolio_number, df_factor, 0, cluster_num, method_name)
## HSDR_naive.rolling(0)
## Output_csv.return_info(HSDR_naive, rfr_data)
#
## ## Homogeneous Case: factor
## method_name = "HoSDR Mean-CVaR (Moment, factor-based)"
## HSDR_factor = FCluster_framework(df_select, df_train, rolling_day, portfolio_number, df_factor, 1, cluster_num, method_name)
## HSDR_factor.rolling(0)
## Output_csv.return_info(HSDR_factor, rfr_data)
#
### Pretrain for Heterogeneous
cluster_num = 4
df_state = pd.read_csv("../factor model/weathers_state.csv")
str_state = [str(each_state) for each_state in list(df_state['Date'])]
df_state['Date'] = str_state
factor_state = df_state[((df_state['Date'])>=str(start_time))&(df_state['Date']<str(end_time))]['state']

df_state = pd.read_csv("../factor model/HMM_state.csv")
str_state = [str(each_state) for each_state in list(df_state['Date'])]
df_state['Date'] = str_state
naive_state = df_state[((df_state['Date'])>=str(start_time))&(df_state['Date']<str(end_time))]['state']

hmm_type = -1
## Moment
# Kang's setup
## Homogeneous Case: factor
#method_name = 'HeSDR Mean-CVaR (Moment, factor-based)'
#HESDR_factor = mean_FCVaR(df_select, df_train, rolling_day, portfolio_number, method_name, df_factor, factor_state, 1, 4, mean_constr)
#HESDR_factor.rolling()
#Output_csv.return_info(HESDR_factor, rfr_data)
#[method_list, return_list] = HESDR_factor.finish_flag(method_list, return_list)
#
### Homogeneous Case: naive
#method_name = 'HeSDR Mean-CVaR (Moment, naive-based)'
#HESDR_naive = mean_FCVaR(df_select, df_train, rolling_day, portfolio_number, method_name, df_factor, naive_state, 0, 4, mean_constr)
#HESDR_naive.rolling()
#Output_csv.return_info(HESDR_naive, rfr_data)
#[method_list, return_list] = HESDR_naive.finish_flag(method_list, return_list)


# General Setup
# method_name = "HeSDR Mean-CVaR (Moment, naive-based)"
# HESDR_naive = FCluster_framework(df_select, df_train, rolling_day, portfolio_number, df_factor, 0, cluster_num, method_name, naive_state)
# HESDR_naive.rolling(0)
# Output_csv.return_info(HESDR_naive, rfr_data)

# method_name = "HeSDR Mean-CVaR (Moment, factor-based)"
# HESDR_factor = FCluster_framework(df_select, df_train, rolling_day, portfolio_number, df_factor, 1, cluster_num, method_name, factor_state)
# HESDR_factor.rolling(0)
# Output_csv.return_info(HESDR_factor, rfr_data)

cv_type = 2
# Wasserstein
method_name = "HeSDR (Wasserstein, factor-based)"
mean_fcvar_factor = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, factor_state, hmm_type, cv_type, mean_constr)
mean_fcvar_factor.rolling()
Output_csv.return_info(mean_fcvar_factor, rfr_data)
[method_list, return_list] = mean_fcvar_factor.finish_flag(method_list, return_list)

method_name = "HeSDR (Wasserstein, naive-based)"
mean_fcvar_hmm = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, naive_state, hmm_type, cv_type, mean_constr)
mean_fcvar_hmm.rolling()
Output_csv.return_info(mean_fcvar_hmm, rfr_data)
[method_list, return_list] = mean_fcvar_hmm.finish_flag(method_list, return_list)


plt_return_tran(method_list,return_list)



