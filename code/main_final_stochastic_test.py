import pandas as pd
import numpy as np
import math
from CVaR_parameter import *
from gurobipy import *
from main_head import *

import matplotlib.pyplot as plt
from method.strategy import *
from method.support import *

method_list = list()
return_list = list()

portfolio_number = 16
#different kinds of datasets (6*/10/17/30/48)
freq = "Monthly"
#Daily/Weekly/Monthly

value = ""

#dta type
data_type = "IndustryIndices"
#select part of the data
start_time = 20050101
end_time = 201001
train_test_split = 20071231
#    train_test_next = 200701
#------------------------------------------------#




#data input
Input_csv = Input(portfolio_number, freq, value, start_time, end_time, train_test_split, data_type)
[data_head, data_parameter, csv_name] = Input_csv.parameter_output()
[df_select, df_train] = Input_csv.data_load()

df_factor = Input_csv.three_factor_load()
rfr_data = 0
# output the head of all policies here
#Output_csv = output(csv_name, data_head, data_parameter)

#--------------------------models------------------#    


# Pretrain for Heterogeneous
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
mean_constr = False
cv_type = -1

dta_param = math.pow(36, 1/16)
k = 0.01

#Stochastic Comparison
##stationary
method_base = 'DR'
# meancvar = mean_CVaR(df_select, df_train, rolling_day, portfolio_number, method_base + '-stationary')
# meancvar.rolling()
# meancvar.weight_pair_illustration('stochastic-stationary')

#non-stationary
#Robust Comparison
#stationary
# method_base = 'DR'
# method_name = "DR (Wasserstein)"
# mean_fcvar_wasserstein = FCVaR_wasserstein2(df_select, df_train, rolling_day, portfolio_number, 1, method_name, cv_type, mean_constr)
# mean_fcvar_wasserstein.theta = k/dta_param
# mean_fcvar_wasserstein.rolling()
# mean_fcvar_wasserstein.weight_pair_illustration(method_base + '-stationary' + str(k))

# #non-stationary
# method_name = 'RSDR CVaR (HMM)'
# mean_fcvar_hmm = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, factor_state, hmm_type, cv_type, mean_constr)
# mean_fcvar_hmm.theta = k/dta_param
# mean_fcvar_hmm.rolling()
# mean_fcvar_hmm.weight_pair_illustration(method_base + '-nonstationary-HMM' + str(k))

method_name = 'RSDR CVaR (Weathers)'
mean_fcvar_hmm = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, naive_state, hmm_type, cv_type, mean_constr)
mean_fcvar_hmm.theta = k/dta_param
mean_fcvar_hmm.rolling()
mean_fcvar_hmm.weight_pair_illustration(method_base + '-nonstationary-Weathers' + str(k))

# cluster_num = 2
# df_state_case = pd.read_csv('../factor model/BB_state.csv')
# str_state = [str(each_state) for each_state in list(df_state_case['Date'])]
# df_state_case['Date'] = str_state
# mkt_state = df_state_case[((df_state_case['Date'])>=str(start_time))&(df_state_case['Date']<str(end_time))]['state']


# method_name = 'RSDR CVaR (Bull&Bear)'
# mean_fcvar_hmm = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, mkt_state, hmm_type, cv_type, mean_constr)
# mean_fcvar_hmm.theta = k/dta_param
# mean_fcvar_hmm.rolling()
# mean_fcvar_hmm.weight_pair_illustration(method_base + '-nonstationary_bull&bear' + str(k))

