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

# Mean-Var Portfolio
for rate in [0.5]:
    risk_aversion = rate
    method_name = "Mean-Variance (gammma = " + str(risk_aversion*2) + ')'
    markowitz = Markowitz(risk_aversion, df_select, df_train, rolling_day, portfolio_number, df_factor, method_name)
    #the fourth parameter means the robust level, 1 indicates we consider the robust case.
    markowitz.rolling(0, 0, 1, 0) 
    Output_csv.return_info(markowitz, rfr_data)



# SAA-CVaR model
# saa_CVaR = SAA_CVaR(df_select, df_train, rolling_day, portfolio_number, 'CVaR')
# saa_CVaR.rolling()  



# meancvar = mean_CVaR(df_select, df_train, rolling_day, portfolio_number, 'Stochastic')
# meancvar.rolling()
#meancvar.weight_pair_illustration()

#Output_csv.head()
#Output_csv.return_info(naive, rfr_data)

# mean_constr = False
# cv_type = 1
# #mean-FCVaR model (Wasserstein)
# method_name = "DR CVaR (Wasserstein)"
# mean_fcvar_wasserstein = FCVaR_wasserstein2(df_select, df_train, rolling_day, portfolio_number, 1, method_name, cv_type, mean_constr)
# mean_fcvar_wasserstein.rolling()
# Output_csv.return_info(mean_fcvar_wasserstein, rfr_data)
# #[method_list, return_list] = mean_fcvar_wasserstein.finish_flag(method_list, return_list)
# #mean_fcvar_wasserstein.weight_pair_illustration()
# #print(np.mean(mean_fcvar_wasserstein.return_array)/np.std(mean_fcvar_wasserstein.return_array))

# df_return = pd.DataFrame(pd.Series(list(mean_fcvar_wasserstein.return_array)))
# df_return.to_csv('../return_cv/' + data_type + '_' + str(portfolio_number) +'_benchmark.csv', index = None)


#meancvar = mean_CVaR(df_select, df_train, rolling_day, portfolio_number, 'Stochastic')
#meancvar.rolling()


# # Pretrain for Heterogeneous
# cluster_num = 4
# df_state = pd.read_csv("../factor model/weathers_state.csv")
# str_state = [str(each_state) for each_state in list(df_state['Date'])]
# df_state['Date'] = str_state
# factor_state = df_state[((df_state['Date'])>=str(start_time))&(df_state['Date']<str(end_time))]['state']

# df_state = pd.read_csv("../factor model/HMM_state.csv")
# str_state = [str(each_state) for each_state in list(df_state['Date'])]
# df_state['Date'] = str_state
# naive_state = df_state[((df_state['Date'])>=str(start_time))&(df_state['Date']<str(end_time))]['state']

# hmm_type = -1


# # Wasserstein
# # method_name = "RSDR CVaR (Weathers)"
# # mean_fcvar_factor = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, factor_state, hmm_type, cv_type, mean_constr)
# # #mean_fcvar_factor.method_name = method_name + ' 2-norm'
# # mean_fcvar_factor.rolling()
# #Output_csv.return_info(mean_fcvar_factor, rfr_data)
# #[method_list, return_list] = mean_fcvar_factor.finish_flag(method_list, return_list)

# # method_name = "RS CVaR (HMM)"
# # mean_fcvar_hmm = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, naive_state, hmm_type, cv_type, mean_constr)
# # mean_fcvar_hmm.rolling()
# # #mean_fcvar_hmm.weight_pair_illustration()
# # #print(np.mean(mean_fcvar_hmm.return_array)/np.std(mean_fcvar_hmm.return_array))
# # Output_csv.return_info(mean_fcvar_hmm, rfr_data)
# # [method_list, return_list] = mean_fcvar_hmm.finish_flag(method_list, return_list)

# cv_type = 1
# #for theta_type in [0.015, 0.03, 0.045]:
# method_name = "RSDR CVaR (HMM) with cv"
# mean_fcvar_hmm2 = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, naive_state, hmm_type, cv_type, mean_constr)
# mean_fcvar_hmm2.theta = 0.1
# mean_fcvar_hmm2.rolling()

# #mean_fcvar_hmm2.weight_pair_illustration()
# Output_csv.return_info(mean_fcvar_hmm2, rfr_data)
# [method_list, return_list] = mean_fcvar_hmm2.finish_flag(method_list, return_list)

# df_return = pd.DataFrame(pd.Series(list(mean_fcvar_hmm2.return_array)))
# df_return.to_csv('../return_cv/' + data_type + '_' + str(portfolio_number) +'.csv', index = None)
# plt_return(method_list,return_list, 1991, 2003, 
#            str(portfolio_number) + data_type, suffix = '_test')