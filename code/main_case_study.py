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
start_time = 20040101
end_time = 201905
train_test_split = 20061231

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

naive = naive_strategy(df_select, df_train, rolling_day, portfolio_number, "EW")
naive.rolling()

Output_csv.head()
Output_csv.return_info(naive, rfr_data)



#SAA-CVaR model
method_name = 'CVaR'
meancvar = mean_CVaR(df_select, df_train, rolling_day, portfolio_number, 'Mean-CVaR')
# meancvar = SAA_CVaR(df_select, df_train, rolling_day, portfolio_number, method_name)
meancvar.rolling()


[method_list, return_list] = meancvar.finish_flag(method_list, return_list)
Output_csv.return_info(meancvar, rfr_data)




mean_constr = 'worst-case'
cv_type = -1
# mean-FCVaR model (Wasserstein)
method_name = "DR Mean-CVaR (Wasserstein)"
mean_fcvar_wasserstein = FCVaR_wasserstein2(df_select, df_train, rolling_day, portfolio_number, 1, method_name, cv_type, mean_constr)
mean_fcvar_wasserstein.rolling()
Output_csv.return_info(mean_fcvar_wasserstein, rfr_data)
[method_list, return_list] = mean_fcvar_wasserstein.finish_flag(method_list, return_list)
#print(np.mean(mean_fcvar_wasserstein.return_array)/np.std(mean_fcvar_wasserstein.return_array))



#meancvar = mean_CVaR(df_select, df_train, rolling_day, portfolio_number, 'Stochastic')
#meancvar.rolling()


#---------------------------Our method-------------------------#
# Overall Param
hmm_type = -1

cv_type = 1

# Pretrain for Heterogeneous I
# cluster_num = 2
# df_state_case = pd.read_csv('../factor model/BB_state.csv')
# str_state = [str(each_state) for each_state in list(df_state_case['Date'])]
# df_state_case['Date'] = str_state
# mkt_state = df_state_case[((df_state_case['Date'])>=str(start_time))&(df_state_case['Date']<str(end_time))]['state']

# method_name = 'RSDR Mean-CVaR (Bull & Bear)'
# mean_fcvar_mkt = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, mkt_state, hmm_type, cv_type, mean_constr)
# mean_fcvar_mkt.rolling()
# Output_csv.return_info(mean_fcvar_mkt, rfr_data)
# [method_list, return_list] = mean_fcvar_mkt.finish_flag(method_list, return_list)



# # Pretrain for Heterogeneous II
# cluster_num = 4
# df_state = pd.read_csv("../factor model/weathers_state.csv")
# str_state1 = [str(each_state) for each_state in list(df_state['Date'])]
# df_state['Date'] = str_state1
# factor_state = df_state[((df_state['Date'])>=str(start_time))&(df_state['Date']<str(end_time))]['state']


# method_name = "RSDR Mean-CVaR (Weathers)"
# mean_fcvar_factor = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, factor_state, hmm_type, cv_type, mean_constr)
# mean_fcvar_factor.rolling()
# Output_csv.return_info(mean_fcvar_factor, rfr_data)
# [method_list, return_list] = mean_fcvar_factor.finish_flag(method_list, return_list)

cluster_num = 4
df_state = pd.read_csv("../factor model/HMM_state.csv")
str_state2 = [str(each_state) for each_state in list(df_state['Date'])]
df_state['Date'] = str_state2
naive_state = df_state[((df_state['Date'])>=str(start_time))&(df_state['Date']<str(end_time))]['state']

method_name = "RSDR Mean-CVaR (HMM)"
mean_fcvar_hmm = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, naive_state, hmm_type, cv_type, mean_constr)
mean_fcvar_hmm.rolling()
#print(np.mean(mean_fcvar_hmm.return_array)/np.std(mean_fcvar_hmm.return_array))
Output_csv.return_info(mean_fcvar_hmm, rfr_data)



method_name = "RS Mean-CVaR (HMM)"
mean_constr = 'worst-case'
fcvar_hmm_rs = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, naive_state, hmm_type, -1, mean_constr, ambiguity_param)
fcvar_hmm_rs.theta = 0
fcvar_hmm_rs.rolling()
[method_list, return_list] = fcvar_hmm_rs.finish_flag(method_list, return_list)

[method_list, return_list] = mean_fcvar_hmm.finish_flag(method_list, return_list)
Output_csv.return_info(fcvar_hmm_rs, rfr_data)



[method_list, return_list] = naive.finish_flag(method_list, return_list)
#visualization
plt_return(method_list,return_list, 2007, 2020, str(portfolio_number) + data_type, suffix = '-----(HMM)')
