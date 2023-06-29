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

cv_type = 1
#fix_theta = 0.025


mean_sign = 'Mean-'
choice_index = 2



if mean_sign == "":
    mean_constr = False
else:
    mean_constr = 'worst-case'


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
Output_csv.return_info(naive, rfr_data)
#[method_list, return_list] = naive.finish_flag(method_list, return_list)

# (Mean-)CVaR portfolio
method_name = mean_sign + 'CVaR'
if mean_constr != '':
    meancvar = mean_CVaR(df_select, df_train, rolling_day, portfolio_number, method_name)
else:
    meancvar = SAA_CVaR(df_select, df_train, rolling_day, portfolio_number, method_name)
meancvar.rolling()
Output_csv.return_info(meancvar, rfr_data)

[method_list, return_list] = meancvar.finish_flag(method_list, return_list)


# DR portfolio
method_name = "DR " + mean_sign + "CVaR (Wasserstein)"
mean_fcvar_wasserstein = FCVaR_wasserstein2(df_select, df_train, rolling_day, portfolio_number, 1, method_name, cv_type, mean_constr)
mean_fcvar_wasserstein.rolling()
Output_csv.return_info(mean_fcvar_wasserstein, rfr_data)
[method_list, return_list] = mean_fcvar_wasserstein.finish_flag(method_list, return_list)



#---------------------------Our method-------------------------#
# Overall Param
hmm_type = -1




cls_num_choice = [2, 4, 4]
state_choice = ['BB', 'weathers', 'HMM']
method_choice = ['(Bull & Bear)', '(Weathers)', '(HMM)']


# Pretrain for Heterogeneous I
cluster_num = cls_num_choice[choice_index]
df_state_case = pd.read_csv('../factor model/' + state_choice[choice_index] + '_state.csv')
str_state = [str(each_state) for each_state in list(df_state_case['Date'])]
df_state_case['Date'] = str_state
mkt_state = df_state_case[((df_state_case['Date'])>=str(start_time))&(df_state_case['Date']<str(end_time))]['state']

# another benchmark (RS)
method_name = 'RS ' + mean_sign + 'CVaR ' + method_choice[choice_index]
cv_type = -1
mean_cvar_mkt = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, mkt_state, hmm_type, cv_type, mean_constr)
mean_cvar_mkt.theta = 0
mean_cvar_mkt.rolling()
Output_csv.return_info(mean_cvar_mkt, rfr_data)
[method_list, return_list] = mean_cvar_mkt.finish_flag(method_list, return_list)


cv_type = 1
method_name = 'RSDR ' + mean_sign + 'CVaR ' + method_choice[choice_index]
mean_fcvar_mkt = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, mkt_state, hmm_type, cv_type, mean_constr)
#mean_fcvar_mkt.theta = fix_theta
mean_fcvar_mkt.rolling()
#mean_fcvar_mkt.weight_pair_illustration(mean_sign + method_choice[cv_type])
Output_csv.return_info(mean_fcvar_mkt, rfr_data)
[method_list, return_list] = mean_fcvar_mkt.finish_flag(method_list, return_list)


[method_list, return_list] = naive.finish_flag(method_list, return_list)
#plt_return(method_list,return_list, 2007, 2020, str(portfolio_number) + data_type, suffix = mean_sign + method_choice[choice_index])
df = pd.DataFrame(np.array(return_list).T, columns = method_list)
df.to_csv('../return_cv/' + data_type + '_' + str(portfolio_number) + mean_sign + str(choice_index) + '2.csv', index = None)




