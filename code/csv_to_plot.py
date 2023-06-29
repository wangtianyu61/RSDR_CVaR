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

def DTA_select(data_type):
    if data_type[-1] == '2':
        return [19970101, 201904, 20061231]
    else:
        if data_type == 'FF' or 'IndustryPortfolios' or 'MKT':
            return [19630701, 200412, 19730631]
        elif data_type == 'S&P_sectors':
            return [19810101, 200212, 19901231]
        else:
            return [19700101, 200108, 19791231]
        
    
    return [start_time, end_time, train_test_split]

#possible_num = [3, 9, 11, 11, 21, 24]
#DataType = ['MKT', 'Country', 'S&P_sectors', 'IndustryPortfolios', 'FF', 'FF']
#start_time_list = [19630701, 19700101, 19810101, 19630701, 19630701, 19630701]
#train_test_split_list = [19730601, 19791231, 19901231, 19730631, 19730631, 19730631]
#end_time_list = [200412, 200108, 200301, 200412, 200412, 200412]
#for (i, j, s1, s2, s3) in zip(possible_num, DataType, start_time_list, train_test_split_list, end_time_list):
#for (i, j) in zip(possible_num, DataType):

portfolio_number = 3
#different kinds of datasets (6*/10/17/30/48)
freq = "Monthly"
#Daily/Weekly/Monthly

value = ""

#dta type
data_type = 'MKT2'
#select part of the data
[start_time, end_time, train_test_split] = DTA_select(data_type)



if data_type == 'IndustryIndices':
    classify_label = ['(Bull&Bear)', '(Weathers)', '(HMM)']
    for mean_sign in ['', 'Mean-']:
        for i, label_class in enumerate(classify_label):
            df = pd.read_csv('../return_cv/' + data_type + '_' + str(portfolio_number) + mean_sign + str(i) + '.csv')
            method_list = list(df.columns)
            for method_name in method_list:
                return_list.append(np.array(list(df[method_name])))
            plt_return(method_list, return_list, 2007, 2020, str(portfolio_number) + data_type, 
                       suffix = mean_sign + label_class)
            return_list = list()


else:
    hmm_type = -1
    mean_constr = False
    cluster_num = 4
    df_state_case = pd.read_csv('../factor model/HMM_state.csv')
    str_state = [str(each_state) for each_state in list(df_state_case['Date'])]
    df_state_case['Date'] = str_state
    mkt_state = df_state_case[((df_state_case['Date'])>=str(start_time))&(df_state_case['Date']<str(end_time))]['state']

    Input_csv = Input(portfolio_number, freq, value, start_time, end_time, train_test_split, data_type)
    [data_head, data_parameter, csv_name] = Input_csv.parameter_output()
    [df_select, df_train] = Input_csv.data_load()
    df_factor = Input_csv.three_factor_load()
    rfr_data = 0
    
    naive = naive_strategy(df_select, df_train, rolling_day, portfolio_number, "EW")
    naive.rolling()
    [method_list, return_list] = naive.finish_flag(method_list, return_list)

    method_name = 'RS CVaR (HMM)'
    cv_type = -1
    mean_cvar_mkt = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, mkt_state, hmm_type, cv_type, mean_constr)
    mean_cvar_mkt.theta = 0
    mean_cvar_mkt.rolling()
    #Output_csv.return_info(mean_cvar_mkt, rfr_data)
    [method_list, return_list] = mean_cvar_mkt.finish_flag(method_list, return_list)

    df_rsdr = pd.read_csv('../return_cv/' + data_type + '_' + str(portfolio_number) + '.csv')
    method_list.append('RSDR CVaR (HMM)')
    return_list.append(np.array(list(df_rsdr['0'])))

    #updated version
    if data_type.endswith('2'):
        plt_return(method_list, return_list, 2007, 2020, str(portfolio_number) + data_type)
    #original version
    else:
        if data_type == 'S&P_sectors' or data_type == 'Country':
            plt_return(method_list, return_list, int(train_test_split/10000) + 1, int(end_time/100) + 1,
                       str(portfolio_number) + data_type)
        else:
            plt_return(method_list, return_list, 1973, 2005, str(portfolio_number) + data_type)
    method_list = list()
    return_list = list()
        
        
