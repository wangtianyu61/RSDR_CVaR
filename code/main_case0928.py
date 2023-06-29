
from main_head import *
#Data Set Choice Generator
def DTA_select(data_type):
    if data_type == 'IndustryIndices':
        return [20040101, 201905, 20061231]
    if data_type[-1] == '2':
        return [19970101, 201904, 20061231]
    else:
        if data_type in ['FF', 'IndustryPortfolios', 'MKT']:
            return [19630701, 200412, 19730631]
        elif data_type == 'S&P_sectors':
            return [19810101, 200212, 19901231]
        else:
            return [19700101, 200108, 19791231]
        
    
    return [start_time, end_time, train_test_split]

freq = "Monthly"
#Here as we all use the Monthly Data Set, we can fix that parameter
value = ""
#eq/not eq csv in the . Here we all use the original dataset class.

#dta type
## In DeMiguel's paper, the dta type are selected to be one in ['MKT', 'Country', 'S&P_sectors', 'IndustryPortfolios', 'FF']
## When his dataset extended to 2019 Q1, the dta type are selected to be one in ['MKT2', 'Country2', 'S&P_sectors2', 'IndustryPortfolios2', 'FF22']
## Their corresponding portfolio number is 3 (MKT), 9 (Country), 11 (S&P_sectors), 11 (IndustryPortfolios), 21 /24 for FF.
## And in our case study, dta type are selected to be 'IndustryIndices' (16)
data_type = "IndustryIndices"

portfolio_number = 16

#select part of the data in the .csv file.
## from start_time to train_test_split as the default train dataset; from train_test_split (not include) as the test dataset.
[start_time, end_time, train_test_split] = DTA_select(data_type)

cv_type = 1
#data input
Input_csv = Input(portfolio_number, freq, value, start_time, end_time, train_test_split, data_type)
[data_head, data_parameter, csv_name] = Input_csv.parameter_output()
[df_select, df_train] = Input_csv.data_load()

df_factor = Input_csv.three_factor_load()
#we do not include the risk free rate item into the computation of sharpe ratio
if sharpe_ratio_open == False:
    rfr_data = 0
else:
    rfr_data = Input_csv.risk_free_rate()
    
cluster_num = 4
df_state = pd.read_csv("../factor model/HMM_state.csv")
str_state = [str(each_state) for each_state in list(df_state['Date'])]
df_state['Date'] = str_state
df_state_hmm = df_state[((df_state['Date'])>=str(start_time))&(df_state['Date']<str(end_time))]['state']
##hmm_type == -1 represents we use the pretrained result for our classification.
hmm_type = -1

# RSDR CVaR (HMM)
method_name = "RSDR CVaR (HMM)"
mean_constr = False
fcvar_hmm_star = FCVaR_HMM_wasserstein(df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_num, method_name, df_state_hmm, hmm_type, cv_type, mean_constr, ambiguity_param)




# #initiation for the test metric module
# #print the output into a separate csv file, separated by the dataset type + header
Output_csv = output(csv_name, data_head, data_parameter)
#Output_csv.head()
#return statistics computation for our policy


fcvar_hmm_star.rolling()
Output_csv.return_info(fcvar_hmm_star, rfr_data)

# # #DR CVaR
# # mean_constr = False
# cv_type = 1
# #mean-FCVaR model (Wasserstein)
# method_name = "DR CVaR (Wasserstein)"
# mean_fcvar_wasserstein = FCVaR_wasserstein2(df_select, df_train, rolling_day, portfolio_number, 1, method_name, cv_type, mean_constr)
# mean_fcvar_wasserstein.rolling()
# Output_csv.return_info(mean_fcvar_wasserstein, rfr_data)

# df_return = pd.DataFrame(pd.Series(list(fcvar_hmm_star.return_array)))
# df_return.to_csv('../return_cv/' + data_type + '_' + str(portfolio_number) +'.csv', index = None)

# df_return2 = pd.DataFrame(pd.Series(list(mean_fcvar_wasserstein.return_array)))
# df_return2.to_csv('../return_cv/' + data_type + '_' + str(portfolio_number) + 'bc.csv', index = None)

