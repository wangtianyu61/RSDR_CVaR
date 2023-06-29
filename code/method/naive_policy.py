# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:17:11 2019

@author: wangtianyu6162
"""

#naive_policy.py is meant to give the performance 
#give each portfolio the same weight:1/N
import pandas as pd
import numpy as np
from CVaR_parameter import *
from method.strategy import *
    
class naive_strategy(strategy):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, method_name):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.method_name = method_name
    def optimize(self, test_return, weight_pre):
        (num_of_sample,port_num) = test_return.shape 
        portfolio_weight =  np.ones(port_num)/port_num
        
        tran_cost = tran_cost_p*np.sum(abs(portfolio_weight - weight_pre))
        #print(date)        
        self.turnover = self.turnover + np.sum(abs(portfolio_weight - weight_pre))
        
        [return_naivepolicy, portfolio_weight] = self.show_return(test_return, portfolio_weight)                
        #print(portfolio_weight, weight_pre)
        return [portfolio_weight, return_naivepolicy*(1 - tran_cost)]
    def rolling(self):
        i = 0
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        print(num_of_sample, num_of_train)
        while i < num_of_sample - num_of_train:
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)])
            [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize(test_return, self.weight)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day
  
        
        
