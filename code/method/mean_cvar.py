# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:38:41 2020

@author: wangtianyu6162
"""

import math
import pandas as pd
import numpy as np
from CVaR_parameter import *
from gurobipy import *
import matplotlib.pyplot as plt
from method.strategy import *
from method.support import *

class mean_CVaR(strategy, resample):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, method_name,mean_target = False):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.method_name = method_name
        self.mean_target = mean_target
        self.theta_list = theta_param
        self.cv_type = cv_type
    def optimize(self, train_return, test_return, weight_pre):#balance between cvar and mean
        m = Model('SAA_CVaR')

        #Create the Variables
        (train_num,port_num) = train_return.shape
        
        train_return_mean = np.array(train_return.mean())
        #print(train_return.iloc[0])
        
        return_weight = pd.Series(m.addVars(train_num)) # - mu*x - v
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
            
        weight_dif = pd.Series(m.addVars(port_num))
        #print(weight)
        v = m.addVar(name = 'v', lb = -GRB.INFINITY, ub = GRB.INFINITY)

        #Set the objective
        obj = v + (return_weight.sum() + trade_off*np.dot(train_return_mean, weight))/(epsilon*train_num)
        m.setObjective(obj, GRB.MINIMIZE)
        #Set the constraints

        if self.mean_target != False:
            max_value = max(np.array(train_return.mean()))
            m.addConstr(np.dot(np.array(train_return.mean()), weight) >= (1 - np.sign(max_value)*self.mean_target)*max_value, "to draw mean-cvar frontier")

        m.addConstrs((return_weight[i] >= 0
                      for i in range(train_num)),'nonnegative for positive expectation')
        m.addConstrs((return_weight[i] >=  -np.dot(train_return.iloc[i],weight)+ tran_cost_p*np.sum(weight_dif) - v
                      for i in range(train_num)),'c0')
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(port_num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(port_num)),'abs2')
    
        #minimum return constraint \mu^{\prime}x >= R
        m.addConstr(np.dot(train_return_mean, weight) - self.min_mean >= 0, 'min_mean_constraint')
    

        m.addConstr(weight.sum() == 1,'budget')        
        #Solve the Linear Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()
        flag = 0      
        try:
            self.base_line = m.objVal#Retrieve the weight    
            weight = [v.x for v in weight]
            #print(weight)
            # print('equities: ', sum(weight[0:10]))
            # print('bond: ', weight[10])
            # print('TIPS: ', weight[15])
            # print('Gold: ', weight[11])
            # print('Other commodities: ', sum(weight[12:14]))
            # print('Credit: ', weight[14])
            # print('======================================')
            #print(weight)
        except Exception as e:
            flag = 1
            print(e)
            weight = np.ones(self.portfolio_number)/self.portfolio_number

        #print(weight) 
        (num_of_sample,port_num) = test_return.shape
        
        tran_cost = 0
        for i in range(port_num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
            
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre)) 
        [return_mean_cvar, weight] = self.show_return(test_return,weight)    

        return [weight, return_mean_cvar*(1 - tran_cost), flag]
       
    def rolling(self):
        i = 0
        pre_weight = np.zeros(self.portfolio_number)
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        self.all_weight_pair = []
        while i < num_of_sample - num_of_train:
            #print(i)
            train_return = self.df_select[i: (i + num_of_train)]
            #print(train_return.iloc[0])
            #cross-validation part: (default) set to be 4 folds (leave the first, the actual number is 3)
            temp_turnover = self.turnover
            ## require cross validation
            if self.cv_type >= 0:
                self.cross_validation2(train_return, fold_number)
            else:
                #fix parameter
                train_array = np.array(train_return)
                (train_num, port_num) = train_array.shape
                #find the percentile = 20% for each time
                Rmin_list = np.array(train_array).reshape(train_num*port_num, )        
                self.min_mean = np.percentile(Rmin_list, lower_percentile)

            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)])
            self.turnover = temp_turnover
            [self.weight, self.return_array[i:i + self.rolling_day], flag] = self.optimize(train_return, test_return, self.weight)
            self.all_weight_pair.append(self.weight)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day
        return train_return
