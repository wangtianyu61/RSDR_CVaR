# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:17:39 2020

@author: wangtianyu6162
"""

import math
from os import mkdir
import pandas as pd
import numpy as np
from CVaR_parameter import *
from gurobipy import *
from method.strategy import *
from method.support import *

class mean_FCVaR(strategy, strategy_cluster, resample):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, method_name, df_factor = None, switch = False, cluster_sign = -1, cluster_number = 1, mean_constr = True):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
#        self.df_factor = df_factor
        self.method_name = method_name
        #help determine the value of gamma1, gamma2 and the minimum mean of the return
        self.robust_level_mean_cvar(df_train)
        self.mean_constr = mean_constr
        self.cv_type = cv_type
        self.switch = switch
        if type(switch)!= bool:
            self.factor_state = np.array(list(switch))
        self.cluster_sign = cluster_sign
        self.cluster_number = cluster_number
        self.df_factor = df_factor
        self.hmm_type = -1

        
    def optimize(self, train_return,  test_return, weight_pre):
    #Create a Model
        m = Model("mean_cvar")
        train_return = np.array(train_return)
        train_return_mean = np.mean(train_return, axis = 0)
        train_return_covar = np.cov(train_return, rowvar = False, ddof = 1)
        
        # Create variables
        ## the notations follows the same in p. 8 in Kang et al. (2019)
        #print(column_name)
        mu = pd.Series(train_return_mean.tolist())
        covar = pd.DataFrame(train_return_covar,index = None)
        (num_of_sample,num) = test_return.shape
        k = math.sqrt((1-epsilon)/epsilon)
        unit_var = np.ones(self.portfolio_number)
        ## adjusted covariance matrix 
        #print(covar)
        #eigenvalue, eigenvector = np.linalg.eig(np.array(covar))
        #print(eigenvalue)
        covar_unit = np.dot(covar, unit_var)
        self.covar_readjust = covar - covar_unit*np.transpose([covar_unit])/(np.dot(covar, unit_var).dot(unit_var))
        covar_overload = np.array(covar) + self.gamma2*np.eye(self.portfolio_number)
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(num))
        else:
            weight = pd.Series(m.addVars(num, lb = -GRB.INFINITY))
        
        weight_dif = pd.Series(m.addVars(num)) #abs(x(i,t+1)-x(i,t))
        #auxiliary variables, default value to be 0
        delta = m.addVar(name = 'v')
        w = m.addVar(name = 'w')
        
        
        obj = k*delta - np.dot(mu, weight) + math.sqrt(self.gamma1)*w
        
        #Set the general constraint:
        m.addConstr(weight.sum() == 1,'budget')
        
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(num)),'abs2')
        
        #Set the Specific constraint
        #m.addConstr(np.dot(mu, weight) - self.min_mean >= 0, 'min_mean_constraint')
        ##the following constraint represents the minimum return constraint
        if self.mean_constr == True:
            m.addConstr((0<= np.dot(mu, weight) - self.min_mean), 'c0')
            #m.addConstr(self.gamma1*(np.dot(self.covar_readjust, weight).dot(weight)) <= (np.dot(mu, weight) - self.min_mean)*(np.dot(mu, weight) - self.min_mean), 'covar_constraint')
        ## additional
        m.addConstr(np.dot(covar_overload, weight).dot(weight) <= delta*delta, "covar_constraint2")
        m.addConstr(np.dot(self.covar_readjust, weight).dot(weight) <= w*w, "covar_constraint3")
        m.setObjective(obj, GRB.MINIMIZE)
                            
        #Solve the Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()

        flag = 0      
        try:
            self.base_line = m.objVal#Retrieve the weight    
            weight = [v.x for v in weight]
            #print(weight)
        except Exception as e:
            flag = 1
            #print(e)
            weight = np.ones(self.portfolio_number)/self.portfolio_number
    
        tran_cost = 0
        for i in range(num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i]) 
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))    

        [return_mean_cvar, weight] = self.show_return(test_return, weight) 

        return [weight, return_mean_cvar*(1 - tran_cost), flag]
    
    def optimize_cluster(self, cluster_freq, mean_info, cov_info,test_return, weight_pre):
    #Create a Model
        m = Model("mean_cvar")
        covar = list()
        for i in range(self.cluster_number):
            covar.append(cov_info.iloc[self.portfolio_number*i : self.portfolio_number*(i + 1)])
        
        #robust approximate version
        self.gamma1 = np.ones(self.cluster_number)*0.005
        self.gamma2 = np.ones(self.cluster_number)*0.005
        # self.gamma1 = np.array([math.sqrt(0.1/math.sqrt(cluster_freq[i])) for i in range(self.cluster_number)])
        # self.gamma2 = np.array([1 + 0.5/math.sqrt(cluster_freq[i]) for i in range(self.cluster_number)])
        

        # Create variables
        ## the notations follows the same in p. 8 in Kang et al. (2019)

        (num_of_sample,num) = test_return.shape
        k = math.sqrt((1-epsilon)/epsilon)
        unit_var = np.ones(self.portfolio_number)
        ## adjusted covariance matrix 
        #print(covar)
        #eigenvalue, eigenvector = np.linalg.eig(np.array(covar))
        #print(eigenvalue)
        covar_readjust = []
        covar_overload = []
        for i in range(self.cluster_number):
            covar_unit = np.dot(covar[i], unit_var)
            covar_overload.append(np.array(covar[i]) + self.gamma2[i]*np.eye(self.portfolio_number))
            covar_readjust.append(covar[i] - covar_unit*np.transpose([covar_unit])/(np.dot(covar[i], unit_var).dot(unit_var)))

        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(num))
        else:
            weight = pd.Series(m.addVars(num, lb = -GRB.INFINITY))
        
        weight_dif = pd.Series(m.addVars(num)) #abs(x(i,t+1)-x(i,t))
        #auxiliary variables, default value to be 0
        delta = m.addVars(self.cluster_number)
        w = m.addVars(self.cluster_number)
        return_mean = 0
        for i in range(self.cluster_number):
            return_mean += cluster_freq[i]*np.dot(mean_info.iloc[i], weight)
        
        obj = -return_mean
        for i in range(self.cluster_number):
            obj += cluster_freq[i]*((math.sqrt(self.gamma1[i])*w[i] + k*delta[i]))
        
        #Set the general constraint:
        m.addConstr(weight.sum() == 1,'budget')
        
        m.addConstrs((weight[i] - weight_pre[i] <= weight_dif[i]
                    for i in range(num)),'abs1')
        m.addConstrs((weight_pre[i] - weight[i] <= weight_dif[i]
                    for i in range(num)),'abs2')
        
        #Set the Specific constraint
        #m.addConstr(np.dot(mu, weight) - self.min_mean >= 0, 'min_mean_constraint')
        ##the following constraint represents the minimum return constraint
        if self.mean_constr == True:
            m.addConstr((0<= return_mean - self.min_mean), 'c0')
            #m.addConstr(self.gamma1*(np.dot(self.covar_readjust, weight).dot(weight)) <= (np.dot(mu, weight) - self.min_mean)*(np.dot(mu, weight) - self.min_mean), 'covar_constraint')
        ## additional
        m.addConstrs((np.dot(covar_overload[i], weight).dot(weight) <= delta[i]*delta[i]
                    for i in range(self.cluster_number)), "covar_constraint2")
        m.addConstrs((np.dot(covar_readjust[i], weight).dot(weight) <= w[i]*w[i]
                    for i in range(self.cluster_number)), "covar_constraint3")
        m.setObjective(obj, GRB.MINIMIZE)
                            
        #Solve the Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()

        flag = 0      
        try:
            self.base_line = m.objVal#Retrieve the weight    
            weight = [v.x for v in weight]
            #print(weight)
        except Exception as e:
            flag = 1
            #print(e)
            weight = np.ones(self.portfolio_number)/self.portfolio_number
    
        tran_cost = 0
        for i in range(num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i]) 
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))    

        [return_mean_cvar, weight] = self.show_return(test_return, weight) 

        return [weight, return_mean_cvar*(1 - tran_cost), flag]
    def rolling(self):
        i = 0
        pre_weight = np.zeros(self.portfolio_number)
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        self.all_weight_pair = []
        while i < num_of_sample - num_of_train:   
            train_return = self.df_select[i: (i + num_of_train)]
            
            
            #cross-validation part: (default) set to be 4 folds (leave the first, the actual number is 3)
            temp_turnover = self.turnover
            ## require cross validation
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)])

            
            train_array = np.array(train_return)
            (train_num, port_num) = train_array.shape
            #find the percentile = 20% for each time
            Rmin_list = np.array(train_array).reshape(train_num*port_num, )        
            self.min_mean = np.percentile(Rmin_list, lower_percentile)
            #fix parameter
            train_array = np.array(train_return)
            (train_num, port_num) = train_array.shape
            #find the percentile = 20% for each time
            Rmin_list = np.array(train_array).reshape(train_num*port_num, )        
            self.min_mean = np.percentile(Rmin_list, lower_percentile)

            MD = np.linalg.norm(np.max(train_array, axis = 0) - np.min(train_array, axis = 0), ord = 1)
            #print(i, MD)
            
            
            if self.cluster_sign == -1:
                if self.cv_type >= 0 and self.mean_constr == True:
                    self.cross_validation2(train_return, fold_number)
                self.turnover = temp_turnover
                [self.weight, self.return_array[i:i + self.rolling_day], flag] = self.optimize(train_return, test_return, self.weight)
            else:
                #homogeneous
                if type(self.switch) == bool:
                    if self.cluster_sign == 1:
                        #use factor to cluster
                        factor_data = self.df_factor[i: i + num_of_train]
                        [cluster_freq, mean_info, cov_info] = self.factor_cluster(train_return, factor_data, self.column_name,self.cluster_number)
                    else:
                        #use return to cluster
                        [cluster_freq, mean_info, cov_info] = self.return_cluster(train_return,self.column_name,self.cluster_number)

                #heterogeneous
                else:
                    #use the output state to define 
                    train_state = self.factor_state[i:(i + num_of_train)]
                    [num_time_in_cluster, cluster_freq, train_return_full] = self.hmm_train(train_return, self.cluster_number, train_state, self.hmm_type)
                    #divide by cluster
                    mean_info_array = np.zeros((self.cluster_number, self.portfolio_number))
                    cov_info_array = np.zeros((self.portfolio_number*self.cluster_number, self.portfolio_number))
                    for j in range(self.cluster_number):
                        mean_info_array[j] = np.mean(np.array(train_return_full[j]), axis = 0)
                        cov_info_array[(self.portfolio_number*j):self.portfolio_number*(j+1)] = np.cov(np.array(train_return_full[j]).T)
                    mean_info = pd.DataFrame(mean_info_array)
                    cov_info = pd.DataFrame(cov_info_array)
                #self.cov_info = cov_info
                self.turnover = temp_turnover
                [self.weight, self.return_array[i:i + self.rolling_day], flag] = self.optimize_cluster(cluster_freq, mean_info, cov_info, test_return, self.weight)
            self.all_weight_pair.append(self.weight)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day
