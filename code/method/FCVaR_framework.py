# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:50:22 2020

@author: wangtianyu6162
"""

from numpy.lib.shape_base import _put_along_axis_dispatcher
import pandas as pd
import numpy as np
import math
from CVaR_parameter import *
from gurobipy import *
from method.strategy import *
from method.support import *

class FCluster_framework(strategy, strategy_cluster, resample):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_sign, cluster_number, method_name, switch = False, mean_constr = True):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.df_factor = df_factor
        self.method_name = method_name
        self.cluster_sign = cluster_sign
        #cluster_sign means choose which method to cluster
        #if cluster_sign = 1, use factor to build clusters
        #else if cluster_sign = 0, just use return to build clusters
        self.cluster_number = cluster_number
        self.min_mean = 0
        #homogeneous or heterogeneous
        self.switch = switch
        if type(switch) != bool:
            self.factor_state = np.array(list(switch))
        self.hmm_type = -1
        self.mean_constr = mean_constr
   

    
    def optimize_cluster(self, cluster_freq, mean_info, cov_info,test_return, weight_pre, netzeros_sign = 0):
        (num_of_sample, port_num) = test_return.shape
        covar = list()
        for i in range(self.cluster_number):
            covar.append(cov_info.iloc[port_num*i:port_num*(i+1)])


        # Create a new model
        m = Model("FCVaR_frame_clusters")

        # Create variables

        v = m.addVar(name = 'v', lb = -GRB.INFINITY, ub = GRB.INFINITY)
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
        #alpha = m.addVars(self.cluster_number, lb = 0)
        covar_bound = m.addVars(self.cluster_number, lb = 0)
        covar_mean = m.addVars(self.cluster_number, lb = 0, name = 'covar')#sqrt((mu*x+v)^2+x'sigmax
        
        #robust approximate version
        self.gamma1 = np.ones(self.cluster_number)*0.005
        self.gamma2 = 1 + np.ones(self.cluster_number)*0.005
        # self.gamma1 = np.array([math.sqrt(0.1/math.sqrt(cluster_freq[i])) for i in range(self.cluster_number)])
        # self.gamma2 = np.array([1 + 0.5/math.sqrt(cluster_freq[i]) for i in range(self.cluster_number)])
        
        ## Set the objectinve: 
        obj = v
        for i in range(self.cluster_number):
            obj = obj + cluster_freq[i]*(covar_mean[i] + self.gamma1[i]*covar_bound[i] - np.dot(mean_info.iloc[i], weight) - v)/(2*epsilon) 

        # Set the constraints:
        if netzeros_sign == 0:
            m.addConstrs((np.dot(covar[i], weight).dot(weight) <= covar_bound[i]*covar_bound[i]
                        for i in range(self.cluster_number)),'first0')
        else:
            covar_re = list()
            unit_var = np.ones(port_num)
            for i in range(self.cluster_number):
                if np.array(covar[i])[0][0] != 0:
                    covar_readjust = covar[i] - 0.05*np.dot(covar[i], unit_var).dot(np.dot(unit_var.T, covar[i]))/(np.dot(covar[i], unit_var).dot(unit_var))
                    if np.all(np.linalg.eigvals(covar_readjust)>0):
                        covar_re.append(covar_readjust)
                    else:
                        covar_re.append(np.zeros(shape = (port_num, port_num)))
                else:
                    covar_re.append(np.zeros(shape = (port_num, port_num)))
            #print(covar_re)
            m.addConstrs((np.dot(covar_re[i], weight).dot(weight)<= covar_bound[i]*covar_bound[i]
                        for i in range(self.cluster_number)), 'first0')
            
        m.addConstrs((self.gamma2[i]*covar_bound[i]*covar_bound[i] + (np.dot(mean_info.iloc[i],weight) + v + math.sqrt(self.gamma1[i])*covar_bound[i])*(np.dot(mean_info.iloc[i],weight) + v + self.gamma1[i]*covar_bound[i]) <= covar_mean[i]*covar_mean[i]
                        for i in range(self.cluster_number)), "c0")
        m.addConstrs((self.gamma2[i]*covar_bound[i]*covar_bound[i] + (np.dot(mean_info.iloc[i],weight) + v - math.sqrt(self.gamma1[i])*covar_bound[i])*(np.dot(mean_info.iloc[i],weight) + v - self.gamma1[i]*covar_bound[i]) <= covar_mean[i]*covar_mean[i]
                        for i in range(self.cluster_number)), "c0")
        
        m.addConstr(weight.sum() == 1,'budget')
        
        min_return_LHS = 0
        for i in range(self.cluster_number):
            min_return_LHS += cluster_freq[i]*(np.dot(mean_info.iloc[i], weight) - math.sqrt(self.gamma1[i])*covar_bound[i])
        m.addConstr(min_return_LHS >= self.min_mean, 'minimum return constraint')
        
        if self.mean_constr == True:
            return_target = 0
            for i in range(self.cluster_number):
                return_target += cluster_freq[i]*(np.dot(mean_info.iloc[i], weight) - math.sqrt(self.gamma1[i]) * covar_bound[i])
            m.addConstr(return_target >= self.min_mean)
        m.setObjective(obj, GRB.MINIMIZE)

        
        
       
        #Solve the Optimization Problem
        m.setParam('OutputFlag',0)
        m.optimize()
        
        #Retrieve the weight
        #print("The optimal weight portfolio from the Popescu Bound with clusters is :")
        weight = [v.x for v in weight]
        
        tran_cost = 0
        for i in range(port_num):
            tran_cost = tran_cost + tran_cost_p*abs(weight[i] - weight_pre[i])
            #print(tran_cost)  
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))  

        [return_FCluster,weight] = self.show_return(test_return,weight)
        
        return [weight, return_FCluster*(1 - tran_cost)]
    
    def rolling(self, netzeros_sign):
        #netzeros-sign show whether to adjust the uncertainty set's parameters to a stable level
        
        i = 0
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        while i < num_of_sample - num_of_train:   
            print(i)
            train_return = self.df_select[i: i + num_of_train]
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)])
           
            if type(self.switch) == bool:
                if self.cluster_sign == 1:
                    #use factor to cluster
                    factor_data = self.df_factor[i: i + num_of_train]
                    [cluster_freq, mean_info, cov_info] = self.factor_cluster(train_return, factor_data, self.column_name,self.cluster_number)
                else:
                    #use return to cluster
                    [cluster_freq, mean_info, cov_info] = self.return_cluster(train_return,self.column_name,self.cluster_number)
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
            cov_info = cov_info.fillna(0)
            #fix parameter
            train_array = np.array(train_return)
            (train_num, port_num) = train_array.shape
            #find the percentile = 20% for each time
            Rmin_list = np.array(train_array).reshape(train_num*port_num, )        
            self.min_mean = np.percentile(Rmin_list, lower_percentile)

            #self.robust_level_cluster(train_return, cluster_freq, mean_info, cov_info)
            [self.weight, self.return_array[i:i + self.rolling_day]] = self.optimize_cluster(cluster_freq, mean_info, cov_info, test_return, self.weight, netzeros_sign)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day