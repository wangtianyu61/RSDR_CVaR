import pandas as pd
import numpy as np
import math
from CVaR_parameter import *
from gurobipy import *

import matplotlib.pyplot as plt
from method.strategy import *
from method.support import *

#The classical method of DRO based on wasserstein distance
## SET as the benchmark for RSDR-HMM Problem
class FCVaR_wasserstein2(strategy, strategy_cluster, resample):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    #adj_level means the type of robustness optimization
    #cv_type means the type of cross-validation, 0 means gridsearch (even infeasible possibly), 1 means search from the possiblem maximum (pre_cv_lp), -1 means no cv.
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, cluster_number, method_name, cv_type = 1, mean_constr = True):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.method_name = method_name
        self.cluster_number = cluster_number
        self.mean_target = False
        self.theta = 0.05
        self.cv_type = cv_type
        self.mean_constr = mean_constr
        self.theta_param = theta_param
        
    def optimize(self, train_return, test_return, weight_pre):
        (num_of_train, port_num) = train_return.shape
        train_return = np.array(train_return)
        m = Model("RSDR-benchmark-2021")
        #Create variables
        v = m.addVar(name = "v", lb = -GRB.INFINITY)
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
        #initiation for additional variables
        aux_alpha = pd.Series(m.addVars(num_of_train))
        aux_lambda = m.addVar(name = 'lambda')

        m.addConstrs((-1/epsilon*np.dot(train_return[cls_index], weight) + v*(1 - 1/epsilon) <= aux_alpha[cls_index]
                        for cls_index in range(num_of_train)), 'cmain')
        m.addConstrs((aux_alpha[cls_index] >= v for cls_index in range(num_of_train)), 'c0')
        m.addConstrs((weight[i] <= epsilon*aux_lambda for i in range(port_num)), 'c1')
        #worst-case min returns
        if self.mean_constr == 'worst-case':
            #add the additional variables with regards to return constraint
            aux_beta = pd.Series(m.addVars(num_of_train, lb = -GRB.INFINITY))
            aux_l = m.addVar(name = 'l')
            for i in range(num_of_train):
                constr_LHS = aux_l*self.theta + aux_beta[i]/num_of_train
            m.addConstr(constr_LHS <= -self.min_mean)
            m.addConstrs((-np.dot(train_return[cls_index], weight)<=aux_beta[cls_index]
                            for cls_index in range(num_of_train)), 'c_appendix1')
            m.addConstrs((weight[i]<= aux_l for i in range(port_num)), 'c_appendix2')

        elif self.mean_constr == 'empirical':
            mean_obj = 0
            for cls_index in range(num_of_train):
                mean_obj += np.dot(train_return[cls_index], weight)/num_of_train
            m.addConstr(mean_obj >= self.min_mean, "mean_constr")


        #set the objective
        obj = aux_lambda*self.theta 
        for cls_index in range(num_of_train):
            obj += aux_alpha[cls_index]/num_of_train

        m.addConstr(weight.sum() == 1,'budget')    
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
            print(e)
            weight = np.ones(self.portfolio_number)/self.portfolio_number
        
        tran_cost = tran_cost_p*np.sum(abs(weight - weight_pre))
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))   
        [return_HMM,weight] = self.show_return(test_return, weight)
        return [weight, return_HMM*(1 - tran_cost), flag]


    
    def rolling(self):
        i = 0
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        self.all_weight_pair = []      
        dta_param = math.pow(num_of_train, 1/self.portfolio_number)
        #create the weight array
        while i < num_of_sample - num_of_train:
            
            train_return = self.df_select[i: (i + num_of_train)]
            #cross-validation part: (default) set to be 4 folds (leave the first, the actual number is 3)
            temp_turnover = self.turnover
            ## require cross validation

            MD = np.max(np.array(train_return)) - np.min(np.array(train_return))
            if self.cv_type >= 0:
                self.theta_list = [param/dta_param for param in self.theta_param] 
                self.cross_validation3(train_return, fold_number)
                #print(self.theta)
            else:
                #fix parameter
                
                train_array = np.array(train_return)
                (train_num, port_num) = train_array.shape
                #find the percentile = 20% for each time
                Rmin_list = np.array(train_array).reshape(train_num*port_num, )        
                self.min_mean = np.percentile(Rmin_list, lower_percentile)
            
            train_return = np.array(train_return)
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day][0: self.portfolio_number])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)][0: self.portfolio_number])
            
            #MD = np.linalg.norm(np.max(train_return, axis = 0) - np.min(train_return, axis = 0), ord = 1)
            #print(i, MD)
            #self.theta = MD*0.005
            #remove the effect of weight and turnover part
    
            temp_weight = self.weight
            self.turnover = temp_turnover
            #print(train_return)
            [self.weight, self.return_array[i:i + self.rolling_day], flag] = self.optimize(train_return, test_return, temp_weight)
            self.all_weight_pair.append(self.weight)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day