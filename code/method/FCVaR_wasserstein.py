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
class FCVaR_wasserstein(strategy, strategy_cluster, resample):
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
        self.theta_list = theta_list
        
    def optimize(self, train_return, test_return, weight_pre):
        
        train_return = np.array(train_return)
        (num_of_train, port_num) = train_return.shape
        m = Model("RSDR-benchmark-2021")
        #Create variables
        v = m.addVar(name = "v", lb = -GRB.INFINITY)
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
    
        #initiation for the auxiliary variables
        ## expansions due to the requirements in Gurobi
        aux_b = []
        ##initiation
        aux_a = pd.Series(m.addVars(num_of_train, lb = -GRB.INFINITY))
        
        for cls_index in range(num_of_train):
            aux_b.append(pd.Series(m.addVars(port_num, lb = -GRB.INFINITY)))
        #budget constraint for c, "c6"
        aux_c = pd.Series(m.addVars(num_of_train, lb = 0))
        aux_alpha = pd.Series(m.addVars(num_of_train, lb = -GRB.INFINITY))
        aux_beta = m.addVar(lb = -GRB.INFINITY)
        #objective
        obj = v
        target_inside = 0
        for cls_index in range(num_of_train):
            target_inside += 1/num_of_train*aux_alpha[cls_index]
        obj += (target_inside + aux_beta*self.theta)/epsilon
        #constraints
        #print(train_return)
        m.addConstrs((-np.dot(train_return[cls_index], aux_b[cls_index]) >= aux_a[cls_index] - aux_alpha[cls_index]
                        for cls_index in range(num_of_train)), "c1")
        m.addConstrs((aux_beta - aux_c[cls_index] >= 0 
                    for cls_index in range(num_of_train)), "c2")
        m.addConstrs((aux_beta - aux_c[cls_index] >= aux_b[cls_index][port_index]
                    for port_index in range(port_num) for cls_index in range(num_of_train)), "c3")

        m.addConstrs((aux_beta - aux_c[cls_index] >= -aux_b[cls_index][port_index]
                    for port_index in range(port_num) for cls_index in range(num_of_train)), "c4")

        m.addConstrs((np.dot(train_return[cls_index], aux_b[cls_index]) + np.dot(train_return[cls_index], weight) >= -aux_a[cls_index] - v
                    for cls_index in range(num_of_train)), "c5")
        
        m.addConstrs((aux_c[cls_index] >= aux_b[cls_index][port_index] + weight[port_index]
                        for port_index in range(port_num) for cls_index in range(num_of_train)), "c7")
        m.addConstrs((aux_c[cls_index] >= -aux_b[cls_index][port_index] - weight[port_index]
                        for port_index in range(port_num) for cls_index in range(num_of_train)), "c8")
        m.addConstrs((np.dot(train_return[cls_index], aux_b[cls_index]) >= -aux_a[cls_index]
                        for cls_index in range(num_of_train)), "c9")
        m.addConstrs((aux_c[cls_index] >= aux_b[cls_index][port_index]
                        for port_index in range(port_num) for cls_index in range(num_of_train)), "c10")
        m.addConstrs((aux_c[cls_index] >= -aux_b[cls_index][port_index]
                        for port_index in range(port_num) for cls_index in range(num_of_train)), "c11")
        #worst-case min returns
        if self.mean_constr == True:
            #add the additional variables
            aux_v = m.addVar(name = 'aux_v')
            aux_eta = pd.Series(m.addVars(num_of_train, lb = -GRB.INFINITY))
            ##add the additional constraints
            mean_obj = 0
            m.addConstrs((aux_v - weight[i] >=0 for i in range(port_num)), "mean1")
            m.addConstrs((aux_v + weight[i] >=0 for i in range(port_num)), "mean2")
            m.addConstrs((np.dot(train_return[cls_index], weight) >= aux_eta[cls_index]
                            for cls_index in range(num_of_train)), "mean3")
            ###important budge constraints
            for cls_index in range(num_of_train):
                mean_obj += aux_eta[cls_index]/num_of_train
            mean_obj -= aux_v*self.theta
            m.addConstr(mean_obj >= self.min_mean, "mean_constr")
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
    def pre_cv_lp(self, train_return, shortsale_sign):
        if self.cv_type == 1:
            (num_of_train, port_num) = train_return.shape
            m = Model("Wasserstein_pre_cv_2021")
            if shortsale_sign == 0:
                weight = pd.Series(m.addVars(port_num))
            else:
                weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
            m.addConstr(weight.sum() == 1,'budget')
            ##add the additional variables
            aux_v = m.addVar(self.cluster_number)
            aux_eta = pd.Series(m.addVars(num_of_train, lb = -GRB.INFINITY))
            ##add the additional constraints
            mean_obj = 0
            m.addConstrs((aux_v - weight[i] >=0 for i in range(port_num)), "mean1")
            m.addConstrs((aux_v + weight[i] >=0 for i in range(port_num)), "mean2")
            m.addConstrs((np.dot(train_return[cls_index], weight) >= aux_eta[cls_index]
                            for cls_index in range(num_of_train)), "mean3")
            ###important budge constraints
            for cls_index in range():
                mean_obj += aux_eta[cls_index]/num_of_train
            mean_obj -= aux_v*self.theta
            m.setObjective(mean_obj, GRB.MAXIMIZE)
            #Solve the Optimization Problem
            m.setParam('OutputFlag',0)
            m.optimize()  
            return m.objVal
        else:
            return False

        
    def rolling(self):
        i = 0
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        self.all_weight_pair = []
        #create the weight array
        while i < num_of_sample - num_of_train:
            print(i)
            train_return = self.df_select[i: (i + num_of_train)]
            #cross-validation part: (default) set to be 4 folds (leave the first, the actual number is 3)
            temp_turnover = self.turnover
            ## require cross validation
            if self.cv_type >= 0:
                self.cross_validation3(train_return, fold_number)
            else:
                #fix parameter
                train_array = np.array(train_return)
                (train_num, port_num) = train_array.shape
                #find the percentile = 20% for each time
                Rmin_list = np.array(train_array).reshape(train_num*port_num, )        
                self.min_mean = np.percentile(Rmin_list, lower_percentile)
            
            
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day][0: self.portfolio_number])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)][0: self.portfolio_number])
            
            #remove the effect of weight and turnover part
    
            temp_weight = self.weight
            
            self.turnover = temp_turnover

            #print(train_return)
            [self.weight, self.return_array[i:i + self.rolling_day], flag] = self.optimize(train_return, test_return, temp_weight)
            self.all_weight_pair.append(self.weight)
            self.weight = np.array(self.weight)
            i = i + self.rolling_day
        