# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 01:47:22 2020

@author: wangtianyu6162
"""
import pandas as pd
import numpy as np
import math
from CVaR_parameter import *
from gurobipy import *

import matplotlib.pyplot as plt
from method.strategy import *
from method.support import *

#The paper shown in Zhenzhen.
class FCVaR_HMM_wasserstein(strategy, strategy_cluster, resample):
    portfolio_number = 10
    weight = np.zeros(portfolio_number)
    method_name = "strategy"
    #adj_level means the type of robustness optimization
    #cv_type means the type of cross-validation, 0 means gridsearch (even infeasible possibly), 1 means search from the possiblem maximum (pre_cv_lp), -1 means no cv.
    def __init__(self, df_select, df_train, rolling_day, portfolio_number, df_factor, cluster_number, method_name, df_state, hmm_type = 2, cv_type = 1, mean_constr = True, theta_type = 0, ambiguity_param = 0):
        strategy.__init__(self, df_select, df_train, rolling_day, portfolio_number)
        self.df_factor = df_factor
        self.method_name = method_name
        self.cluster_number = cluster_number
        self.df_state = np.array(list(df_state))
        self.mean_target = False
        self.adj_level = False
        self.hmm_state_estimate = False
        self.hmm_type = hmm_type
        #help determine the value of gamma1, gamma2 and the minimum mean of the return
        self.robust_level_mean_cvar(df_train)
        self.theta = 0.0
        self.theta_type = theta_type
        self.cv_type = cv_type
        self.mean_constr = mean_constr
        
        self.ambiguity_param = ambiguity_param
        self.theta_param = theta_param
        self.coef = 1
        if self.portfolio_number < 10:
            self.coef = self.cluster_number
    def optimize_test(self, cluster_freq, num_time_in_cluster, train_return, test_return, weight_pre):
        (num_of_sample, port_num) = test_return.shape
        for i in range(self.cluster_number):
            train_return_array = np.array(train_return[i])
        m = Model('HMM_2021')
        v = m.addVar(name = 'v', lb = -GRB.INFINITY)
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
        aux_lambda = m.addVar(name = 'lambda')
        aux_b = []
        num_train = len(self.df_train)
        for i in range(self.cluster_number):
            aux_b.append(pd.Series(m.addVars(num_train, lb = -GRB.INFINITY)))
        for i in range(self.cluster_number):
            m.addConstrs((-np.dot(train_return[i][cls_index], weight)/epsilon + v*(1- 1/epsilon) <= aux_b[i][cls_index]
                            for cls_index in range(num_time_in_cluster[i])), 'c0')
            m.addConstrs((v <= aux_b[i][cls_index] for cls_index in range(num_time_in_cluster[i])), 'c1')
        if self.cost_norm == 1:
            m.addConstrs((aux_lambda >= weight[i]/epsilon for i in range(port_num)), 'c2')
        elif self.cost_norm == 2:
            m.addConstr(quicksum(weight[i] * weight[i] for i in range(port_num)) <= aux_lambda * aux_lambda * epsilon ** 2, 'c2')

        #objective need to be change
        obj = 0
        for i in range(self.cluster_number):
            obj += self.theta*cluster_freq[i]*aux_lambda
        for i in range(self.cluster_number):
            for cls_index in range(num_time_in_cluster[i]):
                obj += cluster_freq[i]*aux_b[i][cls_index]/num_time_in_cluster[i]
        m.addConstr(weight.sum() == 1,'budget')

        if self.mean_constr == 'worst-case':
            #additional mean constraints for the worst-case mean
            ##define another two new sets of variables
            aux_eta = []
            aux_v = m.addVar(name = 'v')
            for i in range(self.cluster_number):
                aux_eta.append(pd.Series(m.addVars(num_train, lb = -GRB.INFINITY)))
            ##add the additional constraints
            mean_obj = 0
    
            if self.cost_norm == 1:
                m.addConstrs((aux_v - weight[i] >=0 for i in range(port_num)), "mean1")
                m.addConstrs((aux_v + weight[i] >=0 for i in range(port_num)), "mean2")
            elif self.cost_norm == 2:
                m.addConstr(quicksum(weight[i]*weight[i] for i in range(port_num)) <= aux_v * aux_v, "mean2-0")
            for k in range(self.cluster_number):
                m.addConstrs((np.dot(train_return[k][cls_index], weight) >= aux_eta[k][cls_index]
                                    for cls_index in range(num_time_in_cluster[k])), "mean3")
                mean_obj -= aux_v*cluster_freq[k]*self.theta
                ###important budge constraints
                for cls_index in range(num_time_in_cluster[k]):
                    mean_obj += cluster_freq[k]*aux_eta[k][cls_index]/num_time_in_cluster[k]
                
            m.addConstr(mean_obj >= self.min_mean, "mean_constr")


    
        m.setObjective(obj, GRB.MINIMIZE)
        #Solve the Optimization Problem
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
        except Exception as e:
            flag = 1
            print(e)
            weight = np.ones(self.portfolio_number)/self.portfolio_number
        
        tran_cost = tran_cost_p*np.sum(abs(weight - weight_pre))
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))   
        [return_HMM,weight] = self.show_return(test_return, weight)
        #print(weight, return_HMM, flag)
        return [weight, return_HMM*(1 - tran_cost), flag]
    
    def optimize(self, cluster_freq, num_time_in_cluster, train_return, test_return, weight_pre):

        
        #the variable of train return is a 3-d array, cluster * cluster_index * portfolio_dim
        #data preprocessing    
        

        (num_of_sample, port_num) = test_return.shape        
        #data overview
        for i in range(self.cluster_number):
            train_return_array = np.array(train_return[i])

        # preprocessing the data to get the num_time_in_cluster
        # other parameter
        # create a model
        m = Model("HMM_2020")
        # Create variables
        v = m.addVar(name = 'v', lb = -GRB.INFINITY)
        if shortsale_sign == 0:
            weight = pd.Series(m.addVars(port_num))
        else:
            weight = pd.Series(m.addVars(port_num, lb = -GRB.INFINITY))
        
        #initiation for the auxiliary variables
        ## expansions due to the requirements in Gurobi
        aux_a = []
        aux_b = [[] for i in range(self.cluster_number)]
        aux_c = []
        aux_alpha = []
        ## train for 10 years
        num_train = len(self.df_train)
        
        for i in range(self.cluster_number):
            aux_a.append(pd.Series(m.addVars(num_train, lb = -GRB.INFINITY)))
            
            for cls_index in range(num_time_in_cluster[i]):
                aux_b[i].append(pd.Series(m.addVars(port_num, lb = -GRB.INFINITY)))
            #budget constraint for c, "c6"
            aux_c.append(pd.Series(m.addVars(num_train, lb = 0)))
            aux_alpha.append(pd.Series(m.addVars(num_train, lb = -GRB.INFINITY)))
        
        aux_beta = pd.Series(m.addVars(self.cluster_number, lb = -GRB.INFINITY))
        obj = v
        for i in range(self.cluster_number):
            #inner loop of the target
            target_inside = 0
            for cls_index in range(num_time_in_cluster[i]):
                target_inside += cluster_freq[i]/num_time_in_cluster[i]*aux_alpha[i][cls_index]
            obj += 1/epsilon*(target_inside + aux_beta[i]*cluster_freq[i]*self.theta)
        # Set the constraints:
        ## aux constraint
        for i in range(self.cluster_number):
            m.addConstrs((-np.dot(train_return[i][cls_index], aux_b[i][cls_index]) >= aux_a[i][cls_index] - aux_alpha[i][cls_index]
                        for cls_index in range(num_time_in_cluster[i])), "c1")
            m.addConstrs((np.dot(train_return[i][cls_index], aux_b[i][cls_index]) + np.dot(train_return[i][cls_index], weight) >= -aux_a[i][cls_index] - v
                        for cls_index in range(num_time_in_cluster[i])), "c5")            
            m.addConstrs((np.dot(train_return[i][cls_index], aux_b[i][cls_index]) >= -aux_a[i][cls_index]
                            for cls_index in range(num_time_in_cluster[i])), "c9")

            
            if self.cost_norm == 1:
                m.addConstrs((aux_beta[i] - aux_c[i][cls_index] >= 0 
                            for cls_index in range(num_time_in_cluster[i])), "c2")
                m.addConstrs((aux_beta[i] - aux_c[i][cls_index] >= aux_b[i][cls_index][port_index]
                            for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c3")
                m.addConstrs((aux_beta[i] - aux_c[i][cls_index] >= -aux_b[i][cls_index][port_index]
                            for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c4")
                m.addConstrs((aux_c[i][cls_index] >= aux_b[i][cls_index][port_index] + weight[port_index]
                                for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c7")
                m.addConstrs((aux_c[i][cls_index] >= -aux_b[i][cls_index][port_index] - weight[port_index]
                                for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c8")

                m.addConstrs((aux_c[i][cls_index] >= aux_b[i][cls_index][port_index]
                                for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c10")
                m.addConstrs((aux_c[i][cls_index] >= -aux_b[i][cls_index][port_index]
                                for port_index in range(port_num) for cls_index in range(num_time_in_cluster[i])), "c11")
            elif self.cost_norm == 2:
                #handle the case of constraint when Q not PSD
                new_tran_obj = []
                for i in range(self.cluster_number):
                    new_tran_obj.append(pd.Series(m.addVars(num_train, lb = 0)))
                    #for cls_index in range(num_time_in_cluster[i]):
                    m.addConstrs(((quicksum(aux_b[i][cls_index][port_index]*aux_b[i][cls_index][port_index] for port_index in range(port_num)))<= aux_c[i][cls_index]*aux_c[i][cls_index]
                                for cls_index in range(num_time_in_cluster[i])), 'c10 + c11')
                    m.addConstrs(((quicksum(aux_b[i][cls_index][port_index]*aux_b[i][cls_index][port_index] + weight[port_index]*weight[port_index] for port_index in range(port_num))) <= aux_c[i][cls_index]*aux_c[i][cls_index] 
                                for cls_index in range(num_time_in_cluster[i])), 'c7 + c8')
                    #for cls_index in range(num_time_in_cluster[i]):
                    m.addConstrs((new_tran_obj[i][cls_index] == aux_beta[i] - aux_c[i][cls_index]
                                for cls_index in range(num_time_in_cluster[i])), 'c_add')
                    #obj = aux_beta[i] - aux_c[i][cls_index]
                    m.addConstrs(((quicksum(aux_b[i][cls_index][port_index]*aux_b[i][cls_index][port_index] for port_index in range(port_num)))<= new_tran_obj[i][cls_index]*new_tran_obj[i][cls_index]                                    
                                for cls_index in range(num_time_in_cluster[i])), 'c3 + c4')
                

        ## budget constraint in the worst-case
        ### m.addConstr(np.dot(train_return_mean, weight) - self.min_mean >= 0, 'min_mean_constraint')
        if self.mean_constr == 'worst-case':
            #additional mean constraints for the worst-case mean
            ##define another two new sets of variables
            aux_eta = []
            aux_v = pd.Series(m.addVars(self.cluster_number))
            for i in range(self.cluster_number):
                aux_eta.append(pd.Series(m.addVars(num_train, lb = -GRB.INFINITY)))
            ##add the additional constraints
            mean_obj = 0
            for k in range(self.cluster_number):
                if self.cost_norm == 1:
                    m.addConstrs((aux_v[k] - weight[i] >=0 for i in range(port_num)), "mean1")
                    m.addConstrs((aux_v[k] + weight[i] >=0 for i in range(port_num)), "mean2")
                    m.addConstrs((np.dot(train_return[k][cls_index], weight) >= aux_eta[k][cls_index]
                                    for cls_index in range(num_time_in_cluster[k])), "mean3")
                elif self.cost_norm == 2:
                    m.addConstr(quicksum(weight[i]*weight[i] for i in range(port_num)) <= aux_v[k]*aux_v[k], "mean2-0")
                ###important budge constraints
                for cls_index in range(num_time_in_cluster[k]):
                    mean_obj += cluster_freq[k]*aux_eta[k][cls_index]/num_time_in_cluster[k]
                mean_obj -= aux_v[k]*cluster_freq[k]*self.theta
            m.addConstr(mean_obj >= self.min_mean, "mean_constr")
                
        #threshold of avg of empiricial return        
        elif self.mean_constr == 'empirical':
            mean_obj = 0
            for i in range(self.cluster_number):
                for cls_index in range(num_time_in_cluster[i]):
                    mean_obj += cluster_freq[i]*np.dot(train_return[i][cls_index], weight)/num_time_in_cluster[i]
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
            beta = [v.x for v in aux_beta]
            #print(beta)
            #print(weight)
            # print('equities: ', sum(weight[0:10]))
            # print('bond: ', weight[10])
            # print('TIPS: ', weight[15])
            # print('Gold: ', weight[11])
            # print('Other commodities: ', sum(weight[12:14]))
            # print('Credit: ', weight[14])
            # print('======================================')
        except Exception as e:
            flag = 1
            print(e)
            weight = np.ones(self.portfolio_number)/self.portfolio_number
        
        tran_cost = tran_cost_p*np.sum(abs(weight - weight_pre))
        self.turnover = self.turnover + np.sum(abs(weight - weight_pre))   
        [return_HMM,weight] = self.show_return(test_return, weight)
        #print(weight, return_HMM, flag)
        return [weight, return_HMM*(1 - tran_cost), flag]
    
    #determine the value of the possible Rmin can attain beforehand to determine the upper bound of R.
    #how the cross validation result related to theta
    def pre_cv_lp(self, num_time_in_cluster, cluster_freq, train_return, shortsale_sign):
        #require to ensure the return can be attained 
        if self.cv_type == 1:
            m = Model("HMM_pre_cv_2021")
            if shortsale_sign == 0:
                weight = pd.Series(m.addVars(self.portfolio_number))
            else:
                weight = pd.Series(m.addVars(self.portfolio_number, lb = -GRB.INFINITY))
            m.addConstr(weight.sum() == 1,'budget')
            aux_eta = []
            aux_v = pd.Series(m.addVars(self.cluster_number))
            for i in range(self.cluster_number):
                aux_eta.append(pd.Series(m.addVars(len(self.df_train), lb = -GRB.INFINITY)))
            ##add the additional constraints
            mean_obj = 0
            for k in range(self.cluster_number):
                m.addConstrs((aux_v[k] - weight[i] >=0 for i in range(self.portfolio_number)), "mean1")
                m.addConstrs((aux_v[k] + weight[i] >=0 for i in range(self.portfolio_number)), "mean2")
                m.addConstrs((np.dot(train_return[k][cls_index], weight) >= aux_eta[k][cls_index]
                                for cls_index in range(num_time_in_cluster[k])), "mean3")

                ###important budge target
                for cls_index in range(num_time_in_cluster[k]):
                    mean_obj += cluster_freq[k]*aux_eta[k][cls_index]/num_time_in_cluster[k]
                
                mean_obj -= aux_v[k]*cluster_freq[k]*self.theta_list[len(self.theta_list) - 1]
            
            m.setObjective(mean_obj, GRB.MAXIMIZE)
            #Solve the Optimization Problem
            m.setParam('OutputFlag',0)
            m.optimize()  
            
            return m.objVal
        else:
            return False
    def cross_validation(self, train_return, train_state, fold_num, Rmin_pre = False):
        right_theta_param = 0
        right_Rmin_param = 0
        #all the parameters
        if Rmin_pre == False:
            Rmin_list = [-0.0003, 0, 0.0003, 0.0006]
        else:
            #max-min for the pre linear program
            Rmin_list = [Rmin_pre-0.00045, Rmin_pre-0.0003, Rmin_pre-0.00015]
        
        #drop the first bin
        for serial_number in range(2, fold_num + 1):
            #hmm train for each problem
            ## generate the overall state list given the hmm type
            ## determine the state in the cross validation part if hmm_type!=-1
            if self.hmm_type != -1:
                train_state = self.hmm_state_pretrain(train_return, self.cluster_number, hmm_type)
            
            [train_train, train_validation] = self.divide(train_return, fold_num, serial_number)
            
            train_return_mean = np.mean(train_train, axis = 0)
            [num_time_in_cluster, cluster_freq, train_return_cv] = self.hmm_train_cv(train_train, self.cluster_number, train_state, self.hmm_type, fold_num, serial_number)
            
            count = 0
            #for each fold
            temp_SR = np.zeros(len(self.theta_list)*len(Rmin_list))
            for i in range(len(self.theta_list)):
                self.theta = self.theta_list[i]
                for j in range(len(Rmin_list)):
                    weight = np.ones(self.portfolio_number)/self.portfolio_number
                    self.min_mean = Rmin_list[j]
                    [weight, return_array, flag] = self.optimize_test(cluster_freq, num_time_in_cluster, train_return_cv, train_validation, weight)                               
                    temp_SR[count] = np.mean(return_array)/np.std(return_array)
                    count += 1
            right_theta_param += self.theta_list[int(np.argmax(temp_SR)/len(Rmin_list))]
            right_Rmin_param += Rmin_list[np.argmax(temp_SR)%len(Rmin_list)]
        self.theta = right_theta_param/(fold_num - 1)
        self.min_mean = right_Rmin_param/(fold_num - 1)
    
    def cross_validation2_test(self, hmm_cv_info, train_validation):
        #classify into at most four possible regimes
        (vali_num, port_num) = train_validation.shape
        
        return_array = np.zeros(vali_num)
        select_columns = list(range(port_num))
        test_array = pd.DataFrame(train_validation, columns = select_columns)
        test_array['last_state'] = hmm_cv_info['last_state']
        weight = np.ones(self.portfolio_number)/self.portfolio_number
        k = 0
        for i in range(self.cluster_number):
            train_validation = np.array(test_array[test_array['last_state'] == i][select_columns])
            cluster_vali_num = len(train_validation)
            #print(train_validation.shape)
            if cluster_vali_num > 0:
                cluster_freq = hmm_cv_info[i][1]
                num_time_in_cluster = hmm_cv_info[i][0]
                train_return_cv = hmm_cv_info[i][2]
                #[weight, return_array[k:(k + cluster_vali_num)], flag] 

                [weight, return_array[k:(k + cluster_vali_num)], flag]= self.optimize_test(cluster_freq, num_time_in_cluster, train_return_cv, train_validation, weight)  
                
                if flag == 1:
                    return [return_array, flag]
                else:
                    k += cluster_vali_num
        return [return_array, flag]
    
    #use more detailed split 
    def cross_validation2(self, train_return, train_state, fold_num):
        right_theta_param = 0
        right_Rmin_param = 0
        #grid search
        ## Rmin_list = [Rmin_pre - 0.0005*(i) for i in range(100)]
        #empirical quantile search
        ##here, we apply EW portfolio to the overall train return and compute the lower quantile = 0.05 or 0.1
        train_array = np.array(train_return)
        (train_num, port_num) = train_array.shape
        #find the percentile = 40/20% for each time
        Rmin_list = np.array(train_array).reshape(train_num*port_num, )
        temp_Rmin = [np.percentile(Rmin_list, 5*i) for i in range(8, 0, -4)]
        self.Rmin_baseline1.append(temp_Rmin[0])
        self.Rmin_baseline2.append(temp_Rmin[1])
        #self.Rmin_baseline3.append(temp_Rmin[2])
        adj_id = 0
        adj_num = 5
        for serial_number in range(2, fold_num + 1):
            #print('For the serial id = ', serial_number)
            
            if self.hmm_type != -1:
                train_state = self.hmm_state_pretrain(train_return, self.cluster_number, hmm_type)
            
            Rmin_list = temp_Rmin[(adj_num*adj_id):(adj_num*(adj_id + 1))]
            [train_train, train_validation] = self.divide(train_return, fold_num, serial_number)
            
            #train_return_mean = np.mean(train_train, axis = 0)
            hmm_cv_info = self.hmm_train_cv(train_train, self.cluster_number, train_state, self.hmm_type, fold_num, serial_number)
            
            #for each fold we take the dictionary of {SharpeRatio, theta, Rmin} as the list element
            temp_SR = list()
            for i in range(len(self.theta_list)):
                
                self.theta = self.theta_list[i]
                infeasible_list = list()
                #print(Rmin_list)
                if self.mean_constr == False:
                    
                    [return_array, flag] = self.cross_validation2_test(hmm_cv_info, train_validation)
                    temp_SR.append({"Sharpe Ratio":np.mean(return_array)/np.std(return_array),
                                    "theta": self.theta,
                                    "Rmin": self.min_mean})

                else:
                    for Rmin_item in Rmin_list:
                        #print('For the case with Rmin = ', Rmin_item)
                        self.min_mean = Rmin_item
                        [return_array, flag] = self.cross_validation2_test(hmm_cv_info, train_validation)       
                        #the optimization did not reach its optimal solution
                        #then we delete that item
                        if flag == 1:
                            infeasible_list.append(Rmin_item)
                            if len(infeasible_list) == len(Rmin_list) and adj_id < int(len(temp_Rmin)/adj_num):
                                Rmin_list += temp_Rmin[(adj_num*(adj_id + 1)):(adj_num*(adj_id + 2))]
                                adj_id += 1
                        else:
                            #print('Success!')
                            temp_SR.append({"Sharpe Ratio":np.mean(return_array)/np.std(return_array),
                                            "theta": self.theta,
                                            "Rmin": Rmin_item})
                    #remove the infeasible items
                    Rmin_list = Rmin_list[len(infeasible_list):]
                #print('The left rmin are: ', Rmin_list)
            #find the max one in the Sharpe Ratio List
            #print(temp_SR)
            SR_base_index = 0
            SR_base = temp_SR[SR_base_index]["Sharpe Ratio"]
            for i in range(1, len(temp_SR)):
                if temp_SR[i]["Sharpe Ratio"] > SR_base:
                    SR_base = temp_SR[i]["Sharpe Ratio"]
                    SR_base_index = i
            right_theta_param += temp_SR[SR_base_index]["theta"]
            #self.theta_list[int(np.argmax(temp_SR)/len(Rmin_list))]
            right_Rmin_param += temp_SR[SR_base_index]["Rmin"]
            #Rmin_list[np.argmax(temp_SR)%len(Rmin_list)]
        #check whether we need EMA
        if self.previous_theta == 0:
            self.theta = right_theta_param/(fold_num - 1)
            self.min_mean = right_Rmin_param/(fold_num - 1)
        else:
            self.theta = self.ambiguity_param*self.previous_theta + (1 -  self.ambiguity_param)*right_theta_param/(fold_num - 1)
            self.min_mean = right_Rmin_param/(fold_num - 1)
            
        #print('Parameters for theta and mean are: ', self.theta, self.min_mean)
        self.hist_theta.append(self.theta)
        self.hist_rmin.append(self.min_mean)
    #days = 1
    def cross_validation2_upd(self, train_return, train_state, fold_num):
        train_array = np.array(train_return)
        (train_num, port_num) = train_array.shape
        Rmin_list = np.array(train_array).reshape(train_num*port_num, )
        temp_Rmin = [np.percentile(Rmin_list, 5*i) for i in range(8, 0, -4)]
        #initiation
        temp_SR = list()
        if self.mean_constr == False:
            for i, theta in enumerate(self.theta_list):
                temp_SR.append({"Sharpe Ratio":0, "theta":theta, "Rmin":temp_Rmin[0]})
        else:
            for i, theta in enumerate(self.theta_list):
                for j, Rmin_item in enumerate(Rmin_list):
                    temp_SR.append({"Sharpe Ratio":0, "theta":theta, "Rmin":Rmin_item})


        for serial_number in range(2, fold_num + 1):
            if self.hmm_type != -1:
                train_state = self.hmm_state_pretrain(train_return, self.cluster_number, hmm_type)
            
            Rmin_list = temp_Rmin
            [train_train, train_validation] = self.divide(train_return, fold_num, serial_number)
            
            #train_return_mean = np.mean(train_train, axis = 0)
            
            hmm_cv_info = self.hmm_train_cv(train_train, self.cluster_number, train_state, self.hmm_type, fold_num, serial_number)

            
            for i in range(len(self.theta_list)):
                
                self.theta = self.theta_list[i]
                print(self.theta)
                infeasible_list = list()
                #print(Rmin_list)
                if self.mean_constr == False:
                    
                    [return_array, flag] = self.cross_validation2_test(hmm_cv_info, train_validation)
                    temp_SR[i]["Sharpe Ratio"] += np.mean(return_array)/np.std(return_array)
                else:
                    for j, Rmin_item in enumerate(Rmin_list):
                        #print('For the case with Rmin = ', Rmin_item)
                        self.min_mean = Rmin_item
                        [return_array, flag] = self.cross_validation2_test(hmm_cv_info, train_validation)       
                        #print('Success!')
                        temp_SR[i*len(self.theta_list) + j]["Sharpe Ratio"] += np.mean(return_array)/np.std(return_array)
                    #remove the infeasible items
                    # Rmin_list = Rmin_list[len(infeasible_list):]
                #print('The left rmin are: ', Rmin_list)

        #find the max one in the Sharpe Ratio List
        #compare after computing all the slices
        SR_base_index = 0
        SR_base = temp_SR[SR_base_index]["Sharpe Ratio"]
        for i in range(1, len(temp_SR)):
            if temp_SR[i]["Sharpe Ratio"] > SR_base:
                SR_base = temp_SR[i]["Sharpe Ratio"]
                SR_base_index = i

        self.theta = temp_SR[SR_base_index]["theta"]
        #self.theta_list[int(np.argmax(temp_SR)/len(Rmin_list))]
        self.min_mean = temp_SR[SR_base_index]["Rmin"]
 
        #check whether we need EMA
        if self.previous_theta != 0:
            self.theta = self.ambiguity_param*self.previous_theta + (1 -  self.ambiguity_param)*self.theta

        self.hist_theta.append(self.theta)
        self.hist_rmin.append(self.min_mean)

    def rolling(self, cost_norm = 1):
        
        self.cost_norm = cost_norm
        #reduce to the standard parameter
        if self.theta == 0:
            self.theta_type = 0


        i = 0
        num_of_sample = len(self.df_select)
        num_of_train = len(self.df_train)
        dta_param = math.pow(num_of_train/self.coef, 1/self.portfolio_number)

        pre_info = 0
        self.hist_theta = []
        self.hist_rmin = []
        #create the weight array
        self.all_weight_pair = []
        #compute the maximum minimum return at each point.
        self.Rmin_pre = np.zeros(num_of_sample - num_of_train)
        #quantile of each history return with EW policy
        self.Rmin_baseline1 = []
        self.Rmin_baseline2 = []
        self.previous_theta = 0
        ##when cv_type == 2, we would only require the additional theta
        if self.cv_type == 2:
            df = pd.read_csv("../factor model/fcvar_hmm_theta_industryindices.csv")
            self.history_theta = list(df['theta'])
            
        while i < num_of_sample - num_of_train:
            print(i)
            #pre retrieve for the state and returns of training samples
            if self.hmm_type == -1:
                train_state = self.df_state[i:(i + num_of_train)]
            #set based on the results of interior samples
            else:
                train_state = None
            
            train_return = self.df_select[i: (i + num_of_train)]
            #print(len(train_return))
            train_return_mean = np.array(train_return.mean())
            if i + num_of_train + self.rolling_day < len(self.df_select):       
                test_return = np.array(self.df_select[i + num_of_train : i + num_of_train + self.rolling_day][0: self.portfolio_number])
            else:
                test_return = np.array(self.df_select[i + num_of_train : len(self.df_select)][0: self.portfolio_number])
            
            #remove the effect of weight and turnover part
            temp_turnover = self.turnover
            temp_weight = self.weight
            
            # hmm estimate
            ## num_time_in_cluster shows the history cluster and cluster_freq shows the transitin probability for the last data point
            [num_time_in_cluster, cluster_freq, train_return_full] = self.hmm_train(train_return, self.cluster_number, train_state, self.hmm_type)
            
            #self.Rmin_pre[i] = self.pre_cv_lp(num_time_in_cluster, cluster_freq, train_return_full, shortsale_sign)
            #cross-validation part: (default) set to be 4 folds (leave the first, the actual number is 3)
            ## require cross validation


            if self.cv_type >= 0:
                #use the out data
                if self.cv_type == 2:
                    if i == 0:
                        self.theta = self.history_theta[0]
                    else:
                        self.theta = self.ambiguity_param*self.history_theta[i - 1] + (1 - self.ambiguity_param)*self.history_theta
                
                else:
                    self.theta_list = [param/dta_param for param in self.theta_param] 
                    self.cross_validation2(train_return, train_state, fold_number)
                    #print(self.theta) 
            else:
                #fix parameter
                train_array = np.array(train_return)
                (train_num, port_num) = train_array.shape
                #find the percentile = 20% for each time
                Rmin_list = np.array(train_array).reshape(train_num*port_num, )        
                self.min_mean = np.percentile(Rmin_list, lower_percentile)
                if self.theta_type != 0:
                    self.theta = self.theta_type/dta_param

            self.turnover = temp_turnover

            [self.weight, self.return_array[i:i + self.rolling_day], flag] = self.optimize_test(cluster_freq, num_time_in_cluster, train_return_full, test_return, temp_weight)
            #print(self.weight)
            self.all_weight_pair.append(self.weight)
            self.weight = np.array(self.weight)
            self.previous_theta = self.theta
            i = i + self.rolling_day
        #return hist_return
    
