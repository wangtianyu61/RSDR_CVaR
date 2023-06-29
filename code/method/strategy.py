# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:08:21 2020

@author: wangt
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from CVaR_parameter import *
import matplotlib.pyplot as plt
import seaborn as sns


class strategy:
    rolling_day = 1
    turnover = 0
    method_name = "strategy"
    def __init__(self, df_select, df_train, rolling_day, portfolio_number):
        self.df_select = df_select
        self.df_train = df_train
        self.rolling_day = rolling_day
        self.portfolio_number = portfolio_number
        self.weight = np.zeros(self.portfolio_number)
        num_of_sample = len(df_select) - len(df_train)
        self.return_array = np.zeros(num_of_sample)
        self.column_name = list(df_select.columns)
        if 'Date' in self.column_name:
            self.column_name.remove('Date')
    #divide the history return into train and validation part
    ##serial shows the sequential id for that fold number
    def divide(self, train_return, fold_num, serial):
        #fold -  1, fold
        train_return = train_return[list(set(train_return.columns) - {"tag_cluster"})]
        train_return_array = np.array(train_return)
        (train_num, port_num) = train_return_array.shape
        #locate the start and end point for train-validation-split
        vali_start = int(train_num/fold_num)*(serial - 1)
        if serial < fold_num:
            vali_end = int(train_num/fold_num)*serial
        else:
        ##case of in the last section
            vali_end = train_num 
        train_validation = np.array(train_return[vali_start:vali_end])
        train_train = pd.concat([train_return[0:vali_start], train_return[vali_end:]])
        
        return [train_train, train_validation]
    
    #a virtual interface for the cross validation whose details are implemented in each method
    def cross_validation(self):
        pass
    #just set the minimum return constraint for cross validation
    def cross_validation2(self, train_return, fold_num):
        right_Rmin_param = 0
        train_array = np.array(train_return)
        (train_num, port_num) = train_array.shape
        Rmin_list = np.array(train_array).reshape(train_num*port_num, )
        Rmin_list = [np.percentile(Rmin_list, 5*i) for i in range(4, 0, -2)]
        
        weight = np.ones(self.portfolio_number)/self.portfolio_number
        cnt = 0
        for serial_number in range(2, fold_num + 1):
            [train_train, train_validation] = self.divide(train_return, fold_num, serial_number)

            #cross validation to obtain the parameter with the maximum out-of-sample Sharpe Ratio
            temp_SR = list()
            infeasible_list = list()
            for Rmin_item in Rmin_list:
                print('inner', serial_number, cnt)
                cnt += 1
                
                self.min_mean = Rmin_item
                [weight, return_array, flag] = self.optimize(train_train, train_validation, weight)
                if flag != 1:
                    temp_SR.append({"Sharpe Ratio": np.mean(return_array)/np.std(return_array),
                                    "Rmin": Rmin_item})
                else:
                    infeasible_list.append(Rmin_item)
            Rmin_list = Rmin_list[len(infeasible_list):]
            SR_base_index = 0
            SR_base = temp_SR[SR_base_index]["Sharpe Ratio"]
            for i in range(1, len(temp_SR)):
                if temp_SR[i]["Sharpe Ratio"] > SR_base:
                    SR_base = temp_SR[i]["Sharpe Ratio"]
                    SR_base_index = i
            right_Rmin_param += temp_SR[SR_base_index]["Rmin"]
        self.min_mean = right_Rmin_param/(fold_num - 1)
        
            
        print('Parameters for minimum return are: ', self.min_mean)
    def cross_validation3(self, train_return, fold_num):
        right_theta_param = 0
        right_Rmin_param = 0
        train_array = np.array(train_return)
        (train_num, port_num) = train_array.shape
        Rmin_list = np.array(train_array).reshape(train_num*port_num, )
        Rmin_list = [np.percentile(Rmin_list, 5*i) for i in range(8, 0, -4)]

        weight = np.ones(self.portfolio_number)/self.portfolio_number
        for serial_number in range(2, fold_num + 1):
            [train_train, train_validation] = self.divide(train_return, fold_num, serial_number)
            #cross validation to obtain the parameter with the maximum out-of-sample Sharpe Ratio
            temp_SR = list()
            for i in range(len(self.theta_list)):
                self.theta = self.theta_list[i]
                if self.mean_constr == False:
                    [weight, return_array, flag] = self.optimize(train_train, train_validation, weight)
                    temp_SR.append({"Sharpe Ratio": np.mean(return_array)/np.std(return_array),
                                    "theta": self.theta})
                else:
                    infeasible_list = []
                    for Rmin_item in Rmin_list:
                        self.min_mean = Rmin_item
                        [weight, return_array, flag] = self.optimize(train_train, train_validation, weight)
                        if flag != 1:
                            temp_SR.append({"Sharpe Ratio": np.mean(return_array)/np.std(return_array),
                                            "theta": self.theta,
                                            "Rmin": Rmin_item})
                        else:
                            infeasible_list.append(Rmin_item)
                    Rmin_list = Rmin_list[len(infeasible_list):]
            SR_base_index = 0
            SR_base = temp_SR[SR_base_index]["Sharpe Ratio"]
            for i in range(1, len(temp_SR)):
                if temp_SR[i]["Sharpe Ratio"] > SR_base:
                    SR_base = temp_SR[i]["Sharpe Ratio"]
                    SR_base_index = i
            right_theta_param += temp_SR[SR_base_index]["theta"]
            if self.mean_constr == True:
                right_Rmin_param += temp_SR[SR_base_index]["Rmin"]

            
        self.theta = right_theta_param/(fold_num - 1)
        if self.mean_constr == True:    
            self.min_mean = right_Rmin_param/(fold_num - 1)
        
    def show_return(self, test_return, weight):
        (num_of_sample,num) = test_return.shape
        return_list = np.zeros(num_of_sample)
        for i in range(num_of_sample):
            return_list[i] = np.dot(test_return[i],weight)#the next time point
            weight = np.multiply(test_return[i]/unit + 1, weight)
            weight = weight/np.sum(weight) #normalization
        return [return_list,weight]
    
    def finish_flag(self, method_list, return_list):
        method_list.append(self.method_name)
        print(self.return_array)
        return_list.append(self.return_array)
        print("Finish "+ self.method_name + " policy!")
        return [method_list, return_list]

    #weight pair illustration from 2008/01/01 to 2009/12/31 for IndustryIndices.csv
    def weight_pair_illustration(self, title_name):
        #sns.set_style('darkgrid')
        plt.rc('font',family='Times New Roman')

        #0-9 equities; 10 means 10-yr treasury note; 11 - 13 Commodities; 14 IG; 15 TIPS
        select_weight_pair = self.all_weight_pair[0:24]
        time_length = len(select_weight_pair)
        equities = [sum(weight_item[0:10]) for weight_item in select_weight_pair] #0-9
        bond = [weight_item[10] for weight_item in select_weight_pair] #10
        TIPS = [weight_item[15] for weight_item in select_weight_pair] #15
        Gold = [weight_item[11] for weight_item in select_weight_pair] #11
        Other_Commodities = [sum(weight_item[12:14]) for weight_item in select_weight_pair] #12-13
        Credit = [weight_item[14] for weight_item in select_weight_pair] #14

        true_index = [0, 4, 8, 12, 16, 20]
        xtickslabel = [200801, 200805, 200809, 200901, 200905, 200909]
        plt.figure(figsize = (10,6), dpi = 500)
        plt.plot(range(time_length), equities, label = "equities", marker = '.', linestyle = '-', markersize = 4)
        plt.plot(range(time_length), bond, label = "bond", marker = '.', linestyle = ':', markersize = 4)
        plt.plot(range(time_length), TIPS, label = "TIPS", marker = 'o', linestyle = '-', markersize = 4)
        plt.plot(range(time_length), Gold, label = "Gold", marker = '*', linestyle = '-', markersize = 4)
        plt.plot(range(time_length), Other_Commodities, label = "Other_Commodities", marker = 'o', linestyle = ':', markersize = 4)
        plt.plot(range(time_length), Credit, label = "Credit", marker = '*', linestyle = ':', markersize = 4)
        plt.xlabel("TimeLine")
        plt.ylabel("Portfolio Weights")
        plt.legend(loc = 'upper right')
        plt.title("Asset Allocation during the 2008-2009 recession")
        plt.xticks(true_index, xtickslabel, size = 12)
        plt.savefig('../figures_new/recession_weight2/' + title_name + '.pdf')
        
        plt.show()
    
class strategy_cluster:
    def factor_return_cluster(self, df_train, factor_data, column_name, cluster_number):
        if cluster_type == "KMeans":
            clf = KMeans(n_clusters = cluster_number, random_state = 0, algorithm = 'auto')
            clf = clf.fit(factor_data)
            factor_data['tag_cluster'] = clf.labels_
#        #print(clf.cluster_centers_)
            df_train['tag_cluster'] = clf.labels_
            
        # GMM Algorithms
        elif cluster_type == "GMM":
            gmm = GaussianMixture(n_components = cluster_number, random_state = 0)
            gmm.fit(factor_data)
            
            factor_data['tag_cluster'] = gmm.predict(factor_data)
            df_train['tag_cluster'] = factor_data['tag_cluster']
            
        
        grouped = df_train.groupby(df_train['tag_cluster'])
        factor_group = factor_data.groupby(factor_data['tag_cluster'])
        ## frequence
        ## mean and covariance
        mean_info = grouped.mean()
        cov_info = grouped.cov()
        factor_center = factor_group.mean()
        return [factor_center, mean_info, cov_info]
    
    # compute the hmm model frequency
    ## hmm learning
    def hmm_fit(self, df_train, side_info):
        pass
    #compute the frequency of the transition matrix
    def hmm_state_compute(self, df_train, cluster_number):
        #the initial transition matrix
        transition_matrix = np.zeros((cluster_number, cluster_number))
        state = np.zeros(cluster_number)
        train_tag_cluster = list(df_train['tag_cluster'])
        #compute the frequency 
        for i in range(len(df_train) - 1):
            transition_matrix[train_tag_cluster[i]][train_tag_cluster[i + 1]] += 1
            state[train_tag_cluster[i]] += 1
            
        #normalization
        for i in range(cluster_number):
            transition_matrix[i] = transition_matrix[i]/state[i]
        #print(transition_matrix)
        #just the previous month beforehand
        last_state = train_tag_cluster[len(df_train) - 1]
        return transition_matrix[last_state]
    
    def factor_cluster(self,df_train,factor_data,column_name,cluster_number, hmm_state_estimate = False):
    #the process and result of the clustering
        cluster_freq = np.zeros(cluster_number)
        
        # KMeans Algorithms
        if cluster_type == "KMeans":
            clf = KMeans(n_clusters = cluster_number, random_state = 0, algorithm = 'auto')
            clf = clf.fit(factor_data)
#           
#        #print(clf.cluster_centers_)
            df_train['tag_cluster'] = clf.labels_
            
        # GMM Algorithms
        elif cluster_type == "GMM":
            gmm = GaussianMixture(n_components = cluster_number, random_state = 0)
            gmm.fit(factor_data)
        
            df_train['tag_cluster'] = gmm.predict(factor_data)
        
        grouped = df_train[column_name].groupby(df_train['tag_cluster'])
        
        ## frequence
        countall = len(df_train)
        counter = grouped.count()
        for index in range(cluster_number):
            cluster_freq[index] = counter.iloc[index,0]/countall

        ## mean and covariance
        mean_info = grouped.mean()
        cov_info = grouped.cov()
        
        #keep the same as the original model
        if hmm_state_estimate == False:
            return [cluster_freq, mean_info, cov_info]
        #change to a new estimate
        else:
            #frequency estimate for the transition matrix
            cluster_freq = self.hmm_state_compute(df_train, cluster_number)
            return [cluster_freq, mean_info, cov_info]
    
    def return_cluster(self,df_train,column_name,cluster_number, hmm_state_estimate = False):
    
        cluster_freq = np.zeros(cluster_number)
        portfolio_data = np.array(df_train[column_name])
        #choose the data to classify
        #the process and the result of the clustering 
        # KMeans Algorithms
        if cluster_type == "KMeans":
            clf = KMeans(n_clusters = cluster_number, random_state = 0, algorithm = 'auto')
            clf = clf.fit(df_train)
#        
#        
#        #print(clf.cluster_centers_)
            df_train['tag_cluster'] = clf.labels_
        
        # GMM Algorithms
        elif cluster_type == "GMM":
            gmm = GaussianMixture(n_components = cluster_number, random_state = 0)
            gmm.fit(df_train)
        
            df_train['tag_cluster'] = gmm.predict(df_train)
        #get the information of each cluster
        grouped = df_train[column_name].groupby(df_train['tag_cluster'])

    ## frequence
        countall = len(df_train)
        counter = grouped.count()
        for index in range(cluster_number):
            cluster_freq[index] = counter.iloc[index,0]/countall
            #print(cluster_freq)

        ## mean and covariance
        mean_info = grouped.mean()
        #print(mean_info)
        #print(type(mean_info.iloc[0]))
        cov_info = grouped.cov()
        
        
        #keep the same as the original model
        if hmm_state_estimate == False:
            return [cluster_freq, mean_info, cov_info]
        #change to a new estimate
        else:
            #frequency estimate for the transition matrix
            cluster_freq = self.hmm_state_compute(df_train, cluster_number)
            return [cluster_freq, mean_info, cov_info]
    def hmm_estimate(self, train_return, hmm_type):
        train_return_array = np.array(train_return)
        (train_num, port_num) = train_return_array.shape
        if hmm_type != 1:
             #find the one with the highest return and name its index
             if hmm_type == 2:
                 return [np.argmax(train_return_array, axis = 1), train_num]
            #choose among 5 indices
             else:
                 #the particular data for the current slice
                 equities = np.mean(train_return_array[:,0:10], axis = 1)
                 treasury_note = np.mean(train_return_array[:,10:11], axis = 1)
                 TIPS = np.mean(train_return_array[:,15:16], axis = 1)
                 commodities = np.mean(train_return_array[:,11:14], axis = 1)
                 credit = np.mean(train_return_array[:,14:15], axis = 1)
                 indices_list = [equities, treasury_note, TIPS, commodities, credit]
                 train_return_variant = np.zeros((train_num, 5))
                 for index in range(len(indices_list)):
                     train_return_variant[:, index] = indices_list[index].T
                     
                 return [np.argmax(train_return_variant, axis = 1), train_num]
        else:
            return [train_return_array, train_num]
    #determine the state in the cross validation part if hmm_type!=-1
    def hmm_state_pretrain(self, train_return, regime_num, hmm_type):
        if hmm_type == -1:
            return
        else:
            [df_seq, train_num] = self.hmm_estimate(train_return, hmm_type)
            if hmm_type != 1:
                remodel = hmm.MultinomialHMM(n_components = regime_num, n_iter = 100)            
            else:
                remodel = hmm.GaussianHMM(n_components = regime_num, n_iter = 100)
            dfseq = df_seq.reshape(-1, 1)    
            #hmm_data = np.array(df[df.columns[1:]])
            remodel.fit(dfseq)
            time_state = np.array(list(remodel.predict(dfseq)))
            #in accordance with the result in state.csv currently
            return time_state + 1

    #for cross validation use 
    def hmm_train_cv(self, train_return, regime_num, time_state, hmm_type, fold_num, serial):
        #list all the possible state value given the time series
        possible_states = [i - 1 for i in list(set(time_state))]
        train_return = np.array(train_return)
        #locate the start and end point for the state
        train_return_array = np.array(train_return)
        (train_num, port_num) = train_return_array.shape
        ##rescale
        train_num = train_num*(fold_num)/(fold_num - 1)
        vali_start = int(train_num/fold_num)*(serial - 1)
        if serial < fold_num:
            vali_end = int(train_num/fold_num)*serial
        else:
        ##case of in the last section
            vali_end = int(train_num) 
        #handle the state condition, concat the formal and latter timestamps
        state_list = list(time_state)
        ##cross validation in the rolling step
        hmm_cv_info = {}
        ## create the last state series
        
        hmm_cv_info['last_state'] = [state_list[i - 1] - 1 for i in range(vali_start, vali_end)]
        
        state_list_formal = state_list[0:vali_start]
        #print('validation end is ', vali_end, 'And the length of state is ', len(state_list))
        state_list_formal.extend(state_list[vali_end:])
        time_state = state_list_formal
        #print('the length of time state is ', len(time_state))
        #delete the link point if have one
        
        time_state = np.array([time_element - 1 for time_element in time_state])
        for last_state in possible_states:
            cluster_freq = np.zeros(regime_num)
            train_return_regime = [[] for i in range(regime_num)]
            #show the state of the last date of the train return
            #print(len(time_state), last_state, vali_start, vali_end)
            for i in range(len(time_state) - 1):
                
                if time_state[i] == last_state and i!= vali_start:
                    #print(time_state[i + 1] - 1)
                    cluster_freq[time_state[i + 1]] = cluster_freq[time_state[i + 1]] + 1
            
            
            num_time_in_cluster = [len(time_state[time_state == i]) for i in range(regime_num)]
            
            #print(cluster_freq)
            for tid in range(len(time_state)):
                #add each time into the regime    
                #print(time_state[tid], type(time_state[tid]))
                train_return_regime[time_state[tid]].append(train_return[tid])
            
           
            ##corner case when cluster_freq has no num
            if sum(cluster_freq) == 0:
                for i in range(len(cluster_freq)):
                    if len(train_return_regime[i]) > 0:
                        cluster_freq[i] = 1
            cluster_freq = cluster_freq/sum(cluster_freq)
            hmm_cv_info[last_state] = [np.array(num_time_in_cluster), cluster_freq, train_return_regime]
        return hmm_cv_info
        
    def hmm_train(self, train_return, regime_num, time_state, hmm_type):
        #revise the train return into different dimensions: cluster_number * cls_index * portfolio_number
        ## hmm_type == 1 means the GMM mixture
        ## hmm_type == 2 means among N portfolios, we treat them equally
        ## hmm_type == 3 means among 5 indices of the N portfolios, we treat them equally 
        ## hmm_type == -1 means just using the state is enough

        train_return_regime = [[] for i in range(regime_num)]
        #print(train_return.shape)
        #the df state is enough
        if hmm_type == -1:
            last_state = time_state[len(time_state) - 1] - 1
            cluster_freq = np.zeros(regime_num)
            time_state = np.array([time_element - 1 for time_element in time_state])
            #show the state of the last date of the train return
        
            for i in range(len(time_state) - 1):
                if time_state[i] == last_state:
                    #print(time_state[i + 1] - 1)
                    cluster_freq[time_state[i + 1]] = cluster_freq[time_state[i + 1]] + 1
            #print(cluster_freq)
            cluster_freq = cluster_freq/sum(cluster_freq)
            #print(cluster_freq)
        else:
            [df_seq, train_num] = self.hmm_estimate(train_return, hmm_type)
            if hmm_type != 1:
                remodel = hmm.MultinomialHMM(n_components = regime_num, n_iter = 100)            
                dfseq = df_seq.reshape(-1, 1)
            else:
                remodel = hmm.GaussianHMM(n_components = regime_num, n_iter = 100)
                dfseq = df_seq    
            #hmm_data = np.array(df[df.columns[1:]])
            remodel.fit(dfseq)
            print(dfseq)
            time_state = np.array(list(remodel.predict(dfseq)))
            #shows the in the last training data point
            cluster_freq = remodel.transmat_[time_state[train_num - 1]]
            #print(type(remodel.predict(df_seq)))
        num_time_in_cluster = [len(time_state[time_state == i]) for i in range(regime_num)]
        train_return = np.array(train_return)
        #print(time_state, len(time_state))
        for tid in range(len(time_state)):
            #add each time into the regime    
            #print(time_state[tid], type(time_state[tid]))
            train_return_regime[time_state[tid]].append(train_return[tid])
            #give an illustration of how many time points in each regime
            #print(time_state, np.sum(time_state[time_state == 0]), np.sum(time_state[time_state == 1]))
        
        
        
        
        return [np.array(num_time_in_cluster), cluster_freq, train_return_regime] 

#    def cluster_output(self,df_train, column_name, cluster_number, cluster_sign):
#        if cluster_sign == 0:
#            dataname = "../cluster_tag/trainset_" + str(len(column_name)) + "_" + str(cluster_number) + " clusters.csv" 
#        elif cluster_sign == 1:
#            dataname = "../cluster_tag/trainset_" + str(len(column_name)) + "_" + str(cluster_number) + " clusters (3 factor).csv"
#        else:
#            dataname = "../cluster_tag/trainset_" + str(len(column_name)) + "_" + str(cluster_number) + " clusters (5 factor).csv"
#        df_train.to_csv(dataname) 
#    
#
#    def cluster_convert(self,cluster_number, portfolio_number, mean_info, cov_info):
#        cluster_mean = list()
#        cluster_covariance = list()
#        for i in range(cluster_number):
#            cluster_mean.append(tuple(np.array(mean_info)[i]))
#            cluster_covariance.append(np.array(cov_info)[portfolio_number*i: portfolio_number*(i+1)])
#        return [cluster_mean, cluster_covariance]
        