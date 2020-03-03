# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:41:46 2019

@author: User
"""


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import pickle
import time

def weekday_time_slot5(dt):
    x = dt.hour
    
    if x >= 7 and x < 10:
        return 0
    elif x >= 10 and x < 17:
        return 1
    elif x >= 17 and x < 20:
        return 2
    elif x >= 20 and x < 24:
        return 3
    else :
        return 4
  

def weekend_time_slot5(dt):
    x = dt.hour
    
    if x >= 5 and x < 10:
        return 0
    elif x >= 10 and x < 14:
        return 1
    elif x >= 14 and x < 20:
        return 2
    elif x >= 20 and x < 24:
        return 3
    else :
        return 4


def season_divide(x):
    if x == 12 or x < 3:
        return 0
    else :
        return 1




#%%
    
# cluster_save_path, w_str, slot, K1, K2, clus_method, cent_method, ok_bike_list

class new_clustering:

    def __init__(self, cluster_save_path, w_str, ok_bike_list, slot):
        self.cluster_save_path = cluster_save_path
        self.w_str = w_str
        self.ok_bike_list = ok_bike_list
        self.slot = slot
        
        """
        알고리즘
        1. 대여(또는 대여와 반납 패턴 기준) 패턴을 기준으로 clustering 함.
        2. 각 클러스터에서 위도와 경도에 따라 클러스터를 정하고 클러스터링 함.
        """
        default_path = os.getcwd()
        bike_path = os.path.join(default_path, 'data\\')
        self.bike_path = bike_path
        
        slot = self.slot
        w_str = self.w_str
        
        # 자전거 수요 관련 파일
        if w_str == 'weekday':
            with open(bike_path+'weekday_lent_data.pickle', 'rb') as f:
                df1 = pickle.load(f)
        elif w_str == 'weekend':
            with open(bike_path+'weekend_lent_data.pickle', 'rb') as f:
                df1 = pickle.load(f)
        
        station_info = pd.read_csv(bike_path+'station_info.csv')
        station_info = station_info[station_info['sta_num'].isin(self.ok_bike_list)].reset_index(drop=True)
        station_info = station_info.drop('hold_num', axis=1)
        station_info.columns = ['sta_num','latitude','longitude']
        
        station = station_info.copy()
        station = station.reset_index().drop('index', axis=1)
        
            # clustering
            # cluster와 비교 시 mean값을 기준으로 비교하게 됨.
                # 1. week에 따른 평일 또는 주말 값 계산
                
                # 평일 : 23 ~ 6, 7~ 9, 10~ 16, 17 ~ 19, 20 ~ 22, 
                # 주말 : 0 ~ 5, 6 ~ 17, 18 ~ 23
            
        df1['hour'] = df1['lent_date'].apply(lambda x : x.hour)
        if slot == 5:
            if w_str == 'weekday':
                df1['slot'] = df1['lent_date'].apply(lambda x : weekday_time_slot5(x))
            elif w_str == 'weekend':
                df1['slot'] = df1['lent_date'].apply(lambda x : weekend_time_slot5(x))
            
        elif slot == 24:
            df1['slot'] = df1['hour']
        
    
        df1['month'] = df1['lent_date'].apply(lambda x : x.month)
        df1['month'] = df1['month'].apply(lambda x : season_divide(x))
        df1['month'] = df1['month']*slot
        
        df1['slot'] = df1['slot'] + df1['month']
        df1 = df1.drop('month', axis=1)
                
        lent_df = df1.groupby(['lent_sta','slot']).sum().reset_index()[['lent_sta','slot','value']]
        return_df = df1.groupby(['return_sta','slot']).sum().reset_index()[['return_sta','slot','value']]
        
        self.lent_df = lent_df
        self.return_df = return_df
        self.station = station
    
    

    
    def k_moid_clustering(self, k_, final_pattern_df_key__, p_dis_mat_):
        
        vj_list = []
        
        p_dis_mat_v_ = p_dis_mat_.values
        for m in range(len(p_dis_mat_v_)):
            # dij에서 j를 구하려는 목적.
            d_j = p_dis_mat_v_[:,m]
            vj = 0
            
            for n in range(len(d_j)):
                dij = d_j[n]
                sum_dil = np.sum(p_dis_mat_v_[n,:])
                vj += dij / sum_dil
            vj_list.append(vj)
        
            # Step 1-3. 
        vj_df = pd.DataFrame({'sta_num':final_pattern_df_key__,'vj':vj_list})
        vj_df = vj_df.sort_values('vj').reset_index(drop=True)
        
        cent_medoids_sta = vj_df.iloc[:k_,0].values
        
        cent_df = self.make_cent_df(final_pattern_df_key__, cent_medoids_sta)    
        h_cluster_df_ = self.make_h_cluster_df(cent_df, p_dis_mat_, final_pattern_df_key__)
        
        past_cent_medoids = cent_medoids_sta.copy()
        count = 0
        
        # 무한 반복될 때 체크하는 변수들
        past_h_cluster_df_ = pd.DataFrame({'clus_num':np.arange(len(p_dis_mat_))})
        past_check_same_value = 0
        check_same_value = 0
        loop_num = 0
        loop_list = []
        first_value = 0
        check_interval_value = 0
        
        
        while True:
            cent_medoids_list = []
            
            # new centmoid 계산  
            for clus_n in np.unique(h_cluster_df_['clus_num']):
                t_sta_list = h_cluster_df_[h_cluster_df_['clus_num'] == clus_n]['sta_num'].tolist()
                t_dist_mat = p_dis_mat_.loc[t_sta_list, t_sta_list]
                
                t_dist_mat_key = t_dist_mat.columns
                t_dist_mat_v = t_dist_mat.values
                
                min_ind = np.argmin(np.sum(t_dist_mat_v, axis=1))
                min_sta_n = t_dist_mat_key[min_ind]
                cent_medoids_list.append(min_sta_n)
                
                # Step 1-5. 거리 구하기
            
            if sum(~(past_cent_medoids == np.array(cent_medoids_list))) == 0:
                break
            
            past_cent_medoids = np.array(cent_medoids_list)
        
            cent_df = self.make_cent_df(final_pattern_df_key__, np.array(cent_medoids_list))
            h_cluster_df_ = self.make_h_cluster_df(cent_df, p_dis_mat_, final_pattern_df_key__)
            
            # 무한 루프 빠지는지 검사 구간. 만약 일정 횟수 이상 같은 숫자 반복시 loop 탈출
            check_same_value = sum(h_cluster_df_['clus_num'].values == past_h_cluster_df_['clus_num'].values)
                
            if past_check_same_value == check_same_value:
                loop_num +=1
    
            if loop_num >= 5:
                break
            count +=1
            
            if count >= 100:
                """
                무한 루프에 빠지는지 검사함.
                1. 위의 코드에서 5번이상 같은 클러스터 경우가 나오면 타룿ㄹ
                2. 간격을 조사함.
                   - 과거 클러스터 값을 가져옴.
                   - first_value : interval 비교를 위해 이용할 변수
                   - 만약 loop_array에서 같은 수가 반복되는지 체크함
                   - 같은 수가 2번이상 나왔을 경우 interval을 체크함
                   - 만약 같은수가 5번 이상 나오면 : first_value의 과거와 현재 숫자 반복 간격이 같은지 확인하고 같으면 이를 루프로 간주함 -> 2개 이상 반복 시 이를 루프로 간주함.
                """
                
                past_h_cluster_df_ = h_cluster_df_.copy()
                past_check_same_value = check_same_value   
                
                # loop 검사
                if first_value == 0:
                    first_value = check_same_value
                
                loop_list.append(check_same_value)
                loop_array = np.array(loop_list)
                
                if len(np.where(loop_array == first_value)[0]) == 2:
                    check_interval_value = np.where(loop_array == first_value)[0][1] - np.where(loop_array == first_value)[0][0]
                    
                if len(np.where(loop_array == first_value)[0]) >= 5:
                    check_target_interval_list = np.where(loop_array == first_value)[0]
                    pattern_check_num = 0
                    for i in range(len(check_target_interval_list)-1):
                        past_v = check_target_interval_list[i]
                        next_v = check_target_interval_list[i+1]
                        if (next_v - past_v) == check_interval_value:
                            pattern_check_num +=1
                        
                    if pattern_check_num >=2:
                        break

        cluster_df_ = h_cluster_df_.copy()
        cluster_df_['f_clus_num'] = [1000+i for i in range(len(cluster_df_))]
        
        return cluster_df_

    
#%%    

    def extract_row(self, lent_df_, return_df_, sta_list_, slot_num_):
        """
        sta_list_를 기준으로 클러스터에 이용할 matrix 추출함.
        """
        
        arr_list = []
        
        for num in range(2):
            t_slot_list = [s_n for s_n in range(num*slot_num_, (num+1)*slot_num_)]
            slot_df_ = pd.DataFrame({'slot': t_slot_list}, columns=['slot'])
            lent_df_.columns = ['lent_sta','slot','l_value']
            return_df_.columns = ['return_sta','slot','r_value']
            
            sloted_lent_df_ = lent_df_[lent_df_['slot'].isin(t_slot_list)]
            sloted_return_df_ = return_df_[return_df_['slot'].isin(t_slot_list)]
            
            final_arr = np.array([]).reshape((-1,slot_num_*2))
            for i in range(len(sta_list_)):
                sta_n = sta_list_[i]
                t_lent_df_ = sloted_lent_df_[sloted_lent_df_['lent_sta'] == sta_n]
                t_return_df_ = sloted_return_df_[sloted_return_df_['return_sta'] == sta_n]
                
                l_value_list = (t_lent_df_['l_value'] / sum(t_lent_df_['l_value'])).tolist()
                r_value_list = (t_return_df_['r_value'] / sum(t_return_df_['r_value'])).tolist()
                
                t_lent_df_ = t_lent_df_.drop('l_value', axis=1)
                t_return_df_ = t_return_df_.drop('r_value', axis=1)
    
                t_lent_df_['l_value'] = l_value_list
                t_return_df_['r_value'] = r_value_list
                
            
            # 반납 시간이 1개라도 빈 경우 : 0을 넣어줌.
                
                ok_df_ = pd.merge(slot_df_, t_lent_df_[['slot','l_value']], how='left', on='slot')
                ok_df_ = pd.merge(ok_df_, t_return_df_[['slot','r_value']], how='left', on='slot')
                ok_df_ = ok_df_.fillna(0)
    
                final_arr = np.r_[final_arr, np.r_[ok_df_['l_value'].values, ok_df_['r_value'].values].reshape((-1,slot_num_*2))]
            
            arr_list.append(final_arr)
            
        real_final_arr = np.c_[arr_list[0], arr_list[1]]
        return real_final_arr, sta_list_
        
    
    
    def check_size(self, arr_, clus_dist_list__, K2__):
        
        # 만약 0인 원소가 있다면 그 값에 1을 넣어줌.
        for i in range(len(arr_)):
            target_clus_size = arr_[i]
            if target_clus_size == 0:
                max_ind = np.argmax(arr_)
                max_value = arr_[max_ind]
                arr_[max_ind] = max_value - 1
                arr_[i] = arr_[i]+1
    
        if sum(arr_) != K2__:
        
            if sum(arr_) > K2__:
                
                """
                값이 크면 줄이고 값이 작으면 늘리기 : 거리 비례
                ratio_arr : 클러스터에 포함된 개수 / 목표 클러스터 개수
                
                1. 현재 클러스터 개수 > 목표 개수 :
                     ㅇ 클러스터 개수를 줄여야 함
                       -> ratio_arr이 제일 작은 값을 늘려줌 => 이는 비율값을 맞추기 위함.
                          => 이 경우, arr_에서 1개를 감소시킴.
                       -> 단, 클러스터 개수가 1개인 곳은 제외 시킬 것!!
                
                2. 현재 클러스터 개수 < 목표 개수 :
                     ㅇ 클러스터 개수를 늘려야 함
                       -> ratio_arr이 제일 큰 값을 줄여줌 => 이는 비율값을 맞추기 위함.
                          => 이 경우, arr_에서 1개를 늘림.              
                """           
                
                while sum(arr_) != K2__ :
                    ratio_arr_ = np.array(clus_dist_list__) / arr_
                    one_clus_ind_ = np.where(arr_ == 1)[0]
                    
                    # ratio 값에 큰값을 주어서 제외시킴.
                    ratio_arr_[one_clus_ind_] = 10000
                    
                    # 제일 작은 값 구하기
                    min_ind_ = np.argmin(ratio_arr_)
                    arr_[min_ind_] = arr_[min_ind_] - 1
            
            
            elif sum(arr_) < K2__:
                while sum(arr_) != K2__ :
                    ratio_arr_ = np.array(clus_dist_list__) / arr_
                    
                    max_ind_ = np.argmax(ratio_arr_)
                    arr_[max_ind_] = arr_[max_ind_] + 1     
    
        return arr_
        
    
       
    def cluster_num_by(self, clus_dist_list_, K2_):
        clus_dist_sum_ = sum(clus_dist_list_)
        each_clus_size_list = []
               
        for i in range(len(clus_dist_list_)):
            dist_n = clus_dist_list_[i]
            clus_size = np.round((dist_n / clus_dist_sum_)*K2_,0)
            each_clus_size_list.append(clus_size)
        
        each_clus_size_arr = np.array(each_clus_size_list)
        
        each_clus_size_arr = self.check_size(each_clus_size_arr, clus_dist_list_, K2_)
        
        return each_clus_size_arr
    
    
    
    def make_dist_matrix(self, final_pattern_df_):
        """
        index가 정류소, column이 속성인 DataFrame을 기준으로 거리 matrix를 추출함.
        
        ** input
        final_pattern_df_ : index가 정류소 column이 속성인 DataFrame
        
        ** output
        p_dis_mat_ : index가 정류소인 값을 기준으로 distance를 추출함.
        
        """
        pattern_df_key_ = final_pattern_df_.index.values
        pattern_mat_ = final_pattern_df_.values
        
        # distance matrix 만들기.
        p_dis_mat_ = pd.DataFrame()
        
        for i in range(len(pattern_df_key_)):
            target_vector_ = pattern_mat_[i,:]
            target_dis = np.sqrt(np.sum(np.power(pattern_mat_ - target_vector_,2), axis=1))
            target_s_num = pattern_df_key_[i]
            p_dis_mat_.loc[:,target_s_num] = target_dis
        
        p_dis_mat_.index = pattern_df_key_
        
        return p_dis_mat_, pattern_df_key_
        
    
    
    def make_cent_df(self, final_pattern_df_key_, cent_medoids_sta_):
        """
        pattern_df의 key값인 col의 ind_num과 sta_num 값을 겹치게 만드는 DataFrame 형성함
        
        ** input
        final_pattern_df_key_ :  pattern_df의 key값
        cent_medoids_sta_ : cent medoids station list(like array)
        
        ** output
        cent_df_ : key_값에 해당하는 index num과 station num을 가진 DataFrame
        """
        
        cent_df_ = pd.DataFrame({'sta_num':[], 'ind_num':[]})
        for sta_num in cent_medoids_sta_:
            cent_meo_ind = np.where(final_pattern_df_key_ == sta_num)[0][0]
            cent_df_ = cent_df_.append({'sta_num':sta_num, 'ind_num':cent_meo_ind}, ignore_index=True)
        
        cent_df_['clus_num'] = [i+10000 for i in np.arange(len(cent_df_))]
        
        return cent_df_
        
    
    
    def make_h_cluster_df(self, cent_df_, p_dis_mat_, final_pattern_df_key_):
        """
        centmoid 값을 기준으로 cluster를 만듦.
        
        ** input
        cent_df_ : centmoid와 sta_num, clus_num을 가진 DataFrame
        p_dis_mat_ : 정류소 별로 dist 값을 가진 dataFrame(symmetric)
        final_pattern_df_key_ : key값
        
        ** output
        h_cluster_df : 각 정류소에 centmoid 값을 기준으로 클러스터 할당한 DataFrame
        """
        h_cluster_df = pd.DataFrame({'sta_num':final_pattern_df_key_, 'clus_num':[0]*len(final_pattern_df_key_)})
        
        cent_list = []
        
        for sta_num in cent_df_['sta_num'].tolist():
            dis_vector = p_dis_mat_[sta_num].values
            cent_list.append(dis_vector)
        cent_array = np.array(cent_list)
        clus_list = np.argmin(cent_array, axis=0)
        h_cluster_df['clus_num'] = clus_list+10000
        
        return h_cluster_df
    
    
    
    def find_density_cent_value(self, k_, X_):
    
        thres = 0.75*(len(X_)/k_)
        
        if int(thres) % 2 != 0:
            thres = int(thres) - 1
        if thres < 1:
            thres = 1
        n,m = X_.shape
        
        # step1
        m_set_list = []
        
        # step 2
        dis_mat, key = self.make_dist_matrix(X_)
        max_v = np.max(np.max(dis_mat))
        
            # 대각선 값
        np.fill_diagonal(dis_mat.values, max_v*100)
        
        for i in range(k_):
            t_m_set_list = []
            
            while len(t_m_set_list) < int(thres):
                # step 3
                row, col = dis_mat.stack().index[np.argmin(dis_mat.values)]
                for r in [row,col]:
                    t_m_set_list.append(r)
                dis_mat = dis_mat.drop([row,col], axis=0)
                dis_mat = dis_mat.drop([row,col], axis=1)
                
            
            m_set_list.append(t_m_set_list)
        
        point_arr = np.array([]).reshape((-1,m))
        for i in range(len(m_set_list)):
            temp_arr = np.array([]).reshape((-1,m))
            
            for sta in m_set_list[i]:
                temp_arr = np.concatenate([temp_arr,X_.loc[sta,:].values.reshape((1,m))])
        
            point_arr = np.concatenate([point_arr, np.mean(temp_arr, axis=0).reshape((1,m))])
        
        return point_arr
    
    
    def find_median_cent_value(self, k_, final_pattern_df_key__, p_dis_mat_, cent_option=False):
        vj_list = []
        
        p_dis_mat_v_ = p_dis_mat_.values
        for m in range(len(p_dis_mat_v_)):
            # dij에서 j를 구하려는 목적.
            d_j = p_dis_mat_v_[:,m]
            vj = 0
            
            for n in range(len(d_j)):
                dij = d_j[n]
                sum_dil = np.sum(p_dis_mat_v_[n,:])
                vj += dij / sum_dil
            vj_list.append(vj)
        
            # Step 1-3. 
        vj_df = pd.DataFrame({'sta_num':final_pattern_df_key__,'vj':vj_list})
        vj_df = vj_df.sort_values('vj').reset_index(drop=True)
        
        if cent_option == True:
            vj_df_len = int(len(vj_df)/2)
            cent_medoids_sta = vj_df.iloc[vj_df_len - k_: vj_df_len + k_,0].values
        
        else : 
            cent_medoids_sta = vj_df.iloc[:k_,0].values
        
        return cent_medoids_sta
    
                

#%%    
    def fit(self, clus_method, cent_method, K1, K2):
        
        if clus_method == 'kmoids' and cent_method == 'density':
            raise NameError('clus_method and cent_method are not matched')
        
        print('1단계 시작')
        #save file name 정의하기.
        station = self.station
        bike_path = self.bike_path
        lent_df = self.lent_df
        return_df = self.return_df
        
        sta_dist = pd.read_csv(bike_path+'dis_matrix.csv')
        
        sta_dist.columns = sta_dist.columns.astype('int')
        sta_dist.index = sta_dist.columns
        sta_dist = sta_dist.loc[self.ok_bike_list, self.ok_bike_list]
        
        X = np.array(station[['latitude', 'longitude']])
        
        # pattern 기준 클러스터링
        sta_list = station['sta_num'].values
        
        t_matrix_df = station[['latitude','longitude']]
        t_matrix_df.index = sta_list
        
        save_file_name = '%s_%s_%s_%s_%s_slot%s_season2_clustering'%(self.w_str, clus_method, cent_method, K1, K2, self.slot)
                
        # 파일이 있는지 검사함
        if save_file_name in os.listdir(self.cluster_save_path):
            return 0
        
        # pattern_mat : shape 바꿔야 함.
        input_matrix, matrix_key = self.make_dist_matrix(t_matrix_df)
        
        if clus_method == 'kmeans':
            if cent_method == 'density':
                cent_data = self.find_density_cent_value(K1, t_matrix_df)
            elif cent_method == 'median':
                cent_sta_num = self.find_median_cent_value(K1, matrix_key, input_matrix)
                cent_data = station[station['sta_num'].isin(cent_sta_num)][['latitude','longitude']].values
                
                
                
            station['bcluster'] = KMeans(n_clusters = int(K1), init=cent_data, n_init=1).fit(X).labels_
        
        elif clus_method == 'kmoids':
                
            t_cluster_df = self.k_moid_clustering(K1, matrix_key, input_matrix)
            t_cluster_df.columns = ['sta_num','bcluster','f_clus_num']
            
            station['bcluster'] = t_cluster_df['bcluster']
        
        else :
            raise NameError('this is not in clus_method')
        
        
        
        # 각 클러스터에서 위도와 경도에 따라 클러스터를 정하고 클러스터링 함.
        clus_list = np.unique(station['bcluster'])
        clus_cluster_list = []
        
        # 클러스터를 나눌 때 기준에서 거리 vs 클러스터 개수를 기준으로 해봄.
        for clus_n in clus_list:
            target_sta_list = station[station['bcluster'] == clus_n]['sta_num'].tolist()
            clus_cluster_list.append(len(target_sta_list))
        each_clus_num_list = self.cluster_num_by(clus_cluster_list, K2)
        
        new_cluster_df = pd.DataFrame()
        new_cluster_df['sta_num'] = station['sta_num']
        new_cluster_df['bcluster'] = 0
        
    #%%    
    
        step1_station_df = station[['sta_num','bcluster']].copy()
        step1_station_df.columns = ['sta_num','f_clus_num']
        
        with open(self.cluster_save_path+'/'+save_file_name+'_step1.pickle', 'wb') as f:
            pickle.dump(step1_station_df,f)
        print('1단계 완료')
    
        # 2차 클러스터링.
        
        print('2단계 시작')
        off_set = 0
        for i in range(len(each_clus_num_list)):
            t_split_clus_n = each_clus_num_list[i]
            clus_n = clus_list[i]
            
            target_station = station[station['bcluster'] == clus_n]
            target_ind = target_station.index
            sta_n_list = target_station['sta_num'].tolist()
            
            target_lent_df = lent_df[lent_df['lent_sta'].isin(sta_n_list)]
            target_return_df = return_df[return_df['return_sta'].isin(sta_n_list)]
    
            pattern_mat, unique_sta_list = self.extract_row(target_lent_df, target_return_df, sta_n_list, self.slot)
    
            t_matrix_df = pd.DataFrame(pattern_mat)
            t_matrix_df.index =sta_n_list
            input_matrix, matrix_key = self.make_dist_matrix(t_matrix_df)
            
            if clus_method == 'kmeans':
                if cent_method == 'density':
                    cent_data = self.find_density_cent_value(int(t_split_clus_n), t_matrix_df)
                
                elif cent_method == 'median':
                    cent_sta_num = self.find_median_cent_value(int(t_split_clus_n), matrix_key, input_matrix)
                    cent_data = t_matrix_df.loc[cent_sta_num].values
        
                    
                
                labels_ = KMeans(n_clusters=int(t_split_clus_n), init = cent_data,n_init=1).fit(pattern_mat).labels_ 
                labels_value = labels_ + off_set
                new_cluster_df.loc[target_ind,'bcluster'] = labels_value
    
            elif clus_method == 'kmoids':
                
                t_cluster_df = self.k_moid_clustering(int(t_split_clus_n), matrix_key, input_matrix)
                labels_ = t_cluster_df['clus_num'].values - 10000
                labels_value = labels_ + off_set
                new_cluster_df.loc[target_ind,'bcluster'] = labels_value
    
            else :
                raise NameError('this is not in clus_method')
                
            off_set = off_set + max(labels_) + 1
    
    #%%    
        
        new_cluster_df.columns = ['sta_num','f_clus_num']
        
        with open(self.cluster_save_path+'/'+save_file_name+'_step2.pickle', 'wb') as f:
            pickle.dump(new_cluster_df, f)
        print('2단계 완료')
        
        print('3단계 시작')
        merged_cluster_df = pd.merge(new_cluster_df, station[['sta_num','longitude','latitude', 'bcluster']], on='sta_num', how='left')
        
        final_cluster_df = merged_cluster_df[['sta_num']].copy()
        final_cluster_df['f_clus_num'] = 0
        final_cluster_df.index = final_cluster_df['sta_num'].tolist()
        
        off_set = 0
        for clus_n in np.unique(merged_cluster_df['bcluster']) :
            t_df = merged_cluster_df[merged_cluster_df['bcluster'] == clus_n]
            uni_f_clus = np.unique(t_df['f_clus_num'])
            clus_number = len(uni_f_clus)
            
            temp_clus_list = []
            for f_clus_n in uni_f_clus :
                t_f_df = t_df[t_df['f_clus_num'] == f_clus_n]
                y_mean = np.mean(t_f_df['longitude'].values)
                x_mean = np.mean(t_f_df['latitude'].values)
                temp_clus_list.append([x_mean,y_mean])
            
            cent_data = np.array(temp_clus_list)
            
            X = t_df[['latitude','longitude']]
            target_sta_list = t_df['sta_num'].values
            labels_ = KMeans(n_clusters=clus_number, init = np.array(temp_clus_list),n_init=1).fit(X).labels_ 
            labels_value = labels_ + off_set
    
            final_cluster_df.loc[target_sta_list,'f_clus_num'] = labels_value
            off_set = off_set + max(labels_) + 1
        
        final_cluster_df = final_cluster_df.reset_index(drop=True)
    #%% 
        with open(self.cluster_save_path+'\\'+save_file_name+'_step3.pickle', 'wb') as f:
            pickle.dump(final_cluster_df, f)
        print('3단계 완료')
    
        return 1
