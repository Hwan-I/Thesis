# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:42:57 2020

@author: Lee
"""
#cluster_save_path_, w_str, slot, K1, K2, clus_method, ok_bike_list
import new_clustering as nc
import pickle
import os

path = os.getcwd()

with open(path+'/data/ok_bike_sta_num.pickle', 'rb') as f:
    ok_b = pickle.load(f)

weekday_clf = nc.new_clustering(cluster_save_path = r'C:\Users\Lee\Desktop\temp\result\\',w_str='weekday', ok_bike_list=ok_b, slot=5)
weekday_clf.fit(clus_method='kmoids', cent_method='median', K1=20, K2=60)


