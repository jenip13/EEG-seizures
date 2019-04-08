#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:59:03 2019

@author: jenisha
"""

# Libraries imports
import mne

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import os
import re
import glob



sampling_rate = 256

def get_raw_data(filename):
    """
    Load the edf file in a Numpy array
    
    Input: 
        file_name: where the file is located
            
    Output:
        data: raw data where the rows is the data and the column
    
    
    """
    # Load the edf file   
    raw = mne.io.read_raw_edf(filename,preload=True)
    data = raw.get_data()
    scaling_factor = 10**6 #Data in V; want in mV
    
    data = data * scaling_factor
    
    #Remove STI
    data = data[:-1]
    channel_names = raw.ch_names[:-1]
    times = raw.times
    
    return (data, times, channel_names )

def extract_spectral_features(data_channel,name_channel, interval, window_size = 2):
    """
        Extract the psd features for a channel
        
        Input: 
            data_channel: data in a channel
            name_channel: name of the channel of interest
            interval: from where data to transform starts
            window_size: size of epoch
    
    
    """
    i = interval[0]
    name_channel_delta=name_channel+" Delta"
    name_channel_theta=name_channel+" Theta"
    name_channel_alpha=name_channel+" Alpha"
    name_channel_beta=name_channel+" Beta"

#     
    spectral_features = []
    while (i+window_size*sampling_rate  < interval[1]):
        all_psd, all_freqs = mlab.psd(np.squeeze(data_channel[i:i+window_size*sampling_rate]),
                                      NFFT=256,Fs=256,scale_by_freq=True)
        delta_indices = np.where(np.logical_and(all_freqs>=1,all_freqs<=4))
        theta_indices = np.where(np.logical_and(all_freqs>4,all_freqs<=8))
        alpha_indices = np.where(np.logical_and(all_freqs>8,all_freqs<=13))
        beta_indices = np.where(np.logical_and(all_freqs>13,all_freqs<=30))
        
        delta= np.mean(all_psd[delta_indices])
        theta= np.mean(all_psd[theta_indices])
        alpha= np.mean(all_psd[alpha_indices])
        beta= np.mean(all_psd[beta_indices])
        
        spectral_features.append((delta,theta,alpha,beta))
        
        
        i = i + window_size*sampling_rate
     
    df_spectral_features = pd.DataFrame(spectral_features,columns=[name_channel_delta,
                                              name_channel_theta,
                                              name_channel_alpha,
                                              name_channel_beta])
    
    return df_spectral_features




def feature_vector_file(subject_directory,regex,dict_interval={},state=0, sample_size=200):
    """
        Extract the spectral (psd) and spatial (channel) for every file in a directory
        
        Input: 
            subject_directory: directory representing all the data of a subject
            regex: regex to determine number of file
            dict_interval: dictinary of intervals to use for each file in directory
            state: 
                seizure state is 1
                non-seizure state is 0
            sample size: for non-seizure file recordings, how many samples to randomly extract
    
    
    """
    #Files in directory:
    files_subject = glob.glob(subject_directory+"*edf")
    
    
    r = re.compile(regex)
    
    
    df_feature_vector = pd.DataFrame()
    for file_name in files_subject:
        num_file = (r.search(file_name )).group(1)
        
        # Read Data
        data,_, channel_names  = get_raw_data(file_name) 
        if not dict_interval:
            interval=(0,(data.shape)[1])
        
        else:
            i = dict_interval[str(num_file)]
            interval = (i[0] *sampling_rate, i[1] *sampling_rate)

        #Obtain features
        df_all_spectral_features = pd.DataFrame()
        for i,name in enumerate(channel_names):
            tmp_df = extract_spectral_features(data[i],name_channel=name,interval=interval)
            df_all_spectral_features = pd.concat([df_all_spectral_features,tmp_df],
                                                 axis=1)
        #Sample features
        if sample_size == 0:
            df_sample = df_all_spectral_features.copy()
        else:
            df_sample= df_all_spectral_features.sample(sample_size)   
            df_sample = df_sample.reset_index(drop=True)
        
        df_sample['File'] = num_file
        df_sample['State'] = state
        
        df_feature_vector = pd.concat([df_feature_vector, df_sample])
    
    df_feature_vector = df_feature_vector.reset_index(drop=True)
    
    return df_feature_vector


dir_files_subject1 = "/Users/jenisha/chb01/"
#df_feature_vector_subject1 = feature_vector_file(dir_files_subject1,
#                                                 '/Users/jenisha/chb01/chb01_(.*?).edf')
        
        
seizure_subject1 = {"03":(2996,3036),"04":(1467,1494),"15":(1732,1772),
                    "16":(1015,1066),"18":(1720,1810),"21":(327,420),"26":(1862,1963)}        
dir_files_subject1_seizure =  dir_files_subject1 + "seizures/"       
#df_feature_vector_subject1_seizure = feature_vector_file(dir_files_subject1_seizure,
#                                                 '/Users/jenisha/chb01/seizures/chb01_(.*?).edf',
#                                                 seizure_subject1,sample_size=0,state=1)



df_power_s1 = pd.concat([df_feature_vector_subject1,df_feature_vector_subject1_seizure])
df_power_s1 =df_power_s1.reset_index(drop=True)


#dir_files_subject2 = "/Users/jenisha/chb02/"
#df_feature_vector_subject2 = feature_vector_file(dir_files_subject2,
#                                                 '/Users/jenisha/chb02/chb02_(.*?).edf')


seizure_subject2 = {"16":(130,212),"16+":(2972,3053),"19":(3369,3378)}        
dir_files_subject2_seizure =  dir_files_subject2 + "seizures/"       
#df_feature_vector_subject2_seizure = feature_vector_file(dir_files_subject2_seizure,
#                                                 '/Users/jenisha/chb02/seizures/chb02_(.*?).edf',
#                                                 seizure_subject2,sample_size=0,state=1)



