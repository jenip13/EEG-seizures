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

import re
import glob

from matplotlib import mlab
from scipy.signal import periodogram

sampling_rate = 256
channel_names= ['FP1-F7',
 'F7-T7',
 'T7-P7',
 'P7-O1',
 'FP1-F3',
 'F3-C3',
 'C3-P3',
 'P3-O1',
 'FP2-F4',
 'F4-C4',
 'C4-P4',
 'P4-O2',
 'FP2-F8',
 'F8-T8',
 'T8-P8-0',
 'P8-O2',
 'FZ-CZ',
 'CZ-PZ',
 'P7-T7',
 'T7-FT9',
 'FT9-FT10',
 'FT10-T8']

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
    i = int(interval[0])
    name_channel_delta=name_channel+" Delta"
    name_channel_theta=name_channel+" Theta"
    name_channel_alpha=name_channel+" Alpha"
    name_channel_beta=name_channel+" Beta"

#     
    spectral_features = []
    while (i+window_size*sampling_rate  <= interval[1]):
        all_psd, all_freqs = mlab.psd(np.squeeze(data_channel[i:i+window_size*sampling_rate]),
                                      NFFT=256,Fs=256,scale_by_freq=True)
        
        #all_freqs, all_psd = periodogram(np.squeeze(data_channel[i:i+window_size*sampling_rate]), fs=256, nfft=256)
        
        
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
        #data,_, channel_names  = get_raw_data(file_name)
        data,_, _  = get_raw_data(file_name)
        if not dict_interval:
            interval=(0,(data.shape)[1]-1)
        
        else:
            i = dict_interval[str(num_file)]
            interval = (i[0]*sampling_rate , i[1]*sampling_rate)

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


def change_state(df, dict_intervals,window_size=2):
    df_all = pd.DataFrame()
    for num_file in dict_intervals.keys():
        interval = dict_intervals[num_file]
        df_temp = df.loc[df['File'] == num_file]
        df_temp = df_temp.reset_index(drop=True)
        scaling=window_size
        df_temp.loc[int(interval[0]//scaling):int(interval[1]//scaling), 'State'] = 1
        df_all = pd.concat([df_all,df_temp])
      
    df_all = df_all.reset_index(drop=True)
    return df_all
        


def feature_matrix(dir_files_subject,re_dir,re_seizure,seizure_dict,patient_number):
    df_feature_vector_subject = feature_vector_file(dir_files_subject,re_dir)
    dir_files_subject_seizure =  dir_files_subject + "seizures/"  
    df_feature_vector_subject_seizure = feature_vector_file(dir_files_subject_seizure,re_seizure,
                                                seizure_dict,sample_size=0,state=1)
    
    df_power = pd.concat([df_feature_vector_subject,df_feature_vector_subject_seizure])
    df_power =df_power.reset_index(drop=True)
    name_file = "df_power_s" + str(patient_number) + ".csv"
    
    df_power.to_csv(name_file)
    
    df_seizures_power_all = feature_vector_file(dir_files_subject_seizure,re_seizure,sample_size=0)
    
    df_seizures_power_states = change_state(df_seizures_power_all, seizure_dict)
    
    name_file = "df_power_states_s" + str(patient_number) + ".csv"
    
    df_seizures_power_states.to_csv(name_file)
    
    return (df_power,df_seizures_power_states )

    


#dir_files_subject1 = "/Users/jenisha/chb01/"
#df_feature_vector_subject1 = feature_vector_file(dir_files_subject1,
#                                                 '/Users/jenisha/chb01/chb01_(.*?).edf')
#        
#        
#seizure_subject1 = {"03":(2996,3036),"04":(1467,1494),"15":(1732,1772),
#                    "16":(1015,1066),"18":(1720,1810),"21":(327,420),"26":(1862,1963)}        
#dir_files_subject1_seizure =  dir_files_subject1 + "seizures/"       
#df_feature_vector_subject1_seizure = feature_vector_file(dir_files_subject1_seizure,
#                                                 '/Users/jenisha/chb01/seizures/chb01_(.*?).edf',
#                                                 seizure_subject1,sample_size=0,state=1)
#
#
#
#df_power_s1 = pd.concat([df_feature_vector_subject1,df_feature_vector_subject1_seizure])
#df_power_s1 =df_power_s1.reset_index(drop=True)
##df_power_s1.to_csv("df_power_s1_4s.csv")
#df_seizures_power_s1 = feature_vector_file(dir_files_subject1_seizure,
#                                           '/Users/jenisha/chb01/seizures/chb01_(.*?).edf',
#                                           sample_size=0)

#df_seizures_power_s1_states = change_state(df_seizures_power_s1, seizure_subject1)
#
#test1, test2=  feature_matrix(dir_files_subject1,'/Users/jenisha/chb01/chb01_(.*?).edf',
#                              '/Users/jenisha/chb01/seizures/chb01_(.*?).edf',seizure_subject1,1)

#dir_files_subject2 = "/Users/jenisha/chb02/"
#df_feature_vector_subject2 = feature_vector_file(dir_files_subject2,
#                                                 '/Users/jenisha/chb02/chb02_(.*?).edf')
#
#
#seizure_subject2 = {"16":(130,212),"16+":(2972,3053),"19":(3369,3378)}        
#dir_files_subject2_seizure =  dir_files_subject2 + "seizures/"       
#df_feature_vector_subject2_seizure = feature_vector_file(dir_files_subject2_seizure,
#                                                 '/Users/jenisha/chb02/seizures/chb02_(.*?).edf',
#                                                 seizure_subject2,sample_size=0,state=1)
#
#df_power_s2 = pd.concat([df_feature_vector_subject2,df_feature_vector_subject2_seizure])
#df_power_s2 =df_power_s2.reset_index(drop=True)
##df_power_s2.to_csv("df_power_s2_4s.csv")
#
#df_seizures_power_s2 = feature_vector_file(dir_files_subject2_seizure,
#                                           '/Users/jenisha/chb02/seizures/chb02_(.*?).edf',sample_size=0)

#df_seizures_power_s2_states = change_state(df_seizures_power_s2, seizure_subject2)


dir_files_subject11 = "/Users/jenisha/chb11/"
seizure_subject11 = {"82":(298,320),"92":(2695,2727),"99":(1454,2206)}  
df_power_s11, df_seizures_power_s11_states=  feature_matrix(dir_files_subject11,'/Users/jenisha/chb11/chb11_(.*?).edf',
                              '/Users/jenisha/chb11/seizures/chb11_(.*?).edf',seizure_subject11,11)


dir_files_subject22 = "/Users/jenisha/chb22/"
seizure_subject22 = {"20":(3367,3425 ),"25":(3139,3213),"38":(1263,1335)}  
df_power_s22, df_seizures_power_22_states=  feature_matrix(dir_files_subject22,'/Users/jenisha/chb22/chb22_(.*?).edf',
                              '/Users/jenisha/chb22/seizures/chb22_(.*?).edf',seizure_subject22,22)
