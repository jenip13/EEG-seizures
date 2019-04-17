#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:20:49 2019

@author: jenisha
"""
# Library imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from processing_data import get_raw_data
from matplotlib import mlab


def plot_4d_psd(df_power_subset, col_names):
    """
    Plots the 4D PSD plots (delta, theta, alpha, beta) of a subset of channels
    
    Input:
        df_power_subset: PSD of a channel (delta, theta, alpha, beta) and its state
        col_names: Name of columns for that channel + state channel
    
    
    """
    channel_name = (col_names[0].split())[0]
    colors = {0:'blue', 1:'red'}

    fig = plt.figure()
    
    
    for i in range(4):
        for j in range(4):
            index = i*4+j +1
            ax = fig.add_subplot(4, 4, index)
            ax.scatter(df_power_subset.iloc[:,j],df_power_subset.iloc[:,-i-2],
              c=df_power_subset.iloc[:,4].apply(lambda s: colors[s]))
            
            if j == 0:
                ax.set_ylabel(col_names[-i-2])
            if i == 3:
                ax.set_xlabel(col_names[j])
                
    
    red_patch = mpatches.Patch(color='red', label='Seizure')
    blue_patch = mpatches.Patch(color='blue', label='Non-seizure')
    plt.suptitle(channel_name)
    plt.legend(handles=[red_patch, blue_patch])
    name_file = "./psd_plots/" + channel_name
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig(name_file)
    plt.close(fig)
    
#df_power_spectrum = pd.read_csv("df_power_s1.csv")
#col_names = list(df_power_spectrum.columns)[1:-2]
#list_channel_groups = list(zip(*(iter(col_names),) * 4))
#for group in list_channel_groups:
#    new_group = list(group)+ ['State']
#    #print(new_group)
#    df_power_subset = df_power_spectrum.loc[:,new_group]
#    plot_4d_psd(df_power_subset, new_group)
#    
def psd_compare(data, start1,start2):
    plt.figure()
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    
    df_signals = data

    for i in range(5):
        power1, freqs1 = mlab.psd(np.squeeze(df_signals[10:11,(512*i):(512*(i+1))]), NFFT=256, Fs=256)
        power2, freqs2 = mlab.psd(np.squeeze(df_signals[10:11,(766976+512*i):(766976+512*(i+1))]), NFFT=256, Fs=256)
        ax1.plot(freqs1, power1)
        ax2.plot(freqs2, power2)
    ax1.axvspan(0, 4, ymin=0, ymax=1, facecolor='m')
    ax1.axvspan(4, 8, ymin=0, ymax=1, facecolor='b')
    ax1.axvspan(8, 13, ymin=0, ymax=1, facecolor='c')
    ax1.axvspan(13,30, ymin=0, ymax=1, facecolor='g')
    ax2.axvspan(0, 4, ymin=0, ymax=1, facecolor='m')
    ax2.axvspan(4, 8, ymin=0, ymax=1, facecolor='b')
    ax2.axvspan(8, 13, ymin=0, ymax=1, facecolor='c')
    ax2.axvspan(13,30, ymin=0, ymax=1, facecolor='g')
    ax1.set_xlim((0, 30))    
    ax2.set_xlim((0, 30))
    ax1.set_xlabel('Frequenzy(Hz)')   
    ax2.set_xlabel('Frequency(Hz)')
    ax1.set_ylabel('PSD')
    #plt.suptitle("Fig4: Normal (left) and Epileptic (right) activity of 20 epochs of 2 seconds (FP1-F7)", fontsize=14)
    plt.show()
    
data_plot,_,channel_names = get_raw_data('/Users/jenisha/chb01/seizures/chb01_03.edf')
a = psd_compare(data_plot ,512, 766976)