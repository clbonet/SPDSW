#!/usr/bin/env python3

## Taken from https://github.com/MultiScale-BCI/IV-2a

'''	Loads the dataset 2a of the BCI Competition IV
available on http://bnci-horizon-2020.eu/database/data-sets
'''

import numpy as np
import scipy.io as sio

from .filters import load_filterbank, butter_fir_filter

__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"



def get_cov(data):
    fs = 250 # Sampling frequency
    bw = [25] ## bandwidth
    forder = 8
    max_freq = 30
    ftype = "butter"

    time_windows_flt = np.array([[2.5,4.5], [4,6], [2.5,6],
                                [2.5,3.5], [3,4], [4,5]])*fs
    time_windows = time_windows_flt.astype(int)
    # restrict time windows and frequency bands 
    time_windows = time_windows[2:3] 
    


    filter_bank = load_filterbank(bandwidth = bw, fs = fs, order = forder, 
                                  max_freq = max_freq, ftype = ftype)
    
    n_tr_trial, n_channel, _ = data.shape
    n_riemann = int((n_channel+1)*n_channel/2)

    n_temp = time_windows.shape[0]
    n_freq = filter_bank.shape[0]
    rho = 0.1

    temp_windows = time_windows

    cov_mat = np.zeros((n_tr_trial, n_temp, n_freq, n_channel, n_channel))

    # calculate training covariance matrices  
    for trial_idx in range(n_tr_trial):	

        for temp_idx in range(n_temp): 
            t_start, t_end  = temp_windows[temp_idx,0], temp_windows[temp_idx,1]
            n_samples = t_end-t_start

            for freq_idx in range(n_freq): 
                # filter signal 
                data_filter = butter_fir_filter(data[trial_idx,:,t_start:t_end], filter_bank[freq_idx])
                # regularized covariance matrix 
                cov_mat[trial_idx,temp_idx,freq_idx] = 1/(n_samples-1)*np.dot(data_filter,np.transpose(data_filter)) + rho/n_samples*np.eye(n_channel)
    
    return cov_mat


def get_data(subject, training, PATH):
	'''	Loads the dataset 2a of the BCI Competition IV
	available on http://bnci-horizon-2020.eu/database/data-sets

	Keyword arguments:
	subject -- number of subject in [1, .. ,9]
	training -- if True, load training data
				if False, load testing data
	
	Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
			class_return 	numpy matrix 	size = NO_valid_trial
	'''
	NO_channels = 22
	NO_tests = 6*48 	
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

	NO_valid_trial = 0
	if training:
		a = sio.loadmat(PATH+'A0'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(PATH+'A0'+str(subject)+'E.mat')
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2=[a_data1[0,0]]
		a_data3=a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_fs 		= a_data3[3]
		a_classes 	= a_data3[4]
		a_artifacts = a_data3[5]
		a_gender 	= a_data3[6]
		a_age 		= a_data3[7]
		for trial in range(0,a_trial.size):
			if(a_artifacts[trial]==0):
				data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
				class_return[NO_valid_trial] = int(a_y[trial])
				NO_valid_trial +=1


	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]
