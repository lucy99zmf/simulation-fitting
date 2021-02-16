import numpy as np
from IPython import embed as shell
import os, sys, datetime, pickle
import scipy as sp
import numpy as np
import numpy.matlib
import matplotlib.pylab as plt
from matplotlib.colors import hex2color, rgb2hex
import pandas as pd
import os
import h5py
from scipy import stats, polyval, polyfit
import seaborn as sns
import math 

hdf5_path = '/Users/lucyzhang/Desktop/analysis/ready'
DF = pd.read_hdf(os.path.join(hdf5_path,"DF.h5"))

softmax = lambda y1, y0, b: np.exp(b*y1)/(np.exp(b*y0)+np.exp(b*y1))
logOdds = lambda x: np.log(x/(1-x)) # lambda indicates a simple function. this line is equivalent to: "def logOdds(x): <br><space> return np.log(x/(1-x))"
sigmoid = lambda x: 1/(1+np.exp(-x))

#slicing dataphrame to take the stimuli schedule 


def stimuli_array (curr):
	if curr: 
		DF_curr = DF[DF.condition==0]
		n = DF_curr.sample()['subject'].values; n = n[0]
		DF_curr = DF_curr[DF.subject==n]
		xs_curr = DF_curr[['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']].to_numpy()
		xs_curr = np.insert(xs_curr, 0, 1, axis = 1)
		return (xs_curr)
	else:
		DF_para = DF[DF.condition==1]
		n = DF_para.sample()['subject'].values; n = n[0]
		DF_para = DF_para[DF.subject==n]
		xs_para = DF_para[['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']].to_numpy()
		xs_para = np.insert(xs_para, 0, 1, axis = 1)
		return (xs_para)

#calculate outcome and reward schedule 
#A1 refers to the right option, A0 the left option, coding (1 right, 0 left)
p = [.1, .2, .3, .4, .6, .7, .8, .9]

def set_up (xs, p):
	woes = logOdds(np.array(p)); woes = np.insert(woes, 0, 0)
	stim_to_woe = woes*xs
	woe_A1 = np.nansum(stim_to_woe, axis=1)
	p_A1 = sigmoid(woe_A1)
	outcome = (np.random.uniform(0,1,len(xs)) < p_A1).astype('int')
	pA1_outcome_sche = np.hstack((p_A1[:,None], outcome[:, None]))

	return (pA1_outcome_sche)



def TTB_used_curr(para):
	alpha = para[0]
	beta = para[1]
	k = para[2]
	qs = np.zeros((2,9)) #qs[0] for A1(choosing right), qs[1] for A0(choosing left)
	p_choice_A1 = []
	A1_chosen = []
	corr = []

	curr = True  
	xs_curr = stimuli_array(curr)
	curriculum = set_up(xs_curr, p)


	#training / testing
	for trial in range(len(curriculum)):
		x = xs_curr[trial, :]; #print(x)
		#search for k most predictive cues, return an array of the index of used cues
		present_cues = np.nonzero(x)[0]; #print(x_temp)
		present_qs = [qs[0][i] for i in present_cues]
		abs_qs = np.absolute(np.reshape(present_qs, len(present_cues)))
		used_cues = []

		if np.sum(x)>2:
			for i in range(math.ceil(k) if k%1 > np.random.uniform(0, 1) else math.floor(k)):
				cue = np.argmax(abs_qs[1:])+1; 
				used_cues.append(present_cues[cue]); 
				abs_qs[cue] = 0; 
		
			#only use those cues: change x so that only the used cues are saved 
			x_new = np.zeros(9)
			for i in used_cues:
				x_new[i] = x[i]; 
			x = x_new
			#print(x)

		y_sum = np.matmul(qs, x) #output decision variable
		#choice prob of A1 stochastic decision rule
		p_choice_right = softmax(y_sum[0], y_sum[1], beta)
		p_choice_A1.append(p_choice_right)
		#A1 chosen based on p pchoice 
		A1_chosen.append(p_choice_right > np.random.uniform(0, 1)) 
		#whether the chosen outcome is correct or not 
		right = 1 if curriculum[trial][0]>.5 else 0
		corr.append(1 if A1_chosen[trial] == right else 0) 
		#whether that option is rewarded or not 
		r = curriculum[trial][1] 
		#update
		qs[0,:] += alpha*(r - p_choice_right)*x
		qs[1,:] += alpha*((1-r) - (1 - p_choice_right))*x 

	#performance stats 

	performance = np.array(corr)
	
	return [qs, performance] 


def TTB_used_para(para):

	################################
	#parallel
	################################
	alpha = para[0]
	beta = para[1]
	k = para[2]
	qs = np.zeros((2,9)) #qs[0] for A1(choosing right), qs[1] for A0(choosing left)
	p_choice_A1 = []
	A1_chosen = []
	corr = []

	curr = False  
	xs_para = stimuli_array(curr)
	parallel = set_up(xs_para, p)


	#training / testing
	for trial in range(len(parallel)):
		x = xs_para[trial, :]; #print(x)
		#search for k most predictive cues, return an array of the index of used cues
		present_cues = np.nonzero(x)[0]; #print(x_temp)
		present_qs = [qs[0][i] for i in present_cues]
		abs_qs = np.absolute(np.reshape(present_qs, len(present_cues)))
		used_cues = []

		for i in range(math.ceil(k) if k%1 > np.random.uniform(0, 1) else math.floor(k)):
			cue = np.argmax(abs_qs[1:])+1; 
			used_cues.append(present_cues[cue]); 
			abs_qs[cue] = 0; 
		
		#print(index)
		#only use those cues: change x so that only the used cues are saved 
		x_new = np.zeros(9)
		for i in used_cues:
			x_new[i] = x[i]; 
		x = x_new
		#print(x)

		y_sum = np.matmul(qs, x) #output decision variable
		#choice prob of A1 stochastic decision rule
		p_choice_right = softmax(y_sum[0], y_sum[1], beta)
		p_choice_A1.append(p_choice_right)
		#A1 chosen based on p pchoice 
		A1_chosen.append(p_choice_right > np.random.uniform(0, 1)) 
		#whether the chosen outcome is correct or not 
		right = 1 if parallel[trial][0]>.5 else 0
		corr.append(1 if A1_chosen[trial] == right else 0) 
		#whether that option is rewarded or not 
		r = parallel[trial][1] 
		#update
		qs[0,:] += alpha*(r - p_choice_right)*x
		qs[1,:] += alpha*((1-r) - (1 - p_choice_right))*x 

	#performance stats 

	performance = np.array(corr)

	
	return [qs, performance]