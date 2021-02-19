from IPython import embed as shell
import os, sys, datetime, pickle
import scipy as sp
import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import hex2color, rgb2hex
import pandas as pd
import h5py
from scipy import stats, polyval, polyfit
import seaborn as sns
import math
from joblib import Parallel, delayed
 
hdf5_path = '/Users/lucyzhang/Desktop/analysis/model_fitting'

DF_ideal_fitted = pd.read_hdf(os.path.join(hdf5_path,'ideal_fitted_parameters.h5'), key = 'ideal_observer')
DF_cue_fitted = pd.read_hdf(os.path.join(hdf5_path,'cue_fitted_parameters.h5'), key = 'cue_sign')


####
#ideal observer
###

### distribution of all three variables 
fig, (ax1, ax2, ax3) = plt.subplots(3)
plt.tight_layout()
fig.tight_layout()
fig.suptitle('Distribution of fitted parameters ideal observer')
sns.histplot(data = DF_ideal_fitted, x = 'alpha', ax = ax1)
# ax1.set_title('alpha')
sns.histplot(data = DF_ideal_fitted, x = 'beta', ax = ax2)
# ax2.set_title('beta')
sns.histplot(data = DF_ideal_fitted, x = 'negLL', ax = ax3)
# ax3.set_title('error')
plt.savefig('distribution of fitted parameters ideal ideal observer')
# plt.show


### scatter plot of alpha and beta 
fig = plt.figure()
fig.suptitle('alpha beta error ideal observer')
sns.scatterplot(data = DF_ideal_fitted, x = 'alpha', y = 'beta', hue = 'negLL')
# sns.scatterplot(data = DF_ideal_fitted[DF_ideal_fitted.negLL > 540], x = 'alpha', y = 'beta', marker = 'X', color = 'red')
plt.title('error')
plt.savefig('alpha beta error scatter ideal observer')
plt.show()




####
#cue sign
###

### distribution of all three variables 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.tight_layout()
fig.suptitle('Distribution of fitted parameters cue sign')
sns.histplot(data = DF_cue_fitted, x = 'alpha', ax = ax1)
# ax1.set_title('alpha')
sns.histplot(data = DF_cue_fitted, x = 'beta', ax = ax2)
sns.histplot(data = DF_cue_fitted, x = 'sensitivity', ax = ax3)
# ax2.set_title('beta')
sns.histplot(data = DF_cue_fitted, x = 'negLL', ax = ax4)
# ax3.set_title('error')
plt.savefig('distribution of fitted parameters cue sign learner')
# plt.show


### scatter plot of alpha and beta 
fig = plt.figure()
fig.suptitle('alpha beta error cue sign')
sns.scatterplot(data = DF_cue_fitted, x = 'alpha', y = 'beta', hue = 'negLL')
# sns.scatterplot(data = DF_ideal_fitted[DF_ideal_fitted.negLL > 540], x = 'alpha', y = 'beta', marker = 'X', color = 'red')
plt.savefig('alpha beta error scatter cue sign')
plt.show()