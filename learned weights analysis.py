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


hdf5_path = '/Users/lucyzhang/Desktop/analysis/model_simulation'

DF_curriculum = pd.read_hdf(os.path.join(hdf5_path,"learned_weights_curriculum.h5"))
DF_parallel = pd.read_hdf(os.path.join(hdf5_path,"learned_weights_parallel.h5"))

DF_curriculum = DF_curriculum.rename(columns = {'assigned weights': 'assigned_weights', 'learned weights': 'learned_weights'})
DF_parallel = DF_parallel.rename(columns = {'assigned weights': 'assigned_weights', 'learned weights': 'learned_weights'})

cue_curriculum = DF_curriculum[DF_curriculum['assigned_weights']!=0]
cue_parallel = DF_parallel[DF_parallel['assigned_weights']!=0]


x = np.linspace(-2.5, 2.5, 1000)
y = np.linspace(-2.5, 2.5, 1000)


fig = plt.figure()
sns.set_style('white')
sns.regplot(data=cue_curriculum[cue_curriculum.model == 'ideal_learner'], label = 'ideal learner', x = 'assigned_weights', y = 'learned_weights', x_estimator=np.mean, x_ci = 95, fit_reg = False, scatter_kws={'s':4})
# sns.regplot(data=cue_parallel[cue_parallel.model == 'TTB_used'], x = 'assigned_weights', y = 'learned_weights', x_estimator=np.mean, x_ci = 95, fit_reg = False, scatter_kws={'s':4}, color = 'blue')
sns.regplot(data=cue_curriculum[cue_curriculum.model == 'cue_sign'], label = 'cue sign', x = 'assigned_weights', y = 'learned_weights', x_estimator=np.mean, fit_reg = False, scatter_kws={'s':4})
sns.regplot(data=cue_curriculum[cue_curriculum.model == 'TTB_used'], label = 'TTB used', x = 'assigned_weights', y = 'learned_weights', x_estimator=np.mean, fit_reg = False, scatter_kws={'s':4})
sns.regplot(data=cue_curriculum[cue_curriculum.model == 'TTB_all'], label = 'TTB all', x = 'assigned_weights', y = 'learned_weights', x_estimator=np.mean, fit_reg = False,  scatter_kws={'s':4})
plt.plot(x, y)
plt.legend()
plt.savefig('learned weights curriculum')


fig2 = plt.figure()
sns.set_style('white')
sns.regplot(data=cue_parallel[cue_parallel.model == 'ideal_learner'], label = 'ideal learner', x = 'assigned_weights', y = 'learned_weights', x_estimator=np.mean, x_ci = 95, fit_reg = False, scatter_kws={'s':4})
sns.regplot(data=cue_parallel[cue_parallel.model == 'cue_sign'], label = 'cue sign', x = 'assigned_weights', y = 'learned_weights', x_estimator=np.mean, fit_reg = False, scatter_kws={'s':4})
sns.regplot(data=cue_parallel[cue_parallel.model == 'TTB_used'], label = 'TTB used', x = 'assigned_weights', y = 'learned_weights', x_estimator=np.mean, fit_reg = False, scatter_kws={'s':4})
sns.regplot(data=cue_parallel[cue_parallel.model == 'TTB_all'], label = 'TTB all', x = 'assigned_weights', y = 'learned_weights', x_estimator=np.mean, fit_reg = False,  scatter_kws={'s':4})
plt.plot(x, y)
plt.legend()
plt.savefig('learned weights parallel')

plt.show()






softmax = lambda y1, y0, b: np.exp(b*y1)/(np.exp(b*y0)+np.exp(b*y1))

x1 = np.linspace(-3, 3, 1000)
x2 = np.linspace(3, -3, 1000)

y = softmax(x1, x2, 1)

weightmin1 = -0.955704
weightmax1 = 0.955704
ymin1 = softmax(weightmin1, weightmax1, 1)
ymax1 = softmax(weightmax1, weightmin1, 1)

weightmin2 = -0.990853
weightmax2 = 0.990853
ymin2 = softmax(weightmin2, weightmax2, 1)
ymax2 = softmax(weightmax2, weightmin2, 1)

weightmin3 = -0.76
weightmax3 = 0.76
ymin3 = softmax(weightmin3, weightmax3, 1)
ymax3 = softmax(weightmax3, weightmin3, 1)

weightmin4 = -4.468968
weightmax4 = 4.468968
ymin4 = softmax(weightmin4, weightmax4, 1)
ymax4 = softmax(weightmax4, weightmin4, 1)


fig, ax = plt.subplots()
ax.plot(x1, y)
ax.plot(weightmin1, ymin1, marker = 'x', color = 'red', label = 'ideal')
ax.plot(weightmax1, ymax1, marker = 'x', color = 'red')
ax.plot(weightmin2, ymin2, marker = '+', color = 'blue', label = 'TTBused')
ax.plot(weightmax2, ymax2, marker = '+', color = 'blue')
ax.plot(weightmin3, ymin3, marker = '*', color = 'green', label = 'cue sign')
ax.plot(weightmax3, ymax3, marker = '*', color = 'green')
ax.plot(weightmin4, ymin4, marker = 'D', color = 'pink', label = 'TTBall')
ax.plot(weightmax4, ymax4, marker = 'D', color = 'pink')

plt.legend()

plt.show()












