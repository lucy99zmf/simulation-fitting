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

from ideal_observer import ideal_curr, ideal_para
from cue_sign_learner import cuesign_curr, cuesign_para 
from TTB_used_cue_update import TTB_used_curr, TTB_used_para
from TTB_all_cue_update import TTB_all_curr, TTB_all_para



hdf5_path = '/Users/lucyzhang/Desktop/analysis/ready'
DF = pd.read_hdf(os.path.join(hdf5_path,"DF.h5"))

softmax = lambda y1, y0, b: np.exp(b*y1)/(np.exp(b*y0)+np.exp(b*y1))
logOdds = lambda x: np.log(x/(1-x)) # lambda indicates a simple function. this line is equivalent to: "def logOdds(x): <br><space> return np.log(x/(1-x))"
sigmoid = lambda x: 1/(1+np.exp(-x))

p = [.1, .2, .3, .4, .6, .7, .8, .9]
woes = logOdds(np.array(p)); woes = np.insert(woes, 0, 0)

df_qs_curriculum = pd.DataFrame()
df_qs_parallel = pd.DataFrame()


###############################
###############################
#running models and evaluate performance 
###############################
###############################
alpha = .05
beta = 1
para = [alpha, beta]
iterations = 50


###############################
#ideal observer
###############################

performance_curr = []; qs_curr = []
performance_para = []; qs_para = []

#curriculum
for i in range(iterations): 
	[qs, performance] = ideal_curr(para) 
	qs_curr.append(qs)
	performance_curr.append(performance)

#store weights learned 
model = []
assigned = []
weights = []
for i in range(len(qs_curr)):
	for k in range(len(qs_curr[i][0])): 
		model.append('ideal_learner')
		assigned.append(woes[k])
		weights.append(qs_curr[i][0][k])
d = {'model':model, 'assigned weights': assigned, 'learned weights': weights}
df_qs_curr = pd.DataFrame(d)
df_qs_curriculum = df_qs_curr


#calculate acc per block
performance_curr = np.array(performance_curr)
block_perf_curr = []
block = []
for i in range(len(performance_curr)): #for each iteration of training 
	for n in range(16): #calculate performance per block
		perf = np.mean(performance_curr[i][n*50:(n+1)*50-1])
		block.append(n+1)
		block_perf_curr.append(perf) 
d = {'block': block, 'accuracy': block_perf_curr}
df_block_acc_curr = pd.DataFrame(d)




#parallel
for i in range(iterations): 
	[qs, performance] = ideal_para(para) 
	qs_para.append(qs)
	performance_para.append(performance)

#store weights learned 
model = []
assigned = []
weights = []
for i in range(len(qs_para)):
	for k in range(len(qs_para[i][0])): 
		model.append('ideal_learner')
		assigned.append(woes[k])
		weights.append(qs_para[i][0][k])
d = {'model':model, 'assigned weights': assigned, 'learned weights': weights}
df_qs_para = pd.DataFrame(d)
df_qs_parallel = df_qs_para


performance_para = np.array(performance_para)
block_perf_para = []
block = []
for i in range(len(performance_para)): #for each iteration of training 
	for n in range(16): #calculate performance per block
		perf = np.mean(performance_para[i][n*50:(n+1)*50-1])
		block.append(n+1)
		block_perf_para.append(perf) 
d = {'block': block, 'accuracy': block_perf_para}
df_block_acc_para = pd.DataFrame(d)


#plot accuracy 
fig = plt.figure(figsize=(8,4))
ax1 = sns.lineplot(data = df_block_acc_curr, x = 'block', y = 'accuracy', ci = 95, color = 'darkblue')
ax2 = sns.lineplot(data = df_block_acc_para, x = 'block', y = 'accuracy', ci = 95, color = 'darkgreen')
plt.xlabel('blocks')
plt.ylabel('accuracy')
plt.legend(('curriculum', 'parallel'))
plt.title('Per Block Accuracy Ideal Learner')
plt.savefig('Per Block Accuracy Ideal Learner')

#plot weights learned 
# fig = plt.subplots(2,1)

# plt.subplot(2, 1, 1)
# ax1 = sns.barplot(data = df_qs_curr, ci = 95) 
# plt.title('curriculum')
# plt.ylabel('weights')

# plt.subplot(2, 1, 2)
# ax2 = sns.barplot(data = df_qs_para, ci = 95)
# plt.title('parallel')
# plt.xlabel('cues')
# plt.ylabel('weights')

# plt.show()


###############################
#cue sign learner
###############################
alpha = .05
beta = 6
para = [alpha, beta]
iterations = 50

performance_curr = []; qs_curr = []
performance_para = []; qs_para = []

#curriculum
for i in range(iterations): #run the model
	[qs, performance] = cuesign_curr(para) 
	qs_curr.append(qs)
	performance_curr.append(performance)

#plot weights learned 
model = []
assigned = []
weights = []
for i in range(len(qs_curr)):
	for k in range(len(qs_curr[i][0])): 
		model.append('cue_sign')
		assigned.append(woes[k])
		weights.append(qs_curr[i][0][k])
d = {'model':model, 'assigned weights': assigned, 'learned weights': weights}
df_qs_curr = pd.DataFrame(d)
df_qs_curriculum = pd.concat([df_qs_curriculum, df_qs_curr], axis = 0)



#calculate acc 
performance_curr = np.array(performance_curr)
block_perf_curr = []
block = []
for i in range(len(performance_curr)): #for each iteration of training 
	for n in range(16): #calculate performance per block
		perf = np.mean(performance_curr[i][n*50:(n+1)*50-1])
		block.append(n+1)
		block_perf_curr.append(perf) 
d = {'block': block, 'accuracy': block_perf_curr}
df_block_acc_curr = pd.DataFrame(d)





#parallel
for i in range(iterations): #run the model 
	[qs, performance] = cuesign_para(para) 
	qs_para.append(qs)
	performance_para.append(performance)

#plot the weights learned
model = []
assigned = []
weights = []
for i in range(len(qs_para)):
	for k in range(len(qs_para[i][0])): 
		model.append('cue_sign')
		assigned.append(woes[k])
		weights.append(qs_para[i][0][k])
d = {'model':model, 'assigned weights': assigned, 'learned weights': weights}
df_qs_para = pd.DataFrame(d)
df_qs_parallel = pd.concat([df_qs_parallel, df_qs_para], axis = 0)


performance_para = np.array(performance_para)
block_perf_para = []
block = []
for i in range(len(performance_para)): #for each iteration of training 
	for n in range(16): #calculate performance per block
		perf = np.mean(performance_para[i][n*50:(n+1)*50-1])
		block.append(n+1)
		block_perf_para.append(perf) 
d = {'block': block, 'accuracy': block_perf_para}
df_block_acc_para = pd.DataFrame(d)


#plot accuracy 
fig = plt.figure(figsize=(8,4))
ax1 = sns.lineplot(data = df_block_acc_curr, x = 'block', y = 'accuracy', ci = 95, color = 'darkblue')
ax2 = sns.lineplot(data = df_block_acc_para, x = 'block', y = 'accuracy', ci = 95, color = 'darkgreen')
plt.xlabel('blocks')
plt.ylabel('accuracy')
plt.legend(('curriculum', 'parallel'))
plt.title('Per Block Accuracy Cue Sign Learner')
plt.savefig('Per Block Accuracy Cue Sign Learner')

#plot weights learned 
# fig = plt.subplots(2,1)

# plt.subplot(2, 1, 1)
# ax1 = sns.barplot(data = df_qs_curr, ci = 95) 
# plt.title('curriculum')
# plt.ylabel('weights')

# plt.subplot(2, 1, 2)
# ax2 = sns.barplot(data = df_qs_para, ci = 95)
# plt.title('parallel')
# plt.xlabel('cues')
# plt.ylabel('weights')

# plt.show()






###############################
#TTB used cue update
###############################
alpha = .05
beta = 1
k = 2.5
para = [alpha, beta, k]
iterations = 50


performance_curr = []; qs_curr = []
performance_para = []; qs_para = []

#curriculum
for i in range(iterations): #run the model
	[qs, performance] = TTB_used_curr(para) 
	qs_curr.append(qs)
	performance_curr.append(performance)

#plot weights learned 
model = []
assigned = []
weights = []
for i in range(len(qs_curr)):
	for k in range(len(qs_curr[i][0])): 
		model.append('TTB_used')
		assigned.append(woes[k])
		weights.append(qs_curr[i][0][k])
d = {'model':model, 'assigned weights': assigned, 'learned weights': weights}
df_qs_curr = pd.DataFrame(d)
df_qs_curriculum = pd.concat([df_qs_curriculum, df_qs_curr], axis = 0)



#calculate acc 
performance_curr = np.array(performance_curr)
block_perf_curr = []
block = []
for i in range(len(performance_curr)): #for each iteration of training 
	for n in range(16): #calculate performance per block
		perf = np.mean(performance_curr[i][n*50:(n+1)*50-1])
		block.append(n+1)
		block_perf_curr.append(perf) 
d = {'block': block, 'accuracy': block_perf_curr}
df_block_acc_curr = pd.DataFrame(d)





#parallel
for i in range(iterations): #run the model 
	[qs, performance] = TTB_used_para(para) 
	qs_para.append(qs)
	performance_para.append(performance)

#plot the weights learned
model = []
assigned = []
weights = []
for i in range(len(qs_para)):
	for k in range(len(qs_para[i][0])): 
		model.append('TTB_used')
		assigned.append(woes[k])
		weights.append(qs_para[i][0][k])
d = {'model':model, 'assigned weights': assigned, 'learned weights': weights}
df_qs_para = pd.DataFrame(d)
df_qs_parallel = pd.concat([df_qs_parallel, df_qs_para], axis = 0)


performance_para = np.array(performance_para)
block_perf_para = []
block = []
for i in range(len(performance_para)): #for each iteration of training 
	for n in range(16): #calculate performance per block
		perf = np.mean(performance_para[i][n*50:(n+1)*50-1])
		block.append(n+1)
		block_perf_para.append(perf) 
d = {'block': block, 'accuracy': block_perf_para}
df_block_acc_para = pd.DataFrame(d)


#plot accuracy 
fig = plt.figure(figsize=(8,4))
ax1 = sns.lineplot(data = df_block_acc_curr, x = 'block', y = 'accuracy', ci = 95, color = 'darkblue')
ax2 = sns.lineplot(data = df_block_acc_para, x = 'block', y = 'accuracy', ci = 95, color = 'darkgreen')
plt.xlabel('blocks')
plt.ylabel('accuracy')
plt.legend(('curriculum', 'parallel'))
plt.title('Per Block Accuracy TTB Used Cue Update')
plt.savefig('Per Block Accuracy TTB Used Cue Update')

#plot weights learned 
# fig = plt.subplots(2,1)

# plt.subplot(2, 1, 1)
# ax1 = sns.barplot(data = df_qs_curr, ci = 95) 
# plt.title('curriculum')
# plt.ylabel('weights')

# plt.subplot(2, 1, 2)
# ax2 = sns.barplot(data = df_qs_para, ci = 95)
# plt.title('parallel')
# plt.xlabel('cues')
# plt.ylabel('weights')

# plt.show()




###############################
#TTB all cue update
###############################
alpha = .05
beta = 1
k = 2.5
para = [alpha, beta, k]
iterations = 50


performance_curr = []; qs_curr = []
performance_para = []; qs_para = []

#curriculum
for i in range(iterations): #run the model
	[qs, performance] = TTB_all_curr(para) 
	qs_curr.append(qs)
	performance_curr.append(performance)

#plot weights learned 
model = []
assigned = []
weights = []
for i in range(len(qs_curr)):
	for k in range(len(qs_curr[i][0])): 
		model.append('TTB_all')
		assigned.append(woes[k])
		weights.append(qs_curr[i][0][k])
d = {'model':model, 'assigned weights': assigned, 'learned weights': weights}
df_qs_curr = pd.DataFrame(d)
df_qs_curriculum = pd.concat([df_qs_curriculum, df_qs_curr], axis = 0)



#calculate acc 
performance_curr = np.array(performance_curr)
block_perf_curr = []
block = []
for i in range(len(performance_curr)): #for each iteration of training 
	for n in range(16): #calculate performance per block
		perf = np.mean(performance_curr[i][n*50:(n+1)*50-1])
		block.append(n+1)
		block_perf_curr.append(perf) 
d = {'block': block, 'accuracy': block_perf_curr}
df_block_acc_curr = pd.DataFrame(d)




#parallel
for i in range(iterations): #run the model 
	[qs, performance] = TTB_all_para(para) 
	qs_para.append(qs)
	performance_para.append(performance)

#plot the weights learned
model = []
assigned = []
weights = []
for i in range(len(qs_para)):
	for k in range(len(qs_para[i][0])): 
		model.append('TTB_all')
		assigned.append(woes[k])
		weights.append(qs_para[i][0][k])
d = {'model':model, 'assigned weights': assigned, 'learned weights': weights}
df_qs_para = pd.DataFrame(d)
df_qs_parallel = pd.concat([df_qs_parallel, df_qs_para], axis = 0)


performance_para = np.array(performance_para)
block_perf_para = []
block = []
for i in range(len(performance_para)): #for each iteration of training 
	for n in range(16): #calculate performance per block
		perf = np.mean(performance_para[i][n*50:(n+1)*50-1])
		block.append(n+1)
		block_perf_para.append(perf) 
d = {'block': block, 'accuracy': block_perf_para}
df_block_acc_para = pd.DataFrame(d)



#plot accuracy 
fig = plt.figure(figsize=(8,4))
ax1 = sns.lineplot(data = df_block_acc_curr, x = 'block', y = 'accuracy', ci = 95, color = 'darkblue')
ax2 = sns.lineplot(data = df_block_acc_para, x = 'block', y = 'accuracy', ci = 95, color = 'darkgreen')
plt.xlabel('blocks')
plt.ylabel('accuracy')
plt.legend(('curriculum', 'parallel'))
plt.title('Per Block Accuracy TTB All Cue Update')
plt.savefig('Per Block Accuracy TTB All Cue Update')


#plot weights learned 
# fig = plt.subplots(2,1)

# plt.subplot(2, 1, 1)
# ax1 = sns.barplot(data = df_qs_curr, ci = 95) 
# plt.title('curriculum')
# plt.ylabel('weights')

# plt.subplot(2, 1, 2)
# ax2 = sns.barplot(data = df_qs_para, ci = 95)
# plt.title('parallel')
# plt.xlabel('cues')
# plt.ylabel('weights')


#save learned weights for analysis 
hdf5_path = '/Users/lucyzhang/Desktop/analysis/model_simulation'

df_qs_parallel.to_hdf('learned_weights_parallel.h5', key = 'parallel', mode = 'w')
df_qs_curriculum.to_hdf('learned_weights_curriculum.h5', key = 'curriculum', mode = 'w')








