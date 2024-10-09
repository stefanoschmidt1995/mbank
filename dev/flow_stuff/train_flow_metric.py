import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

import sys
sys.path.insert(0,'..')

import warnings
warnings.simplefilter('ignore', UserWarning)
import pickle

from mbank.flow import GW_Flow, TanhTransform, STD_GW_Flow
from mbank.flow.utils import plot_loss_functions, create_gif, plotting_callback

def my_callback(model, epoch):
	return plotting_callback(model, epoch, dirname+'img/', validation_samples, variable_format)

########################################

train = True
variable_format = 'logMq_nonspinning'
data_file = 'data/metric_mcmc_2D.dat'

	#Loading data
data = np.loadtxt(data_file)
np.random.shuffle(data)

N, D = data.shape[0], 2
train_factor = 0.8
print("N, D: ",N,D)
	
samples, metric = data[:,:D], data[:,D:]
metric = metric.reshape((N,D,D))

id_ = 0 #np.random.choice(metric.shape[0])
reference_metric = metric[id_]
reference_sample = samples[id_]

L = np.linalg.cholesky(reference_metric).T
L_inv = np.linalg.inv(L)
assert np.allclose(np.einsum('lk,lm,mn->kn', L_inv, reference_metric ,L_inv), np.eye(D))

	#transforming the metric and the samples with a standard transformation
metric = np.einsum('lk,ilm,mn->ikn', L_inv, metric ,L_inv)
samples = np.einsum('ij, lj -> il', samples - reference_sample, L)

	#splitting training/validation
train_samples, validation_samples = samples[:int(train_factor*N),:], samples[int(train_factor*N):,:]
train_metric, validation_metric = metric[:int(train_factor*N),:], metric[int(train_factor*N):,:]

n_layers, hidden_features = 2, 4

flow = STD_GW_Flow(D, n_layers, hidden_features)
#flow.load_weigths('standard_flow_2D/weights')
flow.load_weigths('test_metric_flow/weights_KL_div')
optimizer = optim.Adam(flow.parameters(), lr=0.001)

dirname = 'test_metric_flow/'
print('saving model in ', dirname)

if train:
	if False:
		history = flow.train_flow_forward_KL(N_epochs=1000,
			train_data=train_samples, validation_data=validation_samples,
			batch_size = 100,
			optimizer=optimizer,
			callback = None, #(my_callback, 1000) if args.callback_step>0 else None, 
			validation_metric = 'cross_entropy',
			verbose = True)

	if True:
		history = flow.train_flow_metric(N_epochs=1000,
			train_data=(train_samples, train_metric), validation_data=(validation_samples, validation_metric),
			batch_size = 100,
			alpha = 100,
			optimizer=optimizer,
			callback = None, #(my_callback, 1000) if args.callback_step>0 else None, 
			validation_metric = 'cross_entropy',
			verbose = True)
	my_callback(flow, 100)
	torch.save(flow.state_dict(), dirname+'weights')
	with open(dirname+'history.pkl', 'wb') as f:
		pickle.dump(history, f)
	print('Saved weights')

plot_loss_functions(history, savefolder = dirname)






















