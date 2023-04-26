import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
import itertools
import collections
import torch

import lal 
import lalsimulation as lalsim

from tqdm import tqdm

import ray

import mbank
from mbank.flow.flowmodel import STD_GW_Flow
from mbank.flow.utils import compare_probability_distribution

from torch import optim

import os
import subprocess

def get_samples(N, boundaries):
	theta = np.random.uniform(boundaries[0], boundaries[1], (N,3))
	return theta

##########################################################################################################################

compute_tiling = not False
compute_thetas = not False
theta_file_suffix ='_Mq_s1xz_iota'# '_fromtiling'
n_layers, hidden_features = 6,8

variable_format = 'Mq_s1xz'
boundaries = np.array([[10, 1, 0., -np.pi/2], [100, 5, 0.9, np.pi]])
f_min, f_max = 15, 1024.
psd_file = 'aligo_O3actual_H1.txt'
if not os.path.isfile(psd_file):
	subprocess.run('wget https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt', shell = True)
f, PSD = mbank.utils.load_PSD(psd_file, True, 'H1')
m_obj = mbank.cbc_metric(variable_format, (f,PSD), 'IMRPhenomXP', f_min = f_min, f_max = f_max)

	####
	# Loading & computing stuff
if compute_tiling:
	t_obj = mbank.tiling_handler()
	t_obj = t_obj.create_tiling(boundaries, 0.1, m_obj.get_hessian_symphony, max_depth = 4, verbose = True)
	t_obj.train_flow(N_epochs=1000, N_train_data = 20000, n_layers=n_layers, hidden_features = hidden_features)
	t_obj.save('files/tiling.npy', 'files/flow.zip')
else:
	t_obj = mbank.tiling_handler('files/tiling.npy')
	t_obj.load_flow('files/flow.zip')

if compute_thetas:
	train_theta = t_obj.sample_from_tiling(10000)
	valid_theta = t_obj.sample_from_tiling(1000)
	#train_theta = np.random.uniform(*boundaries, (20000,3))
	#valid_theta = np.random.uniform(*boundaries, (2000,3))
	train_pdf = np.sqrt(m_obj.get_metric_determinant(train_theta))
	valid_pdf = np.sqrt(m_obj.get_metric_determinant(valid_theta))

	np.savetxt('files/train_theta{}.dat'.format(theta_file_suffix), np.concatenate([train_theta,train_pdf[:,None]], axis = 1 ))
	np.savetxt('files/valid_theta{}.dat'.format(theta_file_suffix), np.concatenate([valid_theta,valid_pdf[:,None]], axis = 1 ))
else:
	train_theta = np.loadtxt('files/train_theta.dat'.format(theta_file_suffix))
	train_theta, train_pdf = train_theta[:,:-1], train_theta[:,-1]

	valid_theta = np.loadtxt('files/train_theta.dat'.format(theta_file_suffix))
	valid_theta, valid_pdf = valid_theta[:,:-1], valid_theta[:,-1]

train_w = train_pdf/np.sqrt(np.linalg.det(t_obj.get_metric(train_theta)))
valid_w = valid_pdf/np.sqrt(np.linalg.det(t_obj.get_metric(valid_theta)))

#Training flow with IS
flow = STD_GW_Flow(4, n_layers = n_layers, hidden_features = hidden_features)
optimizer = optim.Adam(flow.parameters(), lr=0.001)
history = flow.train_flow_importance_sampling(1000, train_theta, train_w, valid_theta, valid_w, optimizer,
	batch_size = None, validation_step = 30, callback = None, verbose = True)


compare_probability_distribution(flow.sample(1000).detach().numpy(), data_true = t_obj.sample_from_tiling(1000), variable_format = variable_format, title = None, hue_labels = ('flow', 'tiling'), savefile = None, show = False)
#compare_probability_distribution(t_obj.flow.sample(1000).detach().numpy(), data_true = t_obj.sample_from_tiling(1000), variable_format = variable_format, title = None, hue_labels = ('flow tiling', 'tiling'), savefile = None, show = False)

#Computing the discrepancy between tiling and flow in terms of PDF!!

volume_element_flow_std = np.sqrt(np.linalg.det(t_obj.get_metric(valid_theta, flow = True)))
volume_element_noflow = np.sqrt(np.linalg.det(t_obj.get_metric(valid_theta, flow = False)))

t_obj.flow = flow
volume_element_flow = np.sqrt(np.linalg.det(t_obj.get_metric(valid_theta, flow = True)))+1e-10
log_pdf_flow = flow.log_prob(valid_theta.astype(np.float32)).detach().numpy()
log_pdf_flow = log_pdf_flow - log_pdf_flow[100] + valid_pdf[100]
bins = int(np.sqrt(len(volume_element_noflow)))

plt.figure()
plt.hist(np.log10(volume_element_flow/valid_pdf), bins = bins, histtype = 'step', label = 'flow')
#plt.hist(np.log10(log_pdf_flow/valid_pdf), bins = bins, histtype = 'step', label = 'flow log_pdf')
plt.hist(np.log10(volume_element_flow_std/valid_pdf), bins = bins, histtype = 'step', label = 'flow tiling')
plt.hist(np.log10(volume_element_noflow/valid_pdf), bins = bins, histtype = 'step', label = 'no flow')
plt.legend()
plt.show()
















