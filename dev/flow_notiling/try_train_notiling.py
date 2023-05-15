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
from mbank.utils import split_boundaries

from torch import optim

import os
import subprocess

def get_metric_determinant(theta, m_obj, **kwargs):
	N, N_batch = theta.shape[0], 50
	metric_list = []
	overlap = kwargs.pop('overlap', False)
		
	for i in tqdm(range(0, N, N_batch)):
		try:
			metric_list.append(
				m_obj.get_metric_determinant(theta[i:i+N_batch], overlap = overlap, **kwargs)
			)
		except ValueError:
			metric_list.append(
				np.zeros((theta[i:i+N_batch].shape[0],))+np.nan
			)
			continue
			
	return np.concatenate(metric_list, axis = 0)

##########################################################################################################################

compute_tiling = not True
compute_thetas = not True

variable_format = 'Mq_s1xz_s2z_iota'
theta_file_suffix ='_'+variable_format
boundaries = np.array([[10, 1, 0., -np.pi, -0.9, 0], [100, 5, 0.9, np.pi, 0.9, np.pi]])
f_min, f_max = 15, 1024.
psd_file = 'aligo_O3actual_H1.txt'
if not os.path.isfile(psd_file):
	subprocess.run('wget https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt', shell = True)
f, PSD = mbank.utils.load_PSD(psd_file, True, 'H1')
m_obj = mbank.cbc_metric(variable_format, (f,PSD), 'IMRPhenomXP', f_min = f_min, f_max = f_max)

n_layers, hidden_features = 10, 20

	####
	# Loading & computing stuff
if compute_tiling:
	t_obj = mbank.tiling_handler()
	boundaries_list = split_boundaries(boundaries, [1,1,1,4,2,1])
	t_obj = t_obj.create_tiling_from_list(boundaries_list, 0.1, m_obj.get_hessian_symphony, use_ray = True, max_depth = 8, verbose = True)
	t_obj.train_flow(N_epochs=1500, N_train_data = 200000, n_layers=n_layers, hidden_features = hidden_features)
	t_obj.save('files/tiling.npy', 'files/flow.zip')
else:
	t_obj = mbank.tiling_handler('files/tiling.npy')
	t_obj.load_flow('files/flow.zip')

if compute_thetas:

		#Checking if the metric is right...
	metric_from_tiling = t_obj[0].metric
	metric_from_obj = m_obj.get_metric(t_obj[0].center, metric_type = 'symphony')
	assert np.allclose(metric_from_tiling, metric_from_obj, atol = 0., rtol = 1e-6), "The metric is not compatibile with the tiling!"

	train_theta = t_obj.sample_from_tiling(500_000)
	valid_theta = t_obj.sample_from_tiling(50_000)
	#train_theta = np.random.uniform(*boundaries, (20000,3))
	#valid_theta = np.random.uniform(*boundaries, (2000,3))
	train_pdf = np.sqrt(get_metric_determinant(train_theta, m_obj, metric_type = 'symphony', overlap = False))
	valid_pdf = np.sqrt(get_metric_determinant(valid_theta, m_obj, metric_type = 'symphony', overlap = False))
	ids_train_ok, = np.where(~np.isnan(train_pdf))
	ids_valid_ok, = np.where(~np.isnan(valid_pdf))

	np.savetxt('files/train_theta{}.dat'.format(theta_file_suffix),
		np.concatenate([train_theta[ids_train_ok],train_pdf[ids_train_ok,None]], axis = 1 ))
	np.savetxt('files/valid_theta{}.dat'.format(theta_file_suffix),
		np.concatenate([valid_theta[ids_valid_ok],valid_pdf[ids_valid_ok,None]], axis = 1 ))
else:
	train_theta = np.loadtxt('files/train_theta{}.dat'.format(theta_file_suffix))
	np.random.shuffle(train_theta)
	train_theta, train_pdf = train_theta[:,:-1], train_theta[:,-1]

	valid_theta = np.loadtxt('files/valid_theta{}.dat'.format(theta_file_suffix))#[:2000]
	np.random.shuffle(valid_theta)
	valid_theta, valid_pdf = valid_theta[:,:-1], valid_theta[:,-1]

#n_layers, hidden_features = t_obj.flow.n_layers, t_obj.flow.hidden_features

train_w = train_pdf/np.sqrt(np.linalg.det(t_obj.get_metric(train_theta, kdtree = True)))
valid_w = valid_pdf/np.sqrt(np.linalg.det(t_obj.get_metric(valid_theta, kdtree = True)))

assert np.all(~np.isnan(train_w)) and np.all(~np.isnan(valid_w))

#resample_theta_ids = np.random.choice(len(valid_theta), size = 1000, p = valid_w/np.sum(valid_w), replace = False)
#compare_probability_distribution(valid_theta[resample_theta_ids], data_true = valid_theta[:1000],
#	hue_labels=('resampling', 'tiling'),
#	variable_format = variable_format, title = None,
#	savefile = 'files/pdf_resampling_{}.png'.format(theta_file_suffix), show = True)

#Training flow with IS
print('Flow architecure: #layers | #hidden features ', n_layers, hidden_features)
flow = STD_GW_Flow(train_theta.shape[-1], n_layers = n_layers, hidden_features = hidden_features)
optimizer = optim.Adam(flow.parameters(), lr=0.0005)
#history = flow.train_flow_forward_KL(1000, train_theta, valid_theta, optimizer,
#	batch_size = 5000, validation_step = 30, callback = None, verbose = True)
history = flow.train_flow_importance_sampling(1000, train_theta, train_w, valid_theta, valid_w, optimizer,
	batch_size = 50000, validation_step = 30, callback = None, verbose = True)


if False:
	old_train_w = train_pdf/np.sqrt(np.linalg.det(t_obj.get_metric(train_theta, kdtree= True, flow = True)))
	t_obj.flow = flow
	new_train_w = train_pdf/np.sqrt(np.linalg.det(t_obj.get_metric(train_theta, kdtree= True, flow = True)))
	plt.figure()
	plt.hist(np.log10(train_w), bins = 1000, label = 'no flow', histtype = 'step')
	plt.hist(np.log10(old_train_w), bins = 1000, label = 'old weights', histtype = 'step')
	plt.hist(np.log10(new_train_w), bins = 1000, label = 'new weights', histtype = 'step')
	plt.legend()
	plt.show()
	quit()

	#Deleting the dataset
del train_theta, train_w

#compare_probability_distribution(flow.sample(1000).detach().numpy(), data_true = t_obj.sample_from_tiling(1000),
#	variable_format = variable_format, title = None, hue_labels = ('flow', 'tiling'),
#	savefile = 'files/pdf_comparison_{}.png'.format(theta_file_suffix), show = False)

#compare_probability_distribution(t_obj.flow.sample(1000).detach().numpy(), data_true = t_obj.sample_from_tiling(1000), variable_format = variable_format, title = None, hue_labels = ('flow tiling', 'tiling'), savefile = None, show = False)

#Computing the discrepancy between tiling and flow in terms of PDF!!

volume_element_flow_std = np.sqrt(np.linalg.det(t_obj.get_metric(valid_theta, flow = True, kdtree= True)))
volume_element_noflow = np.sqrt(np.linalg.det(t_obj.get_metric(valid_theta, flow = False, kdtree= True)))

t_obj.flow = flow
volume_element_flow = np.sqrt(np.linalg.det(t_obj.get_metric(valid_theta, flow = True, kdtree= True)))+1e-10
log_pdf_flow = flow.log_prob(valid_theta.astype(np.float32)).detach().numpy()
log_pdf_flow = log_pdf_flow - log_pdf_flow[10] + valid_pdf[10]
bins = int(np.sqrt(len(volume_element_noflow)))

plt.figure()
plt.hist(np.log10(volume_element_flow/valid_pdf), bins = bins, histtype = 'step', label = 'flow')
#plt.hist(np.log10(log_pdf_flow/valid_pdf), bins = bins, histtype = 'step', label = 'flow log_pdf')
plt.hist(np.log10(volume_element_flow_std/valid_pdf), bins = bins, histtype = 'step', label = 'flow tiling')
plt.hist(np.log10(volume_element_noflow/valid_pdf), bins = bins, histtype = 'step', label = 'no flow')
plt.legend()
plt.savefig('files/comparison_hist_{}.png'.format(theta_file_suffix))
plt.show()
















