import sys
sys.path.insert(0,'..')
import os

from mbank.flow import GW_Flow, TanhTransform, STD_GW_Flow
from mbank.flow.utils import plot_loss_functions, create_gif, plotting_callback

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from nflows.distributions.normal import StandardNormal

import re
import warnings
warnings.simplefilter('ignore', UserWarning)

from build_flow import Std2DTransform, Std3DTransform, Std6DTransform, Std5DTransform, Std8DTransform, GW_SimpleRealNVP
import pickle

import argparse

#############

def my_callback(model, epoch):
	return plotting_callback(model, epoch, dirname+'img/', validation_data, variable_format)
	
#############

if __name__ == '__main__':

	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument(
	"--train",  default = False, action='store_true',
	help="Whether to train the model. If not present, the model is loaded and some plots will be made.")
	parser.add_argument(
	"--n-dim",  type=int, required = True,
	help="Number of dimensions of the flow.")
	parser.add_argument(
	"--dirname",  type=str,
	help="Name of the directory to store the model. If not given, it will be set to 'standard_flow_nD' with n being the number of dimensions.")
	parser.add_argument(
	"--dataset",  type=str,
	help="Path to dataset. If not given, a default based on `n-dim` will be provided.")
	parser.add_argument(
	"--variable-format",  type=str,
	help="Variable format. If not given, a default based on --n-dim will be set")	
	parser.add_argument(
	"--N-epochs",  type=int, default = 2000,
	help="Number of training epochs")
	parser.add_argument(
	"--callback-step",  type = int, default = 0,
	help="Whether to produce plots during the training. Default is 0, meaning no callback is called")
	

	args, _ = parser.parse_known_args()

	train = args.train

	#datafile = '../runs/out_mcmc_2D_Taylor/chain_mcmc_example.dat'; variable_format = 'logMq_nonspinning'; dirname = 'test_flow_2D/'
	#datafile = '../../miscellanea/norm_flow/data/chain_mcmc_3D_example.dat'; variable_format = 'Mq_chi'

	if args.n_dim ==2:
		datafile = 'data/samples_mcmc_2D.dat'; variable_format = 'logMq_nonspinning' ; dirname = 'standard_flow_2D/'
		n_layers, hidden_features = 10, 2
	elif args.n_dim ==3:
		datafile = 'data/samples_mcmc_3D.dat'; variable_format = 'logMq_chi' ; dirname = 'standard_flow_3D/'
		n_layers, hidden_features = 10, 3
	elif args.n_dim ==4:
		datafile = 'data/samples_mcmc_4D.dat'; variable_format = 'logMq_s1xz' ; dirname = 'standard_flow_4D/'
		transform = None
		raise NotImplementedError("Please define a transformation here")
	elif args.n_dim == 5:
		datafile = 'data/samples_mcmc_5D_lowq.dat'; variable_format = 'logMq_s1xz_s2z' ; dirname = 'standard_flow_5D/'
		transform = Std5DTransform()
	elif args.n_dim == 6:
		datafile = 'data/samples_mcmc_6D.dat'; variable_format = 'logMq_s1xz_s2z_iota' ; dirname = 'standard_flow_6D/'
		transform = Std6DTransform()
	elif args.n_dim == 8:
		datafile = 'data/samples_mcmc_8D.dat'; variable_format = 'logMq_fullspins' ; dirname = 'standard_flow_8D/'
		transform = Std8DTransform()

	if isinstance(args.variable_format, str): variable_format = args.variable_format
	if isinstance(args.dataset, str): datafile = args.dataset
	if isinstance(args.dirname, str): dirname = args.dirname
	if not dirname.endswith('/'): dirname = dirname+'/'
	if not os.path.exists(dirname): os.makedirs(dirname)
	
		#Loading data
	data = np.loadtxt(datafile)
	np.random.shuffle(data)
	
	data[:,0] = np.log10(data[:,0])
	N, D = data.shape
	
	assert D==args.n_dim, "Wrong dimensionality for the file given"

	train_factor = 0.9
	train_data, validation_data = data[:int(train_factor*N),:], data[int(train_factor*N):,:]

		#####
		## Summary
	print("Saving stuff to folder: ", dirname)
	print("Training data: ", datafile)
	print("Number of data (train|valid): ", train_data.shape[0], validation_data.shape[0])
	print("Dimensionality of the data: ", train_data.shape[1])
	print("Variable format: ", variable_format)
	print("Boundaries of the training set: min|max\n\t{}\n\t{}".format(np.min(data, axis =0), np.max(data, axis =0)))
	
		#####
		# Training the model
	#base_dist = StandardNormal(shape=[D])
	#flow = GW_Flow(transform=transform, distribution=base_dist)
	flow = STD_GW_Flow(D, n_layers, hidden_features)
	#flow = GW_SimpleRealNVP(D, hidden_features = 4, num_layers = 2, num_blocks_per_layer = 2)
	optimizer = optim.Adam(flow.parameters(), lr=0.001)

		#training the flow
	if train:
		history = flow.train_flow_forward_KL(N_epochs=args.N_epochs, train_data=train_data, validation_data=validation_data,
			batch_size = None,
			optimizer=optimizer,
			callback = (my_callback, args.callback_step) if args.callback_step>0 else None, 
			validation_metric = 'cross_entropy',
			verbose = True)

		my_callback(flow, len(history['train_loss']))
		torch.save(flow.state_dict(), dirname+'weights')
		with open(dirname+'history.pkl', 'wb') as f:
			pickle.dump(history, f)
		print('Saved weights')
	else:
		with open(dirname+'history.pkl', 'rb') as f:
			history = pickle.load(f)
		#flow.load_state_dict(torch.load(dirname+'weights'))
		flow.load(dirname+'weights')

	create_gif(dirname+'img', dirname+'/training_history.gif', fps = 1)
	
		#plotting train and validation data
	#plotting_callback(flow, len(history['train_loss']), dirname, train_data[:4000,:], variable_format, basefilename = 'train')
	#plotting_callback(flow, len(history['train_loss']), dirname, validation_data[:4000,:], variable_format, basefilename = 'validation')
	#print("params: ",[p.data for p in flow._transform._transforms[0].parameters()])
	
	plot_loss_functions(history, savefolder = dirname)

	plt.show()
























