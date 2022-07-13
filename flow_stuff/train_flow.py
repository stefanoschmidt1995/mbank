import sys
sys.path.insert(0,'..')
import os

from mbank.flow import GW_Flow, TanhTransform
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

from build_flow import Std2DTransform, Std3DTransform, Test5DTransform
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

	args, _ = parser.parse_known_args()

	train = args.train
	
	#datafile = '../runs/out_mcmc_2D/chain_mcmc_example.dat'; variable_format = 'logMq_nonspinning'
	#datafile = '../../miscellanea/norm_flow/data/chain_mcmc_3D_example.dat'; variable_format = 'Mq_chi'
	
	#datafile = 'data/samples_mcmc_2D.dat'; variable_format = 'logMq_nonspinning' ; dirname = 'standard_flow_2D/'
	#datafile = 'data/samples_mcmc_3D.dat'; variable_format = 'logMq_chi' ; dirname = 'standard_flow_3D/'
	datafile = 'data/samples_mcmc_5D.dat'; variable_format = 'logMq_s1xz_s2z'; dirname = 'test_flow_5D/'

	if not os.path.exists(dirname): os.makedirs(dirname)
	
		#Loading data
	data = np.loadtxt(datafile)
	#data = np.delete(data, np.where(data[:,1]<1.5), axis =0) #This is just for 2D metric... :D
	data = np.delete(data, np.where(data[:,1]>6,), axis =0) #This is just for 5D metric...
	data = np.delete(data, np.where(data[:,2]>0.7,), axis =0) #This is just for 5D metric...
	
	data[:,0] = np.log10(data[:,0])
	N, D = data.shape

	N_epochs = 8000
	train_factor = 0.85
	train_data, validation_data = data[:int(train_factor*N),:], data[int(train_factor*N):,:]

	base_dist = StandardNormal(shape=[D])
	#transform = Std2DTransform()
	#transform = Std3DTransform()
	transform = Test5DTransform()
	
		#####
		## Summary
	print("Saving stuff to folder: ", dirname)
	print("Training data: ", datafile)
	print("Number of data (train|valid): ", train_data.shape[0], validation_data.shape[0])
	print("Dimensionality of the data: ", train_data.shape[1])
	print("Variable format: ", variable_format)
	
		#####
		# Training the model
	flow = GW_Flow(transform=transform, distribution=base_dist)
	optimizer = optim.Adam(flow.parameters(), lr=0.0001)

		#training the flow
	if train:
		history = flow.train_flow(N_epochs=N_epochs, train_data=train_data, validation_data=validation_data,
			batch_size = None,
			optimizer=optimizer,
			#callback = (my_callback, 50), 
			verbose = True)

		my_callback(flow, len(history['train_loss']))
		torch.save(flow.state_dict(), dirname+'weights')
		with open(dirname+'history.pkl', 'wb') as f:
			pickle.dump(history, f)
	else:
		with open(dirname+'history.pkl', 'rb') as f:
			history = pickle.load(f)
		flow.load_state_dict(torch.load(dirname+'weights'))

	create_gif(dirname+'img', dirname+'/training_history.gif', fps = 1)
	
		#plotting and validation data
	plotting_callback(flow, len(history['train_loss']), dirname, train_data[:4000,:], variable_format, basefilename = 'train')
	plotting_callback(flow, len(history['train_loss']), dirname, validation_data[:4000,:], variable_format, basefilename = 'validation')
	
	plot_loss_functions(history, savefolder = dirname)
	plt.show()

	plt.show()
























