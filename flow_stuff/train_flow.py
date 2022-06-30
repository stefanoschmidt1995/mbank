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

from build_flow import Std2DTransform, Std3DTransform

#############

def my_callback(model, epoch):
	return plotting_callback(model, epoch, dirname, validation_data[:500,:], variable_format)
	
#############

if __name__ == '__main__':

	dirname = 'out_test_3D/'
	datafile = '../runs/out_mcmc_2D/chain_mcmc_example.dat'; variable_format = 'Mq_nonspinning'
	datafile = '../../miscellanea/norm_flow/data/chain_mcmc_3D_example.dat'; variable_format = 'Mq_chi'
	
	#datafile = 'data/samples_mcmc_2D.dat'; variable_format = 'Mq_nonspinning'
	#datafile = 'data/samples_mcmc_3D.dat'; variable_format = 'Mq_chi'

	data = np.loadtxt(datafile)
	data = np.delete(data, np.where(data[:,1]<1.3), axis =0)
	#data[:,0] = np.log10(data[:,0])
	N, D = data.shape

	N_epochs = 600
	train_factor = 0.85
	train_data, validation_data = data[:int(train_factor*N),:], data[int(train_factor*N):,:]

	base_dist = StandardNormal(shape=[D])
	#transform = Std2DTransform()
	transform = Std3DTransform()
	
		#####
		# Training the model
	flow = GW_Flow(transform=transform, distribution=base_dist)
	optimizer = optim.Adam(flow.parameters(), lr=0.001)

		#training the flow
	train_loss, val_loss, val_metric = flow.train_flow(N_epochs=N_epochs, train_data=train_data, validation_data=validation_data,
		batch_size= None,
		optimizer=optimizer,
		callback = (my_callback, 50), 
		verbose = True)

	torch.save(flow.state_dict(), dirname+'weights')

	my_callback(flow, N_epochs)

	create_gif(dirname, dirname+'train.gif', fps = 1)

	plot_loss_functions(train_loss, val_loss, val_metric, savefolder = None)

	plt.show()
























