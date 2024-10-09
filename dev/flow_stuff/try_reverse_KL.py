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

from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform,  MaskedPiecewiseLinearAutoregressiveTransform
from nflows.transforms.linear import NaiveLinear
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal
from torch.distributions.multivariate_normal import MultivariateNormal

from scipy.stats import multivariate_normal

import re
import warnings
warnings.simplefilter('ignore', UserWarning)

from build_flow import Std2DTransform, Std3DTransform, Test5DTransform, GW_SimpleRealNVP
import pickle

import argparse
#################################

folder = 'test_reverse_KL/'

	#####
	# Instatiating the distribution
cov = torch.tensor([[1,.3],[.3,3]], dtype=torch.float32)
mu = torch.tensor([1,0], dtype=torch.float32)
#distribution = multivariate_normal(mean=[1,0], cov= cov, allow_singular=False)
distribution = MultivariateNormal(mu, cov)

validation_data = distribution.rsample([2000]).detach().numpy()
D = validation_data.shape[-1]

def target_logpdf(x):
	logpdf = -0.5*torch.einsum('...j, ...jk, ...k -> ...', torch.atleast_2d(x-mu), torch.inverse(cov), torch.atleast_2d(x-mu)) - 0.5*torch.log( np.power(2*np.pi,D)*torch.det(cov))
	return logpdf
#print(target_logpdf(mu), distribution.log_prob(mu))

base_dist = StandardNormal(shape=[D])
transform = []

#transform.append(TanhTransform(low=low, high=high))

for _ in range(10):
			
	transform.append(NaiveLinear(features=D))
	transform.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))
	
transform = CompositeTransform(transform)

	#####
	# Training the model
flow = GW_Flow(transform=transform, distribution=base_dist)
optimizer = optim.Adam(flow.parameters(), lr=0.001)

N_epochs = 1000

flow.train_flow_reverse_KL(N_epochs=N_epochs, target_logpdf=target_logpdf, validation_data=validation_data,
			batch_size = 10000,
			optimizer=optimizer,
			#callback = (my_callback, 50), 
			verbose = True)


	
	
	
	
	
	
	
	
	
	
	
	
	
	

