"""
This script defines a bunch of normalizing flow models that can be easily built and loaded.
The architectures are hard coded and can be easily retrieved.

"""

import sys
sys.path.insert(0,'..')
import os

from mbank.flow import GW_Flow, TanhTransform
from mbank.flow.utils import compare_probability_distribution, plot_loss_functions

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.nonlinearities import Sigmoid, LeakyReLU
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform,  MaskedPiecewiseLinearAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.linear import NaiveLinear

from nflows.flows.realnvp import SimpleRealNVP
from torch.nn import functional as F

import imageio.v2 as iio

import re
import warnings
warnings.simplefilter('ignore', UserWarning)

###########################################################################################

class Std2DTransform(CompositeTransform):
	def __init__(self):
		transform_list = self.get_transformation_list()
		super().__init__(transform_list)
		return
		
	def get_transformation_list(self):
		"""
		This defines the architecture of the flow.
		"""
		D = 2
		N_layers = 15

		low, high = [0.98, 1.42], [np.log10(40)+0.01, 7.06]

		base_dist = StandardNormal(shape=[D])
		
		transform_list = []

		transform_list.append(TanhTransform(low=low, high=high))

		for _ in range(N_layers):
			
			transform_list.append(NaiveLinear(features=D))
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))
			
				#Another possible architecture	
			#transform_list.append(ReversePermutation(features=D))
			#transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))
		return transform_list

class Std3DTransform(CompositeTransform):
	def __init__(self):
		transform_list = self.get_transformation_list()
		super().__init__(transform_list)
		return
		
	def get_transformation_list(self):
		"""
		This defines the architecture of the flow.
		"""
		
		D = 3
		N_layers = 15

		low, high = [0.98, 0.98, -0.93], [np.log10(40)+0.01, 7.06, 0.93]

		transform_list = []
		transform_list.append(TanhTransform(low=low, high=high))

		for _ in range(N_layers):
			
			transform_list.append(NaiveLinear(features=D))
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))
			
				#Another possible architecture	
			#transform_list.append(ReversePermutation(features=D))
			#transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))
		return transform_list

class Test5DTransform(CompositeTransform):
	def __init__(self):
		transform_list = self.get_transformation_list()
		super().__init__(transform_list)
		return
		
	def get_transformation_list(self):
		"""
		This defines the architecture of the flow.
		"""
		
		D = 5
		N_layers = 20

		low, high = [0.98, 0.98, -0.05, -0.05, -0.93], [np.log10(40)+0.01, 7.06, 0.93, np.pi+0.01, 0.93]

		transform_list = []
		transform_list.append(TanhTransform(low=low, high=high))

		for _ in range(N_layers):
			
			#transform_list.append(NaiveLinear(features=D))
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=2*D))

		return transform_list

class GW_SimpleRealNVP(SimpleRealNVP, GW_Flow):
	def __init__(
		self,
		features,
		hidden_features,
		num_layers,
		num_blocks_per_layer,
		use_volume_preserving=False,
		activation=F.relu,
		dropout_probability=0.0,
		batch_norm_within_layers=False,
		batch_norm_between_layers=False,
	):
		SimpleRealNVP.__init__(self, features, hidden_features,	num_layers, num_blocks_per_layer,
			use_volume_preserving, activation, dropout_probability, batch_norm_within_layers, batch_norm_between_layers)
		
		return
	#def train_flow(self, **kwargs):
	#	return GW_Flow.train_flow(self, **kwargs)






















