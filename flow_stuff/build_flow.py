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

from torch.nn import functional as F

from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
	AdditiveCouplingTransform,
	AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm

import imageio.v2 as iio

import re
import warnings

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
		N_layers = 10

		base_dist = StandardNormal(shape=[D])
		
		transform_list = []

		transform_list.append(TanhTransform(D))

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
		N_layers = 10

		transform_list = []
		transform_list.append(TanhTransform(D))

		for _ in range(N_layers):
			
			transform_list.append(NaiveLinear(features=D))
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))
			
				#Another possible architecture	
			#transform_list.append(ReversePermutation(features=D))
			#transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))
		return transform_list

class Std6DTransform(CompositeTransform):
	def __init__(self):
		transform_list = self.get_transformation_list()
		super().__init__(transform_list)
		return
		
	def get_transformation_list(self):
		"""
		This defines the architecture of the flow.
		"""
		
		D = 6
		N_layers = 15

		#logMq_s1xz_s2z_iota
		transform_list = []
		transform_list.append(TanhTransform(D))

		for _ in range(N_layers):
			
			transform_list.append(NaiveLinear(features=D))
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))

		return transform_list



class Std5DTransform(CompositeTransform):
	def __init__(self):
		transform_list = self.get_transformation_list()
		super().__init__(transform_list)
		return
		
	def get_transformation_list(self):
		"""
		This defines the architecture of the flow.
		"""
		
		D = 5
		N_layers = 10

		transform_list = []
		transform_list.append(TanhTransform(D))

		for _ in range(N_layers):
			
			#transform_list.append(NaiveLinear(features=D))
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=2*D))
			
		return transform_list

class Std8DTransform(CompositeTransform):
	def __init__(self):
		transform_list = self.get_transformation_list()
		super().__init__(transform_list)
		return
		
	def get_transformation_list(self):
		"""
		This defines the architecture of the flow.
		"""
		
		D = 8
		N_layers = 10

		transform_list = []
		transform_list.append(TanhTransform(D))

		for _ in range(N_layers):
			
			#transform_list.append(NaiveLinear(features=D))
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=2*D))

		return transform_list

class GW_SimpleRealNVP(GW_Flow):
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
		#very ugly copy & paste from: https://github.com/bayesiains/nflows/blob/master/nflows/flows/realnvp.py#L17
		
		if use_volume_preserving:
			coupling_constructor = AdditiveCouplingTransform
		else:
			coupling_constructor = AffineCouplingTransform

		mask = torch.ones(features)
		mask[::2] = -1

		def create_resnet(in_features, out_features):
			return nets.ResidualNet(
				in_features,
				out_features,
				hidden_features=hidden_features,
				num_blocks=num_blocks_per_layer,
				activation=activation,
				dropout_probability=dropout_probability,
				use_batch_norm=batch_norm_within_layers,
			)

		layers = [TanhTransform(features)]
		for _ in range(num_layers):
			transform = coupling_constructor(
				mask=mask, transform_net_create_fn=create_resnet
			)
			layers.append(transform)
			mask *= -1
			if batch_norm_between_layers:
				layers.append(BatchNorm(features=features))
		
		super().__init__(
			transform=CompositeTransform(layers),
			distribution=StandardNormal([features]),
		)
		
		return






















