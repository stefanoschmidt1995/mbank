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
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.linear import NaiveLinear

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

		low, high = [10, 1.3], [40, 7]

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

		low, high = [10, 1.3, -1], [40, 7, 1]

		transform_list = []
		transform_list.append(TanhTransform(low=low, high=high))

		for _ in range(N_layers):
			
			transform_list.append(NaiveLinear(features=D))
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))
			
				#Another possible architecture	
			#transform_list.append(ReversePermutation(features=D))
			#transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=D))
		return transform_list






