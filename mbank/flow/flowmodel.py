"""
mbank.flow.flowmodel
====================
		This module implements the basic normalizing flow model, useful for sampling from the Binary Black Hole parameter space.
		It requires `torch` and `nflows`, which are not among the `mbank` dependencies.
"""

import numpy as np
import random #garbage
from tqdm import tqdm 

try:
	import torch
	from torch import distributions
	from torch.utils.data import DataLoader, random_split
	from torch.distributions.utils import broadcast_all

	from nflows.flows.base import Flow
	from nflows.distributions.base import Distribution
	from nflows.distributions.uniform import BoxUniform
	from nflows.utils import torchutils
	from nflows.transforms.base import InputOutsideDomain

	from nflows.transforms.base import Transform
except:
	raise ImportError("Unable to find packages `torch` and/or `nflows`: try installing them with `pip install torch nflows`.")

from .utils import ks_metric

########################################################################

class TanhTransform(Transform):
	"""
	Implements the Tanh transformation. This maps a Rectangle [low, high] into a R^D.
	It is *very* recommended to use this as the last layer of every flow you will ever train on GW data.
	"""
	def __init__(self, low, high):
		"""
		Initialize the transformation.
		
		Parameters
		----------
			low: torch.tensor
				Low corner of the rectangle
			high: torch.tensor
				High corner of the rectangle
		"""
		super().__init__()
		self.low = torch.tensor(low, dtype=torch.float32)
		self.high = torch.tensor(high, dtype=torch.float32)
	
	def inverse(self, inputs, context=None):
		inputs = torch.tanh(inputs)
		inputs = inputs.mul(self.high-self.low)
		inputs = inputs.add(self.high+self.low)
		outputs = inputs.div(2)
		logabsdet = torch.log(1 - outputs ** 2)
		logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
		return outputs, logabsdet

	def forward(self, inputs, context=None):
		inputs = inputs.mul(2)
		inputs = inputs.add(-self.high-self.low)
		inputs = inputs.div(self.high-self.low)
		if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
			raise InputOutsideDomain()
		outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
		logabsdet = -torch.log(1 - inputs ** 2)
		logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
		return outputs, logabsdet


class GW_Flow(Flow):
	
	def __init__(self, transform, distribution):
		"""
		Constructor.
		
		Args:
			transform: nflows.transforms.base.Transform
				A bijection that transforms data into noise (in the `nflows` style)
			distribution: nflows.distributions.base.Distribution
				The base distribution of the flow that generates the noise (in the `nflows` style)
		"""
		super().__init__(transform=transform, distribution=distribution)
		
	def train_flow(self, N_epochs, train_data, validation_data, optimizer, batch_size = None, validation_step = 10, callback = None, verbose = False):
		"""
		Trains the flow.
		
		Parameters
		----------
			N_epochs: int
				Number of training epochs
		
		Returns
		-------
			
		
		"""
		if isinstance(callback, tuple): callback, callback_step = callback
		else: callback_step = 1
		
			#Are you sure you want float32?
		train_data = torch.tensor(train_data, dtype=torch.float32)
		validation_data = torch.tensor(validation_data, dtype=torch.float32)
		
		N_train = train_data.shape[0]

		metric_computer = ks_metric(validation_data, self, 1000)
		val_loss=[]
		train_loss=[]
		metric = [] #Kolmogorov–Smirnov metric (kind of)
				
		desc_str = 'Training loop - loss: {:5f}|{:5f}'
		if verbose: it = tqdm(range(N_epochs), desc = desc_str.format(np.inf, np.inf))
		else: it = range(N_epochs)
		
		try:
			for i in it:

				if isinstance(batch_size, int):
					ids_ = torch.randperm(N_train)[:batch_size]
				else:
					ids_ = range(N_train)


				optimizer.zero_grad()
				loss = -self.log_prob(inputs=train_data[ids_,:]).mean()
				loss.backward()
				optimizer.step()

				train_loss.append(loss.item())


				if not (i%validation_step):
					with torch.no_grad():			
						loss = -self.log_prob(inputs=validation_data).mean()
					val_loss.append(loss)
					
					metric.append(metric_computer.get_metric())
					
				if callable(callback) and not (i%callback_step): callback(self, i)
				if not (i%int(N_train//1000+1)) and verbose: it.set_description(desc_str.format(train_loss[i], val_loss[-1]))
		except KeyboardInterrupt:
			print("KeyboardInterrupt: quitting the training loop")

		history = {'validation_step': validation_step,
			'train_loss': np.array(train_loss),
			'validation_loss': np.array(val_loss),
			'log_pvalue': np.array(metric),
			'log_pvalue_mean': metric_computer.metric_mean,
			'log_pvalue_std': metric_computer.metric_std
		}

		return history












