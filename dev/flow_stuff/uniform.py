"Implementation of a uniform base distribution."
#If this turns to be elegant, we can consider a patch for the normflow code

import numpy as np
import torch
from torch import nn

from nflows.distributions.base import Distribution
from nflows.utils import torchutils

class BoxUniform(Distribution):
	"""A Uniform distribution in the square [low, high]^D"""

	def __init__(self, shape, low = 0., high = 1.):
		super().__init__()
		self._shape = torch.Size(shape)
		self.dist = torch.distributions.Uniform(
					low=torch.tensor([low]*self._shape[0], dtype = torch.float32), 
					high=torch.tensor([high]*self._shape[0], dtype = torch.float32)
			)


		self.register_buffer("_log_z",
							 torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
										  dtype=torch.float64),
							 persistent=False)

	def _log_prob(self, inputs, context):
		# Note: the context is ignored.
		if inputs.shape[1:] != self._shape:
			raise ValueError(
				"Expected input of shape {}, got {}".format(
					self._shape, inputs.shape[1:]
				)
			)

		log_prob = self.dist.log_prob(inputs)
		log_prob = torch.sum(log_prob, dim = 1)

		return log_prob

	def _sample(self, num_samples, context):
		if context is None:
			return self.dist.sample([num_samples]).to(self._log_z.device)
		else:
			raise NotImplementedError("Cannot sample here with a context!")

	def _mean(self, context):
		raise NotImplementedError("Please do the mean")
		if context is None:
			return torch.tensor((self.dist.low+self.dist.high)/2.).to(self._log_z.device)
			#return self._log_z.new_zeros(self._shape)
		else:
			# The value of the context is ignored, only its size is taken into account.
			raise NotImplementedError("Cannot use here a context!")
