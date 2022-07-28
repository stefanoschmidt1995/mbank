"""
mbank.flow.flowmodel
====================

		This module implements the basic normalizing flow model, useful for sampling from the Binary Black Hole parameter space.
		It requires `torch` and `nflows`, which are not among the `mbank` dependencies.
"""

import numpy as np
from tqdm import tqdm 
import warnings

try:
	import torch
	from torch import distributions
	from torch.utils.data import DataLoader, random_split
	from torch.distributions.utils import broadcast_all

	from nflows.flows.base import Flow
	from nflows.distributions.base import Distribution
	from nflows.distributions.normal import StandardNormal
	from nflows.utils import torchutils
	from nflows.transforms.base import InputOutsideDomain
	from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
	from nflows.transforms.linear import NaiveLinear

	from nflows.transforms.base import Transform, CompositeTransform
except:
	raise ImportError("Unable to find packages `torch` and/or `nflows`: try installing them with `pip install torch nflows`.")

from .utils import ks_metric, cross_entropy_metric

########################################################################

class TanhTransform(Transform):
	"""
	Implements the Tanh transformation. This maps a Rectangle [low, high] into a R^D.
	It is *very* recommended to use this as the last layer of every flow you will ever train on GW data.
	"""
	def __init__(self, D):
		"""
		Initialize the transformation.
		
		Parameters
		----------
			D: int
				Dimensionality of the space
		"""
		super().__init__()
			#Placeholders for the true values
			#They will be fitted as a first thing in the training procedure
		self.low = torch.nn.Parameter(torch.randn([D], dtype=torch.float32), requires_grad = False)
		self.high = torch.nn.Parameter(torch.randn([D], dtype=torch.float32), requires_grad = False)
	
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

############################################################################################################
############################################################################################################

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
		
	def train_flow_forward_KL(self, N_epochs, train_data, validation_data, optimizer, batch_size = None, validation_step = 10, callback = None, validation_metric = 'cross_entropy', verbose = False):
		"""
		Trains the flow with forward KL: see eq. (13) of `1912.02762 <https://arxiv.org/abs/1912.02762>`_.
		
		Parameters
		----------
			N_epochs: int
				Number of training epochs

			validation_metric: str
				Name for the validation metric to use: options are `cross_entropy` and `ks` (Kolmogorov-Smirnov). Default is cross entropy
		
		Returns
		-------
			history: dict
				A dictionary keeping the historical values of training & validation loss function + KS metric.
				It has the following entries:
					- validation_step: number of epochs between two consecutive evaluation of the validation metrics
					- train_loss: values of the loss function
		
		"""
		#TODO: implement early stopping!!
		#FIXME: there's something weird with cross entropy: why do you exceed the threshold even though the loss function still goes down?
		if isinstance(callback, tuple): callback, callback_step = callback
		else: callback_step = 10
		
			#Are you sure you want float32?
		train_data = torch.tensor(train_data, dtype=torch.float32)
		validation_data = torch.tensor(validation_data, dtype=torch.float32)
		
		N_train = train_data.shape[0]
		
				#Dealing with the first transformation
				#Setting low and high
		if isinstance(self._transform._transforms[0], TanhTransform):
			low, _ = torch.min(train_data, axis = 0)
			high, _ = torch.max(train_data, axis = 0)
				#the interval between low and high is made larger by a factor epsilon
			epsilon_ = 0.02
			low = low*(1-torch.sign(low)*epsilon_)
			high = high*(1+torch.sign(high)*epsilon_)

			for param, val in zip(self._transform._transforms[0].parameters(), [low, high]):
				param.data = val
			#print("params: ",[p.data for p in self._transform._transforms[0].parameters()])
			#print("train low high ", [torch.min(train_data, axis = 0)[0],  torch.max(train_data, axis = 0)[0]])
			#print("val low high ",[torch.min(validation_data, axis = 0)[0],
			#			torch.max(validation_data, axis = 0)[0]])

		else:
			msg = "The first layer is not of the kind 'TanhTransform': although this will not break anything, it is *really* recommended to have it as a first layer. It is much needed to transform bounded data into unbounded ones."
			warnings.warn(msg)

		if validation_metric == 'ks': metric_computer = ks_metric(validation_data, self, 1000)
		elif validation_metric == 'cross_entropy': metric_computer = cross_entropy_metric(validation_data, self, 1000)
		else: raise ValueError("Wrong value '{}' given for the validation metric: please choose either 'ks' or 'cross_entropy'.".format(validation_metric))
		
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
					
					metric.append(metric_computer.get_validation_metric())
					
				if callable(callback) and not (i%callback_step): callback(self, i)
				if not (i%int(N_epochs//1000+1)) and verbose: it.set_description(desc_str.format(train_loss[i], val_loss[-1]))
		except KeyboardInterrupt:
			print("KeyboardInterrupt: quitting the training loop")

		#TODO: change the name of log_pvalue to validation_metric (or something)
		history = {'validation_step': validation_step,
			'train_loss': np.array(train_loss),
			'validation_loss': np.array(val_loss),
			'log_pvalue': np.array(metric),
			'log_pvalue_mean': metric_computer.metric_mean,
			'log_pvalue_std': metric_computer.metric_std,
			'metric_type': validation_metric
		}

		return history
	
	def get_distance(self, theta1, theta2, thetac, metric):
		"""
		Computes the squared distance between ```theta1``` and ```theta2``` using the metric evaluated at ```thetac``` and the flow.
		"""
		theta1, theta2 = np.atleast_2d(theta1), np.atleast_2d(theta2)
		weight_factor = self.integrate_flow(torch.tensor(theta1, dtype=torch.float32), 
					torch.tensor(theta2, dtype=torch.float32),
					torch.tensor(thetac, dtype=torch.float32),
					N_steps = 100, d = 1).detach().numpy()
		diff = theta2-theta1
		dist = np.einsum('ij,jk,ik->i', diff, metric, diff)
		print('\t',weight_factor, dist, dist*weight_factor)
		dist = dist*weight_factor
		print('\t', 1-(np.exp(dist)-1))

		#return dist
		return np.exp(dist)-1 #better, as does not go crazy!



	def integrate_flow(self, theta1, theta2, thetac, N_steps = 100, d = 1):
		"""
		It performs the following integral:
		
		.. math::
			\int_{0}^{1} \\text{d}t \, \left(\\frac{|M_{\\text{flow}}(\\theta(t))|}{|M_{\\text{flow}}(\\theta_C)|} \\right)^d
		
		where
		
		.. math::
			\\theta(t) = \\theta_1 + (\\theta_2 -\\theta_1) t
			
		and where :math:`|M_{\\text{flow}}(\\theta)|` is estimated by the flow.
		
		It is useful to weigth the metric distance obtained by assuming a constant metric (i.e. the standard way).
		
		Parameters
		----------
			theta1: torch.tensor
				shape (D,)/(N,D) -
				Starting point of the line integral
			
			theta2: torch.tensor
				shape (D,)/(N,D) -
				Ending point of the line integral

			thetac: torch.tensor
				shape (D,)/(N,D) -
				Center which the metric is compared at
			
			N_steps: int
				The number of points to be used for integral estimation
			
			d: float
				Exponent appearing in the integral
		
		Returns
		-------
			integral: torch.tensor
				shape (1,)/(N,) -
				The result of the integral
		"""
		#TODO: should this be a member of GW_flow? It would make sense...

		assert (theta1.ndim >= thetac.ndim) or (theta2.ndim >= thetac.ndim), "The center should have a number of dimensions less or equal than the extrema"

		theta1, theta2 = torch.atleast_2d(theta1), torch.atleast_2d(theta2)
		steps = theta1 + torch.einsum("ij,k->kij", theta2-theta1, torch.linspace(0, 1, N_steps))
		#steps = theta1 + torch.outer(theta2-theta1, torch.linspace(0, 1, N_steps))

		old_shape = steps.shape
		steps = torch.flatten(steps, 0, 1) #(N*N_steps, 3)
		
		log_pdfs = self.log_prob(steps) #(N*N_steps, )
		log_pdf_center = self.log_prob(torch.atleast_2d(thetac)) #(N, )/()
		
		log_pdfs = torch.reshape(log_pdfs, old_shape[:-1]) #(N_steps, N, )
		log_pdfs = log_pdfs - log_pdf_center #(N_steps, N, )

		det_M = torch.pow(torch.exp(log_pdfs), 2*d) #(N*N_steps, )
		
		integral = torch.trapezoid(det_M, dx =1/N_steps, axis =0)
		
		return integral








	def train_flow_reverse_KL(self, N_epochs, target_logpdf, optimizer, batch_size = 1000, validation_data =None, validation_step = 10, callback = None, out_dir=None, verbose = False):
		"""
		Trains the flow with reverse KL: see eq. (17) of `1912.02762 <https://arxiv.org/abs/1912.02762>`_.
		
		Parameters
		----------
			N_epochs: int
				Number of training epochs
		
		Returns
		-------
			history: dict
				A dictionary keeping the historical values of training & validation loss function + KS metric.
				It has the following entries:
					- validation_step: number of epochs between two consecutive evaluation of the validation metrics
					- train_loss: values of the loss function
		"""
		
		msg = "Fitting with reverse KL is not feasible as it requires the evaluation of the GRADIENT of the true PDF!\n"
		msg += "Can you express the gradients of the metric determinant in terms of the first derivatives only of the WFs? I really doubt that, as the grad|M| tells something about curvature, which depends on the second derivative (at least in GR).\n"
		msg += "Moreover, also with a simple gaussian, there seems to be a huuuge problem of overflow: the flow is not able to identify the interesting region.\n"
		msg += "Take home message: use always the forward KL!"
		
		raise NotImplementedError(msg)
		
		if isinstance(callback, tuple): callback, callback_step = callback
		else: callback_step = 10

		if isinstance(validation_data, np.ndarray) or isinstance(validation_data, torch.tensor):
			validation_data = torch.tensor(validation_data, dtype=torch.float32)
			metric_computer = ks_metric(validation_data, self, 1000)
			val_loss=[]
			metric = [] #Kolmogorov–Smirnov metric (kind of)
			do_validation = True
		else:
			do_validation = False

		train_loss=[]
				
		desc_str = 'Training loop - loss: {:5f}|{:5f}'
		if verbose: it = tqdm(range(N_epochs), desc = desc_str.format(np.inf, np.inf if do_validation else ''))
		else: it = range(N_epochs)
		
		try:
			for i in it:
			
				samples, log_prob_flow = self.sample_and_log_prob(batch_size)
				log_prob_true = target_logpdf(samples)

				loss = -(log_prob_flow - log_prob_true).mean()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				train_loss.append(loss.item())


				if do_validation and not (i%validation_step):
					with torch.no_grad():			
						loss = -self.log_prob(inputs=validation_data).mean()
					val_loss.append(loss)
					
					metric.append(metric_computer.get_validation_metric())
					
				if callable(callback) and not (i%callback_step): callback(self, i)
				if not (i%int(N_epochs//1000+1)) and verbose: it.set_description(desc_str.format(train_loss[i], val_loss[-1]))
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


############################################################################################################
############################################################################################################

class STD_GW_Flow(GW_Flow):
	"""
	An implementation of the standard flow: the flow is composed by one TanhTransform layer and a stack of layers made by NaiveLinear+MaskedAffineAutoregressiveTransform
	"""
	def __init__(self, D, n_layers, hidden_features = 2):
		"""
		Initialization of the flow
		"""
		base_dist = StandardNormal(shape=[D])
		
		transform_list = [TanhTransform(D)]

		for _ in range(n_layers):
			transform_list.append(NaiveLinear(features=D))
				#FIXME: are you sure you want D hidden features?
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=hidden_features))
		transform_list = CompositeTransform(transform_list)
		
		super().__init__(transform=transform_list, distribution=base_dist)
		return















