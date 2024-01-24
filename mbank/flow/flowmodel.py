"""
mbank.flow.flowmodel
====================

Implements the basic normalizing flow model, useful for sampling from the Binary Black Hole parameter space.
It requires `torch` and `glasflow.nflows`, which are not among the `mbank` dependencies.
"""

import numpy as np
from tqdm import tqdm 
import warnings

import torch
from torch import distributions
from torch.utils.data import DataLoader, random_split
from torch.distributions.utils import broadcast_all
from torch.autograd.functional import jacobian

from glasflow.nflows.flows.base import Flow
from glasflow.nflows.distributions.base import Distribution
from glasflow.nflows.distributions.normal import StandardNormal
from glasflow.nflows.utils import torchutils
from glasflow.nflows.transforms.base import InputOutsideDomain
from glasflow.nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from glasflow.nflows.transforms.linear import NaiveLinear

from glasflow.nflows.transforms.base import Transform, CompositeTransform

from .utils import ks_metric, cross_entropy_metric

import re

########################################################################

class powerlawTransform(Transform):
	"""
	Implements a trainable power law transformation for each dimension of the flow
	"""
	def __init__(self, D, alphas = None, low = None, high = None):
		"""
		Initialize the transformation.
		
		Parameters
		----------
			f_low: float
				Lower frequency cut-off for the transformation
		"""
		super().__init__()
		for l in [alphas, low, high]:
			if l is None: l = torch.randn([D], dtype=torch.float32)

		self.alphas = torch.nn.Parameter(torch.tensor(alphas, dtype=torch.float32), requires_grad = True)
		self.set_boundaries(low, high)
		
	
	def set_boundaries(self, low, high):
		print(low, high)
		low = torch.tensor(low, dtype=torch.float32)
		high = torch.tensor(high, dtype=torch.float32)
		self.diff =  high - low
		low = low - self.diff*1e-1
		high = high + self.diff*1e-1		
		self.low = torch.nn.Parameter(low, requires_grad = False)
		self.high = torch.nn.Parameter(high, requires_grad = False)
		
		print('###### Boundaries')
		print(self.low, self.high, self.alphas)
		
	def inverse(self, inputs, context=None):
		outputs = torch.pow(inputs, 1/self.alphas)*self.diff + self.low
		logabsdet = -torch.sum(torch.log(torch.abs(self.alphas/self.diff))) + torch.sum((1/self.alphas-1)*torch.log(torch.abs(inputs)), dim = -1)
		return outputs, logabsdet

	def forward(self, inputs, context=None):
		scaled_inputs = (inputs-self.low)/self.diff
		outputs = torch.pow(scaled_inputs, self.alphas)
		logabsdet = torch.sum(torch.log(torch.abs(self.alphas/self.diff))) + torch.sum((self.alphas-1)*torch.log(torch.abs(scaled_inputs)), dim = -1)
		return outputs, logabsdet

class tau0tau3Transform(Transform):
	"""
	Implements the tau0tau3 transformation as in https://arxiv.org/pdf/0706.4437.pdf
	"""
	def __init__(self):
		"""
		Initialize the transformation.
		
		Parameters
		----------
			f_low: float
				Lower frequency cut-off for the transformation
		"""
		super().__init__()
		f_low = 15
		self.A0 = 1#5./(256*(np.pi * f_low)**(8./3))  # eqn B3
		self.A3 = 0.05#np.pi/(8*(np.pi*f_low)**(5./3))  # eqn B3
		self.logabsdet_prefactor = np.log(self.A0*self.A3)
	
	def logabsdet_fwd(self, mc, eta):
		return -10/3.*torch.log(mc)-8/5.*torch.log(eta)
	
	def inverse(self, inputs, context=None):
		tau0, tau3 = inputs[:,0], inputs[:,1]
		mc = torch.pow(tau0/self.A0, -3/5.)
		eta = torch.pow(tau0/self.A0, 2/3.)*torch.pow(tau3/self.A3, -5/3.)
		logabsdet = - self.logabsdet_fwd(mc, eta)
		outputs = torch.stack([mc, eta, *inputs[:,2:].T], dim = 1)
		return outputs, logabsdet

	def forward(self, inputs, context=None):
		mc, eta = inputs[:,0], inputs[:,1]
		tau0 = self.A0*torch.pow(mc, -5/3.)
		tau3 = self.A3*torch.pow(mc, -2/3.)*torch.pow(eta, -3/5.)
		logabsdet = self.logabsdet_fwd(mc, eta)
		outputs = torch.stack([tau0, tau3, *inputs[:,2:].T], dim = 1)
		return outputs, logabsdet

class TanhTransform(Transform):
	"""
	Implements the Tanh transformation. This maps a Rectangle [low, high] into R^D.
	It is *very* recommended to use this as the last layer of every flow you will ever train on GW data.
	"""
	def __init__(self, D, low = None, high = None):
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
		if low is None:
			self.low = torch.nn.Parameter(torch.randn([D], dtype=torch.float32), requires_grad = False)
		else:
			self.low = torch.nn.Parameter(torch.tensor(low, dtype=torch.float32), requires_grad = False)
		if high is None:
			self.high = torch.nn.Parameter(torch.randn([D], dtype=torch.float32), requires_grad = False)
		else:
			self.high = torch.nn.Parameter(torch.tensor(high, dtype=torch.float32), requires_grad = False)
	
	def inverse(self, inputs, context=None):
		th_inputs = torch.tanh(inputs)
		outputs = (th_inputs*(self.high-self.low)+self.high+self.low)/2
		logabsdet = torch.log((1 - th_inputs ** 2)*(self.high-self.low)*0.5)
		logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
		return outputs, logabsdet

	def forward(self, inputs, context=None):
		inside = torch.logical_and(torch.prod(inputs>self.low, dim = -1), torch.prod(inputs<self.high, dim = -1))
		inputs = inputs.mul(2)
		inputs = inputs.add(-self.high-self.low)
		inputs = inputs.div(self.high-self.low)

		if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
			raise InputOutsideDomain()
		outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
		logabsdet = -torch.log((1 - inputs ** 2)*0.5*(self.high-self.low))
		logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
		return outputs, logabsdet

############################################################################################################
############################################################################################################

class GW_Flow(Flow):
	"""
	A Normalizing flow suitable to reproduce the uniform distribution over the parameter space.
	It offers an interface to loading and saving the model as well as to the training.
	"""
	
	def __init__(self, transform, distribution, has_constant = True):
		"""
		Constructor.
		
		Parameters
		----------
			transform: glasflow.nflows.transforms.base.Transform
				A bijection that transforms data into noise (in the ``glasflow.nflows`` style)
			distribution: glasflow.nflows.distributions.base.Distribution
				The base distribution of the flow that generates the noise (in the ``glasflow.nflows`` style)
		"""
		super().__init__(transform=transform, distribution=distribution)
		if has_constant:
			self.constant = torch.nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad = True)
		else: self.constant = 0.
		
		self.loss_dict = {
			'forward_KL':self.forward_KL_loss,
			'll_mse': self.ll_mse_loss,
			'weighted_ll_mse': self.weighted_ll_mse_loss
		}
		self.D = distribution.sample(1).shape[-1]
	
	@property
	def available_losses(self):
		return list(self.loss_dict.keys())
	
	def save_weigths(self, filename):
		"""
		Saves the weigths to filename.
		It is equivalent to:
		
		::
		
			torch.save(self.state_dict(), filename)

		Parameters
		----------
			filename: str
				Name of the file to save the weights at (a file-like object)
		"""
		#TODO: this should be called save_weights
		torch.save(self.state_dict(), filename)

	
	def load_weights(self, filename):
		"""
		Load the weights of the flow from file. The weights must match the architecture of the flow.
		It is equivalent to
		
		::
		
			self.load_state_dict(torch.load(filename))
		
		Parameters
		----------
			filename: str
				File to load the weights from
		"""
		#TODO: this should be called load_weights
		try:
			self.load_state_dict(torch.load(filename))
		except:
			msg = "The given weigth file does not match the architecture of this flow model."
			raise ValueError(msg)
		return
	
	def sample_within_boundaries(self, num_samples, boundaries = None, seed = None):
		"""
		Generate a number of samples within the given boundaries
		"""
		if isinstance(seed, int):
			torch.manual_seed(seed)
			
		if boundaries is None:
			return self.sample_and_log_prob(num_samples)
		
		num_samples_per_trial = min(num_samples, 3000)

		with torch.no_grad():		
			samples, log_prob = None, None
			while True:
				new_points, new_log_prob = self.sample_and_log_prob(num_samples_per_trial)
				ids_ = boundaries(new_points)
				if sum(ids_)>0:
					samples = new_points[ids_] if samples is None else torch.cat([samples, new_points[ids_]], dim = 0)
					log_prob = new_log_prob[ids_] if log_prob is None else torch.cat([log_prob, new_log_prob[ids_]], dim = 0)
					
					if samples.shape[0]>=num_samples: break
		
		return samples[:num_samples], log_prob[:num_samples]
		
	
	def forward_KL_loss(self, data, weights):
		return -self.log_prob(inputs=data).mean()

	def ll_mse_loss(self, data, weights):
		return torch.square(self.log_prob(inputs=data) - weights + self.constant).mean()

	def weighted_ll_mse_loss(self, data, weights):
		w = torch.sqrt(weights - torch.min(weights))
		#w /= torch.sum(w)
		return torch.square((self.log_prob(inputs=data) - weights + self.constant)*w).mean()

	def get_jacobian(self, theta):
		theta = torch.tensor(theta, dtype = torch.float32)
		theta = torch.atleast_2d(theta)
	
		jac_fun = lambda x: self._transform.forward(torch.t(x))[0]
		noise = self.transform_to_noise(theta)
		p_u = torch.exp(self._distribution.log_prob(noise)) #(N,)
		jac_list = []
		for n in theta:
			jac_ = jacobian(jac_fun, n[:, None])[0,:,:,0]
			jac_list.append(jac_)
		jac = torch.stack(jac_list, axis = 0)
		M_flow = torch.einsum('k, kia, kja -> kij', p_u, jac, jac)
		
		return M_flow

	def train_flow(self, loss, N_epochs, train_data, validation_data, optimizer, train_weights = None, validation_weights = None, batch_size = None, validation_step = 10, callback = None, lr_scheduler = None, boundaries = None, verbose = False):
		"""
		Trains the normalizing flow.
		
		It can use several loss functions:
			- 'forward_KL': uses the forward KL entropy. In this case, train_data and validation_data must be *samples* and train_weights and validation_weights will be ignored. See eq. (13) of `1912.02762 <https://arxiv.org/abs/1912.02762>`_.
			- 'll_mse': uses the means squared error of the log likelihood between the flow and the target distribution. In this case, train_weights and validation_weights must be provided and have the meaning of log likelihood of each train_data and validation_data respectively. The data doesn't need to be drawn from any particular distribution
		
		The training supports callbacks and learning rate decay. Early stop can be implemented through callbacks.

		
		Parameters
		----------
			N_epochs: int
				Number of training epochs
			
			train_data: :class:`~numpy:numpy.ndarray`
				Training data. They need to fit the dimensionality of the flow and be convertible into a torch tensor
			
			validation_data: :class:`~numpy:numpy.ndarray`
				Validation data. They need to fit the dimensionality of the flow and be convertible into a torch tensor

			optimizer: torch.optim
				An torch optimizer object. Typical usage is:
				
				.. code-block:: python
				
					from torch import optim
					flow = GW_Flow(**args)
					optimizer = optim.Adam(flow.parameters(), lr=0.001)

			train_weights: :class:`~numpy:numpy.ndarray`
				Weights for each of the training data
			
			validation_weights: :class:`~numpy:numpy.ndarray`
				Weights for each of the validation data

			batch_size: int
				Batch size: number of points of the training set to be used at each iteration
			
			validation_step: int
				Number of training steps after which the validation metric is computed
			
			callback: callable
				A callable, called at each validation step of the training.
				It has to have call signature ``callback(GW_Flow, epoch)``: see :func:`mbank.flow.utils.plotting_callback` for an example.
			
			lr_scheduler: torch.optim.lr_scheduler
				A torch learning rate scheduler. Typical example would be:
				
				.. code-block:: python
				
					from torch import optim
					flow = GW_Flow(**args)
					optimizer = optim.Adam(flow.parameters(), lr=0.001)
					scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

			
			boundaries: :class:`~numpy:numpy.ndarray`
				shape: (2,D) -
				Rectangular boundaries for the dataset: the flow will only be able to sample points within this rectangle.
				If not given, its value will be inferred from the data.

			verbose: bool
				Whether to print a progress bar during the training

		Returns
		-------
			history: dict
				A dictionary keeping the historical values of training & validation loss function + KS metric.
				It has the following entries:
				
				- `validation_step`: number of epochs between two consecutive evaluation of the validation metrics
				- `train_loss`: values of the loss function at each iteration
				- `validation_loss`: values of the validation loss function at each validation iteration
				- `valmetric_value`: values of the validation metric at each validation iteration
				- `valmetric_mean`: expected value of the validation metric for a perfectly trained flow
				- `valmetric_std`: standard deviation of the validation metric for a perfectly trained flow
				- `validation_metric`: name of the validation metric being used
		"""
		#TODO: implement early stopping!!
		
			#Are you sure you want float32?
		train_data = torch.tensor(train_data, dtype=torch.float32)
		validation_data = torch.tensor(validation_data, dtype=torch.float32)
		if train_weights is not None: train_weights = torch.squeeze(torch.tensor(train_weights, dtype=torch.float32)) #squeeze is suuuuper important!
		if validation_weights is not None: validation_weights = torch.squeeze(torch.tensor(validation_weights, dtype=torch.float32))
		
		N_train = train_data.shape[0]
		if not isinstance(batch_size, int):
			batch_size = N_train//10
		
				#Dealing with the first transformation
				#Setting low and high
		if isinstance(self._transform._transforms[0], TanhTransform):
			if boundaries is None:
				low, _ = torch.min(train_data, axis = 0)
				high, _ = torch.max(train_data, axis = 0)
				#high[0] = high[0]*1.1 #DEBUG
				#warnings.warn("Exploration for high limits of the flow :D")
			else:
				assert np.asarray(boundaries).shape == (2, self.D), "Wrong shape for the boundaries of the training flow"
				low, high = torch.Tensor(boundaries[0]), torch.Tensor(boundaries[1])
				#the interval between low and high is made larger by a factor epsilon

			epsilon_ = 0.01 #TODO: tune this number!! That's very very hard to set. Maybe you can make it trainable?
			diff = high-low
			assert torch.all(torch.abs(diff)>1e-20), "The training set has at least one degenerate dimension! Unable to continue with the training"
			low = low - diff*epsilon_
			high = high + diff*epsilon_

			for param, val in zip(self._transform._transforms[0].parameters(), [low, high]):
				param.data = val
			#print("params: ",[p.data for p in self._transform._transforms[0].parameters()])
			#print("train low high ", [torch.min(train_data, axis = 0)[0],  torch.max(train_data, axis = 0)[0]])
			#print("val low high ",[torch.min(validation_data, axis = 0)[0],
			#			torch.max(validation_data, axis = 0)[0]])

		else:
			msg = "The first layer is not of the kind 'TanhTransform': although this will not break anything, it is *really* recommended to have it as a first layer. It is much needed to transform bounded data into unbounded ones."
			warnings.warn(msg)

		val_loss=[]
		train_loss=[]
		metric = [] #Kolmogorov–Smirnov metric (kind of)
				
		if isinstance(self.constant, torch.Tensor):
			#It's usually a good idea to set the constant to a large number. The training will go to the minimum faster
			#warnings.warn('Scaling the LLs')
			#validation_weights = validation_weights - torch.max(train_weights) + 0
			#train_weights = train_weights - torch.max(train_weights) + 0
			with torch.no_grad():
				self.constant[0] = torch.quantile(train_weights, 0.9).item()
			#self.constant.requires_grad = False
			if verbose: print('Initialing scaling constant to: ', self.constant[0].item())
		
		desc_str = 'Training loop - lr: {:2f} - loss: {:5f}|{:5f}'
		it = tqdm(range(N_epochs), desc = desc_str.format(optimizer.state_dict()['param_groups'][0]['lr'], np.inf, np.inf), disable = not verbose)
		
		try:
			for i in it:

				if isinstance(batch_size, int):
					ids_ = torch.randperm(N_train)[:batch_size]
				else:
					ids_ = range(N_train)

				optimizer.zero_grad()
				loss_ = self.loss_dict[loss](train_data[ids_,:], None if train_weights is None else train_weights[ids_])
				#with torch.no_grad():
				#	print('###', i)
				#	print('\tnan in prediction: ',np.any(np.isnan(self.log_prob(inputs=train_data[ids_,:]).numpy())))
				#	print('\tnan in dataset: ',np.any(np.isnan(train_weights[ids_].numpy())))
				#	print('\tnan in loss: ',np.any(np.isnan(loss_.numpy())))
				loss_.backward()
				optimizer.step()

				train_loss.append(loss_.item())
				
				if not (i%validation_step):

					with torch.no_grad():		
						loss_ = self.loss_dict[loss](validation_data, validation_weights)
					val_loss.append(loss_)

					if lr_scheduler: lr_scheduler.step(loss_)
					
					if callable(callback):
						if callback(self, i, train_loss[-1], val_loss[-1]): break
	
				if not (i%int(N_epochs//1000+1)) and verbose:
					lr_ = optimizer.state_dict()['param_groups'][0]['lr']
					it.set_description(desc_str.format(lr_, train_loss[i], val_loss[-1]))
		except KeyboardInterrupt:
			if verbose: print("KeyboardInterrupt: quitting the training loop")

		history = {'validation_step': validation_step,
			'train_loss': np.array(train_loss),
			'validation_loss': np.array(val_loss),
			'valmetric_value': np.array(metric),
		}

		return history

	
	def train_flow_metric(self, N_epochs, train_data, validation_data, optimizer, batch_size = None, validation_step = 10, alpha = 100, callback = None, validation_metric = 'cross_entropy', verbose = False):
		#"Train the flow with PDF and metric"
		#raise NotImplementedError("Implement a nice and viable loss function :D")
		if isinstance(callback, tuple): callback, callback_step = callback
		else: callback_step = 10
		
			#Setting the data nicely
		train_samples, train_metric_data = train_data
		train_samples = torch.tensor(train_samples, dtype=torch.float32)
		train_metric_data = torch.tensor(train_metric_data, dtype=torch.float32)
		validation_samples, validation_metric_data = validation_data
		validation_samples = torch.tensor(validation_samples, dtype=torch.float32)
		validation_metric_data = torch.tensor(validation_metric_data, dtype=torch.float32)
		
		N_train, D = train_samples.shape
		
				#Dealing with the first transformation
				#Setting low and high
		if isinstance(self._transform._transforms[0], TanhTransform):
			low, _ = torch.min(train_samples, axis = 0)
			high, _ = torch.max(train_samples, axis = 0)
					#the interval between low and high is made larger by a factor epsilon
			epsilon_ = 0.01 #TODO: tune this number: previously it was 0.2
			diff = high-low
			assert torch.all(torch.abs(diff)>1e-20), "The training set has at least one degenerate dimension! Unable to continue with the training"
			low = low - diff*epsilon_
			high = high + diff*epsilon_

			for param, val in zip(self._transform._transforms[0].parameters(), [low, high]):
				param.data = val
			#print("params: ",[p.data for p in self._transform._transforms[0].parameters()])
			#print("train low high ", [torch.min(train_samples, axis = 0)[0],  torch.max(train_samples, axis = 0)[0]])
			#print("val low high ",[torch.min(validation_samples, axis = 0)[0],
			#			torch.max(validation_samples, axis = 0)[0]])

		#else:
		#	msg = "The first layer is not of the kind 'TanhTransform': although this will not break anything, it is *really* recommended to have it as a first layer. It is much needed to transform bounded data into unbounded ones."
		#	warnings.warn(msg)

		if validation_metric == 'ks': metric_computer = ks_metric(validation_samples, self, 1000)
		elif validation_metric == 'cross_entropy': metric_computer = cross_entropy_metric(validation_samples, self, 1000)
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

					#computing the loss metric (tricky)
				jac_fun = lambda x: self._transform.forward(torch.t(x))[0]
				noise = self.transform_to_noise(train_samples[ids_,:])
				p_u = torch.exp(self._distribution.log_prob(noise)) #(N,)
				jac_list = []
				for n in train_samples[ids_,:]:
					jac_ = jacobian(jac_fun, n[:, None])[0,:,:,0]
					jac_list.append(jac_)
				jac = torch.stack(jac_list, axis = 0)
				M_flow = torch.einsum('k, kia, kja -> kij', p_u, jac, jac)
				M_flow = (M_flow.T/torch.pow(torch.linalg.det(M_flow), 1/D)).T
				M_true = (train_metric_data[ids_,...].T/torch.pow(torch.linalg.det(train_metric_data[ids_,...]), 1/D)).T
				#print("True metric: ",M_true[0])
				#print("Flow metric: ",M_flow[0])
				loss_metric = torch.log(torch.linalg.norm(M_true - M_flow, ord = None, dim = [1,2]))

					#Stuff in the noise space
				if False:
					inverse_fun = lambda x: self._transform.inverse(torch.t(x))[0]
					noise = self.transform_to_noise(train_samples[ids_,:])
					jac_list = []
					for n in noise:
						jac_ = jacobian(inverse_fun, n[:, None])[0,:,:,0]
						jac_list.append(jac_)
					jac_inv = torch.stack(jac_list, axis = 0)
					M_transformed = torch.einsum('kij, kai, kbj -> kab', train_metric_data[ids_,...], jac_inv, jac_inv)
					p_u = torch.exp(self._distribution.log_prob(noise))
					I = torch.stack([torch.eye(D)]*len(ids_), axis = 0)
					expected_M_transformed = torch.einsum('i, ijk -> ijk', p_u, I)
					
					print(M_transformed[0], expected_M_transformed[0])
					
					print("ratio sqrt(det): ", torch.sqrt(torch.linalg.det(M_transformed)/torch.linalg.det(expected_M_transformed))[:10])
					A = torch.sqrt(torch.linalg.det(M_transformed)/torch.linalg.det(expected_M_transformed)).mean()
					print(torch.pow(A, 2/D))
					loss_metric = torch.linalg.norm(M_transformed - expected_M_transformed/torch.pow(A, 2/D) , ord = None, dim = [1,2])



					#standard NF loss & summing stuff together
				gamma = torch.tensor(i/alpha)
				#loss = -self.log_prob(inputs=train_samples[ids_,:]).mean() * torch.exp(-gamma)
				loss = loss_metric.mean() #*(1-torch.exp(-gamma))
				#print("Loss_NF, Loss_metric, e^{-gamma}: ", self.log_prob(inputs=train_samples[ids_,:]).mean(), loss_metric.mean(), torch.exp(-gamma))
				loss.backward()
				optimizer.step()
				
				train_loss.append(loss.item())

				if not (i%validation_step):
					with torch.no_grad():			
						loss = -self.log_prob(inputs=validation_samples).mean()
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
	
	def train_flow_reverse_KL(self, N_epochs, target_logpdf, optimizer, batch_size = 1000, validation_data =None, validation_step = 10, callback = None, out_dir=None, verbose = False):
		#"Trains the flow with reverse KL: see eq. (17) of `1912.02762 <https://arxiv.org/abs/1912.02762>`_."
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
	An implementation of the standard flow: the flow is composed by one TanhTransform layer and a stack of layers made by NaiveLinear+MaskedAffineAutoregressiveTransform.
	All the applications within mbank uses of this class.
	"""
	def __init__(self, D, n_layers, hidden_features, has_constant = True):
		"""
		Initialization of the flow
		
		Parameters
		----------
			D: int
				Dimensionality of the flow
			
			n_layers: int
				Number of layers (each made by NaiveLinear+MaskedAffineAutoregressiveTransform)
			
			hidden_features: int
				Number of hidden features of the ``MaskedAffineAutoregressiveTransform``. If a list is given, it is intended to be the number of hidden features for each layer
		"""
		base_dist = StandardNormal(shape=[D])
		
		transform_list = [TanhTransform(D)]

		if isinstance(hidden_features, int):
			hidden_features = [hidden_features]
		if isinstance(hidden_features, list):
			if len(hidden_features) == 1 and n_layers>1:
				hidden_features = hidden_features*n_layers
		assert isinstance(hidden_features, (int, (list, tuple))), "hidden_features must be a int or a list of ints"
		
		for i in range(n_layers):
			transform_list.append(NaiveLinear(features=D))
				#FIXME: are you sure you want D hidden features?
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=hidden_features[i]))
		transform_list = CompositeTransform(transform_list)
		
		super().__init__(transform=transform_list, distribution=base_dist, has_constant = has_constant)
		self.n_layers = n_layers
		self.hidden_features = hidden_features
		self.D = D
		return

	def log_volume_element(self, theta):
		"""
		Returns an estimation to the (log) volume element :math:`\log\sqrt{M(\theta)}` givev by the normalizing flow. This is equivalent to the flow log_pdf, scaled by a constant.
		
		Parameters
		----------
			theta: :class:`~numpy:numpy.ndarray`
				Input points to evaluate the at
		
		Returns
		-------
			log_sqrt_metric: :class:`~numpy:numpy.ndarray`
				The natural logarithm of the volume element :math:`\log\sqrt{M(\theta)}`.
		"""
		theta = torch.tensor(theta, dtype=torch.float32)
		with torch.no_grad():
			log_volume_element = self.log_prob(theta) + self.constant
		
		return log_volume_element.numpy()
	
	@property
	def boundary_box(self):
		"""
		Returns the boundaries of the support of the flow (a rectangle)
		"""
		if isinstance(self._transform._transforms[0], TanhTransform):
			with torch.no_grad():
				bbox = [v.numpy() for v in self._transform._transforms[0].parameters()]
		return np.array(bbox)
		
	
	@classmethod
	def load_flow(cls, weigth_file):
		"""
		Loads the flow from a file. The architecture is inferred from the loaded weights.
		
		.. code-block:: python

			new_flow = STD_GW_Flow.load_flow('weight.zip')
		
		Parameters
		----------
			weigth_file: str
				File to load the flow from
		
		Returns
		-------
			new_flow: STD_GW_Flow
				Initialized flow
		"""
	
		w = torch.load(weigth_file)
		try:
			has_constant = ('constant' in w)
			D = w['_transform._transforms.0.low'].shape[0]
			n_layers = re.findall(r'\d+', list(w.keys())[-1])
			assert len(n_layers)==1
			n_layers = int(n_layers[0])//2

			hidden_features = []
			for k in w.keys():
				if k.find('.autoregressive_net.final_layer.mask') > -1:
					hidden_features.append(w[k].shape[1])
		except:
			raise ValueError("The given weight file does not match the architecture of a `STD_GW_Flow`")
		assert len(hidden_features)==n_layers, "Number of layers and features do not match"
		new_flow = cls(D, n_layers, hidden_features, has_constant)
		new_flow.load_state_dict(w)
		return new_flow

