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
	from torch.autograd.functional import jacobian

	from nflows.flows.base import Flow
	from nflows.distributions.base import Distribution
	from nflows.distributions.normal import StandardNormal
	from nflows.utils import torchutils
	from nflows.transforms.base import InputOutsideDomain
	from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
	from nflows.transforms.linear import NaiveLinear

	from nflows.transforms.base import Transform, CompositeTransform
except:
	#raise ImportError("Unable to find packages `torch` and/or `nflows`: try installing them with `pip install torch nflows`.")
	msg = "Unable to find packages `torch` and/or `nflows`: you will not be able to use the flow functionalities.\nIf you want to use them, try `pip install torch nflows`."
	warnings.warn(msg)

from .utils import ks_metric, cross_entropy_metric

import re

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
		th_inputs = torch.tanh(inputs)
		outputs = (th_inputs*(self.high-self.low)+self.high+self.low)/2
		logabsdet = torch.log((1 - th_inputs ** 2)*(self.high-self.low)*0.5)
		logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
		return outputs, logabsdet

	def forward(self, inputs, context=None):
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
	
	def save_weigths(self, filename):
		"""
		Saves the weigths to filename.
		It is equivalent to:
		
		::
		
			torch.save(self.state_dict(), filename)

		Parameters
		----------
			filename: str
				Name of the file to save the weights at
		"""
		#TODO: this should be called save_weights
		torch.save(self.state_dict(), filename)

	
	def load_weigths(self, filename):
		"""
		Load the weigths of the flow from file. The weights must match the architecture of the flow.
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
	
	#TODO: train_data and validation_data should be called train_samples and validation_samples!!
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
			epsilon_ = 0.01 #TODO: tune this number: previously it was 0.2
			diff = high-low
			assert torch.all(torch.abs(diff)>1e-20), "The training set has at least one degenerate dimension! Unable to continue with the training"
			low = low - diff*epsilon_
			high = high + diff*epsilon_

			for param, val in zip(self._transform._transforms[0].parameters(), [low, high]):
				param.data = val
			print("params: ",[p.data for p in self._transform._transforms[0].parameters()])
			print("train low high ", [torch.min(train_data, axis = 0)[0],  torch.max(train_data, axis = 0)[0]])
			print("val low high ",[torch.min(validation_data, axis = 0)[0],
						torch.max(validation_data, axis = 0)[0]])

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
	
	def train_flow_metric(self, N_epochs, train_data, validation_data, optimizer, batch_size = None, validation_step = 10, alpha = 100, callback = None, validation_metric = 'cross_entropy', verbose = False):
		"Train the flow with PDF and metric"
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
			print("params: ",[p.data for p in self._transform._transforms[0].parameters()])
			print("train low high ", [torch.min(train_samples, axis = 0)[0],  torch.max(train_samples, axis = 0)[0]])
			print("val low high ",[torch.min(validation_samples, axis = 0)[0],
						torch.max(validation_samples, axis = 0)[0]])

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
	
	@classmethod
	def load_flow(cls, weigth_file):
		w = torch.load(weigth_file)
		try:
			D = w['_transform._transforms.0.low'].shape[0]
			n_layers = re.findall(r'\d+', list(w.keys())[-1])
			assert len(n_layers)==1
			n_layers = int(n_layers[0])//2

			hidden_features = '_transform._transforms.{}.autoregressive_net.final_layer.mask'.format(2*n_layers)
			hidden_features = w[hidden_features].shape[1]
		except:
			raise ValueError("The given weight file does not match the architecture of a `STD_GW_Flow`")
		new_flow = cls(D, n_layers, hidden_features)
		new_flow.load_state_dict(w)
		return new_flow
		
		

class GW_Flow_Uniform(GW_Flow):
	"""
	An implementation of the standard flow with an uniform distribution, instead of a random normal
	"""
	def __init__(self, D, n_layers, hidden_features = 2):
		"""
		Initialization of the flow
		"""
		#FIXME: this is a scam and does not work!!
		from uniform import BoxUniform
		#base_dist = BoxUniform(low=torch.zeros(D), high = torch.ones(1))
		base_dist = BoxUniform([D])
		
		transform_list = []#[TanhTransform(D)]

		for _ in range(n_layers):
			transform_list.append(NaiveLinear(features=D))
				#FIXME: are you sure you want D hidden features?
			transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=hidden_features))

		transform_list.append(InverseTanhTransform(D))
		
		transform_list = CompositeTransform(transform_list)

		super().__init__(transform=transform_list, distribution=base_dist)

			#Initializing the noise transform
		#low = torch.zeros(D)
		#high = torch.ones(D)

		#for param, val in zip(self._transform._transforms[-1].parameters(), [low, high]):
		#	param.data = val

		return














