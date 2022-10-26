"""
mbank.metric
============
	This module implements the metric computation for cbc signals. It provides a class ``cbc_metric`` that, besides the metric computations, offers some functions to compute waveforms and the match between them.
	
	The metric is a D dimensional square matrix that approximates the match between two waveforms. The metric M is defined such that:
	
	.. math::
		<h(\\theta) | h(\\theta + \Delta\\theta) > = 1 - M(\\theta)_{ij} \Delta\\theta_i \\Delta\\theta_j
	
	The metric is a useful local approximation of the match and it is the physical input for the bank generation.
	The explicit expression for the metric is a complicated expression of the gradients of the waveform and it is a function of theta.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import itertools

import lal 
import lalsimulation as lalsim

import warnings

from .handlers import variable_handler
from .utils import project_metric

	#TODO: understand whether it's a good idea to use anycache
#from anycache import anycache

#############DEBUG LINE PROFILING
try:
	from line_profiler import LineProfiler

	def do_profile(follow=[]):
		def inner(func):
			def profiled_func(*args, **kwargs):
				try:
					profiler = LineProfiler()
					profiler.add_function(func)
					for f in follow:
						profiler.add_function(f)
					profiler.enable_by_count()
					return func(*args, **kwargs)
				finally:
					profiler.print_stats()
			return profiled_func
		return inner
except:
	pass

#pip install line_profiler
#add decorator @do_profile(follow=[]) before any function you need to track

####################################################################################################################
####################################################################################################################

####################################################################################################################
####################################################################################################################

class cbc_metric(object):
	"""
	This class implements the metric on the space, defined for each point of the space.
	The metric corresponds to the hessian of the match function, when evaluated around a point.
	Besides the metric computation, this class allows for waveform (WF) generation and for match computation, both with and without metric. 
	"""
	
	def __init__(self, variable_format, PSD, approx, f_min = 10., f_max = None):
		"""
		Initialize the class.
		
		Parameters
		----------
			
		variable_format: string
			How to handle the variables. Different options are possible and which option is set, will decide the dimensionality D of the parameter space (hence of the input).
			Variable format can be changed with ``set_variable_format()`` and can be accessed under name ``cbc_metric.variable_format``. See ``mbank.handlers.variable_handler`` for more details.

		PSD: tuple
			PSD for computing the scalar product.
			It is a tuple with a frequency grid array and a PSD array (both one dimensional and with the same size).
			PSD should be stored in an array which stores its value on a grid of evenly spaced positive frequencies (starting from f0 =0 Hz).

		approx: string
			Which approximant to use. It can be any lal approx.
			The approximant can be changed with set_approximant() and can be accessed under name cbc_metric.approx
		
		f_min: float
			Minimum frequency at which the scalar product is computed (and the WF is generated from)
		
		f_max: float
			Cutoff for the high frequencies in the PSD. If not None, frequencies up to f_max will be removed from the computation
			If None, no cutoff is applied
		
		"""
		self.var_handler = variable_handler() #this obj is to keep in a single place all the possible spin manipulations that may be required

		self.set_approximant(approx)
		self.set_variable_format(variable_format)
		
		self.f_min = float(f_min)
		
		if not isinstance(PSD, tuple):
			raise ValueError("Wrong format for the PSD. Expected a tuple of np.ndarray, got {}".format(type(PSD)))

		#####Tackling the PSD
		#FIXME: do you really need to store the PSD and the frequency grid all the way to zero? It seems like an useless waste of memory/computational time
		self.f_grid = PSD[0]
		self.delta_f = self.f_grid[1]-self.f_grid[0]
			#checking that grid is equally spaced
		assert np.all(np.diff(self.f_grid)-self.delta_f<1e-10), "Frequency grid is not evenly spaced!"
		
		#####applying a high frequency cutoff to the PSD
		if isinstance(f_max, (int, float)): #high frequency cutoff
			self.f_max = f_max
		else:
			self.f_max = self.f_grid[-1]
		self.f_grid = np.linspace(0., self.f_max, int(self.f_max/self.delta_f)+1)
		self.PSD = np.interp(self.f_grid, PSD[0], PSD[1])

		if np.any(self.PSD ==0):
			self.PSD[self.PSD ==0.] = np.inf
			
		self.f_max = float(self.f_max)
			#To compute the sampling rate:
			#https://electronics.stackexchange.com/questions/12407/what-is-the-relation-between-fft-length-and-frequency-resolution
		self.srate = 2*(self.f_grid[1] - self.f_grid[0])*len(self.f_grid)
		
		return
	
	def set_approximant(self, approx):
		"""
		Change the lal approximant used to compute the WF
		
		Parameters
		----------
		
		approx: string
			Which approximant to use. It can be any lal FD approximant.
		"""
			#checking if the approximant is right
		try:
			lal_approx = lalsim.SimInspiralGetApproximantFromString(approx) #getting the approximant
		except RuntimeError:
			raise RuntimeError("Wrong approximant name: it must be an approximant supported by lal")

			#changing the approximant
		self.approx = approx

		return
	
	def set_variable_format(self, variable_format):
		"""
		Set the variable_format to be used.
		See ``mbank.handlers.variable_handler`` for more information.
		
		The following snippet prints the allowed variable formats and to display some information about a given format.
		::
		
			from mbank import handlers
			vh = handlers.variable_handler()
			print(vh.valid_formats)
			print(vh.format_info['Mq_s1xz_s2z_iota'])
		
		Parameters
		----------
		
		variable_format: string
			A string to specify the variable format
		"""
		assert variable_format in self.var_handler.valid_formats, "Wrong variable format '{}'. Available formats are: ".format(variable_format)+str(self.var_handler.valid_formats)
		
		self.variable_format = variable_format
		
		self.D = self.var_handler.format_D[variable_format]

		return
	
	def get_space_dimension(self):
		"""
		Returns the dimensionality `D` of the metric.
		
		Returns
		-------
			D: float
				The dimensionality of the metric
		"""
		return self.D

	def get_volume_element(self, theta, overlap = False):
		"""
		Returns the volume element. It is equivalent to `sqrt{|M(theta)|}`
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on the variable format set for the metric
		
		overlap: bool
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		Returns
		-------
		
		vol_element : :class:`~numpy:numpy.ndarray`
			shape: (N,) -
			Volume element of the metric for the given input
			
		"""
		return np.sqrt(np.abs(np.linalg.det(self.get_metric(theta, overlap = overlap)))) #(N,)
	
	def get_metric_determinant(self, theta, overlap = False):
		"""
		Returns the metric determinant
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on the variable format set for the metric
		
		overlap: bool
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		Returns
		-------
		
		det : :class:`~numpy:numpy.ndarray`
			shape: (N,) -
			Determinant of the metric for the given input
			
		"""
		return np.linalg.det(self.get_metric(theta, overlap = overlap)) #(N,)

	def log_pdf(self, theta, boundaries = None):
		"""
		Returns the logarithm log(pdf) of the probability distribution function we want to sample from:
		.. math::
		
			pdf = p(theta) ~ sqrt(|M(theta)|)

		imposing the boundaries
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			parameters of the BBHs. The dimensionality depends on the variable format set for the metric
		
		boundaries: :class:`~numpy:numpy.ndarray`
			shape: (2,D)
			An optional array with the boundaries for the model. If a point is asked below the limit, -10000000 is returned
			Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
			If None, no boundaries are implemented
		
		Returns
		-------
		
		log_pdf : :class:`~numpy:numpy.ndarray`
			shape: (N,) -
			Logarithm of the pdf, ready to use for sampling
		"""
		theta = np.asarray(theta)
		
		if theta.ndim == 1:
			theta = theta[None,:]
			reshape = True
		else:
			reshape = False
		
		if isinstance(boundaries,np.ndarray):
			if boundaries.shape != (2,self.D):
				raise ValueError("Wrong shape of boundaries given: expected (2,{}), given {}".format(self.D, boundaries.shape))
			
			ids_ok = np.logical_and(np.all(theta > boundaries[0,:], axis =1), np.all(theta < boundaries[1,:], axis = 1)) #(N,)
		else:
			ids_ok = np.full((theta.shape[0],), True)
			
		
		res = np.zeros((theta.shape[0],)) -10000000
		
		if np.any(ids_ok):
			det = self.get_metric_determinant(theta[ids_ok,:])
			det = np.log(np.abs(det))*0.5 #log(sqrt(|det|))
			res[ids_ok] = det
		
		return res
	
	def log_pdf_gauss(self, theta, boundaries = None):
		return -0.5*np.sum(np.square(theta-10.), axis =-1) #DEBUG!!!!!
	
	def get_WF_grads(self, theta, approx = None, order = None, epsilon = 1e-6):
		"""
		Computes the gradient of the WF with a given lal FD approximant. The gradients are computed with finite difference methods.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			parameters of the BBHs. The dimensionality depends on the variable format set for the metric
	
		approx: string
			Which approximant to use. It can be any lal FD approximant
			If None, the default approximant will be used
		
		order: int
			Order of the finite difference scheme for the gradient computation.
			If None, some defaults values will be set, depending on the total mass.
		
		epsilon: list
			Size of the jump for the finite difference scheme for each dimension. If a float, the same jump will be used for all dimensions

		Returns
		-------
		
		grad_h : :class:`~numpy:numpy.ndarray`
			shape: (N, K, D) -
			Complex array holding the gradient of the WFs evaluated on the default frequency/time grid
		"""
		#Take home message, to get a really nice metric:
		# - The order of the integration is crucial. You can set this adaptively, depending on total mass
		# - The espilon is EVEN MORE CRUCIAL
		#FIXME: find a nice way to set some defaults for the finite difference step! Maybe hardcode it inside the variable_handler?
		#FIXME: assert boundaries when computing gradients... Maybe find a preprocessing for the vars that makes every variable unbounded?

		squeeze = False
		theta = np.asarray(theta)
		if theta.ndim == 1:
			theta = theta[None,:]
			squeeze = True
		
		assert order in [None, 1,2,4,6,8], "Wrong order '{}' for the finite difference scheme given: options are 'None' or '{}'".format(order, [1,2,4,6,8])

		def get_WF(theta_value, df_):
			#return self.get_WF_lal(theta_value, approx, df_)[0].data.data
			return self.get_WF(theta_value)

			#setting epsilon value
		if isinstance(epsilon, float):
			epsilon_list = [epsilon for _ in range(12)]
			epsilon_list[0] = epsilon_list[0]/10 #the first variable needs a smaller step?
		elif isinstance(epsilon, (list, np.ndarray)):
			epsilon_list = epsilon
		else:
			raise ValueError("epsilon should be a list or a float")
		
		 	#doing finite difference methods
		delta_ij = lambda i,j: 1 if i ==j else 0
		grad_h_list = []
		
		def get_order(M):
			#TODO:fix the thresholds
			if isinstance(order, int): return order
			if M>50.: order_ = 1
			if M<=50. and M>10.: order_ = 4
			if M<=10. and M>1.: order_ = 6
			if M<=1.: order_ = 8
			#print(M, order_) #DEBUG
			return order_
		
		for theta_ in theta:
			df = self.delta_f#*10.
			order_theta = get_order(theta_[0])
			grad_theta_list = []
			if order_theta == 1: WF = get_WF(theta_, df) #only for forward Euler
			
				#loops on the D dimensions of the theta vector
			for i in range(theta.shape[1]):

				deltax = np.zeros(theta.shape[1])
				deltax[i] = 1.
				epsilon = epsilon_list[i]
				
				#computing WFs
				WF_p = get_WF(theta_ + epsilon * deltax, df)
				if order_theta>1:
					WF_m = get_WF(theta_ - epsilon * deltax, df)
	
				if order_theta>2:
					WF_2p = get_WF(theta_ + 2.*epsilon * deltax, df)
					WF_2m = get_WF(theta_ - 2.*epsilon * deltax, df)
					
				if order_theta>4:
					WF_3p = get_WF(theta_ + 3.*epsilon * deltax, df)
					WF_3m = get_WF(theta_ - 3.*epsilon * deltax, df)				

				if order_theta>6:
					WF_4p = get_WF(theta_ + 4.*epsilon * deltax, df)
					WF_4m = get_WF(theta_ - 4.*epsilon * deltax, df)

				
				#######
				# computing gradients with finite difference method
				# see: https://en.wikipedia.org/wiki/Finite_difference_coefficient

					#forward euler: faster but less accurate
				if order_theta ==1:
					grad_i = (WF_p - WF )/(epsilon) #(N,D) 
					#second order method
				elif order_theta==2:
					grad_i = (WF_p - WF_m )/(2*epsilon) #(N,D)
					#fourth order method
				elif order_theta==4:
					grad_i = (-WF_2p/4. + 2*WF_p - 2.*WF_m + WF_2m/4. )/(3*epsilon) #(N,D)
					#sixth order method
				elif order_theta==6:
					grad_i = (WF_3p -9.*WF_2p + 45.*WF_p \
						- 45.*WF_m + 9.*WF_2m -WF_3m)/(60.*epsilon) #(N,D)
					#eight order method
				elif order_theta==8:
					grad_i = (- WF_4p/56. + (4./21.)*WF_3p - WF_2p + 4.*WF_p \
						- 4. *WF_m + WF_2m - (4./21.)* WF_3m + WF_4m/56.)/(5*epsilon) #(N,D)
				else:
					raise ValueError("Wrong value for the derivative order")

				grad_theta_list.append(grad_i)
			grad_h_list.append(grad_theta_list)

		grad_h = np.stack(grad_h_list, axis = -1).T #(N,K,D)
		
		if squeeze: grad_h = grad_h[0]
		return grad_h
	
	#@anycache()
	def get_WF_lal(self, theta, approx = None, df = None):
		"""
		Returns the lal WF with a given approximant with parameters theta. The WFs are in FD and are evaluated on the grid set by ``lal``
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (D, ) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
	
		approx: string
			Which approximant to use. It can be FD lal approx
			If None, the default approximant will be used
		
		df: float
			The frequency step used for the WF generation.
			If None, the default, given by the PSD will be used

		Returns
		-------
		
		hp, hc : :class:`~numpy:numpy.ndarray`
			shape: (N,K) -
			lal frequency series holding the polarizations
		"""
		if approx is None: approx = self.approx
		theta = np.squeeze(theta)
		if theta.ndim == 2: raise RuntimeError("Theta can be only one dimensional")

		if df is None: df = self.delta_f
		else: assert isinstance(df, float), "Wrong type {} for df: expected float".format(type(df))

		try:
			lal_approx = lalsim.SimInspiralGetApproximantFromString(approx) #getting the approximant
		except RuntimeError:
			raise RuntimeError("Given approximant not supported by lal")

			#generating the WF
		if not lalsim.SimInspiralImplementedFDApproximants(lal_approx):
			raise RuntimeError("Approximant {} is TD: only FD approximants are supported".format(approx)) #must be FD approximant
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi = self.var_handler.get_BBH_components(theta, self.variable_format)
		#print("mbank pars - {}: ".format(self.variable_format),m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi, e, meanano) #DEBUG
		#warnings.warn("Set non-zero spins!"); s1z = s1z + 0.4; s2z = s2z -0.7
		
		try:
			hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(m1*lalsim.lal.MSUN_SI,
                        m2*lalsim.lal.MSUN_SI,
                        float(s1x), float(s1y), float(s1z),
                        float(s2x), float(s2y), float(s2z),
                        1e6*lalsim.lal.PC_SI,
                        iota, phi, 0., #inclination, phi0, longAscNodes
                        e, meanano, # eccentricity, meanPerAno
                        df,
                        self.f_min, #flow
                        self.f_max, #fhigh
                        self.f_min, #fref
                        lal.CreateDict(),
                        lal_approx)
		except RuntimeError:
			msg = "Failed to call lal waveform with parameters: ({} {} {} {} {} {} {} {} {} {} {} {})".format(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi, e, meanano)
			raise ValueError(msg)
		#f_grid = np.linspace(0., self.f_max, len(hptilde.data.data))
		
		return hptilde, hctilde
	
	def get_WF(self, theta, approx = None, plus_cross = False):
		"""
		Computes the WF with a given approximant with parameters theta. The WFs are in FD and are evaluated on the grid on which the PSD is evauated (``self.f_grid``)
		An any lal FD approximant.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
	
		approx: string
			Which approximant to use. It can be FD lal approx
			If None, the default approximant will be used
		
		plus_cross: bool
			Whether to return both polarizations. If False, only the plus polarization will be returned

		Returns
		-------
		
		h : :class:`~numpy:numpy.ndarray`
			shape: (N,K) -
			Complex array holding the WFs evaluated on the default frequency/time grid
		"""
		theta = np.asarray(theta)
		if approx is None: approx = self.approx
		
		if theta.ndim == 1:
			theta = theta[None,:]
			squeeze = True
		else:
			squeeze = False

		WF_list = []
		for i in range(theta.shape[0]):
				#lal WF evaluated in the given f_grid
			hp, hc = self.get_WF_lal(theta[i,:], approx)
			
				#trimming the WF to the proper PSD (this amounts to enforcing the high frequency cutoff)
			hp = hp.data.data[:self.PSD.shape[0]]
			hc = hc.data.data[:self.PSD.shape[0]]

			if plus_cross: WF_list.append((hp, hc))
			else: WF_list.append(hp)
		
		h = np.stack(WF_list, axis = -1).T #(N,D)/(N,D,2)
		
		if squeeze: h = h[0,...]
		
		if plus_cross: return h[...,0], h[...,1]
		else: return h
	
	#@do_profile(follow = [])
	def get_metric(self, theta, overlap = False, metric_type = 'hessian', **kwargs):
		"""
		Returns the metric. Depending on ``metric_type``, it uses different approximations.
		It can accept any argument of the underlying function, specified by ``metric_type``
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
		
		overlap: bool
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		metric_type: str
			How to compute the metric. Available options are:
				- 'hessian': Computes the hessian (calls ``cbc_metric.get_hessian``)
				- 'numerical_hessian': computes the numerical hessian with finite difference methods (uses package ``numdifftools`` and calls ``cbc_metric.get_numerical_hessian``)
				- 'WRITEME!'

		Returns
		-------
		
		metric : :class:`~numpy:numpy.ndarray`
			shape: (N,D,D)/(D,D) -
			Array containing the metric in the given parameters
			
		"""
		#TODO: allow to implement a custom metric function...
		metric_dict ={
			'hessian': self.get_hessian,
			'projected_hessian': self.get_projected_hessian,
			'numerical_hessian': self.get_numerical_hessian,
			'parabolic_fit_hessian': self.get_parabolic_fit_hessian,
			'block_diagonal_hessian': self.get_block_diagonal_hessian,
			'fisher':self.get_fisher_matrix
			}
		
		if metric_type not in metric_dict.keys():
			msg = "Unknown metric_type '{}' given. It must be one of {}".format(metric_type, list(metric_dict.keys()))
			raise ValueError(msg)
		
		return metric_dict[metric_type](theta, overlap = overlap, **kwargs)

	def get_projected_hessian(self, theta, overlap = False,  min_eig = 1e-3, order = None, epsilon = 1e-5):
		"""
		Returns the projected Hessian matrix. The projections happens on the subspace spanned the eigenvectors with eigenvalues larger than ``min_eig``.
		
		See ``mbank.utils.get_projected_metric`` for more information.
		
		The metric obtained with this procedure is a nicer accuracy but it is **degenerate** (i.e. very close to zero eigenvalues). Moreover, the volume element obtained by the metric is not representative of the actual volume element, since it makes sense in a lower dimensional space.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
		
		overlap: bool
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		min_eig: float
			Minimum tolerable eigenvalue for ``metric``. The metric will be projected on the directions of smaller eigenvalues

		order: int
			Order of the finite difference scheme for the gradient computation.
			If None, some defaults values will be set, depending on the total mass.

		epsilon: list
			Size of the jump for the finite difference scheme for each dimension. If a float, the same jump will be used for all dimensions

		Returns
		-------
		
		metric : :class:`~numpy:numpy.ndarray`
			shape: (N,D,D)/(D,D) -
			Array containing the metric Hessian in the given parameters
			
		"""
		theta = np.asarray(theta)
		squeeze = False
		if theta.ndim == 1:
			theta = theta[None,:]
			squeeze = True
		
		metric = self.get_hessian(theta,  overlap = overlap, epsilon = epsilon, order = order)

		for i, metric_ in enumerate(metric):
			
			if self.D <= 2: break
			metric[i, ...] = get_projected_metric(metric, min_eig = min_eig)

		if squeeze: return metric[0,...]
		else: return metric

	
	def get_block_diagonal_hessian(self, theta, overlap = False, order = None, epsilon = 1e-5):
		"""
		Computes the hessian with a block diagonal method
		
		#WRITEME!!
		"""
		#FIXME: this is trash!! Remove?
		theta = np.asarray(theta)
		squeeze = False
		if theta.ndim == 1:
			theta = theta[None,:]
			squeeze = True
		
		metric = self.get_hessian(theta,  overlap = overlap, epsilon = epsilon, order = order)
		#metric = self.get_parabolic_fit_hessian(theta,  overlap = overlap, target_match = 0.99,
		#		N_epsilon_points = 5, log_epsilon_range = (-7, -4), full_output = False)

		ax_list = [k for k in range(2, self.D)]
		for i, metric_ in enumerate(metric):
			
			if self.D <= 2: break
			metric[i, 2:,2:] = project_metric(metric_, ax_list)
			for j in range(2):
				metric[i, j,2:] = 0.
				metric[i, 2:,j] = 0.

		if squeeze: return metric[0,...]
		else: return metric


	def get_parabolic_fit_hessian(self, theta, overlap = False, target_match = 0.9, N_epsilon_points = 7, log_epsilon_range = (-4, 1), full_output = False, **kwargs):
		"""
		Returns the hessian with the adjusted eigenvalues.
		The eigenvalues are adjusted by fitting a parabola on the match along each direction.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
		
		overlap: bool
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		target_match: float
			Target match for the eigenvalues fixing. Maximum range for the optimization problem
		
		N_epsilon_points: int
			Number of points to evaluate the match along each dimension
		
		log_epsilon_range: tuple
			Tuple of floats. They represent the range in the log10(epsilon) to explore
		
		full_output: bool
			Whether to return (together with the metric) a list of array, one for each dimension. The arrays have rows ``(step, match)``
		

		Returns
		-------
		
		metric : :class:`~numpy:numpy.ndarray`
			shape: (N,D,D)/(D,D) -
			Array containing the metric in the given parameters

		parabolae : list
			List of array being used to compute the eigenvalues along each dimension.
		
		original_metric : :class:`~numpy:numpy.ndarray`
			shape: (N,D,D)/(D,D) -
			The metric being used to compute the new eigenvalues. 
			
		"""
		theta = np.asarray(theta)
		squeeze = False
		if theta.ndim ==1:
			theta = theta[None,:]
			squeeze = True
		
		metric_hessian = self.get_hessian(theta, overlap = overlap, order = None) #(N,D,D)
		#metric_hessian = self.get_block_diagonal_hessian(theta, overlap = overlap, order = None); warnings.warn("Using block diagonal hessian")#(N,D,D)
		parabolae = []
		metric = []

		for center, metric_ in zip(theta, metric_hessian):
			WF1 = self.get_WF(center, self.approx)
			eigvals, eigvecs = np.linalg.eig(metric_)
			
			parabolae_ = []
			new_eigvals = []
			
				#recomputing the eigenvalues by parabolic fitting
			for d, eigvec in enumerate(eigvecs.T):
					
				epsilon_list = np.logspace(*log_epsilon_range, N_epsilon_points)
				parabola_list_d = [(0.,1.)]
				max_epsilon = [np.inf, np.inf]
				
				for epsilon, s in itertools.product(epsilon_list,[+1,-1]):
					id_s = (s+1)//2
					if max_epsilon[id_s] < epsilon: continue #this is to make sure not to explore weird regions
					
					theta_ = np.array(center)+s*epsilon*eigvec
					if self.var_handler.is_theta_ok(theta_, self.variable_format):
						WF2 = self.get_WF(theta_, self.approx)
						temp_match = self.WF_match(WF1, WF2, overlap = overlap) #Why was it set to True??
						if temp_match >= target_match:
							parabola_list_d.append((s*epsilon, temp_match))
						else:
							max_epsilon[id_s] = epsilon
					else:
						max_epsilon[id_s] = epsilon

				parabolae_.append(np.array(parabola_list_d))
				p = np.polyfit(parabolae_[-1][:,0]**2, parabolae_[-1][:,1], 1)[0] #parabolic fit
				#p = np.polyfit(parabolae_[-1][:,0]**2, parabolae_[-1][:,1], 2)[1] #quartic fit
				new_eigvals.append(np.abs(p)) #abs is dangerous, as if you have a negative eigenvalues, you wouldn't note
		
			parabolae.append(parabolae_)
			metric.append(np.linalg.multi_dot([eigvecs, np.diag(new_eigvals), eigvecs.T]))
		
		metric = np.stack(metric, axis = 0)
		if squeeze:
			metric = np.squeeze(metric)
			parabolae = parabolae[0]
			metric_hessian = metric_hessian[0]
		
		if full_output: return metric, parabolae, metric_hessian
		else: return metric
		
	
	def get_numerical_hessian(self, theta, overlap = False, epsilon = 1e-6, target_match = 0.97):
		"""
		Returns the Hessian matrix, obtained by finite difference differentiation. Within numerical erorrs, it should reproduce `cbc_metric.get_hessian_metric`.
		This function is slower and most prone to numerical errors than its counterparts, based on waveform gradients. For this reason it is mostly intended as a check of the recommended function `cbc_metric.get_hessian_metric`.
		In particular the step size epsilon must be tuned carefully and the results are really sensible on this choice.
		Uses package ``numdifftools``, not among the dependencies.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
		
		overlap: bool
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		epsilon: float
			Size of the jump for the finite difference scheme.
			If `None`, it will be authomatically computed.
		
		target_match: float
			Target match for the authomatic epsilon computation. Only applies is epsilon is None.
		
		Returns
		-------
		
		metric : :class:`~numpy:numpy.ndarray`
			shape: (N,D,D)/(D,D) -
			Array containing the metric Hessian in the given parameters
		"""
		##
		# Defining a match functions
		
		def match_t(x):
			t, theta_ = x[0], x[1:]
			WF2 = self.get_WF(theta_, self.approx)*np.exp(-1j*2*np.pi*self.f_grid*t)
			return self.WF_match(WF1, WF2, False) #WF1 is in global scope
		
		###
		# Loss function
		def loss_epsilon(log10_epsilon, axis, center):
			"Loss function to compute the proper value of epsilon"
			theta_p, theta_m = np.array(center), np.array(center)
			theta_p[axis] = center[axis]+10**log10_epsilon
			theta_m[axis] = center[axis]-10**log10_epsilon
			if not (self.var_handler.is_theta_ok(theta_p, self.variable_format)
				and self.var_handler.is_theta_ok(theta_m, self.variable_format)):
					return np.inf
			match = match_t(theta_p)
			res = np.square(target_match - match)
			return res
		
		
		##
		# Try to import numdifftools (in a future version, this may be in the package dependencies)	
		
		try:
			import numdifftools as nd
		except ImportError:
			raise ImportError("Cannot compute numerical metric as package `numdifftools` is not installed")
		
		##
		# Preprocessing theta
		
		theta = np.asarray(theta)
		squeeze = False
		if theta.ndim ==1:
			theta = theta[None,:]
			squeeze = True

		##		
		# Dealing with epsilon
		if not (isinstance(epsilon, (float, list, np.ndarray)) or epsilon is None):
			raise ValueError("epsilon should be a float, list or None")
	
		##
		# Computing the metric
		
		metric = []
		for theta_i in theta:
			WF1 = self.get_WF(theta_i, self.approx) #used by
			center = np.array([0, *theta_i])

			#setting epsilon (if it's the case)
			#FIXME: this does not work!!
			if epsilon is None:			
				epsilon_list = np.full(center.shape, 1e-5)
			
				for ax in range(self.D+1):
					res = scipy.optimize.minimize_scalar(loss_epsilon, bounds=(-10, -1), args = (ax, center),
						#method='brent', options={'xtol': 1e-2, 'maxiter': 100})
						method='bounded', options={'xatol': 1e-2, 'maxiter': 100})
					if res.success and res.fun != 1000.:
						epsilon_list[ax] = 10**res.x
				#print(theta_i, epsilon_list)
			
	
			#Computing the hessian
			step = epsilon if epsilon is not None else epsilon_list #/2 because numdifftools uses a second order accurate finite difference scheme
			H_function = nd.Hessian(match_t, base_step = 1e-2,
				num_steps=40, step_ratio=2, num_extrap=16) #adaptive method from https://git.ligo.org/chad-hanna/manifold/-/blob/main/manifold/metric.py#L273
			#H_function = nd.Hessian(match_t, step = step)
					
			H = 0.5*H_function(center)
	
			if overlap: H = H[1:,1:]
			else: H = H[1:,1:] - np.outer(H[0,1:], H[0,1:])/H[0,0]
			H = -H
			
			#enforcing positive eigenvalues (WTF??)
			eigval, eigvec = np.linalg.eig(H)
			#H = np.linalg.multi_dot([eigvec, np.diag(np.abs(eigval)), eigvec.T])
	
			metric.append(H)
		
		metric = np.stack(metric, axis = 0)
		if squeeze:	metric = np.squeeze(metric)
		return metric

	def get_fisher_matrix(self, theta, overlap = False, order = None, epsilon = 1e-3):
		"""
		Returns the Fisher matrix.
		
		::
		
			M_{ij} = 0.5 <d_i h | d_j h>
			
		```##Check this!```
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
		
		overlap: bool
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		order: int
			Order of the finite difference scheme for the gradient computation.
			If None, some defaults values will be set, depending on the total mass.

		epsilon: list
			Size of the jump for the finite difference scheme for each dimension. If a float, the same jump will be used for all dimensions

		Returns
		-------
		
		metric : :class:`~numpy:numpy.ndarray`
			shape: (N,D,D)/(D,D) -
			Array containing the metric Fisher matrix according to the given parameters
			
		"""
		scalar_ = lambda h1, h2: 0.5*np.sum(h1*h2.conj() + h2*h1.conj(), axis = -1).real
		
		theta = np.asarray(theta)
		squeeze = False
		if theta.ndim ==1:
			theta = theta[None,:]
			squeeze = True

		h = self.get_WF(theta, approx = self.approx) #(N,D)
		grad_h = self.get_WF_grads(theta, approx = self.approx, order = order, epsilon = epsilon) #(N,D, K)
			#whithening		
		h_W = h / np.sqrt(self.PSD) #whithened WF
		grad_h_W = grad_h/np.sqrt(self.PSD[:,None]) #whithened grads

			#Stacking the gradients w.r.t. t,phi
			#h*np.exp(1j*(2*np.pi*m_obj.f_grid*t+phi))
		grad_h_W_t = h_W*1j*(2*np.pi*self.f_grid) #(N,D)
		grad_h_W_phi = 1j*h_W #(N,D)
		grad_h_W = np.concatenate([grad_h_W, grad_h_W_t[...,None], grad_h_W_phi[...,None]], axis = -1) #(N,D, K+2)

			#overlaps
		h_h = np.sum(np.multiply(np.conj(h_W), h_W), axis =1).real #(N,)
		h_grad_h_real = np.einsum('ij,ijk->ik', np.conj(h_W), grad_h_W).real #(N,K+2)
		grad_h_grad_h_real = np.einsum('ijk,ijl->ikl', np.conj(grad_h_W), grad_h_W).real #(N,K+2, K+2)
		
			#assembling metric
		metric_tphi = np.einsum('ij,ik->ijk', h_grad_h_real, h_grad_h_real) 
		metric_tphi = np.einsum('ijk,i->ijk', metric_tphi , 1./np.square(h_h))
		metric_tphi = -metric_tphi + np.divide(grad_h_grad_h_real, h_h[:,None,None])
		
		metric = np.zeros((metric_tphi.shape[0], self.D, self.D))
		for i, M_ in enumerate(metric_tphi):
			metric[i,...] = project_metric(M_, [j for j in range(theta.shape[-1])])
		metric = 0.5*metric
				
		if squeeze: metric = metric[0]
		
		return metric


	def get_hessian(self, theta, overlap = False, order = None, epsilon = 1e-5):
		"""
		Returns the Hessian matrix.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
		
		overlap: bool
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		order: int
			Order of the finite difference scheme for the gradient computation.
			If None, some defaults values will be set, depending on the total mass.

		epsilon: list
			Size of the jump for the finite difference scheme for each dimension. If a float, the same jump will be used for all dimensions

		Returns
		-------
		
		metric : :class:`~numpy:numpy.ndarray`
			shape: (N,D,D)/(D,D) -
			Array containing the metric Hessian in the given parameters
			
		"""
			#TODO: understand whether the time shift is an issue here!!
			#		Usually match is max_t0 <h1(t)|h2(t-t0)>. How to cope with that? It this may make the space larger than expected
		theta = np.asarray(theta)
		squeeze = False
		if theta.ndim ==1:
			theta = theta[None,:]
			squeeze = True

		####
		#computing the metric
		####
			#M(theta) = - 0.5 * { (h|d_i h)(h|d_j h) / <h|h>^2 + [h|d_i h][h|d_j h] / <h|h>^2 - (d_i h|d_j h) / <h|h> }

		#The following outputs grad_h_grad_h_real (N,D,4,4), h_grad_h.real/h_grad_h.imag (N,D,4) and h_h (N,D), evaluated on self.f_grid (or in a std grid if PSD is None)

		### scalar product in FD
		h = self.get_WF(theta, approx = self.approx) #(N,D)
		grad_h = self.get_WF_grads(theta, approx = self.approx, order = order, epsilon = epsilon) #(N,D, K)
		
		h_W = h / np.sqrt(self.PSD) #whithened WF
		grad_h_W = grad_h/np.sqrt(self.PSD[:,None]) #whithened grads
		
		h_h = np.sum(np.multiply(np.conj(h_W), h_W), axis =1).real #(N,)
		h_grad_h = np.einsum('ij,ijk->ik', np.conj(h_W), grad_h_W) #(N,K)
		grad_h_grad_h_real = np.einsum('ijk,ijl->ikl', np.conj(grad_h_W), grad_h_W).real #(N,K,K)
		
		###
		#computing the metric, given h_grad_h.real (N,D,4), h_grad_h.imag (N,D,4) and h_h (N,D)
		metric = np.einsum('ij,ik->ijk', h_grad_h.real, h_grad_h.real) 
		metric = metric + np.einsum('ij,ik->ijk', h_grad_h.imag, h_grad_h.imag)
		metric = np.einsum('ijk,i->ijk', metric , 1./np.square(h_h))
		metric = metric - np.divide(grad_h_grad_h_real, h_h[:,None,None])

		if not overlap:
			#including time dependence
			h_h_f = np.sum(np.multiply(np.conj(h_W), h_W*self.f_grid), axis =1) #(N,)
			h_h_f2 = np.sum(np.multiply(np.conj(h_W), h_W*np.square(self.f_grid)), axis =1) #(N,)
			h_grad_h_f = np.einsum('ij,ijk->ik', np.conj(h_W)*self.f_grid, grad_h_W) #(N,4)
			
			g_tt = np.square(h_h_f.real/h_h) - h_h_f2.real/h_h #(N,)
			
			g_ti = (h_grad_h.imag.T * h_h_f.real + h_grad_h.real.T * h_h_f.imag).T #(N,K)
			g_ti = (g_ti.T/np.square(h_h)).T
			g_ti = g_ti - (h_grad_h_f.imag.T/h_h).T
			
			time_factor = np.einsum('ij,ik,i->ijk', g_ti, g_ti, 1./g_tt)
			metric = metric - time_factor
		
			#adding the -0.5 factor
		metric = -0.5*metric
		
		if squeeze:	metric = np.squeeze(metric)

		return metric
	
	def WF_symphony_match(self, h1, h2, overlap = False, F_p = 1., F_c = 0.):
		"""
		Computes the symphony match line by line between two WFs. The WFs shall be evaluated on the custom grid 
		No checks will be done on the input
		The symphony match is defined in eq (13) of `1709.09181 <https://arxiv.org/abs/1709.09181>`_
		
		To be computed, the symphony match requires the specification of antenna pattern functions. They are typically a function of sky location and a (conventional) polarization angle.
		
		Parameters
		----------
		
		h1: tuple
			(:class:`~numpy:numpy.ndarray`, :class:`~numpy:numpy.ndarray`) (N,K) -
			First WF: tuple (hp, hc)

		h1: tuple
			(:class:`~numpy:numpy.ndarray`, :class:`~numpy:numpy.ndarray`) (N,K) -
			Second WF: tuple (hp, hc)
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed
		
		F_p: float
			Value for the :math:`F_+` antenna pattern function. Used to build the signal at ifo

		F_c: float
			Value for the :math:`F_\\times` antenna pattern function. Used to build the signal at ifo			
		
		Returns
		-------
		
		sym_match : :class:`~numpy:numpy.ndarray`
			shape: (N,) -
			Array containing the symphony match of the given WFs
			
		"""
		#TODO: insert the zero padding to the frequency series also here (if it makes sense...)
		sigmasq = lambda WF: np.sum(np.multiply(np.conj(WF), WF), axis = -1)
		
			#whithening and normalizing
		s_WN = F_p*h2[0] + F_c*h2[1] #depends on the antenna pattern
		h1p_WN = (h1[0]/np.sqrt(self.PSD)) #whithened WF
		h1c_WN = (h1[1]/np.sqrt(self.PSD)) #whithened WF
		s_WN = (s_WN/np.sqrt(self.PSD)) #whithened WF
		
		h1p_WN = (h1p_WN.T/np.sqrt(sigmasq(h1p_WN))).T #normalizing WF
		h1c_WN = (h1c_WN.T/np.sqrt(sigmasq(h1c_WN))).T #normalizing WF
		s_WN = (s_WN.T/np.sqrt(sigmasq(s_WN))).T #normalizing s
	
			#computing frequency series, time series and denominator
			#TODO: pad with zeros the frequency series here!!
		SNR_fs_p = np.multiply(np.conj(s_WN), h1p_WN) #(N,D) #frequency series
		SNR_fs_c = np.multiply(np.conj(s_WN), h1c_WN) #(N,D) #frequency series
		
		SNR_ts_p = np.fft.ifft(SNR_fs_p, axis =-1).real*SNR_fs_p.shape[-1]
		SNR_ts_c = np.fft.ifft(SNR_fs_c, axis =-1).real*SNR_fs_c.shape[-1]

			#This correlation does not agree with sbank!!
		h1pc = np.sum(np.multiply(np.conj(h1p_WN), h1c_WN), axis = -1).real
		den = 1- np.square(h1pc)
		
		SNR_ts = np.square(SNR_ts_p).T + np.square(SNR_ts_c).T - 2*SNR_ts_p.T*SNR_ts_c.T*h1pc #(N,D)
		SNR_ts = SNR_ts/den #(N,D)
		SNR_ts = SNR_ts.T
		SNR_ts = np.sqrt(SNR_ts) #remember this!! It is important :)
		
		#print("mbank hphccorr: ", h1pc)
		
		if overlap: #no time maximization
			return SNR_ts[...,0]

		match = np.max(SNR_ts, axis = -1)

		return match
	
	def WF_match(self, h1, h2, overlap = False):
		"""
		Computes the match line by line between two WFs. The WFs shall be evaluated on the custom grid 
		No checks will be done on the input
		
		The
		
		Parameters
		----------
		
		h1: :class:`~numpy:numpy.ndarray`
			shape: (N,K) -
			First WF frequency series

		h2: :class:`~numpy:numpy.ndarray`
			shape: (N,K)/(K,) -
			Second WF frequency series
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed
		
		Returns
		-------
		
		match : :class:`~numpy:numpy.ndarray` (N,)
			Array containing the match of the given WFs
			
		"""
		sigmasq = lambda WF: np.sum(np.multiply(np.conj(WF), WF), axis = -1)

			##
			# The whithened WF will be zero until kmin. No need to remove it (except maybe for speed up)
			# kmin = int(self.f_min / self.delta_f)
		
			#whithening and normalizing	
		h1_WN = (h1/np.sqrt(self.PSD)) #whithened WF
		h2_WN = (h2/np.sqrt(self.PSD)) #whithened WF
		
		h1_WN = (h1_WN.T/np.sqrt(sigmasq(h1_WN))).T #normalizing WF
		h2_WN = (h2_WN.T/np.sqrt(sigmasq(h2_WN))).T #normalizing WF
	
		SNR_fs = np.multiply(np.conj(h1_WN), h2_WN) #(N,D) #frequency series
		
		if overlap: #no time maximization
			overlap = np.abs(np.sum(SNR_fs, axis =-1))
			return overlap
		
			#padding the frequency series with zeros: this mimick pycbc correlate
			#Is it really required?
		D = SNR_fs.shape[-1]
		pad_zeros = np.zeros(SNR_fs.shape)
		SNR_fs = np.concatenate([pad_zeros[...,:D//2], SNR_fs, pad_zeros[...,D//2:]], axis = -1)

		SNR_ts = np.fft.ifft(SNR_fs, axis =-1)*SNR_fs.shape[-1]
		match = np.max(np.abs(SNR_ts), axis = -1)
		
		return match
	
	
	#@do_profile(follow=[])
	def match(self, theta1, theta2, symphony = False, overlap = False):
		"""
		Computes the elementwise match between waveforms defined by theta1 and theta2
		
		If symphony is False, the match is the standard non-precessing one 
		.. math::
		
			|<h1p|h2p>|^2
			
		If symphony is True, it returns the symphony match (as in `1709.09181 <https://arxiv.org/abs/1709.09181>`_)
		.. math::

			[(s|h1p)^2+(s|h1c)^2 - 2 (s|h1p)(s|h1c)(h1c|h1p)]/[1-(h1c|h1p)^2]
		
		Parameters
		----------
		
		theta1: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the first BBHs. The dimensionality depends on self.variable_format

		theta2: :class:`~numpy:numpy.ndarray`
			shape: (N,D) /(D,) -
			Parameters of the second BBHs. The dimensionality depends on self.variable_format
	
		symphony: bool
			Whether to compute the symphony match (default False)
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed
		
		Returns
		-------
		
		match : :class:`~numpy:numpy.ndarray`
			shape: (N,) -
			Array containing the match of the given WFs
			
		"""
		#FIXME: the atleast_2d shouldn't be necessary...
		theta1 = np.asarray(theta1)
		theta2 = np.asarray(theta2)
		squeeze = ((theta1.ndim == 1) and (theta2.ndim == 1))
		theta1 = np.atleast_2d(theta1)
		theta2 = np.atleast_2d(theta2)
		
			#checking for shapes
		if theta1.shape != theta2.shape:
			if theta1.shape[-1] != theta1.shape[-1]:
				raise ValueError("Last dimension of the two imputs should be the same!")
		
		h1 = self.get_WF(theta1, self.approx, plus_cross = symphony)
		h2 = self.get_WF(theta2, self.approx, plus_cross = symphony)

		if symphony:
			match = self.WF_symphony_match(h1, h2, overlap)
		else:
			match = self.WF_match(h1, h2, overlap)

		if squeeze: match = match[0]

		return match
	
	def metric_match(self, theta1, theta2, metric = None, overlap = False):
		"""
		Computes the metric match line by line between elements in theta1 and elements in theta2.
		The match is approximated by the metric:
		
		.. math::
		
			match(theta1, theta2) = 1 - M_ij(theta1) (theta1 - theta2)_i (theta1 - theta2)_j
		
		Parameters
		----------
		
		theta1: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the first BBHs. The dimensionality depends on self.variable_format

		theta2: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the second BBHs. The dimensionality depends on self.variable_format
		
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D) -
			metric to use for the match (if None, it will be computed from scratch)
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed

		Returns
		-------
		
		match : :class:`~numpy:numpy.ndarray`
			shape: (N,) -
			Array containing the metric approximated match of the given WFs
			
		"""
		theta1 = np.asarray(theta1)
		theta2 = np.asarray(theta2)
		assert theta1.shape[-1] == theta2.shape[-1] == self.D, "Dimension of theta must be D = {}".format(self.D)
		squeeze = False
		if theta1.ndim ==1:
			theta1 = theta1[None,:]
			squeeze = True

		delta_theta = theta2 - theta1  #(N,D)
		
		if metric is None:
			metric = self.get_metric((theta1+theta2)/2., overlap)
			#metric = self.get_metric(theta1, overlap) #DEBUG
			match = 1 - np.einsum('ij, ijk, ik -> i', delta_theta, metric, delta_theta) #(N,)
		else:
			match = 1 - np.einsum('ij, jk, ik -> i', delta_theta, metric, delta_theta) #(N,)
		
		return match
		
	def get_points_at_match(self, N_points, theta, match, metric = None, overlap = False):
		"""
		Given a central theta point, it computes ``N_points`` couples of random points with constant metric match. The metric is evaluated at `theta`, hence the couple of points returned will be symmetric w.r.t. to theta and their distance from theta will be `dist/2`.
		
		The match is related to the distance between templates in the metric as:
		
		.. math::
		
			dist = sqrt(1-match)
		
		The returned points `points1`, `points2` will be such that
		
		::
		
			m_obj.metric_match(*m_obj.get_points_at_match(N_points, center, match = match))
		
		is equal to match
		
		
		Parameters
		----------
		
		N_points: int
			Number of random couples to be drawn
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (D,) -
			Parameters of the central point. The dimensionality depends on ``self.variable_format``

		match: float
			Match between the randomly drawn points and the central point ``theta``.
			The metric distance between such points and the center is ``d = sqrt(1-M)``
		
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D) -
			Metric to use for the match (if None, it will be computed from scratch)
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed

		Returns
		-------
		
		points1: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			N points close to the center. Each will have a constant metric distance from its counterpart in `points2`
		
		points2: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			N points close to the center. Each will have a constant metric distance from its counterpart in `points1`
			
		"""
		dist = np.sqrt(1-match)
		if not isinstance(metric, np.ndarray): metric = self.get_metric(theta, overlap)
		assert metric.shape == (self.D, self.D)
		
		L = np.linalg.cholesky(metric).T
		L_inv = np.linalg.inv(L)
		
		theta_prime = np.matmul(L, theta)
		
			#picking N_points random directions
		n_hat = np.random.normal(0,1, (N_points, self.D)) #(N,D)
		n_hat = (n_hat.T / np.linalg.norm(n_hat, axis=1)).T
		
		points1 = theta_prime + n_hat*dist/2.
		points2 = theta_prime - n_hat*dist/2.
		
		points1 = np.matmul(points1, L_inv.T)		
		points2 = np.matmul(points2, L_inv.T)
		
		return points1, points2
	
	
	def get_points_on_ellipse(self, N_points, theta, match, metric = None, inside = False, overlap = False):
		"""
		Given a central theta point, it computes ``N_points`` random point on the ellipse of constant match center in theta.
		The points returned will have approximately the the given metric match, although the actual metric match may differ as the metric is evaluated at `theta` and not at the baricenter of the points in question.
		If the option ``inside`` is set to ``True``, the points will be inside the ellipse, i.e. they will have a metric match larger than the given ``match``.
		The match is related to the distance between templates in the metric as:
		
		.. math::
		
			dist = sqrt(1-match)
		
		Parameters
		----------
		
		N_points: int
			Number of random points to be drawn
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (D,) -
			Parameters of the central point the metric is evaluated at (i.e. the center of the ellipse).
			The dimensionality depends on ``self.variable_format``

		match: float
			Match between the randomly drawn points and the central point ``theta``.
			The metric distance between such points and the center is ``d = sqrt(1-M)``
		
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D) -
			Metric to use for the match (if None, it will be computed from scratch)
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed

		Returns
		-------
		
		points : :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Points with distance dist from the center
		"""
		dist = np.sqrt(1-match)
		if not isinstance(metric, np.ndarray): metric = self.get_metric(theta, overlap)
		assert metric.shape == (self.D, self.D)
		
		L = np.linalg.cholesky(metric).T
		L_inv = np.linalg.inv(L)
		
		theta_prime = np.matmul(L, theta)
		
			#generating points on the unit sphere
		v = np.random.normal(0, 1, (N_points, theta.shape[0]))
		norm = 1.0 / np.linalg.norm(v, axis = 1) #(N_points,)
		
		if inside: r = np.random.uniform(0.,1., norm.shape)
		else: r = 1.
		
		points_prime = theta_prime + dist*(v.T*norm*r).T
		
		points = np.matmul(points_prime, L_inv.T)
		
		return points
	
	
	
	
		
