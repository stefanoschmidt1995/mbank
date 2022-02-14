"""
mbank.metric
============
	This module implements the metric computation for cbc signals. It provides a class ``cbc_metric`` that, besides the metric computations, offers some functions to compute waveforms and the match between them.
	
	The metric is a D dimensional square matrix that approximates the match between two waveforms. The metric M is defined such that:
	
	:math: `<h(theta) | h(theta + Deltatheta) > = 1 - M(theta)_{ij} Deltatheta_i Deltatheta_j`
	
	The metric is a useful local approximation of the match and it is the physical input for the bank generation.
	The explicit expression for the metric is a complicated expression of the gradients of the waveform and it is a function of theta.
"""

import numpy as np
import matplotlib.pyplot as plt

import lal 
import lalsimulation as lalsim

from .handlers import variable_handler

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
#FIXME: you are not able to perform the FFT of the WFs... Learn how to do it and do it well!
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
			How to handle the spin variables. Different options are possible and which option is set, will decide the dimensionality D of the parameter space (hence of the input).
			Variable format can be changed with ``set_variable_format()`` and can be accessed under name ``cbc_metric.variable_format``. See ``mbank.handlers.variable_handler`` for more details.

		PSD: tuple
			PSD for computing the scalar product.
			It is a tuple with a frequency grid array and a PSD array (both one dimensional and with the same size).
			PSD should be stored in an array which stores its value on a grid of evenly spaced positive frequencies (starting from f0 =0 Hz).
			If None, the PSD is assumed to be flat

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
		
		self.f_min = f_min
		
		if not isinstance(PSD, tuple):
			raise ValueError("Wrong format for the PSD. Expected a tuple of np.ndarray, got {}".format(type(PSD)))

		#####Tackling the PSD
		#FIXME: do you really need to store the PSD and the frequency grid all the way to zero? It seems like an useless waste of memory/computational time
		self.PSD = PSD[1]
		self.f_grid = PSD[0]
		self.delta_f = self.f_grid[1]-self.f_grid[0]
			#checking that grid is equally spaced
		assert np.all(np.diff(self.f_grid)-self.delta_f<1e-10), "Frequency grid is not evenly spaced!"
		
		#####applying a high frequency cutoff to the PSD
		if isinstance(f_max, (int, float)): #high frequency cutoff
			self.f_max = f_max
			self.f_grid = np.linspace(0., f_max, int(f_max/self.delta_f)+1)
			self.PSD = np.interp(self.f_grid, PSD[0], PSD[1])
		else:
			self.f_max = self.f_grid[-1]
		
		return
	
	def set_approximant(self, approx):
		"""
		Change the lal approximant used to compute the WF
		
		Parameters
		----------
		
		approx: string
			Which approximant to use. It can be any lal FD approximant.
		"""
		self.approx = approx
		
			#checking if the approximant is right
		try:
			lal_approx = lalsim.SimInspiralGetApproximantFromString(approx) #getting the approximant
		except RuntimeError:
			raise RuntimeError("Wrong approximant name: it must be an approximant supported by lal")

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
			An string to specify the variable format
		"""
		assert variable_format in self.var_handler.valid_formats, "Wrong variable format '{}'. Available formats are: ".format(variable_format)+str(self.var_handler.valid_formats)
		
		self.variable_format = variable_format
		
		self.D = self.var_handler.format_D[variable_format]

		return
	
	def get_metric_determinant(self, theta):
		"""
		Returns the metric determinant
		
		Parameters
		----------
		
		theta: np.ndarray
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on the variable format set for the metric

		Returns
		-------
		
		det : np.ndarray
			shape: (N,) -
			Determinant of the metric for the given input
			
		"""
		return np.linalg.det(self.get_metric(theta)) #(N,)

	def log_pdf(self, theta, boundaries = None):
		"""
		Returns the logarithm log(pdf) of the probability distribution function we want to sample from:
		.. math::
		
			pdf = p(theta) ~ sqrt(|M(theta)|)

		imposing the boundaries
		
		Parameters
		----------
		
		theta: np.ndarray
			shape: (N,D) -
			parameters of the BBHs. The dimensionality depends on the variable format set for the metric
		
		boundaries: np.ndarray
			shape: (2,D)
			An optional array with the boundaries for the model. If a point is asked below the limit, -10000000 is returned
			Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
			If None, no boundaries are implemented
		
		Returns
		-------
		
		log_pdf : np.ndarray
			shape: (N,) -
			Logarithm of the pdf, ready to use for sampling
		"""
		theta = np.array(theta)
		
		if theta.ndim == 1:
			theta = theta[None,:]
			reshape = True
		else:
			reshape = False
		ids_ok = np.full((theta.shape[0],), True)
		if isinstance(boundaries,np.ndarray):
			if boundaries.shape != (2,self.format_D[self.variable_format]):
				raise ValueError("Wrong shape of boundaries given: expected (2,{}), given {}".format(self.format_D[self.variable_format], boundaries.shape))
			
			ids_ok = np.logical_and(np.all(theta > boundaries[0,:], axis =1), np.all(theta < boundaries[1,:], axis = 1)) #(N,)
			
			#TODO: you might want to implement in the boundaries, the fact that m1>=m2!
		
		res = np.zeros((theta.shape[0],)) -10000000
		
		det = self.get_metric_determinant(theta[ids_ok,:])
		
		det = np.log(np.abs(det))*0.5 #log(sqrt(|det|))
		
		res[ids_ok] = det
		
		return res
	
	def get_WF_grads(self, theta, approx, order = None, epsilon = 1e-5):
		"""
		Computes the gradient of the WF with a given lal FD approximant. The gradients are computed with finite difference methods.
		
		Parameters
		----------
		
		theta: np.ndarray
			shape: (N,D) -
			parameters of the BBHs. The dimensionality depends on the variable format set for the metric
	
		approx: string
			Which approximant to use. It can be any lal FD approximant
		
		order: int
			Order of the finite difference scheme for the gradient computation.
			If None, some defaults values will be set, depending on the total mass.
		
		epsilon: float
			Size of the jump for the finite difference scheme

		Returns
		-------
		
		h : np.ndarray
			shape: (N, K, D) -
			Complex array holding the gradient of the WFs evaluated on the default frequency/time grid
		"""
		#Take home message, to get a really nice metric:
		# - The order of the integration is crucial. You can set this adaptively, depending on total mass

		def get_WF(theta_value, df_):
			#return self.get_WF_lal(theta_value, approx, df_)[0].data.data
			return self.get_WF(theta_value)
		
		 	#doing finite difference methods
		delta_ij = lambda i,j: 1 if i ==j else 0
		grad_h_list = []
		epsilon_list = [epsilon for _ in range(12)]
		epsilon_list[0] = epsilon_list[0]/5 #the first variable needs a smaller step?
		
		def get_order(M):
			#TODO:fix the thresholds
			if isinstance(order, int): return order
			if M>50.: order_ = 1
			if M<=50. and M>15.: order_ = 2
			if M<=15. and M>5.: order_ = 4
			if M<=5. and M>1.: order_ = 6
			if M<=1.: order_ = 8
			return order_
		
		for theta_ in theta:
			df = self.delta_f#*10.
			order_theta = get_order(theta_[0])
			grad_theta_list = []
			if order==1: WF = get_WF(theta_, df) #only for forward euler
			
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
					#second order method: slower but more accurate
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
		
		return grad_h
	
	def get_WF_lal(self, theta, approx = None, df = None):
		"""
		Returns the lal WF with a given approximant with parameters theta. The WFs are in FD and are evaluated on the grid set by ``lal``
		
		Parameters
		----------
		
		theta: np.ndarray
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
		
		hp, hc : np.ndarray
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
		#print("mbank pars: ",m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi) #DEBUG
		
			#FIXME: WTF is this grid??
		hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(m1*lalsim.lal.MSUN_SI,
                        m2*lalsim.lal.MSUN_SI,
                        s1x, s1y, s1z,
                        s2x, s2y, s2z,
                        1e6*lalsim.lal.PC_SI,
                        iota, phi, 0., #inclination, phi0, longAscNodes
                        e, meanano, # eccentricity, meanPerAno
                        df,
                        self.f_min, #flow
                        self.f_max, #fhigh
                        self.f_min, #fref
                        lal.CreateDict(),
                        lal_approx)
		#f_grid = np.linspace(0., self.f_max, len(hptilde.data.data))
		
		return hptilde, hctilde
	
	def get_WF(self, theta, approx = None, plus_cross = False):
		"""
		Computes the WF with a given approximant with parameters theta. The WFs are in FD and are evaluated on the grid on which the PSD is evauated (``self.f_grid``)
		An any lal FD approximant.
		
		Parameters
		----------
		
		theta: np.ndarray
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
	
		approx: string
			Which approximant to use. It can be FD lal approx
			If None, the default approximant will be used
		
		plus_cross: bool
			Whether to return both polarizations. If False, only the plus polarization will be returned

		Returns
		-------
		
		h : np.ndarray
			shape: (N,K) -
			Complex array holding the WFs evaluated on the default frequency/time grid
		"""
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
			
				#trimming the WF to the proper PSD
			hp = hp.data.data[:self.PSD.shape[0]]
			hc = hc.data.data[:self.PSD.shape[0]]

			if plus_cross: WF_list.append((hp, hc))
			else: WF_list.append(hp)
		
		h = np.stack(WF_list, axis = -1).T #(N,D)/(N,D,2)
		
		if squeeze: h = h[0,...]
		
		if plus_cross: return h[...,0], h[...,1]
		else: return h
	
	
	def get_metric(self, theta, overlap = False):
		"""
		Returns the metric
		
		Parameters
		----------
		
		theta: np.ndarray
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on self.variable_format
		
		overlap: bool
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		Returns
		-------
		
		metric : np.ndarray
			shape: (N,D,D) -
			Array containing the metric in the given parameters
			
		"""
			#TODO: add an option for choosing the gradient accuracy
			#TODO: understand whether the time shift is an issue here!!
			#		Usually match is max_t0 <h1(t)|h2(t-t0)>. How to cope with that? It this may make the space larger than expected
		theta = np.array(theta)
		squeeze = False
		if theta.ndim ==1:
			theta = theta[None,:]
			squeeze = True
		
		####
		#computing the metric
		####
			#M(theta) = 0.5 * { (h|d_i h)(h|d_j h) / <h|h>^2 + [h|d_i h][h|d_j h] / <h|h>^2 - (d_i h|d_j h) / <h|h> }

		#The following outputs grad_h_grad_h_real (N,D,4,4), h_grad_h.real/h_grad_h.imag (N,D,4) and h_h (N,D), evaluated on self.f_grid (or in a std grid if PSD is None)

		### scalar product in FD
		h = self.get_WF(theta, approx = self.approx) #(N,D)
		grad_h = self.get_WF_grads(theta, approx = self.approx) #(N,D, K)
			
		h_W = h / np.sqrt(self.PSD) #whithened WF
		grad_h_W = grad_h/np.sqrt(self.PSD[:,None]) #whithened grads
		
		h_h = np.sum(np.multiply(np.conj(h_W), h_W), axis =1).real #(N,)
		h_grad_h = np.einsum('ij,ijk->ik', np.conj(h_W), grad_h_W) #(N,4)
		grad_h_grad_h_real = np.einsum('ijk,ijl->ikl', np.conj(grad_h_W), grad_h_W).real #(N,D,D)
		
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
			
			g_ti = (h_grad_h.imag.T * h_h_f.real + h_grad_h.real.T * h_h_f.imag).T #(N,4)
			g_ti = (g_ti.T/np.square(h_h)).T
			g_ti = g_ti - (h_grad_h_f.imag.T/h_h).T
			
			time_factor = np.einsum('ij,ik,i->ijk', g_ti, g_ti, 1./g_tt)
			metric = metric - time_factor
		
			#adding the 0.5 factor
		metric = 0.5*metric
		
		if squeeze:
			metric = np.squeeze(metric)

		return metric
	
	def WF_symphony_match(self, h1, h2, overlap = False):
		"""
		Computes the symphony match line by line between two WFs. The WFs shall be evaluated on the custom grid 
		No checks will be done on the input
		The symphony match is defined in eq (13) of `1709.09181 <https://arxiv.org/abs/1709.09181>`_
		
		Parameters
		----------
		
		h1: tuple
			(np.ndarray,np.ndarray) (N,K) -
			First WF: tuple (hp, hc)

		h1: tuple
			(np.ndarray,np.ndarray) (N,K) -
			Second WF: tuple (hp, hc)
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed
		
		Returns
		-------
		
		sym_match : np.ndarray
			shape: (N,) -
			Array containing the symphony match of the given WFs
			
		"""
		sigmasq = lambda WF: np.sum(np.multiply(np.conj(WF), WF), axis = -1)
		
			#whithening and normalizing
		s_WN = h2[0] + 0*h2[1] #TODO: set a proper linear combination coeff!!
		h1p_WN = (h1[0]/np.sqrt(self.PSD)) #whithened WF
		h1c_WN = (h1[1]/np.sqrt(self.PSD)) #whithened WF
		s_WN = (s_WN/np.sqrt(self.PSD)) #whithened WF
		
		h1p_WN = (h1p_WN.T/np.sqrt(sigmasq(h1p_WN))).T #normalizing WF
		h1c_WN = (h1c_WN.T/np.sqrt(sigmasq(h1c_WN))).T #normalizing WF
		s_WN = (s_WN.T/np.sqrt(sigmasq(s_WN))).T #normalizing s
	
			#computing frequency series, time series and denominator
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
		
		h1: np.ndarray
			shape: (N,K) -
			First WF frequency series

		h2: np.ndarray
			shape: (N,K)/(K,) -
			Second WF frequency series
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed
		
		Returns
		-------
		
		match : np.ndarray (N,)
			Array containing the match of the given WFs
			
		"""
		sigmasq = lambda WF: np.sum(np.multiply(np.conj(WF), WF), axis = -1)
		
			#whithening and normalizing			
		h1_WN = (h1/np.sqrt(self.PSD)) #whithened WF
		h2_WN = (h2/np.sqrt(self.PSD)) #whithened WF
			
		h1_WN = (h1_WN.T/np.sqrt(sigmasq(h1_WN))).T #normalizing WF
		h2_WN = (h2_WN.T/np.sqrt(sigmasq(h2_WN))).T #normalizing WF
	
		SNR_fs = np.multiply(np.conj(h1_WN), h2_WN) #(N,D) #frequency series
		
		if overlap: #no time maximization
			overlap = np.abs(np.sum(SNR_fs, axis =-1))
			return overlap
		
		SNR_ts = np.fft.ifft(SNR_fs, axis =-1)*SNR_fs.shape[-1]

		match = np.max(np.abs(SNR_ts), axis = -1)

		return match
	
	
	#@do_profile(follow=[])
	def match(self, theta1, theta2, symphony = False, overlap = False):
		"""
		Computes the match line by line between elements in theta1 and elements in theta2
		
		If symphony is False, the match is the standard non-precessing one 
		.. math::
		
			|<h1p|h2p>|^2
			
		If symphony is True, it returns the symphony match (as in `1709.09181 <https://arxiv.org/abs/1709.09181>`_)
		.. math::

			[(s|h1p)^2+(s|h1c)^2 - 2 (s|h1p)(s|h1c)(h1c|h1p)]/[1-(h1c|h1p)^2]
		
		Parameters
		----------
		
		theta1: np.ndarray
			shape: (N,D) -
			Parameters of the first BBHs. The dimensionality depends on self.variable_format

		theta2: np.ndarray
			shape: (N,D) /(D,) -
			Parameters of the second BBHs. The dimensionality depends on self.variable_format
	
		symphony: bool
			Whether to compute the symphony match (default False)
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed
		
		Returns
		-------
		
		match : np.ndarray
			shape: (N,) -
			Array containing the match of the given WFs
			
		"""
		#FIXME: this function agrees with pycbc but disagree with sbank!! WFT?? Time alignment is the way!
		theta1 = np.array(theta1)
		theta2 = np.array(theta2)
		
			#checking for shapes
		if theta1.shape != theta2.shape:
			if theta1.shape[-1] != theta1.shape[-1]:
				raise ValueError("Shapes of the two inputs should be the same!")
		
		squeeze = False
		if theta1.ndim ==1:
			theta1 = theta1[None,:]
			theta2 = theta2[None,:]
			squeeze = True
		
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
		
			match(theta1, theta2) = 1 + M_ij(theta1) (theta1 - theta2)_i (theta1 - theta2)_j
		
		Parameters
		----------
		
		theta1: np.ndarray
			shape: (N,D) -
			Parameters of the first BBHs. The dimensionality depends on self.variable_format

		theta2: np.ndarray
			shape: (N,D) -
			Parameters of the second BBHs. The dimensionality depends on self.variable_format
		
		metric: np.ndarray
			shape: (D,D) -
			metric to use for the match (if None, it will be computed from scratch)
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed

		Returns
		-------
		
		match : np.ndarray
			shape: (N,) -
			Array containing the metric approximated match of the given WFs
			
		"""
		theta1 = np.array(theta1)
		theta2 = np.array(theta2)
		assert theta1.shape == theta2.shape, "Shapes of the two inputs should be the same!"
		squeeze = False
		if theta1.ndim ==1:
			theta1 = theta1[None,:]
			theta2 = theta2[None,:]
			squeeze = True
		
		theta1 = theta1[:,:self.var_handler.D(self.variable_format)]
		theta2 = theta2[:,:self.var_handler.D(self.variable_format)]

		delta_theta = theta2 - theta1  #(N,D)
		
		if metric is None:
			metric = self.get_metric((theta1+theta2)/2., overlap)
			#metric = self.get_metric(theta1, overlap) #DEBUG
			match = 1 + np.einsum('ij, ijk, ik -> i', delta_theta, metric, delta_theta) #(N,)
		else:
			match = 1 + np.einsum('ij, jk, ik -> i', delta_theta, metric, delta_theta) #(N,)
		
		return match
		
	def metric_accuracy(self, theta1, theta2, overlap = False):
		"""
		Computes the metric accuracy at the given points of the parameter space.
		The accuracy is the absolute value of the difference between true and metric mismatch (as computed by match and metric_match)
		
		Parameters
		----------
		
		theta1: np.ndarray
			shape: (N,D) -
			Parameters of the first BBHs. The dimensionality depends on self.variable_format

		theta2: np.ndarray
			shape: (N,D) -
			Parameters of the second BBHs. The dimensionality depends on self.variable_format

		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed

		Returns
		-------
		
		accuracy : np.ndarray
			shape: (N,) -
			Array the accuracy of the metric approximation
		
		"""
		accuracy = self.metric_match(theta1, theta2, overlap = overlap) - self.match(theta1, theta2, overlap = overlap) #(N,)
		return np.abs(accuracy)
		
	
	def get_points_atmatch(self, N_points, theta, match, overlap = False):
		"""
		Given a central theta point, it computes ``N_points`` random point with a given metric match.
		The match is related to the distance between templates in the metric as:
		
		.. math::
		
			dist = sqrt(1-match)
		
		Parameters
		----------
		
		N_points: int
			Number of random points to be drawn
		
		theta: np.ndarray
			shape: (D,) -
			Parameters of the central point. The dimensionality depends on ``self.variable_format``

		match: float
			Match between the randomly drawn points and the central point ``theta``.
			The metric distance between such points and the center is ``d = sqrt(1-M)``
		
		overlap: bool
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed

		Returns
		-------
		
		points : np.ndarray
			shape: (N,D) -
			Points with distance dist from the center
		"""
		dist = np.sqrt(1-match)
		metric = -self.get_metric(theta, overlap)
		
		L = np.linalg.cholesky(metric).T
		L_inv = np.linalg.inv(L)
		
		theta_prime = np.matmul(L, theta)
		
			#generating points on the unit sphere
		v = np.random.normal(0, 1, (N_points, theta.shape[0]))
		norm = 1.0 / np.linalg.norm(v, axis = 1) #(N_points,)
		
		points_prime = theta_prime + dist*(v.T*norm).T
		
		points = np.matmul(points_prime, L_inv.T)
		
		if False:
			plt.scatter(*theta_prime.T)
			plt.scatter(*points_prime.T)
			plt.figure()
			plt.scatter(*theta.T)
			plt.scatter(*points.T)
			plt.show()
		
		return points