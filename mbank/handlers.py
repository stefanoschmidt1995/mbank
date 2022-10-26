"""
mbank.handlers
==============
	Two two important handlers class for ``mbank``:
	
	- ``variable_handler``: takes care of the BBH parametrization
	- ``tiling_handler``: takes care of the tiling of the space
	
	The handlers are used extensively throughout the package
"""
####################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
import itertools
import collections

	#ligo.lw imports for xml files: pip install python-ligo-lw
from ligo.lw import utils as lw_utils
from ligo.lw import ligolw
from ligo.lw import table as lw_table
from ligo.lw import lsctables
from ligo.lw.utils import process as ligolw_process

import lal 
import lalsimulation as lalsim

from tqdm import tqdm

import ray

import scipy.stats
import scipy.integrate
import scipy.spatial

#TODO: You should use a GMM with infinite dirichlet prior to interpolate the metric! It's gonna be much much faster (and effective) 
#https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
from .flow.flowmodel import STD_GW_Flow
import torch

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

####################################################################################################################

###
class variable_handler(object):
	"""
	Class to handle a large number of variable layouts.
	The full BBH space is characterized by the following variables
	
	::
	
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi
	
	
	These are standard quantities in gravitational waves and the ligo waveform infrastructure can accept all these paramters.

	A *variable layout* is a set of variables that parametrize a subspace of such 12 dimensional BBH parameter space.
	The conversion between a the full space and the chosen subspace is made by means of a projection, where any variable not used in the paramterization is set to a default value of 0.
	For instance, ``(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi)`` can be mapped to ``(m1, m2, s1x, 0, s1z, 0, 0, s2z, iota, 0, e, 0)`` with a suitable variable format.
	The variable parametrizing the projected BBH will alway be labeled as ``theta`` and has a variable dimension ``D``, depending on the chosen variable format.
	
	The variable layout is specified by a string ``variable_format``, that can be passed to any function of this class.
	
	The general variable format can be built as any of the follwing:
	
	::
	
		MassFormat_SpinFormat
		MassFormat_SpinFormat_AnglesFormat	
		MassFormat_SpinFormat_EccentricityFormat
		MassFormat_SpinFormat_EccentricityFormat_AnglesFormat

	Valid format for the two masses are:
	
	- ``Mq``: Total mass and q
	- ``logMq``: Log10 of total mass and q
	- ``mceta``: chirp mass and eta
	- ``m1m2``: mass1 and mass2
	
	Valid formats for spins are:
	
	- ``nonspinning``: no spins are considered (only two masses), D = 0
	- ``chi``: only ``chi_eff`` is considered. That means that ``s1z=s2z=chi``
	- ``s1z``: only the z spin component of most massive BH is considered, D = 3
	- ``s1z_s2z``: only the z components of the spins are considered (no precession), D = 2
	- ``s1xz``: spin components assigned to one BH in plane xz, D = 2
	- ``s1xyz``: spin components assigned to one BH, D = 3
	- ``s1xz_s2z``: the two z components are assigned as well as s1x, D = 3
	- ``s1xyz_s2z``: the two z components are assigned as well as s1x, s1y,  D = 4
	- ``fullspins``: all the 6 dimensional spin parameter is assigned,  D = 6
	
	Regarding the spins, we stick to the following conventions:

	- If a spin is aligned to the z axis the spin variable is the value of the z component of the spin (with sign): ``s1z`` or ``s2z``.
	- If a generic spin is assigned to a BH, the spin is *always** expressed in sperical coordinates ``s1``, ``theta1``, ``phi``. ``s1`` (or ``s2``) represents the spin magnitude (between 0 and 1). The angle ``theta1`` (``theta2``) corresponds to the polar angle of the spin, which controls the magnitude of in-plane spin. The angle ``phi1``(``phi2``) is the aximuthal angle (if set), which controls the mixing between x and y components.

	On top of the spins, the user can **optionally** specify a format for Angles and for the Eccentricity
	
	Valid formats for the angles are:
	
	- ``iota``: the inclination angle is included
	- ``iotaphi``: the inclination angle and reference phase are included

	Valid formats for the eccentricity are:
	
	- ``e``: to include the orbital eccentricity
	- ``emeanano``: to include the orbital eccentricity and mean periastron anomaly

	Note that some waveform approximants do not support all these paramters. Using a format that sets them to be non zero will likely fail. 
	
	For example, valid formats are: ``mceta_s1xz_s2z_e_iotaphi``, ``m1m2_nonspinning_e``, ``Mq_s1xz_s2z_iotaphi``, ``m1m2_s1z_s2z``
	"""
	#TODO: hard code here the values of epsilons?

	def __init__(self):
		"Initialization. Creates a dict of dict with all the info for each format" 
		
			#hard coding valid formats for masses, spins, eccentricity and angles
		self.m_formats = ['m1m2', 'Mq', 'logMq', 'mceta'] #mass layouts
		self.s_formats = ['nonspinning', 'chi', 's1z', 's1z_s2z', 's1xz', 's1xyz', 's1xz_s2z', 's1xyz_s2z', 'fullspins'] #spin layouts
		self.e_formats = ['', 'e', 'emeanano'] #eccentric layouts
		self.angle_formats = ['', 'iota', 'iotaphi'] #angles layouts
		
			#hard coding dimensions for each format
		D_spins = {'nonspinning':0, 'chi':1, 's1z':1, 's1z_s2z':2, 's1xz':2, 's1xyz':3, 's1xz_s2z':3, 's1xyz_s2z':4, 'fullspins': 6} #dimension of each spin format
		D_ecc = {'':0, 'e':1, 'emeanano':2} #dimension of each eccentric format
		D_angles = {'':0, 'iota':1, 'iotaphi':2} #dimension of each angle format

			#creating info dictionaries
		self.format_info = {}
		self.valid_formats = []
		self.format_D = {}
			
		for m_, s_, e_, a_ in itertools.product(self.m_formats, self.s_formats, self.e_formats, self.angle_formats):
				#nonspinning and noneccentric formats don't sample angles...
			#if s_ == 'nonspinning' and (a_ != '' or e_ == ''): continue 
			format_to_add = ''
			for f_ in [m_, s_, e_, a_]:
				if f_ != '': format_to_add += '_{}'.format(f_)
			format_to_add = format_to_add[1:] #removing '_' in the first position
			
				#adding the format to the different dicts
			self.valid_formats.append(format_to_add)
			self.format_D[format_to_add] = 2+ D_spins[s_] + D_ecc[e_] + D_angles[a_] #dimension of the variable format
			self.format_info[format_to_add] = {'D':self.format_D[format_to_add],
				'mass_format': m_, 'spin_format': s_, 'eccentricity_format': e_, 'angle_format':a_,
				'e': (e_.find('e')>-1), 'meanano': (e_.find('meanano')>-1),
				'iota': (a_.find('iota')>-1) ,'phi': (a_.find('phi')>-1)}
			
		self.MAX_SPIN = 0.999 #defining the constant maximum value for the spin (used for any check that's being done)
		self.MAX_Q = 100.
		
		self.constraints = {'M':(0.,np.inf), 'logM':(-np.inf,np.inf), 'q': (1./self.MAX_Q, self.MAX_Q),
				'Mc':(0., np.inf), 'eta':(1./self.MAX_Q, 0.25),
				'mass1':(0, np.inf), 'mass2':(0, np.inf),
				'chi': (-self.MAX_SPIN, self.MAX_SPIN), 
				's1z': (-self.MAX_SPIN, self.MAX_SPIN), 's2z': (-self.MAX_SPIN, self.MAX_SPIN),
				's1': (0., self.MAX_SPIN), 's2': (0., self.MAX_SPIN),
				'theta1':(0., np.pi), 'phi1':(-np.pi, np.pi),
				'theta2':(0., np.pi), 'phi1':(-np.pi, np.pi),
				'iota': (0.,np.pi), 'phi': (-np.inf, np.inf),
				'e': (0., 1.), 'meanano': (0, 1.)
				} #allowed ranges for each label
				
		
		return

	def is_theta_ok(self, theta, variable_format, raise_error = False):
		"""
		Given a value of theta, it checks whether it is an acceptable value for the given spin format.
		It calls `get_BBH_components` internally.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on variable_format
		
		variable_format: string
			How to handle the BBH variables.
		
		raise_error: bool
			Whether to raise a ValueError if theta is not acceptable
		
		Returns
		-------
			is_ok: bool
				`True` if `theta` is an acceptable value. `False` otherwise
		"""
		labels = self.labels(variable_format, latex = False)
		theta = np.atleast_2d(theta)
		
		is_ok = True
		if raise_error: bad_labels = []
		
		for i, l in enumerate(labels):
			is_ok_l = np.logical_and( self.constraints[l][0]<theta[...,i], theta[...,i]<self.constraints[l][1])
			is_ok = np.logical_and(is_ok, is_ok_l)
			if not np.all(is_ok_l) and raise_error: bad_labels.append(l)
		
		if raise_error and not np.all(is_ok):
			raise ValueError("The given theta does not have an acceptable value for the quantities: {}".format(*bad_labels))
		
		return is_ok


	def switch_BBH(self, theta, variable_format):
		"""
		Given theta, it returns the theta components of the system with switched BBH masses (so that m1>m2)
		If only BH1 has an in-plane component, only the z components of the spins will be switched: this is equivalent to assume that the in-plane spin of BH1 is a collective spin.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on variable_format
		
		variable_format: string
			How to handle the BBH variables.
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			ids = np.where(theta[:,0]<theta[:,1])[0]
			theta[ids,0], theta[ids,1] = theta[ids,1], theta[ids,0] #switching masses
		elif self.format_info[variable_format]['mass_format'] in ['Mq', 'logMq']:
			ids = np.where(theta[:,1]<1)[0]
			theta[ids,1] = 1./theta[ids,1] #switching masses
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			return theta #this mass configuration is symmetric, no further action is required
		
		if len(ids)<1: return theta
		
		if self.format_info[variable_format]['spin_format'] in ['nonspinning', 'chi','s1z']:
			pass #In chi and nonspinning there's nothing to switch #s1z is always on the larger spin
		elif self.format_info[variable_format]['spin_format'] == 's1z_s2z':
			theta[ids,2], theta[ids,3] = theta[ids,3], theta[ids,2] #switching spins
		elif self.format_info[variable_format]['spin_format'] == 's1xz':
			pass #chiP is always intended to be on the largest BH (pay attention to this)
		elif self.format_info[variable_format]['spin_format'] == 's1xyz':
			pass #chiP is always intended to be on the largest BH (pay attention to this)
		elif self.format_info[variable_format]['spin_format'] == 's1xz_s2z':
			theta[ids,3], theta[ids,4] = theta[ids,4], theta[ids,3] #switching spins
		elif self.format_info[variable_format]['spin_format'] == 's1xyz_s2z':
			theta[ids,4], theta[ids,5] = theta[ids,5], theta[ids,4] #switching spins
		elif self.format_info[variable_format]['spin_format'] == 'fullspins':
			theta[ids,[2,3,4]], theta[ids,[5,6,7]] =  theta[ids,[5,6,7]], theta[ids,[2,3,4]] #switching spins


		if squeeze: theta = np.squeeze(theta)
		return theta
	
	def labels(self, variable_format, latex = False):
		"""
		List the names of the variables for each entry of the BBH parameter vector
		
		Parameters
		----------
		
		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
		
		labels: list
			List of labels for the parmams in the BBH (each a str)
		
		latex: bool
			Whether the labels should be in latex
		"""
		assert variable_format in self.valid_formats, "Wrong variable format given"
		
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			if latex: labels = [r'$m_1$', r'$m_2$']
			else: labels = ['mass1', 'mass2']
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
			if latex: labels = [r'$M$', r'$q$']
			else: labels = ['M', 'q']
		elif self.format_info[variable_format]['mass_format'] == 'logMq':
			if latex: labels = [r'$\log_{10}M$', r'$q$']
			else: labels = ['logM', 'q']
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			if latex: labels = [r'$\mathcal{M}_c$', r'$\eta$']
			else: labels = ['Mc', 'eta']
		
		if self.format_info[variable_format]['spin_format'] =='nonspinning':
			pass
		elif self.format_info[variable_format]['spin_format'] == 'chi':
			if latex: labels.extend([r'$\chi$'])
			else: labels.extend(['chi'])
		elif self.format_info[variable_format]['spin_format'] == 's1z':
			if latex: labels.extend([r'$s_{1z}$'])
			else: labels.extend(['s1z'])
		elif self.format_info[variable_format]['spin_format'] == 's1z_s2z':
			if latex: labels.extend([r'$s_{1z}$', r'$s_{2z}$'])
			else: labels.extend(['s1z', 's2z'])
		elif self.format_info[variable_format]['spin_format'] == 's1xz':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$'])
			else: labels.extend(['s1', 'theta1'])
		elif self.format_info[variable_format]['spin_format'] == 's1xyz':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$\phi_1$'])
			else: labels.extend(['s1', 'theta1', 'phi1'])
		elif self.format_info[variable_format]['spin_format'] == 's1xz_s2z':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$s_{2z}$'])
			else: labels.extend(['s1', 'theta1', 's2z'])
		elif self.format_info[variable_format]['spin_format'] == 's1xyz_s2z':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$\phi_1$', r'$s_{2z}$'])
			else: labels.extend(['s1','theta1', 'phi1', 's2z'])
		elif self.format_info[variable_format]['spin_format'] == 'fullspins':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$\phi_1$', r'$s_{2}$', r'$\theta_2$', r'$\phi_2$'])
			else: labels.extend(['s1','theta1', 'phi1', 's2z', 'theta2', 'phi2'])
		
		if self.format_info[variable_format]['e'] and latex: labels.append(r'$e$')
		if self.format_info[variable_format]['e'] and not latex: labels.append('e')

		if self.format_info[variable_format]['meanano'] and latex: labels.append(r'$meanano$')
		if self.format_info[variable_format]['meanano'] and not latex: labels.append('meanano')
		
		if self.format_info[variable_format]['iota'] and latex: labels.append(r'$\iota$')
		if self.format_info[variable_format]['iota'] and not latex: labels.append('iota')

		if self.format_info[variable_format]['phi'] and latex: labels.append(r'$\phi$')
		if self.format_info[variable_format]['phi'] and not latex: labels.append('phi')
		
		return labels
	
	def D(self, variable_format):
		"""
		Returns the dimensionality of the parameter space required
		
		Parameters
		----------
		
		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
		
		D: int
			Dimensionality of the BBH parameter vector			
		"""
		assert variable_format in self.valid_formats, "Wrong variable format given"
		return self.format_info[variable_format]['D']

	def format_info(self, variable_format):
		"""
		Returns the a dict with some information about the format.
		The dict has the following entries:
		
		- ``mass_format`` : format for the masses
		- ``spin_format`` : format for the spins
		- ``eccentricity_format`` : format for the eccentricities
		- ``angle_format`` : format for the angles
		- ``D`` : dimensionality of the BBH space
		- ``e`` : whether the variables include the eccentricity e
		- ``meanano`` : whether the variables include the mean periastron anomaly meanano
		- ``iota`` : whether the variables include the inclination iota
		- ``phi`` : whether the variables include the reference phase phi
		
		Parameters
		----------
		
		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
		
		format_info: int
			Dictionary with the info for the format
		"""
		assert variable_format in self.valid_formats, "Wrong variable format given"
		return self.format_info[variable_format]

	def get_theta(self, BBH_components, variable_format):
		"""
		Given the ``BBH components``, it returns the components suitable for the bank.
		
		Parameters
		----------
		
		BBH_components: :class:`~numpy:numpy.ndarray`
			shape: (N,12)/(12,) -
			Parameters of the BBHs.
			Each row should be: m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
			theta: :class:`~numpy:numpy.ndarray`
				shape: (N,D)/(D,) -
				Components of the BBH in the format suitable for the bank.
				The dimensionality depends on variable_format
		"""
		BBH_components, squeeze = self._check_theta_and_format(BBH_components, variable_format)
		
		assert BBH_components.shape[1] == 12, "The number of BBH parameter is not enough. Expected 12 [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi], given {}".format(BBH_components.shape[1])
		
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			theta = [BBH_components[:,0], BBH_components[:,1]]
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
			q = np.maximum(BBH_components[:,1] / BBH_components[:,0], BBH_components[:,0] / BBH_components[:,1])
			theta = [BBH_components[:,0] + BBH_components[:,1], q]
		elif self.format_info[variable_format]['mass_format'] == 'logMq':
			q = np.maximum(BBH_components[:,1] / BBH_components[:,0], BBH_components[:,0] / BBH_components[:,1])
			theta = [np.log10(BBH_components[:,0] + BBH_components[:,1]), q]
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			eta = np.divide(BBH_components[:,1] * BBH_components[:,0], np.square(BBH_components[:,0] + BBH_components[:,1]) )
			theta = [(BBH_components[:,0] + BBH_components[:,1])*np.power(eta, 3./5.), eta]

			#starting a case swich
		if self.format_info[variable_format]['spin_format'] =='nonspinning':
			pass
		elif self.format_info[variable_format]['spin_format'] == 'chi':
			chi = (BBH_components[:,0]*BBH_components[:,4] + BBH_components[:,7]*BBH_components[:,1])/ \
					(BBH_components[:,0]+BBH_components[:,1])
			theta.append(chi)
		elif self.format_info[variable_format]['spin_format'] == 's1z':
			theta.append(BBH_components[:,4])
		elif self.format_info[variable_format]['spin_format'] == 's1z_s2z':
			theta.append(BBH_components[:,4])
			theta.append(BBH_components[:,7])
		elif self.format_info[variable_format]['spin_format'] == 's1xz':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1) #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			theta.extend([s1, theta1])
		elif self.format_info[variable_format]['spin_format'] == 's1xyz':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1) #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			phi1 = np.arctan2(BBH_components[:,3], BBH_components[:,2])
			theta.extend([s1, theta1, phi1])
		elif self.format_info[variable_format]['spin_format'] == 's1xz_s2z':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1)+1e-10 #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			theta.append(s1)
			theta.append(theta1)
			theta.append(BBH_components[:,7])
		elif self.format_info[variable_format]['spin_format'] == 's1xyz_s2z':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1)+1e-10 #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			phi1 = np.arctan2(BBH_components[:,3], BBH_components[:,2])
			theta.extend([s1, theta1, phi1, BBH_components[:,7]])
		elif self.format_info[variable_format]['spin_format'] == 'fullspins':
			s1 = np.maximum(np.linalg.norm(BBH_components[:,2:5], axis =1), 1e-20) #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			phi1 = np.arctan2(BBH_components[:,3], BBH_components[:,2])
			s2 = np.maximum(np.linalg.norm(BBH_components[:, 5:8], axis =1), 1e-20) #(N,)
			theta2 = np.arccos(BBH_components[:,7]/s2)
			phi2 = np.arctan2(BBH_components[:,6], BBH_components[:,5])
			theta.extend([s1, theta1, phi1, s2, theta2, phi2])
		else:
			raise RuntimeError("Wrong setting for variable_format")
			
			#dealing with eccentricity
		if self.format_info[variable_format]['e']:
			theta.append(BBH_components[:,8])
		if self.format_info[variable_format]['meanano']:
			theta.append(BBH_components[:,9])
		
			#dealing with angles
		if self.format_info[variable_format]['iota']:
			theta.append(BBH_components[:,10])
		if self.format_info[variable_format]['phi']:
			theta.append(BBH_components[:,11])
		
		theta = np.column_stack(theta)
		
		if squeeze: theta = np.squeeze(theta)
		
		return theta


	def get_BBH_components(self, theta, variable_format):
		"""
		Given ``theta``, it returns the components suitable for lal
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray` (N,D)
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
		
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano iota, phi: :class:`~numpy:numpy.ndarray`
			Components of the BBH in the std parametrization.
			Each has shape (N,)
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		assert theta.shape[1]==self.D(variable_format), "The number of BBH parameter doesn't fit into the given variable format. Expected {}, given {}".format(self.D(variable_format), theta.shape[1])
		
			#setting the masses
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			m1, m2 = theta[:,0], theta[:,1]
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
			m1, m2 = theta[:,0]*theta[:,1]/(1+theta[:,1]), theta[:,0]/(1+theta[:,1])
			m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2) #this is to make sure that m1>m2, also if q is less than 1
		elif self.format_info[variable_format]['mass_format'] == 'logMq':
			M = 10**theta[:,0]
			m1, m2 = M*theta[:,1]/(1+theta[:,1]), M/(1+theta[:,1])
			m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2) #this is to make sure that m1>m2, also if q is less than 1
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
				#see https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/tau0tau3.py#L215
			M = theta[:,0] / np.power(theta[:,1], 3./5.)
			if not np.all(theta[:,1]<=0.25):
				raise ValueError("Values of the symmetric mass ratio should be all <= 0.25. The given array has some entries not satisfying this: {}".format(theta[:,1]))
			temp_ = np.power(1.0 - 4.0 * theta[:,1], 0.5)
			m1 = 0.5 * M * (1.0 + temp_)
			m2 = 0.5 * M * (1.0 - temp_)
		
			#allocating variables for spins
		s1x, s1y, s1z = np.zeros(m1.shape), np.zeros(m1.shape), np.zeros(m1.shape)
		s2x, s2y, s2z = np.zeros(m1.shape), np.zeros(m1.shape), np.zeros(m1.shape)

			#dealing with spins
		if self.format_info[variable_format]['spin_format'] =='nonspinning':
			pass
		elif self.format_info[variable_format]['spin_format'] == 'chi':
			s1z, s2z = theta[:,2], theta[:,2]
		elif self.format_info[variable_format]['spin_format'] == 's1z':
			s1z = theta[:,2]
		elif self.format_info[variable_format]['spin_format'] == 's1z_s2z':
			s1z, s2z = theta[:,2], theta[:,3]
		elif self.format_info[variable_format]['spin_format'] == 's1xz':
			s1x, s1z = theta[:,2]*np.sin(theta[:,3]), theta[:,2]*np.cos(theta[:,3])
		elif self.format_info[variable_format]['spin_format'] == 's1xyz':
			s1x, s1y, s1z = theta[:,2]*np.sin(theta[:,3])*np.cos(theta[:,4]), theta[:,2]*np.sin(theta[:,3])*np.sin(theta[:,4]), theta[:,2]*np.cos(theta[:,3])
		elif self.format_info[variable_format]['spin_format'] == 's1xz_s2z':
			s1x, s1z, s2z = theta[:,2]*np.sin(theta[:,3]), theta[:,2]*np.cos(theta[:,3]), theta[:,4]
		elif self.format_info[variable_format]['spin_format'] == 's1xyz_s2z':
			s1x, s1y, s1z, s2z = theta[:,2]*np.sin(theta[:,3])*np.cos(theta[:,4]), theta[:,2]*np.sin(theta[:,3])*np.sin(theta[:,4]), theta[:,2]*np.cos(theta[:,3]), theta[:,5]
		elif self.format_info[variable_format]['spin_format'] == 'fullspins':
			s1x, s1y, s1z = theta[:,2]*np.sin(theta[:,3])*np.cos(theta[:,4]), theta[:,2]*np.sin(theta[:,3])*np.sin(theta[:,4]), theta[:,2]*np.cos(theta[:,3])
			s2x, s2y, s2z = theta[:,5]*np.sin(theta[:,6])*np.cos(theta[:,7]), theta[:,5]*np.sin(theta[:,6])*np.sin(theta[:,7]), theta[:,5]*np.cos(theta[:,6])

			#dealing with angles and eccentricity (tricky!!)
		assign_var =  [self.format_info[variable_format]['e'], self.format_info[variable_format]['meanano'],
				self.format_info[variable_format]['iota'], self.format_info[variable_format]['phi']]
		N_to_assign = sum(assign_var)
		
		vars_to_assign = []
		
		k = 0
		for av_ in assign_var:
			if av_:
				vars_to_assign.append(theta[:,-N_to_assign+k] )
				k += 1
			else:
				vars_to_assign.append(np.zeros(m1.shape))
		
		e, meanano, iota, phi = vars_to_assign
	
			#setting spins to zero, if they need to be
		def set_zero_spin(s):
			ids_ = np.where(np.abs(s)<1e-10)[0]
			if len(ids_)>0: s[ids_] = 0.
			return s
		s1x, s1y, s1z = set_zero_spin(s1x), set_zero_spin(s1y), set_zero_spin(s1z)
		s2x, s2y, s2z = set_zero_spin(s2x), set_zero_spin(s2y), set_zero_spin(s2z)
		
		
		if squeeze:
			m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,  e, meanano, iota, phi = m1[0], m2[0], s1x[0], s1y[0], s1z[0], s2x[0], s2y[0], s2z[0], e[0], meanano[0], iota[0], phi[0]
		
		return m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi
	
	def get_mchirp(self, theta, variable_format):
		"""
		Given theta, it returns the chirp mass

		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
			mchirp: :class:`~numpy:numpy.ndarray`
				Chirp mass of each BBH
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			mchirp = np.power(theta[:,0]*theta[:,1], 3./5.) / np.power(theta[:,0]+theta[:,1], 1./5.)
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
			mchirp = theta[:,0] * np.power(theta[:,1]/np.square(theta[:,1]+1), 3./5.)
		elif self.format_info[variable_format]['mass_format'] == 'logMq':
			mchirp = 10**theta[:,0] * np.power(theta[:,1]/np.square(theta[:,1]+1), 3./5.)
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			mchirp = theta[:,0]

		if squeeze: mchirp = mchirp[0]
		return mchirp

	def get_massratio(self, theta, variable_format):
		"""
		Given theta, it returns the mass ratio.

		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
			q: :class:`~numpy:numpy.ndarray`
				Chirp mass of each BBH
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		if self.format_info[variable_format]['mass_format'] =='m1m2':
			q = np.maximum(theta[:,1]/theta[:,0], theta[:,0]/theta[:,1])
		elif self.format_info[variable_format]['mass_format'] in ['Mq', 'logMq']:
			q = theta[:,1]
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			q = theta[:,1] #FIXME: compute this properly! 

		if squeeze: q = q[0]
		return q
	
	def get_chiP(self, m1, m2, s1x, s1y, s1z, s2x, s2y):
		"""
		Computes the precessing spin parameter (one dimensional) as in `1408.1810 <https://arxiv.org/abs/1408.1810>`_
		Also described in `2012.02209 <https://arxiv.org/abs/2012.02209>`_
		
		Parameters
		----------
		
		m1, m2: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Masses of the two BHs
			It assumes m1>=m2

		s1x, s1y: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			In-plane spins of the primary black hole
		
		s1z: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Aligned spin for the primary black hole. Used to enforce Kerr limit in the spin parameter
		
		s2x, s2y: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			In-plane spins of the secondary black hole
		
		Returns
		-------
		
		chiP: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			The precessing spin parameter
		"""
		m1, m2, s1x, s1y, s1z, s2x, s2y = np.asarray(m1), np.asarray(m2), \
			np.asarray(s1x), np.asarray(s1y), np.asarray(s1z), \
			np.asarray(s2x), np.asarray(s2y)
		
		q = m1/m2
		assert np.all(q>=1), 'm1 should be greater or equal than m2' #assert q to be greater than 1
		A1 = 2+1.5/q 
		A2 = 2+1.5*q
		s1_perp = np.sqrt(s1x**2+s1y**2) * m1**2 #(N,)/()
		s2_perp = np.sqrt(s2x**2+s2y**2) * m2**2 #(N,)/()
		sp = np.maximum(s1_perp*A1, s2_perp*A2) #(N,)/()
		
		chip = sp/(A1*m1**2) #(N,)/()
		
			#enforcing kerr limit
		norm = np.sqrt(chip**2+s1z**2) #(N,)/()
		ids_ = np.where(norm >=self.MAX_SPIN)[0]
		if len(ids_)>0: chip[ids_] = np.sqrt(self.MAX_SPIN - s1z[ids_]**2 )

		return chip

	def get_chiP_2D(self, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, only_spin1 = False):
		"""
		Computes the two dimensional precessing spin parameter as in `2012.02209 <https://arxiv.org/abs/2012.02209>`_
		The approximantion assigns a two dimensional in-plane spin to one of the two BHs.
		If the option only_spin1 is set, the in-plane spin will be always assigned to the primary BH (as in (7) in `2012.02209 <https://arxiv.org/abs/2012.02209>`_).
		
		Parameters
		----------
		
		m1, m2: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Masses of the two BHs

		s1x, s1y: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			In-plane spins of the primary black hole
		
		s1z: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Aligned spin for the primary black hole. Used to enforce Kerr limit in the spin parameter (if it is the case)
		
		s2x, s2y: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			In-plane spins of the secondary black hole

		s2z: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Aligned spin for the secondary black hole. Used to enforce Kerr limit in the spin parameter (if it is the case)
		
		only_spin1: bool
			Whether to assign the precessing spin only always to the primary BH.
			The default is False, as 2012.02209 suggests.
		
		Returns
		-------
		
		chiP_2D_1: :class:`~numpy:numpy.ndarray`
			shape: (N,2)/(2,) -
			In-plane (x and y) components of the two dimensional precessing spin parameter on the primary BH
		
		chiP_2D_2: :class:`~numpy:numpy.ndarray`
			shape: (N,2)/(2,) -
			In-plane (x and y) components of the two dimensional precessing spin parameter on the secondary BH
		
		"""
		#TODO: check the accuracy of this function
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = np.asarray(m1), np.asarray(m2), \
			np.asarray(s1x), np.asarray(s1y), np.asarray(s1z), \
			np.asarray(s2x), np.asarray(s2y),  np.asarray(s2z)
		
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = np.atleast_1d(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z)

		assert np.all(m1>=m2), 'm1 should be greater or equal than m2' #assert q to be greater than 1

		S1_perp = (m1**2*np.column_stack([s1x, s1y]).T).T #(N,2)
		S2_perp = (m2**2*np.column_stack([s2x, s2y]).T).T #(N,2)
		
		S_perp = S1_perp + S2_perp
		
		S1_perp_norm= np.linalg.norm(S1_perp, axis = 1) #(N,)
		S2_perp_norm= np.linalg.norm(S2_perp, axis = 1) #(N,)
		
		if only_spin1: BH1_gtr_BH2 = np.ones(S1_perp_norm.shape, dtype = bool) #only the primary BH will have the in-plane spin
		else: BH1_gtr_BH2 = S1_perp_norm >= S2_perp_norm
	
			#computing effective spin parameters
		chi_eff_1, chi_eff_2 = np.zeros(S1_perp.shape), np.zeros(S2_perp.shape) #(N,2)
		
		if np.any(BH1_gtr_BH2): chi_eff_1[BH1_gtr_BH2,:] = S_perp[BH1_gtr_BH2] / (m1[BH1_gtr_BH2]**2+ S2_perp_norm[BH1_gtr_BH2])
		if np.any(~BH1_gtr_BH2): chi_eff_2[~BH1_gtr_BH2,:] = S_perp[~BH1_gtr_BH2] / (m2[~BH1_gtr_BH2]**2+ S1_perp_norm[~BH1_gtr_BH2])
		
			#enforcing Kerr limit
		norm_1 = np.linalg.norm( np.column_stack([*chi_eff_1.T, s1z]), axis =1) #(N,)/()
		norm_2 = np.linalg.norm( np.column_stack([*chi_eff_2.T, s2z]), axis =1) #(N,)/()
		
		ids_1 = np.where(norm_1 >self.MAX_SPIN)[0]
		ids_2 = np.where(norm_2 >self.MAX_SPIN)[0]
		
			#self.MAX_SPIN is a upper bound for the spin
		if len(ids_1)>0: chi_eff_1[ids_1] = (chi_eff_1[ids_1].T * np.sqrt(self.MAX_SPIN - s1z[ids_1]**2 ) / np.linalg.norm(chi_eff_1[ids_1], axis =1)).T
		if len(ids_2)>0: chi_eff_2[ids_2] = (chi_eff_2[ids_2].T * np.sqrt(self.MAX_SPIN - s2z[ids_2]**2 ) / np.linalg.norm(chi_eff_2[ids_2], axis =1)).T
		
		return chi_eff_1, chi_eff_2
	
	def get_chiP_BBH_components(self, BBH_components, chiP_type = 'chiP'):
		"""
		Given a set of BBH components (in the output format of ``get_BBH_components``) it returns the components of the same BBH in the precessing spin parameter approximation.
		This implements a mapping between the 4 dimensional in-plane spin parameter (s1x, s1y, s2x, s2y) onto a smaller space. Several options are available:
		
		- ``chiP``: performs the mapping described in eq (4.1) of arxiv/1408.1810, where the only non-zero componennt is s1x
		- ``chiP_2D``: performs the mapping described in eq (10-11) of arxiv/2012.02209. It consist in assigning a two dimensional in-plane spin to the BH which exibit more precession
		- ``chiP_2D_BH1``: performs the mapping described in eq (7) of arxiv/2012.02209. The spin parameter is the same as above but it is always assigned to the primary BH
		
		Parameters
		----------
		
		BBH_components: :class:`~numpy:numpy.ndarray`
			shape: (N,12) -
			Parameters of the BBHs.
			Each row should keep: ``m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi``
		
		chiP_type: str
			A string to determine which precessing spin approximation to set
			
			- ``chiP``: one dimensional spin parameter eq (5-6) of 2012.02209
			- ``chiP_2D``: two dimensional spin parameter eq (10-11) of 2012.02209
			- ``chiP_2D_BH1``: two dimensional spin parameter, always placed on BH1, as in eq (7) of 2012.02209
		
		Returns
		-------
		
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano iota, phi: :class:`~numpy:numpy.ndarray`
			Components of the BBH in the std parametrization after the spin parameter mapping has been applied.
			Each has shape (N,)
		"""
		#TODO: is this name right? get_BBH_components accepts theta, this accepts BBH_components. It may be confusing...
		
		assert chiP_type in ['chiP', 'chiP_2D', 'chiP_2D_BH1'], "Wrong format for the chiP_type: it should be one among ['chiP', 'chiP_2D', 'chiP_2D_BH1'], not {}".format(chiP_type)
		
		BBH_components, squeeze = self._check_theta_and_format(BBH_components, None)
		assert BBH_components.shape[1] == 12, "The number of BBH parameter is not enough. Expected 12 [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi], given {}".format(BBH_components.shape[1])
		chiP_BBH_components = np.array(BBH_components) #copying into a fresh new array
		
		if chiP_type == 'chiP':
			chiP = self.get_chiP(*chiP_BBH_components[:,:7].T)
			chiP_BBH_components[:,2] = chiP
			chiP_BBH_components[:,[3,5,6]] = 0.
		else:
			chiP_1, chiP_2 = self.get_chiP_2D(*chiP_BBH_components[:,:8].T, only_spin1 = (chiP_type == 'chiP_2D_BH1') )
			chiP_BBH_components[:,2:4] = chiP_1
			chiP_BBH_components[:,5:7] = chiP_2
		
		if squeeze: chiP_BBH_components = chiP_BBH_components[0,:]
		return tuple(comp_ for comp_ in chiP_BBH_components.T)

	def _check_theta_and_format(self, theta, variable_format):
		"""
		Performs some standard checks and preprocessing to the theta vector.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
			theta: :class:`~numpy:numpy.ndarray`
				Theta with two dimensions (N,D)
			
			squeeze: bool
				Whether the output array shall be squeezed
		
		"""
		if isinstance(variable_format, str):
			assert variable_format in self.valid_formats, "Wrong variable format given"
		
		theta = np.asarray(theta)
		if theta.ndim == 1:
			theta = theta[None, :]
			squeeze = True
		else:
			squeeze = False
		
		return theta, squeeze
	
####################################################################################################################

#TODO: tiling_handler should have a variable format! It should be a string attribute and should be stored in the tiling numpy array as a metadata. See https://numpy.org/devdocs/reference/generated/numpy.dtype.metadata.html
#It must be given at initialization (if it's not read from file).
#The idea is that, whenever a tiling file is read, there is no need to specify explicitly the variable_format

class tile(tuple):
	"""
	Class to represent a tile. A tile is a tuple
	
	::
	
		(Rectangle, Metric)
	
	where rectangle is represented by a :class:`~scipy.spatial.Rectangle` object and metric is a square matrix, stored as a :class:`~numpy:numpy.ndarray`.
	
	The rectangle and the metric can be accessed with `tile.rectangle` and `tile.metric`.
	The volume and the center of the tile can be accessed with `tile.center` and `tile.volume` respectively.
	"""
	def __new__(cls, rectangle, metric = None):
		"""
		Create a new tile.
		
		Parameters
		----------
			rectangle: :class:`~scipy.spatial.Rectangle`/:class:`~numpy:numpy.ndarray`/tuple
				A scipy rectangle.
				If an `np.ndarray` is given, it must be of shape `(2,D)` and it is interpreted s.t. `rectangle[0,:]` is the minimum and `rectangle[1,:]` is the maximum.
				If metric is None, it can be initialized with a tuple: in this case it is understood that the tuple is (Rect, metric) and it will be unwrapped authomatically
				
			metric: :class:`~numpy:numpy.ndarray`
				shape (D,D)
		"""
		#FIXME: check here can be done a bit better...
		if metric is None:  rectangle, metric = rectangle
		if isinstance(rectangle, np.ndarray) or isinstance(rectangle, list):
			rectangle = np.asarray(rectangle)
			assert rectangle.ndim ==2 and rectangle.shape[0] == 2, "The given rectangle must be of shape (2,D) but {} was given".format(rectangle.shape)
			rectangle = scipy.spatial.Rectangle(rectangle[0,:], rectangle[1,:])
		assert isinstance(rectangle, scipy.spatial.Rectangle), "The given rectangle must be either a scipy.spatial.Rectangl or a numpy array but a {} was given".format(type(rectangle))
		
		metric = np.asarray(metric)
		assert isinstance(metric, np.ndarray), "The metric must be a numpy array"
		metric_shape = (rectangle.maxes.shape[0], rectangle.maxes.shape[0])
		assert metric.shape == metric_shape, "The metric must have shape {} but shape {} was given".format(metric_shape, metric.shape)

		return super().__new__(cls, [rectangle, metric])

	@property
	def rectangle(self):
		return self[0]
	
	@property
	def metric(self):
		return self[1]

	@property
	def det(self):
		return np.abs(np.linalg.det(self[1]))
	
	@property
	def center(self):
		return (self.rectangle.maxes+self.rectangle.mins)/2.
	
	@property
	def volume(self):
		return np.sqrt(np.abs(np.linalg.det(self.metric)))*self.rectangle.volume()

	@property
	def D(self):
		return self.metric.shape[0]
	
	def split(self, axis = None, n = 2):
		"""
		Splits the tile along the given axis. It returns n split tiles (with the same metric)
		If axis is None, the axis is set to be along the direction of longest proper dimension.
		
		Parameters
		----------
			axis: int
				Dimension along which the tile shall be split. If `None`, the dimension will be set to be the largest
		 Returns
		 -------
		 	*split: tile
		 		The splitted
		"""
		if n<=1: return self
		
		if axis is None:
			d_vector = self.rectangle.maxes - self.rectangle.mins #(D,)
			dist = np.square(d_vector)*np.diag(self.metric)
			axis = np.argmax(dist)

		new_tiles = []
		tile_to_split = self
		len_d = self.rectangle.maxes[axis] - self.rectangle.mins[axis]

		for i in range(1, n):
			left_rect, right_rect = tile_to_split.rectangle.split(axis,  self.rectangle.mins[axis] + i/float(n)*len_d)
			left, right = tile(left_rect, self.metric), tile(right_rect, self.metric)
			new_tiles.append(left)
			tile_to_split = right
		new_tiles.append(right)

		return tuple(new_tiles)

	def N_templates(self, avg_dist):
		"""
		Computes the approximate number of templates inside the tile, given the typical distance between templates

		`N_templates` is compute as:
		
		::
			
			N_templates = rect.volume() * sqrt(abs(det(metric))) / avg_dist**D

		`N_templates` is a measure of volume if `avg_dist` is kept fixed

		Parameters
		----------
			avg_dist: float
				Desidered average distance between templates

		Returns
		-------
			N_templates: float
				The number or templates that should lie inside the tile (i.e. the rectangle with the given metric)
			
		"""
		return self.volume / np.power(avg_dist, self.D)


class tiling_handler(list, collections.abc.MutableSequence):
	"""
	Class for a tiling with I/O helpers.
	A tiling is a list of tiles that cover a larger space.
	Each tile, consists in:
	
	- an hypercubes (:class:`~scipy.spatial.Rectangle` object) that defines its boundaries
	- a metric that it's used to compute distances between points of the tile. It is represented by a DxD matrix (``np.ndarray`` object), where D is the dimensionality of the space.
	
	Overall a tiling handler looks like:
	
	::
	
		[(rect_1, metric_1), (rect_2, metric_2), ..., (rect_N, metric_N)]
	"""
	
	def __init__(self, ini = None):
		"""
		Initializes the tiling handler.
		
		Parameters
		----------
			ini: str/list
				Optional initialization.
				If a string is given, it is understood as the file the tiling handler is loaded from. The file shall be the same format as produced by ``tiling_handler.save()``
				If a list is given, it is understood as a list of tiles and the list will be initialized accordingly. Providing a single tile/tuple is also possible.
		"""
		list.__init__(self)
		self.lookup_table = None
		self.boundaries = None
		self.flow = None
		
			#storing the volume and the volume of each tile, for fast retrival
		self.volume = None
		self.tiles_volume = []

		if ini is None: return
		if isinstance(ini, tuple): ini = [ini]
		if isinstance(ini, str): self.load(ini)
		elif isinstance(ini, list):
			for t in ini:
				assert len(t)==2, "The elements in the input list have the wrong length. They should be broadcastable to a tile object, representing a tuple (Rectangle, Metric)"
				self.append(tile(*t))
		else:
			msg = "Type of ini argument not understood (expected list, tuple or str). The tiling_handler is initialized empty."
			warnings.warn(msg)
		self._set_boundaries()
		return
	
	def __getitem__(self, key):
		item = list.__getitem__(self, key)
		if isinstance(item, tuple): return item
		else: return tiling_handler(item)
		
	def get_centers(self):
		"""
		Returns an array with the centers of the tiling
		
		Returns
		-------
			centers: :class:`~numpy:numpy.ndarray`
				shape (N,D) -
				All the centers of the tiles
			
		"""
		centers = np.stack( [t.center for t in self.__iter__()], axis =0)
		return np.atleast_2d(centers)

	def _set_boundaries(self):
		"""
		Sets the boundaries of the tiling, stored as a Rectangle
		"""
		boundaries = []
		for R, _ in self.__iter__():
			boundaries.append([R.mins, R.maxes])
		if len(boundaries)>0:
			boundaries = np.array(boundaries)
			self.boundaries = scipy.spatial.Rectangle(np.min(boundaries[:,0,:], axis =0), np.max(boundaries[:,1,:], axis =0))
		else:
			self.boundaries = None
		return

	def update_KDTree(self):
		"""
		Updates the lookup table to compute the tile each point falls in.
		It should be called if you update the tiling and you want to compute the tile each point falls in.
		"""
		self.lookup_table = scipy.spatial.KDTree(self.get_centers())
		return

	#@do_profile(follow=[])
	def get_tile(self, points, kdtree = True):
		"""
		Given a set points, it computes the tile each point is closest to.
		
		Parameters
		----------
			points: :class:`~numpy:numpy.ndarray`
				shape (N,D)/(D,) - 
				A set of points
			
			kdtree: bool
				Whether to use a kdtree method to compute the tile. This method is much faster but it may be less accurate as it relies on euclidean distance rather than on the rectangles of the tiling.
		
		Returns
		-------
			id_tile: list
				A list of length N of the indices of the closest tile for each point.
		"""
		#TODO: Understand whether the look-up table is fine. It is FASTER, but maybe this is unwanted...
		points = np.atleast_2d(np.asarray(points))


		if kdtree:
			if self.lookup_table is None: self.update_KDTree()
			_, id_tiles = self.lookup_table.query(points, k=1)

		else:
			#FIXME: this is super super slow!!
			distance_points = []
			for R, _ in self.__iter__():
				distance_points.append( R.min_distance_point(points) ) #(N_points,)
			distance_points = np.stack(distance_points, axis = 1) #(N_points, N_tiles)
			id_tiles = np.argmin(distance_points, axis = 1) #(N_points,)
			del distance_points
		
		return id_tiles

	def is_inside(self, points):
		"""
		Returns whether each point is inside the tiling.
		
		Parameters
		----------
			points: :class:`~numpy:numpy.ndarray`
				shape (N,D)/(D,) - 
				A set of points
		
		Returns
		-------
			is_inside: :class:`~numpy:numpy.ndarray`
				shape (N,)/() - 
				Bool array stating whether each of the input is inside the tiling
			
			ids_inside :class:`~numpy:numpy.ndarray`
				shape (N',) - 
				Indices of the points inside the tiling

			ids_outside :class:`~numpy:numpy.ndarray`
				shape (N',) - 
				Indices of the points outside the tiling
		"""
		points = np.asarray(points)
		if self.boundaries is None: self._set_boundaries()
		if self.boundaries is None: raise ValueError("Tiling is empty: unable to check whether points are inside it")
		
		is_inside = self.boundaries.min_distance_point(points) == 0
		ids_inside = np.where(is_inside)[0]
		ids_outside = np.where(~is_inside)[0]
		
		return is_inside, ids_inside, ids_outside

	def create_tiling_from_list(self, boundaries_list, tolerance, metric_func, max_depth = None, use_ray = False, verbose = True):
		"""
		Creates a tiling of the space, starting from a list of rectangles.
		Each rectangle in `boundaries_list` is treated separately and if `use_ray is set to `True` they run in parallel.
		
		This function is useful to create a new tiling which covers disconnected regions or to heavily parallelize the computation.
		
		The generated tiling will **overwrite** any previous tiling (i.e. the tiling will be emptied before executing).

		Parameters
		----------

		boundaries_list: list
			A list of boundaries for a coarse tiling. Each box will have its own independent hierarchical tiling
			Each element of the list must be (max, min), where max, min are array with the upper and lower point of the hypercube.
			Each element can also be a (2,D) `np.ndarray`.
			If a single `np.ndarray` is given

		tolerance: float
			Maximum tolerated relative change between the metric determinant of the child and the parent :math:`|M|`.
			This means that a tile will be split if
			
			.. math ::
			
				0.5 \\bigg\\rvert{\log_{10}\\frac{|M_{\\text{parent}}|}{|M_{\\text{child}}|}} \\bigg\\rvert > tolerance
			
			If tolerance is greater than 10, no further tiling will be performed
			
		metric_func: function
			A function that accepts theta and returns the metric.
			A common usage would be:
			
			::
			
				metric_obj = mbank.metric.cbc_metric(**args)
				metric_func = metric_obj.get_metric
		
		max_depth: int
			Maximum number of splitting before quitting the iteration. If None, the iteration will go on until the volume condition is not met
				
		use_ray: bool
			Whether to use ray to parallelize

		verbose: bool
			whether to print to screen the output
		
		Returns
		-------
					
		self: tiling_handler 
			A list of tiles ready to be used for the bank generation
		"""
			#emptying the tiling & initializing ray
		self.clear()
		if use_ray:
			ray.init()
		
			#checking on boundaries_list
		if not isinstance(boundaries_list, list):
			if isinstance(boundaries_list, np.ndarray):
				if boundaries_list.ndim ==2 and boundaries_list.shape[0]==2: boundaries_list = [boundaries_list]
				else: raise ValueError("If `boundaries_list` is an array, must have shape (2,D)")
			else:
				raise ValueError("Wrong value for the entry `boundaries_list`")

		t_ray_list = []
		
		for i, b in enumerate(boundaries_list):
			temp_t_obj = tiling_handler() #This must be emptied at every iteration!! Otherwise, it gives lots of troubles :(
			if use_ray:
				t_ray_list.append( temp_t_obj.create_tiling_ray.remote(temp_t_obj, b,
							tolerance, metric_func, max_depth = max_depth, verbose = verbose , worker_id = i) )
			else:
				self.extend(temp_t_obj.create_tiling(b, tolerance, metric_func, max_depth = max_depth, verbose = verbose, worker_id = None)) #adding the newly computed templates to the tiling object
			
		if use_ray:
			t_ray_list = ray.get(t_ray_list)
			ray.shutdown()
			if verbose: print("All ray jobs are done")
			t_obj = tiling_handler()
			for t in t_ray_list: self.extend(t)
		
		self.compute_volume()
		self.update_KDTree()
		self._set_boundaries()
		
		return self


	@ray.remote
	def create_tiling_ray(self, boundaries, tolerance, metric_func, max_depth = None, verbose = True, worker_id = None):
		"Wrapper to `create_tiling` to allow for `ray` parallel execution. See `handlers.tiling_hander.create_tiling()` for more information."
		return self.create_tiling(boundaries, tolerance, metric_func, max_depth, verbose, worker_id)
	
	def create_tiling(self, boundaries, tolerance, metric_func, max_depth = None, verbose = True, worker_id = None):
		"""
		Create a tiling within a rectangle using a hierarchical iterative splitting method.
		If there is already a tiling, the splitting will be continued.
		
		Parameters
		----------
		
		boundaries: :class:`~numpy:numpy.ndarray`
			shape: (2,D) -
			Boundaries of the space to tile.
			Lower limit is ``boundaries[0,:]`` while upper limits is ``boundaries[1,:]``
			If the tiling is non empty and not consistent with the boundaries, a warning will be raised but the boundaries will be ignored

		tolerance: float
			Maximum tolerated relative change between the metric determinant of the child and the parent ``|M|``.
			This means that a tile will be split if
			
			::
			
				|log10(sqrt(M_p/M_c))| > tolerance
			
			If tolerance is greater than 10, no further tiling will be performed
			
		metric_func: function
			A function that accepts theta and returns the metric.
			A common usage would be:
			
			::
			
				metric_obj = mbank.metric.cbc_metric(**args)
				metric_func = metric_obj.get_metric
		
		max_depth: int
			Maximum number of splitting before quitting the iteration. If None, the iteration will go on until the volume condition is not met
			
		verbose: bool
			Whether to print the progress of the computation
		
		worker_id: int
			If given, it will print the worker id in the progress bar.
		
		Returns
		-------
			self: tiling_handler
				Return this object filled with the desired tiling
		
		"""
		boundaries = tuple([np.array(b) for b in boundaries])
		D = boundaries[0].shape[0]
		
		##
		# In this context, we build a list where each element is (tile, is_ok, depth)
		# is_ok checks whether the tile is ready to be returned to the user or shall be splitted more
		##
		
		####
		#Defining the initial list of tiles. It will be updated accordingly to the splitting algorithm
		####
			
			#initializing with the center of the boundaries and doing the first metric evaluation
		start_rect = scipy.spatial.Rectangle(boundaries[0], boundaries[1])
		if len(self) ==0:
			start_metric = metric_func((boundaries[1]+boundaries[0])/2.)

			#tiles_list = [(start_rect, start_metric, N_temp+1)] 
			tiles_list = [(tile(start_rect, start_metric), tolerance >= 10, 0)]

			#else the tiling is already full! We do:
			#	- a consistency check of the boundaries
			#	- initialization of tiles_list
		else:
			centers = self.get_centers()
			if np.any(start_rect.min_distance_point(centers)!=0):
				warnings.warn("The given boundaries are not consistent with the previous tiling. This may not be what you wanted")
				print(centers, start_rect.min_distance_point(centers), start_rect)
			tiles_list = [(tile(R, M), tolerance >= 10, 0) for R, M in self.__iter__()]
		
		tiles_list_old = []
		
		
		if start_rect.volume()<1e-19:
			warnings.warn("The given boundaries are degenerate (i.e. zero volume) and a single tile was generated: this may not be what you expected")
			self.clear() #empty whatever was in the old tiling
			self.extend( [tile(start_rect, start_metric)])
			return self
		
		####
		#Defining some convenience function
		####
		def get_deltaM(metric1, metric2):
			det1, det2 = np.linalg.det(metric1), np.linalg.det(metric2)
			#return np.abs(det1-det2)/np.maximum(det1,det2)
			return 0.5*np.abs(np.log10(det1/det2))
		
			 #progress bar = % of volume covered
		if verbose:
			if worker_id is None: desc = 'Volume covered by the tiling'
			else:  desc = 'Worker {} - Volume covered by tiling'.format(worker_id)
			pbar = tqdm(total=100, desc = desc,
				bar_format = "{desc}: {percentage:.1f}%|{bar}| [{elapsed}<{remaining}]")
			V_tot = start_rect.volume()
		
		####
		#Iterative splitting
		####
		#TODO: consider using log ratio, instead of the absolute distance...
		while not (tiles_list == tiles_list_old):
				#updating old one
			tiles_list_old = list(tiles_list)
			V_covered = 0.
				#loops on tiles an updating tiles_list
			for t in tiles_list:

				if t[1]:
					if verbose: V_covered += t[0].rectangle.volume()
					continue
				if isinstance(max_depth, (int, float)):
					if t[2]>= max_depth: continue #stop splitting
				
				nt = t[0].split(None, 3) #new tiles (len = 3)

					#Computing the new metric				
				metric_0 = metric_func(nt[0].center)
				metric_2 = metric_func(nt[2].center)
					#Computing stopping conditions
				split_tile = get_deltaM(metric_2, metric_0) < tolerance

				extended_list = [ 	(tile(nt[0].rectangle, metric_0), split_tile, t[2]+1),
									(nt[1],	split_tile, t[2]+1),
									(tile(nt[2].rectangle, metric_2), split_tile, t[2]+1) ]
				
					#replacing the old tile with the new ones
				tiles_list.remove(t)
				tiles_list.extend(extended_list)

			if verbose:
				pbar.n = V_covered/V_tot*100.
				pbar.refresh()

		if verbose: pbar.close()
		
		self.clear() #empty whatever was in the old tiling
		self.extend( [t for t,_,_ in tiles_list] )
		
		self.update_KDTree()
		self.compute_volume()
		self._set_boundaries()
		
		return self
	
	def compute_volume(self, metric_func = None):
		"""
		Compute the volume of the space an the volume of each tile.
		The volume will be computed with the metric approximation by default.
		If `metric_function` is a function, the volume will be computed by means of an integration.
		It must returns a metric approximation given a point theta in D dimension.
		
		The volume is computed as the integral of :math:`sqrt{|M(theta)|}` over the D dimensional space of the tile.
		
		Using a metric function for the integration, may be veeeery slow!
		
		Parameters
		----------
			metric_func: function
				A function that accepts theta and returns the metric.
				A common usage would be:
				
				::
				
					metric_obj = mbank.metric.cbc_metric(**args)
					metric_func = metric_obj.get_metric
			
		Returns
		-------
			volume: float
				The volume of the space covered by the tiling
			
			tiles_volume: list
				The volume covered by each tile
		"""
		if metric_func is None:
			#tiles_volume = [rect.volume()*np.sqrt(np.abs(np.linalg.det(metric))) for rect, metric in self.__iter__()]
			if self.volume is None or len(self.tiles_volume) != len(self):
				self.tiles_volume = [t.volume for t in self.__iter__()]
				self.volume = sum(self.tiles_volume)
			volume, tiles_volume = self.volume, self.tiles_volume
		else:
			tiles_volume = []

				#casting metric function to the weird format required by scipy
			def metric_func_(*args):
				theta = np.column_stack(args)
				return metric_func(theta)

			for rect, _ in tqdm(self.__iter__(), desc = 'Computing the volume of each tile'):
				ranges = [(rect.mins[i], rect.maxes[i]) for i in range(rect.maxes.shape[0])]
				tile_volume, abserr, out_dict = scipy.integrate.nquad(metric_func_, ranges, opts = {'limit':2}, full_output = True)
				tiles_volume.append(tile_volume)
				#print(tile_volume)
		
			volume = sum(tiles_volume)
			
		return volume, tiles_volume
	
	def sample_from_flow(self, N_samples):
		"""
		Samples random points from the normalizing flow model defined on the tiling. It makes sure that the sampled points are inside the tiling.
		The flow must be trained in advance using `tiling_handler.train_flow`.
		
		Parameters
		----------
			N_samples: int
				Number of samples to draw from the normalizing flow
		
		Returns
		-------
			samples: :class:`~numpy:numpy.ndarray`
				shape: (N_samples, D) - 
				`N_samples` samples drawn from the normalizing flow inside the tiling
		"""
		if self.flow is None: raise ValueError("You must train the flow before sampling from it!")
		
		with torch.no_grad():
			samples = self.flow.sample(N_samples).detach().numpy()
		
			_, _, ids_outside = self.is_inside(samples)
			while len(ids_outside)>0:
				new_samples = self.flow.sample(len(ids_outside)).detach().numpy()
				samples[ids_outside] = new_samples
				ids_outside = ids_outside[self.is_inside(new_samples)[2]]

		return np.array(samples)
		
	#@do_profile(follow=[])
	def sample_from_tiling(self, N_samples, seed = None, qmc = False, dtype = np.float64, tile_id = False, p_equal = False):
		"""
		Samples random points from the tiling. It uses Gibb's sampling.
		
		Parameters
		----------
			N_samples: int
				Number of samples to draw from the tiling
			
			seed: int
				Seed for the random points. If `None` no seed will be set
			
			qmc: int
				Whether to use a quasi-Monte carlo method to sample points inside a tile. It uses `scipy.stats.qmc`
			
			dtype: type
				Data type for the sampling (default np.float64)
			
			tile_id: bool
				Whether to output the id of the tile each random point belongs to. Default is `False`
			
			p_equal: bool
				Whether all the tiles should be chosen with equal probability. If False, the probability of having a point in each tile is proportional to its volume
		
		Returns
		-------
			samples: :class:`~numpy:numpy.ndarray`
				shape: (N_samples, D) - 
				`N_samples` samples drawn from the tiling
		"""
		D = self[0][1].shape[0]
		gen = np.random.default_rng(seed = seed)
		if qmc: sampler = scipy.stats.qmc.LatinHypercube(d=D, seed = gen)

		tot_vols, vols = self.compute_volume()
		vols = np.array(vols)/tot_vols #normalizing volumes
		
		
		tiles_rand_id = gen.choice(len(vols), N_samples, replace = True, p = None if p_equal else vols )
		tiles_rand_id, counts = np.unique(tiles_rand_id, return_counts = True)
		
		if qmc:
			samples = [	sampler.random(n=c)*(self[t_id][0].maxes-self[t_id][0].mins)+self[t_id][0].mins
						for t_id, c in zip(tiles_rand_id, counts)]
		else:
			samples = [	gen.uniform(self[t_id][0].mins, self[t_id][0].maxes, (c, D) )
						for t_id, c in zip(tiles_rand_id, counts)]
		samples = np.concatenate(samples, axis = 0, dtype = dtype)

		if tile_id:
			tile_id_vector = np.concatenate([np.full(c, t_id) for t_id, c in zip(tiles_rand_id, counts)])
			return samples, tile_id_vector
		return samples


	def split_tiling(self, d, split):
		"""
		Produce two tilings by splitting the parameter space along dimension `d`. The splitting is done by the threshold `split`.
		It is roughly the equivalent for a tiling to the ``split`` method of :class:`~scipy.spatial.Rectangle`.
		
		Parameters
		----------
			d: int
				Axis to split the tiling along.

			split: float
				Threshold value for the splitting along axis `d`
		
		Returns
		-------
			left: tiling_handler
				Tiling covering the space `<threshold` the given axis
			
			right: tiling_handler
				Tiling covering the space `>threshold` the given axis
		"""
		left, right = tiling_handler(), tiling_handler()
		
		for R, M in self.__iter__():
			if R.maxes[d]<split: #the tile fits in the left tiling
				left.append(tile(R,M))
			elif R.mins[d]>split: #the tile fits in the right tiling
				right.append(tile(R,M))
			else:
				R_left, R_right = R.split(d, split)
				left.append(tile(R_left, M))
				right.append(tile(R_right, M))
		return left, right

	
	def save(self, filename, flowfilename = None):
		"""
		Save the tiling to a file in npy format
		
		Parameters
		----------
		
		filename: str
			File to save the tiling to (in npy format)
		
		flowfilename: str
			File to save the flow to. If None, the flow will not be saved
			
		"""
		#The tiling is saved as a np.array: (N, 2+D, D)
		out_array = []
		for t in self.__iter__():
			out_array.append(np.concatenate([[t[0].mins], [t[0].maxes], t[1]], axis = 0)) #(D+2,D)
		out_array = np.stack(out_array, axis =0) #(N, D+2, D) [[min], [max], metric]

		np.save(filename, out_array)
		
		if isinstance(flowfilename, str):
			if self.flow is None: warnings.warn("The flow model is not trained: unable to save it")
			else:
				self.flow.save_weigths(flowfilename)
		
		return
	
	def load(self, filename):
		"""
		Loads the tiling from a file in npy format.
		The file should be the same layout as produced by ``tiling_handler.load()``
		
		Parameters
		----------
		
		filename: str
			File to load the tiling from (in npy format)
		"""
		try:
			in_array = np.load(filename)
		except FileNotFoundError:
			raise FileNotFoundError("Input tiling file {} does not exist".format(filename))
		for in_ in in_array:
			self.append( tile(scipy.spatial.Rectangle(in_[0,:], in_[1,:]), in_[2:,:]) )

			#updating look_up table
		self.update_KDTree()
		self.compute_volume()
		self._set_boundaries()
		
		return
	
	def load_flow(self, filename):
		"""
		Loads the flow from file. The architecture of the flow must be specified at input.
		
		Parameters
		----------
		
		filename: str
			File to load the tiling from (in npy format)
		
		n_layers: int
			Number of layers of the flow
			See `mbank.flow.STD_GW_flow` for more information

		hidden_features: int
			Number of hidden features for the masked autoregressive flow in use.
			See `mbank.flow.STD_GW_flow` for more information
		"""
		self.flow = STD_GW_Flow.load_flow(filename)
		return
	

	def train_flow(self, N_epochs = 1000, N_train_data = 10000, n_layers = 2, hidden_features = 4, batch_size = None, lr = 0.001, verbose = False):
		"""
		Train a normalizing flow model on the space covered by the tiling, using points sampled from the tiling.
		The flow can be useful for smooth sampling from the tiling and to interpolate the metric within each tiles.
		
		It uses the architecture defined in `mbank.flow.STD_GW_flow`.
		
		Parameters
		----------
			N_epochs: int
				Number of training epochs
			
			N_train_data: int
				Number of training samples from the tiling to be used for training. The validation will be performed with 10% of the training data
			
			n_layers: int
				Number of layers of the flow
				See `mbank.flow.STD_GW_flow` for more information

			hidden_features: int
				Number of hidden features for the masked autoregressive flow in use.
				See `mbank.flow.STD_GW_flow` for more information
			
			batch_size: int
				Amount of training data to be used for each iteration. If None, all the training data will be used.
			
			lr: float
				Learning rate for the training
			
			verbose: bool
				Whether to print the training output

		Returns
		-------
			history: dict
				A dict with the training history.
				See `mbank.flow.GW_flow.train_flow_forward_KL` for more information
		"""
		from torch import optim
		
		assert len(self)>0, "In order to train the flow, the tiling must not be empty"
		D = self[0].D

			#Computing max and mins so that the limits of the flow will be initialized accordingly
		max_val = np.max([ R.maxes for R,_ in self.__iter__()], axis = 0)
		min_val = np.min([ R.mins for R,_ in self.__iter__()], axis = 0)
		
		train_data = self.sample_from_tiling(N_train_data)
		train_data[[0,1],:] = [max_val, min_val]
		validation_data = self.sample_from_tiling(int(0.1*N_train_data))
		
		self.flow = STD_GW_Flow(D, n_layers, hidden_features)
		
		optimizer = optim.Adam(self.flow.parameters(), lr=0.001)
		
		history = self.flow.train_flow_forward_KL(N_epochs, train_data, validation_data, optimizer, batch_size = batch_size, validation_step = 30, callback = None, validation_metric = 'cross_entropy', verbose = verbose)
	
		return history

	#@do_profile(follow=[])
	def get_metric(self, theta, flow = False, kdtree = False):
		"""
		Computes the approximation to the metric evaluated at points theta, as provided by the tiling.
		If `flow` is true, a normalizing flow model will be used to interpolate the metric.
		
		Givent the metric :math:`M_{\\text{T}}` in a given tile with center :math:`\\theta_{\\text{T}}`, the interpolated determinant is

		.. math::
			|M|(\\theta) = \\left( \\frac{p_{\\text{NF}}(\\theta)}{p_{\\text{NF}}(\\theta_{\\text{T}})} \\right)^2 |M_{\\text{T}}|  = f(\\theta; \\theta_{\\text{T}}) |M_{\\text{T}}| 
		
		where :math:`p_{\\text{NF}}(\\theta)` is the probability distribution function defined by the normalizing flow model. :math:`p_{\\text{NF}}` is proportional by construction to the square root of the metric determinant.
		
		The interpolated metric then becomes:
		
		.. math::
			M_{ij}(\\theta) = f^{2/D}(\\theta; \\theta_{\\text{T}}) {M_{\\text{T}}}_{ij}
		
		
		Parameters
		----------
			theta: :class:`~numpy:numpy.ndarray`
				shape: (N, D)/(D,) - 
				Points of the parameter space to evaluate the metric at.
			
			flow: bool
				Whether to use the normalizing flow model. The flow model must be trained in advance using `tiling_handler.train_flow`.
		
			kdtree: bool
				Whether to use a kdtree method to compute the tile. This method is much faster but it may be less accurate as it relies on euclidean distance rather than on the rectangles of the tiling. Argument is passed to `get_tile`
		
		Returns
		-------
			metric: :class:`~numpy:numpy.ndarray`
				shape: (N, D, D)/(D, D) - 
				The metric evaluated according to the tiling
		"""
		theta = np.asarray(theta)
		if theta.ndim ==1: squeeze=True
		else: squeeze=False
		theta = np.atleast_2d(theta)
		
		id_tiles = self.get_tile(theta, kdtree = kdtree)
		metric = np.stack([self[id_].metric for id_ in id_tiles], axis = 0)
		
		if flow:
			if self.flow is None:
				raise ValueError("Cannot evaluate the flow PDF, as the flow is not trained. You can do that with `tiling_handler.train_flow`")

			#TODO: check this factor VERY carefully
					
			centers = np.stack([self[id_].center for id_ in id_tiles], axis = 0)
	
			with torch.no_grad():
				log_pdf_centers = self.flow.log_prob(centers.astype(np.float32))
				log_pdf_theta = self.flow.log_prob(theta.astype(np.float32))
			
				D = self[0].D
				factor = (2/D)*(log_pdf_theta-log_pdf_centers)
				factor = torch.exp(factor).detach().numpy()
			
			metric = (metric.T*factor).T
		
		if squeeze: metric = metric[0]
		return metric

	def get_metric_distance(self, theta1, theta2, flow = False):
		"""
		Computes the squared *metric* distance between ``theta1`` and ``theta2``, using the metric provided by tiling.
		The distance between to points is related to the match :math:`\mathcal{M}(\\theta_1, \\theta_2)` as follow:
		
		.. math::
			d(\\theta_1, \\theta_2) = e^{M_{ij}\Delta\\theta_i \Delta\\theta_j} - 1 \simeq 1 - \mathcal{M}(\\theta_1, \\theta_2) 
			
		where :math:`M_{ij}\Delta\\theta_i \Delta\\theta_j` is the metric approximation to the distance.
		The exponential is required to constrain the distance between 0 and 1.

		If the option flow is True, the distance is computed with an integration of the flow PDF (this may be quite expensive). See :meth:`mbank.handlers.tiling_handler.compute_integral` for more details.
		
		Parameters
		----------
			theta1: :class:`~numpy:numpy.ndarray`
				shape: (N, D)/(D,) - 
				First point

			theta2: :class:`~numpy:numpy.ndarray`
				shape: (N, D)/(D,) - 
				Second point
			
			flow: bool
				Whether to use the flow to compute the distance. Default is False
		Returns
		-------
			dist: :class:`~numpy:numpy.ndarray`
				shape: (N, )/(,) - 
				Distance between ``theta1`` and ``theta2`` computed according to the tiling
		"""
		theta1, theta2 = np.asarray(theta1), np.asarray(theta2)
		if theta1.ndim ==1: squeeze=True
		else: squeeze=False
		theta1, theta2 = np.atleast_2d(theta1), np.atleast_2d(theta2)
		thetac = (theta1+theta2)/2.
		metric = self.get_metric(thetac, flow)
		diff = theta2-theta1
		dist = np.einsum('ij,ijk,ik->i', diff, metric, diff)
		
		if flow:
			if self.flow is None: raise ValueError("Cannot compute the distance using the flow integral as the flow is not trained. You can do so by calling `self.train_flow`.")
			
			D = theta1.shape[-1]
			weight_factor = self.integrate_flow(torch.tensor(theta1, dtype=torch.float32), 
					torch.tensor(theta2, dtype=torch.float32),
					torch.tensor(thetac, dtype=torch.float32),
					N_steps = 100).detach().numpy()
			dist = dist*weight_factor

		if squeeze: dist = dist[0]

		#return dist
		return np.exp(dist)-1 #better, as does not go crazy!

	def integrate_flow(self, theta1, theta2, thetac, N_steps = 100):
		"""
		It performs the following integral:
		
		.. math::
			\int_{0}^{1} \\text{d}t \, \left(\\frac{|M_{\\text{flow}}(\\theta(t))|}{|M_{\\text{flow}}(\\theta_C)|} \\right)^{2/D}
		
		where `D` is the dimensionality of the space and 
		
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
		
		Returns
		-------
			integral: torch.tensor
				shape (1,)/(N,) -
				The result of the integral
		"""
		if self.flow is None: raise ValueError("You should train the flow before computing this integral")

		assert (theta1.ndim >= thetac.ndim) or (theta2.ndim >= thetac.ndim), "The center should have a number of dimensions less or equal than the extrema"

		theta1, theta2 = torch.atleast_2d(theta1), torch.atleast_2d(theta2)
		D = theta1.shape[-1]
		steps = theta1 + torch.einsum("ij,k->kij", theta2-theta1, torch.linspace(0, 1, N_steps))
		#steps = theta1 + torch.outer(theta2-theta1, torch.linspace(0, 1, N_steps))

		old_shape = steps.shape
		steps = torch.flatten(steps, 0, 1) #(N*N_steps, 3)
		
		with torch.no_grad():
			log_pdfs = self.flow.log_prob(steps) #(N*N_steps, )
			log_pdf_center = self.flow.log_prob(torch.atleast_2d(thetac)) #(N, )/()
		
			log_pdfs = torch.reshape(log_pdfs, old_shape[:-1]) #(N_steps, N, )
			log_pdfs = log_pdfs - log_pdf_center #(N_steps, N, )

			factor = torch.pow(torch.exp(log_pdfs), 2./D) #(N*N_steps, )
			
			integral = torch.trapezoid(factor, dx =1/N_steps, axis =0)
		
		return integral


####################################################################################################################







