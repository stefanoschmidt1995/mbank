"""
mbank.handlers
==============
	Two two important handlers class for ``mbank``:
	
	- ``spin_handler``: takes care of the BBH parametrization
	- ``tiling_handlers``: takes care of the tiling of the space
	
	The handlers are used extensively throughout the package
"""
####################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
import itertools

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

from .utils import project_metric

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
	- ``mceta``: chirp mass and eta
	- ``m1m2``: mass1 and mass2
	
	Valid formats for spins are:
	
	- ``nonspinning``: no spins are considered (only two masses), D = 0
	- ``s1z_s2z``: only the z components of the spins are considered (no precession), D = 2
	- ``s1xz``: spin components assigned to one BH in plane xz, D = 2
	- ``s1xyz``: spin components assigned to one BH, D = 3
	- ``s1xz_s2z``: the two z components are assigned as well as s1x, D = 3
	- ``s1xyz_s2z``: the two z components are assigned as well as s1x, s1y,  D = 4
	- ``fullspins``: all the 6 dimensional spin parameter is assigned,  D = 6
	
	Whenever a precessing spin is assigned to a BH, the spin is *always* expressed in sperical coordinates s1, theta1, phi1.

	On top of this, the user can *optionally* specify a format for Angles and for the Eccentricity
	
	Valid formats for the angles are:
	
	- ``iota``: the inclination angle is included
	- ``iotaphi``: the inclination angle and reference phase are included

	Valid formats for the eccentricity are:
	
	- ``e``: to include the orbital eccentricity
	- ``emeanano``: to include the orbital eccentricity and mean periastron anomaly

	Note that some waveform approximants do not support all these paramters. Using a format that sets them to be non zero will likely fail. 
	
	For example, valid formats are: ``mceta_s1xz_s2z_e_iotaphi``, ``m1m2_nonspinning_e``, ``Mq_s1xz_s2z_iotaphi``, ``m1m2_s1z_s2z``
	"""

	def __init__(self):
		"Initialization. Creates a dict of dict with all the info for each format" 
		
			#hard coding valid formats for masses, spins, eccentricity and angles
		self.m_formats = ['m1m2', 'Mq', 'mceta'] #mass layouts
		self.s_formats = ['nonspinning', 's1z_s2z', 's1xz', 's1xyz', 's1xz_s2z', 's1xyz_s2z', 'fullspins'] #spin layouts
		self.e_formats = ['', 'e', 'emeanano'] #eccentric layouts
		self.angle_formats = ['', 'iota', 'iotaphi'] #angles layouts
		
			#hard coding dimensions for each format
		D_spins = {'nonspinning':0, 's1z_s2z':2, 's1xz':2, 's1xyz':3, 's1xz_s2z':3, 's1xyz_s2z':4, 'fullspins': 6} #dimension of each spin format
		D_ecc = {'':0, 'e':1, 'emeanano':2} #dimension of each eccentric format
		D_angles = {'':0, 'iota':1, 'iotaphi':2} #dimension of each angle format

			#creating info dictionaries
		self.format_info = {}
		self.valid_formats = []
		self.format_D = {}
			
		for m_, s_, e_, a_ in itertools.product(self.m_formats, self.s_formats, self.e_formats, self.angle_formats):
				#nonspinning and noneccentric formats don't sample angles...
			if s_ == 'nonspinning' and a_ != '' and e_ == '': continue 
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
		
		return

	def switch_BBH(self, theta, variable_format):
		"""
		Given theta, it returns the theta components of the system with switched BBH masses (so that m1>m2)
		If only BH1 has an in-plane component, only the z components of the spins will be switched: this is equivalent to assume that the in-plane spin of BH1 is a collective spin.
		
		Parameters
		----------
		
		theta: np.ndarray
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on variable_format
		
		variable_format: string
			How to handle the BBH variables.
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			ids = np.where(theta[:,0]<theta[:,1])[0]
			theta[ids,0], theta[ids,1] = theta[ids,1], theta[ids,0] #switching masses
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
			ids = np.where(theta[:,1]<1)[0]
			theta[ids,1] = 1./theta[ids,1] #switching masses
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			return theta #this mass configuration is symmetric, no further action is required
		
		if len(ids)<1: return theta
		
		if self.format_info[variable_format]['spin_format'] =='nonspinning':
			pass
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
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			if latex: labels = [r'$\mathcal{M}_c$', r'$\eta$']
			else: labels = ['Mc', 'eta']
		
		if self.format_info[variable_format]['spin_format'] =='nonspinning':
			pass
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
		
		BBH_components: np.ndarray
			shape: (N,12)/(12,) -
			Parameters of the BBHs.
			Each row should be: m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
			theta: np.ndarray
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
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			eta = np.divide(BBH_components[:,1] * BBH_components[:,0], np.square(BBH_components[:,0] + BBH_components[:,1]) )
			theta = [(BBH_components[:,0] + BBH_components[:,1])*np.power(eta, 3./5.), eta]

			#starting a case swich
		if self.format_info[variable_format]['spin_format'] =='nonspinning':
			pass
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
		
		theta: np.ndarray (N,D)
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
		
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano iota, phi: np.ndarray
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
		
		if squeeze:
			m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,  e, meanano, iota, phi = m1[0], m2[0], s1x[0], s1y[0], s1z[0], s2x[0], s2y[0], s2z[0], e[0], meanano[0], iota[0], phi[0]
		
		return m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi
	
	def get_mchirp(self, theta, variable_format):
		"""
		Given theta, it returns the chirp mass

		Parameters
		----------
		
		theta: np.ndarray
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
			mchirp: np.ndarray
				Chirp mass of each BBH
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			mchirp = np.power(theta[:,0]*theta[:,1], 3./5.) / np.power(theta[:,0]+theta[:,1], 1./5.)
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
			mchirp = theta[:,0] * np.power(theta[:,1]/np.square(theta[:,1]+1), 3./5.)
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			mchirp = theta[:,0]

		if squeeze: mchirp = mchirp[0]
		return mchirp

	def get_massratio(self, theta, variable_format):
		"""
		Given theta, it returns the mass ratio.

		Parameters
		----------
		
		theta: np.ndarray
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
			q: np.ndarray
				Chirp mass of each BBH
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		if self.format_info[variable_format]['mass_format'] =='m1m2':
			q = np.maximum(theta[:,1]/theta[:,0], theta[:,0]/theta[:,1])
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
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
		
		m1, m2: np.ndarray
			shape: (N,)/() -
			Masses of the two BHs
			It assumes m1>=m2

		s1x, s1y: np.ndarray
			shape: (N,)/() -
			In-plane spins of the primary black hole
		
		s1z: np.ndarray
			shape: (N,)/() -
			Aligned spin for the primary black hole. Used to enforce Kerr limit in the spin parameter
		
		s2x, s2y: np.ndarray
			shape: (N,)/() -
			In-plane spins of the secondary black hole
		
		Returns
		-------
		
		chiP: np.ndarray
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
		
		m1, m2: np.ndarray
			shape: (N,)/() -
			Masses of the two BHs

		s1x, s1y: np.ndarray
			shape: (N,)/() -
			In-plane spins of the primary black hole
		
		s1z: np.ndarray
			shape: (N,)/() -
			Aligned spin for the primary black hole. Used to enforce Kerr limit in the spin parameter (if it is the case)
		
		s2x, s2y: np.ndarray
			shape: (N,)/() -
			In-plane spins of the secondary black hole

		s2z: np.ndarray
			shape: (N,)/() -
			Aligned spin for the secondary black hole. Used to enforce Kerr limit in the spin parameter (if it is the case)
		
		only_spin1: bool
			Whether to assign the precessing spin only always to the primary BH.
			The default is False, as 2012.02209 suggests.
		
		Returns
		-------
		
		chiP_2D_1: np.ndarray
			shape: (N,2)/(2,) -
			In-plane (x and y) components of the two dimensional precessing spin parameter on the primary BH
		
		chiP_2D_2: np.ndarray
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
		
		BBH_components: np.ndarray
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
		
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano iota, phi: np.ndarray
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
		
		theta: np.ndarray
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: string
			How to handle the BBH variables.
		
		Returns
		-------
			theta: np.ndarray
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

class tiling_handler(list):
	"""
	Class for a tiling with I/O helpers.
	A tiling is a list of tiles that cover a larger space.
	Each tile, consists in:
	
	- an hypercubes (``scipy.spatial.Rectangle`` object) that defines its boundaries
	- a metric that it's used to compute distances between points of the tile. It is represented by a DxD matrix (``np.ndarray`` object), where D is the dimensionality of the space.
	
	Overall a tiling handler looks like:
	
	::
	
		[(rect_1, metric_1), (rect_2, metric_2), ..., (rect_N, metric_N)]
	"""
	
	def __init__(self, filename = None):
		"""
		Initializes the tilin handler.
		
		Parameters
		----------
			filename: str
				Optional filename. If it is given, the tiling handler is loaded from it.
				The file shall be the same format as produced by ``tiling_handler.save()``
		"""
		super().__init__()
		if isinstance(filename, str): self.load(filename)
		return
	
	def get_centers(self):
		"""
		Returns an array with the centers of the tiling
		
		Returns
		-------
			centers: np.ndarray
				shape (N,D) -
				All the centers of the tiles
			
		"""
		centers = np.stack( [(t[0].maxes+t[0].mins)/2. for t in self.__iter__()], axis =0)
		return centers

	def N_templates(self, rect, metric, avg_dist):
		"""
		Computes the approximate number of templates given metric, rect and avg_dist

		`N_templates` is compute as:
		
		::
			
			N_templates = rect.volume() * sqrt(abs(det(metric))) / avg_dist**D

		`N_templates` is a measure of volume if `avg_dist` is kept fixed

		Parameters
		----------
			rect: scipy.spatial.Rectangle
				Rectangle object defining the boundaries of the tile

			metric: np.ndarray
				shape: (D,D) -
				Metric to use within the tile
				
			avg_dist: float
				Desidered average distance between templates

		Returns
		-------
			N_templates: float
				The number or templates that should lie inside the tile (i.e. the rectangle with the given metric)
			
		"""
		#return rect.volume() * np.sqrt(np.prod(np.diag(metric))) / np.power(avg_dist, metric.shape[0])
		return rect.volume() * np.sqrt(np.abs(np.linalg.det(metric))) / np.power(avg_dist, metric.shape[0])

			#This trash is to look for projecting over many variable and check which gives the maximum number of templates
		#D =  metric.shape[0]
		#rect_diff = rect.maxes - rect.mins
		#N_templates_list = []
		#for l in range(1, D+1):
		#	for axes in itertools.combinations(range(D),l):	
		#		axes = list(axes)
		#		proj_metric = project_metric(metric, axes)
		#		dist_ = avg_dist*np.sqrt(D/len(axes))
		#		N_templates_list.append( np.prod(rect_diff[axes])* np.sqrt(np.abs(np.linalg.det(proj_metric))) / np.power(dist_, len(axes)) )

		#return np.max(N_templates_list)

	@ray.remote
	def create_tiling_ray(self, boundaries, N_temp, metric_func, avg_dist = 0.1, verbose = True, worker_id = None):
		"Wrapper to `create_tiling` to allow for `ray` parallel execution. See `handlers.tiling_hander.create_tiling()` for more information."
		return self.create_tiling(boundaries, N_temp, metric_func, avg_dist, verbose, worker_id)
	
	def create_tiling(self, boundaries, N_temp, metric_func, avg_dist = 0.1, verbose = True, worker_id = None):
		"""
		Create a tiling within the boundaries by a hierarchical iterative splitting
		
		Parameters
		----------
		
		boundaries: np.ndarray
			shape: (2,D) -
			Boundaries of the space to tile.
			Lower limit is ``boundaries[0,:]`` while upper limits is ``boundaries[1,:]``

		N_temp: int
			Maximum number of templates that shall lie within the tile. The number of templates in each tile will be always smaller than ``N_temp``
			Default is 0.1, the reference value for `cbc_bank.generate_bank()`.
			
		metric_func: function
			A function that accepts theta and returns the metric.
			A common usage would be:
			
			::
			
				metric_obj = mbank.metric.cbc_metric(**args)
				metric_func = metric_obj.get_metric
		
		avg_dist: float
				Desidered average distance between templates. To compute the number of templates in each tile via ``mbank.utils.N_templates()``
			
		verbose: bool
			Whether to print the progress of the computation
		
		worker_id: int
			If given, it will print the worker id in the progress bar.
		
		Returns
		-------
			self: tiling_handler
				Return this object filled with the desired tiling
		
		"""
		#boundaries is a tuple (max, min)
		boundaries = tuple([np.array(b) for b in boundaries])
		D = boundaries[0].shape[0]
		start_rect = scipy.spatial.Rectangle(boundaries[0], boundaries[1])
		start_metric = metric_func((boundaries[1]+boundaries[0])/2.)
		
		if start_rect.volume()<1e-19:
			warnings.warn("The given boundaries are degenerate (i.e. zero volume) and a single tile was generated: this may not be what you expected")
			self.clear() #empty whatever was in the old tiling
			self.extend( [(start_rect, start_metric)])
			return self
		
		####
		#Defining some convenience function
		
			#splitting procedure
		def split(rect, d):
			len_d = rect.maxes[d] - rect.mins[d]
			rect_1, rect_23 = rect.split(d, rect.mins[d] + len_d/3.)
			rect_2, rect_3 = rect_23.split(d, rect.mins[d] + len_d*2./3.)
			#print(rect, rect_1, rect_23, rect_2, rect_3);quit()
			return [rect_1, rect_2, rect_3]

			#which axis should be split?
		def which_split_axis(rect, metric):
			d_vector = rect.maxes - rect.mins #(D,)
			dist = np.square(d_vector)*np.diag(metric)
			
			#dist = - np.einsum('ij,jk,ik -> i', np.diag(d_vector), metric, np.diag(d_vector)) #distance is -match !!
			#print("\t", dist, -np.einsum('ij,jk,ik -> i', np.diag(d_vector), metric, np.diag(d_vector)))
			
			return np.argmax(dist)
		
		def get_center(rect):
			return (rect.maxes + rect.mins)/2.
		
			#tiles list is initialized with a start cell. The first split is done no matter what
		#tiles_list = [(start_rect, start_metric, N_temp+1)] 
		tiles_list = [(start_rect, start_metric, self.N_templates(start_rect, start_metric, avg_dist))] 
		tiles_list_old = []
		
			 #progress bar = % of volume covered
		if verbose:
			if worker_id is None: desc = 'Volume covered by the tiling'
			else:  desc = 'Worker {} - Volume covered by tiling'.format(worker_id)
			pbar = tqdm(total=100, desc = desc,
				bar_format = "{desc}: {percentage:.2f}%|{bar}| [{elapsed}<{remaining}]")
			V_tot = tiles_list[0][0].volume()
		
		while not (tiles_list == tiles_list_old):
				#updating old one
			tiles_list_old = list(tiles_list)
			V_covered = 0.
				#loops on tiles an updating tiles_list
			for t in tiles_list:

				if t[2] <= N_temp:
					if verbose: V_covered += t[0].volume()
					continue
				
				nt = split(t[0], which_split_axis(t[0], t[1])) #new tiles list (len = 3)
				
				metric_0 = metric_func(get_center(nt[0]))
				metric_2 = metric_func(get_center(nt[2]))
				extended_list = [ (nt[0], metric_0, self.N_templates(nt[0], metric_0, avg_dist)),
								(nt[1], t[1],       self.N_templates(nt[1], t[1], avg_dist)),
								(nt[2], metric_2,   self.N_templates(nt[2], metric_2, avg_dist)) ]
				
				
					#replacing the old tile with the new ones
				tiles_list.remove(t)
				tiles_list.extend(extended_list)

			if verbose: pbar.update(np.round(min(V_covered/V_tot*100. - pbar.last_print_n, 100),2))

		if verbose: pbar.close()
		#plot_tiles(tiles_list, boundaries)
		#plt.show()
		
		self.clear() #empty whatever was in the old tiling
		self.extend( [(t[0],t[1]) for t in tiles_list] )
		#TODO: this might have an easier interface with
		#self.extend( [{'rect':t[0], 'metric':t[1]} for t in tiles_list] )
		
		return self
	
	def compute_volume(self, metric_func = None):
		"""
		Compute the volume of the space an the volume of each tile.
		The volume will be computed with the metric approximation by default.
		If `metric_function` is a function, the volume will be computed by means of an integration.
		It must returns a metric approximation given a point theta in D dimension.
		
		The volume is computed as the integral of ``sqrt{|M(theta)|}`` over the D dimensional space of the tile.
		
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
			tiles_volume = [rect.volume()*np.sqrt(np.abs(np.linalg.det(metric))) for rect, metric in self.__iter__()]
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
	
	def sample_from_tiling(self, N_samples):
		"""
		Samples random points from the tiling. It uses Gibb's sampling.
		
		Parameters
		----------
			N_samples: int
				Number of samples to draw from the tiling
		
		Returns
		-------
			samples: np.ndarray
				shape: (N_samples, D) - 
				`N_samples` samples drawn from the tiling
		"""
		tot_vols, vols = self.compute_volume()
		vols = np.array(vols)/tot_vols #normalizing volumes
		
		tiles_rand_id = np.random.choice(len(vols), N_samples, replace = True, p = vols)
		tiles_rand_id, counts = np.unique(tiles_rand_id, return_counts = True)
		
		D = self[0][1].shape[0]
		
		samples = np.concatenate([
					np.random.uniform(self[t_id][0].mins, self[t_id][0].maxes, (c, D) )
					for t_id, c in zip(tiles_rand_id, counts)], axis = 0)
		return samples		
	
	def save(self, filename):
		"""
		Save the tiling to a file in npy format
		
		Parameters
		----------
		
		filename: str
			File to save the tiling to (in npy format)
			
		"""
		#The tiling is saved as a np.array: (N, 2+D, D)
		out_array = []
		for t in self.__iter__():
			out_array.append(np.concatenate([[t[0].mins], [t[0].maxes], t[1]], axis = 0)) #(D+2,D)
		out_array = np.stack(out_array, axis =0) #(N, D+2, D) [[min], [max], metric]

		np.save(filename, out_array)
		
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
			self.append( (scipy.spatial.Rectangle(in_[0,:], in_[1,:]), in_[2:,:]) )
		
		return


####################################################################################################################







