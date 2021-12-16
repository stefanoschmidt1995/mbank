"""
Some handlers for mbank:
	spin_handler -> takes care of the variables to use
	tiling_handlers -> takes care of the tiling
"""
####################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pandas as pd
import seaborn
import warnings

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
import scipy.spatial

####################################################################################################################

#TODO: change this name
class spin_handler(object):
	"""
	Class to handle a large number of variable layouts. Everything is specified by a string spin_format, available at each call.
	Valid formats for spins are:
		- 'nonspinning': no spins are considered (only two masses), D = 2
		- 's1z_s2z': only the z components of the spins are considered (no precession), D = 4
		- 's1xz': spin components assigned to one BH, everythihng in polar coordinates, D = 4
		- 's1xz_s2z': the two z components are assigned as well as s1x, spin1 in polar coordinates, D = 5
		- 's1xyz_s2z': the two z components are assigned as well as s1x, s1y, spin1 in polar coordinates,  D = 6
		- 'fullspins': all the 6 dimensional spin parameter is assigned,  D = 8
	On top of those, one can specify the mass formats:
		- 'Mq': Total mass and q
		- 'mceta': chirp mass and eta
		- 'm1m2': mass1 and mass2
	If 'iota' or 'iotaphi' is postponed to the format string, also the iota and phase are sampled.
	
	The format shall be provided as 'massformat_spinformat_angles' or as 'massformat_spinformat'.
	For example, valid formats are: 'Mq_s1xz_s2z_iotaphi' or 'm1m2_s1z_s2z'
	"""

	def __init__(self):
		"Initialization. Creates a dict of dict with all the info for each format" 
		self.s_formats = ['nonspinning', 's1z_s2z', 's1xz', 's1xz_s2z', 's1xyz_s2z', 'fullspins'] #spin layouts
		self.m_formats = ['m1m2', 'Mq', 'mceta'] #mass layouts
		temp_format_D = {'nonspinning':2, 's1z_s2z':4, 's1xz':4, 's1xz_s2z':5, 's1xyz_s2z':6, 'fullspins': 8} #dimension of each spin format
		self.format_info = {}
		
		self.valid_formats = []
		self.format_D = {}
			
			#adding m1m2 and Mq to the formats
			#adding iota to precessing modes
		N = len(self.s_formats)
		for i in range(N):
			s_format = self.s_formats[i]
			
			for m_f in self.m_formats:
				s_format_m = m_f +'_'+ s_format
				s_format_iota = s_format_m +'_'+ 'iota'
				s_format_iotaphi = s_format_m + '_'+'iotaphi'
				
				self.valid_formats.append(s_format_m)
				self.format_D[s_format_m] = temp_format_D[s_format]
				self.format_info[s_format_m] = {'D': temp_format_D[s_format], 'm_format': m_f, 's_format': s_format, 'iota':False, 'phi':False}

				self.valid_formats.append(s_format_iota)
				self.format_D[s_format_iota] = temp_format_D[s_format]+1
				self.format_info[s_format_iota] = {'D': temp_format_D[s_format]+1, 'm_format': m_f, 's_format': s_format, 'iota':True, 'phi':False}				
				
				self.valid_formats.append(s_format_iotaphi)
				self.format_D[s_format_iotaphi] = temp_format_D[s_format]+2
				self.format_info[s_format_iotaphi] = {'D': temp_format_D[s_format]+2, 'm_format': m_f, 's_format': s_format, 'iota':True, 'phi':True}

		return

	def switch_BBH(self, theta, spin_format):
		"""
		Given theta, it returns the theta components of the system with switched BBH masses (so that m1>m2)
		Any collective spin (chiP/chi_eff) is attributed to the heavier BH
		
		Parameters
		----------
		
		theta: 'np.ndarray' (N,D)
			Parameters of the BBHs. The dimensionality depends on spin_format
		
		spin_format: 'string'
			How to handle the spin variables.
		"""
		assert spin_format in self.valid_formats, "Wrong spin format given"
		
		if self.format_info[spin_format]['m_format'] == 'm1m2':
			ids = np.where(theta[:,0]<theta[:,1])[0]
			theta[ids,0], theta[ids,1] = theta[ids,1], theta[ids,0] #switching masses
		elif self.format_info[spin_format]['m_format'] == 'Mq':
			ids = np.where(theta[:,1]<1)[0]
			theta[ids,1] = 1./theta[ids,1] #switching masses
		elif self.format_info[spin_format]['m_format'] == 'mceta':
			return theta #this mass configuration is symmetric, no further action is required
		
		if len(ids)<1: return theta
		
		if self.format_info[spin_format]['s_format'] =='nonspinning':
			pass
		elif self.format_info[spin_format]['s_format'] == 's1z_s2z':
			theta[ids,2], theta[ids,3] = theta[ids,3], theta[ids,2] #switching spins
		elif self.format_info[spin_format]['s_format'] == 's1xz':
			pass #chiP is always intended to be on the largest BH (pay attention to this)
		elif self.format_info[spin_format]['s_format'] == 's1xz_s2z':
			theta[ids,3], theta[ids,4] = theta[ids,4], theta[ids,3] #switching spins
		elif self.format_info[spin_format]['s_format'] == 's1xyz_s2z':
			theta[ids,4], theta[ids,5] = theta[ids,5], theta[ids,4] #switching spins
		elif self.format_info[spin_format]['s_format'] == 'fullspins':
			theta[ids,[2,3,4]], theta[ids,[5,6,7]] =  theta[ids,[5,6,7]], theta[ids,[2,3,4]] #switching spins


		return theta
	
	def labels(self, spin_format, latex = False):
		"""
		List the names of the variables for each entry of the BBH parameter vector
		
		Parameters
		----------
		
		spin_format: 'string'
			How to handle the spin variables.
		
		Returns
		-------
		
		labels: 'list'
			List of labels for the parmams in the BBH (each a str)
		
		latex: bool
			Whether the labels should be in latex
		"""
		assert spin_format in self.valid_formats, "Wrong spin format given"
		
		if self.format_info[spin_format]['m_format'] == 'm1m2':
			if latex: labels = [r'$m_1$', r'$m_2$']
			else: labels = ['mass1', 'mass2']
		elif self.format_info[spin_format]['m_format'] == 'Mq':
			if latex: labels = [r'$M$', r'$q$']
			else: labels = ['M', 'q']
		elif self.format_info[spin_format]['m_format'] == 'mceta':
			if latex: labels = [r'$\mathcal{M}_c$', r'$\eta$']
			else: labels = ['Mc', 'eta']
		
		if self.format_info[spin_format]['s_format'] =='nonspinning':
			pass
		elif self.format_info[spin_format]['s_format'] == 's1z_s2z':
			if latex: labels.extend([r'$s_{1z}$', r'$s_{2z}$'])
			else: labels.extend(['s1z', 's2z'])
		elif self.format_info[spin_format]['s_format'] == 's1xz':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$'])
			else: labels.extend(['s1', 'theta1'])
		elif self.format_info[spin_format]['s_format'] == 's1xz_s2z':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$s_{2z}$'])
			else: labels.extend(['s1', 'theta1', 's2z'])
		elif self.format_info[spin_format]['s_format'] == 's1xyz_s2z':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$\phi_1$', r'$s_{2z}$'])
			else: labels.extend(['s1','theta1', 'phi1', 's2z'])
		elif self.format_info[spin_format]['s_format'] == 'fullspins':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$\phi_1$', r'$s_{2}$', r'$\theta_2$', r'$\phi_2$'])
			else: labels.extend(['s1','theta1', 'phi1', 's2z', 'theta2', 'phi2'])
		
		if self.format_info[spin_format]['iota'] and latex: labels.append(r'$\iota$')
		if self.format_info[spin_format]['iota'] and not latex: labels.append('iota')

		if self.format_info[spin_format]['phi'] and latex: labels.append(r'$\phi$')
		if self.format_info[spin_format]['phi'] and not latex: labels.append('phi')
		
		return labels
	
	def D(self, spin_format):
		"""
		Returns the dimensionality of the parameter space required
		
		Parameters
		----------
		
		spin_format: 'string'
			How to handle the spin variables.
		
		Returns
		-------
		
		D: 'int'
			Dimensionality of the BBH parameter vector			
		"""
		assert spin_format in self.valid_formats, "Wrong spin format given"
		return self.format_info[spin_format]['D']

	def format_info(self, spin_format):
		"""
		Returns the a dict with some information about the format.
		The dict has the following entries:
			- m_format : format for the masses
			- s_format : format for the spins
			- D : dimensionality of the BBH space
			- iota : whether the variables include iota
			- phi : whether the variables include phi
		
		Parameters
		----------
		
		spin_format: 'string'
			How to handle the spin variables.
		
		Returns
		-------
		
		format_info: 'int'
			Dictionary with the info for the format
		"""
		assert spin_format in self.valid_formats, "Wrong spin format given"
		return self.format_info[spin_format]

	def get_theta(self, BBH_components, spin_format):
		"""
		Given the BBH components, it returns the components suitable for the bank.
		This function inverts get_BBH_components
		
		Parameters
		----------
		
		BBH_components: 'np.ndarray' (N,10)
			Parameters of the BBHs.
			Each row should be: m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi

		spin_format: 'string'
			How to handle the spin variables.
		
		Returns
		-------
			theta: 'np.ndarray' (N,D)
				Components of the BBH in the format suitable for the bank.
				The dimensionality depends on spin_format
		"""
		assert spin_format in self.valid_formats, "Wrong spin format given"
		
		BBH_components = np.array(BBH_components) #(N,10)/(10,)
		if BBH_components.ndim == 1:
			BBH_components = BBH_components[None,:] #(N,10)
		
		#TODO: fix the metric to accept phi
		assert BBH_components.shape[1] == 10, "The number of BBH parameter is not enough. Expected 10 [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi], given {}".format(BBH_components.shape[1])
		
		if self.format_info[spin_format]['m_format'] == 'm1m2':
			theta = [BBH_components[:,0], BBH_components[:,1]]
		elif self.format_info[spin_format]['m_format'] == 'Mq':
			q = np.maximum(BBH_components[:,1] / BBH_components[:,0], BBH_components[:,0] / BBH_components[:,1])
			theta = [BBH_components[:,0] + BBH_components[:,1], q]
		elif self.format_info[spin_format]['m_format'] == 'mceta':
			eta = np.divide(BBH_components[:,1] * BBH_components[:,0], np.square(BBH_components[:,0] + BBH_components[:,1]) )
			theta = [(BBH_components[:,0] + BBH_components[:,1])*np.power(eta, 3./5.), eta]

			#starting a case swich
		if self.format_info[spin_format]['s_format'] =='nonspinning':
			pass
		elif self.format_info[spin_format]['s_format'] == 's1z_s2z':
			theta.append(BBH_components[:,4])
			theta.append(BBH_components[:,7])
		elif self.format_info[spin_format]['s_format'] == 's1xz':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1) #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			theta.append(s1)
			theta.append(theta1)
		elif self.format_info[spin_format]['s_format'] == 's1xz_s2z':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1)+1e-10 #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			theta.append(s1)
			theta.append(theta1)
			theta.append(BBH_components[:,7])
		elif self.format_info[spin_format]['s_format'] == 's1xyz_s2z':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1)+1e-10 #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			phi1 = np.arctan2(BBH_components[:,3], BBH_components[:,2])
			theta.extend([s1, theta1, phi1, BBH_components[:,7]])
		elif self.format_info[spin_format]['s_format'] == 'fullspins':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1)+1e-10 #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			phi1 = np.arctan2(BBH_components[:,3], BBH_components[:,2])
			s2 = np.linalg.norm(BBH_components[:, 5:8], axis =1)+1e-10 #(N,)
			theta2 = np.arccos(BBH_components[:,7]/s1)
			phi2 = np.arctan2(BBH_components[:,6], BBH_components[:,5])
			theta.extend([s1, theta1, phi1, s2, theta2, phi2])
		else:
			raise RuntimeError("Wrong setting for self.spin_format")
		
			#dealing with iota
		if self.format_info[spin_format]['iota']:
			theta.append(BBH_components[:,8])
		if self.format_info[spin_format]['phi']:
			theta.append(BBH_components[:,9])
		
		theta = np.column_stack(theta)
		
		return theta


	def get_BBH_components(self, theta, spin_format):
		"""
		Given theta, it returns the components suitable for lal
		
		Parameters
		----------
		
		theta: 'np.ndarray' (N,D)
			Parameters of the BBHs. The dimensionality depends on spin_format

		spin_format: 'string'
			How to handle the spin variables.
		
		Returns
		-------
			m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi: 'np.ndarray'
				Components of the BBH in the std parametrization.
				Each has shape (N,)
		"""
		assert spin_format in self.valid_formats, "Wrong spin format given"
		
		theta = np.array(theta)
		if theta.ndim == 1:
			theta = theta[None, :]
			squeeze = True
		else:
			squeeze = False
		
		assert theta.shape[1]>=self.D(spin_format), "The number of BBH parameter is not enough for the given spin format. Expected {}, given {}".format(self.D(spin_format), theta.shape[1])
		
			#setting the masses
		if self.format_info[spin_format]['m_format'] == 'm1m2':
			m1, m2 = theta[:,0], theta[:,1]
		elif self.format_info[spin_format]['m_format'] == 'Mq':
			m1, m2 = theta[:,0]*theta[:,1]/(1+theta[:,1]), theta[:,0]/(1+theta[:,1])
			m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2) #this is to make sure that m1>m2, also if q is less than 1
		elif self.format_info[spin_format]['m_format'] == 'mceta':
				#see https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/tau0tau3.py#L215
			M = theta[:,0] / np.power(theta[:,1], 3./5.)
			if not np.all(theta[:,1]<0.25):
				raise ValueError("Values of the symmetric mass ratio should be all <= 0.25. The given array has some entries not satisfying this: {}".format(theta[:,1]))
			temp_ = np.power(1.0 - 4.0 * theta[:,1], 0.5)
			m1 = 0.5 * M * (1.0 + temp_)
			m2 = 0.5 * M * (1.0 - temp_)
		
			#allocating variables for spins
		s1x, s1y, s1z = np.zeros(m1.shape), np.zeros(m1.shape), np.zeros(m1.shape)
		s2x, s2y, s2z = np.zeros(m1.shape), np.zeros(m1.shape), np.zeros(m1.shape)

			#dealing with spins
		if self.format_info[spin_format]['s_format'] =='nonspinning':
			pass
		elif self.format_info[spin_format]['s_format'] == 's1z_s2z':
			s1z, s2z = theta[:,2], theta[:,3]
		elif self.format_info[spin_format]['s_format'] == 's1xz':
			s1x, s1z = theta[:,2]*np.sin(theta[:,3]), theta[:,2]*np.cos(theta[:,3])
		elif self.format_info[spin_format]['s_format'] == 's1xz_s2z':
			s1x, s1z, s2z = theta[:,2]*np.sin(theta[:,3]), theta[:,2]*np.cos(theta[:,3]), theta[:,4]
		elif self.format_info[spin_format]['s_format'] == 's1xyz_s2z':
			s1x, s1y, s1z, s2z = theta[:,2]*np.sin(theta[:,3])*np.cos(theta[:,4]), theta[:,2]*np.sin(theta[:,3])*np.sin(theta[:,4]), theta[:,2]*np.cos(theta[:,3]), theta[:,5]
		elif self.format_info[spin_format]['s_format'] == 'fullspins':
			s1x, s1y, s1z = theta[:,2]*np.sin(theta[:,3])*np.cos(theta[:,4]), theta[:,2]*np.sin(theta[:,3])*np.sin(theta[:,4]), theta[:,2]*np.cos(theta[:,3])
			s2x, s2y, s2z = theta[:,5]*np.sin(theta[:,6])*np.cos(theta[:,7]), theta[:,5]*np.sin(theta[:,6])*np.sin(theta[:,7]), theta[:,5]*np.cos(theta[:,6])

			#dealing with iota and phi (tricky!!)
		if self.format_info[spin_format]['iota'] and self.format_info[spin_format]['phi']:
			iota, phi = theta[:,-2], theta[:,-1]
		elif self.format_info[spin_format]['iota'] and not self.format_info[spin_format]['phi']:
			iota = theta[:,-1]
			phi = np.zeros(m1.shape)
		elif not self.format_info[spin_format]['iota'] and self.format_info[spin_format]['phi']:
			phi = theta[:,-1]
			iota = np.zeros(m1.shape)
		else:
			iota, phi = np.zeros(m1.shape), np.zeros(m1.shape)
		
		if squeeze:
			m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi = m1[0], m2[0], s1x[0], s1y[0], s1z[0], s2x[0], s2y[0], s2z[0], iota[0], phi[0]
		
		return m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi
	
	def get_mchirp(self, theta, spin_format):
		"""
		Given theta, it returns the chirp mass

		Parameters
		----------
		
		theta: 'np.ndarray' (N,D)
			Parameters of the BBHs. The dimensionality depends on spin_format

		spin_format: 'string'
			How to handle the spin variables.
		
		Returns
		-------
			mchirp: 'np.ndarray'
				Chirp mass of each BBH
		"""
		assert spin_format in self.valid_formats, "Wrong spin format given"
		
		if self.format_info[spin_format]['m_format'] == 'm1m2':
			mchirp = np.power(theta[:,0]*theta[:,1], 3./5.) / np.power(theta[:,0]+theta[:,1], 1./5.)
		elif self.format_info[spin_format]['m_format'] == 'Mq':
			mchirp = theta[:,0] * np.power(theta[:,1]/np.square(theta[:,1]+1), 3./5.)
		elif self.format_info[spin_format]['m_format'] == 'mceta':
			mchirp = theta[:,0]

		return mchirp

	def get_massratio(self, theta, spin_format):
		"""
		Given theta, it returns the mass ratio.

		Parameters
		----------
		
		theta: 'np.ndarray' (N,D)
			Parameters of the BBHs. The dimensionality depends on spin_format

		spin_format: 'string'
			How to handle the spin variables.
		
		Returns
		-------
			q: 'np.ndarray'
				Chirp mass of each BBH
		"""
		assert spin_format in self.valid_formats, "Wrong spin format given"
		
		if self.format_info[spin_format]['m_format'] =='m1m2':
			mchirp = np.maximum(theta[:,1]/theta[:,0], theta[:,0]/theta[:,1])
		elif self.format_info[spin_format]['m_format'] == 'Mq':
			mchirp = theta[:,1]
		elif self.format_info[spin_format]['m_format'] == 'mceta':
			mchirp = theta[:,1] #FIXME: compute this properly! 

		return mchirp

	
####################################################################################################################

class tiling_handler(list):
	"Class for a tiling with I/O helpers"
	
	def __init__(self, filename = None):
		"Creates a tile class"
		super().__init__()
		if isinstance(filename, str): self.load(filename)
		return
	
	def N_templates(self, rect, metric, avg_dist):
		"Computes the number of templates given metric, rect and avg_dist"
		#return rect.volume() * np.sqrt(np.abs(np.prod(np.diag(metric)))) / np.power(avg_dist, metric.shape[0])
		return rect.volume() * np.sqrt(np.abs(np.linalg.det(metric))) / np.power(avg_dist, metric.shape[0])
		#print(create_mesh(avg_dist, (rect, metric) ).shape[0], rect.volume() * np.sqrt(np.abs(np.linalg.det(metric))) / np.power(avg_dist, metric.shape[0]))
		#return create_mesh(avg_dist, (rect, metric) ).shape[0]
		
	
	def get_centers(self):
		"Returns an array with the centers of the tiling"
		centers = np.stack( [(t[0].maxes+t[0].mins)/2. for t in self.__iter__()], axis =0)
		return centers
	
	@ray.remote
	def create_tiling_ray(self, boundaries, N_temp, metric_func, avg_dist, verbose = True, worker_id = None):
		return self.create_tiling(boundaries, N_temp, metric_func, avg_dist, verbose, worker_id)
	
	def create_tiling(self, boundaries, N_temp, metric_func, avg_dist, verbose = True, worker_id = None):
		"Creates a tile list"
		#boundaries is a tuple (max, min)
		boundaries = tuple([np.array(b) for b in boundaries])
		D = boundaries[0].shape[0]
		start_rect = scipy.spatial.Rectangle(boundaries[0], boundaries[1])
		start_metric = metric_func((boundaries[1]+boundaries[0])/2.)
		
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
			dist = - np.square(d_vector)*np.diag(metric)
			
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
			else:  desc = 'Worker {}: volume covered by tiling'.format(worker_id)
			pbar = tqdm(total=100, desc = desc,
				bar_format = "{desc}: {percentage:.2f}%|{bar}| [{elapsed}<{remaining}]")
			V_tot = tiles_list[0][0].volume()
		
		while not (tiles_list == tiles_list_old):
				#updating old one
			tiles_list_old = list(tiles_list)
			V_covered = 0.
				#loops on tiles an updating tiles_list
			for t in tiles_list:

				if t[2] <= N_temp: #a bit of tolerance? Good idea?
					if verbose: V_covered += t[0].volume()
					continue
				
				nt = split(t[0], which_split_axis(t[0], t[1])) #new tiles list (len = 3)
				
				metric_0 = metric_func(get_center(nt[0]))
				metric_2 = metric_func(get_center(nt[2]))
				extended_list = [ (nt[0], metric_0, self.N_templates(nt[0], metric_0, avg_dist)),
								(nt[1], t[1],       self.N_templates(nt[1], t[1], avg_dist)),
								(nt[2], metric_2,   self.N_templates(nt[2], metric_2, avg_dist)), ]
				
				
					#replacing the old tile with a new one
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
	
	def save(self, filename):
		"Save a tiling to file"
		#The tiling is saved as a np.array: (N, 2+D, D)
		out_array = []
		for t in self.__iter__():
			out_array.append(np.concatenate([[t[0].mins], [t[0].maxes], t[1]], axis = 0)) #(D+2,D)
		out_array = np.stack(out_array, axis =0) #(N, D+2, D) [[min], [max], metric]

		np.save(filename, out_array)
		
		return
	
	def load(self, filename):
		try:
			in_array = np.load(filename)
		except FileNotFoundError:
			raise FileNotFoundError("Input tiling file {} does not exist".format(filename))
		for in_ in in_array:
			self.append( (scipy.spatial.Rectangle(in_[0,:], in_[1,:]), in_[2:,:]) )
		
		return


####################################################################################################################







