"""
mbank
=====
	A geometric placement method for a template bank in gravitational waves data analysis
"""

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

import poisson_disc #https://pypi.org/project/poisson-disc/

import lal 
import lalsimulation as lalsim

from tqdm import tqdm

import ray

	#for older versions of the code...
#from sklearn.cluster import KMeans
#import emcee

import scipy.stats
import scipy.spatial

from .utils import plot_tiles, get_templates_ray, plawspace, create_mesh, create_mesh_new, points_in_hull, all_line_hull_intersection, sample_from_hull, sample_from_hull_boundaries, get_pad_points_2d, get_pad_points, get_boundary_box #, plot_tiles_templates

#TODO: create a package for placing N_points in a box with lloyd algorithm (extra)

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
#TODO: TEST GRADS & WF GENERATION

#FIXME: the trained model for IMR does not agree with the one of TEOB
#FIXME: there are some problems with the metric when the PSD is included...
#TODO: insert the correct formula for the time-frequency relation

#FIXME: you are not able to perform the FFT of the WFs... Learn how to do it and do it well!

#TODO: Understand how to tune the hyperparameters: len(seed_bank) and N_samples from hull

#TODO: eventually you should set a better import chain

####################################################################################################################
def plot_tiles_templates(t_obj, templates, spin_format, save_folder, show = False):
	#FIXME: here the import statement is fucked up... you can do better!
	s_handler = spin_handler()
	if not save_folder.endswith('/'): save_folder.out_dir = save_folder.out_dir+'/'
	if templates.shape[0] >50000:
		ids_ = np.random.choice(templates.shape[0], 50000, replace = False)
	else:
		ids_ = range(templates.shape[0])
	seaborn.pairplot(pd.DataFrame(data=templates[ids_,:], columns = s_handler.labels(spin_format, True)),  markers = 'o', plot_kws={"s": 3}, corner=True)
	plt.savefig(save_folder+'bank.png')
	
	centers = np.array([ (t[0].maxes + t[0].mins)/2. for t in t_obj])
	
	if centers.shape[0] > 5000: s = 10
	else: s = 50
	
	seaborn.pairplot(pd.DataFrame(data=centers, columns = s_handler.labels(spin_format, True)), diag_kind = None,  markers = 'o', plot_kws={"s": s, 'c': ['red' for i in range(len(centers))]}, corner=True)
	plt.savefig(save_folder+'tiling.png')
	
	if templates.shape[1] ==2:
			#plotting rectangles
		plt.figure()
		currentAxis = plt.gca()
		for t in t_obj:
			d = t[0].maxes- t[0].mins
			currentAxis.add_patch(matplotlib.patches.Rectangle(t[0].mins, d[0], d[1], fill = None, alpha =1))
		plt.scatter(*centers.T, c = 'r')
		plt.scatter(*templates.T, s = 3, c ='b')
		plt.xlabel(s_handler.labels(spin_format, True)[0])
		plt.ylabel(s_handler.labels(spin_format, True)[1])
		plt.savefig(save_folder+'rectangles.png')
	if show: plt.show()
	
	return

####################################################################################################################

class DefaultSnglInspiralTable(lsctables.SnglInspiralTable):
	"""
	#NOT VERY ELEGANT... FIND A BETTER WAY OF DOING THINGS
	This is a copy of SnglInspiralTable with implmented defaults.
	Implemented as here: https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/waveforms.py#L39
	"""
	def __init__(self, *args, **kwargs):
		lsctables.SnglInspiralTable.__init__(self, *args, **kwargs)
		for entry in self.validcolumns.keys():
			if not(hasattr(self, entry)):
				if self.validcolumns[entry] in ['real_4', 'real_8']:
					setattr(self, entry, 0.)
				elif self.validcolumns[entry] == 'int_4s':
					setattr(self, entry, 0)
				elif self.validcolumns[entry] == 'lstring':
					setattr(self, entry, '')
				elif self.validcolumns[entry] == 'ilwd:char':
					setattr(self, entry, '')
			else:
				print("Column %s not recognized" % entry, file=sys.stderr)
				raise ValueError

class DefaultSimInspiralTable(lsctables.SimInspiralTable):
	"""
	#NOT VERY ELEGANT... FIND A BETTER WAY OF DOING THINGS
	This is a copy of SnglInspiralTable with implmented defaults.
	Implemented as here: https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/waveforms.py#L39
	"""
	def __init__(self, *args, **kwargs):
		lsctables.SnglInspiralTable.__init__(self, *args, **kwargs)
		for entry in self.validcolumns.keys():
			if not(hasattr(self, entry)):
				if self.validcolumns[entry] in ['real_4', 'real_8']:
					setattr(self, entry, 0.)
				elif self.validcolumns[entry] == 'int_4s':
					setattr(self, entry, 0)
				elif self.validcolumns[entry] == 'lstring':
					setattr(self, entry, '')
				elif self.validcolumns[entry] == 'ilwd:char':
					setattr(self, entry, '')
			else:
				print("Column %s not recognized" % entry, file=sys.stderr)
				raise ValueError

####################################################################################################################


#TODO: change this name
class spin_handler(object):
	"""
	Class to handle a large number of variable layouts. Everything is specified by a string spin_format, available at each call.
	Valid formats for spins are:
		- 'nonspinning': no spins are considered (only two masses), D = 2
		- 's1z_s2z': only the z components of the spins are considered (no precession), D = 4
		- 'chiP_chieff': spin components assigned to one BH, s1x = chiP, s1z = chieff, everythihng in polar coordinates, D = 4
		- 'chiP_s1z_s2z': the two z components are assigned as well as s1x, spin1 in polar coordinates, D = 5
		- 'chiP2d_s1z_s2z': the two z components are assigned as well as s1x, s1y, spin1 in polar coordinates,  D = 6
		- 'fullspins': all the 6 dimensional spin parameter is assigned,  D = 8
	On top of those, one can specify the mass formats:
		- 'Mq': Total mass and q
		- 'mceta': chirp mass and eta
		- 'm1m2': mass1 and mass2
	If 'iota' or 'iotaphi' is postponed to the format string, also the iota and phase are sampled.
	
	The format shall be provided as 'massformat_spinformat_angles' or as 'massformat_spinformat'.
	For example, valid formats are: 'Mq_chiP_s1z_s2z_iotaphi' or 'm1m2_s1z_s2z'
	"""

	def __init__(self):
		"Initialization. Creates a dict of dict with all the info for each format" 
		self.s_formats = ['nonspinning', 's1z_s2z', 'chiP_chieff', 'chiP_s1z_s2z', 'chiP2d_s1z_s2z', 'fullspins'] #spin layouts
		self.m_formats = ['m1m2', 'Mq', 'mceta'] #mass layouts
		temp_format_D = {'nonspinning':2, 's1z_s2z':4, 'chiP_chieff':4, 'chiP_s1z_s2z':5, 'chiP2d_s1z_s2z':6, 'fullspins': 8} #dimension of each spin format
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
		elif self.format_info[spin_format]['s_format'] == 'chiP_chieff':
			pass #chiP is always intended to be on the largest BH (pay attention to this)
		elif self.format_info[spin_format]['s_format'] == 'chiP_s1z_s2z':
			theta[ids,3], theta[ids,4] = theta[ids,4], theta[ids,3] #switching spins
		elif self.format_info[spin_format]['s_format'] == 'chiP2d_s1z_s2z':
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
		elif self.format_info[spin_format]['s_format'] == 'chiP_chieff':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$'])
			else: labels.extend(['s1', 'theta1'])
		elif self.format_info[spin_format]['s_format'] == 'chiP_s1z_s2z':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$s_{2z}$'])
			else: labels.extend(['s1', 'theta1', 's2z'])
		elif self.format_info[spin_format]['s_format'] == 'chiP2d_s1z_s2z':
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
		elif self.format_info[spin_format]['s_format'] == 'chiP_chieff':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1) #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			theta.append(s1)
			theta.append(theta1)
		elif self.format_info[spin_format]['s_format'] == 'chiP_s1z_s2z':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1)+1e-10 #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			theta.append(s1)
			theta.append(theta1)
			theta.append(BBH_components[:,7])
		elif self.format_info[spin_format]['s_format'] == 'chiP2d_s1z_s2z':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1)+1e-10 #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			phi1 = atan2(BBH_components[:,3], BBH_components[:,2])
			theta.extend([s1, theta1, phi1, BBH_components[:,7]])
		elif self.format_info[spin_format]['s_format'] == 'fullspins':
			s1 = np.linalg.norm(BBH_components[:,2:5], axis =1)+1e-10 #(N,)
			theta1 = np.arccos(BBH_components[:,4]/s1)
			phi1 = atan2(BBH_components[:,3], BBH_components[:,2])
			s2 = np.linalg.norm(BBH_components[:, 5:8], axis =1)+1e-10 #(N,)
			theta2 = np.arccos(BBH_components[:,7]/s1)
			phi2 = atan2(BBH_components[:,6], BBH_components[:,5])
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
		elif self.format_info[spin_format]['m_format'] == 'mceta':
				#see https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/tau0tau3.py#L215
			M = theta[:,0] / np.power(theta[:,1], 3./5.)
			if not np.all(theta[:,1]<0.25): print("WTF?? eta> 0.25!! ",theta[:,1])
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
		elif self.format_info[spin_format]['s_format'] == 'chiP_chieff':
			s1x, s1z = theta[:,2]*np.sin(theta[:,3]), theta[:,2]*np.cos(theta[:,3])
		elif self.format_info[spin_format]['s_format'] == 'chiP_s1z_s2z':
			s1x, s1z, s2z = theta[:,2]*np.sin(theta[:,3]), theta[:,2]*np.cos(theta[:,3]), theta[:,4]
		elif self.format_info[spin_format]['s_format'] == 'chiP2d_s1z_s2z':
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

class WF_metric(object):
	"This class implements the metric on the space, defined for each point of the space. The metric is defined as M(theta)_ij = <d_i h | d_j h>"
	
	def __init__(self, PSD = None, f_min = 10., f_high = None, approx = 'mlgw', spin_format = 'nonspinning'):
		"""
		Initialize the class.
		
		Parameters
		----------
		
		PSD: ('np.ndarray', 'np.ndarray')
			PSD for computing the scalar product.
			It is a tuple with a frequency grid array and a PSD array
			PSD should be stored in an array which stores its value on a grid of evenly spaced positive frequencies.
			If None, the PSD is assumed to be flat
		
		f_min: 'float'
			Minimum frequency at which the scalar product is computed (and the WF is generated from)
		
		f_high: `float`
			Cutoff for the high frequencies in the PSD. If not None, frequencies up to f_high will be removed from the computation
			If None, no cutoff is applied
		
		approx: 'string'
			Which approximant to use. It can be 'mlgw' or any lal approx.
			The approximant can be changed with set_approximant() and can be accessed under name WF_metric.approx
		
		spin_format: 'string'
			How to handle the spin variables. Different options are possible and which option is set, will decide the dimensionality D of the parameter space (hence of the input).
			Spin format can be changed with set_spin_format() and can be accessed under name WF_metric.spin_format
			Valid formats are:
				- 'nonspinning': no spins are considered (only two masses), D = 2
				- 's1z_s2z': only the z components of the spins are considered (no precession), D = 4
				- 'chieff_chiP': spin components assigned to one BH, s1x = chiP, s1z = chieff, D = 4
				- 'chiP_s1z_s2z': the two z components are assigned as well as s1x, D = 5
				- 'chiP2d_s1z_s2z': the two z components are assigned as well as s1x, s1y, D = 6
				- 'fullspins': all the 6 dimensional spin parameter is assigned,  D = 8
		"""
		self.s_handler = spin_handler() #this obj is to keep in a single place all the possible spin manipulations that may be required

		self.set_approximant(approx)
		self.set_spin_format(spin_format)
		
		self.f_min = f_min
		
		if not isinstance(PSD, tuple):
			raise ValueError("Wrong format for the PSD. Expected a tuple of np.ndarray, got {}".format(type(PSD)))

		#####Tackling the PSD
		self.PSD = PSD[1]
		self.f_grid = PSD[0]
		self.delta_f = self.f_grid[1]-self.f_grid[0]
			#checking that grid is equally spaced
		assert np.all(np.diff(self.f_grid)-self.delta_f<1e-10), "Frequency grid is not evenly spaced!"
		
		#####applying a high frequency cutoff to the PSD
		if isinstance(f_high, (int, float)): #high frequency cutoff
			self.f_high = f_high
			self.f_grid = np.linspace(0., f_high, int(f_high/self.delta_f)+1)
			self.PSD = np.interp(self.f_grid, PSD[0], PSD[1])
		else:
			self.f_high = self.f_grid[-1]
		
		return
	
	def set_approximant(self, approx):
		"""
		Change the WF approximant used
		
		Parameters
		----------
		
		approx: 'string'
			Which approximant to use. It can be 'mlgw' or any lal approx.
		"""
		self.approx = approx
		
			#checking if the approximant is right
		try:
			lal_approx = lalsim.SimInspiralGetApproximantFromString(approx) #getting the approximant
		except RuntimeError:
			raise RuntimeError("Wrong approximant name: it must be either \'mlgw\' either an approximant supported by lal")

		return
	
	def set_spin_format(self, spin_format):
		"""
		Set the spin_format to be used.
		
		Valid formats are:
			- 'nonspinning': no spins are considered (only two masses), D = 2
			- 's1z_s2z': only the z components of the spins are considered (no precession), D = 4
			- 'chieff_chiP': spin components assigned to one BH, s1x = chiP, s1z = chieff, D = 4
			- 'chiP_s1z_s2z': the two z components are assigned as well as s1x, D = 5
			- 'chiP2d_s1z_s2z': the two z components are assigned as well as s1x, s1y, D = 6
			- 'fullspins': all the 6 dimensional spin parameter is assigned,  D = 8

		where D is the dimensionality of the BBH space considered
		
		Parameters
		----------
		
		spin_format: 'string'
			An identifier for the spin format
		"""
		assert spin_format in self.s_handler.valid_formats, "Wrong spin format '{}'. Available formats are: ".format(spin_format)+str(self.s_handler.valid_formats)
		
		self.spin_format = spin_format

		return
	
	def get_metric_determinant(self, theta):
		"""
		Returns the metric determinant
		
		Parameters
		----------
		
		theta: 'np.ndarray' (N,D)
			parameters of the BBHs. The dimensionality depends on the spin format set for the metric

		Returns
		-------
		
		det : 'np.ndarray' (N,)
			Determinant of the metric for the given input
			
		"""
		return np.linalg.det(self.get_metric(theta)) #(N,)

	def log_pdf_gauss(self, theta, boundaries = None):
		"Gaussian PDF, for the purpose of debugging and testing the MCMC..."
		return -0.5*np.sum(np.square(theta+10.), axis =1) #DEBUG!!!!!

	def log_pdf(self, theta, boundaries = None):
		"""
		Returns the logarithm log(pdf) of the probability distribution function we want to sample from:
			pdf = p(theta) ~ sqrt(|M(theta)|)
		imposing the boundaries
		
		Parameters
		----------
		
		theta: 'np.ndarray' (N,D)
			parameters of the BBHs. The dimensionality depends on the spin format set for the metric
		
		boundaries: 'np.ndarray' (2,D)
			An optional array with the boundaries for the model. If a point is asked below the limit, -10000000 is returned
			Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
			If None, no boundaries are implemented
		
		Returns
		-------
		
		log_pdf : 'np.ndarray' (N,)
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
			if boundaries.shape != (2,self.format_D[self.spin_format]):
				raise ValueError("Wrong shape of boundaries given: expected (2,{}), given {}".format(self.format_D[self.spin_format], boundaries.shape))
			
			ids_ok = np.logical_and(np.all(theta > boundaries[0,:], axis =1), np.all(theta < boundaries[1,:], axis = 1)) #(N,)
			
			#TODO: you might want to implement in the boundaries, the fact that m1>=m2!
		
		res = np.zeros((theta.shape[0],)) -10000000
		
		det = self.get_metric_determinant(theta[ids_ok,:])
		
		det = np.log(np.abs(det))*0.5 #log(sqrt(|det|))
		
		res[ids_ok] = det
		
		return res
	
	def get_WF_grads(self,theta, approx):
		"""
		Computes the gradient of the WF with a given approximant, both in TD or FD, depending on the option set at initialization.
		An approximant can be either 'mlgw' either any lal approximant. In the case of mlgw an analytical expression for the gradient is used. In the case of lal, the gradients are computed with finite difference methods.
		
		Parameters
		----------
		
		theta: 'np.ndarray' (N,D)
			parameters of the BBHs. The dimensionality depends on the spin format set for the metric
	
		approx: 'string'
			Which approximant to use. It can be 'mlgw' or any lal approx

		Returns
		-------
		
		h : 'np.ndarray' (N,D, 4)
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
		epsilon_list = [1e-6, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6]  #FIXME: find a way to tune it
			#PAY ATTENTION TO THE GRADIENTS! THEY REALLY MAKE A DIFFERENCE IN ACCURACY
			#IT IS HARD TO COMPUTE GRADIENTS IN THE LOW MASS REGION! WHY??
		
		def get_order(M):
			#TODO:fix the thresholds
			if M>50.: order = 1
			if M<=50. and M>15.: order = 2
			if M<=15. and M>10.: order = 4
			if M<=10. and M>1.: order = 6
			if M<=1.: order = 8
			return order
		
		for theta_ in theta:
			df = self.delta_f#*10.
			order = get_order(theta_[0])
			grad_theta_list = []
			if order==1: WF =  get_WF(theta_, df) #only for forward euler
			
			for i in range(theta.shape[1]):

				deltax = np.zeros(theta.shape[1])
				deltax[i] = 1.
				epsilon = epsilon_list[i]
				
				#computing WFs
				if order>1:
					WF_p = get_WF(theta_ + epsilon * deltax, df)
					WF_m = get_WF(theta_ - epsilon * deltax, df)
	
				if order>2:
					WF_2p = get_WF(theta_ + 2.*epsilon * deltax, df)
					WF_2m = get_WF(theta_ - 2.*epsilon * deltax, df)
					
				if order>4:
					WF_3p = get_WF(theta_ + 3.*epsilon * deltax, df)
					WF_3m = get_WF(theta_ - 3.*epsilon * deltax, df)				

				if order>6:
					WF_4p = get_WF(theta_ + 4.*epsilon * deltax, df)
					WF_4m = get_WF(theta_ - 4.*epsilon * deltax, df)

				
				#######
				# computing gradients with finite difference method
				# see: https://en.wikipedia.org/wiki/Finite_difference_coefficient

					#forward euler: faster but less accurate
				if order ==1:
					grad_i = (WF_p - WF )/(epsilon) #(N,D) 
					#second order method: slower but more accurate
				if order==2:
					grad_i = (WF_p - WF_m )/(2*epsilon) #(N,D)
					#fourth order method
				elif order==4:
					grad_i = (-WF_2p/4. + 2*WF_p - 2.*WF_m + WF_2m/4. )/(3*epsilon) #(N,D)
					#sixth order method
				elif order==6:
					grad_i = (WF_3p -9.*WF_2p + 45.*WF_p \
						- 45.*WF_m + 9.*WF_2m -WF_3m)/(60.*epsilon) #(N,D)
					#eight order method
				elif order==8:
					grad_i = (- WF_4p/56. + (4./21.)*WF_3p - WF_2p + 4.*WF_p \
						- 4. *WF_m + WF_2m - (4./21.)* WF_3m + WF_4m/56.)/(5*epsilon) #(N,D)
				else:
					raise ValueError("Wrong value for the derivative order")

				#TODO: maybe this should be replaced by scipy.signal.decimate?				
				#grad_i = np.interp(self.f_grid, np.linspace(0, df*len(grad_i),len(grad_i)), grad_i)
				
				grad_theta_list.append(grad_i)
			grad_h_list.append(grad_theta_list)

		grad_h = np.stack(grad_h_list, axis = -1).T #(N,K,D)
		
		return grad_h
	
	def get_WF_lal(self, theta, approx = None, df = None):
		"""
		Returns the lal WF with a given approximant with parameters theta. The WFs are in FD and are evaluated on the grid set by lal
		
		Parameters
		----------
		
		theta: 'np.ndarray' (D, )
			Parameters of the BBHs. The dimensionality depends on self.spin_format
	
		approx: 'string'
			Which approximant to use. It can be FD lal approx
			If None, the default approximant will be used
		
		df: 'float'
			The frequency step used for the WF generation.
			If None, the default, given by the PSD will be used

		Returns
		-------
		
		hp, hc : 'np.ndarray' (N,D)
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
			raise RuntimeError("Approximant {} is TD".format(approx)) #must be FD approximant
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi = self.s_handler.get_BBH_components(theta, self.spin_format)
			#FIXME: WTF is this grid??
		hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(m1*lalsim.lal.MSUN_SI,
                        m2*lalsim.lal.MSUN_SI,
                        s1x, s1y, s1z,
                        s2x, s2y, s2z,
                        1e6*lalsim.lal.PC_SI,
                        iota, phi, 0., 0.,0., #inclination, phi0, longAscNodes, eccentricity, meanPerAno
                        df,
                        self.f_min, #flow
                        self.f_high, #fhigh
                        self.f_min, #fref
                        lal.CreateDict(),
                        lal_approx)
		#f_grid = np.linspace(0., self.f_high, len(hptilde.data.data))
		
		return hptilde, hctilde
	
	def get_WF(self, theta, approx = None):
		"""
		Computes the WF with a given approximant with parameters theta. The WFs are in FD and are evaluated on the grid on which the PSD is evauated (self.f_grid)
		An any lal FD approximant.
		
		Parameters
		----------
		
		theta: 'np.ndarray' (N,D)
			Parameters of the BBHs. The dimensionality depends on self.spin_format
	
		approx: 'string'
			Which approximant to use. It can be FD lal approx
			If None, the default approximant will be used

		Returns
		-------
		
		h : 'np.ndarray' (N,D)
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

			WF_list.append(hp)
		
		h = np.stack(WF_list, axis =0)
		
		if squeeze: h = h[0,:]
		
		return h
	
	
	def get_metric(self, theta, overlap = False):
		"""
		Returns the metric
		
		Parameters
		----------
		
		theta: 'np.ndarray' (N,D)
			Parameters of the BBHs. The dimensionality depends on self.spin_format
		
		overlap: 'bool'
			Whether to compute the metric based on the local expansion of the overlap rather than of the match
			In this context the match is the overlap maximized over time

		Returns
		-------
		
		metric : 'np.ndarray' (N,4,4)/(N,2,2)
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
	
	def WF_match(self, h1, h2, overlap = False):
		"""
		Computes the match line by line between two WFs. The WFs shall be evaluated on the custom grid 
		No checks will be done on the input
		
		Parameters
		----------
		
		h1: 'np.ndarray' (N,D)
			First WF

		h2: 'np.ndarray' (N,D)
			Second WF
		
		overlap: 'bool'
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed
		
		Returns
		-------
		
		match : 'np.ndarray' (N,)
			Array containing the match of the given WFs
			
		"""
		sigmasq = lambda WF: np.sum(np.multiply(np.conj(WF), WF), axis = 1)
		
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
	def match(self, theta1, theta2, overlap = False):
		"""
		Computes the match line by line between elements in theta1 and elements in theta2
		
		Parameters
		----------
		
		theta1: 'np.ndarray' (N,D)
			Parameters of the first BBHs. The dimensionality depends on self.spin_format

		theta2: 'np.ndarray' (N,D)
			Parameters of the second BBHs. The dimensionality depends on self.spin_format
		
		overlap: 'bool'
			Whether to compute the overlap between WFs (rather than the match)
			In this case, the time maximization is not performed
		
		Returns
		-------
		
		match : 'np.ndarray' (N,)
			Array containing the match of the given WFs
			
		"""
		#FIXME: this function agrees with pycbc but disagree with sbank!! WFT?? Time alignment is the way!
		theta1 = np.array(theta1)
		theta2 = np.array(theta2)
		assert theta1.shape == theta2.shape, "Shapes of the two inputs should be the same!"
		squeeze = False
		if theta1.ndim ==1:
			theta1 = theta1[None,:]
			theta2 = theta2[None,:]
			squeeze = True
		
		h1 = self.get_WF(theta1, self.approx)
		h2 = self.get_WF(theta2, self.approx)

		match = self.WF_match(h1, h2, overlap)

		if squeeze: match = match[0]

		return match
	
	def metric_match(self, theta1, theta2, metric = None):
		"""
		Computes the metric match line by line between elements in theta1 and elements in theta2.
		The match is approximated by the metric:
			match(theta1, theta2) = 1 + M_ij(theta1) (theta1 - theta2)_i (theta1 - theta2)_j
		
		Parameters
		----------
		
		theta1: 'np.ndarray' (N,D)
			Parameters of the first BBHs. The dimensionality depends on self.spin_format

		theta2: 'np.ndarray' (N,D)
			Parameters of the second BBHs. The dimensionality depends on self.spin_format
		
		metric: 'np.ndarray' (D,D)
			metric to use for the match (if None, it will be computed from scratch)

		Returns
		-------
		
		match : 'np.ndarray' (N,)
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
		
		theta1 = theta1[:,:self.s_handler.D(self.spin_format)]
		theta2 = theta2[:,:self.s_handler.D(self.spin_format)]

		delta_theta = theta2 - theta1  #(N,D)
		
		if metric is None:
			metric = self.get_metric(theta1)
			match = 1 + np.einsum('ij, ijk, ik -> i', delta_theta, metric, delta_theta) #(N,)
		else:
			match = 1 + np.einsum('ij, jk, ik -> i', delta_theta, metric, delta_theta) #(N,)
		
		return match
		
	def metric_accuracy(self, theta1, theta2):
		"""
		Computes the metric accuracy at the given points of the parameter space.
		The accuracy is the absolute value of the difference between true and metric mismatch (as computed by match and metric_match)
		
		Parameters
		----------
		
		theta1: 'np.ndarray' (N,D)
			Parameters of the first BBHs. The dimensionality depends on self.spin_format

		theta2: 'np.ndarray' (N,D)
			Parameters of the second BBHs. The dimensionality depends on self.spin_format

		Returns
		-------
		
		accuracy : 'np.ndarray' (N,)
			Array the accuracy of the metric approximation
		
		"""
		accuracy = self.metric_match(theta1, theta2) - self.match(theta1, theta2) #(N,)
		return np.abs(accuracy)
		
	
	def get_points_atdist(self, N_points, theta, dist):
		"""
		Given a central theta point, it computes N_points random point with a metric match equals to distance.
		
		Parameters
		----------
		
		N_points: int
			Number of random points to be drawn
		
		theta: 'np.ndarray' (D,)
			Parameters of the center. The dimensionality depends on self.spin_format

		dist: float
			Distance from the center theta

		Returns
		-------
		
		points : 'np.ndarray' (N,D)
			Points with distance dist from the center
		"""
		metric = -self.get_metric(theta)
		
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
		return rect.volume() * np.sqrt(np.abs(np.linalg.det(metric))) / np.power(avg_dist, metric.shape[0])
		#return rect.volume() * np.sqrt(np.abs(np.prod(np.diag(metric)))) / np.power(avg_dist, metric.shape[0])
	
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
		tiles_list = [(start_rect, start_metric, N_temp+1)] 
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

			if verbose: pbar.update(np.round((V_covered/V_tot)*100. - pbar.last_print_n,2))

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
		in_array = np.load(filename)
		for in_ in in_array:
			self.append( (scipy.spatial.Rectangle(in_[0,:], in_[1,:]), in_[2:,:]) )
		
		return


####################################################################################################################

class GW_bank():
	"""
	This implements a bank. A bank is a collection of templates (saved as the numpy array bank.templates) that holds masses and spins of the templates.
	A bank is generated from a tiling object (created internally): each template belongs to a tile. This can be useful to speed up the match computation
	It can be generated with a MCMC and can be saved in txt or in the std ligo xml file.
	The class implements some methods for computing the fitting factor with a number of injections.
	"""
	def __init__(self, filename = None, spin_format = 'nonspinning'):
		"""
		Initialize the bank
		
		Parameters
		----------
		
		filename: 'str'
			Optional filename to load the bank from (if None, the bank will be initialized empty)
		
		spin_format: 'str'
			How to handle the spin variables.
			See class spin_handler for more details
		
		"""
		#TODO: start dealing with spins properly from here...
		self.s_handler = spin_handler()
		self.spin_format = spin_format
		self.templates = None #empty bank
		
		self.D = self.s_handler.D(self.spin_format) #handy shortening
		
		if isinstance(filename, str):
			self.load(filename)

		return
	
	def avg_dist(self, avg_match):
		"""
		Average distance between templates in the proper volume, s.t. they have a given average match between each other
		
		Parameters
		----------
			MM: 'float'
				Minimum match
		Returns
		-------
			avg_dist: 'float'
				Average distance between templates
		"""
		#return 2*np.sqrt((1-avg_match))
		return 2*np.sqrt((1-avg_match)/self.D) #Owen

	def load(self, filename):
		"""
		Load a bunch of templates from file. They are added to existing templates (if any)
		
		Parameters
		----------
			
		filename: str
			Filename to load the bank from
		"""
		if filename.endswith('.npy'):
			templates_to_add = np.load(filename)
		if filename.endswith('.txt') or filename.endswith('.dat'):
			templates_to_add = np.loadtxt(filename)
		if filename.endswith('.xml') or filename.endswith('.xml.gz'):
		
				#for some reason this content handles is needed... Why??
			class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
				pass
			lsctables.use_in(LIGOLWContentHandler)

			xmldoc = lw_utils.load_filename(filename, verbose = True, contenthandler = LIGOLWContentHandler)
			sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)
		
			BBH_components = []
			
			for row in sngl_inspiral_table:
				BBH_components.append([row.mass1, row.mass2, row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z, row.alpha3, row.alpha5])
			
			BBH_components = np.array(BBH_components) #(N,10)
			
				#making the templates suitable for the bank
			templates_to_add = self.s_handler.get_theta(BBH_components, self.spin_format) #(N,D)
			
		self.add_templates(templates_to_add)

		return

	def _save_xml(self, filename):
		"""
		Save the bank to an xml file suitable for LVK applications
		
		Parameters
		----------
			
		filename: str
			Filename to save the bank at
		"""
			#getting the masses and spins of the rows
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi = self.s_handler.get_BBH_components(self.templates, self.spin_format)
		
			#preparing the doc
			#See: https://git.ligo.org/RatesAndPopulations/lvc-rates-and-pop/-/blob/master/bin/lvc_rates_injections#L168
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		signl_inspiral_table = lsctables.New(lsctables.SnglInspiralTable)

			#register a process_table about what code made the file
		process = ligolw_process.register_to_xmldoc(
			xmldoc,
			program="mbank",
			paramdict={},#process_params, #what should I enter here?
			comment="A bank of (possibly precessing) BBH, generated using a metric approach")
		
			#here we add the rows one by one
		for i in range(m1.shape[0]):
				#defining the row
			row =  DefaultSnglInspiralTable() #This is a dirty trick for a std initialization (works)
			#row = lsctables.New(lsctables.SnglInspiralTable).RowType()
			
				#setting bank parameters
			row.mass1, row.mass2 = m1[i], m2[i]
			row.spin1x, row.spin1y, row.spin1z = s1x[i], s1y[i], s1z[i]
			row.spin2x, row.spin2y, row.spin2z = s2x[i], s2y[i], s2z[i]
			row.alpha3 = iota[i]
			row.alpha5 = phi[i] #are you sure it's alpha5? See here: https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/waveforms.py#L845
			
				#shall I need to set other things by hand? E.g. taus...
			row.mtotal = row.mass1 + row.mass2
			row.eta = row.mass1 * row.mass2 / row.mtotal**2
			row.mchirp = ((row.mass1 * row.mass2)**3/row.mtotal)**0.2
			row.chi = (row.mass1 *row.spin1z + row.mass2 *row.spin2z) / row.mtotal #is this the actual chi?
				#this is chi from https://git.ligo.org/lscsoft/gstlal/-/blob/master/gstlal-inspiral/python/_spawaveform.c#L896
			#row.chi = (np.sqrt(row.spin1x**2+row.spin1y**2+row.spin1z**2)*m1 + np.sqrt(row.spin2x**2+row.spin2y**2+row.spin2z**2)*m2)/row.mtotal
			
			row.f_final = 2500 /(row.mtotal) #dirty trick (again) this is a very very very crude estimation of maximum frequency (in Hz)
			row.ifo = 'L1' #default ifo #FIXME:make it better
			
				#Setting additional parameters
			row.process_id = process.process_id #This must be an int
			row.event_id = i
			row.Gamma0 = float(i) #apparently Gamma0 is the template id in gstlal (for some very obscure reason)
			
			#for k, v in std_extra_params.items():
			#	setattr(row, k, v)
			signl_inspiral_table.append(row)
			
		#xmldoc.appendChild(ligolw.LIGO_LW()).appendChild(signl_inspiral_table)
		ligolw_process.set_process_end_time(process)
		xmldoc.childNodes[-1].appendChild(signl_inspiral_table)
		lw_utils.write_filename(xmldoc, filename, gz=filename.endswith('.xml.gz'), verbose=False)
		xmldoc.unlink()
		
		return
		
	def save_bank(self, filename, full_space = False):
		"""
		Save the bank to file
		
		Parameters
		----------
			
		filename: str
			Filename to save the bank at
		
		full_space: False
			Whether to save the masses and the full spins of each template (does not apply if the input file is an xml).
		"""
		if self.templates is None:
			raise ValueError('Bank is empty: cannot save an empty bank!')

		if not full_space: to_save = self.templates
		else: to_save = np.column_stack([ self.s_handler.get_BBH_components(self.templates, self.spin_format) ])

		if filename.endswith('.npy'):
			templates_to_add = np.save(filename, to_save)
		elif filename.endswith('.txt') or filename.endswith('.dat'):
			templates_to_add = np.savetxt(filename, to_save)
		elif filename.endswith('.xml') or filename.endswith('.xml.gz'):
			self._save_xml(filename)
		else:
			raise RuntimeError("Type of file not understood. The file can only end with 'npy', 'txt', 'data, 'xml', 'xml.gx'.")
		
		return

	def save_injs(self, filename, GPS_start, GPS_end, time_step, approx, luminosity_distance = 100, f_min = 20., multiple_template = False ):
		"""
		Save the bank to a standard injection file.
		
		Parameters
		----------
			
		filename: str
			Filename to save the injections at
		
		GPS_start: int
			Start GPS time for the injections

		GPS_end: int
			End GPS time for the injections

		time_step: float
			Time step between consecutive injections
			Warning: no checks are made for overlapping injections
		
		approx: str
			Lal approximant to use to perform injections
		
		luminosity_distance: float/tuple
			Luminosity distance in Mpc for the all the injections
			If a tuple, it has the meaning max luminosity/min luminosity

		f_min: float
			Starting frequency (in Hz) for the injection
		
		multiple_template: bool
			Whether to allow the same template to appear more than once in the injection set
		
		"""
		#For inspiration see: https://git.ligo.org/RatesAndPopulations/lvc-rates-and-pop/-/blob/master/bin/injection_utils.py#L675
		#https://git.ligo.org/RatesAndPopulations/lvc-rates-and-pop/-/blob/master/bin/lvc_rates_injections#L168
			#defining detectors
		detectors = {
				'h' : lal.CachedDetectors[lal.LHO_4K_DETECTOR],
				'l' : lal.CachedDetectors[lal.LLO_4K_DETECTOR],
				'v' : lal.CachedDetectors[lal.VIRGO_DETECTOR]
			}
			
			#if luminosity_distance is an int, the max/min value shall change a bit, otherwise the dag won't run
		if isinstance(luminosity_distance, float): luminosity_distance = (luminosity_distance, luminosity_distance*1.001)
		else: assert isinstance(luminosity_distance, tuple), "Wrong format for luminosity distance. Must be a float or a tuple of float, not {}".format(type(luminosity_distance))
		
			#getting the masses and spins of the rows
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi = self.s_handler.get_BBH_components(self.templates, self.spin_format)
		
			#opening a document
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		sim_inspiral_table = lsctables.New(lsctables.SimInspiralTable)

		process = ligolw_process.register_to_xmldoc(
			xmldoc,
			program="mbank",
			paramdict={},
			comment="Injections matching a bank")

			#loops on rows (i.e. injections)
		for i, t_inj in enumerate(np.arange(GPS_start, GPS_end, time_step)):

			if multiple_template: id_template = np.random.randint(0, len(m1))
			else: id_template = i
			
			if i>= len(m1) and not multiple_template: break

				#boring initialization stuff
			row =  DefaultSimInspiralTable()
			row.process_id = process.process_id
			row.simulation_id = i #int?
			row.waveform = approx
			row.f_lower = f_min
			row.taper = "TAPER_START"
			row.bandpass = 0

				#setting interesting row paramters
			row.inclination = iota[id_template]
			row.coa_phase = phi[id_template]
			row.polarization = np.random.uniform(0.0, 2.0 * np.pi)
			row.longitude = np.random.uniform(0.0, 2.0 * np.pi)
			row.latitude = np.arcsin(np.random.uniform(-1.0, 1.0))
			row.distance = np.random.uniform(*luminosity_distance)

				#setting masses/spins and other related quantities
			row.mass1, row.mass2 = m1[id_template], m2[id_template]
			row.spin1x, row.spin1y, row.spin1z = s1x[id_template], s1y[id_template], s1z[id_template]
			row.spin2x, row.spin2y, row.spin2z = s2x[id_template], s2y[id_template], s2z[id_template]
			
			row.mtotal = row.mass1 + row.mass2
			row.eta = row.mass1 * row.mass2 / row.mtotal**2
			row.mchirp = ((row.mass1 * row.mass2)**3/row.mtotal)**0.2
			row.chi = (row.mass1 *row.spin1z + row.mass2 *row.spin2z) / row.mtotal #is this the actual chi?
			#row.chi = (np.sqrt(row.spin1x**2+row.spin1y**2+row.spin1z**2)*m1 + np.sqrt(row.spin2x**2+row.spin2y**2+row.spin2z**2)*m2)/row.mtotal
			
			row.f_final = 2500 /(row.mtotal)
			
				#dealing with geocentric time for the injections
			tj = lal.LIGOTimeGPS(float(t_inj)) #do you want to jitter it?
			row.geocent_end_time = tj.gpsSeconds
			row.geocent_end_time_ns = tj.gpsNanoSeconds
			row.end_time_gmst = lal.GreenwichMeanSiderealTime(tj)

				# calculate and set detector-specific columns
			for site, det in detectors.items():
				tend = tj + lal.TimeDelayFromEarthCenter(det.location, row.longitude, row.latitude, tj)
				setattr(row, site + "_end_time", tend.gpsSeconds)
				setattr(row, site + "_end_time_ns", tend.gpsNanoSeconds)
				setattr(row, "eff_dist_" + site, row.distance) #this is not d_eff, but there is not such a thing as d_eff for precessing searches...
			row.alpha4 = 20 #This is H1 SNR?? Required?
			row.alpha5 = 20 #This is L1 SNR?? Required?

				#appending row
			sim_inspiral_table.append(row)

		ligolw_process.set_process_end_time(process)
		xmldoc.childNodes[-1].appendChild(sim_inspiral_table)
		lw_utils.write_filename(xmldoc, filename, gz=filename.endswith('.xml.gz'), verbose=False)
		xmldoc.unlink()

		return 

	def add_templates(self, new_templates):
		"""
		Adds a bunch of templates to the bank.
		They will be saved in the format set by spin_format
		
		Parameters
		----------
		
		new_templates: np.ndarray
			New templates to add.
			They need to be stored in an array of shape (N,D) or (D,), where D is the dimensionality of the bank
		"""
		new_templates = np.array(new_templates)
		
		if new_templates.ndim == 1:
			new_templates = new_templates[None,:]
		
		assert new_templates.ndim == 2, "The new templates are provided with a wrong shape!"
		
		assert self.D == new_templates.shape[1], "The templates to add have the wrong dimensionality. Must be {}, but given {}".format(self.D, new_templates.shape[1])
		
		new_templates = self.s_handler.switch_BBH(new_templates, self.spin_format)
		
		if self.templates is None:
			self.templates = new_templates
		else:
			self.templates = np.concatenate([self.templates, new_templates], axis = 0) #(N,4)
		
		return
	def get_templates(self, metric_obj, avg_dist, lower_boxes, upper_boxes, lower_boxes_i, upper_boxes_i, p_disc, verbose = True):
		"""
		Place the templates on a sub-tile.
		"""
		#TODO:create a proper docstring for that
		center_1d = (upper_boxes+lower_boxes)/2.
		new_templates = [] #to store the newly added templates
		metric_values = [] #to store the metric evaluation on the centroids
		
			#setting verbosity
		if verbose: it = tqdm(range(len(upper_boxes_i)), desc = 'Generating a bank - loops on sub-tiles', leave = False)
		else: it = range(len(upper_boxes_i))
		
		for j in it:
			center = np.concatenate([[center_1d], (lower_boxes_i[j]+upper_boxes_i[j])/2.])
			boundaries_ij = np.column_stack([np.concatenate([[lower_boxes], lower_boxes_i[j]]),
				np.concatenate([[upper_boxes], upper_boxes_i[j]]) ]).T

				#TODO: edit create_mesh, so that it produces meaningful results :)
			#metric =  metric_obj.get_metric(center) #metric
			#new_templates = create_mesh(avg_dist, (scipy.spatial.Rectangle(boundaries_ij[0,:],boundaries_ij[1,:]) , metric))
			#self.add_templates(new_templates)
			#continue
				
				######computing N_templates
				#volume computation
			volume = np.abs(np.prod(boundaries_ij[1,:] - boundaries_ij[0,:]))
			volume_factor = np.sqrt(np.abs(metric_obj.get_metric_determinant(center))) #volume factor (from the metric)
			volume = volume*volume_factor
			
			metric_values.append(np.concatenate([center, [volume_factor], [int(volume / np.power(avg_dist, self.D))] ]))
			#DEBUUUUUUUUUUUUUUUUUUUG
			#pay attention to that!!
			new_templates.append(np.random.uniform(*boundaries_ij, (3, self.D))) #DEBUG
			continue
			
				#extracting the templates and placing
			if p_disc:
				#FIXME: how the hell shall I tune this parameter??
				radius = 0.5*avg_dist/np.power(volume_factor, 1/self.D)
				new_templates.append(poisson_disc.Bridson_sampling((boundaries_ij[1,:]-boundaries_ij[0,:]), radius = radius) + boundaries_ij[0,:])
			else:
				lambda_N_templates = volume / np.power(avg_dist, self.D)
				N_templates = np.random.poisson(lambda_N_templates)
				#print(center, N_templates)
				#print(avg_dist)
				new_templates.append(np.random.uniform(*boundaries_ij, (N_templates, self.D)))
		
			if new_templates[-1].ndim ==1: new_templates[-1]= new_templates[-1][None,:]
			
			if False:
				corners = get_cube_corners(boundaries_ij[:,:2])
				plt.scatter(corners[:,0], corners[:,1], c = 'r', s = 15)
				plt.plot(*corners[[0,1],:].T, c = 'k', lw = 2)
				plt.plot(*corners[[2,3],:].T, c = 'k', lw = 2)
				plt.plot(*corners[[1,3],:].T, c = 'k', lw = 2)
				plt.plot(*corners[[0,2],:].T, c = 'k', lw = 2)
				plt.scatter(*center[:2], c = 'r', marker = 'x')
				plt.scatter(*new_templates[-1][:,:2].T, c = 'r', s = 1)

		new_templates = np.concatenate(new_templates, axis =0)
		metric_values = np.stack(metric_values, axis =0)
		
		return new_templates, metric_values
	
	#@do_profile(follow=[])
	def generate_bank_hypercubes(self, metric_obj, avg_match, boundaries, grid_list = None, p_disc = False, plot_debug = None, use_ray = False):
		"""
		Generates a bank using an hypercube tesselation.
		Works only if spinf format includes M and q
		
		Parameters
		----------

		metric_obj: WF_metric
			A WF_metric object to compute the match with

		avg_match: float
			Average match between templates
		
		boundaries: 'np.ndarray' (2,D)
			An array with the boundaries for the model. Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
			
		grid_list: 'list'
			A list holding the number of split to be performed along each dimension (included M and q)
			If the entry is a list, they are understood as the threshold values for the division (handy for a non equally spaced grid)
			##SHITTY NAME: FIND A BETTER ONE

		p_disc: 'bool'
			Whether to use Poisson disc sampling methods for sampling on the hypercubes. It ensures a higher uniformity in the points drawn
			(Needs module poisson_disc)

		plot_debug: str
			String with the folder to save the plots at.
			If None, no plots will be produced. If 'show', the results will be shown.

		use_ray: bool
			Whether to use ray to parallelize
		
		Returns
		-------
		
		metric_values: 'np.ndarray' (N_centers,D+1)
			The evaluation of the metric in the centers of the hypercubes.
			Each row has the format [center (D,), sqrt(|M|)]
		"""
		avg_dist = self.avg_dist(avg_match) #desired average distance between templates
		
		if use_ray: ray.init()
		
		if self.spin_format.startswith('m1m2_'):
			raise RuntimeError("The current placing method does not support m1m2 format for the masses")
		
		if grid_list is None:
			raise ValueError("Parameter grid_list not provided")
		assert len(grid_list) == self.D, "Wrong number of grid sizes. Expected {}; given {}".format(self.D, len(grid_list))
		
			#creating a proper grid list
		grid_list_ = []
		for i in range(self.D):
			if isinstance(grid_list[i], int):
				if i ==0:
					#g_list = np.geomspace(boundaries[0,i], boundaries[1,i], grid_list[i]+1) #not based on physics but apparently better
						#placing m_tot or M_chirp according the scaling relation: mc**(-8/3)*l ~ const.
						#(but maybe it is better to use geomspace)
					#g_list = plawspace(boundaries[0,i], boundaries[1,i], -8./3., grid_list[i]+1) #power law spacing
					g_list = np.linspace(boundaries[0,i], boundaries[1,i], grid_list[i]+1) #linear spacing
				else: g_list = np.linspace(boundaries[0,i], boundaries[1,i], grid_list[i]+1)
				grid_list_.append( g_list )
			elif isinstance(grid_list[i], np.ndarray) or isinstance(grid_list[i], list):
				grid_list_.append( np.array(grid_list[i]))
			else:
				raise ValueError("Grid type not understood. Expected list/np.array or int, got {}".format(type(grid_list[i])))
		grid_list = grid_list_
		
		lower_boxes, upper_boxes = get_boundary_box([grid_list[0]])
		lower_boxes, upper_boxes = lower_boxes[:,0], upper_boxes[:,0]

			#loop on the first dimension (i.e. M)
		new_templates, metric_values = [], []
		
		for i in tqdm(range(len(upper_boxes)), desc = 'Generating a bank - loops on M/Mc tiles'):
			#grid_list_i = [np.sort([g[0], *np.random.uniform(g[0],g[-1],len(g)-2), g[-1]]) for g in grid_list[1:]]
			
				#this is ugly but look alright! Probably, it is better to do a linspace
			grid_list_i =  [np.array([g[0], 
					*((g[-1]-g[0])*np.sort(np.linspace(0,1,len(g))*np.random.uniform(1-.5/(len(g)),1+.5/(len(g)),len(g)))[1:-1]+g[0]),
					g[-1]]) for g in grid_list[1:]]
			grid_list_i = grid_list[1:] #DEBUUUUUUUUUUUUUUUG
			
			lower_boxes_i, upper_boxes_i = get_boundary_box(grid_list_i)
			
			#getting the templates in this M/mc strip
			if not use_ray:
				new_templates_temp, metric_values_temp = self.get_templates(metric_obj, avg_dist, lower_boxes[i], upper_boxes[i], lower_boxes_i, upper_boxes_i, p_disc, True)
				new_templates.append(new_templates_temp)
				metric_values.append(metric_values_temp)
			if use_ray:
				#only the first one will have its progress bar: this gives a rough idea of the time that everything takes
				#this new templates is a hack: it rather keeps a tuple of array [new_templates, metric_values]
				new_templates.append(get_templates_ray.remote(self, metric_obj, avg_dist, lower_boxes[i], upper_boxes[i], lower_boxes_i, upper_boxes_i, p_disc, verbose = (i == 0)))
		
		if use_ray:
			#gathering the ray results (that are executed in parallel)
			#TODO: do a proper progress bar here
			
			new_templates = ray.get(new_templates)
			ray.shutdown()
			print("All ray jobs are done")
			metric_values = [nt[1] for nt in new_templates]
			new_templates = [nt[0] for nt in new_templates]

		new_templates = np.concatenate(new_templates, axis =0) #concatenate a list of templates
		metric_values = np.concatenate(metric_values, axis =0) #concatenate a list of metric values
		self.add_templates(new_templates)

		if isinstance(plot_debug, str):
			import pandas as pd
			if self.templates.shape[0] >50000:
				ids_ = np.random.choice(self.templates.shape[0], 50000, replace = False)
			else:
				ids_ = range(self.templates.shape[0])
			seaborn.pairplot(pd.DataFrame(data=self.templates[ids_,:], columns = self.s_handler.labels(self.spin_format, True)),  markers = 'o', plot_kws={"s": 3})
			if plot_debug != 'show': plt.savefig(plot_debug+"pairplot.png")
			if plot_debug == 'show': plt.show()
			plt.show() #DEBUG
			plt.close('all')

		return metric_values
	
	def place_templates(self, t_obj, avg_match, placing_method = 'p_disc', verbose = True):
		"""
		Given a tiling, it places the templates and adds them to the bank
		
		Parameters
		----------

		t_obj: 'tiling_handler'
			A tiling handler with a non-empty tiling
		
		avg_match: 'float'
			Average match for the bank: it controls the distance between templates
		
		placing_method: 'str'
			The placing method to set templates in each tile
			It can be:	'p_disc' 	-> Poisson disc sampling
						'uniform'	-> Uniform drawing in each hyper-rectangle
						'geometric'	-> Geometric placement
		verobse: 'bool'
			Print output?
		
		"""
		assert placing_method in ['p_disc', 'uniform', 'geometric', 'iterative'], ValueError("Wrong placing method selected")
		assert self.D == t_obj[0][0].maxes.shape[0], ValueError("The tiling doesn't match the chosen spin format (space dimensionality mismatch)")
		
		avg_dist = self.avg_dist(avg_match) #desired average distance between templates
		new_templates = []
		
		if verbose: it = tqdm(t_obj, desc = 'Placing the templates within a tile')
		else: it = t_obj
		
		for t in it:
			boundaries_ij = np.stack([t[0].mins, t[0].maxes], axis =0) #boundaries of the tile
			volume_factor = np.sqrt(np.abs(np.linalg.det(t[1]))) #* t[0].volume()
			
			if placing_method == 'p_disc':
					#radius controls the relative distance between templates (not the radius of the circle!)
				radius = 0.5* avg_dist/np.power(volume_factor, 1/self.D)
				new_templates_ = poisson_disc.Bridson_sampling((boundaries_ij[1,:]-boundaries_ij[0,:]), radius = radius) + boundaries_ij[0,:]
				
				#print(new_templates_.shape[0], len(create_mesh(avg_dist, t )))
				
			elif placing_method == 'uniform':
				lambda_N_templates = t_obj.N_templates(*t, avg_dist) #volume / np.power(avg_dist, self.D) #Computed by tiling_handler
				N_templates = int(lambda_N_templates)+1
				new_templates_ = np.random.uniform(*boundaries_ij, (N_templates, self.D))
				
				#print(avg_match, avg_dist, N_templates)
				#print(new_templates_)
				
			elif placing_method == 'geometric':
				new_templates_ = create_mesh(avg_dist, t )
				#new_templates_ = create_mesh_new(avg_dist, t )
			
			elif placing_method == 'iterative':
				temp_t_obj = tiling_handler()
				temp_metric_fun = lambda theta: t[1]
				temp_t_obj.create_tiling((t[0].mins, t[0].maxes), 1, temp_metric_fun, avg_dist, verbose = False, worker_id = None)
				
				print(len(temp_t_obj), t_obj.N_templates(*t, avg_dist))
				
				new_templates_ = temp_t_obj.get_centers()
		
			new_templates.append(new_templates_)

		new_templates = np.concatenate(new_templates, axis =0)
		self.add_templates(new_templates)
		return 			

	def create_grid_tiling(self):
		"This would be equivalent to the old version of the code"
		return
	
	def _generate_tiling(self, metric_obj, coarse_boundaries, avg_dist, N_temp, use_ray = False ):
		"""
		Creates a tiling of the space, starting from a coarse tile.
		
		Parameters
		----------
		
		metric_obj: 'WF_metric'
			A WF_metric object to compute the match with

		N_temp: 'int'
			Maximum number of templates that each tile may contain

		avg_dist: 'float'
			Average distance (in the metric space) between templates
		
		coarse_boundaries: 'list'
			A list of boundaries for a coarse tiling. Each box will have its own independent hierarchical tiling
			Each element of the list must be (max, min), where max, min are array with the upper and lower point of the hypercube
		
		use_ray: 'bool'
			Whether to use ray to parallelize
		
		Returns
		-------
					
		tiling: 'tiling_handler' 
			A list of tiles ready to be used for the bank generation
		"""
		t_obj = tiling_handler() #empty tiling handler
		temp_t_obj = tiling_handler()
		t_ray_list = []
		
		for i, b in enumerate(coarse_boundaries):
			if use_ray:
				t_ray_list.append( temp_t_obj.create_tiling_ray.remote(temp_t_obj, b,
							N_temp, metric_obj.get_metric, avg_dist, verbose = True , worker_id = i) )
			else:
				t_obj += temp_t_obj.create_tiling(b, N_temp, metric_obj.get_metric, avg_dist, verbose = True, worker_id = None) #adding the newly computed templates to the tiling object
			
		if use_ray:
			t_ray_list = ray.get(t_ray_list)
			ray.shutdown()
			print("All ray jobs are done")
			t_obj = tiling_handler()
			for t in t_ray_list:
				t_obj += t
		
		return t_obj
			
	def generate_bank(self, metric_obj, avg_match, boundaries, grid_list, N_temp = 200, placing_method = 'p_disc', plot_folder = None, use_ray = False):
		"""
		Generates a bank using a hierarchical hypercube tesselation. 
		It works only if spin format includes M and q
		
		Parameters
		----------

		metric_obj: WF_metric
			A WF_metric object to compute the match with

		avg_match: float
			Average match between templates
		
		boundaries: 'np.ndarray' (2,D)
			An array with the boundaries for the model. Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
		
		grid_list: 'list'
			A list of ints, each representing the number of coarse division of the space.
			If use ray option is set, the subtiling of each coarse division will run in parallel
		
		N_temp: 'int'
			Maximum number of templates that each tile may contain

		placing_method: 'str'
			The placing method to set templates in each tile
			It can be:	'p_disc' 	-> Poisson disc sampling
						'uniform'	-> Uniform drawing in each hyper-rectangle
						'geometric'	-> Geometric placement

		plot_folder: str
			String with the folder to save the plots at.
			If None, no plots will be produced. If 'show', the results will be shown.

		use_ray: bool
			Whether to use ray to parallelize
		
		Returns
		-------
		
		tiling: 'tiling_handler' 
			A list of tiles used for the bank generation		
		"""
		#TODO: add an option to avoid the hierarchical tiling??
			##
			#Initialization
		avg_dist = self.avg_dist(avg_match) #desired average distance in the metric space between templates
		
		if use_ray: ray.init()
		
		if self.spin_format.startswith('m1m2_'):
			raise RuntimeError("The current placing method does not support m1m2 format for the masses")
		
		assert len(grid_list) == self.D, "Wrong number of grid sizes. Expected {}; given {}".format(self.D, len(grid_list))
		
			###
			#creating a proper grid list for a coarse boundary creation
		grid_list_ = []
		for i in range(self.D):
			if i ==0:
					#placing m_tot or M_chirp according the scaling relation: mc**(-8/3)*l ~ const.
					#(but maybe it is better to use geomspace)
				g_list = plawspace(boundaries[0,i], boundaries[1,i], -8./3., grid_list[i]+1) #power law spacing
				#g_list = np.geomspace(boundaries[0,i], boundaries[1,i], grid_list[i]+1) #not based on physics
				#g_list = np.linspace(boundaries[0,i], boundaries[1,i], grid_list[i]+1) #linear spacing
			else:
				g_list = np.linspace(boundaries[0,i], boundaries[1,i], grid_list[i]+1)
			grid_list_.append( g_list )
		grid_list = grid_list_
		
		lower_boxes, upper_boxes = get_boundary_box(grid_list)
		coarse_boundaries = [(low, up) for low, up in zip(lower_boxes, upper_boxes) ]
		
			###
			#creating the tiling
		t_obj = self._generate_tiling(metric_obj, coarse_boundaries, avg_dist, N_temp, use_ray = use_ray )	
		
			##
			#placing the templates
		self.place_templates(t_obj, avg_match, placing_method = placing_method, verbose = True)
		
			##
			#plot debug
		if isinstance(plot_folder, str):
			plot_tiles_templates(t_obj, self.templates, self.spin_format, plot_folder, show = True)
		
		return t_obj
				
	def enforce_boundaries(self, boundaries):
		"""
		Remove from the bank the templates that do not lie within the boundaries
		
		Parameters
		----------

		boundaries: 'np.ndarray' (2,4)/(2,2)
			An array with the boundaries for the model. Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
		"""	
		if self.templates is None: return

		ids_ok = np.logical_and(np.all(self.templates > boundaries[0,:], axis =1), np.all(self.templates < boundaries[1,:], axis = 1)) #(N,)
		if len(ids_ok) == 0:
			self.templates = None
			warnings.warn("No template fits into the boundaries")
		elif len(ids_ok) < self.templates.shape[0]:
			self.templates = self.templates[ids_ok,:]
		else:
			pass
			#print("The bank already fits into the boundaries")

		return

####################
####################
# OLD FF STUFF
####################
####################
	
	def _compute_cross_match(self, metric_obj, injections_set, N_templates, N_neigh):
		"""
		For each template in injections_set, it computes the optimal match with an element in the bank
		
		Parameters
		----------

		metric_obj: WF_metric
			A WF_metric object to compute the match with

		injections_set: np.ndarray (N,2)/(N,4)
			An array containing the set of injections

		N_templates: int
			Number of radom templates to be chosen from the bank. If None, all the element in the bank will be chosen

		N_neigh: int
			Number of neigbours to consider for the match calculation
		
		Returns
		-------
		
		FF : 'np.ndarray' (M,)
			Best match for each injection
		
		ids_templates: 'np.ndarray' (M,)
			Indices of the templates used for the computation
		"""
		match_function = metric_obj.match #function to compute the match
		metric_obj.set_spin_format(self.spin_format)
		
		injections_set = injections_set[:,:self.D]
		bank = self.templates[:,:self.D]

			#in the case where the inj_set and the bank are the same, we don't want to include the first neighbour: otherwise the FF =1 always
		if np.all(bank==injections_set):
			id_start = 1
		else:
			id_start = 0
		
		if isinstance(N_templates, int):
			ids_templates = np.random.choice(bank.shape[0], size = (N_templates,), replace = False)
			bank = bank[ids_templates,:]
			if id_start ==1:
				injections_set = injections_set[ids_templates,:]
		else:
			ids_templates = np.array(range(bank.shape[0]))
		
		bank_lookup_table = scipy.spatial.cKDTree(bank)
		d, ids = bank_lookup_table.query(injections_set, k = N_neigh)
		
		FF_list = [] #this list keeps the match for each neighbours
		for j in tqdm(range(id_start, N_neigh), desc="Computing fitting factor - evaluating neighbours"): #loop on neighbour order
			temp_match = match_function(injections_set, bank[ids[:,j],:], True)
			FF_list.append(temp_match)

		FF = np.array(FF_list).T
		FF = np.max(FF, axis =1)
		
		return FF, ids_templates
	
	def _compute_cross_match_opt(self, metric_obj, injections_set, N_templates, N_neigh):
		"""
		For each template in injections_set, it computes the optimal match with an element in the bank.
		It is optimized so that the injections are computed only once.
		
		Parameters
		----------

		metric_obj: WF_metric
			A WF_metric object to compute the match with

		injections_set: np.ndarray (N,2)/(N,4)
			An array containing the set of injections

		N_templates: int
			Number of radom templates to be chosen from the bank. If None, all the element in the bank will be chosen

		N_neigh: int
			Number of neigbours to consider for the match calculation
		
		Returns
		-------
		
		FF : 'np.ndarray' (M,)
			Best match for each injection
		
		ids_templates: 'np.ndarray' (M,)
			Indices of the templates used for the computation
		"""
		match_function = metric_obj.match #function to compute the match
		metric_obj.set_spin_format(self.spin_format)
		
		injections_set = injections_set[:,:self.D]
		bank = self.templates[:,:self.D]

			#in the case where the inj_set and the bank are the same, we don't want to include the first neighbour: otherwise the FF =1 always
		if np.all(bank==injections_set):
			id_start = 1
		else:
			id_start = 0
		
			#trimming the bank to a given number of templates N_templates
		if isinstance(N_templates, int):
			ids_templates = np.random.choice(bank.shape[0], size = (N_templates,), replace = False)
			bank = bank[ids_templates,:]
			if id_start ==1:
				injections_set = injections_set[ids_templates,:]
		else:
			ids_templates = np.array(range(bank.shape[0]))
		
			#precomputing the injections
			#injections_set -> holds the parameters for the injs
			#whiteWFs_injections_set -> holds the whithened WFs
			#norm_injWFs -> holds the normalization constant
		whiteWFs_injections_set = metric_obj.get_WF(injections_set, metric_obj.approx)
		whiteWFs_injections_set = whiteWFs_injections_set/np.sqrt(metric_obj.PSD)
		norm_injWFs = np.sum(np.multiply(np.conj(whiteWFs_injections_set), whiteWFs_injections_set),axis =1).real #(N,)
		print("Generated {} injections".format(len(norm_injWFs)))

		mchirp_bank = self.s_handler.get_mchirp(bank, self.spin_format)
		mchirp_injs = self.s_handler.get_mchirp(injections_set, self.spin_format)

		FF_list = [] #this list keeps the match for each neighbours
		
		bank_lookup_table = scipy.spatial.cKDTree(mchirp_bank[:,None])
		d, ids = bank_lookup_table.query(mchirp_injs[:,None], k = N_neigh)
		
		for i in tqdm(range(bank.shape[0]), desc = 'Looping on the templates'):
			template = bank[i,None,:]
			
				#computing the injections falling close to the template
			#ids_template = np.where(np.abs(mchirp_bank[i] - mchirp_injs)<1)[0] #FIXME: this is garbage, do it better
			ids_template = np.where(ids == i)[0]
			
			if len(ids_template)<=0: continue

			temp_match = np.zeros(mchirp_injs.shape)

			whiteWFs_bank = metric_obj.get_WF(template, metric_obj.approx)/np.sqrt(metric_obj.PSD)
			norm_bank  = np.sum(np.multiply(np.conj(whiteWFs_bank), whiteWFs_bank),axis =1).real #(N,) 
			
			overlap = np.matmul(np.conj(whiteWFs_injections_set[ids_template,:]), whiteWFs_bank.T) #(N,) #np.einsum('ij,j->i')
			#match_ = np.squeeze(overlap).real/np.sqrt(norm_injWFs[ids_template]*np.squeeze(norm_bank)) #this is WITHOUT phi maximization
			match_ = np.abs(np.squeeze(overlap))/np.sqrt(norm_injWFs[ids_template]*np.squeeze(norm_bank)) #this is WITH phi maximization

			temp_match[ids_template] = match_
			
			FF_list.append(temp_match)

		FF = np.array(FF_list).T
		FF = np.max(FF, axis =1)
		
		return FF, ids_templates


	def compute_effectualness(self, metric_obj, N_templates = None, N_neigh = 5):
		"""
		It computes the effectualness of the bank. It is defined as:
			MM_i = min_{j != i} <h_i|h_j>
		where h_i is a generic template in the bank	
		
		Parameters
		----------

		metric_obj: WF_metric
			A WF_metric object to compute the match with

		N_templates: int
			Number of radom templates to be chosen from the bank. If None, all the element in the bank will be chosen

		N_neigh: int
			Number of neigbours to consider for the match calculation
		
		Returns
		-------
		
		MM : 'np.ndarray' (N_templates,)
			Best minimum match for each template
		
		ids_templates: 'np.ndarray' (M,)
			Indices of the templates used for the computation.
			This is only returned if N_templates is not None
		"""
		MM, ids_templates = self._compute_cross_match(metric_obj, self.templates, N_templates, N_neigh)

		if N_templates is not None:
			return MM, ids_templates
		else:
			return MM

	def compute_fitting_factor(self, metric_obj, N_inj, boundaries, N_templates = None, N_neigh = 5):
		"""
		It computes the fitting factor of the bank by drawing random WFs within the boundaries.

		Parameters
		----------
		
		metric_obj: WF_metric
			A WF_metric object to compute the match with
		
		N_inj: int
			Number of random injections to compute the match

		boundaries: 'np.ndarray' (2,4)/(2,2)
			An optional array with the boundaries for the random extraction
			Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]

		N_templates: int
			Number of radom templates to be chosen from the bank. If None, all the element in the bank will be chosen

		N_neigh: 'int'
			Number of neighbours in the Euclidean space to compute the match with
		
		Returns
		-------
		
		FF : 'np.ndarray' (N,)
			Best match for each injection
		
		theta_inj: 'np.ndarray' (N,4)
			Points in the space at which match is computed
		
		ids_templates: 'np.ndarray' (M,)
			Indices of the templates used for the computation.
			This is only returned if N_templates is not None
		"""
		boundaries = np.array(boundaries) #(2,2/(2,4)
		injs = np.random.uniform(*boundaries, (N_inj, boundaries.shape[1]))
			#switching s.t. m1>m2
		ids = np.where(injs[:,0]<injs[:,1])[0]
		injs[ids,0], injs[ids,1] = injs[ids,1], injs[ids,0]
		
		FF, ids_templates = self._compute_cross_match_opt(metric_obj, injs, N_templates, N_neigh)		
	
		if N_templates is not None:
			return  FF, injs, ids_templates
		return FF, injs, ids_templates
	
	def save_obj(self):
		return

###########################
###########################
# OLD GARBAGE
###########################
###########################


	def generate_bank_MCMC(self, metric_obj, N_templates, boundaries, fitting_factor = None, n_walkers = 100, thin_factor = None, load_chain = None, save_chain = None, verbose = True):
		#FIXME: shall I also compute the minimum match as a stopping condition? Right now, I am not using it because the FF is needed to correct the bank at later stages... In future, things can change!
		#FIXME: the sampling does not work well in the equal mass region (perhaps the probability is low?)
		#FIXME: qualitatively, there are some differences between sbank and mbank. Understand why: sampler or PDF?
		"""
		Fills the bank with a MCMC (uses emcee package).
		
		### This function is not up to date!!
		
		Parameters
		----------

		metric_obj: WF_metric
			A WF_metric objec to compute the PDF to distribute the templates
		
		N_templates: int
			Number of new templates to add.
			If fitting_factor is specified, this option has no effect and an indefinite number of new templates will be added
		
		boundaries: 'np.ndarray' (2,4)/(2,2)
			An optional array with the boundaries for the model. If a point is asked below the limit, -10000000 is returned
			Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
			If None, no boundaries are implemented
			
		fitting_factor: (float, int)
			A tuple of (max_FF, N_injs)
			If not None, the fitting factor of the bank will be computed with N_injs.
			Whenever the bank fitting factor is below max_FF, the bank generation will end
		
		n_walkers: int
			Number of independent walkers during the chain
		
		thin_factor: int
			How many MC steps to discard before selecting one.
			This value is computed authomatically based on the autocorrelation: it is recommended not to set it by hand 
		
		load_chain: str
			Path to a file where the position of each walker is stored, togheter with integrated aucorellation tau.
			The file must keep a np.array of dimensions (n_walkers, 2/4). The first line of the file is intended to be the autocorrelation time for each variable. If it is not provided, a standard value of 4 (meaning a thin step of 2) is assumed.
			If not None, the sampler will start from there and the burn-in phase will not be required.
		
		save_chain: str
			If not None, it saves the path in which to save the status of the sampler.
			The file saved is ready to be given to load chain
		
		verbose: 'bool'
			whether to print to screen the output
		"""
		ndim = np.array([2,4])[[self.nonspinning, not self.nonspinning]][0]
		sampler = emcee.EnsembleSampler(n_walkers, ndim, metric_obj.log_pdf, args=[boundaries], vectorize = True)
		n_burnin = 0
		
		if load_chain is not None:
			#this will output an estimate of tau and a starting chain. The actual sampling will start straight away
			burnin = False
			loaded_chain = np.loadtxt(load_chain)
			if loaded_chain.shape[0] == n_walkers:
				start = loaded_chain
				tau = 4 + np.zeros((ndim,))
			else:
				tau, start = loaded_chain[0,:], loaded_chain[1:,:]
			print('tau', tau)
			assert start.shape == (n_walkers, ndim), "Wrong shape for the starting chain. Unable to continue"
		else:
			burnin = True
			start = np.random.uniform(*boundaries, (n_walkers, ndim))
		
			###########
			#This part has two purposes:
			#	- Give a first estimation for tau parameters (required to decide the size of burn-in steps and the thin step)
			#	- Do a burn in phase (discard some samples to achieve stationariety)
			###########
			
		if burnin:
			tau_list = []
			step = 30

			def dummy_generator(): #dummy generator for having an infinite loop: only required for tqdm (is there a better way for doing this)
				while True: yield
			
			if verbose:
				it_obj = tqdm(dummy_generator(), desc='Burn-in/calibration phase')
			else:
				it_obj = dummy_generator()

			for _ in it_obj:
				n_burnin += step
				#if verbose: print("\tIteration ", n_burnin//step)
				state = sampler.run_mcmc(start, nsteps = step, progress = False, tune = False)
				start = state.coords #very important! The chain will start from here
						
				tau = sampler.get_autocorr_time(tol = 0)
				tau_list.append(tau)
				
				if len(tau_list)>1 and np.all(np.abs(tau_list[-2]-tau_list[-1]) < 0.001*tau_list[-1]):
					tau = tau_list[-1]
					break
			#print(tau)		
			#plt.plot(tau_list)
			#plt.show()
			if verbose: print("")
			###########
			#doing the actual sampling
		#FIXME: this eventually should have a check on the FF
			
			#first estimate of thin
		if thin_factor is None:
			thin = max(int(0.5 * np.min(tau)),1)
		else:
			thin = thin_factor
		
		print('Thin, burn-in: ', thin, int(2 * np.max(tau)))

		n_steps = int((N_templates*thin)/n_walkers) - int(n_burnin) #steps left to do...
		print("Steps done/ steps new", n_burnin, n_steps)
		
			#remember to start from a proper position!!!!! You idiot!!		
		if n_steps > 0:
			state = sampler.run_mcmc(start, n_steps, progress = verbose, tune = False)
	
		tau = sampler.get_autocorr_time(tol = 0)
		
			#setting burn-in steps (if not set yet...)
		burnin_steps = 0
		if burnin:
			burnin_steps = int(2 * np.max(tau))

		#The lines commented below look useless
			#FIXME: understand whether you want to change the thin factor... it is likely it is underestimated during the burn-in phase
		if thin_factor is None:
			thin = max(int(0.5 * np.min(tau)),1)
			print('##updated thin## Thin, burn-in: ', thin, burnin_steps)

		chain = sampler.get_chain(discard = burnin_steps, thin = thin, flat=True)[-N_templates:,:]
		
		if save_chain is not None:
			chain_to_save = state.coords #(n_walkers, 4)/(n_walkers, 2)
			to_save = np.concatenate([tau[None,:], chain_to_save], axis = 0)
			np.savetxt(save_chain, to_save)

			#adding chain to the bank
		self.add_templates(chain)
				
		return

	def generate_bank_hull_sampling(self, metric_obj, avg_match, seed_bank, boundaries, plot_debug = None):
		"""
		Generates a bank using the Voronoi tesselation method. It enforces boundaries by hull sampling
		
		Parameters
		----------

		metric_obj: WF_metric
			A WF_metric object to compute the match with

		avg_match: float
			Average match between templates

		seed_bank: 'np.ndarray'/'string'/tuple
			A starting seed bank for the tesselation algorithm. It can be generated with MCMC and has to have the dimensions of the space
			If a string is given, the seed bank is load from that file with np.loadtxt
			If a tuple is given, the first is intended to be a string, the second is intended to be a the number of templates to be loaded
		
		boundaries: 'np.ndarray' (2,4)/(2,2)
			An array with the boundaries for the model. Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
		
		plot_debug: str
			String with the folder to save the plots at.
			If None, no plots will be produced. If 'show', the results will be shown.
		"""
		PLOT_DEBUG = True
			#loading seed_bank, if it's the case
		if isinstance(seed_bank, str):
			seed_bank = np.loadtxt(seed_bank)
		if isinstance(seed_bank, tuple):
			seed_bank = np.loadtxt(seed_bank[0], max_rows = seed_bank[1])
			
			#making sure that seed bank and boundaries are of the right size
		seed_bank = seed_bank[:,:self.D]
		boundaries = boundaries[:,:self.D]

			#making sure that seed_bank is within boundaries
		ids_ok = np.logical_and(np.all(seed_bank > boundaries[0,:], axis =1), np.all(seed_bank < boundaries[1,:], axis = 1)) #(N,)
		seed_bank = seed_bank[ids_ok,:]
		seed_bank = self.s_handler.switch_BBH(seed_bank, self.spin_format)

			#padding seed_bank, to avoid infinities
		pad_points = get_pad_points(5, boundaries)
		seed_bank = np.concatenate([seed_bank, pad_points], axis = 0)

		if PLOT_DEBUG:
			N_pad = pad_points.shape[0]
			plt.figure()#PLOT_DEBUG
			plt.title('Seed bank') #PLOT_DEBUG
			plt.scatter(seed_bank[:-N_pad,0], seed_bank[:-N_pad,1], c = 'b', marker = 'o', s = 5)#PLOT_DEBUG
			plt.scatter(pad_points[:,0], pad_points[:,1], c = 'r', marker = 'o', s = 12)#PLOT_DEBUG
			plt.xlabel("m1");plt.ylabel("m1")
			plt.figure()#PLOT_DEBUG
			plt.title('tiling')#PLOT_DEBUG

			#scaling to make voronoi better
		scaling_consts = np.abs(boundaries[1,:] - boundaries[0,:])
		scaling_vol = np.prod(scaling_consts)
		seed_bank = seed_bank/scaling_consts
		unscaled_boundaries = np.array(boundaries)
		boundaries = boundaries/scaling_consts
		
			#creating Voronoi tesselation
		vor = scipy.spatial.Voronoi(seed_bank)
		#scipy.spatial.voronoi_plot_2d(vor)
		#plt.show()
		
			#looping on regions
			#each region is trimmed inside the boundaries and a number of templates are evenly spaced in each region
			#FIXME: this can be parallelized
		for id_region in tqdm(range(len(vor.regions)), desc = 'Generating a bank - loops on Voronoi regions'):
		#for id_region in range(len(vor.regions)):
			region = vor.regions[id_region]
			if -1 in region: #region at infinity, discard (hopefully, it should be out of the boundaries)
				continue
			else:
				points = vor.vertices[region]

			try: #why this??
				id_vertex = np.where(vor.point_region == id_region)[0][0]
			except:
				#weird region
				continue

			center = seed_bank[id_vertex]
				#if center is outside, skip the region
			if not np.logical_and(np.all(center > boundaries[0,:]), np.all(center < boundaries[1,:])):
				continue
	
	
				#this line can fail, I might better insert this in a try except statement
			try:
				hull = scipy.spatial.ConvexHull(points) #hull of the current region
			except:
				#import pandas as pd
				#seaborn.pairplot(pd.DataFrame(data=points))
				#plt.show()
				warnings.warn("Region {}: the hull construction failed ".format(id_region))
				continue

				#making the hull inside the boundaries
			hull_generators = sample_from_hull_boundaries(hull, 5000, boundaries, max_iter = 3000)
			
			if hull_generators is not None and len(hull_generators) >10:
				try:
					hull = scipy.spatial.ConvexHull(hull_generators) #this hull now has proper volume
				except:
					#import pandas as pd
					#seaborn.pairplot(pd.DataFrame(data=hull_generators))
					#plt.show()
					warnings.warn("Region {}: the hull construction failed after the sampling: whyyyy?".format(id_region))
					continue
			else:
				warnings.warn("Region {} is badly sampled: too close to the boundary!!".format(id_region))
				continue

				#TODO: think about splitting a volume, if it's too large: might improve convergence at almost no cost
				
				#recomputing the center
			center = np.mean(hull_generators, axis = 0) #good improvement
				#computing the volume of the center
			volume_factor = np.sqrt(np.abs(metric_obj.get_metric_determinant(center*scaling_consts)))
			volume = hull.volume * volume_factor * scaling_vol
			#print(center*scaling_consts, volume_factor,  hull.volume, scaling_vol, volume) #DEBUG
			
			N_templates = int( volume / np.power(1-avg_match, seed_bank.shape[1]) ) #find a good way to set this number
			#print(N_templates) #DEBUG

			if N_templates < 1: continue
			
			if False: #this is good but slow, find a better one
				#FIXME: lloyd's algorithm is slow... Find a good alternative
				kmeans = KMeans(n_clusters=N_templates, random_state=0).fit(hull_generators)
				templates_in_hull = kmeans.cluster_centers_
			else:
				#FIXME: this part can fail easily... you need to make it more robust
				if hull_generators.shape[0] > N_templates:
					templates_in_hull = hull_generators[np.random.choice(hull_generators.shape[0], N_templates, False),:]
					#PLOT_DEBUG = False
				else:
					templates_in_hull = hull_generators
					#PLOT_DEBUG = True
					warnings.warn("Region {}: too few templates were sampled. Result in this region are likely to be biased".format(id_region))
					print(center*scaling_consts, volume_factor,  hull.volume, scaling_vol, volume) #DEBUG
			
			if PLOT_DEBUG:
				if False:
					for simplex in hull.simplices: #PLOT_DEBUG
						plt.plot(hull_generators[simplex, 0]*scaling_consts[0], hull_generators[simplex, 1]*scaling_consts[1], 'k-', zorder = 1)  #PLOT_DEBUG
				if True:
					import pandas as pd
					seaborn.pairplot(pd.DataFrame(data=hull_generators*scaling_consts, columns = self.s_handler.labels(self.spin_format)))
					plt.show()
			
			self.add_templates(templates_in_hull*scaling_consts)
		
		if PLOT_DEBUG:
			plt.scatter(self.templates[:,0], self.templates[:,1], marker = 'o', s = 2, c = 'r', zorder = 0)#PLOT_DEBUG
			plt.xlabel("m1");plt.ylabel("m2")
			plt.figure()#PLOT_DEBUG
			plt.title('final bank')#PLOT_DEBUG
			plt.scatter(self.templates[:,0], self.templates[:,1], marker = 'o', s = 2, c = 'b', zorder = 0)#PLOT_DEBUG
			plt.xlabel("m1");plt.ylabel("m2")
			plt.show() #PLOT_DEBUG
			
		return

	#@do_profile(follow=[])
	def generate_bank_voronoi(self, metric_obj, avg_match, seed_bank, boundaries, grid_list = None, p_disc = False, plot_debug = None):
		"""
		Generates a bank using the Voronoi tesselation method.
		
		Parameters
		----------

		metric_obj: WF_metric
			A WF_metric object to compute the match with

		avg_match: float
			Average match between templates

		seed_bank: 'np.ndarray'/'string'/tuple
			A starting seed bank for the tesselation algorithm. It can be generated with MCMC and has to have the dimensions of the space
			If a string is given, the seed bank is load from that file with np.loadtxt
			If a tuple is given, the first is intended to be a string, the second is intended to be a the number of templates to be loaded
		
		boundaries: 'np.ndarray' (2,D)/(2,2)
			An array with the boundaries for the model. Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
			
		grid_list: 'list'
			A list holding the number of split to be performed along each dimension (excluded the masses)
			If the entry is a list, they are understood as the threshold values for the division (handy for a non equally spaced grid)
			##SHITTY NAME: FIND A BETTER ONE
		
		p_disc: 'bool'
			Whether to use Poisson disc sampling methods for sampling on the hypercubes. It ensures a higher uniformity in the points drawn
			Not implemeted yet :(
			(Needs module poisson_disc)

		plot_debug: str
			String with the folder to save the plots at.
			If None, no plots will be produced. If 'show', the results will be shown.
		"""
		if self.spin_format.startswith('Mq_') or self.spin_format.startswith('mceta_'):
			raise RuntimeError("The current placing method does not support Mq format for the masses")
		
		if qmc:
			sampler = scipy.stats.qmc.LatinHypercube(d=self.D-2)

			#loading seed_bank, if it's the case
		if isinstance(seed_bank, str):
			seed_bank = np.loadtxt(seed_bank)
		if isinstance(seed_bank, tuple):
			seed_bank = np.loadtxt(seed_bank[0], max_rows = seed_bank[1])
		seed_bank = seed_bank[:,:2] #making sure the seed bank is of the right size

			#taking care of the grid on extra dimensions
		upper_boxes, lower_boxes = [], []
		if self.D >2:
			if grid_list is None:
				raise ValueError("Parameter grid_list not provided")
			assert len(grid_list) == self.D-2, "Wrong number of grid sizes"
				#creating a proper grid list
			grid_list_ = []
			for i in range(self.D-2):
				if isinstance(grid_list[i], int):
					grid_list_.append( np.linspace(boundaries[0,2+i], boundaries[1,2+i], grid_list[i]+1) )
				elif isinstance(grid_list[i], np.ndarray) or isinstance(grid_list[i], list):
					grid_list_.append( np.array(grid_list[i]))
				else:
					raise ValueError("Grid type not understood. Expected list/np.array or int, got {}".format(type(grid_list[i])))
			grid_list = grid_list_
			
			lower_boxes, upper_boxes = get_boundary_box(grid_list)
			
		else:
			upper_boxes = [np.array([])]
			lower_boxes = [np.array([])]
			ids_region_bounds = np.array([[]])
			mesh_grid = np.array([[]])
		
		boundaries = boundaries[:,:2] #forgetting about the extra dim boundaries

			#making sure that seed_bank is within boundaries
		ids_ok = np.logical_and(np.all(seed_bank > boundaries[0,:], axis =1), np.all(seed_bank < boundaries[1,:], axis = 1)) #(N,)
		seed_bank = seed_bank[ids_ok,:]
		seed_bank = self.s_handler.switch_BBH(seed_bank, self.spin_format)

			#padding seed_bank, to avoid infinities
		pad_points = get_pad_points_2d(5, boundaries)
		seed_bank = np.concatenate([seed_bank, pad_points], axis = 0)

			#Voronoi tassellation
		vor = scipy.spatial.Voronoi(seed_bank)

		m1, M1, m2, M2 = boundaries[0,0], boundaries[1,0], boundaries[0,1], boundaries[1,1]
		box_2d = np.array([[m1, m1],[m1, m2],[M1, M1], [M1,M2], [M1,m2]])
		box_2d = np.unique(box_2d, axis =0)
		box_hull = scipy.spatial.ConvexHull(box_2d)

		for id_region in tqdm(range(len(vor.regions)), desc = 'Generating a bank - loops on Voronoi regions'):
			region = vor.regions[id_region]
			
			########
			#Creating a tesselation on the mass space
			########
			
			if -1 in region: #region at infinity: the center of the region is added instead of the infinities
				id_vertex = np.where(vor.point_region == id_region)[0][0]
				center = seed_bank[id_vertex] #getting the center
				region = list(set(region)) #same as unique
				region.remove(-1)
				vertices = vor.vertices[region]
				vertices = np.concatenate([vertices, center[None,:]], axis =0)
			else:
				vertices = vor.vertices[region]

			if len(vertices) <1: continue #for some reason there is an empty region

			vertices_in_box = points_in_hull(vertices, box_hull) #whether each vertex is in the box

			if np.all(~vertices_in_box):
				continue #all the vertices are outside

			hull = scipy.spatial.ConvexHull(vertices)

			if not np.all(vertices_in_box):
					#intersecting with boundaries
					#check whether the hull intersects with the boundaries
				b_centers = np.column_stack(vertices[hull.simplices[:,0]]).T #centers of all the vertices of a hull
				b_normals = np.column_stack([vertices[hull.simplices[:,1]]- vertices[hull.simplices[:,0]]]) #direction of all the vertices of a hull
					#computing all the intersections between hull vertices and box hull
				p_intersect = all_line_hull_intersection(b_normals, b_centers, box_hull)
				p_intersect = np.concatenate([box_2d, p_intersect], axis =0) #to add centers
				
					#compute whether the intersections between current hull and box are inside both the hull and the box
				p_to_add = p_intersect[points_in_hull(p_intersect, hull)]
				p_to_add = p_to_add[points_in_hull(p_to_add, box_hull)]

				new_vertices = np.column_stack([*vertices[vertices_in_box,:], *p_to_add]).T
				
				if new_vertices.shape[0] <1: continue #this shouldn't happen but it's better to check :)
				
				vertices = new_vertices
				
				hull = scipy.spatial.ConvexHull(vertices)
			
			if isinstance(plot_debug, str):
				plt.plot(hull.points[hull.vertices,0], hull.points[hull.vertices,1], 'o', c ='g', ms = 5)
				center = np.mean(hull.points[hull.vertices,:], axis = 0)
				plt.plot(center[0], center[1], 'x', c ='r', ms = 7)
				for simplex in hull.simplices: #PLOT_DEBUG
					plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'r-', zorder = 10)

			########
			#Generating the templates
			########
			
			hull_volume_2d = hull.volume
			center_2d = np.mean(hull.points[hull.vertices,:], axis = 0)
				#loops on the boxes in which each mass tile is divided in d dimensions
				#For each box it computes the geometric volume
			extra_dim_samples = []
			N_tot_templates = 0

			if True: #crude square approx
			
				if self.D>2:
					temp_grid_list = [np.sort([g[0], *np.random.uniform(g[0],g[-1],len(g)-2), g[-1]]) for g in grid_list]
					lower_boxes, upper_boxes = get_boundary_box(temp_grid_list)
			
				for i in range(len(upper_boxes)):
					center = np.concatenate([center_2d, (upper_boxes[i]+lower_boxes[i])/2.]) #center of the box
					volume_extra_dim = np.prod(np.abs(upper_boxes[i] - lower_boxes[i]))	#euclidean volume given by the extra dimension
					volume_factor = np.sqrt(np.abs(metric_obj.get_metric_determinant(center))) #volume factor (from the metric)
					volume = hull_volume_2d*volume_extra_dim*volume_factor	#volume of the box
					
						#FIXME: the choice here is absolutely crucial!!
						#These two links might help:
				#https://mathoverflow.net/questions/22592/mean-minimum-distance-for-k-random-points-on-a-n-dimensional-hyper-cube
				#https://arxiv.org/abs/math/0212230
				#You will probably need the inverse of the gamma function: https://mathoverflow.net/questions/12828/inverse-gamma-function

						#eq 2.16 https://arxiv.org/pdf/gr-qc/9511032.pdf
					#lambda_N_templates = volume / np.power(1-avg_match, self.D) #this is very wrong (look better at eq. 2.16)
					lambda_N_templates = volume / np.power(np.sqrt(4*(1-avg_match)/self.D), self.D)
					try:
						N_templates = np.random.poisson(lambda_N_templates)
					except:
						print("Failed N_templates: ", lambda_N_templates, center)
						N_templates = 0
					
					if self.D>2:
						extra_dim_samples.append( np.random.uniform(lower_boxes[i], upper_boxes[i], (N_templates, len(lower_boxes[i]) )) )

					N_tot_templates += N_templates

			if N_tot_templates <1: continue #skipping boxes that shall be empty...
			
			new_templates_2d = sample_from_hull(hull, N_tot_templates) #(N_templates, 2)

			if self.D>2:
				extra_dim_samples = np.concatenate(extra_dim_samples, axis =0)
				new_templates = np.concatenate([new_templates_2d, extra_dim_samples], axis =1)
			else:
				new_templates = new_templates_2d
			
			self.add_templates(new_templates)

		if isinstance(plot_debug, str):
			plt.scatter(*self.templates[:,:2].T, s = 1)
			plt.xlabel(r'$m_1$')
			plt.ylabel(r'$m_2$')
			if plot_debug != 'show': plt.savefig(plot_debug+"voronoi.png")
			import pandas as pd
			seaborn.pairplot(pd.DataFrame(data=self.templates, columns = self.s_handler.labels(self.spin_format)),  markers = 'o', plot_kws={"s": 1})
			if plot_debug != 'show': plt.savefig(plot_debug+"pairplot.png")
			if plot_debug == 'show': plt.show()
			
		return








		
		
		
