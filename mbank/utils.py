"""
mbank.utils
===========

Some utilities for ``mbank``, where you find lots of useful stuff for some boring operations on the templates.
It keeps functions for plotting, injection recovery computation, I/O operations with ligo xml format and other useful operations useful for the package.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
from itertools import combinations, permutations, product
import argparse
import lal.series
import os
import ast
import sys
import scipy
from scipy.stats import binned_statistic_2d

import json
import pickle

from scipy.sparse import lil_array

	#ligo.lw imports for xml files: pip install python-ligo-lw
from ligo.lw import utils as lw_utils
from ligo.lw import ligolw
from ligo.lw import table as lw_table
from ligo.lw import lsctables
from ligo.lw.utils import process as ligolw_process
from ligo.lw.utils import load_filename

from tqdm import tqdm
import ray

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

def dummy_iterator():
	while True:
		yield

####################################################################################################################

class DefaultSnglInspiralTable(lsctables.SnglInspiralTable):
	"""
	This is a copy of ``ligo.lw.lsctables.SnglInspiralTable`` with implemented defaults.
	Implemented as in `sbank.waveform <https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/waveforms.py#L39>`_
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
	This is a copy of ``ligo.lw.lsctables.SimInspiralTable`` with implemented defaults.
	Implemented as in `sbank.waveform <https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/waveforms.py#L39>`_
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

@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
	pass
lsctables.use_in(LIGOLWContentHandler)

####################################################################################################################

def avg_dist(avg_match, D):
	"""
	The average distance between templates such that an injection have an average match with a cubic lattice of templates. This sets the spacing between templates.
		
	Parameters
	----------
		MM: float
			Minimum match
	Returns
	-------
		avg_dist: float
			Average distance between templates
	"""
	#return np.sqrt((1-avg_match)) #like 2202.09380
	return 2*np.sqrt((1-avg_match)/D) #Owen
	#return 2*np.sqrt(1-avg_match)


####################################################################################################################

def get_boundaries_from_ranges(variable_format, M_range, q_range,
	s1_range = (-0.99,0.99), s2_range = (-0.99,0.99), chi_range = (-0.99,0.99), theta_range = (-np.pi, np.pi), phi_range = (-np.pi/2., np.pi/2.),
	iota_range = (0, np.pi), ref_phase_range = (-np.pi, np.pi), e_range = (0., 0.5), meanano_range = (0.1, 1.)):
	"""
	Given the ranges of each quantity, it combines them in a bondary array, suitable for other uses in the package (for instance in the bank generation).
	No checks are performed whether the given ranges make sense.
	
	Parameters
	----------
		variable_format: str
			A string to specify the variable format.
			See :class:`mbank.handlers.variable_handler` for more information
		
		M_range, q_range, s1_range, s2_range, chi_range, theta_range, phi_range, iota_range, ref_phase_range, e_range, meanano_range: tuple
			Ranges for each physical quantity. They will be used whenever required by the ``variable_format``
			If ``mchirpeta`` mass format is set, ``M_range`` and ``q_range`` are interpreted as mchirp and eta respectively.
			If ``logMq`` mass format is set, ``M_range`` is still interpreted as the mass and **not** the log mass.
	
	Returns
	-------
		boundaries: :class:`~numpy:numpy.ndarray`
			shape (2,D) -
			An array with the boundaries.
	"""
	format_info = variable_handler().format_info[variable_format]
	
	######
	#	Setting boundaries: shape (2,D)
	######
	if format_info['spin_format'].find('1x') >-1 and s1_range[0]<0.:
		s1_range = (0, s1_range[1])
	if format_info['spin_format'] == 'fullspins':
		if s1_range[0]< 0: s1_range = (0, s1_range[1])
		if s2_range[0]< 0: s2_range = (0, s2_range[1])
	
	if format_info['mass_format'] == 'logMq':
		M_range = np.log10(np.asarray(M_range))
	if format_info['mass_format'] == 'logm1logm2':
		M_range = np.log10(np.asarray(M_range))
		q_range = np.log10(np.asarray(q_range))
		
		#setting spin boundaries
	if format_info['spin_format'] == 'nonspinning':
		boundaries = np.array([[M_range[0], q_range[0]],[M_range[1], q_range[1]]])
	elif format_info['spin_format'] == 'chi':
		boundaries = np.array([[M_range[0], q_range[0], chi_range[0]],[M_range[1], q_range[1], chi_range[1]]])
	elif format_info['spin_format'] == 's1z':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0]],[M_range[1], q_range[1], s1_range[1]]])
	elif format_info['spin_format'] == 's1z_s2z':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], s2_range[0]],[M_range[1], q_range[1], s1_range[1], s2_range[1]]])
	elif format_info['spin_format'] == 's1xz':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0]],[M_range[1], q_range[1], s1_range[1], theta_range[1]]])
	elif format_info['spin_format'] == 's1xyz':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0], phi_range[0]],[M_range[1], q_range[1], s1_range[1], theta_range[1], phi_range[1]]])
	elif format_info['spin_format'] == 's1xz_s2z':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0], s2_range[0]],[M_range[1], q_range[1], s1_range[1], theta_range[1], s2_range[1]]])
	elif format_info['spin_format'] == 's1xyz_s2z':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0], phi_range[0], s2_range[0]],[M_range[1], q_range[1], s1_range[1], theta_range[1], phi_range[1], s2_range[1]]])
	elif format_info['spin_format'] == 'fullspins':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0], phi_range[0], s2_range[0], theta_range[0], phi_range[0],],[M_range[1], q_range[1], s1_range[1], theta_range[1], phi_range[1], s2_range[1], theta_range[1], phi_range[1]]])
	else:
		raise RuntimeError("Boundaries current not implemented for the required format of spins {}: apologies :(".format(format_info['spin_format']))

	if format_info['e']:
		boundaries = np.concatenate([boundaries, [[e_range[0]], [e_range[1]]]], axis =1)
	if format_info['meanano']:
		boundaries = np.concatenate([boundaries, [[meanano_range[0]], [meanano_range[1]]]], axis =1)
	if format_info['iota']:
		boundaries = np.concatenate([boundaries, [[iota_range[0]], [iota_range[1]]]], axis =1)
	if format_info['phi']:
		boundaries = np.concatenate([boundaries, [[ref_phase_range[0]], [ref_phase_range[1]]]], axis =1)
	
	return boundaries #(2,D)
	

####################################################################################################################
def load_PSD(filename, asd = False, ifo = 'H1', df = None):
	"""
	Loads a PSD from file and returns a grid of frequency and PSD values
	
	Parameters
	----------
		filename: str
			Name of the file to load the PSD from (can be a txt file or an xml file)

		asd: bool
			Whether the file contains an ASD rather than a PSD
		
		ifo: str
			Interferometer which the PSD refers to. Only for loading a PSD from xml
		
		df: float
			Spacing for the grid on which the PSD is evaluated. It controls the resolution of the PSD (and also the speed of metric computation)
			If `None` is given, the grid of the PSD won't be changed
	
	Returns
	-------
		f: :class:`~numpy:numpy.ndarray`
			Frequency grid

		PSD: :class:`~numpy:numpy.ndarray`
			PSD evaluated on the frequency grid
	"""
	
	if filename.endswith('xml') or filename.endswith('xml.gz'):
		PSD_fseries = lal.series.read_psd_xmldoc(
				load_filename(filename, verbose=False,
				contenthandler=lal.series.PSDContentHandler)
			)
		try:
			PSD_fseries = PSD_fseries[ifo]
		except KeyError:
			raise ValueError("The given PSD file doesn't have an entry for the chosen interferometer {}".format(ifo))
		f = np.linspace(PSD_fseries.f0, PSD_fseries.deltaF*PSD_fseries.data.length, PSD_fseries.data.length)
		PSD = PSD_fseries.data.data
	else:
		f, PSD = np.loadtxt(filename)[:,:2].T

	if asd: PSD = np.square(PSD)
	
	if df:
		new_f = np.arange(f[0], f[-1]-f[0], df)
		PSD = np.interp(new_f, f, PSD)
		f = new_f

	return f, PSD	


####################################################################################################################
def ray_compute_injections_match(inj_dict, bank, metric_obj, mchirp_window = 0.1, symphony_match = False, max_jobs = 8, verbose = True):
	"""
	Wrapper to :func:`compute_injections_match` to allow for parallel execution.
	Given an injection dictionary, generated by :func:`compute_injections_metric_match` it computes the actual match (without the metric approximation) between injections and templates. It updates ``inj_dict`` with the new computed results.
	The injections are generic (not necessarly projected on the bank submanifold).
	
	Parameters
	----------
		inj_dict: dict
			A dictionary with the data injection as computed by `compute_injections_metric_match`.
		
		bank: :class:`mbank.bank.cbc_bank`
			A bank object

		metric_obj: cbc_metric
			A cbc_metric object to compute the match with.

		mchirp_window: float
			Window in mchirp where the match between templates and injections is evaluated.
			The window is expressed in terms of :math:`\Delta\mathcal{M}/\mathcal{M}`, where :math:`\Delta\mathcal{M}` is the distance in chirp mass between injection and each template

		symphony_match: bool
			Whether to use the symphony match

		max_jobs: int
			Maximum number of parallel ray jobs to be instantiated
		
		verbose: 'bool'
			Whether to print the output
		
	Returns
	-------
		inj_dict: dict
			The output dictionary with the updated matches
	"""
		###
		# Split injections
	n_injs_per_job = max(25, int(inj_dict['theta_inj'].shape[0]/max_jobs)) 
	
		###
		# Initializing ray and performing the computation
	inj_dict_ray_list = []
	ray.init()
	for id_, i in enumerate(range(0, inj_dict['theta_inj'].shape[0], n_injs_per_job)):
		inj_dict_ray_list.append( _compute_injections_match_ray.remote(i, i+n_injs_per_job, inj_dict,
			bank, metric_obj, mchirp_window, symphony_match, id_, verbose))
	inj_dict_ray_list = ray.get(inj_dict_ray_list)
	ray.shutdown()
	
		###
		# Concatenating the injections
	inj_dict = {}
	for k in inj_dict_ray_list[0].keys():
		if isinstance(inj_dict_ray_list[0][k], np.ndarray):
			inj_dict[k] = np.concatenate([inj_dict_[k] for inj_dict_ in inj_dict_ray_list ])
		elif isinstance(inj_dict_ray_list[0][k], list):
			inj_dict[k] = []
			for inj_dict_ in inj_dict_ray_list:
				inj_dict[k].extend(inj_dict_[k])
		else:
			inj_dict[k] = inj_dict_ray_list[0][k]
	
	return inj_dict


@ray.remote
def _compute_injections_match_ray(start_id, end_id, inj_dict, bank, metric_obj, mchirp_window = 0.1, symphony_match = False, worker_id = 0, verbose = True):
	"""
	Wrapper to :fun:`compute_injections_match` to allow for parallelization with ray.
	"""
	local_dict = {} #ray wants a local dict for some reason...

		#splitting the dictionary
	for k in inj_dict.keys():
		if k == 'sky_loc':
			if np.asarray(inj_dict[k]).ndim ==1:
				local_dict[k] = inj_dict[k]
				continue
	
		if isinstance(inj_dict[k], np.ndarray):
			local_dict[k] = np.copy(inj_dict[k][start_id:end_id])
		elif isinstance(inj_dict[k], list):
			local_dict[k] = inj_dict[k][start_id:end_id].copy()
		else:
			local_dict[k] = inj_dict[k]
	
	return compute_injections_match(local_dict, bank, metric_obj, mchirp_window, symphony_match, worker_id, verbose)

def compute_injections_match(inj_dict, bank, metric_obj, mchirp_window = 0.1, symphony_match = False, worker_id = None, verbose = True):
	"""
	Given an injection dictionary, generated by :func:`compute_injections_metric_match` it computes the actual match (not the metric approximation) between injections and templates of a given bank. It updates the ``match`` and ``id_match`` entries of ``inj_dict`` with the newly computed match values.
	The injections are generic (not necessarly projected on the bank submanifold).
	
	For each injection, it identifies the templates within the mchirp window. At a second stage, it loops on the templates and computes the injection match, whenever required.
	
	Parameters
	----------
		inj_dict: dict
			A dictionary with the data injection as computed by :func:`compute_injections_metric_match`.
		
		bank: :class:`mbank.bank.cbc_bank`
			A bank object

		metric_obj: cbc_metric
			A cbc_metric object to compute the match with.
		
		mchirp_window: float
			Window in mchirp where the match between templates and injections is evaluated.
			The window is expressed in terms of :math:`\Delta\mathcal{M}/\mathcal{M}`, where :math:`\Delta\mathcal{M}` is the distance in chirp mass between injection and each template
		
		symphony_match: bool
			Whether to use the symphony match
		
		worker_id: 'int'
			Id of the ray worker being used. If None, it is assumed that ray is not called
		
		verbose: 'bool'
			Whether to print the output
		
	Returns
	-------
		inj_dict: dict
			The output dictionary with the updated matches
	"""
	inj_dict = dict(inj_dict)
	sky_locs = inj_dict['sky_loc']
	
	old_format = metric_obj.variable_format
	if metric_obj.variable_format != 'BBH_components':
		metric_obj.set_variable_format('BBH_components')

		#allocating memory for the match
	inj_dict['id_match'] = np.zeros((inj_dict['theta_inj'].shape[0],), int)
	inj_dict['match'] = np.zeros((inj_dict['theta_inj'].shape[0],))
	inj_dict['symphony_SNR'] = symphony_match
	inj_dict['mchirp_window'] = mchirp_window

		#putting injections and templates with the format 'm1m2_fullspins_emeanano_iotaphi'
		# The format is basically the full 12 dimensional space, with spins in spherical coordinates
	injs = inj_dict['theta_inj']
	templates = bank.BBH_components
	
	chirp_injs = metric_obj.var_handler.get_mchirp(injs[:,[0,1]], 'm1m2_nonspinning')
	chirp_templates = metric_obj.var_handler.get_mchirp(templates[:,[0,1]], 'm1m2_nonspinning')

		#Dealing with antenna patterns
	if sky_locs is not None:
		sky_locs = np.asarray(sky_locs)
		assert sky_locs.shape[-1]==3, "Each row of sky location must be composed of three angles! {} given".format(sky_locs.shape[-1])
		F_p, F_c = get_antenna_patterns(*sky_locs.T)
		F_p, F_c = np.atleast_1d(F_p), np.atleast_1d(F_c)
		if chirp_injs.shape[0]>F_p.shape[0]:
			F_p = np.repeat(F_p[0], chirp_injs.shape[0])
			F_c = np.repeat(F_c[0], chirp_injs.shape[0])
	else:
		if symphony_match: raise ValueError("The sky localization must be given if the symphony match is used!")

	#print("Sky locs: ",sky_locs, F_p, F_c)

		######
		# Creating the mapping table
	inj_template_mapping = lil_array((injs.shape[0], templates.shape[0]), dtype = int)

	for i in tqdm(range(injs.shape[0]), desc = 'Generating an injection-template mapping table', disable = not verbose):
	
		mcw = mchirp_window
		relative_diff = np.abs(chirp_templates-chirp_injs[i])/chirp_injs[i]
	
			#Here we adaptively tune the mchirp window to make sure that even the high mass injections will have at least 1500 templates to match
		while True:
			ids_, = np.where(relative_diff<mcw)
			mcw += 0.05
			if len(ids_) >= min(templates.shape[0], 1500): break
		
		#print(i, len(ids_), mcw, chirp_injs[i])

		inj_template_mapping[[i], ids_] = 1

		######
		# Computing the match between injections and templates
		
		#Generating the injected signal
		
	injs_WFs = metric_obj.get_WF(injs, plus_cross = True)
	if sky_locs is not None:
		s_WFs = (injs_WFs[0].T*F_p + injs_WFs[1].T*F_c).T
	else:
		s_WFs, _ = injs_WFs

	if worker_id is None: desc = 'Computing the {} match: loop on the templates'.format('symphony' if symphony_match else 'std')
	else: desc = 'Worker {} - Computing the {} match: loop on the templates'.format(worker_id, 'symphony' if symphony_match else 'std')
	it_ = tqdm(range(templates.shape[0]), desc = desc, leave = True, mininterval = 5, maxinterval = 20) if verbose else range(templates.shape[0])

	for i in it_:
		ids_injs, _ = inj_template_mapping[:, [i]].nonzero()
		if len(ids_injs)>0:
			template_WF = metric_obj.get_WF(templates[i], plus_cross = symphony_match)
			
			if symphony_match:
				current_match_injs = metric_obj.WF_symphony_match(s_WFs[ids_injs], *template_WF)
			else:
				current_match_injs = metric_obj.WF_match(s_WFs[ids_injs], template_WF)
			#print(current_match_injs)
			
			matches_injs_old = inj_dict['match'][ids_injs]
			
			#print(inj_dict['match'][ids_injs])
			
			ids_to_change = np.where(inj_dict['match'][ids_injs]<current_match_injs)[0]
			if len(ids_to_change)>0:
				inj_dict['match'][ids_injs[ids_to_change]] = current_match_injs[ids_to_change]
				inj_dict['id_match'][ids_injs[ids_to_change]] = i

			#print(ids_to_change)
			#print(inj_dict['match'][ids_injs])
			
	metric_obj.set_variable_format(old_format)
	
	return inj_dict
		


####################################################################################################################

def initialize_inj_stat_dict(injs, sky_locs = None):
	"""
	Creates an injection stat dictionary for the given injections and (possibly) sky localization.
	The injection stat dictionary can be passed to :func:`compute_injections_metric_match` and/or :func:`compute_injections_match` which fill the relevant entries, with the results of fitting factors calculations.

	An injection stat dictionary is dictionary with entries:
		
	- ``theta_inj``: the parameters of the injections
	- ``id_tile``: index of the tile the injections belongs to in a given tiling (filled by :func:`compute_injections_metric_match`)
	- ``mchirp_window``: the window in relative chirp mass inside which the templates are considered for full match
	- ``match``: match of the closest template (filled by :func:`compute_injections_match`)
	- ``id_match``: index of the closest template (filled by :func:`compute_injections_match`)
	- ``metric_match``: metric match of the closest template (filled by :func:`compute_injections_metric_match`)
	- ``id_metric_match``: metric index of the closest template (filled by :func:`compute_injections_metric_match`)
	- ``sky_loc``: the sky location for each injection

	Parameters
	----------
		injs: :class:`~numpy:numpy.ndarray`
			shape: (N,12) -
			A set of injections. They must be in the full format of the :meth:`mbank.metric.cbc_metric.get_BBH_components`.
			Each row keeps the following entries: m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi
	
		sky_locs: :class:`~numpy:numpy.ndarray`
			shape (N,3) -
			Sky localization for the injections. Each row corresponds to longitude, latitude and polarization angle for the injections. Sky localization will be used to compute the antenna pattern: see :func:`get_random_sky_loc` and :func:`get_antenna_patterns` for more information.
			If ``None``, only the plus polarization will be used for injection recovery: this can only happens if the standard match is used (``symphony = False`` in :func:`compute_injections_match`).
	
	Returns
	-------
		inj_dict: dict
			An injection stat dictionary
	"""
		#Making a copy of the original injs
	injs = np.array(injs)
	if injs.shape[1] != 12:
		raise ValueError("Wrong input size for the injections: each injection must have 12 dimensions but {} given".format(injs.shape[1]))
	if isinstance(sky_locs, (list, np.ndarray)):
		sky_locs = np.array(sky_locs)
	
		#storing injections in the full format (so that they can be generic)
	inj_dict = {'theta_inj': injs, 'sky_loc': sky_locs,
				'mchirp_window': None, 'symphony_SNR': None,
				'match': None, 'id_match':  None,
				'id_tile': None, 'metric_match': None, 'id_metric_match': None
				}
	return inj_dict


#@do_profile(follow=[])
def compute_injections_metric_match(inj_dict, bank, tiling, verbose = True):
	"""
	Computes the match of the injections in ``inj_dict`` with the bank by using the metric approximation.
	It makes use of a brute force approach where each injection is checked against each template of the bank. The metric used is the one of the tile each injection belongs to.
	
	It updates the entries ``id_tile``, ``metric_match`` and ``id_metric_match`` of the injection stat dictionary.
	
	Parameters
	----------
		inj_dict: dict
			A valid injection stat dictionary with injections. Can be easily filled with :func:`initialize_inj_stat_dict`
		
		bank: cbc_bank
			A :class:`~mbank.bank:mbank.bank.cbc_bank` object
		
		tiling: tiling_handler
			A tiling object to compute the metric match between templates and injections
		
		verbose: bool
			Whether to print the output
		
	Returns
	-------
		inj_dict: dict
			The updated injection stat dictionary with the result of metric fitting factor computation
	
	"""
	inj_dict = dict(inj_dict)
	try:
		injs = inj_dict['theta_inj']
	except KeyError:
		raise ValueError("The injection stat dict must have a 'theta_inj' entry")
	
		#Initializing the relevant entries
	inj_dict['id_tile'] = np.zeros((injs.shape[0],), int)
	inj_dict['metric_match'] = np.zeros((injs.shape[0],))
	inj_dict['id_metric_match'] = np.empty((injs.shape[0],), dtype = int)
	
		#casating the injections to the metric type
	injs = bank.var_handler.get_theta(injs, bank.variable_format)
	
	template_dist = np.allclose(bank.templates, injs) if (bank.templates.shape == injs.shape) else False
	N_argpartiton = 20000
	id_diff_ok = np.arange(bank.templates.shape[0])
	
		#loops on the injections
	if verbose: inj_iter = tqdm(range(injs.shape[0]), desc = 'Evaluating metric match for injections', leave = True)
	else: inj_iter = range(injs.shape[0])
	
	if tiling.flow: metric_list = tiling.get_metric(injs, flow = True, kdtree = True)
	#TODO: optimize injections with flow (now it's too slow in the metric computation approach)
	
	for i in inj_iter:

		inj_dict['id_tile'][i] = tiling.get_tile(injs[i])[0]

		diff = bank.templates - injs[i] #(N_templates, D)
		
			#these are the indices being checked
		if N_argpartiton < bank.templates.shape[0]:
			id_diff_ok = np.argpartition(np.linalg.norm(diff, axis=1), N_argpartiton)[:N_argpartiton]

			#using the flow to compute the true tiling metric (if available)
		if tiling.flow: metric = metric_list[i]
		else: metric = tiling[inj_dict['id_tile'][i]].metric
		
		match_i = 1 - np.sum(np.multiply(diff[id_diff_ok], np.matmul(diff[id_diff_ok], metric)), axis = -1)
		match_i = np.exp(-(1-match_i))
		
		inj_dict['id_metric_match'][i] = np.argmax(match_i)
		inj_dict['metric_match'][i] = match_i[inj_dict['id_metric_match'][i]]

	return inj_dict


####################################################################################################################
def get_ellipse(metric, center, dist, **kwargs):
	"""
	Given a two dimensional metric and a center, it returns the `matplotlib.Patch` that represent the points at constant distance `dist` according to the metric.
	It accepts as an additional parameter, anything that can be given to `matplotlib.patches.Ellipse`.
	
	Parameters
	----------
		metric: :class:`~numpy:numpy.ndarray`
			shape: (2,2) - 
			A two dimensional metric
		
		center: :class:`~numpy:numpy.ndarray`
			shape: (2,) - 
			The center for the ellipse
		
		dist: float
			The distance between points
	
	Returns
	-------
		ellipse: matplotlib.patches.Ellipse
			The ellips of constant match
	"""
	
		#setting some defaults...
	if 'fill' not in kwargs.keys():	kwargs['fill'] = None
	if 'alpha' not in kwargs.keys(): kwargs['alpha'] =1

	eig, eig_vals = np.linalg.eig(metric)
	w, h = 2*np.sqrt(dist**2/eig)
	angle = np.arctan2(eig_vals[1,0], eig_vals[0,0])*180/np.pi
	ellipse = matplotlib.patches.Ellipse(center, w, h, angle = angle, **kwargs)

	return ellipse	

def plot_match_histogram(matches_metric = None, matches = None, mm = None, bank_name = None, save_folder = None):
	"""
	Makes a simple histogram of the injection recovery.
	
	Parameters
	----------
		matches_metric: :class:`~numpy:numpy.ndarray`
			shape: (N,) -
			Values for the matches computed by the metric, if any.
		
		matches: :class:`~numpy:numpy.ndarray`
			shape: (N,) -
			Values for the true matches, if any.
		
		mm: float
			Minimum match requirement used to create the bank. Used for visualization purposes.
		
		bank_name: str
			Name of the bank tested. Used to set a title.
		
		save_folder: str
			Folder where to save the plots
			If `None`, no plots will be saved
	"""
	fs = 15
	plt.figure(figsize = (15,15))
	if bank_name: plt.title("Injection recovery for bank {}".format(bank_name), fontsize = fs+10)
	plt.gca().tick_params(axis='x', labelsize=fs)
	plt.gca().tick_params(axis='y', labelsize=fs)
	if matches_metric is not None:
		nbins = int(np.sqrt(len(matches_metric)))
		#logbins = np.logspace(np.log10(np.percentile(matches_metric, .5)),np.log10(max(matches_metric)), nbins)
		plt.hist(matches_metric, bins = nbins, density = True, cumulative = True,
				color = 'blue',	label = 'metric match',
				histtype = 'step')
	if matches is not None:
		#logbins = np.logspace(np.log10(np.percentile(matches, .5)),np.log10(max(matches)), nbins)
		nbins = int(np.sqrt(len(matches)))
		plt.hist(matches, bins = nbins, histtype='step', cumulative = True,
				density = True, color = 'orange', label = 'match')
	
	if isinstance(mm, (float, int)): plt.axvline(x = mm, c = 'r') #DEBUG
	plt.legend(fontsize = fs+3)
	plt.yscale('log')
	plt.xlabel(r"$\mathcal{M}$")
	plt.ylabel("Cumulative fraction")

	if save_folder:
		if not save_folder.endswith('/'): save_folder = save_folder+'/'
		plt.savefig(save_folder+'FF_hist.png', transparent = False)

	return 

def plot_colormap(datapoints, values, variable_format, statistics = 'mean', bins = 10, fs = 15, values_label = None, savefile = None, show = False, title = None):
	"""
	Plots a colormap values for the given datapoints
		
	Parameters
	----------
		datapoints: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Datapoints to plot. They must be compatible with the chosen variable format
		
		values: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Datapoints to plot. They must be compatible with the chosen variable format
		
		variable_format: str
			How to handle the BBH variables.
		
		statistics: str
			Statistics to use for the :func:`~scipy:scipy.stats.binned_statistic_2d`

		bins: int
			Bins to use along each dimension (as in :func:`~scipy:scipy.stats.binned_statistic_2d`)

		fs: int
			Font size for the labels and axis

		values_label: str
			Labels for the colorbar.

		savefile: str
			File where to save the plots. If `None`, no plots will be saved
		
		show: bool
			Whether to show the plots
		
		title: str
			A title for all the plots
	"""
	var_handler = variable_handler()
		###
		#Plotting datapoints
		###
	
		###
		#Plotting
		###
	fsize = 4* datapoints.shape[1]-1
	fig, axes = plt.subplots(datapoints.shape[1]-1, datapoints.shape[1]-1, figsize = (fsize, fsize))
	if title: fig.suptitle(title)
	if datapoints.shape[1]-1 == 1:
		axes = np.array([[axes]])
	for i,j in permutations(range(datapoints.shape[1]-1), 2):
		if i<j:	axes[i,j].remove()

		#Plot the datapoints
	for ax_ in combinations(range(datapoints.shape[1]), 2):
		currentAxis = axes[ax_[1]-1, ax_[0]]
		ax_ = list(ax_)
		
		stat, x_edges, y_edges, binnumber = binned_statistic_2d(datapoints[:,ax_[0]], datapoints[:,ax_[1]], values = values,
			statistic = statistics, bins = bins)
		X, Y = np.meshgrid(x_edges,y_edges)
		mesh = currentAxis.pcolormesh(X, Y, stat.T)
		
		cbar = plt.colorbar(mesh, ax = currentAxis)
		if values_label: cbar.set_label(values_label, rotation=270, labelpad = 15)

		if ax_[0] == 0:
			currentAxis.set_ylabel(var_handler.labels(variable_format, latex = True)[ax_[1]], fontsize = fs)
		else:
			currentAxis.set_yticks([])
		if ax_[1] == datapoints.shape[1]-1:
			currentAxis.set_xlabel(var_handler.labels(variable_format, latex = True)[ax_[0]], fontsize = fs)
		else:
			currentAxis.set_xticks([])
		currentAxis.tick_params(axis='x', labelsize=fs)
		currentAxis.tick_params(axis='y', labelsize=fs)

	if isinstance(savefile, str): plt.savefig(savefile, transparent = False)
	if show: plt.show()

def plot_tiles_templates(templates, variable_format, tiling = None, injections = None, inj_cmap = None, dist_ellipse = None, save_folder = None, fs = 15, show = False, savetag = '', title = None):
	"""
	Make some plots of a bunch of templates, possibly with a tiling and/or injections
		
	Parameters
	----------
		templates: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			The templates to plot, as stored in :attr:`mbank.bank.cbc_bank.templates`
		
		variable_format: str
			How to handle the BBH variables.
		
		tiling: tiling_handler
			Tiling handler that tiles the parameter space. If `None`, no tiling will be plotted	
			
		injections: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			An extra set of injections to plot. If `None`, no extra points will be plotted.
		
		inj_cmap: :class:`~numpy:numpy.ndarray`
			shape: (N,) -
			A colouring value for each injection, tipically the match with the bank. If None, no colouring will be done.
			The argument is ignored if `injections = None`
		
		dist_ellipse: float
			The distance for the match countour ellipse to draw. If `None`, no contour will be drawn.
			Requires a tiling object.
		
		fs: int
			Font size for the labels and axis
		
		save_folder: str
			Folder where to save the plots
			If `None`, no plots will be saved
		
		show: bool
			Whether to show the plots
		
		savetag: str
			A tag to append to the name of each file, to distinguish between different call
		
		title: str
			A title for all the plots

	"""
	templates = np.asarray(templates)
	
	var_handler = variable_handler()
		###
		#Plotting templates
		###
	if isinstance(save_folder, str): 
		if not save_folder.endswith('/'): save_folder = save_folder+'/'
		if savetag: savetag = '_{}'.format(savetag)
	
	if isinstance(dist_ellipse, float): #computing a tile for each template
		if tiling is None: raise ValueError("If ellipses are to be plotted, a tiling object is required but None is given")
		dist_template = []
		for t in tiling:
			dist_template.append( t[0].min_distance_point(templates) ) #(N_templates,)
		dist_template = np.stack(dist_template, axis = 1) #(N_templates, N_tiles)
		id_tile_templates = np.argmin(dist_template, axis = 1) #(N_templates,)
		del dist_template
	
		###
		#Plotting templates & tiles
		###
	if templates.shape[0] >500000: ids_ = np.random.choice(templates.shape[0], 500000, replace = False)
	else: ids_ = range(templates.shape[0])
	
	size_template = 20 if templates.shape[0] < 500 else 2
	fsize = 4* templates.shape[1]-1
	fig, axes = plt.subplots(templates.shape[1]-1, templates.shape[1]-1, figsize = (fsize, fsize))
	if title: fig.suptitle(title)
	#plt.suptitle('Templates of the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
	if templates.shape[1]-1 == 1:
		axes = np.array([[axes]])
	for i,j in permutations(range(templates.shape[1]-1), 2):
		if i<j:	axes[i,j].remove()

		#Plot the templates
	for ax_ in combinations(range(templates.shape[1]), 2):
		currentAxis = axes[ax_[1]-1, ax_[0]]
		ax_ = list(ax_)
		
		currentAxis.scatter(templates[ids_,ax_[0]], templates[ids_,ax_[1]], s = size_template, marker = 'o', c= 'b', alpha = 0.3)
		if ax_[0] == 0:
			currentAxis.set_ylabel(var_handler.labels(variable_format, latex = True)[ax_[1]], fontsize = fs)
		else:
			currentAxis.set_yticks([])
		if ax_[1] == templates.shape[1]-1:
			currentAxis.set_xlabel(var_handler.labels(variable_format, latex = True)[ax_[0]], fontsize = fs)
		else:
			currentAxis.set_xticks([])
		currentAxis.tick_params(axis='x', labelsize=fs)
		currentAxis.tick_params(axis='y', labelsize=fs)

	if isinstance(save_folder, str): plt.savefig(save_folder+'bank{}.png'.format(savetag), transparent = False)

		#Plot the tiling
	if isinstance(tiling,list):
		centers = tiling.get_centers()
		#plt.suptitle('Templates + tiling of the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
		for ax_ in combinations(range(templates.shape[1]), 2):
			currentAxis = axes[ax_[1]-1, ax_[0]]
			ax_ = list(ax_)
			currentAxis.scatter(*centers[:,ax_].T, s = 30, marker = 'x', c= 'r', alpha = 1)
			for t in tiling:
				d = t[0].maxes- t[0].mins
				currentAxis.add_patch(matplotlib.patches.Rectangle(t[0].mins[ax_], d[ax_[0]], d[ax_[1]], fill = None, alpha =1))

		if isinstance(save_folder, str): plt.savefig(save_folder+'tiling{}.png'.format(savetag), transparent = False)

		#Plotting the injections, if it is the case
	if isinstance(injections, np.ndarray):
		inj_cmap = 'r' if inj_cmap is None else inj_cmap
		#plt.suptitle('Templates + tiling of the bank & {} injections'.format(injections.shape[0]), fontsize = fs+10)
		for ax_ in combinations(range(templates.shape[1]), 2):
			currentAxis = axes[ax_[1]-1, ax_[0]]
			ax_ = list(ax_)
			cbar_vals = currentAxis.scatter(injections[:,ax_[0]], injections[:,ax_[1]],
				s = 20, marker = 's', c= inj_cmap)
		if not isinstance(inj_cmap, str):
			cbar_ax = fig.add_axes([0.88, 0.15, 0.015, 0.7])
			cbar_ax.tick_params(labelsize=fs)
			fig.colorbar(cbar_vals, cax=cbar_ax)
		if isinstance(save_folder, str): plt.savefig(save_folder+'injections{}.png'.format(savetag), transparent = False)


		#Plotting the ellipses, if it is the case
	if isinstance(dist_ellipse, float):
		#plt.suptitle('Templates + tiling + ellipses of the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
		for ax_ in combinations(range(templates.shape[1]), 2):
			currentAxis = axes[ax_[1]-1, ax_[0]]
			ax_ = list(ax_)
			for i, templ in enumerate(templates):
				metric_projected = project_metric(tiling[id_tile_templates[i]][1], ax_)
				currentAxis.add_patch(get_ellipse(metric_projected, templ[ax_], dist_ellipse))
			#if ax_[0]!=0: currentAxis.set_xlim([-10,10]) #DEBUG
			#currentAxis.set_ylim([-10,10]) #DEBUG
		if isinstance(save_folder, str): plt.savefig(save_folder+'ellipses{}.png'.format(savetag), transparent = False)
	
		#Plot an histogram
	fig, axes = plt.subplots(1, templates.shape[1], figsize = (4*templates.shape[1], 5), sharey = True)
	if title: fig.suptitle(title)
	#plt.suptitle('Histograms for the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
	hist_kwargs = {'bins': min(50, int(len(templates)/50 +1)), 'histtype':'step', 'color':'orange'}
	for i, ax_ in enumerate(axes):
		ax_.hist(templates[:,i], **hist_kwargs)
		if i==0: ax_.set_ylabel("# templates", fontsize = fs)
		ax_.set_xlabel(var_handler.labels(variable_format, latex = True)[i], fontsize = fs)
		min_, max_ = np.min(templates[:,i]), np.max(templates[:,i])
		d_ = 0.1*(max_-min_)
		ax_.set_xlim((min_-d_, max_+d_ ))
		ax_.tick_params(axis='x', labelsize=fs)
		ax_.tick_params(axis='y', labelsize=fs)
	if isinstance(save_folder, str): plt.savefig(save_folder+'hist{}.png'.format(savetag), transparent = False)
	
	if show: plt.show()
	
	return

####################################################################################################################
def project_metric(metric, axes):
	"""
	Projects the metric on the given axes.
	It follows a discussion on `stackexchange <https://math.stackexchange.com/questions/2431159/how-to-obtain-the-equation-of-the-projection-shadow-of-an-ellipsoid-into-2d-plan>`_
	
	Parameters
	----------
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D) - 
			A D dimensional metric
		
		axes: list, int
			The D' axes to project the metric over
	
	Returns
	-------
		projected_metric: :class:`~numpy:numpy.ndarray`
			shape: (D',D') - 
			The projected dimensional metric

	"""
	#FIXME: this name is misleading: maybe marginalize/reduce is better?

	if isinstance(axes, int): axes = [axes]
	if len(axes) == metric.shape[0]: return metric
	
	other_axes = [ d for d in range(metric.shape[0]) if d not in axes]
	
	J = metric[axes,:][:,axes] #(D',D')
	K = metric[other_axes,:][:,other_axes] #(K,K)
	L = metric[axes,:][:,other_axes] #(D',K) 
	
	K_inv = np.linalg.inv(K)

	proj_metric = J - np.einsum('ij,jk,kl->il', L, K_inv, L.T)
	
	return proj_metric

def clip_eigenvalues(metric, min_eig = 5e-2):
	"""
	Given a metric, it sets to ``min_eig`` the eigenvalues of the metric which are lower than ``min_eig``.
	
	Parameters
	----------
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D)/(N,D,D) - 
			A D dimensional metric
		
		min_eig: float
			The minimum value for the eigenvalues. The metric will be changed accordingly
	
	Returns
	-------
		trimmed_metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D)/(N,D,D) - 
			The clipped-eigenvalues D dimensional metric
	"""

	metric = np.asarray(metric)
	eigval, eigvec = np.linalg.eig(metric)
	print(eigval) #DEBUG
	
	#eigval[eigval<min_eig] = min_eig
	eigval[eigval<min_eig] = np.maximum(10*eigval[eigval<min_eig], min_eig)
	
	print(eigval) #DEBUG

	if metric.ndim <3:
		return np.linalg.multi_dot([eigvec, np.diag(eigval), eigvec.T]) #for 2D matrices
	else:
		return np.einsum('ijk,ik,ilk->ijl', eigvec, eigval, eigvec) #for 1x2D matrices


def get_projector(*directions):
	"""
	Given a set of orthogonal directions it computes the projector matrix :math:`P` on the orthogonal space.
	This means that for each direction :math:`d` in directions and each point :math:`x` of the space:
	
	.. math::
	
		<d, Px> = 0
	
	See this `stack exchange <https://math.stackexchange.com/questions/2320236/projection-on-the-hyperplane-h-sum-x-i-0>`_ page for more info.
	
	Parameters
	----------
		directions: list
			List of the orthogonal directions ``d`` that defines the projector. Each direction has dimension D.
	
	Returns
	-------
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D) -
			The projector (a D dimensional metric)
	
	"""
		#Doing some checks on the input
	directions = list(directions)
	for i, d in enumerate(directions):
		if not np.allclose(np.linalg.norm(d), 1): directions[i] = d/np.linalg.norm(d)
	for n1, n2 in combinations(directions,2):
		assert np.allclose(np.vdot(n1, n2), 0.), "Directions must be orthogonal"
	
		#Building the projector
	P = np.eye(len(directions[0]))
	
	for d in directions:
		P = P - np.outer(d,d)
	
	return P

def get_projected_metric(metric, min_eig = 1e-4):
	"""
	Given a metric :math:`M`, it applies the projector operator :math:`P` along the eigen-direction with eigenvalue smaller than ``min_eig``.
	
	The resulting metric is obtained as:
	
	.. math::
	
		M' = PMP
	
	Parameters
	----------
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D) - 
			A D dimensional metric

		min_eig: float
			Minimum tolerable eigenvalue for ``metric``. The metric will be projected on the directions of smaller eigenvalues

	Returns
	-------
		metric_projected: :class:`~numpy:numpy.ndarray`
			shape: (D,D) -
			The D dimensional metric projected along the large eigenvalues
	
	
	"""
	
	metric = np.asarray(metric)
	eigval, eigvec = np.linalg.eig(metric)
	
	ids_ = np.where(eigval<=min_eig)[0]
	#print(ids_, eigval, min_eig)
	
	if len(ids_)==0: return metric
	
	P = get_projector(*eigvec[:,ids_].T)
	
	new_metric = np.linalg.multi_dot([P, metric,P])

			#creating low dimensional matrix (in the eigenvector basis)
	ids_ = np.where(eigval>min_eig)[0] #(K,)
	eigvec = eigvec[:,ids_] #(D, K)
	new_metric_small = np.linalg.multi_dot([eigvec.T, metric, eigvec])
	
	return new_metric

####################################################################################################################

def plawspace(start, stop, exp, N_points):
	"""
	Generates a grid which is `power law distributed`. It has almost the same behaviour as :func:`~numpy:numpy.logspace`.
	Helpful for a nice grid spacing in the mass sector.
	
	Parameters
	----------
		start: float
			Starting point of the grid

		end: float
			End point of the grid	
		
		exp: float
			Exponent for the equally space points
		
		N_points: int
			Number of points to include in the grid		
	
	"""
	assert exp != 0, "Exponent of plawspace cannot be negative!"
	f_start = np.power(start, exp)
	f_stop = np.power(stop, exp)
	points = np.linspace(f_start, f_stop, N_points)
	points = np.power(points, 1/exp)
	
	points[0]=start
	if N_points>1: points[-1] = stop
	
	return points

def partition_tiling(thresholds, d, tiling):
	"""
	Given a tiling, it partitions the tiling given a list of thresholds. The splitting is performed along axis ``d``.
	
	Parameters
	----------
		thresholds: list
			list of trhesholds for the partitioning
		
		d: int
			Axis to split the tiling along.
		
		tiling: tiling_handler
			Tiling to partion
	
	Returns
	-------
		partitions: list
			List of tiling handlers making the partitions
	"""
	#TODO: this should be a member of tiling_handler
	if not isinstance(thresholds, list): thresholds = list(thresholds)
	thresholds.sort()
	
	partitions = []
	
	t_obj_ = tiling
	
	for threshold in thresholds:
		temp_t_obj, t_obj_ = t_obj_.split_tiling(d, threshold)
		partitions.append(temp_t_obj)
	partitions.append(t_obj_)
	
	return partitions


def get_boundary_box(grid_list):
	lower_grids = [g[:-1] for g in grid_list]
	upper_grids = [g[1:] for g in grid_list]
	
	lower_grids = np.meshgrid(*lower_grids)
	lower_grids = [g.flatten() for g in lower_grids]
	lower_grids = np.column_stack(lower_grids)
	
	upper_grids = np.meshgrid(*upper_grids)
	upper_grids = [g.flatten() for g in upper_grids]
	upper_grids = np.column_stack(upper_grids)
	
	return lower_grids, upper_grids

	
def split_boundaries(boundaries, grid_list, use_plawspace = True):
	"""
	Splits a boundary rectangle by dividing each dimension into a number of evenly spaced segments defined by each entry ``grid_list``.
	
	If option ``plawspace`` is set, the segments will be evenly distributed acconrding to a power law with exponent -8/3
	
	Parameters
	----------
		boundaries: :class:`~numpy:numpy.ndarray`
			shape: (2,D) -
			Boundaries of the space to split.
			Lower limit is ``boundaries[0,:]`` while upper limits is ``boundaries[1,:]``
		
		grid_list: list
			A list of ints, each representing the number of coarse division of the space.
		
		use_plawspace: bool
			Whether to use a power law spacing for the first variable

	Returns
	-------
		boundaries_list: list
			A list of the boundaries arrays obtained after the splitting, each with shape ``(2,D)``

	"""
	D = boundaries.shape[1]
	grid_list_ = []
	for i in range(D):
		if i ==0:
				#placing m_tot or M_chirp according the scaling relation: mc**(-8/3)*l ~ const.
			if use_plawspace: g_list = plawspace(boundaries[0,i], boundaries[1,i], -8./3., grid_list[i]+1) #power law spacing
			else: g_list = np.linspace(boundaries[0,i], boundaries[1,i], grid_list[i]+1) #linear spacing
		else:
			g_list = np.linspace(boundaries[0,i], boundaries[1,i], grid_list[i]+1)
		grid_list_.append( g_list )
	grid_list = grid_list_
		
	lower_boxes, upper_boxes = get_boundary_box(grid_list)
	return [(low, up) for low, up in zip(lower_boxes, upper_boxes) ]

##########################################################################################
def get_antenna_patterns(longitude, latitude, polarization):
	"""
	Returns the antenna pattern functions :math:`F_+, F_\\times` for an hypotetical interferometer located at the north pole.
	Antenna patterns are defined in terms of the longitude and latitude (sky location) :math:`\\alpha, \delta` and the polarization angle angle :math:`\Psi` as:
	
	.. math::
	
		F_+ = - \\frac{1}{2}(1 + \cos(\\theta^2)) \cos(2\\alpha) \cos(2\Psi) - \cos(\\theta)\sin(2\\alpha)\sin(2\Psi)
		
		F_\\times = \\frac{1}{2}(1 + \cos(\\theta^2)) \cos(2\\alpha) \sin(2\Psi) - \cos(\\theta)\sin(2\\alpha)\cos(2\Psi) 
	
	where :math:`\\theta = \\frac{\pi}{2} - \\delta`.
	
	See `pycbc.detector <https://github.com/gwastro/pycbc/blob/ce305cfb9fca3b59b8b9d3b16a3a486ae6c067cb/pycbc/detector.py#L559>`_ for more information.
	
	Parameters
	----------
		longitude: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Longitude of the source (right ascension)
		
		latitude: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Latitude of the source (declination)

		polarization: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Polarization angle

	Returns
	-------
		F_p: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Values for the plus antenna pattern
		
		F_c: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Values for the cross antenna pattern
	"""
	theta = np.pi/2 - np.asarray(latitude)
	
	F_p = - 0.5*(1 + np.cos(theta)**2)* np.cos(2*longitude)* np.cos(2*polarization)
	F_p -= np.cos(theta)* np.sin(2*longitude)* np.sin(2*polarization) 
	F_c = 0.5*(1 + np.cos(theta)**2)* np.cos(2*longitude)* np.sin(2*polarization)
	F_c -= np.cos(theta)* np.sin(2*longitude)* np.cos(2*polarization) 
	
	#print('mbank')
	#print(theta, alpha, psi)
	#print('\t', F_p, F_c)

	return F_p, F_c

def get_random_sky_loc(N = None, seed = None):
	"""
	Returns a random value for the sky location :math:`\\alpha, \delta` and the polarization angle :math:`\Psi`:
	The values for the sky location are randomly drawn uniformly across the sky and the polarization angle is uniformly extracted in the range :math:`[-\pi,\pi]`.
	
	Parameters
	----------
		N: int
			Number of pattern values to be extracted
			If `None`, one value will be extraced and the returned arrays will be one dimensional.
		
		seed: int
			A random seed for the sky location and polarization angles

	Returns
	-------
		longitude: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Longitude of the source (right ascension)
		
		latitude: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Latitude of the source (declination)

		polarization: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Polarization angle
	"""
	if isinstance(seed, int): np.random.seed(seed)
	polarization = np.random.uniform(0, 2*np.pi, N)
	latitude = np.arcsin(np.random.uniform(-1.,1., N)) #FIXME: check this is right!
	longitude = np.random.uniform(-np.pi, np.pi, N)
	return longitude, latitude, polarization

##########################################################################################
#TODO: use this function every time you read an xml!!!
def read_xml(filename, table, N = None):
	"""
	Read an xml file in the ligo.lw standard and extracts the BBH parameters
	
	
	Parameters
	----------
		filename: str
			Name of the file to load
		
		table: ligo.lw.lsctables.table.Table, str
			A ligo.lw table type. User typically will want to set `ligo.lw.lsctables.SnglInspiralTable` for a bank and `ligo.lw.lsctables.SimInspiralTable` for injections.
			A string 'sngl_inspiral' or 'sim_inspiral' can be given, instead of a ligo object, referring to `ligo.lw.lsctables.SnglInspiralTable` and `ligo.lw.lsctables.SimInspiralTable` respectively.
		
		N: int
			Number of rows to be read. If `None` all the rows inside the table will be read
	
	Returns
	-------
		BBH_components: :class:`~numpy:numpy.ndarray`
			shape (N,12) -
			An array with the read BBH components. It has the same layout as in `mbank.handlers.variable_handler`
		
		sky_locs: :class:`~numpy:numpy.ndarray`
			shape (N,3) -
			An array with sky localization and polarization angles. Each row contains longitude :math:`\\alpha`, latitude :math:`\delta` and polarization angle :math:`\Psi` (see :func:`get_antenna_patterns`). Returned only if reading a SimInspiralTable.
	"""
	if isinstance(table, str):
		if table == 'sim_inspiral':
			table = lsctables.SimInspiralTable
		elif table == 'sngl_inspiral':
			table = lsctables.SnglInspiralTable
		else:
			raise ValueError("Table string not understood")

	xmldoc = lw_utils.load_filename(filename, verbose = False, contenthandler = LIGOLWContentHandler)
	table = table.get_table(xmldoc)
	
	if not isinstance(N, int): N = len(table)
	BBH_components, sky_locs = [], []
			
	for i, row in enumerate(table):
		if i>=N: break
		
		if isinstance(table, lsctables.SnglInspiralTable):
			iota, phi = row.alpha3, row.alpha5
			e, meanano = 0, 0
		else:
			iota, phi = row.inclination, row.coa_phase
			e, meanano = row.psi0, row.psi3 #FIXME: This is a hack. Is it correct??
			
		BBH_components.append([row.mass1, row.mass2, #masses
			row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z, #spins
			e, meanano, iota, phi]) #e, meananano, iota, phi
		
		if isinstance(table, lsctables.SimInspiralTable):
			sky_locs.append([row.longitude, row.latitude, row.polarization])
		
	BBH_components = np.array(BBH_components) #(N,12)
	
	if isinstance(table, lsctables.SimInspiralTable):
		return BBH_components, np.array(sky_locs)
	
	return BBH_components
		
def save_injs(filename, injs, GPS_start, GPS_end, time_step, approx, sky_locs = None, luminosity_distance = 100, f_min = 10., f_max = 1024.):
		"""
		Save the given injections to a ligo xml injection file (sim_inspiral table).
		
		Parameters
		----------
			
		filename: str
			Filename to save the injections at
		
		injs: :class:`~numpy:numpy.ndarray`
			shape (N,12) -
			Injection array. It must be in the same layout of :func:`mbank.handlers.get_BBH_components`: ``m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano iota, phi``.
		
		GPS_start: int
			Start GPS time for the injections

		GPS_end: int
			End GPS time for the injections

		time_step: float
			Time step between consecutive injections
			Warning: no checks are made for overlapping injections
		
		approx: str
			Lal approximant to use to perform injections
		
		sky_locs: :class:`~numpy:numpy.ndarray`
			shape (N,3) -
			An array with sky localization and polarization angles. Each row contains longitude :math:`\\alpha`, latitude :math:`\delta` and polarization angle :math:`\Psi` (see :func:`get_antenna_patterns`).
			If ``None``, they will randomly drawn uniformly over the sky.
		
		luminosity_distance: float/tuple
			Luminosity distance in Mpc for the all the injections
			If a tuple, it has the meaning max luminosity/min luminosity

		f_min: float
			Starting frequency (in Hz) for the injections

		f_max: float
			End frequency (in Hz) for the injections
		
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
		if isinstance(luminosity_distance, (int, float, complex)):
			luminosity_distance = (luminosity_distance, luminosity_distance*1.001)
		else:
			assert isinstance(luminosity_distance, tuple), "Wrong format for luminosity distance. Must be a float or a tuple of float, not {}".format(type(luminosity_distance))
		
			#Dealing with sky location & distance
		N = len(injs)
		if sky_locs is None:
			sky_locs = np.stack([*get_random_sky_loc(N)], axis = 1)
		else:
			sky_locs = np.atleast_2d(np.asarray(sky_locs))
			
			if sky_locs.shape[0] ==1:
				sky_locs = np.repeat(sky_locs, N, axis = 0)
			
			assert sky_locs.shape[0] >= N, "The number of sky location values should be the same as the number of injections"
		
			#opening a document
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		sim_inspiral_table = lsctables.New(lsctables.SimInspiralTable)

		process = ligolw_process.register_to_xmldoc(
			xmldoc,
			program="mbank",
			paramdict={},
			comment="")

			#loops on rows (i.e. injections)
		for i, t_inj in enumerate(np.arange(GPS_start, GPS_end, time_step)):

			if i>=N:
				break
			
			m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi = injs[i]


				#boring initialization stuff
			row =  DefaultSimInspiralTable()
			row.process_id = process.process_id
			row.simulation_id = i #int?
			row.waveform = approx
			row.f_lower = f_min
			row.f_final = f_max
			row.taper = "TAPER_START"
			row.bandpass = 0
			
			row.psi0, row.psi3 = e, meanano #FIXME: This is a hack. Is it correct??

				#setting interesting row paramters
			row.inclination = iota
			row.coa_phase = phi
			row.longitude, row.latitude, row.polarization = sky_locs[i]
			row.distance = np.random.uniform(*luminosity_distance)

				#setting masses/spins and other related quantities
			row.mass1, row.mass2 = m1, m2
			row.spin1x, row.spin1y, row.spin1z = s1x, s1y, s1z
			row.spin2x, row.spin2y, row.spin2z = s2x, s2y, s2z
			
			row.mtotal = row.mass1 + row.mass2
			row.eta = row.mass1 * row.mass2 / row.mtotal**2
			row.mchirp = ((row.mass1 * row.mass2)**3/row.mtotal)**0.2
			row.chi = (row.mass1 *row.spin1z + row.mass2 *row.spin2z) / row.mtotal #is this the actual chi?
			#row.chi = (np.sqrt(row.spin1x**2+row.spin1y**2+row.spin1z**2)*m1 + np.sqrt(row.spin2x**2+row.spin2y**2+row.spin2z**2)*m2)/row.mtotal
			
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
			
				#this setting fucks things a bit! (is it really required??)
			#row.alpha4 = 20 #This is H1 SNR. Required?
			#row.alpha5 = 20 #This is L1 SNR. Required?
			#row.alpha6 = 20 #This is V1 SNR. Required?

				#appending row
			sim_inspiral_table.append(row)

		#ligolw_process.set_process_end_time(process)
		xmldoc.childNodes[-1].appendChild(sim_inspiral_table)
		lw_utils.write_filename(xmldoc, filename, verbose=False)
		xmldoc.unlink()

		print("Saved {} injections to {}".format(i+1, filename)) #FIXME: remove this from here!

		return 

def load_inj_stat_dict(filename):
	"""
	Loads an injection statistics dictionary (see :func`compute_injections_match`) from a json or pickle file.
	
	Parameters
	----------
			
		filename: str
			Filename to load the dictionary from.
		
	Returns
	-------
		
		inj_stat_dict: dict
			A dictionary with the results of an injection study. It can be generated by :func`compute_injections_match` and :func:`compute_injections_metric_match`.
	"""
	if filename.endswith('pkl'):
		with open(filename, 'rb') as f:
			inj_stat_dict = pickle.load(f)
	
	elif filename.endswith('json'):
	
		with open(filename, 'r') as f:
			inj_stat_dict = json.load(f)
			#converting list to np.arrays
		for k,v in inj_stat_dict.items():
			if isinstance(v, list): inj_stat_dict[k] = np.array(v)
	else:
		raise ValueError("Only json and pickle are valid formats to load an injection stat dictionary")

	return inj_stat_dict

def save_inj_stat_dict(filename, inj_stat_dict):
	"""
	Saves an injection statistics dictionary (see :func`compute_injections_match`) to file with json or pickle format. The dictionary can be loaded again with :func:`load_inj_stat_dict`.
	The format of the file is determined by the extension of the file ('json' or 'pkl').
	
	Parameters
	----------
			
		filename: str
			Filename to save the dictionary at
		
		inj_stat_dict: dict
			A dictionary with the results of an injection study. It can be generated by :func`compute_injections_match` and :func:`compute_injections_metric_match`.
	"""
	if filename.endswith('pkl'):
		with open(filename, 'wb') as f:
			pickle.dump(inj_stat_dict, f)
	
	elif filename.endswith('json'):
			#converting np. arrays to list (working with a copy)
		inj_stat_dict = dict(inj_stat_dict)
		for k,v in inj_stat_dict.items():
			if isinstance(v, np.ndarray): inj_stat_dict[k] = v.tolist()
		
			#serializing to json
		with open(filename, 'w') as f:
			json.dump(inj_stat_dict, f, indent = 2)
	else:
		raise ValueError("Only json and pickle are valid formats to save an injection stat dictionary")

	return


"""
Snippet for caching


if cache_folder:
				#dealing with cached WFs
			template_WF = []
			for id_ in inj_dict['id_match_neig'][i,:]:
				id_file ='{}/{}'.format(cache_folder, id_)
				try:
					assert id_ in dict_ids
					new_WF = np.loadtxt(id_file, dtype = complex)
				except (OSError, AssertionError):
					new_WF = metric_obj.get_WF(templates[id_,:])
				if id_ in dict_ids:
					np.savetxt(id_file, new_WF)
					dict_ids[id_] = dict_ids[id_]-1
					if dict_ids[id_] == 0: os.remove(id_file)

				template_WF.append(new_WF)
				
				
			template_WF = np.stack(template_WF, axis =0) #(N_neigh_templates, D)
			print(template_WF.shape)
			inj_WF = metric_obj.get_WF(injs[i]) #(D,)
			true_match = metric_obj.WF_match(template_WF, inj_WF) #(N_neigh_templates,)
		else:
			true_match = metric_obj.match(template_, injs[i], symphony = symphony_match, overlap = False)
"""

















	
	
	
