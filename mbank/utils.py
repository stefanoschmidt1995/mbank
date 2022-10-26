"""
mbank.utils
===========
	Some utilities for ``mbank``, where you find lots of useful stuff for some boring operations on the templates.
	It keeps functions for plotting, for template placing, injection recovery computation and other useful operations useful around the package.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
from itertools import combinations, permutations, product
import argparse
import lal.series
import configparser
import os
import ast
import sys
import scipy


	#ligo.lw imports for xml files: pip install python-ligo-lw
from ligo.lw import utils as lw_utils
from ligo.lw import ligolw
from ligo.lw import table as lw_table
from ligo.lw import lsctables
from ligo.lw.utils import process as ligolw_process
from ligo.lw.utils import load_filename

from scipy.spatial import Rectangle

from tqdm import tqdm
import ray

from .handlers import variable_handler

from scipy.spatial import ConvexHull, Rectangle

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

class DefaultSnglInspiralTable(lsctables.SnglInspiralTable):
	"""
	This is a copy of ``ligo.lw.lsctables.SnglInspiralTable`` with implemented defaults.
	Implemented as `here <https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/waveforms.py#L39>`_
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
	Implemented as `here <https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/waveforms.py#L39>`_
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
#Parser stuff

def int_tuple_type(strings): #type for the grid-size parser argument
	strings = strings.replace("(", "").replace(")", "")
	mapped_int = map(int, strings.split(","))
	return tuple(mapped_int)

class parse_from_file(argparse.Action):
	"Convenience class to read the parser arguments from a config file \n\n ``DEPRECATED``"	
	#https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
	def __call__ (self, parser, namespace, values, option_string=None):
		with values as f:
			read_args = [a  for arg in f.readlines() if not arg.strip('\t').startswith('#') for a in arg.split() if not a.startswith('#')]

		# parse arguments in the file and store them in a blank namespace
		data, _  = parser.parse_known_args(read_args, namespace=None)
		for k, v in vars(data).items():
				# set arguments in the target namespace if they haven’t been set yet (i.e. they are not their default value)
			if getattr(namespace, k, None) == parser.get_default(k):
				setattr(namespace, k, v)


def updates_args_from_ini(ini_file, args, parser):
	"""	
	Updates the arguments of Namespace args according to the given `ini_file`.

	Parameters
	----------
		ini_file: str
			Filename of the ini file to load. It must readable by `configparser.ConfigParser()`
		args: argparse.Namespace
			A parser namespace object to be updated
		parser: argparse.ArgumentParser
			A parser object (compatibile with the given namespace)
	
	Returns
	-------
		args: argparse.Namespace
			Updated parser namespace object
	"""
	if not os.path.exists(ini_file):
		raise FileNotFoundError("The given ini file '{}' doesn't exist".format(ini_file))
		
		#reading the ini file
	config = configparser.ConfigParser()
	config.read(ini_file)
	assert len(config.sections()) ==1, "The ini file must have only one section"
	
		#casting to a dict and adding name entry
		#in principle this is not required, but makes things more handy
	ini_info = dict(config[config.sections()[0]])
	ini_info['run-name'] = config.sections()[0]

		#formatting the ini-file args
	args_to_read = []
	for k, v in ini_info.items():
		if v.lower() != 'false': #if it's a bool var, it is always store action
			args_to_read.extend('--{} {}'.format(k,v).split(' '))

	#args, _ = parser.parse_known_args(args_to_read, namespace = args) #this will update the existing namespace with the new values...
	
		#adding the new args to the namespace (if the values are not the default)
	new_data, _ = parser.parse_known_args(args_to_read, namespace = None)
	for k, v in vars(new_data).items():
		# set arguments in the args if they haven’t been set yet (i.e. they are not their default value)
		if getattr(args, k, None) == parser.get_default(k):
			setattr(args, k, v)
	return args

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
	s1_range = (-0.99,0.99), s2_range = (-0.99,0.99), chi_range = (-0.99,0.99), theta_range = (0, np.pi), phi_range = (-np.pi, np.pi),
	iota_range = (0, np.pi), ref_phase_range = (-np.pi, np.pi), e_range = (0., 0.5), meanano_range = (0.1, 1.)):
	"""
	Given the ranges of each quantity, it combines them in a bondary array, suitable for other uses in the package (for instance in the bank generation).
	No checks are performed whether the given ranges make sense.
	
	Parameters
	----------
		variable_format: str
			A string to specify the variable format.
			See `mbank.handler.variable_format` for more information
		
		M_range, q_range, s1_range, s2_range, chi_range, theta_range, phi_range, iota_range, ref_phase_range, e_range, meanano_range: tuple
			Ranges for each physical quantity. They will be used whenever required by the `variable_format`
			If `mchirpeta` mass format is set, `M_range` and `q_range` are interpreted as mchirp and eta respectively.
			If `logMq` mass format is set, `M_range` is still interpreted as the mass and *not* the log mass.
	
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
def load_PSD(filename, asd = False, ifo = 'H1'):
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

	return f, PSD	


####################################################################################################################
def ray_compute_injections_match(inj_dict, templates, metric_obj, symphony_match = False, max_jobs = 8, cache = True):
	"""
	Wrapper to ``compute_injections_match()`` to allow for parallel execution. It calls ``_compute_injections_match_ray()``
	Given an injection dictionary, generated by ``compute_injections_metric_match()`` it computes the actual match (without the metric approximation) between injections and templates. It updates ``inj_dict`` with the new computed results.
	The injections are generic (not necessarly projected on the bank submanifold).
	
	Parameters
	----------
		inj_dict: dict
			A dictionary with the data injection as computed by `compute_injections_metric_match`.
		
		templates: :class:`~numpy:numpy.ndarray`
			An array with the templates. They should have the same layout as lal (given by get_BBH_components)

		metric_obj: cbc_metric
			A cbc_metric object to compute the match with.

		N_neigh_templates: 'int'
			The number of neighbouring templates to consider for each injection
						
		cache: bool
			Whether to cache the WFs
		
		max_jobs: int
			Maximum number of parallel ray jobs to be instantiated
			
		symphony_match: bool
			Whether to use the symphony match
		
	Returns
	-------
		out_dict: 'dict'
			The output dictionary with the updated matches
	"""
		###
		# Split injections
	n_injs_per_job = max(25, int(inj_dict['metric_match'].shape[0]/max_jobs)) 
	
		###
		# Initializing ray and performing the computation
	inj_dict_ray_list = []
	ray.init()
	for id_, i in enumerate(range(0, inj_dict['metric_match'].shape[0], n_injs_per_job)):
		inj_dict_ray_list.append( _compute_injections_match_ray.remote(i, i+n_injs_per_job, inj_dict,
			templates, metric_obj, symphony_match, cache, id_))
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
def _compute_injections_match_ray(start_id, end_id, inj_dict, templates, metric_obj, symphony_match = False, cache = True, worker_id = 0):
	"""
	Wrapper to ``compute_injections_match`` to allow for parallelization with ray.

	Parameters
	----------
		inj_dict: dict
			A dictionary with the data injection as computed by `compute_injections_metric_match`.
		
		templates: :class:`~numpy:numpy.ndarray`
			An array with the templates. They should have the same layout as lal (given by get_BBH_components)

		metric_obj: cbc_metric
			A ``cbc_metric`` object to compute the match with.

		N_neigh_templates: 'int'
			The number of neighbouring templates to consider for each injection
		
		symphony_match: bool
			Whether to use the symphony match
				
		cache: bool
			Whether to cache the WFs
			
		worker_id: 'int'
			Id of the ray worker being used.
		
	Returns
	-------
		out_dict: dict
			The output dictionary with the updated matches
	"""
	local_dict = {} #ray wants a local dict for some reason...

		#splitting the dictionary
	for k in inj_dict.keys():
		if isinstance(inj_dict[k], np.ndarray):
			local_dict[k] = np.copy(inj_dict[k][start_id:end_id])
		elif isinstance(inj_dict[k], list):
			local_dict[k] = inj_dict[k][start_id:end_id].copy()
		else:
			local_dict[k] = inj_dict[k]
	return compute_injections_match(local_dict, templates, metric_obj, symphony_match, cache, worker_id)

def compute_injections_match(inj_dict, templates, metric_obj, symphony_match = False, cache = True, worker_id = None):
	"""
	Given an injection dictionary, generated by ``compute_injections_metric_match`` it computes the actual match (not the metric approximation) between injections and templates. It updates ``inj_dict`` with the new computed results.
	The injections are generic (not necessarly projected on the bank submanifold).
	
	Parameters
	----------
		inj_dict: 'dict'
			A dictionary with the data injection as computed by `compute_injections_metric_match`.
		
		templates: :class:`~numpy:numpy.ndarray`
			An array with the templates. They should have the same layout as lal (given by get_BBH_components)

		metric_obj: cbc_metric
			A cbc_metric object to compute the match with.
		
		symphony_match: bool
			Whether to use the symphony match
		
		cache: bool
			Whether to cache the WFs
		
		worker_id: 'int'
			Id of the ray worker being used. If None, it is assumed that ray is not called
		
	Returns
	-------
		out_dict: 'dict'
			The output dictionary with the updated matches
	"""
	
	old_format = metric_obj.variable_format
	if metric_obj.variable_format != 'm1m2_fullspins_emeanano_iotaphi':
		metric_obj.set_variable_format('m1m2_fullspins_emeanano_iotaphi')

		#allocating memory for the match
	inj_dict['id_match'] = np.empty(inj_dict['id_metric_match'].shape, dtype=inj_dict['id_metric_match'].dtype)
	inj_dict['match'] = np.empty(inj_dict['metric_match'].shape, dtype=inj_dict['metric_match'].dtype)
	inj_dict['symphony_SNR'] = symphony_match

		#putting injections and templates with the format 'm1m2_fullspins_emeanano_iotaphi'
		# The format is basically the full 12 dimensional space, with spins in spherical coordinates
	injs = metric_obj.var_handler.get_theta(inj_dict['theta_inj'], 'm1m2_fullspins_emeanano_iotaphi')
	templates = metric_obj.var_handler.get_theta(templates, 'm1m2_fullspins_emeanano_iotaphi')
	
	if worker_id is None: desc = 'Computing the {} match: loop on the injections'.format('symphony' if symphony_match else 'std')
	else: desc = 'Worker {} - Computing the {} match: loop on the injections'.format(worker_id, 'symphony' if symphony_match else 'std')
	
	for i in tqdm(range(injs.shape[0]), desc = desc, leave = True):
		#Take a look at https://pypi.org/project/anycache/
		template_ = templates[inj_dict['id_match_list'][i],:] #(N', 12)
		
		template_WFs = metric_obj.get_WF(template_, plus_cross = symphony_match)
		inj_WF = metric_obj.get_WF(injs[i], plus_cross = symphony_match)

		if cache:
			raise NotImplementedError("Cache in injections are currently not supported")
		else:
			#true_match = metric_obj.match(template_, injs[i], symphony = symphony_match, overlap = False)
			if symphony_match:
				true_match = metric_obj.WF_symphony_match(template_WFs, inj_WF, False)
			else:
				true_match = metric_obj.WF_match(template_WFs, inj_WF, False)
			
			
			
			#updating the dict
		ids_max = np.argmax(true_match)

		inj_dict['match_list'].append(list(true_match)) #(N_neigh_templates,)
		inj_dict['id_match'][i] = np.argmax(true_match)
		inj_dict['match'][i] = np.max(true_match)

	metric_obj.set_variable_format(old_format)

	return dict(inj_dict)


####################################################################################################################	
#@do_profile(follow=[])
def compute_injections_metric_match(injs, bank, tiling, match_threshold = 0.9, verbose = True):
	"""
	Computes the match of the injection with the bank by using the metric approximation.
	It makes use of a brute force approach where each injection is checked against each template of the bank. The metric used is the one of the tile each injection belongs to
	
	Parameters
	----------
		injs: :class:`~numpy:numpy.ndarray`
			shape: (N,12)/(N,D) -
			A set of injections.
			If the shape is `(N,D)` they are assumed to be the lie in the bank manifold.
			Otherwise, they are in the full format of the ``mbank.metric.cbc_metric.get_BBH_components()``.
			Each row should be: m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi
		
		bank: cbc_bank
			A ``cbc_bank`` object
		
		tiling: tiling_handler
			A tiling object to compute the metric match between templates and injections

		match_threshold: float
			For each injection, the templates with match above `match_threshold` will be stored
		
		verbose: bool
			Whether to print the output
		
	Returns
	-------
		out_dict: dict
			A dictionary with the output. It has the entries:
			
			- ``theta_inj``: the parameters of the injections
			- ``id_tile``: index of the tile the injections belongs to (in the tiling)
			- ``match_threshold``: the given value for `match_threshold`
			- ``match``: match of the closest template (filled by function `compute_injections_match`)
			- ``id_match``: index of the closest template (filled by function `compute_injections_match`)
			- ``metric_match``: metric match of the closest template
			- ``id_metric_match``: metric index of the closest template
			- ``id_match_list``: list of all the templates closer than match_threshold to each injection
			- ``match_list``: list of all the matches between each injection and the templates in `id_match_list` (filled by function `compute_injections_match`)
		
			each entry is ``np.ndarray`` where each row is an injection.
	
	"""
		#storing injections in the full format (so that they can be generic)
	out_dict = {'theta_inj': np.array(bank.var_handler.get_BBH_components(injs, bank.variable_format)).T if injs.shape[1] != 12 else injs ,
				'id_tile': np.zeros((injs.shape[0],), int),
				'match_threshold': match_threshold, 'symphony_SNR': None,
				'match': None, 'id_match':  None,
				'metric_match': np.zeros((injs.shape[0],)), 'id_metric_match':  np.empty((injs.shape[0],), dtype = int),
				'id_match_list': [], 'match_list': []
				}
	
		#csating the injections to the metric type
	if injs.shape[1] == 12:
		injs = bank.var_handler.get_theta(injs, bank.variable_format)
	if injs.shape[1] != bank.D:
		raise ValueError("Wrong input size for the injections")
	
	template_dist = np.allclose(bank.templates, injs) if (bank.templates.shape == injs.shape) else False
	N_argpartiton = 20000
	id_diff_ok = np.arange(bank.templates.shape[0])
	
		#loops on the injections
	if verbose: inj_iter = tqdm(range(injs.shape[0]), desc = 'Evaluating metric match for injections', leave = True)
	else: inj_iter = range(injs.shape[0])
	
	if tiling.flow: metric_list = tiling.get_metric(injs, flow = True, kdtree = True)
	#TODO: optimize injections with flow (now it's too slow in the metric computation approach)
	
	for i in inj_iter:

		out_dict['id_tile'][i] = tiling.get_tile(injs[i])[0]

		diff = bank.templates - injs[i] #(N_templates, D)
		
			#these are the indices being checked
		if N_argpartiton < bank.templates.shape[0]:
			id_diff_ok = np.argpartition(np.linalg.norm(diff, axis=1), N_argpartiton)[:N_argpartiton]

			#using the flow to compute the true tiling metric (if available)
		if tiling.flow: metric = metric_list[i]
		else: metric = tiling[out_dict['id_tile'][i]].metric
		
		match_i = 1 - np.sum(np.multiply(diff[id_diff_ok], np.matmul(diff[id_diff_ok], metric)), axis = -1)
		
		out_dict['id_metric_match'][i] = np.argmax(match_i)
		out_dict['metric_match'][i] = match_i[out_dict['id_metric_match'][i]]

		out_dict['id_match_list'].append(list(np.where(match_i>match_threshold)[0]))
			#if nothing is below match_threshold, we look into the first 100 best matching templates (hopeless)
		if out_dict['id_match_list'][-1] == []: out_dict['id_match_list'][-1] = list(np.argsort(match_i)[-100:])
		
	return out_dict


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
	ellipse = matplotlib.patches.Ellipse(center, w, h, angle, **kwargs)

	return ellipse	


def plot_tiles_templates(templates, variable_format, tiling = None, injections = None, inj_cmap = None, dist_ellipse = None, save_folder = None, fs = 15, show = False):
	"""
	Make some plots of the templates and the tiling.
		
	Parameters
	----------
		templates: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			The templates to plot, as stored in ``cbc_bank.templates``
		
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

	"""
	var_handler = variable_handler()
		###
		#Plotting templates
		###
	if isinstance(save_folder, str): 
		if not save_folder.endswith('/'): save_folder = save_folder+'/'
	
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
	
	size_template = [20 if templates.shape[0] < 5000 else 2][0]
	fsize = 4* templates.shape[1]-1
	fig, axes = plt.subplots(templates.shape[1]-1, templates.shape[1]-1, figsize = (fsize, fsize))
	plt.suptitle('Templates of the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
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

	if isinstance(save_folder, str): plt.savefig(save_folder+'bank.png', transparent = False)

		#Plot the tiling
	if isinstance(tiling,list):
		centers = tiling.get_centers()
		plt.suptitle('Templates + tiling of the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
		for ax_ in combinations(range(templates.shape[1]), 2):
			currentAxis = axes[ax_[1]-1, ax_[0]]
			ax_ = list(ax_)
			currentAxis.scatter(*centers[:,ax_].T, s = 30, marker = 'x', c= 'r', alpha = 1)
			for t in tiling:
				d = t[0].maxes- t[0].mins
				currentAxis.add_patch(matplotlib.patches.Rectangle(t[0].mins[ax_], d[ax_[0]], d[ax_[1]], fill = None, alpha =1))

		if isinstance(save_folder, str): plt.savefig(save_folder+'tiling.png', transparent = False)

		#Plotting the injections, if it is the case
	if isinstance(injections, np.ndarray):
		inj_cmap = 'r' if inj_cmap is None else inj_cmap
		plt.suptitle('Templates + tiling of the bank & {} injections'.format(injections.shape[0]), fontsize = fs+10)
		for ax_ in combinations(range(templates.shape[1]), 2):
			currentAxis = axes[ax_[1]-1, ax_[0]]
			ax_ = list(ax_)
			cbar_vals = currentAxis.scatter(injections[:,ax_[0]], injections[:,ax_[1]],
				s = 20, marker = 's', c= inj_cmap)
		if not isinstance(inj_cmap, str):
			cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
			cbar_ax.tick_params(labelsize=fs)
			fig.colorbar(cbar_vals, cax=cbar_ax)
		if isinstance(save_folder, str): plt.savefig(save_folder+'injections.png', transparent = False)


		#Plotting the ellipses, if it is the case
	if isinstance(dist_ellipse, float):
		plt.suptitle('Templates + tiling + ellipses of the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
		for ax_ in combinations(range(templates.shape[1]), 2):
			currentAxis = axes[ax_[1]-1, ax_[0]]
			ax_ = list(ax_)
			for i, templ in enumerate(templates):
				metric_projected = project_metric(tiling[id_tile_templates[i]][1], ax_)
				currentAxis.add_patch(get_ellipse(metric_projected, templ[ax_], dist_ellipse))
			#if ax_[0]!=0: currentAxis.set_xlim([-10,10]) #DEBUG
			#currentAxis.set_ylim([-10,10]) #DEBUG
		if isinstance(save_folder, str): plt.savefig(save_folder+'ellipses.png', transparent = False)
	
		#Plot an histogram
	fig, axes = plt.subplots(1, templates.shape[1], figsize = (4*templates.shape[1], 5), sharey = True)
	plt.suptitle('Histograms for the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
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
	if isinstance(save_folder, str): plt.savefig(save_folder+'hist.png', transparent = False)
	
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
		
		axes: list
			The D' axes to project the metric over
	
	Returns
	-------
		projected_metric: :class:`~numpy:numpy.ndarray`
			shape: (D',D') - 
			The projected dimensional metric

	"""
	#FIXME: this name is misleading: maybe marginalize/reduce is better?

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
	Given a metric, it sets to `min_eig` the eigenvalues of the metric which are lower than `min_eig`.
	
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
	Given a set of orthogonal directions it computes the projector matrix P on the orthogonal space.
	This means that for each direction ``d`` in directions and each point ``x`` of the space:
	
	::
		<d, Px> = 0
	
	See `here <https://math.stackexchange.com/questions/2320236/projection-on-the-hyperplane-h-sum-x-i-0>`_ for more info.
	
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
	Given a metric, it applies the projector operator along the eigen-direction with eigenvalue smaller than ``min_eig``.
	
	The resulting metric is obtained as:
	
	::
	
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

def get_cube_corners(boundaries):
	"""
	Given the boundaries of an hyper-rectangle, it computes all the corners of it
	
	Parameters
	----------
		boundaries: :class:`~numpy:numpy.ndarray`
			shape: (2,D) -
			An array with the boundaries for the model. Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
	
	Returns
	-------
		corners: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			An array with the corners. Each row is a different corner
	
	"""
	corners = np.meshgrid(*boundaries.T)
	corners = [c.flatten() for c in corners]
	corners = np.column_stack(corners)
	return corners

def plawspace(start, stop, exp, N_points):
	"""
	Generates a grid which is 'power law distributed'. It has almost the same behaviour as ``np.logspace``.
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

def place_stochastically_in_tile(minimum_match, tile):
	"""
	Place templates with a stochastic placing algorithm withing a given tile, by iteratively proposing a new template to add to the bank inside the given tile.
	The proposal is accepted if the match of the proposal with the previously placed templates is smaller than ``minimum_match``. The iteration goes on until no template is found to have a distance smaller than the given threshold ``minimum_match``.
	
	
	Parameters
	----------
		minimum_match: float
			Minimum match between templates.
		
		tile: tuple
			An element of the ``tiling_handler`` object.
			It consists of a tuple ``(scipy.spatial.Rectangle, np.ndarray)``
	
	Returns
	-------
		new_templates: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			A set of templates generated by the stochastic placing algorithm within the given tile
	"""
	dist_sq = 1-minimum_match

		#initial template
	new_templates = np.random.uniform(tile.rectangle.mins, tile.rectangle.maxes, (1, tile.D)) #(1,D)
	
	nothing_new = 0
	while nothing_new < 300:
		proposal = np.random.uniform(tile.rectangle.mins, tile.rectangle.maxes, tile.D) #(D,)
		diff = new_templates - proposal

		min_dist = np.min(np.sum(np.multiply(diff, np.matmul(diff, tile.metric)), axis = -1))

		if min_dist > dist_sq:
			new_templates = np.concatenate([new_templates, proposal[None,:]], axis = 0)
			nothing_new = 0
		else:
			nothing_new += 1
		
	return new_templates

#@do_profile(follow=[])
def place_stochastically(minimum_match, tiling, empty_iterations = 200, seed_bank = None, verbose = True):
	"""
	Place templates with a stochastic placing algorithm.
	It iteratively proposes a new template to add to the bank. The proposal is accepted if the match of the proposal with the previously placed templates is smaller than ``minimum_match``. The iteration goes on until no template is found to have a distance smaller than the given threshold ``minimum_match``.
	It can start from a given set of templates.

	The match of a proposal is computed against all the templats that have been added.
	
	Parameters
	----------
		minimum_match: float
			Minimum match between templates.
		
		tiling: tiling_handler
			A tiling object to compute the match with
		
		empty_iterations: int
			Number of consecutive templates that are not accepted before the placing algorithm is terminated
			
		seed_bank: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			A set of templates that provides a first guess for the bank
		
		verbose: bool
			Whether to print the progress bar
	
	Returns
	-------
		new_templates: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			A set of templates generated by the stochastic placing algorithm
	"""
		#User communication stuff
	def dummy_iterator():
		while True:
			yield
	t_ = tqdm(dummy_iterator()) if verbose else dummy_iterator()

	MM = minimum_match

	if seed_bank is None:
		ran_id_ = np.random.choice(len(tiling))
		new_templates = np.random.uniform(tiling[ran_id_][0].mins, tiling[ran_id_][0].maxes, (1, len(tiling[ran_id_][0].maxes)))
	else:
		new_templates = np.asarray(seed_bank)

	nothing_new, i, max_nothing_new = 0, 0, 0
	
		#optimized version of the above... (not really helpful)
	if tiling.flow:
		import torch
		with torch.no_grad():
			log_pdf_centers = tiling.flow.log_prob(tiling.get_centers().astype(np.float32)).numpy()
		proposal_list, log_pdf_theta_list = [], []
	
	try:
		for _ in t_:

			if verbose and i%100==0:t_.set_description("Templates added {} ({}/{} empty iterations)".format(new_templates.shape[0], int(max_nothing_new), int(empty_iterations)))
			if nothing_new >= empty_iterations: break
			
			if tiling.flow:
					#Unoptimized version - you need to make things in batches!
				#proposal = tiling.sample_from_flow(1)
				#metric = tiling.get_metric(proposal, flow = True) #using the flow if it is trained
				
					#optimized version of the above... (not really helpful)
				with torch.no_grad():
					if len(proposal_list)==0:
						proposal_list, log_pdf_theta_list = tiling.flow.sample_and_log_prob(1000)
						proposal_list, log_pdf_theta_list = list(proposal_list.numpy()), list(log_pdf_theta_list.numpy())

					proposal, log_pdf_theta = proposal_list.pop(0)[None,:], log_pdf_theta_list.pop(0)
						#checking if the proposal is inside the tiling
					if not tiling.is_inside(proposal)[0]: continue

					#proposal, log_pdf_theta = tiling.flow.sample_and_log_prob(1)
					#proposal = proposal.numpy()
					
						#FIXME: this kdtree may mess things up
					id_ = tiling.get_tile(proposal, kdtree = True)[0]
					metric = tiling[id_].metric
					
					factor = (2/metric.shape[0])*(log_pdf_theta-log_pdf_centers[id_])
					factor = np.exp(factor)
			
					metric = (metric.T*factor).T
			else:
				#FIXME: this thing is fucking slooooow! Maybe you should do a fancy buffer to parallelize this?
				proposal, tile_id = tiling.sample_from_tiling(1, tile_id = True)
				metric = tiling[tile_id[0]].metric

			diff = new_templates - proposal #(N_templates, D)
			
			
			max_match = np.max(1 - np.sum(np.multiply(diff, np.matmul(diff, metric)), axis = -1))

			if (max_match < MM):
				new_templates = np.concatenate([new_templates, proposal], axis =0)
				nothing_new = 0
			else:
				nothing_new += 1
				max_nothing_new = max(max_nothing_new, nothing_new)
	
			i+=1
	except KeyboardInterrupt:
		pass
	
	if tiling.flow: del proposal_list, log_pdf_theta_list
	
	return new_templates


def partition_tiling(thresholds, d, tiling):
	"""
	Given a tiling, it partitions the tiling given a list of thresholds. The splitting is performed along axis `d`.
	
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
	if not isinstance(thresholds, list): thresholds = list(thresholds)
	thresholds.sort()
	
	partitions = []
	
	t_obj_ = tiling
	
	for threshold in thresholds:
		temp_t_obj, t_obj_ = t_obj_.split_tiling(d, threshold)
		partitions.append(temp_t_obj)
	partitions.append(t_obj_)
	
	return partitions


def place_iterative(match, t):
	"""
	Given a tile, it returns the templates within the tile obtained by iterative splitting.
	
	Parameters
	----------
	
		match: float
			Match defining the template volume
		
		t: tile
			The tile to cover with templates
	
	Returns
	-------
		new_templates: :class:`~numpy:numpy.ndarray`
			Array with the generated templates 
	"""
	dist = avg_dist(match, t.D)
	is_ok = lambda tile_: tile_.N_templates(dist)<=1
	
	template_list = [(t, is_ok(t))]
	
	while True:
		if np.all([b for _, b in template_list]): break
		for t_ in template_list:
			if t_[1]: continue
			t_left, t_right = t_[0].split(None,2)
			extended_list = [(t_left, is_ok(t_left)), (t_right, is_ok(t_right))]
			template_list.remove(t_)
			template_list.extend(extended_list)
	new_templates = np.array([t_.center for t_, _ in template_list])
	
	return new_templates

#@do_profile(follow=[])
def place_random(minimum_match, tiling, N_points, tolerance = 0.01, verbose = True):
	"""
	Given a tiling object, it covers the volume with points and covers them with templates.
	It follows `2202.09380 <https://arxiv.org/abs/2202.09380>`_
	
	Parameters
	----------
	
		minimum_match: float
			Minimum match between templates.
		
		tiling: tiling_handler
			Tiling handler that tiles the parameter space
		
		N_points: int
			Number of livepoints to cover the space with
		
		tolerance: float
			Fraction of livepoints to be covered before terminating the loop
		
		verbose: bool
			Whether to display the progress bar
	
	Returns
	-------
		new_templates: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			A set of templates generated by the placing algorithm
	"""
	#TODO: maybe here you can use the tiling KDTree for saving some useless computations?
	#i.e. you should add a tiling label to all the generated livepoints and use it somehow...
	
	assert 0<tolerance <=1., "The tolerance should be a fraction in (0,1]"
	assert isinstance(N_points, int) and N_points>0, "N_points should be a positive integer"
	
	MM = minimum_match
	dtype = np.float32 #better to downcast to single precision! There is a mild speedup there
	
		#FIXME: add here the sampling from flow option!
	livepoints, id_tile_livepoints = tiling.sample_from_tiling(N_points,
				dtype = dtype, tile_id = True, p_equal = False) #(N_points, D)
	
	if False:
		#sorting livepoint by their metric determinant...
		_, vols = tiling.compute_volume()
		v_list_livepoints = [np.linalg.det(tiling[i].metric) for i in id_tile_livepoints]
		id_sort = np.argsort(v_list_livepoints)[::-1]
	else:
		#random shuffling
		id_sort = np.random.permutation(N_points)

	livepoints = livepoints[id_sort, :]
	id_tile_livepoints = id_tile_livepoints[id_sort]
	det_list = [t_.det for t_ in tiling]
	det_livepoints = np.array([det_list[id_] for id_ in id_tile_livepoints])
	del det_list
	
		#ordering the tile by volume in ascending order...
	_, vols = tiling.compute_volume()
	
	def dummy_iterator():
		while True:
			yield
	
	new_templates = []
	
	bar_str = 'Loops on tiles ({}/{} livepoints killed | {} templates placed)'
	if verbose: it = tqdm(dummy_iterator(), desc = bar_str.format(N_points -len(livepoints), N_points, len(new_templates)), leave = True)
	else: it = dummy_iterator()
	
	for _ in it:
			#choosing livepoints in whatever order they are set
		id_point = 0
		#id_point = np.random.randint(len(livepoints))
		point = livepoints[id_point,:]
		id_ = id_tile_livepoints[id_point]
		
		if tiling.flow: metric = tiling.get_metric(point, flow = True) #using the flow if it is trained
		else: metric = tiling[id_].metric
		
		diff = livepoints - point #(N,D)
		#TODO: you may insert here a distance cutoff (like 4 or 10 in coordinate distance...): this should account for the very very large unphysical tails of the metric!!
		
				#measuring metric match between livepoints and proposal
				#Doing things with cholesky is faster but requires non degenerate matrix
		L_t = np.linalg.cholesky(metric).astype(dtype) #(D,D)
			#BLAS seems to be faster for larger matrices but slower for smaller ones...
			#Maybe put a threshold on the number of livepoints?
		diff_prime = scipy.linalg.blas.sgemm(1, diff, L_t)
		dist = np.sum(np.square(diff_prime), axis =1) #(N,) #This is the bottleneck of the computation, as it should be
		
		#match = 1 - np.sum(np.multiply(diff, np.matmul(diff, metric)), axis = -1) #(N,)
	
		ids_kill = np.where(dist < 1- MM)[0]
			#This variant kind of works although the way to go (probably) is to use normalizing flows to interpolate the metric
		#scaled_dist = dist * np.power(det_livepoints/np.linalg.det(metric), 1/tiling[0].D)		
		#ids_kill = np.where(np.logical_and(dist < 1- MM, scaled_dist < 1- MM) )[0]

			#This operation is very slow! But maybe there is nothing else to do...
		livepoints = np.delete(livepoints, ids_kill, axis = 0)
		id_tile_livepoints = np.delete(id_tile_livepoints, ids_kill, axis = 0)
		det_livepoints = np.delete(det_livepoints, ids_kill, axis = 0)
		
				#this is very very subtle: if you don't allocate new memory with np.array, you won't decrease the reference to livepoints, which won't be deallocated. This is real real bad!!
		new_templates.append(np.array(point, dtype = np.float64))
		del point
			
			#communication and exit condition
		if len(livepoints)<=tolerance*N_points: break
		if len(new_templates) %100 ==0 and verbose: it.set_description(bar_str.format(N_points -len(livepoints), N_points, len(new_templates)) )
	
	new_templates = np.column_stack([new_templates])
	#if len(livepoints)>0: new_templates = np.concatenate([new_templates, livepoints], axis =0) #adding the remaining livepoints
	
	return new_templates

def create_mesh(dist, tile, coarse_boundaries = None):
	"""
	Creates a mesh of points on an hypercube, given a metric.
	The points are approximately equally spaced with a distance ``dist``.
	
	Parameters
	----------
		dist: float
			Distance between templates
		
		tile: tuple
			An element of the ``tiling_handler`` object.
			It consists of a tuple ``(scipy.spatial.Rectangle, np.ndarray)``
	
		coarse_boundaries: :class:`~numpy:numpy.ndarray`
			shape: (2,D) -
			An array with the coarse boundaries of the tiling.
			If given, each tile is checked to belong to the border of the tiling. If it's the case, some templates are added to cover the boundaries

	Returns
	-------
		mesh: :class:`~numpy:numpy.ndarray`
			shape: (N,D) - 
			A mesh of N templates that cover the tile
	"""
	#dist: float
	#metric: (D,D)
	#boundaries (2,D)
	D = tile[0].maxes.shape[0]
	
		#bound_list keeps the dimension over which the tile is a boundary in the larger space
	if D < 2: coarse_boundaries = None
	if coarse_boundaries is not None:
		up_bound_list = np.where( np.isclose(tile[0].maxes, coarse_boundaries[1,:], 1e-4, 0) )[0].tolist() #axis where there is an up bound
		low_bound_list = np.where( np.isclose(tile[0].mins, coarse_boundaries[0,:], 1e-4, 0) )[0].tolist()
		bound_list = [ (1, up_) for up_ in up_bound_list]
		bound_list.extend( [ (0, low_) for low_ in low_bound_list])
	else: bound_list = []
	
		#Computing cholesky decomposition of the metric	
	metric = tile[1]
	L = np.linalg.cholesky(metric).T
	L_inv = np.linalg.inv(L)
	
		#computing boundaries and boundaries_prime
	boundaries = np.stack([tile[0].mins, tile[0].maxes], axis = 0)	
	corners = get_cube_corners(boundaries)#[0,:], boundaries[1,:])
	corners_prime = np.matmul(corners, L.T)
	center = (tile[0].mins+tile[0].maxes)/2. #(D,) #center
	center_prime = np.matmul(L, center) #(D,) #center_prime
	
		#computing the extrema of the boundaries (rectangle)
	boundaries_prime = np.array([np.amin(corners_prime, axis =0), np.amax(corners_prime, axis =0)])
	
		#creating a mesh in the primed coordinates (centered around center_prime)
	mesh_prime = []
	where_random = [] #list to keep track of the dimensions where templates should be drawn at random!
	
	for d in range(D):
		min_d, max_d = boundaries_prime[:,d]
		
		N = max(int((max_d-min_d)/dist), 1)
			#this tends to overcover...
		#grid_d = [np.linspace(min_d, max_d, N+1, endpoint = False)[1:]] 
			 #this imposes a constant distance but may undercover
		grid_d = [np.arange(center_prime[d], min_d, -dist)[1:][::-1], np.arange(center_prime[d], max_d, dist)]

		grid_d = np.concatenate(grid_d)
		
		if len(grid_d) <=1 and d >1: where_random.append(d)
		
		mesh_prime.append(grid_d)
		
		#creating the mesh in the primed space and inverting
	mesh_prime = np.meshgrid(*mesh_prime)
	mesh_prime = [g.flatten() for g in mesh_prime]
	mesh_prime = np.column_stack(mesh_prime) #(N,D)
	
	mesh = np.matmul(mesh_prime, L_inv.T)

		#we don't check the boundaries for the axis that will be drawn at random
	axis_ok = [i for i in range(D) if i not in where_random]
	ids_ok = np.logical_and(np.all(mesh[:,axis_ok] >= boundaries[0,axis_ok], axis =1), np.all(mesh[:,axis_ok] <= boundaries[1,axis_ok], axis = 1)) #(N,)
	mesh = mesh[ids_ok,:]

	
		#Whenever there is a single point in the grid, the templates along that dimension will be placed at random
	for id_random in where_random:
		mesh[:,id_random] =np.random.uniform(boundaries[0,id_random], boundaries[1,id_random], (mesh.shape[0], )) # center[id_random] #
	#warnings.warn('Random extraction for "non-important" dimensions disabled!')
	return mesh
		####
		#adding the boundaries
		####
		
		#Boundaries are added by creating a mesh in the D-1 plane of the tile boundary
	boundary_mesh = []
		#up_down keeps track whether we are at the min (0) or max (1) value along the d-th dimension
	for up_down, d in bound_list:
		ids_not_d = [d_ for d_ in range(D) if d_ is not d]
		new_dist = dist*np.sqrt(D/(D-1)) #this the distance between templates that must be achieved in the low dimensional manifold
		
			#creating the input for the boundary tiling
		rect_proj = Rectangle( tile[0].mins[ids_not_d], tile[0].maxes[ids_not_d]) #projected rectangle
		metric_proj = metric - np.outer(metric[:,d], metric[:,d]) /metric[d,d]
		metric_proj = metric_proj[tuple(np.meshgrid(ids_not_d,ids_not_d))].T #projected metric on the rectangle
		
		new_coarse_boundaries = np.stack([rect_proj.mins, rect_proj.maxes], axis =0) #(2,D)
		#new_coarse_boundaries = None
		new_mesh_ = create_mesh(new_dist, (rect_proj, metric_proj), new_coarse_boundaries) #(N,D-1) #mesh on the D-1 plane
		boundary_mesh_ = np.zeros((new_mesh_.shape[0], D))
		boundary_mesh_[:,ids_not_d] = new_mesh_
		boundary_mesh_[:,d] = boundaries[up_down,d]
		
		boundary_mesh.extend(boundary_mesh_)
		
	if len(boundary_mesh)>0:
		boundary_mesh = np.array(boundary_mesh)
		mesh = np.concatenate([mesh,boundary_mesh], axis =0)
	
	return mesh

###########################################################################################

#All the garbage here should be removed!!

def points_in_hull(points, hull, tolerance=1e-12):
	#"Check if points (N,D) are in the hull"
	if points.ndim == 1:
		points = point[None,:]
	
	value_list = [np.einsum('ij,j->i', points, eq[:-1])+eq[-1] for eq in hull.equations]
	value_list = np.array(value_list).T #(N, N_eqs)
	
	return np.prod(value_list<= tolerance, axis = 1).astype(bool) #(N,)

def all_line_hull_intersection(v, c, hull):
	#"Compute all the intersection between N_lines and a single hull"
	if c.ndim == 1:
		c = np.repeat(c[None,:], v.shape[0], axis =0)

	eq=hull.equations.T
	n,b=eq[:-1],eq[-1] #(N_faces, D), (N_faces,)
	
	den = np.matmul(v,n)+1e-18 #(N_lines, N_faces)
	num = np.matmul(c,n) #(N_lines, N_faces)
	
	alpha= -(b +num )/den #(N_lines, N_faces)

		#v (N_lines, D)
	res = c[:,None,:] + np.einsum('ij,ik->ijk', alpha,v) #(N_lines, N_faces, D)

	return res.reshape((res.shape[0]*res.shape[1], res.shape[2]))

def sample_from_hull(hull, N_points):
	#"Sample N_points from a convex hull"
	dims = hull.points.shape[-1]
	del_obj = scipy.spatial.Delaunay(hull.points) #Delaunay triangulation obj
	deln = hull.points[del_obj.simplices] #(N_triangles, 3, dims)
	vols = np.abs(np.linalg.det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)	
	sample = np.random.choice(len(vols), size = N_points, p = vols / vols.sum()) #Decide where to sample (Gibbs sampling)
	samples = np.einsum('ijk, ij -> ik', deln[sample], scipy.stats.dirichlet.rvs([1]*(dims + 1), size = N_points))

	if False:
		plt.figure()
		plt.triplot(hull.points[:,0], hull.points[:,1], del_obj.simplices) #plot delaneuy triangulation
		plt.scatter(*samples.T, s = 2)
		plt.show()
	
	return samples

#@do_profile(follow=[])
def sample_from_hull_boundaries(hull, N_points, boundaries = None, max_iter = 1000):
	#"SamplesN_points from a convex hull. If boundaries are given, it will enforce them"
	dims = hull.points.shape[1]
	del_obj = scipy.spatial.Delaunay(hull.points)
	deln = hull.points[del_obj.simplices]
	vols = np.abs(np.linalg.det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)
	
	samples_list = []
	N_samples = 0
	
		#100 is the max number of iterations, after them we break
	for i in range(max_iter):
		sample = np.random.choice(len(vols), size = N_points, p = vols / vols.sum())
		print(sample, sample.shape,  deln[sample].shape)
		samples = np.einsum('ijk, ij -> ik', deln[sample], scipy.stats.dirichlet.rvs([1]*(dims + 1), size = N_points))
	
		if boundaries is not None:
			ids_ok = np.logical_and(np.all(samples > boundaries[0,:], axis =1), np.all(samples < boundaries[1,:], axis = 1)) #(N,)
			samples = samples[ids_ok,:]	#enforcing boundaries on masses and spins
			samples = samples[np.where(samples[:,0]>=samples[:,1])[0],:] #enforcing mass cut

		if samples.shape[0]> 0:
			samples_list.append(samples)
			N_samples += samples.shape[0]

		if N_samples >= N_points: break
	
	if len(samples_list)>0:
		samples = np.concatenate(samples_list, axis =0)
	else:
		samples = None

	return samples

def plot_hull(hull, points = None):
	#"Plot the hull and a bunch of additional points"
	plt.figure()
	for simplex in hull.simplices:
		plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'g-')
	plt.scatter(*hull.points[:,:2].T, alpha = 0.1, c = 'y', marker = 'o')
	if points is not None:
		plt.scatter(*points[:,:2].T, alpha = 0.3, c = 'r', marker = 's')
	plt.plot([10,100],[10,100])
	plt.show()
	return

def get_pad_points_2d(N_grid, boundaries):
	#"Get N points padding the boundary box"
	m1, M1 = boundaries[:,0]
	m2, M2 = boundaries[:,1]
	m, M = min(m1,m2), max(M1, M2)

		#creating points in 2D
	new_points = [*np.column_stack([M1*1.5*np.ones((N_grid,)), np.linspace(m2, M2,N_grid)])]
	new_points = new_points + [*np.column_stack([ np.linspace(m1, M1, N_grid), m2*0.2*np.ones((N_grid,))])]
	new_points = new_points + [*np.column_stack([ np.linspace(m, M, N_grid), np.linspace(m,M, N_grid)*1.5])]
	new_points = new_points + [np.array([.1,.1])]
		#new_points keeps a 2D grid over the mass boundaries
	new_points_2D = np.array(new_points)

	return new_points_2D

def get_pad_points(N_grid, boundaries):
	#"Get N points padding the boundary box"
	#FIXME: this function is shit
	m1, M1 = boundaries[:,0]
	m2, M2 = boundaries[:,1]
	m, M = min(m1,m2), max(M1, M2)

		#creating points in 2D
	new_points = [*np.column_stack([M1*1.52*np.ones((N_grid,)), np.linspace(m2, M2,N_grid)])]
	new_points = new_points + [*np.column_stack([ np.linspace(m1, M1, N_grid), m2*0.18*np.ones((N_grid,))])]
	new_points = new_points + [*np.column_stack([ np.linspace(m, M, N_grid), np.linspace(m,M, N_grid)*1.52])]
	new_points = new_points + [np.array([1,1])]
		#new_points keeps a 2D grid over the mass boundaries
	new_points_2D = np.array(new_points)
	
		#This is to add the rest: it's super super super super super ugly
	if boundaries.shape[1] > 2:
		new_points_list = []

			#creating grids
		s1_grid = 							 np.linspace(boundaries[0,2]*0.8, boundaries[1,2]*1.2, N_grid)
		if boundaries.shape[1] >3: s2_grid = np.linspace(boundaries[0,3]*0.8, boundaries[1,3]*1.2, N_grid)
		else: s2_grid = [0]
		if boundaries.shape[1] >4: s3_grid = np.linspace(boundaries[0,4]*0.8, boundaries[1,4]*1.2, N_grid)
		else: s3_grid = [0]
		if boundaries.shape[1] >5: s4_grid = np.linspace(boundaries[0,5]*0.8, boundaries[1,5]*1.2, N_grid)
		else: s4_grid = [0]
		if boundaries.shape[1] >6: s5_grid = np.linspace(boundaries[0,6]*0.8, boundaries[1,6]*1.2, N_grid)
		else: s5_grid = [0]
	
			#the super ugly way: there is a better way
		for s1 in s1_grid:
			for s2 in s2_grid:
				for s3 in s3_grid:
					for s4 in s4_grid:
						for s5 in s5_grid:
							temp = np.concatenate([new_points_2D, np.repeat([[s1,s2,s3,s4,s5]], new_points_2D.shape[0], axis =0) ], axis = 1)
							new_points_list.append(temp)
		new_points = np.concatenate(new_points_list, axis = 0) #(N,D)
		new_points = new_points[:,:boundaries.shape[1]]
	else:
		new_points = new_points_2D
				

	return new_points

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
	Splits a boundary rectangle by dividing each dimension into a number of evenly spaced segments defined by each entry `grid_list`.
	
	If option `plawspace` is set, the segments will be evenly distributed acconrding to a pwer law with exponent `-8/3`
	
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
			A list of boundaries arrays obtained after the splitting, each with shape `(2,D)`
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
	boundaries_list = [(low, up) for low, up in zip(lower_boxes, upper_boxes) ]
	return boundaries_list

##########################################################################################
#TODO: use this function every time you read an xml!!!
def read_xml(filename, table, N = None):
	"""
	Read an xml file in the ligo.lw standard and extracts the BBH parameters
	
	Parameters
	----------
		filename: str
			Name of the file to load
		
		table: ligo.lw.lsctables.table.Table
			A ligo.lw table type. User typically will want to set `ligo.lw.lsctables.SnglInspiralTable` for a bank and `ligo.lw.lsctables.SimInspiralTable` for injections
		
		N: int
			Number of rows to be read. If `None` all the rows inside the table will be read
	
	Returns
	-------
		BBH_components: :class:`~numpy:numpy.ndarray`
			shape (N,12) -
			An array with the read BBH components. It has the same layout as in `mbank.handlers.variable_handler`
	"""
	@lsctables.use_in
	class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
		pass
	lsctables.use_in(LIGOLWContentHandler)

	xmldoc = lw_utils.load_filename(filename, verbose = False, contenthandler = LIGOLWContentHandler)
	table = table.get_table(xmldoc)
	
	if not isinstance(N, int): N = len(table)
	BBH_components = []
			
	for i, row in enumerate(table):
		if i>=N: break
		#FIXME: read e and meanano properly!!
		BBH_components.append([row.mass1, row.mass2, #masses
			row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z, #spins
			0, 0, row.alpha3, row.alpha5]) #e, meananano, iota, phi
			
	BBH_components = np.array(BBH_components) #(N,12)
	
	return BBH_components
		
def save_injs(filename, injs, GPS_start, GPS_end, time_step, approx, luminosity_distance = 100, f_min = 10., f_max = 1024.):
		"""
		Save the given injections to a ligo xml injection file (sim_inspiral table).
		
		Parameters
		----------
			
		filename: str
			Filename to save the injections at
		
		injs: :class:`~numpy:numpy.ndarray`
			Injection array (N,12)
		
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
		if isinstance(luminosity_distance, float): luminosity_distance = (luminosity_distance, luminosity_distance*1.001)
		else: assert isinstance(luminosity_distance, tuple), "Wrong format for luminosity distance. Must be a float or a tuple of float, not {}".format(type(luminosity_distance))
		
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

			if i>=injs.shape[0]:
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

				#setting interesting row paramters
			row.inclination = iota
			row.coa_phase = phi
			row.polarization = np.random.uniform(0.0, 2.0 * np.pi)
			row.longitude = np.random.uniform(0.0, 2.0 * np.pi)
			row.latitude = np.arcsin(np.random.uniform(-1.0, 1.0)) #FIXME: check whether this is correct!
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
		lw_utils.write_filename(xmldoc, filename, gz=filename.endswith('.xml.gz'), verbose=False)
		xmldoc.unlink()

		print("Saved {} injections to {}".format(i+1, filename))

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

















	
	
	
