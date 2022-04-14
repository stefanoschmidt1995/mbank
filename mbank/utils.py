"""
mbank.utils
===========
	Some utilities for ``mbank``.
	It keeps function for plotting, for template placing and for match computation.

	#FIXME: this may be split in two parts as some functions need to import the tiling handler
	#They are `plot_tiles_templates` and `get_boundaries_from_ranges`
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

#from .handlers import variable_handler

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
	Updates the arguments of Namespace args according to the 

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

def get_boundaries_from_ranges(format_info, M_range, q_range,
	s1_range = (-0.99,0.99), s2_range = (-0.99,0.99), theta_range = (0, np.pi), phi_range = (-np.pi, np.pi),
	iota_range = (0, np.pi), ref_phase_range = (-np.pi, np.pi), e_range = (0., 0.5), meanano_range = (0.1, 1.)):
	"""
	Given the ranges of each quantity, it combines them in a bondary array, suitable for other uses in the package (for instance in the bank generation).
	No checks are performed whether the given ranges make sense.
	
	Parameters
	----------
		format_info: dict
			Dict holding the format information
			It can be generated with: `variable_handler().format_info[variable_format]`
		
		M_range, q_range, s1_range, s2_range, theta_range, phi_range, iota_range, ref_phase_range, e_range, meanano_range: tuple
			Ranges for each physical quantity. They will be used whenever required by the `variable_format`
			If `mchirpeta` mass format is set, `M_range` and `q_range` are interpreted as mchirp and eta respectively.
	
	Returns
	-------
		boundaries: np.ndarray
			shape (2,D) -
			An array with the boundaries.
	"""
	######
	#	Setting boundaries: shape (2,D)
	######
	if format_info['spin_format'].find('1x') >-1 and s1_range[0]<0.:
		s1_range = (0, s1_range[1])
	if format_info['spin_format'] == 'fullspins':
		if s1_range[0]< 0: s1_range = (0, s1_range[1])
		if s2_range[0]< 0: s2_range = (0, s2_range[1])
		
		#setting spin boundaries
	if format_info['spin_format'] == 'nonspinning':
		boundaries = np.array([[M_range[0], q_range[0]],[M_range[1], q_range[1]]])
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
		boundaries = np.concatenate([boundaries, [[e_range[0]], [e_range[0]]]], axis =1)
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
		f: np.ndarray
			Frequency grid

		PSD: np.ndarray
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
def compute_avg_metric_dist(bank, tiling, N_neigh_templates = 10, metric_obj = None):
	"""
	Given a tiling and a bank, it computes the distance of the nearest neighbour for each template.
	It is a wrapper to ``compute_metric_injections_match``.
	
	Parameters
	----------
		
		bank: cbc_bank
			``cbc_bank`` object holding an initialized bank

		tiling: tiling_handler
			``tiling_handler`` object with a valid tiling
		
		N_neigh_templates: int
			Number of neighbours templates to consider
		
		metric_obj: cbc_metric
			A cbc_metric object to compute the match with. If None, only the metric approximation will be used
		
	Returns
	-------
			
		out_dict: dict
			A dictionary with the output. It has the same format of ``compute_metric_injections_match``
	"""
	out_dict = compute_metric_injections_match(bank.templates, bank, tiling, N_neigh_templates)

	if metric_obj is not None:
		templates = np.array(bank.var_handler.get_BBH_components(bank.templates, bank.variable_format)).T
		metric_obj.set_variable_format('m1m2_fullspins_emeanano_iotaphi')
		out_dict = compute_injections_match(out_dict, templates, metric_obj, N_neigh_templates = N_neigh_templates, symphony_match = False, cache = True, worker_id = None)

	return out_dict

def ray_compute_injections_match(inj_dict, templates, metric_obj, N_neigh_templates = 10, symphony_match = False, cache = True):
	"""
	Wrapper to ``compute_injections_match()`` to allow for parallel execution. It calls ``_compute_injections_match_ray()``
	Given an injection dictionary, generated by ``compute_metric_injections_match()`` it computes the actual match (without the metric approximation) between injections and templates. It updates ``inj_dict`` with the new computed results.
	The injections are generic (not necessarly projected on the bank submanifold).
	
	Parameters
	----------
		inj_dict: dict
			A dictionary with the data injection as computed by `compute_metric_injections_match`.
		
		templates: np.ndarray
			An array with the templates. They should have the same layout as lal (given by get_BBH_components)

		metric_obj: cbc_metric
			A cbc_metric object to compute the match with.

		N_neigh_templates: 'int'
			The number of neighbouring templates to consider for each injection
						
		cache: bool
			Whether to cache the WFs
			
		symphony_match: bool
			Whether to use the symphony match
		
	Returns
	-------
		out_dict: 'dict'
			The output dictionary with the updated matches
	"""
		###
		# Split injections
	n_injs_per_job = min(500, int(inj_dict['match'].shape[0]/5))
	
		###
		# Initializing ray and performing the computation
	inj_dict_ray_list = []
	ray.init()
	for id_, i in enumerate(range(0, inj_dict['match'].shape[0], n_injs_per_job)):
		inj_dict_ray_list.append( _compute_injections_match_ray.remote(i, i+n_injs_per_job, inj_dict,
			templates, metric_obj, N_neigh_templates, symphony_match, cache, id_))
	inj_dict_ray_list = ray.get(inj_dict_ray_list)
	ray.shutdown()
	
		###
		# Concatenating the injections
	inj_dict = {}
	for k in inj_dict_ray_list[0].keys():
		if isinstance(inj_dict_ray_list[0][k], np.ndarray):
			inj_dict[k] = np.concatenate([inj_dict_[k] for inj_dict_ in inj_dict_ray_list ])		
		else:
			inj_dict[k] = None	
	
	return inj_dict


@ray.remote
def _compute_injections_match_ray(start_id, end_id, inj_dict, templates, metric_obj, N_neigh_templates = 10, symphony_match = False, cache = True, worker_id = None):
	"""
	Wrapper to ``compute_injections_match`` to allow for parallelization with ray.

	Parameters
	----------
		inj_dict: dict
			A dictionary with the data injection as computed by `compute_metric_injections_match`.
		
		templates: np.ndarray
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
			Id of the ray worker being used. If None, it is assumed that ray is not called
		
	Returns
	-------
		out_dict: dict
			The output dictionary with the updated matches
	"""
	local_dict = {} #ray wants a local dict for some reason...

	for k in inj_dict.keys():
		if isinstance(inj_dict[k], np.ndarray):
			local_dict[k] = np.copy(inj_dict[k][start_id:end_id])
		else:
			local_dict[k] = None
	return compute_injections_match(local_dict, templates, metric_obj, N_neigh_templates, symphony_match, cache, worker_id)

def compute_injections_match(inj_dict, templates, metric_obj, N_neigh_templates = 10, symphony_match = False, cache = True, worker_id = None):
	#TODO: out_dict should have two entries, one for metric match and the other one for full match
	"""
	Given an injection dictionary, generated by ``compute_metric_injections_match`` it computes the actual match (not the metric approximation) between injections and templates. It updates ``inj_dict`` with the new computed results.
	The injections are generic (not necessarly projected on the bank submanifold).
	
	Parameters
	----------
		inj_dict: 'dict'
			A dictionary with the data injection as computed by `compute_metric_injections_match`.
		
		templates: np.ndarray
			An array with the templates. They should have the same layout as lal (given by get_BBH_components)

		metric_obj: cbc_metric
			A cbc_metric object to compute the match with.

		N_neigh_templates: int
			The number of neighbouring templates to consider for each injection
		
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
	#assert metric_obj.variable_format == 'm1m2_fullspins_emeanano_iotaphi', "Wrong variable format given. It must be 'm1m2_fullspins_emeanano_iotaphi'"

	if N_neigh_templates > templates.shape[0]:
		N_neigh_templates = templates.shape[0]

		#putting injections and templates with the format 'm1m2_fullspins_emeanano_iotaphi'
		# The format is basically the full 12 dimensional space, with spins in spherical coordinates
	injs = metric_obj.var_handler.get_theta(inj_dict['theta_inj'], 'm1m2_fullspins_emeanano_iotaphi')
	templates = metric_obj.var_handler.get_theta(templates, 'm1m2_fullspins_emeanano_iotaphi')

		#Only the first N_neigh_templates are considered: this can be different from the one used for the metric match
	inj_dict['id_match_neig'] = inj_dict['id_match_neig'][:,:N_neigh_templates]
	inj_dict['match_neig'] = inj_dict['match_neig'][:,:N_neigh_templates]
	
	if cache:
		unique_ids = inj_dict['id_match_neig'].flatten()
		unique, counts = np.unique(unique_ids, return_counts=True)
		ids_ok = np.where(counts>1)[0]
		dict_ids = dict(zip(unique[ids_ok], counts[ids_ok])) #{'tmplt_id': counts}
		#print(dict_ids)
		dict_WFs = {}
		if len(dict_ids)<1:
			cache = None #if dict_ids is empty, no cache is needed

	
	if worker_id is None: desc = 'Computing the {} match: loop on the injections'.format('symphony' if symphony_match else 'std')
	else: desc = 'Worker {} - Computing the {} match: loop on the injections'.format(worker_id, 'symphony' if symphony_match else 'std')
	
	for i in tqdm(range(injs.shape[0]), desc = desc, leave = False):
		#Take a look at https://pypi.org/project/anycache/
		template_ = templates[inj_dict['id_match_neig'][i,:],:] #(N_neigh_templates, 10)

		if cache:
				#dealing with cached WFs
			template_WF = []
			for id_ in inj_dict['id_match_neig'][i,:]:
				if id_ in dict_WFs:
					new_WF = dict_WFs[id_]
				else:
					new_WF = metric_obj.get_WF(templates[id_,:])
				
				if id_ in dict_ids:
						#saving to cache
					dict_WFs[id_] = new_WF
					dict_ids[id_] = dict_ids[id_]-1
					if dict_ids[id_] == 0: #removing from cache
						dict_WFs.pop(id_)
						dict_ids.pop(id_)

				template_WF.append(new_WF)
			
			template_WF = np.stack(template_WF, axis =0) #(N_neigh_templates, D)
			inj_WF = metric_obj.get_WF(injs[i]) #(D,)
			true_match = metric_obj.WF_match(template_WF, inj_WF) #(N_neigh_templates,)
		else:
			true_match = metric_obj.match(template_, injs[i], symphony = symphony_match, overlap = False)

			#updating the dict
		ids_max = np.argmax(true_match)

		inj_dict['match_neig'][i,:] = true_match #(N_neigh_templates,)
		inj_dict['id_match'][i] = inj_dict['id_match_neig'][i, np.argmax(inj_dict['match_neig'][i,:])]
		inj_dict['match'][i] = np.max(inj_dict['match_neig'][i,:])

	metric_obj.set_variable_format(old_format)

	return dict(inj_dict)


####################################################################################################################	
#@do_profile(follow=[])
def compute_metric_injections_match(injs, bank, tiling, N_neigh_templates = 10, verbose = True):
	#FIXME: this is a shitty name: it should be compute_injections_metric_match
	"""
	Computes the match of the injection with the bank by using the metric approximation.
	It makes use of a brute force approach where each injection is checked against each template of the bank. The metric used is the one of the tile each injection belongs to
	
	Parameters
	----------
		injs: np.ndarray
			shape: (N,12)/(N,D) -
			A set of injections.
			If the shape is `(N,D)` they are assumed to be the lie in the bank manifold.
			Otherwise, they are in the full format of the ``mbank.metric.cbc_metric.get_BBH_components()``.
			Each row should be: m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi
		
		bank: cbc_bank
			A ``cbc_bank`` object

		N_neigh_templates: int
			Number of nearest neigbour to report in the output
		
		verbose: bool
			Whether to print the output
		
	Returns
	-------
		out_dict: dict
			A dictionary with the output. It has the entries:
			
			- ``theta_inj``: the parameters of the injections
			- ``id_tile``: index of the tile the injections belongs to (in the tiling)
			- ``match``: match of the closest template
			- ``id_match``: index of the closest template
			- ``match_neig``: match for the ``N_neigh_templates`` nearest neighbours
			- ``id_match_neig``: index of the ``N_neigh_templates`` nearest neighbours
		
			each entry is ``np.ndarray`` where each row is an injection.
	
	"""
	#TODO: apply a cutoff on the distances... otherwise you could get crazy values!!
	if N_neigh_templates > bank.templates.shape[0]:
		N_neigh_templates = bank.templates.shape[0]

		#storing injections in the full format (so that they can be generic)
	out_dict = {'theta_inj': np.array(bank.var_handler.get_BBH_components(injs, bank.variable_format)).T if injs.shape[1] != 12 else injs ,
				'id_tile': np.zeros((injs.shape[0],), int),
				'match': np.zeros((injs.shape[0],)), 'id_match':  np.empty((injs.shape[0],), dtype = int),
				'match_neig': np.zeros((injs.shape[0], N_neigh_templates)),
				'id_match_neig': np.empty((injs.shape[0], N_neigh_templates), dtype = int)
				}
	
		#csating the injections to the metric type
	if injs.shape[1] == 12:
		injs = bank.var_handler.get_theta(injs, bank.variable_format)
	if injs.shape[1] != bank.D:
		raise ValueError("Wrong input size for the injections")
	
	template_dist = np.allclose(bank.templates, injs) if (bank.templates.shape == injs.shape) else False
	N_argpartiton = 2000
	id_diff_ok = np.arange(bank.templates.shape[0])
	
		#loops on the injections
	if verbose: inj_iter = tqdm(range(injs.shape[0]), desc = 'Evaluating metric match for injections', leave = True)
	else: inj_iter = range(injs.shape[0])
	
	for i in inj_iter:

		out_dict['id_tile'][i] = tiling.get_tile(injs[i])[0]

		diff = bank.templates - injs[i] #(N_templates, D)
		
			#these are the indices being checked
		if N_argpartiton < bank.templates.shape[0]:
			id_diff_ok = np.argpartition(np.linalg.norm(diff, axis=1), N_argpartiton)[:N_argpartiton]
		
		metric = tiling[out_dict['id_tile'][i]][1]
		L_t = np.linalg.cholesky(metric) #(D,D)
		match_i = 1- np.sum(np.square(np.matmul(diff[id_diff_ok], L_t)), axis =1)
		
		#match_i = 1 - np.einsum('ij, jk, ik -> i', diff[id_diff_ok], metric, diff[id_diff_ok]) #old, slower version
		
		indices_i = np.argpartition(match_i, -N_neigh_templates)[-N_neigh_templates:]
		
			#sorting
		ids_sort = np.argsort(-match_i[indices_i])
		out_dict['match_neig'][i,:] = match_i[indices_i][ids_sort]
		out_dict['id_match_neig'][i,:] = id_diff_ok[indices_i[ids_sort]] #indices_i[ids_sort]
		
			#id = 0 if not template dist, 1 if template dist
		out_dict['id_match'][i] = out_dict['id_match_neig'][i, int(template_dist)]
		out_dict['match'][i] = out_dict['match_neig'][i, int(template_dist)]

	#removing the first match in the case of template distance
	out_dict['id_match_neig'] = out_dict['id_match_neig'][:, int(template_dist):]
	out_dict['match_neig'] = out_dict['match_neig'][:, int(template_dist):]
		
	return out_dict

####################################################################################################################
def get_ellipse(metric, center, dist, **kwargs):
	"""
	Given a two dimensional metric and a center, it returns the `matplotlib.Patch` that represent the points at constant distance `dist` according to the metric.
	It accepts as an additional parameter, anything that can be given to `matplotlib.patches.Ellipse`.
	
	Parameters
	----------
		metric: np.ndarray
			shape: (2,2) - 
			A two dimensional metric
		
		center: np.ndarray
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


def plot_tiles_templates(t_obj, templates, variable_format, var_handler, injections = None, inj_cmap = None, dist_ellipse = None, save_folder = None, show = False):
	"""
	Make some plots of the templates and the tiling.
		
	Parameters
	----------
		t_obj: tiling_handler
			Tiling handler that tiles the parameter space
		
		templates: np.ndarray
			shape: (N,D) -
			The templates to plot, as stored in ``cbc_bank.templates``
		
		variable_format: str
			How to handle the BBH variables.	
			
		var_handler: variable_handler
			A variable handler object to handle the labels. Can be instantiated with `mbank.handlers.variable_handler()`
		
		injections: np.ndarray
			shape: (N,D) -
			An extra set of injections to plot. If `None`, no extra points will be plotted.
		
		inj_cmap: np.ndarray
			shape: (N,) -
			A colouring value for each injection, tipically the match with the bank. If None, no colouring will be done.
			The argument is ignored if `injections = None`
		
		dist_ellipse: float
			The distance for the match countour ellipse to draw. If `None`, no contour will be drawn.
		
		save_folder: str
			Folder where to save the plots
			If `None`, no plots will be saved
		
		show: bool
			Wheter to show the plots

	"""
	#var_handler = variable_handler() #old
		###
		#Plotting templates
		###
	if isinstance(save_folder, str): 
		if not save_folder.endswith('/'): save_folder.out_dir = save_folder.out_dir+'/'
	fs = 15 #font size
	
	if isinstance(dist_ellipse, float): #computing a tile for each template
		dist_template = []
		for t in t_obj:
			dist_template.append( t[0].min_distance_point(templates) ) #(N_templates,)
		dist_template = np.stack(dist_template, axis = 1) #(N_templates, N_tiles)
		id_tile_templates = np.argmin(dist_template, axis = 1) #(N_templates,)
		del dist_template
	
		###
		#Plotting templates & tiles
		###
	if templates.shape[0] >500000: ids_ = np.random.choice(templates.shape[0], 500000, replace = False)
	else: ids_ = range(templates.shape[0])
	
	size_template = [20 if templates.shape[0] < 10000 else 2][0]
	centers = t_obj.get_centers()
	fig, axes = plt.subplots(templates.shape[1]-1, templates.shape[1]-1, figsize = (15,15))
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
	plt.suptitle('Templates + tiling of the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
	for ax_ in combinations(range(templates.shape[1]), 2):
		currentAxis = axes[ax_[1]-1, ax_[0]]
		ax_ = list(ax_)
		currentAxis.scatter(*centers[:,ax_].T, s = 30, marker = 'x', c= 'r', alpha = 1)
		for t in t_obj:
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
			fig.colorbar(cbar_vals, cax=cbar_ax)
		if isinstance(save_folder, str): plt.savefig(save_folder+'injections.png', transparent = False)


		#Plotting the ellipses, if it is the case
	if isinstance(dist_ellipse, float):
		plt.suptitle('Templates + tiling + ellipses of the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
		for ax_ in combinations(range(templates.shape[1]), 2):
			currentAxis = axes[ax_[1]-1, ax_[0]]
			ax_ = list(ax_)
			for i, templ in enumerate(templates):
				metric_projected = project_metric(t_obj[id_tile_templates[i]][1], ax_)
				currentAxis.add_patch(get_ellipse(metric_projected, templ[ax_], dist_ellipse))
			#if ax_[0]!=0: currentAxis.set_xlim([-10,10]) #DEBUG
			#currentAxis.set_ylim([-10,10]) #DEBUG
		if isinstance(save_folder, str): plt.savefig(save_folder+'ellipses.png', transparent = False)
	
		#Plot an histogram
	fig, axes = plt.subplots(1, templates.shape[1], figsize = (4*templates.shape[1], 5), sharey = True)
	plt.suptitle('Histograms for the bank: {} points'.format(templates.shape[0]), fontsize = fs+10)
	hist_kwargs = {'bins': min(50, int(len(ids_)/50 +1)), 'histtype':'step', 'color':'orange'}
	for i, ax_ in enumerate(axes):
		ax_.hist(templates[ids_,i], **hist_kwargs)
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
		metric: np.ndarray
			shape: (D,D) - 
			A D dimensional metric
		
		axes: list
			The D' axes to project the metric over
	
	Returns
	-------
		projected_metric: np.ndarray
			shape: (D',D') - 
			The projected dimensional metric

	"""

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
		metric: np.ndarray
			shape: (D,D)/(N,D,D) - 
			A D dimensional metric
		
		min_eig: float
			The minimum value for the eigenvalues. The metric will be changed accordingly
	
	Returns
	-------
		trimmed_metric: np.ndarray
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


####################################################################################################################

def get_cube_corners(boundaries):
	"""
	Given the boundaries of an hyper-rectangle, it computes all the corners of it
	
	Parameters
	----------
		boundaries: np.ndarray
			shape: (2,D) -
			An array with the boundaries for the model. Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
	
	Returns
	-------
		corners: np.ndarray
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

def place_stochastically_in_tile(avg_dist, tile):
	"""
	Place templates with a stochastic placing algorithm withing a given tile.
	It iteratively proposes a new template to add to the bank inside the given tile. The proposal is accepted if the distance of the proposal with the previously placed templates is smaller than ``avg_dist``. The iteration goes on until no template is found to have a distance smaller than the given threshold ``avg_dist``. 
	
	
	Parameters
	----------
		avg_dist: float
			Average distance between templates
		
		tile: tuple
			An element of the ``tiling_handler`` object.
			It consists of a tuple ``(scipy.spatial.Rectangle, np.ndarray)``
	
	Returns
	-------
		new_templates: np.ndarray
			shape: (N,D) -
			A set of templates generated by the stochastic placing algorithm within the given tile
	"""
	dist_sq = avg_dist**2

	metric = tile[1]
	L = np.linalg.cholesky(metric).T
	
	new_templates = np.random.uniform(tile[0].mins, tile[0].maxes, (1, tile[0].maxes.shape[0])) #(1,D)
	new_templates_prime = np.matmul(new_templates, L.T) #(1,D)
	
		#computing the boundaries (only on first 2 variables)
	boundaries = np.stack([tile[0].mins, tile[0].maxes], axis = 0)	
	corners = get_cube_corners(boundaries)#[0,:], boundaries[1,:])
	corners_prime = np.matmul(corners, L.T)[:,[0,1]]
	hull = ConvexHull(corners_prime)
	
	nothing_new = 0
	while nothing_new < 300:
		proposal = np.random.uniform(tile[0].mins, tile[0].maxes, tile[0].maxes.shape) #(D,)
		proposal_prime = np.matmul(L, proposal) #(D,)
		
			#dist from boundaries of convex hull: https://stackoverflow.com/questions/41000123/computing-the-distance-to-a-convex-hull
		dist_boundaries = np.max(np.dot(hull.equations[:, :-1], proposal_prime[[0,1]].T).T + hull.equations[:, -1], axis=-1)
		
		min_dist = np.min(np.sum(np.square(new_templates_prime-proposal_prime), axis =1))

		#print(min_dist, np.square(dist_boundaries), dist_boundaries)
		#plt.scatter(*corners_prime.T)
		#plt.scatter(*proposal_prime)
		#plt.scatter(*new_templates_prime.T)
		#plt.show()
		
		min_dist = min(min_dist, (np.abs(dist_boundaries)))

		if min_dist > dist_sq:
			new_templates = np.concatenate([new_templates, proposal[None,:]], axis = 0)
			new_templates_prime = np.concatenate([new_templates_prime, proposal_prime[None,:]], axis = 0)
			nothing_new = 0
		else:
			nothing_new += 1
		
	return new_templates

#@do_profile(follow=[])
def place_stochastically(avg_dist, t_obj, bank, empty_iterations = 200, seed_bank = None):
	"""
	Place templates with a stochastic placing algorithm
	It iteratively proposes a new template to add to the bank. The proposal is accepted if the distance of the proposal with the previously placed templates is smaller than ``avg_dist``. The iteration goes on until no template is found to have a distance smaller than the given threshold ``avg_dist``.
	It can start from a given set of templates.

	At each iteration a proposal is made in each tile. This means that if the number of tiles is larger than the expected number ot templates, this method will **overcover**.
	
	The match of the proposals is computed with `compute_metric_injections_match`.
	
	The match of a proposal is computed against all the templats that have been added.
	
	Parameters
	----------
		avg_dist: float
			Average distance between templates. It can be tuned with `mbank.utils.get_avg_dist`
		
		t_obj: tiling_handler
			A tiling object to compute the match with
		
		bank: cbc_bank
			A ``cbc_bank`` object. The bank will be filled with the new templates generated
		
		empty_iterations: int
			Number of consecutive templates that are not accepted before the placing algorithm is terminated
			
		seed_bank: np.ndarray
			shape: (N,D) -
			A set of templates that provides a first guess for the bank.
	
	Returns
	-------
		new_templates: np.ndarray
			shape: (N,D) -
			A set of templates generated by the stochastic placing algorithm
	"""
	N_neigh_templates = 20000 #FIXME: This option doesn't work!!

		#User communication stuff
	def dummy_iterator():
		while True:
			yield
	t_ = tqdm(dummy_iterator())

	MM = 1- avg_dist**2

	if seed_bank is None:
		ran_id_ = np.random.choice(len(t_obj))
		new_templates = np.random.uniform(t_obj[ran_id_][0].mins, t_obj[ran_id_][0].maxes, (1, len(t_obj[ran_id_][0].maxes)))
	else:
		new_templates = np.asarray(seed_bank)

	nothing_new = np.zeros((len(t_obj),), dtype = int)
	tiles_to_use = np.array([i for i in range(len(t_obj))], dtype = int)
	
	try:
		for _ in t_:
			#bank.templates = new_templates #updating bank
			
				#checking for stopping conditions and updating for empty tiles
			where_to_remove = (nothing_new > empty_iterations)
			tiles_to_use = np.delete(tiles_to_use, np.where(where_to_remove))
			nothing_new = np.delete(nothing_new, np.where(where_to_remove))

			t_.set_description("Templates added {} ({}/{} tiles full)".format(new_templates.shape[0], len(t_obj)-len(tiles_to_use), len(t_obj)))
			if len(tiles_to_use) == 0: break
			
			id_tiles_to_use = np.random.choice(len(tiles_to_use)) #id of the tile to use in the list tiles_to_use
			tile_id = tiles_to_use[id_tiles_to_use] #id of the tile in the tiling list
			
			proposal = np.random.uniform(t_obj[tile_id][0].mins, t_obj[tile_id][0].maxes, (1, len(t_obj[tile_id][0].maxes)))
			
			diff = new_templates - proposal #(N_templates, D)
			
			L_t = np.linalg.cholesky(t_obj[tile_id][1]) #(D,D)
			match = np.max(1- np.sum(np.square(np.matmul(diff, L_t)), axis =1))
			
				#slower alternative with eisum
			#match = np.max(1 - np.einsum('ij, jk, ik -> i', diff, t_obj[tile_id][1], diff)) #()

			if (match < MM)>0:
				new_templates = np.concatenate([new_templates, proposal], axis =0)
				nothing_new[id_tiles_to_use] = 0
			else:
				nothing_new[id_tiles_to_use] +=1
	except KeyboardInterrupt:
		pass
	
	return new_templates


def partition_tiling(thresholds, d, t_obj):
	"""
	Given a tiling, it partitions the tiling given a list of thresholds. The splitting is performed along axis `d`.
	
	Parameters
	----------
		thresholds: list
			list of trhesholds for the partitioning
		
		d: int
			Axis to split the tiling along.
		
		t_obj: tiling_handler
			Tiling to partion
	
	Returns
	-------
		partitions: list
			List of tiling handlers making the partitions
	"""
	if not isinstance(thresholds, list): thresholds = list(thresholds)
	thresholds.sort()
	
	partitions = []
	
	t_obj_ = t_obj
	
	for threshold in thresholds:
		temp_t_obj, t_obj_ = t_obj_.split_tiling(d, threshold)
		partitions.append(temp_t_obj)
	partitions.append(t_obj_)
	
	return partitions


#@do_profile(follow=[])
def place_random(dist, t_obj, N_points, tolerance = 0.01):
	"""
	Given a tiling object, it covers the volume with points and covers them with templates.
	It follows `2202.09380 <https://arxiv.org/abs/2202.09380>`_
	
	Parameters
	----------
	
		dist: float
			Typical distance between templates. It can be tuned with `mbank.utils.get_avg_dist`.
		
		t_obj: tiling_handler
			Tiling handler that tiles the parameter space
		
		N_points: int
			Number of livepoints to cover the space with
		
		tolerance: float
			Fraction of livepoints to be covered before terminating the loop
	
	Returns
	-------
		new_templates: np.ndarray
			shape: (N,D) -
			A set of templates generated by the placing algorithm
	"""
	#TODO: maybe here you can use the tiling KDTree for saving some useless computations?
	#i.e. you should add a tiling label to all the generated livepoints and use it somehow...
	
	assert tolerance <=1., "The tolerance should be a fraction in (0,1]"
	
	dist_sq = dist**2
	dtype = np.float32 #better to downcast to single precision! There is a mild speedup there
	livepoints = t_obj.sample_from_tiling(N_points, dtype = dtype) #(N_points, D)
	
		#ordering the tile by volume in ascending order...
	_, vols = t_obj.compute_volume()
	
	def dummy_iterator():
		while True:
			yield
	
	new_templates = []
	
	bar_str = 'Loops on tiles ({}/{} livepoints killed | {} templates placed)'
	it = tqdm(dummy_iterator(), desc = bar_str.format(N_points -len(livepoints), N_points, len(new_templates)), leave = True)
	
	for _ in it:
		id_point = np.random.randint(len(livepoints))
		point = livepoints[id_point,:]
		
		if len(livepoints)<=tolerance*N_points:
			break
			
		diff = livepoints - point #(N,D)
		
		id_ = t_obj.get_tile(point)[0]
		metric = t_obj[id_][1]
		
				#measuring metric match between livepoints and proposal
		L_t = np.linalg.cholesky(metric).astype(dtype) #(D,D)
		
			#BLAS seems to be faster for larger matrices but slower for smaller ones...
			#Maybe put a threshold on the number of livepoints?
		diff_prime = scipy.linalg.blas.sgemm(1, diff, L_t)
		#diff_prime_np = np.matmul(diff, L_t)
		#print(np.allclose(diff_prime_np, diff_prime))
		
		dist_ = np.sum(np.square(diff_prime), axis =1) #(N,) #This is the bottleneck of the computation, as it should be
		ids_kill = np.where(dist_< dist_sq)[0]

		del diff, dist_

			#This operation is very slow! But maybe there is nothing else to do...
		livepoints = np.delete(livepoints, ids_kill, axis = 0) 
		
				#this is very very subtle: if you don't allocate new memory with np.array, you won't decrease the reference to livepoints, which won't be deallocated. This is real real bad!!
		new_templates.append(np.array(point, dtype = np.float64))
		del point
			
		if len(livepoints) == 0: break
		if len(new_templates) %2 ==0: it.set_description(bar_str.format(N_points -len(livepoints), N_points, len(new_templates)) )
	
	new_templates = np.column_stack([new_templates])
	#if len(livepoints)>0: new_templates = np.concatenate([new_templates, livepoints], axis =0) #adding the remaining livepoints
	
	return new_templates
	
def create_mesh_new(dist, tile, coarse_boundaries = None):
	"""
	Creates a mesh of points on an hypercube, given a metric.
	The points are approximately equally spaced with a distance ``dist``.
	
	`NEW VERSION: TEST IN PROGRESS. IT SEEMS TO OVERCOVER :( PROBABLY NOT THE WAY TO GO!`
	
	Parameters
	----------
		dist: float
			Distance between templates
		
		tile: tuple
			An element of the ``tiling_handler`` object.
			It consists of a tuple ``(scipy.spatial.Rectangle, np.ndarray)``
	
		coarse_boundaries: np.ndarray
			shape: (2,D) -
			An array with the coarse boundaries of the tiling.
			If given, each tile is checked to belong to the border of the tiling. If it's the case, some templates are added to cover the boundaries

	Returns
	-------
		mesh: np.ndarray
			shape: (N,D) - 
			A mesh of N templates that cover the tile
	"""
	D = tile[0].maxes.shape[0]
	
		#bound_list keeps the dimension over which the tile is a boundary in the larger space
	if coarse_boundaries is not None:
		up_bound_list = np.where( np.isclose(tile[0].maxes, coarse_boundaries[1,:], 1e-4, 0) )[0].tolist() #axis where there is an up bound
		low_bound_list = np.where( np.isclose(tile[0].mins, coarse_boundaries[0,:], 1e-4, 0) )[0].tolist()
		bound_list = [ (1, up_) for up_ in up_bound_list]
		bound_list.extend( [ (0, low_) for low_ in low_bound_list])
	else: bound_list = []
	
		#Computing Choelsky decomposition of the metric	
	metric = tile[1]
	L = np.linalg.cholesky(metric).T
	L_inv = np.linalg.inv(L)
	
		#computing boundaries and boundaries_prime
	boundaries = np.stack([tile[0].mins, tile[0].maxes], axis = 0) #(2,D)
	corners = get_cube_corners(boundaries)#[0,:], boundaries[1,:])
	corners_prime = np.matmul(corners, L.T)
	center = (tile[0].mins+tile[0].maxes)/2. #(D,) #center
	center_prime = np.matmul(L, center) #(D,) #center_prime
	
		#computing the extrema of the boundaries (rectangle)
	boundaries_prime = np.array([np.amin(corners_prime, axis =0), np.amax(corners_prime, axis =0)])
	
		#creating a mesh in the primed coordinates (centered around center_prime)
	mesh_prime = []
	where_single_layer = [] #list to keep track of the dimensions where templates should be drawn at random!
	
	for d in range(D):
		min_d, max_d = boundaries_prime[:,d]
		#if (0,d) not in bound_list: min_d = min_d + dist/2.
		#if (1,d) not in bound_list: max_d = max_d - dist/2.
		
		N = max(int((max_d-min_d)/dist), 1)
			#this tends to overcover...
		#grid_d = [np.linspace(min_d, max_d, N+1, endpoint = False)[1:]] 
			 #this imposes a constant distance but may undercover
		grid_d = [np.arange(center_prime[d], min_d, -dist)[1:][::-1], np.arange(center_prime[d], max_d, dist)]

		grid_d = np.concatenate(grid_d)
		
		if len(grid_d) <=1 and d >1: where_single_layer.append(d)
		
		mesh_prime.append(grid_d)
		
		#creating the mesh in the primed space and inverting
	mesh_prime = np.meshgrid(*mesh_prime)
	mesh_prime = [g.flatten() for g in mesh_prime]
	mesh_prime = np.column_stack(mesh_prime) #(N,D)
	
	mesh = np.matmul(mesh_prime, L_inv.T)

		#we don't check the boundaries for the axis that will be drawn at random
	axis_ok = [i for i in range(D) if i not in where_single_layer]
	ids_ok = np.logical_and(np.all(mesh[:,axis_ok] >= boundaries[0,axis_ok], axis =1), np.all(mesh[:,axis_ok] <= boundaries[1,axis_ok], axis = 1)) #(N,)
	mesh = mesh[ids_ok,:]
	
		#Creating a grid for the others dimensions
	new_mesh = []
	for d in where_single_layer:
		min_d, max_d = boundaries[:,d]

			#teoretical N_points
			#It is not achieved to the finitess of the tile...
		metric_d = project_metric(metric, [d])
		N_teo = (max_d - min_d)*np.sqrt(np.squeeze(metric_d)) / dist

			#checking the distance between the top of the tile and the grid so far
			#If it is too high, we should add more to the grid		
		test_point = np.mean(boundaries, axis = 0)
		test_point[d] = boundaries[1,d] 
		diff = mesh - test_point
		true_dist = np.min(np.sqrt(np.einsum('ij,jk,ik->i', diff, metric, diff)))
		N_true = 2*true_dist/dist

			#adding boundaries
		N_true = N_true - 0.5 * ((0,d) not in bound_list) - 0.5 * ((1,d) not in bound_list)
		#grid_d = np.linspace(min_d, max_d, max(int(N_true), 1)+1, endpoint = ((1,d) in bound_list))[int(((0,d) not in bound_list)):]
		grid_d = np.linspace(min_d, max_d, max(int(N_true), 1)+1, endpoint = False)[1:]
		new_mesh.append(grid_d)
		
		print(d, N_teo, N_true)
		print(grid_d)

	N_new_points = np.prod([len(g) for g in new_mesh])
	
	if N_new_points>1:
			#filling the new grid
		new_mesh = np.meshgrid(*new_mesh)
		new_mesh = [g.flatten() for g in new_mesh]
		new_mesh = np.column_stack(new_mesh) #(N,D)
		
		mesh_list = []
		for i in range(N_new_points):
			temp_mesh = np.array(mesh)
			temp_mesh[:,where_single_layer] = new_mesh[i,:]
			mesh_list.append(temp_mesh)
		mesh = np.concatenate(mesh_list, axis =0)
	
	return mesh
	

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
	
		coarse_boundaries: np.ndarray
			shape: (2,D) -
			An array with the coarse boundaries of the tiling.
			If given, each tile is checked to belong to the border of the tiling. If it's the case, some templates are added to cover the boundaries

	Returns
	-------
		mesh: np.ndarray
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
	
		#Computing Choelsky decomposition of the metric	
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
	#for id_random in where_random:
	#	mesh[:,id_random] =np.random.uniform(boundaries[0,id_random], boundaries[1,id_random], (mesh.shape[0], )) # center[id_random] #
	warnings.warn('Random extraction for "non-important" dimensions disabled!')
	
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
		boundaries: np.ndarray
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
		BBH_components: np.ndarray
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
		
def save_injs(filename, injs, GPS_start, GPS_end, time_step, approx, luminosity_distance = 100, f_min = 20.):
		"""
		Save the given injections to a ligo xml injection file (sim_inspiral table).
		
		Parameters
		----------
			
		filename: str
			Filename to save the injections at
		
		injs: np.ndarray
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
			row.taper = "TAPER_START"
			row.bandpass = 0

				#setting interesting row paramters
			row.inclination = iota
			row.coa_phase = phi
			row.polarization = np.random.uniform(0.0, 2.0 * np.pi)
			row.longitude = np.random.uniform(0.0, 2.0 * np.pi)
			row.latitude = np.arcsin(np.random.uniform(-1.0, 1.0))
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
			
			row.f_final = 2500 /(row.mtotal) #are you sure this is fine??
			
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

















	
	
	
