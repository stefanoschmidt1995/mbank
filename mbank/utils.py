"""
mbank.utils
===========
	Some utilities for mbank, for plotting purposes, for template placing and for match computation
	#TODO: write more here....
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
from itertools import combinations, permutations
import argparse
import lal.series

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

class parse_from_file(argparse.Action):
	"Convenience class to read th arguments from a config file"	
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

####################################################################################################################

def load_PSD(filename, asd = False, ifo = 'H1'):
	"""
	Loads a PSD from file and returns a grid of frequency and PSD values
	
	Parameters
	----------
		filename: 'str'
			Name of the file to load the PSD from (can be a txt file or an xml file)

		asd: 'bool'
			Whether the file contains an ASD rather than a PSD
		
		ifo: 'str'
			Interferometer which the PSD refers to. Only for loading a PSD from xml
	
	Returns
	-------
		f: 'np.ndarray'
			Frequency grid

		PSD: 'np.ndarray'
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
def compute_avg_dist(bank, tiling, metric_obj, metric_approx = True):
	"""
	Given a tiling and a bank, it computes the distance of the NN for each template
	
	Parameters
	----------
		
		bank: 'WF_bank'
			WF_bank object

		tiling: 'tiling_handler'
			A tiling
			
		metric_obj: 'cbc_metric'
			A cbc_metric object to compute the match with
		
		metric_approx: 'bool'
			Whether to use the metric to approximate the match
		
	Returns
	-------
			
		out_list: 'list'
			A list of dict.
			For each tile it is recorded a dict with {'theta_inj': (N_inj,D), 'match': (N_inj,), 'id_match': (N_inj,)}
	"""
	
		#FIXME: the avg distance between templates is far the one we set by hand
		#	On the other hand, the mismatch looks good and most important the templates looks nicely spread
	out_list = []
	id_tile = -1
	for t in tqdm(tiling[:10]):
		id_tile += 1
		dist_t = t[0].min_distance_point(bank.templates)
		ids_t = np.where(dist_t == 0.)[0] #(N,) #ids of the templates that lie within this tile

		if len(ids_t) ==0: continue #skipping this
		
			#getting the template
		templates_t = bank.templates[ids_t,:]
		
		#if metric_approx:
				#you need to do the outer product of the difference... how??
		#	diff = np.einsum('ij,lj->ilj', templates_t, templates_t)
		#	matches_t_ = 1 + np.einsum('ilj,jk,ilk->il', diff, t[1], diff)
		
		if not metric_approx:
			templates_t_WFs = metric_obj.get_WF(templates_t) #(N, D)
		
		matches_t = []
		
		for i in range(templates_t.shape[0]):
		
			if metric_approx:
				template_ = np.repeat([templates_t[i,:]], len(templates_t), axis =0) #(N,D)
				match_i = metric_obj.metric_match(templates_t, template_, t[1]) #(N,)
			else:
				template_ = np.repeat([templates_t_WFs[i,:]], len(templates_t), axis =0) #(N,D)
				match_i = metric_obj.WF_match(templates_t_WFs, template_) #(N,)
			
			#print(match_i)
			
			matches_t.append(match_i)

		matches_t = np.array(matches_t) #(N_t, N_t)
		
		matches_t = matches_t-np.eye(matches_t.shape[0])
		
		ids_max_t = np.argmax(matches_t, axis =1) #(N_t,) #id of the best matching template for each injection (ids refers to ids_t)

		matches_t = np.max(matches_t, axis =1) #(N_t,)
		ids_max_t = [ids_t[id_max] for id_max in ids_max_t] #(N_t,) #id of the best matching template for each injection (ids refers to bank)

		matches_t = metric_obj.metric_match( bank.templates[ids_max_t,:], templates_t, t[1])

		out_list.append({'tile': id_tile,  'match': matches_t, 'id_match': ids_max_t})

	return out_list

def ray_compute_injections_match(inj_dict, templates, metric_obj, N_neigh_templates = 10, symphony_match = False, cache = True):
	"""
	Given an injection dictionary, generated by `compute_metric_injections_match` it computes the actual match (not the metric approximation) between injections and templates.
	The injections are generic (not necessarly projected on the bank submanifold).
	It make use of ray package to parallelize the execution.
	
	Parameters
	----------
		inj_dict: 'dict'
			A dictionary with the data injection as computed by `compute_metric_injections_match`.
		
		templates: 'np.ndarray'
			An array with the templates. They should have the same layout as lal (given by get_BBH_components)

		metric_obj: 'cbc_metric'
			A cbc_metric object to compute the match with.

		N_neigh_templates: 'int'
			The number of neighbouring templates to consider for each injection
						
		cache: 'bool'
			Whether to cache the WFs
			
		symphony_match: 'bool'
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
	Wrapper to compute_injections_match to allow for parallelization with ray.

	Parameters
	----------
		inj_dict: 'dict'
			A dictionary with the data injection as computed by `compute_metric_injections_match`.
		
		templates: 'np.ndarray'
			An array with the templates. They should have the same layout as lal (given by get_BBH_components)

		metric_obj: 'cbc_metric'
			A cbc_metric object to compute the match with.

		N_neigh_templates: 'int'
			The number of neighbouring templates to consider for each injection
		
		symphony_match: 'bool'
			Whether to use the symphony match
				
		cache: 'bool'
			Whether to cache the WFs
			
		worker_id: 'int'
			Id of the ray worker being used. If None, it is assumed that ray is not called
		
	Returns
	-------
		out_dict: 'dict'
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
	"""
	Given an injection dictionary, generated by `compute_metric_injections_match` it computes the actual match (not the metric approximation) between injections and templates.
	The injections are generic (not necessarly projected on the bank submanifold).
	
	Parameters
	----------
		inj_dict: 'dict'
			A dictionary with the data injection as computed by `compute_metric_injections_match`.
		
		templates: 'np.ndarray'
			An array with the templates. They should have the same layout as lal (given by get_BBH_components)

		metric_obj: 'cbc_metric'
			A cbc_metric object to compute the match with.

		N_neigh_templates: 'int'
			The number of neighbouring templates to consider for each injection
		
		symphony_match: 'bool'
			Whether to use the symphony match
		
		cache: 'bool'
			Whether to cache the WFs
		
		worker_id: 'int'
			Id of the ray worker being used. If None, it is assumed that ray is not called
		
	Returns
	-------
		out_dict: 'dict'
			The output dictionary with the updated matches
	"""
	assert metric_obj.variable_format == 'm1m2_fullspins_emeanano_iotaphi', "Wrong variable format given. It must be 'm1m2_fullspins_emeanano_iotaphi'"

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
	
	for i in tqdm(range(injs.shape[0]), desc = desc):
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

	return dict(inj_dict)


####################################################################################################################	
def compute_metric_injections_match_bruteforce(injs, bank, tiling, N_neigh_templates = 10):
	"Brute force match computation"
	if N_neigh_templates > bank.templates.shape[0]:
		N_neigh_templates = bank.templates.shape[0]

	out_dict = {'theta_inj': injs,
				'id_tile': None, 'id_tile_neig':  None,
				'match': np.zeros((injs.shape[0],)), 'id_match':  np.empty((injs.shape[0],), dtype = int),
				'match_neig': np.zeros((injs.shape[0], N_neigh_templates)), 'id_match_neig': np.empty((injs.shape[0], N_neigh_templates), dtype = int)
				}
	
		#Projecting the injection on the template manifold
	injs = bank.var_handler.get_theta(injs, bank.variable_format)
	dist_t = []
	for t in tqdm(tiling, desc='Identifying a tile for each injection', leave = True):
		dist_t.append( t[0].min_distance_point(injs) ) #(N_injs,)
	dist_t = np.stack(dist_t, axis = 1) #(N_injs, N_tiles)
	id_sort = np.argmin(dist_t, axis = 1)
	del dist_t

	if (bank.templates.shape == injs.shape): template_dist = np.allclose(bank.templates, injs)
	else: template_dist = False
	
	out_dict['id_tile'] = id_sort
	N_argpartiton = 20000
	
	for i in tqdm(range(injs.shape[0]), desc = 'Evaluating brute force metric match for injections', leave = True):
		diff = bank.templates - injs[i] #(N_templates, D)
		
		#these are the indices being checked
		#TODO: check this is fine!!!
		
		if N_argpartiton < bank.templates.shape[0]:
			id_diff_ok = np.argpartition(np.linalg.norm(diff, axis=1), N_argpartiton)[:N_argpartiton]
		else:
			id_diff_ok = np.arange(bank.templates.shape[0])
		
		match_i = 1 + np.einsum('ij, jk, ik -> i', diff[id_diff_ok], tiling[out_dict['id_tile'][i]][1], diff[id_diff_ok])
		indices_i = np.argpartition(match_i, -N_neigh_templates)[-N_neigh_templates:]
		
			#sorting
		ids_sort = np.argsort(-match_i[indices_i])
		out_dict['match_neig'][i,:] = match_i[indices_i][ids_sort]
		out_dict['id_match_neig'][i,:] = id_diff_ok[indices_i[ids_sort]] #indices_i[ids_sort]
		
			#id = 0 if not template dist, 1 if template dist
		out_dict['id_match'][i] = out_dict['id_match_neig'][i, int(template_dist)]
		out_dict['match'][i] = out_dict['match_neig'][i, int(template_dist)]
		
	return out_dict

def compute_metric_injections_match(injs, bank, tiling, N_neigh_templates = 10, N_neigh_tiles = 4, tile_id_population = None):
	"""
	Given a tiling, a bank and some injections, it computes the metric match between templates the injections.
	The injection must lie in the same bank submanifold.
	
	Parameters
	----------
		injs: 'np.ndarray' (N_injs, 10)
			An array with the injections. They should be have format (N, 10), as output by var_handler.get_BBH_components
		
		bank: 'WF_bank'
			WF_bank object

		tiling: 'tiling_handler'
			A tiling object
		
		N_neigh_templates: 'int'
			The number of neighbouring templates to consider for each injection
		
		N_neigh_tiles: 'int'
			The number of neighbouring tiles to consider for each injection
		
		tile_id_population: 'list'
			A list of list. 
			tile_id_population[i] keeps the ids of the templates inside tile i
			If None, this object will be created. However, this is time consuming for large banks and might be better to load it from file
		
	Returns
	-------
		out_dict: 'dict'
			A dictionary with the resulting data. Each entry has an array with N_injs rows. The entry are as follow:
				theta_inj	match	id_match	tile_id	id_tile_neig		match_neig				id_match_neig
				(N,D)		(N,)	(N,)		(N,)	(N, N_neigh_tiles)	(N,N_neigh_templates)	(N,N_neigh_templates)
	"""
	###
		#Creating the output structure
		#This is empty expect for the theta values.
		#The whole point of this function is to fill out_dict in a smart way
	###
	out_dict = {'theta_inj': injs,
				'id_tile': None, 'id_tile_neig':  None,
				'match': np.zeros((injs.shape[0],)), 'id_match':  np.empty((injs.shape[0],), dtype = int),
				'match_neig': np.zeros((injs.shape[0],N_neigh_templates)), 'id_match_neig': np.empty((injs.shape[0], N_neigh_templates), dtype = int)
				}
	
		#Projecting the injection on the template manifold
	injs = bank.var_handler.get_theta(injs, bank.variable_format)
	###
		# For each template of the bank, we compute which tile it is in
	###

	if tile_id_population is None:
		tile_id_population = [] #(N_tiles,) #for each tile, brings the indices of the templates that are inside it	
		for t in tqdm(tiling, desc='Computing the tile which each template belongs to', leave = True):
			dist_t = t[0].min_distance_point(bank.templates)
			tile_id_population.append( np.where(dist_t == 0.)[0] )
			if len(tile_id_population[-1]) ==0:
				warnings.warn("The tile {} does not have templates inside. This is pathological: the injection routine may fail".format(tiling.index(t)))

	###
		# Check if there are templates that do not belong to any tile... In this case, they will be assigned to the nearest one
		# This is done only if the tile_id_population object is not provided by the user
	###
		lone_templates_id = list(set(range(bank.templates.shape[0])) - set([id_t_ for id_t in tile_id_population for id_t_ in id_t ]))
		if len(lone_templates_id) > 0:
			dists = []
			for t in tiling:
				dists.append( t[0].min_distance_point(bank.templates[lone_templates_id,:]))
			id_dists = np.argmin(dists, axis = 0) #(N_templates,) #for each lone template, this keeps the tile in which it should be assigned

			for id_tile_, lone_template_id_ in zip(id_dists, lone_templates_id):
				tile_id_population[id_tile_] = np.append(tile_id_population[id_tile_], lone_template_id_)

	###
		#for each injection, we identify the closest N_neigh_tiles tiles
		# FIXME: do this with a better memory scaling. This may be unfeasible for large banks
	###
	dist_t = []
	for t in tqdm(tiling, desc='Identifying neighbouring tiles for each injection', leave = True):
		dist_t.append( t[0].min_distance_point(injs) ) #(N_injs,)
	dist_t = np.stack(dist_t, axis = 1) #(N_injs, N_tiles)
	id_sort = np.argsort(dist_t, axis = 1)[:,:N_neigh_tiles]
	del dist_t
	
	out_dict['id_tile'] = id_sort[:,0] #(N_injs, )
	out_dict['id_tile_neig'] = id_sort #(N_injs, N_neigh_tiles)
	
	###
		#for each injection, we compute the metric match for the templates in the neighbouring tiles and we identify the N_neigh_templates neighbouring templates
		#This will fill the entry 'match_neig' and id_match_neig'
	###
	match_inj = []#[] for i in range(injs.shape[0])]
	for i in tqdm(range(injs.shape[0]), desc = 'Evaluating metric match for injections', leave = True):
		id_tiles = out_dict['id_tile_neig'][i] #these are the ids of the tiles to look at
		
		match_list = []
		id_list = []
		
		for id_ in id_tiles:
			id_list.extend(tile_id_population[id_]) #id of the templates in the tile
				#computing the metric match
			metric = tiling[id_][1]
			
			diff = bank.templates[tile_id_population[id_],:] - injs[i] #(N_templates_in_tile, D)
			match_ = 1 + np.einsum('ij, jk, ik -> i', diff, metric, diff) ##(N_templates_in_tile,)
			match_list.extend(match_)
		
			#selecting the closest N_neigh_templates to the injection
		match_list = np.array(match_list)
		id_list = np.array(id_list)
		ids_sort = np.argsort(-match_list)[:N_neigh_templates]
		match_list = match_list[ids_sort]
		id_list = id_list[ids_sort]
		
		if len(match_list)< N_neigh_templates:
			N_repeats = int(N_neigh_templates/len(match_list))+1
			match_list = np.repeat(match_list, N_repeats)[:N_neigh_templates]
			id_list = np.repeat(id_list, N_repeats)[:N_neigh_templates]
		
		out_dict['match_neig'][i,:] = match_list
		out_dict['id_match_neig'][i,:] = id_list
		out_dict['match'][i] = match_list[0]
		out_dict['id_match'][i] = id_list[0]
		
	return out_dict

####################################################################################################################

def plot_tiles_templates(t_obj, templates, variable_format, save_folder, show = False):
		###
		#Plotting templates
		###
	var_handler = variable_handler()
	if not save_folder.endswith('/'): save_folder.out_dir = save_folder.out_dir+'/'
	
		###
		#Plotting templates & tiles
		###
	if templates.shape[0] >500000: ids_ = np.random.choice(templates.shape[0], 500000, replace = False)
	else: ids_ = range(templates.shape[0])
	
	size_template = [20 if templates.shape[0] < 10000 else 2][0]
	centers = np.array([ (t[0].maxes + t[0].mins)/2. for t in t_obj])
	fig, axes = plt.subplots(templates.shape[1]-1, templates.shape[1]-1, figsize = (15,15))
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
			currentAxis.set_ylabel(var_handler.labels(variable_format, latex = True)[ax_[1]])
		else:
			currentAxis.set_yticks([])
		if ax_[1] == templates.shape[1]-1:
			currentAxis.set_xlabel(var_handler.labels(variable_format, latex = True)[ax_[0]])
		else:
			currentAxis.set_xticks([])

	plt.savefig(save_folder+'bank.png', transparent = True)

		#Plot the tiling
	for ax_ in combinations(range(templates.shape[1]), 2):
		currentAxis = axes[ax_[1]-1, ax_[0]]
		ax_ = list(ax_)
		currentAxis.scatter(*centers[:,ax_].T, s = 30, marker = 'x', c= 'r', alpha = 1)
		for t in t_obj:
			d = t[0].maxes- t[0].mins
			currentAxis.add_patch(matplotlib.patches.Rectangle(t[0].mins[ax_], d[ax_[0]], d[ax_[1]], fill = None, alpha =1))


	plt.savefig(save_folder+'tiling.png', transparent = True)
	
		#Plot an histogram
	fig, axes = plt.subplots(1, templates.shape[1], figsize = (4*templates.shape[1], 5), sharey = True)
	hist_kwargs = {'bins': min(50, int(len(ids_)/50) ), 'histtype':'step', 'color':'orange'}
	for i, ax_ in enumerate(axes):
		ax_.hist(templates[ids_,i], **hist_kwargs)
		if i==0: ax_.set_ylabel("# templates")
		ax_.set_xlabel(var_handler.labels(variable_format, latex = True)[i])
		min_, max_ = np.min(templates[:,i]), np.max(templates[:,i])
		d_ = 0.1*(max_-min_)
		ax_.set_xlim((min_-d_, max_+d_ ))
	plt.savefig(save_folder+'hist.png', transparent = True)
	
	if show: plt.show()
	
	return

####################################################################################################################

def plot_tiles(tiles_list, boundaries):
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	plt.ylim((boundaries[0][1], boundaries[1][1]))
	plt.xlim((boundaries[0][0], boundaries[1][0]))
	
	for t in tiles_list:
		min_, max_ = t[0].mins, t[0].maxes
		center = (min_+max_)/2.
		ax1.scatter(*center[:2], c = 'r')
	return

@ray.remote
def get_templates_ray(bank_obj, metric_obj, avg_dist, lower_boxes, upper_boxes, lower_boxes_i, upper_boxes_i, p_disc, verbose = False):
	return bank_obj.get_templates(metric_obj, avg_dist, lower_boxes, upper_boxes, lower_boxes_i, upper_boxes_i, p_disc, verbose)
	#return test_get_templates()

def get_cube_corners(boundaries):
	"Given the extrema of a cube, it computes the corner of the cube"
	mesh = np.meshgrid(*boundaries.T)
	mesh = [g.flatten() for g in mesh]
	mesh = np.column_stack(mesh)
	return mesh

def plawspace(start, stop, exp, N_points):
	"Generates a grid which is 'power law distributed'. Helpful for a nice grid spacing in the mass sector."
	f_start = np.power(start, exp)
	f_stop = np.power(stop, exp)
	points = np.linspace(f_start, f_stop, N_points)
	points = np.power(points, 1/exp)
	
	return points

def place_stochastically(MM, tile):
	"Place templates stochastically at approximately constant distance dist"
	dist_sq = (1-MM)
	new_templates = np.random.uniform(tile[0].mins, tile[0].maxes, (1, tile[0].maxes.shape[0])) #(1,D)
	
	metric = -tile[1]
	L = np.linalg.cholesky(metric).T
	
	new_templates_prime = np.matmul(new_templates, L.T) #(1,D)
	
	nothing_new = 0
	while nothing_new < 300:
		proposal = np.random.uniform(tile[0].mins, tile[0].maxes, tile[0].maxes.shape) #(D,)
		proposal_prime = np.matmul(L, proposal) #(D,)
		
		min_dist = np.min(np.sum(np.square(new_templates_prime-proposal_prime), axis =1))

		if min_dist > dist_sq:
			new_templates = np.concatenate([new_templates, proposal[None,:]], axis = 0)
			new_templates_prime = np.concatenate([new_templates_prime, proposal_prime[None,:]], axis = 0)
			nothing_new = 0
		else:
			nothing_new += 1
		
	return new_templates

def place_stochastically_globally(MM, t_obj, empty_iterations = 200, first_guess = None):
	"Does stochastic placement on the overall tiling"
	N_neigh_templates = 20000 #FIXME: This option doesn't work!!

		#User communication stuff
	def dummy_iterator():
		while True:
			yield
	t_ = tqdm(dummy_iterator())

	dist_sq = (1-MM)

	if first_guess is None:
		new_templates = np.concatenate([np.random.uniform(t[0].mins, t[0].maxes, (5, len(t[0].maxes))) for t in t_obj], axis=0)
	else:
		new_templates = np.array(first_guess)

	nothing_new = np.zeros((len(t_obj),), dtype = int)
	tiles_to_use = np.array([i for i in range(len(t_obj))], dtype = int)
	
	for _ in t_:
			#checking for stopping conditions
		where_to_remove = (nothing_new > empty_iterations)
		
		if np.all(where_to_remove): break
		tiles_to_use = np.delete(tiles_to_use, np.where(where_to_remove))
		nothing_new = np.delete(nothing_new, np.where(where_to_remove))
		
		proposal = np.concatenate([np.random.uniform(t_obj[id_][0].mins, t_obj[id_][0].maxes, (1, len(t_obj[id_][0].maxes))) for id_ in tiles_to_use], axis = 0)
		proposal = np.atleast_2d(proposal)
		
		match_list = []
		for p_, id_ in zip(proposal, tiles_to_use):
			diff = new_templates - p_ #(N_templates, D)
			if N_neigh_templates < diff.shape[0]:
				id_diff_ok = np.argpartition(np.linalg.norm(diff, axis=1), N_neigh_templates)[:N_neigh_templates]
			else:
				id_diff_ok = range(diff.shape[0])
				#TODO: here you could use argpartition to consider only the first N_neighbours in diff (might speed up)...
			match_i = 1 + np.einsum('ij, jk, ik -> i', diff[id_diff_ok,:], t_obj[id_][1], diff[id_diff_ok,:]) #(N_templates,)
			match_list.append(np.max(match_i))

		accepted = (np.array(match_list) < MM)
		N_accepted = np.sum(accepted)
		
		if N_accepted>0:
			new_templates = np.concatenate([new_templates, proposal[accepted,:]], axis =0)

		t_.set_description("Templates added {} ({}/{} tiles full)".format(new_templates.shape[0], len(t_obj)-len(tiles_to_use), len(t_obj)))

		nothing_new[accepted] = 0
		nothing_new[~accepted] += 1
		
	return new_templates

def create_mesh(dist, tile, coarse_boundaries = None):
	"Creates a mesh of points on an hypercube, given a metric."
	#dist: float
	#metric: (D,D)
	#boundaries (2,D)
	
		#bound_list keeps the dimension over which the tile is a boundary in the larger space
	if tile[0].maxes.shape[0] < 2: coarse_boundaries = None
	if coarse_boundaries is not None:
		up_bound_list = np.where( np.isclose(tile[0].maxes, coarse_boundaries[1,:], 1e-4, 0) )[0].tolist()
		low_bound_list = np.where( np.isclose(tile[0].mins, coarse_boundaries[0,:], 1e-4, 0) )[0].tolist()
		bound_list = [ (1, up_) for up_ in up_bound_list]
		bound_list.extend( [ (0, low_) for low_ in low_bound_list])
	else: bound_list = []
	
		#Computing Choelsky decomposition of the metric	
	metric = -tile[1]
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
	
	for d in range(boundaries_prime.shape[1]):
		
		N = int((boundaries_prime[1,d]-boundaries_prime[0,d])/dist+1)	
		grid_d = [np.linspace(boundaries_prime[0,d], boundaries_prime[1,d], N+1, endpoint = False)[1:]] #FIXME: is this better?
		#grid_d = [np.arange(center_prime[d], boundaries_prime[0,d], -dist)[1:][::-1], np.arange(center_prime[d], boundaries_prime[1,d], dist)]
		grid_d = np.concatenate(grid_d)
		
		if len(grid_d) <=1 and d >1: where_random.append(d)
		
		mesh_prime.append(grid_d)
		
		#creating the mesh in the primed space and inverting
	mesh_prime = np.meshgrid(*mesh_prime)
	mesh_prime = [g.flatten() for g in mesh_prime]
	mesh_prime = np.column_stack(mesh_prime) #(N,D)
	
	mesh = np.matmul(mesh_prime, L_inv.T)

		#we don't check the boundaries for the axis that will be drawn at random
	axis_ok = [i for i in range(mesh.shape[1]) if i not in where_random]
	ids_ok = np.logical_and(np.all(mesh[:,axis_ok] >= boundaries[0,axis_ok], axis =1), np.all(mesh[:,axis_ok] <= boundaries[1,axis_ok], axis = 1)) #(N,)
	mesh = mesh[ids_ok,:]
	
		#Whenever there is a single point in the grid, the templates along that dimension will be placed at random
	for id_random in where_random:
		mesh[:,id_random] =np.random.uniform(boundaries[0,id_random], boundaries[1,id_random], (mesh.shape[0], )) # center[id_random] #
	
		####
		#adding the boundaries
		####
		
		#Boundaries are added by creating a mesh in the D-1 plane of the tile boundary
	boundary_mesh = []
		#up_down keeps track whether we are at the min (0) or max (1) value along the d-th dimension
	for up_down, d in bound_list:
		ids_not_d = [d_ for d_ in range(metric.shape[1]) if d_ is not d]
		new_dist = dist*np.sqrt(metric.shape[1]/(metric.shape[1]-1)) #this the distance between templates that must be achieved in the low dimensional manifold
		
			#creating the input for the boundary tiling
		rect_proj = Rectangle( tile[0].mins[ids_not_d], tile[0].maxes[ids_not_d]) #projected rectangle
		metric_proj = metric - np.outer(metric[:,d], metric[:,d]) /metric[d,d]
		metric_proj = metric_proj[tuple(np.meshgrid(ids_not_d,ids_not_d))].T #projected metric on the rectangle
		
		#new_coarse_boundaries = np.stack([rect_proj.mins, rect_proj.maxes], axis =0) #(2,D)
		new_coarse_boundaries = None
		
		new_mesh_ = create_mesh(new_dist, (rect_proj, -metric_proj), new_coarse_boundaries) #(N,D-1) #mesh on the D-1 plane
		boundary_mesh_ = np.zeros((new_mesh_.shape[0], metric.shape[1]))
		boundary_mesh_[:,ids_not_d] = new_mesh_
		boundary_mesh_[:,d] = boundaries[up_down,d]
		
		boundary_mesh.extend(boundary_mesh_)
		
	if len(boundary_mesh)>0:
		boundary_mesh = np.array(boundary_mesh)
		mesh = np.concatenate([mesh,boundary_mesh], axis =0)
	
	return mesh

###########################################################################################

def points_in_hull(points, hull, tolerance=1e-12):
	"Check if points (N,D) are in the hull"
	if points.ndim == 1:
		points = point[None,:]
	
	value_list = [np.einsum('ij,j->i', points, eq[:-1])+eq[-1] for eq in hull.equations]
	value_list = np.array(value_list).T #(N, N_eqs)
	
	return np.prod(value_list<= tolerance, axis = 1).astype(bool) #(N,)

def all_line_hull_intersection(v, c, hull):
	"Compute all the intersection between N_lines and a single hull"
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
	"Sample N_points from a convex hull"
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
	"SamplesN_points from a convex hull. If boundaries are given, it will enforce them"
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
	"Plot the hull and a bunch of additional points"
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
	"Get N points padding the boundary box"
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
	"Get N points padding the boundary box"
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

##########################################################################################
		
		
def save_injs(filename, injs, GPS_start, GPS_end, time_step, approx, luminosity_distance = 100, f_min = 20.):
		"""
		Save the bank to a ligo xml injection file (sim_inspiral table).
		
		Parameters
		----------
			
		filename: str
			Filename to save the injections at
		
		injs: 'np.ndarray'
			Injection array (N,10)
		
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
			
				#this setting fucks things a bit! (is it really required??)
			#row.alpha4 = 20 #This is H1 SNR. Required?
			#row.alpha5 = 20 #This is L1 SNR. Required?
			#row.alpha6 = 20 #This is V1 SNR. Required?

				#appending row
			sim_inspiral_table.append(row)

		ligolw_process.set_process_end_time(process)
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

















	
	
	
