"""
mbank.bank
==========
	Keeps the cbc_bank class which implements a cbc bank
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

	#ligo.lw imports for xml files: pip install python-ligo-lw
from ligo.lw import utils as lw_utils
from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw.utils import process as ligolw_process

import poisson_disc #https://pypi.org/project/poisson-disc/

from tqdm import tqdm

import ray

	#for older versions of the code...
#import emcee

import scipy.spatial

from .utils import plawspace, create_mesh, get_boundary_box, plot_tiles_templates, get_cube_corners, place_stochastically, place_stochastically_globally, DefaultSnglInspiralTable

from .handlers import variable_handler, tiling_handler
from .metric import cbc_metric

############
#TODO: create a package for placing N_points in a box with lloyd algorithm (extra)
############

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
####################################################################################################################

class cbc_bank():
	"""
	This implements a bank for compact binary coalescence signals (CBC). A bank is a collection of templates (saved as the numpy array bank.templates) that holds masses and spins of the templates.
	A bank is generated from a tiling object (created internally): each template belongs to a tile. This can be useful to speed up the match computation
	It can be generated with a MCMC and can be saved in txt or in the std ligo xml file.
	The class implements some methods for computing the fitting factor with a number of injections.
	"""
	def __init__(self, variable_format, filename = None):
		"""
		Initialize the bank
		
		Parameters
		----------
		variable_format: 'str'
			How to handle the spin variables.
			See class variable_handler for more details
			
		filename: 'str'
			Optional filename to load the bank from (if None, the bank will be initialized empty)
		
		"""
		#TODO: start dealing with spins properly from here...
		self.var_handler = variable_handler()
		self.variable_format = variable_format
		self.templates = None #empty bank
		
		self.D = self.var_handler.D(self.variable_format) #handy shortening
		
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
		#return np.sqrt((1-avg_match)/self.D)
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

			if not self.var_handler.format_info[self.variable_format]['e']: warnings.warn("Currently loading from an xml file does not support eccentricity")
			
				#for some reason this content handles is needed... Why??
			@lsctables.use_in
			class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
				pass
			lsctables.use_in(LIGOLWContentHandler)

			xmldoc = lw_utils.load_filename(filename, verbose = False, contenthandler = LIGOLWContentHandler)
			sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)
		
			BBH_components = []
			
			for row in sngl_inspiral_table:
				BBH_components.append([row.mass1, row.mass2, row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z, 0, 0, row.alpha3, row.alpha5])
			
			BBH_components = np.array(BBH_components) #(N,12)
			
				#making the templates suitable for the bank
			templates_to_add = self.var_handler.get_theta(BBH_components, self.variable_format) #(N,D)
			
		self.add_templates(templates_to_add)

		return

	def _save_xml(self, filename, ifo = 'L1'):
		"""
		Save the bank to an xml file suitable for LVK applications
		
		Parameters
		----------
			
		filename: 'str'
			Filename to save the bank at
		
		ifo: 'str'
			Name of the interferometer the bank refers to 
		
		"""
			#getting the masses and spins of the rows
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi = self.var_handler.get_BBH_components(self.templates, self.variable_format)
		
		if np.any(e != 0.):
			warnings.warn("Currently xml format does not support eccentricity... The saved bank {} will have zero eccentricity".format(filename))
		
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
			row.ifo = ifo #setting the ifo chosen by the user
			
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
		
	def save_bank(self, filename, ifo = 'L1'):
		"""
		Save the bank to file
		
		Parameters
		----------
			
		filename: str
			Filename to save the bank at
		
		ifo: 'str'
			Name of the interferometer the bank refers to (only applies to xml files)
		
		"""
		if self.templates is None:
			raise ValueError('Bank is empty: cannot save an empty bank!')

		if filename.endswith('.npy'):
			templates_to_add = np.save(filename, self.templates)
		elif filename.endswith('.txt') or filename.endswith('.dat'):
			templates_to_add = np.savetxt(filename, self.templates)
		elif filename.endswith('.xml') or filename.endswith('.xml.gz'):
			self._save_xml(filename, ifo)
		else:
			raise RuntimeError("Type of file not understood. The file can only end with 'npy', 'txt', 'data, 'xml', 'xml.gx'.")
		
		return

	def add_templates(self, new_templates):
		"""
		Adds a bunch of templates to the bank.
		They will be saved in the format set by variable_format
		
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
		
		new_templates = self.var_handler.switch_BBH(new_templates, self.variable_format)
		
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
	
	def place_templates(self, t_obj, avg_match, placing_method, verbose = True):
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
		
		Returns
		-------
					
		tile_id_population: 'list' 
			A list of list. 
			tile_id_population[i] keeps the ids of the templates inside tile i
		"""
		assert placing_method in ['p_disc', 'uniform', 'geometric', 'iterative', 'stochastic', 'pure_stochastic'], ValueError("Wrong placing method selected")
		assert self.D == t_obj[0][0].maxes.shape[0], ValueError("The tiling doesn't match the chosen variable format (space dimensionality mismatch)")
		
			#getting coarse_boundaries from the tiling
		#if placing_method in ['geometric', 'uniform']:
		coarse_boundaries = np.min([t_[0].mins for t_ in t_obj], axis = 0) #(D,)
		coarse_boundaries = np.stack([coarse_boundaries, np.max([t_[0].maxes for t_ in t_obj], axis = 0)], axis =0) #(2,D)
		
		avg_dist = self.avg_dist(avg_match) #desired average distance between templates
		new_templates = []
		tile_id_population = [] #for each tile, this stores the templates inside it
		
		if verbose: it = tqdm(t_obj, desc = 'Placing the templates within each tile')
		else: it = t_obj
		
		for t in it:
			boundaries_ij = np.stack([t[0].mins, t[0].maxes], axis =0) #boundaries of the tile
			eigs, _ = np.linalg.eig(t[1]) #determinant of the metric

				#some sanity checks on the metric eigenvalues
			if np.any(eigs > 0):
				warnings.warn("The metric has a positive eigenvalue: the template placing in this tile may be unreliable. This is pathological as the metric computation may have failed. You may improve the stability of the computation by increasing the order of differentiation.")
			
			abs_det = np.abs(np.prod(eigs))
			if abs_det < 1e-50: #checking if the determinant is close to zero...
				raise ValueError("The determinant of the metric is zero! It is impossible to place templates into this tile: maybe the approximant you are using is degenerate with some of the sampled quantities?")
				
			volume_factor = np.sqrt(abs_det)
			
			if placing_method == 'p_disc':
					#radius controls the relative distance between templates (not the radius of the circle!)
				radius = 0.5* avg_dist/np.power(volume_factor, 1/self.D)
				new_templates_ = poisson_disc.Bridson_sampling((boundaries_ij[1,:]-boundaries_ij[0,:]), radius = radius) + boundaries_ij[0,:]
				
				#print(new_templates_.shape[0], len(create_mesh(avg_dist, t )))
				
			elif placing_method == 'uniform':
					#N_templates is computed with a mesh, more realistic...
				N_templates = int(t_obj.N_templates(*t, avg_dist)+1) #Computed by tiling_handler
				new_templates_ = np.random.uniform(*boundaries_ij, (N_templates, self.D))
				
			elif placing_method == 'geometric' or placing_method == 'stochastic':
					#if stochastic option is set, we create a first guess for stochastic placing method 
				new_templates_ = create_mesh(avg_dist, t, coarse_boundaries = None) #(N,D)
			
			elif placing_method == 'pure_stochastic':
				break
			
			elif placing_method == 'iterative':
				temp_t_obj = tiling_handler()
				temp_metric_fun = lambda theta: t[1]

				temp_t_obj.create_tiling((t[0].mins, t[0].maxes), 1, temp_metric_fun, avg_dist, verbose = False, worker_id = None)
				
				new_templates_ = temp_t_obj.get_centers()
			#elif placing_method == 'stochastic':
				#new_templates_ = place_stochastically(avg_match, t)
				#break
		
			tile_id_population.append([i for i in range(len(new_templates), len(new_templates)+ len(new_templates_)) ])
			new_templates.extend(new_templates_)

		if placing_method == 'pure_stochastic' or placing_method == 'stochastic':
			if len(new_templates) == 0: new_templates = None #pure stochastic
			new_templates = place_stochastically_globally(avg_match, t_obj, empty_iterations = 400/self.D, first_guess = new_templates) #this is if I have a first guess
			for t in tqdm(t_obj, desc='Computing the tile which each template belongs to', leave = True):
				dist_t = t[0].min_distance_point(new_templates)
				tile_id_population.append( np.where(dist_t == 0.)[0] )

		new_templates = np.stack(new_templates, axis =0)
		self.add_templates(new_templates)
		return tile_id_population #shall I save it somewhere??	

	def create_grid_tiling(self):
		"This would be equivalent to the old version of the code"
		return
	
	def _generate_tiling(self, metric_obj, coarse_boundaries, avg_dist, N_temp, use_ray = False ):
		"""
		Creates a tiling of the space, starting from a coarse tile.
		
		Parameters
		----------
		
		metric_obj: 'cbc_metric'
			A cbc_metric object to compute the match with

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
			
	def generate_bank(self, metric_obj, avg_match, boundaries, grid_list, N_temp = 200, placing_method = 'geometric', plot_folder = None, use_ray = False, show = True):
		"""
		Generates a bank using a hierarchical hypercube tesselation. 
		It works only if variable format includes M and q
		
		Parameters
		----------

		metric_obj: cbc_metric
			A cbc_metric object to compute the match with

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
		
		show: bool
			Whether to show the plotted output
		
		Returns
		-------
		
		tiling: 'tiling_handler' 
			A list of tiles used for the bank generation
		
		tile_id_population: 'list' 
			A list of list. 
			tile_id_population[i] keeps the ids of the templates inside tile i
			
		"""
		#TODO: add an option to avoid the hierarchical tiling??
			##
			#Initialization
		avg_dist = self.avg_dist(avg_match) #desired average distance in the metric space between templates
		
		if use_ray: ray.init()
		
		if self.variable_format.startswith('m1m2_'):
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
			#(if there is KeyboardInterrupt, the tiling is returned anyway)
		try:
			tile_id_population = self.place_templates(t_obj, avg_match, placing_method = placing_method, verbose = True)
		except KeyboardInterrupt:
			tile_id_population = None
			plot_folder	= None
			self.templates = None
		
			##
			#plot debug
		if isinstance(plot_folder, str):
			plot_tiles_templates(t_obj, self.templates, self.variable_format, plot_folder, show = show)
		
		return t_obj, tile_id_population
				
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

		metric_obj: cbc_metric
			A cbc_metric objec to compute the PDF to distribute the templates
		
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
