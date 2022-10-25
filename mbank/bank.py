"""
mbank.bank
==========
	Module to implement a bank of gravitational waves signals.
	It implement the class ``cbc_bank`` which provides a large number of functionalities to generate a bank, perform I/O operations on files
"""

import numpy as np
import warnings

	#ligo.lw imports for xml files: pip install python-ligo-lw
from ligo.lw import utils as lw_utils
from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw.utils import process as ligolw_process

from tqdm import tqdm

import ray

import scipy.spatial

from .utils import place_stochastically_in_tile, place_stochastically, place_iterative, place_random
from .utils import DefaultSnglInspiralTable, avg_dist, read_xml, partition_tiling, split_boundaries, plawspace, create_mesh, get_boundary_box

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
	The class implements a bank for compact binary coalescence signals (CBC). A bank is a collection of templates (saved as the numpy array ``bank.templates``). Each template is a row of the array; the columns are specified by a given variable format.
	The available variable formats are listed in ``mbank.handlers.variable_handler``.
	A bank is generated from a tiling object (created internally) that speeds up the template placing. However, the tiling file is not part of a bank and lives as an independent object ``tiling_handler``.
	A bank can be saved in txt or in the std ligo xml file.
	"""
	def __init__(self, variable_format, filename = None):
		"""
		Initialize the bank with a given variable format. If a filename is given, the bank is loaded from file.
		
		Parameters
		----------
		variable_format: str
			How to handle the variables.
			See class variable_handler for more details
			
		filename: str
			Optional filename to load the bank from (if None, the bank will be initialized empty)
		
		"""
		#TODO: start dealing with spins properly from here...
		self.var_handler = variable_handler()
		self.variable_format = variable_format
		self.templates = None #empty bank
		
		assert self.variable_format in self.var_handler.valid_formats, "Wrong variable format given"
		
		if isinstance(filename, str):
			self.load(filename)

		return
	
	@property
	def D(self):
		"""
		The dimensionality of the space
		
		Returns
		-------
			D: float
				Keeps the dimensionality of the space
		"""
		return self.var_handler.D(self.variable_format) #handy shortening
	
	@property
	def placing_methods(self):
		"""
		List all the available placing methods
		
		Returns
		-------
		placing_methods: list
			The available methods for placing the templates
		"""
		return ['uniform', 'qmc', 'geometric', 'iterative', 'stochastic', 'random', 'tile_random', 'geo_stochastic', 'random_stochastic', 'tile_stochastic']
	
	def load(self, filename):
		"""
		Load a template bank from file. They are added to the existing templates (if any).
		
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

			if self.var_handler.format_info[self.variable_format]['e']: warnings.warn("Currently loading from an xml file does not support eccentricity")
				#reading the BBH components
			BBH_components = read_xml(filename, lsctables.SnglInspiralTable)
		
				#making the templates suitable for the bank
			templates_to_add = self.var_handler.get_theta(BBH_components, self.variable_format) #(N,D)
			
		self.add_templates(templates_to_add)

		return

	def _save_xml(self, filename, f_max = 1024., ifo = 'L1'):
		"""
		Save the bank to an xml file suitable for LVK applications

		Parameters
		----------
			
		filename: str
			Filename to save the bank at
		
		f_max: float
			End frequency (in Hz) for the templates
		
		ifo: str
			Name of the interferometer the bank refers to 
		
		"""
			#getting the masses and spins of the rows
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi = self.var_handler.get_BBH_components(self.templates, self.variable_format)
		
		if np.any(e != 0.):
			msg = "Currently xml format does not support eccentricity... The saved bank '{}' will have zero eccentricity".format(filename)
			warnings.warn(msg)
		
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
			comment="A bank of BBH, generated using a metric approach")
		
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
			
			row.f_final = f_max
			row.ifo = ifo #setting the ifo chosen by the user
			
				#Setting additional parameters
			row.process_id = process.process_id #This must be an int
			row.event_id = i
			row.Gamma0 = float(i) #apparently Gamma0 is the template id in gstlal (for some very obscure reason)
			
			#for k, v in std_extra_params.items():
			#	setattr(row, k, v)
			signl_inspiral_table.append(row)
			
		#xmldoc.appendChild(ligolw.LIGO_LW()).appendChild(signl_inspiral_table)
		#ligolw_process.set_process_end_time(process)
		xmldoc.childNodes[-1].appendChild(signl_inspiral_table)
		lw_utils.write_filename(xmldoc, filename, gz=filename.endswith('.xml.gz'), verbose=False)
		xmldoc.unlink()
		
		return
		
	def save_bank(self, filename, f_max = 1024., ifo = 'L1'):
		#TODO: change this name to `save`
		"""
		Save the bank to file
		
		``WARNING: xml file format currently does not support eccentricity``
		
		Parameters
		----------
			
		filename: str
			Filename to save the bank at
		
		f_max: float
			End frequency (in Hz) for the templates (applies only to xml format)
		
		ifo: str
			Name of the interferometer the bank refers to (only applies to xml files)
		
		"""
		if self.templates is None:
			raise ValueError('Bank is empty: cannot save an empty bank!')

		if filename.endswith('.npy'):
			templates_to_add = np.save(filename, self.templates)
		elif filename.endswith('.txt') or filename.endswith('.dat'):
			templates_to_add = np.savetxt(filename, self.templates)
		elif filename.endswith('.xml') or filename.endswith('.xml.gz'):
			self._save_xml(filename, f_max, ifo)
		else:
			raise RuntimeError("Type of file not understood. The file can only end with 'npy', 'txt', 'data, 'xml', 'xml.gx'")
		
		return
	
	def BBH_components(self):
		"""
		Returns the BBH components of the templates in the bank.
		They are: `m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano iota, phi`
		
		Returns
		-------
			BBH_components: :class:`~numpy:numpy.ndarray`
				shape: (N,12)
				Array of BBH components of the templates in the bank. They have the same layout as `variable_handler.get_BBH_components`
		"""
		if self.templates is not None:
			return np.array(self.var_handler.get_BBH_components(self.templates, self.variable_format)).T
		return
		
	def add_templates(self, new_templates):
		"""
		Adds a bunch of templates to the bank. They must be of a shape suitable for the variable format
		
		Parameters
		----------
		
		new_templates: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,)
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
		
	def place_templates(self, tiling, avg_match, placing_method, livepoints = 50, empty_iterations = 100, verbose = True):
		"""
		Given a tiling, it places the templates according to the given method and **adds** them to the bank
		
		Parameters
		----------

		tiling: tiling_handler
			A tiling handler with a non-empty tiling
		
		avg_match: float
			Average match for the bank: it controls the distance between templates as in ``utils.avg_dist()``
		
		placing_method: str
			The placing method to set templates in each tile. It can be:
			
			- `uniform`	-> Uniform drawing in each hyper-rectangle, according to the volume
			- `qmc`	-> Quasi Monte Carlo drawing in each hyper-rectangle, according to the volume
			- `geometric` -> Geometric placement
			- `iterative` -> Each tile is split iteratively until the number of templates in each subtile is equal to one
			- `stochastic` -> Stochastic placement
			- `geo_stochastic` -> Geometric placement + stochastic placement
			- `tile_stochastic` -> Stochastic placement for each tile separately
			- `random` -> The volume is covered with some point that are killed by placing the templates
			- `tile_random` -> Random placement for each tile separately
		
			Those methods are listed in `cbc_bank.placing_methods`
		
		livepoints: float
			The ratio between the number of livepoints and the number of templates placed by ``uniform`` placing method. It only applies to the random placing method
		
		empty_iterations: int
			Number of consecutive proposal inside a tile to be rejected before the tile is considered full. It only applies to the ``stochastic`` placing method.
		
		verbose: bool
			Whether to print the output
		
		Returns
		-------
					
		new_templates: :class:`~numpy:numpy.ndarray`
			The templates generated (already added to the bank)

		"""
		assert placing_method in self.placing_methods, ValueError("Wrong placing method '{}' selected. The methods available are: ".format(placing_method, self.placing_methods))
		assert self.D == tiling[0][0].maxes.shape[0], ValueError("The tiling doesn't match the chosen variable format (space dimensionality mismatch)")
		
			#getting coarse_boundaries from the tiling (to cover boundaries for the bank)
		if placing_method in ['geometric', 'geo_stochastic', 'random', 'random_stochastic'] :
			coarse_boundaries = np.stack([np.min([t_[0].mins for t_ in tiling], axis = 0),
					np.max([t_[0].maxes for t_ in tiling], axis = 0)],
					axis =0) #(2,D)
		
		dist = avg_dist(avg_match, self.D) #desired average distance between templates
		if verbose: print("Approx number of templates {}".format(int(tiling.compute_volume()[0] / np.power(dist, self.D))))
			#total number of points according to volume placement
		N_points = lambda t: livepoints*t.compute_volume()[0] / np.power(np.sqrt(1-avg_match), self.D)
		new_templates = []

		if placing_method in ['stochastic', 'random', 'uniform', 'qmc', 'random_stochastic']: it = iter(())		
		elif verbose: it = tqdm(range(len(tiling)), desc = 'Placing the templates within each tile', leave = True)
		else: it = range(len(tiling))

		for i in it:
			
			t = tiling[i] #current tile
			boundaries_ij = np.stack([t[0].mins, t[0].maxes], axis =0) #boundaries of the tile
			eigs, _ = np.linalg.eig(t[1]) #eigenvalues

				#some sanity checks on the metric eigenvalues
			if np.any(eigs < 0):
				warnings.warn("The metric has a negative eigenvalue: the template placing in this tile may be unreliable. This is pathological as the metric computation may have failed.")
			
			abs_det = np.abs(np.prod(eigs))
			if abs_det < 1e-50: #checking if the determinant is close to zero...
				msg = "The determinant of the metric is zero! It is impossible to place templates into this tile: maybe the approximant you are using is degenerate with some of the sampled quantities?\nRectangle: {}\nMetric: {}".format(t[0], t[1])
				raise ValueError(msg)
			
			if placing_method in ['geometric', 'geo_stochastic']:
					#if stochastic option is set, we create a first guess for stochastic placing method 
				#new_templates_ = create_mesh(dist, t, coarse_boundaries = None) #(N,D)
				new_templates_ = create_mesh(2*np.sqrt(1-avg_match), t, coarse_boundaries = None) #(N,D)
			
			elif placing_method == 'iterative':
				new_templates_ = place_iterative(avg_match, t)
			elif placing_method == 'tile_stochastic':
				new_templates_ = place_stochastically_in_tile(avg_match, t)
			elif placing_method == 'tile_random':
				temp_t_ = tiling_handler(t)
				new_templates_ = place_random(avg_match, temp_t_, N_points = int(N_points(temp_t_)), tolerance = 0.0001, verbose = False)
		
			new_templates.extend(new_templates_)

		if placing_method in ['uniform', 'qmc']:
			vol_tot, _ = tiling.compute_volume()
			N_templates = int( vol_tot/(dist**self.D) )+1
			if tiling.flow and placing_method == 'uniform': new_templates = tiling.sample_from_flow(N_templates)
			else: new_templates = tiling.sample_from_tiling(N_templates, qmc = (placing_method=='qmc'))
			
		if placing_method in ['random', 'random_stochastic']:
				#As a rule of thumb, the fraction of templates/livepoints must be below 10% (otherwise, bad injection recovery)
			N_points_max = int(1e6)
			N_points_tot = N_points(tiling)

			if N_points_tot >N_points_max:
				thresholds = plawspace(coarse_boundaries[0,0], coarse_boundaries[1,0], -8./3., int(N_points_tot/N_points_max)+2)[1:-1]
				partition = partition_tiling(thresholds, 0, tiling)
				#print("\tThresholds: ",thresholds)
				#print("\tN_points: ", [int(N_points(p)) for p in partition])
			else:
				partition = [tiling]

			#print(N_points_tot, len(partition))
			new_templates = []
			if verbose: it = tqdm(partition, desc = 'Loops on the partitions for random placement')
			else: it = partition
			for p in it:
				#TODO: make this a ray function? Too much memory expensive, probably...
					#The template volume for random is sqrt(1-MM) (not dist)
				
				new_templates_ = place_random(avg_match, p, N_points = int(N_points(p)), tolerance = 0.0001, verbose = verbose)
				
				new_templates.extend(new_templates_)
			
		if placing_method in ['geo_stochastic', 'random_stochastic', 'stochastic']:
			new_templates = place_stochastically(avg_match, tiling,
					empty_iterations = empty_iterations,
					seed_bank =  None if placing_method == 'stochastic' else new_templates, verbose = verbose)
		#TODO: find a nice way to set free parameters for placing methods stochastic and random

		new_templates = np.stack(new_templates, axis =0)
		self.add_templates(new_templates)
		return new_templates

	def generate_bank(self, metric_obj, avg_match, boundaries, tolerance,
			placing_method = 'random', metric_type = 'hessian', grid_list = None, train_flow = False,
			use_ray = False, livepoints = 50, empty_iterations = 100, max_depth = 6, n_layers = 2, hidden_features = 4, N_epochs = 1000):
		#FIXME: here you should use kwargs, directing the user to the docs of other functions?
		"""
		Generates a bank using a hierarchical hypercube tesselation. 
		The bank generation consists in two steps:
		
		1. Tiling generation by iterative splitting of the parameter space
		2. Template placing in each tile, according to the method given in ``placing_method``
		
		Parameters
		----------
		
		metric_obj: cbc_metric
			A cbc_metric object to compute the match with

		avg_match: float
			Average match between templates
		
		boundaries: :class:`~numpy:numpy.ndarray`
			shape: (2,D) -
			An array with the boundaries for the model. Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]
		
		tolerance: float
			Threshold used for the tiling algorithm. It amounts to the maximum tolerated relative change between the metric determinant of the child and the parent ``|M|``.
			For more information, see `mbank.handlers.tiling_handler.create_tiling`
		
		placing_method: str
			The placing method to set templates in each tile. See `place_templates` for more information.
		
		metric_type: str
			The method computation method to use. For more information, you can check ``metric.cbc_metric.get_metric``.
		
		train_flow: bool
			Whether to train a normalizing flow model after the tiling is generated. It will be used for metric interpolation during the template placing
		
		grid_list: list
			A list of ints, each representing the number of coarse division of the space.
			If use ray option is set, the subtiling of each coarse division will run in parallel
			If None, no prior splitting will be made.

		use_ray: bool
			Whether to use ray to parallelize
		
		livepoints: float
			The ratio between the number of livepoints and the number of templates placed by ``random`` placing method. It only applies to the random placing method
		
		empty_iterations: int
			Number of consecutive proposal inside a tile to be rejected before the tile is considered full. It only applies to the ``stochastic`` placing method.
		
		max_depth: int
			Maximum number of splitting before quitting the iteration. If None, the iteration will go on until the volume condition is not met
		
		n_layers: int
			Number of layers of the flow
			See `mbank.flow.STD_GW_flow` for more information

		hidden_features: int
			Number of hidden features for the masked autoregressive flow in use.
			See `mbank.flow.STD_GW_flow` for more information
		
		N_epochs: int
			Number of epochs for the training of the flow
		
		Returns
		-------
		
		tiling: tiling_handler 
			A list of tiles used for the bank generation
		"""
			##
			#Initialization & various checks
		assert avg_match<1. and avg_match>0., "`avg_match` should be in the range (0,1)!"
		if avg_match <0.9:
			warnings.warn("Average match is set to be smaller than 0.9. Although the code will work, this can give unreliable results as the metric match approximation is less accurate.")
		dist = avg_dist(avg_match, self.D) #desired average distance in the metric space between templates
		
		if self.variable_format.startswith('m1m2_'):
			raise RuntimeError("The current placing method does not support m1m2 format for the masses")
		
		if grid_list is None: grid_list = [1 for i in range(self.D)]
		assert len(grid_list) == self.D, "Wrong number of grid sizes. Expected {}; given {}".format(self.D, len(grid_list))
		
			###
			#creating a proper grid list for a coarse boundary creation
		boundaries_list = split_boundaries(boundaries, grid_list, use_plawspace = True)
		
			###
			#creating the tiling
		metric_fun = lambda center: metric_obj.get_metric(center, overlap = False, metric_type = metric_type)
									#metric_type = 'hessian')
									#metric_type = 'block_diagonal_hessian')
									#metric_type = 'parabolic_fit_hessian', target_match = 0.9, N_epsilon_points = 10, log_epsilon_range = (-4, 1))
		t_obj = tiling_handler() #empty tiling handler
		t_obj.create_tiling_from_list(boundaries_list, tolerance, metric_fun, max_depth = max_depth, use_ray = use_ray )	
		
		if train_flow: t_obj.train_flow(N_epochs = N_epochs, n_layers = n_layers, hidden_features =  hidden_features, verbose = True)
		
			##
			#placing the templates
			#(if there is KeyboardInterrupt, the tiling is returned anyway)
		try:
			self.place_templates(t_obj, avg_match, placing_method = placing_method, livepoints = livepoints, empty_iterations = empty_iterations, verbose = True)
		except KeyboardInterrupt:
			self.templates = None
		
		return t_obj
				
	def enforce_boundaries(self, boundaries):
		"""
		Remove from the bank the templates that do not lie within the given boundaries
		
		Parameters
		----------

		boundaries: :class:`~numpy:numpy.ndarray`
			shape: (2,4)/(2,2) -
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


	def generate_bank_mcmc(self, metric_obj, N_templates, boundaries, n_walkers = 100, use_ray = False, thin_factor = None, load_chain = None, save_chain = None, verbose = True):
		"""
		Fills the bank with a Markov Chain Monte Carlo (MCMC) method.
		The MCMC sample from the probability distribution function induced by the metric:
		
		.. math::
		
			p(theta) \propto \sqrt(|M(theta)|)
		
		The function uses `emcee` package, not in the `mbank` dependencies.
		
		Parameters
		----------

		metric_obj: cbc_metric
			A cbc_metric objec to compute the PDF to distribute the templates
		
		N_templates: int
			Number of new templates to sample from the PDF
		
		boundaries: :class:`~numpy:numpy.ndarray`
			shape: (2,D) -
			An array with the boundaries for the model. Lower limit is boundaries[0,:] while upper limits is boundaries[1,:]

		n_walkers: int
			Number of independent walkers during the chain. If `use_ray` option is `True`, they will be run in parellel.
		
		use_ray: bool
			Whether to use `ray` to parallelize the sampling.
			#DO THIS OPTION
		
		thin_factor: int
			How many MC steps to discard before selecting one.
			If `None` it is computed authomatically based on the autocorrelation: this is the recommended behaviour 
		
		load_chain: str
			Path to a file where the position of each walker is stored, togheter with integrated aucorellation tau.
			The file must keep a np.array of dimensions (n_walkers, D). The first line of the file is intended to be the autocorrelation time for each variable. If it is not provided, a standard value of 4 (meaning a thin step of 2) is assumed.
			If set, the sampler will start from there and the burn-in phase will not be required.
		
		save_chain: str
			If not None, it saves the path in which to save the status of the sampler.
			The file saved is ready to be loaded with option `load_chain`
		
		verbose: bool
			whether to print to screen the output
		"""
		try:
			import emcee
		except ModuleNotFoundError:
			raise ModuleNotFoundError("Unable to sample from the metric PDF as package `emcee` is not installed. Please try `pip install emcee`")
		
		burnin_steps = lambda tau: int(2 * np.max(tau)) if burnin else 0
		
			#initializing the sampler
		sampler = emcee.EnsembleSampler(n_walkers, self.D, metric_obj.log_pdf, args=[boundaries], vectorize = True)
		
			#tau is a (D,) vector holding the autocorrelation for each variable
			#it is used to estimate the thin factor
		
		if isinstance(load_chain, str):
			#this will output an estimate of tau and a starting chain. The actual sampling will start straight away
			burnin = False
			loaded_chain = np.loadtxt(load_chain)
			if loaded_chain.shape[0]<n_walkers:
				raise RuntimeError("The given input file does not have enough walkers. Required {} but given {}".format(n_walkers, loaded_chain.shape[0]))
			elif loaded_chain.shape[0] == n_walkers:
				start = loaded_chain
				tau = 4 + np.zeros((self.D,))
			else:
				tau, start = loaded_chain[0,:], loaded_chain[1:n_walkers+1,:]
			print('tau', tau)
			assert start.shape == (n_walkers, self.D), "Wrong shape for the starting chain. Expected {} but given {}. Unable to continue".format((n_walkers, self.D), start.shape)
		else:
			burnin = True
			start = np.random.uniform(*boundaries, (n_walkers, self.D))
		n_burnin = 0		
			###########
			#This part has two purposes:
			#	- Give a first estimation for tau parameters (required to decide the size of burn-in steps and the thin step)
			#	- Do a burn in phase (discard some samples to achieve stationariety)
			###########
			
		if burnin:
			tau_list = []
			step = 30

			def dummy_generator(): #dummy generator for having an infinite loop
				while True: yield
			
			if verbose:
				it_obj = tqdm(dummy_generator(), desc='Burn-in/calibration phase')
			else:
				it_obj = dummy_generator()

			for _ in it_obj:
				n_burnin += step
				state = sampler.run_mcmc(start, nsteps = step, progress = False, tune = False)
				start = state.coords #very important! The chain will start from here
						
				tau = sampler.get_autocorr_time(tol = 0)
				tau_list.append(tau)
				
				if len(tau_list)>1 and np.all(np.abs(tau_list[-2]-tau_list[-1]) < 0.001*tau_list[-1]):
					tau = tau_list[-1]
					break
			if verbose: print("")


			###########
			#doing the actual sampling
		if thin_factor is None:
			thin = max(int(0.5 * np.min(tau)),1)
		else:
			thin = thin_factor
		
		
		if verbose: print('Thin factor: {} | burn-in: {} '.format( thin, burnin_steps(tau)))

		n_steps = int((N_templates*thin)/n_walkers) - int(n_burnin) + burnin_steps(tau) + thin #steps left to do...
		if verbose: print("Steps done: {} | Steps to do: {}".format(n_burnin, n_steps))
		
		if n_steps > 0:
			try:
				state = sampler.run_mcmc(start, n_steps, progress = verbose, tune = True)
			except KeyboardInterrupt:
				pass
	
			#FIXME: understand whether you want to change the thin factor... it is likely underestimated during the burn-in phase
			#On the other hand, it makes difficult to predict how many steps you will need
			#updating thin factor
			#FIXME: this needs to be taken into account!
		if thin_factor is None and False:
			tau = sampler.get_autocorr_time(tol = 0)
			thin = max(int(0.5 * np.min(tau)),1)
			if verbose: print('Updated -- Thin factor: {} | burn-in: {} '.format( thin, burnin_steps(tau)))

		chain = sampler.get_chain(discard = burnin_steps(tau), thin = thin, flat=True)[-N_templates:,:]
		
		if isinstance(save_chain, str) and (state is not None):
			chain_to_save = state.coords #(n_walkers, D)
			to_save = np.concatenate([tau[None,:], chain_to_save], axis = 0)
			np.savetxt(save_chain, to_save)

			#adding chain to the bank
		self.add_templates(chain)
				
		return
