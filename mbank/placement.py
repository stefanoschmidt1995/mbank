"""
mbank.utils
===========

A lot of placement methods, both for the tiling and the normalizing flow. Most of them are deprecated and are not guaranteed to work properly.
"""

import numpy as np
import scipy
from scipy.spatial import ConvexHull, Rectangle
from tqdm import tqdm

from .utils import dummy_iterator

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
			#max_match = np.exp(-(1-max_match)) #do we need this?

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

#@do_profile(follow = [])
def place_random(minimum_match, tiling, N_livepoints, covering_fraction = 0.01, verbose = True):	
	"""
	Draw templates from the uniform distribution on the manifold. For each proposal, all the livepoints in the ellipse of constant ``minimum_match`` are killed. The iteration goes on until a fraction of ``covering_fraction`` of livepoints are alive.
	It follows `2202.09380 <https://arxiv.org/abs/2202.09380>`_
	
	Parameters
	----------
	
		minimum_match: float
			Minimum match between templates.
		
		tiling: tiling_handler
			Tiling handler that tiles the parameter space
		
		N_livepoints: int
			Number of livepoints to cover the space with
		
		covering_fraction: float
			Fraction of livepoints to be covered before terminating the loop
		
		verbose: bool
			Whether to display the progress bar
	
	Returns
	-------
		new_templates: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			A set of templates generated by the placing algorithm
	
	"""
	assert 0<covering_fraction <=1., "The covering_fraction should be a fraction in (0,1]"
	MM = minimum_match
	dtype = np.float32
	
	livepoints = tiling.sample_from_tiling(N_livepoints,
				dtype = dtype, tile_id = False, p_equal = False)
	new_templates = []
	if tiling.flow:
		import torch
		with torch.no_grad():
			log_pdf_centers = tiling.flow.log_prob(tiling.get_centers().astype(np.float32)).numpy()
	proposal_list, proposal_ids_, log_pdf_theta_list = [], [], []
	
	bar_str = '{} templates placed ({} % livepoints alive)'
	if verbose: it = tqdm(dummy_iterator(), desc = bar_str.format(len(new_templates), 100), leave = True)
	else: it = dummy_iterator()
	
	for _ in it: 
		if len(livepoints)<N_livepoints*covering_fraction: break
		
			#Generating proposals
		if tiling.flow:
			with torch.no_grad():
				if len(proposal_list)==0:
					proposal_list, log_pdf_theta_list = tiling.flow.sample_and_log_prob(1000)
					proposal_list, log_pdf_theta_list = list(proposal_list.numpy()), list(log_pdf_theta_list.numpy())

				#FIXME: this thing is shit!! You should code all these operations inside the tiling!!
				proposal, log_pdf_theta = proposal_list.pop(0), log_pdf_theta_list.pop(0)
					#checking if the proposal is inside the tiling
				if not tiling.is_inside(proposal)[0]: continue

				id_ = tiling.get_tile(proposal, kdtree = True)[0]
				metric = tiling[id_].metric
					
				factor = (2/metric.shape[0])*(log_pdf_theta-log_pdf_centers[id_])
				factor = np.exp(factor)
			
				metric = (metric.T*factor).T
		else:
			if len(proposal_list)==0:
				proposal_list, proposal_ids_ = tiling.sample_from_tiling(1000,
						dtype = dtype, tile_id = True, p_equal = False)
				proposal_list = list(proposal_list)
				proposal_ids_ = list(proposal_ids_)
			proposal, id_ = proposal_list.pop(0), proposal_ids_.pop(0)
			metric = tiling[id_].metric
		
		diff = livepoints - proposal #(N,D)
		L_t = np.linalg.cholesky(metric).astype(dtype) #(D,D)
		diff_prime = scipy.linalg.blas.sgemm(1, diff, L_t)
		dist = np.sum(np.square(diff_prime), axis =1) #(N,) #This is the bottleneck of the computation, as it should be
	
		ids_kill = np.where(dist < 1- MM)[0]

		livepoints = np.delete(livepoints, ids_kill, axis = 0)

		new_templates.append(np.array(proposal, dtype = np.float64))
		if len(new_templates) %100 ==0 and verbose:
			it.set_description(bar_str.format(len(new_templates), round(100*len(livepoints)/N_livepoints, 1)))


	new_templates = np.array(new_templates)
	
	return new_templates

#@do_profile(follow=[])
def place_pruning(minimum_match, tiling, N_points, covering_fraction = 0.01, verbose = True):
	"""
	Given a tiling object, it covers the volume with points and covers them with templates.
	It uses a pruning method, where proposal are chosen from a large set of random points, called livepoints. The bank is created by selecting a proposal from the set of livepoints and removing (killing) the livepoints too close from the proposal. This methods effectively prunes the original set of livepoints, to remove the random points that are too close from each other.
	
	Parameters
	----------
	
		minimum_match: float
			Minimum match between templates.
		
		tiling: tiling_handler
			Tiling handler that tiles the parameter space
		
		N_points: int
			Number of livepoints to cover the space with
		
		covering_fraction: float
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
	
	assert 0<covering_fraction <=1., "The covering_fraction should be a fraction in (0,1]"
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
		if len(livepoints)<=covering_fraction*N_points: break
		if len(new_templates) %100 ==0 and verbose: it.set_description(bar_str.format(N_points -len(livepoints), N_points, len(new_templates)) )
	
	new_templates = np.column_stack([new_templates])
	#if len(livepoints)>0: new_templates = np.concatenate([new_templates, livepoints], axis =0) #adding the remaining livepoints
	
	return new_templates

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
		points = points[None,:]
	
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
