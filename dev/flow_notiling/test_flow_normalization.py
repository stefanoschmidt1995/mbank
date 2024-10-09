import argparse
from argparse import Namespace
from mbank.parser import boundary_keeper
from mbank import cbc_bank, tiling_handler, variable_handler, cbc_metric
from mbank.utils import plot_tiles_templates, avg_dist, load_PSD, plot_colormap
from mbank.placement import place_random_flow, place_stochastically_flow, place_geometric_flow
from mbank.flow import STD_GW_Flow
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm


args = argparse.Namespace()
args.m1_range = (1, 200)
args.m2_range = (1, 200)
args.q_range = (1, 30)
args.chi_range = (0, 1)
args.variable_format = 'm1m2_nonspinning'

n_points = 100000

flow_file = '../../paper/plots/flow_validation/out_m1m2_nonspinning/flow_m1m2_nonspinning.zip'
#flow_file = '../../paper/plots/flow_validation/out_m1m2_chi_e/flow_m1m2_chi_e.zip'

flow = STD_GW_Flow.load_flow(flow_file)

bk = boundary_keeper(args)
coord_vol_flow = np.prod(np.abs(np.diff(flow.boundary_box, axis = 0)))
print('coord vol flow: ', coord_vol_flow)
print('coord vol box: ', bk.volume_box(args.variable_format))
vol_coordinates, err_vol = bk.volume(n_points, args.variable_format, 20)

n_trials = 100
vols = []

for i in tqdm(range(n_trials)):
	#points = bk.sample(n_points, args.variable_format)
	points = np.random.uniform(*flow.boundary_box, (n_points, flow.D))
	print(sum(bk(points, args.variable_format))/n_points)
	
	with torch.no_grad():
		log_pdf = flow.log_prob(points.astype(np.float32)).numpy()

	vol_flow = np.mean(np.exp(log_pdf))*coord_vol_flow
	vols.append(vol_flow)

print('normalization constant of the normalizing flow: ', np.mean(vols), np.std(vols, ddof=1)/np.sqrt(n_trials))

quit()

plt.scatter(*points[:,:2].T, s = 5)
plt.show()

