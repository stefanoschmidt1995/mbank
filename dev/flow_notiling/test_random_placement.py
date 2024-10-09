import mbank
from mbank.utils import plot_tiles_templates
from mbank.placement import 
from mbank.flow import STD_GW_Flow
import numpy as np
import matplotlib.pyplot as plt
import torch
from mbank.placement import place_random_flow
from tqdm import tqdm
import itertools
import scipy

#########################################################################################
################################################################################################################


	#max 45000
N_livepoints = 1500
variable_format = 'm1m2_s1z_s2z'
var_handler = mbank.variable_handler()
D = var_handler.D(variable_format)

flow = STD_GW_Flow.load_flow('out_allsky_bank_flow/flow_allsky_noIMBH.zip')
livepoints_dataset = np.loadtxt('out_allsky_bank_flow/livepoints_allsky_noIMBH.dat')[-N_livepoints:]

print(flow.n_layers, flow.hidden_features)

livepoints, metric = livepoints_dataset[:,:D], livepoints_dataset[:,D:-1]
metric = metric.reshape((N_livepoints, D,D))

	#WEIRD: what the fuck is this??
good_ids, = np.where(np.linalg.det(metric)>0)
livepoints, metric = livepoints[good_ids], metric[good_ids]

	#Checking gridding in the gaussian space
if False:
	N_points = int(1e5**(1/D))+1
	print(N_points, N_points**D)

	samples = scipy.special.erfinv(2*np.linspace(0,1, N_points, endpoint = True)[1:-1] - 1)
	samples = [samples]*D
	grid = []
	for p in itertools.product( *samples):
		grid.append(p)
	grid = torch.tensor(np.array(grid).astype(np.float32))

	samples, _ = flow._transform.inverse(grid)
	plot_tiles_templates(samples.detach().numpy(), variable_format, show = True)


	quit()

new_templates = place_random_flow(0.97, flow, livepoints, metric, covering_fraction = 0.01, verbose = True)
plot_tiles_templates(new_templates, variable_format, show = True)
print(new_templates.shape)


