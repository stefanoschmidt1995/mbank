"""
Some attempts to sample from the conditional probability of the flow

One option could be as done [here](https://arxiv.org/abs/2007.06140)

Another one could be importance sampling (easier but more dirty)
"""

import sys
sys.path.insert(0,'..')
import os

from mbank.flow import GW_Flow, TanhTransform
from mbank.flow.utils import compare_probability_distribution

from nflows.distributions.normal import StandardNormal

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 

import torch

from build_flow import Std2DTransform, Std3DTransform

#######################

flow_2d = GW_Flow(transform=Std2DTransform(), distribution=StandardNormal(shape=[2]))
flow_3d = GW_Flow(transform=Std3DTransform(), distribution=StandardNormal(shape=[3]))

flow_2d.load_state_dict(torch.load('standard_flow_2D/weights'))
flow_3d.load_state_dict(torch.load('standard_flow_3D/weights'))

chi = 0

	#drawing uniform samples in the conditional support and do importance sampling
	#Kind of works
boundaries = np.array([[np.log10(10), 1.], [np.log10(40), 7.]])
points_conditional = []
N_random_samples = 500
for _ in tqdm(range(N_random_samples)):
	N_points = 10000
	points = np.random.uniform(*boundaries, (N_points, 2))
	points = np.concatenate([points, np.full((N_points, 1), chi)], axis=1)
	points = torch.tensor(points, dtype = torch.float32)

	probs = flow_3d.log_prob(points)
	probs = torch.exp(probs-torch.max(probs))

	random_point = points[torch.multinomial(probs, 1)[0],:2]
	points_conditional.append(random_point)
points_conditional = torch.stack(points_conditional, axis =0)

points_true = flow_2d.sample(10*N_random_samples).detach().numpy()
ids_ok = np.logical_and(np.all(points_true > boundaries[0,:], axis =1), np.all(points_true < boundaries[1,:], axis = 1))
points_true = points_true[ids_ok,:]

compare_probability_distribution(points_conditional.detach().numpy(), data_true = points_true,
		variable_format = 'logMq_nonspinning', title = None, 
		hue_labels = ['conditional','true'],
		savefile = None, show = True)

quit()
points_true = flow_2d.sample(10*N_random_samples)

compare_probability_distribution(points_true.detach().numpy(), data_true = None,
		variable_format = 'logMq_nonspinning', title = None, savefile = None, show = True)










