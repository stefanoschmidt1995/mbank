import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
from scipy.stats import norm


import torch
from torch import optim

from mbank import variable_handler, cbc_metric
from mbank.utils import load_PSD, avg_dist, plot_tiles_templates, get_boundaries_from_ranges, plot_colormap
from mbank.parser import get_boundary_box_from_args, boundary_keeper
import mbank.parser

from mbank.flow import STD_GW_Flow, GW_Flow, TanhTransform

from glasflow.nflows.flows.base import Flow
from glasflow.nflows.distributions.normal import StandardNormal
from glasflow.nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from glasflow.nflows.transforms.linear import NaiveLinear
from glasflow.nflows.transforms.base import Transform, CompositeTransform

D = 2

base_dist = StandardNormal(shape=[D])
transform_list = CompositeTransform([NaiveLinear(features=D), MaskedAffineAutoregressiveTransform(features=D, hidden_features=3)])
flow = Flow(transform=transform_list, distribution=base_dist)

flow = STD_GW_Flow.load_flow('out_nsbh_bank_rectangle/flow_nsbh.zip')
#flow = STD_GW_Flow.load_flow('out_high_dimensional_bank_flow_verylarge/flow_high_dimensional.zip')

samples, ll = flow.sample_and_log_prob(2000)

noise = flow.transform_to_noise(samples)

	#forward
v_fwd = samples
t_list = [t for t in flow._transform._transforms]
for transform in t_list:
	v_fwd, _ = transform(v_fwd)

	#inverse
v_inv = noise
t_list = [t for t in flow._transform._transforms]
for transform in t_list[::-1]:
	v_inv, _ = transform.inverse(v_inv)

print(torch.allclose(v_inv, samples, atol = 0, rtol = 1e-2))

plt.figure()
plt.hist(noise[:,0].detach().numpy(), bins = 100, histtype = 'step', density = True)
plt.hist(v_fwd[:,0].detach().numpy(), bins = 100, histtype = 'step', density = True)
plt.hist(v_inv[:,0].detach().numpy(), bins = 100, histtype = 'step', density = True)
plt.hist(samples[:,0].detach().numpy(), bins = 100, histtype = 'step', density = True)
x = np.linspace(-5, 5, 1000)
plt.plot(x, norm.pdf(x),'r-', lw=2, alpha=0.6, label='norm pdf')

tanh_trans = flow._transform._transforms[0]
transformed_samples, _ = tanh_trans(samples)


with torch.no_grad():
	transformed_samples = transformed_samples.numpy()
	samples = samples.numpy()
	ll = ll.numpy()

plt.figure()
plt.title('samples')
plt.scatter(*samples[:,:2].T, c = ll)
plt.colorbar()

plt.figure()
plt.title('TanhTransform samples')
plt.scatter(*transformed_samples[:,:2].T, c = ll)
plt.colorbar()
plt.show()
