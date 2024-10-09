"""
Attempt to place templates using the flow as both proposal generator and metric determinant estimator. Things may work but they are super slow: we are sampling a large space and the sampling from a sub-manifold of the whole space is very expensive...
"""
import sys
sys.path.insert(0,'..')
from mbank.flow import GW_Flow
from mbank.utils import avg_dist

from mbank import cbc_metric, cbc_bank
from mbank.utils import load_PSD, plot_tiles_templates


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 

import torch
from torch import nn
from torch import optim

from nflows.distributions.normal import StandardNormal

from build_flow import Std2DTransform, Std3DTransform, Std6DTransform, Std5DTransform, Std8DTransform, GW_SimpleRealNVP

from scipy.spatial import KDTree

###################################

variable_format = 'logMq_nonspinning'
f, PSD = load_PSD('../H1L1-REFERENCE_PSD-1164556817-1187740818.xml.gz', False, 'L1')
metric = cbc_metric(variable_format,
			PSD = (f,PSD),
			approx = 'IMRPhenomD',
			f_min = 15, f_max = 1024)

D = 2
flow = GW_Flow(transform=Std2DTransform(), distribution=StandardNormal([D]) )
flow.load_state_dict(torch.load('standard_flow_2D/weights'))

center = np.array([[np.log10(30), 3.]])
log_pdf_center_true = metric.log_pdf(center)
log_pdf_center_flow = flow.log_prob(torch.tensor(center, dtype = torch.float32)).detach().numpy()
log_pdf_factor = log_pdf_center_true-log_pdf_center_flow
metric_scaling = metric.get_metric(center)[0,...]
metric_scaling = metric_scaling/np.linalg.det(metric_scaling)
L_t = np.linalg.cholesky(metric_scaling)


dist_templates = avg_dist(0.97, D)

templates = flow.sample(1).detach().numpy()

templates_prime = np.matmul(templates, L_t)

n_discarded = 0

def dummy_iterator():
	while True:
		yield
desc = "Bank size: {}"
it_ = tqdm(dummy_iterator(), desc = desc.format(templates.shape[0]))


for _ in it_:
	try:
		#tree = KDTree(templates_prime)
		
		proposal = flow.sample(1)
		if proposal[0,0]<np.log10(20): continue
		proposal_prime = np.matmul(proposal.detach().numpy(), L_t)
		
		metric_det = np.square(np.exp(flow.log_prob(proposal).detach().numpy()+log_pdf_factor))

		dist = np.sqrt(metric_det) * np.linalg.norm(templates_prime - proposal_prime, axis = 1)

			#DEBUG prints
		#m =  metric.get_metric(np.squeeze(proposal.detach().numpy()))
		#print("\ndets:",np.squeeze(metric_det), np.squeeze(metric.get_metric_determinant(proposal.detach().numpy())))
		#print(dist, '\n',np.sqrt(1-metric.metric_match(templates, proposal.detach().numpy()[0,:], m)))

		if not np.any(dist<dist_templates):
			templates = np.concatenate([templates, proposal.detach().numpy()], axis = 0)
			templates_prime = np.concatenate([templates_prime, proposal_prime], axis = 0)
			n_discarded = 0
			it_.set_description(desc.format(templates.shape[0]))
		else:
			n_discarded +=1
		
		if n_discarded >=100: break
	except KeyboardInterrupt:
		print("KeyboardInterrupt: quitting the loop")
		break

bank = cbc_bank(variable_format)
bank.add_templates(templates)

plot_tiles_templates(bank.templates, variable_format)
plt.show()





	
	
