
import sys
sys.path.insert(0,'..')
import os

from mbank.flow import GW_Flow, TanhTransform
from mbank.flow.utils import plot_loss_functions, create_gif, plotting_callback

from nflows.distributions.normal import StandardNormal

import numpy as np
import matplotlib.pyplot as plt

import torch

from build_flow import Std2DTransform, Std3DTransform

#######################

def integrate_1d(theta1, theta2, flow, N_steps = 100):
	"Old version of the integral, just for 1D tensors"
	steps = theta1 + torch.outer(torch.linspace(0, 1, N_steps), theta2-theta1)
	log_pdfs = flow.log_prob(steps) #(N_steps, )
	log_pdfs = log_pdfs - log_pdfs[0] #(N_steps, )
	det_M = torch.square(torch.exp(log_pdfs)) #(N_steps, )
	integral = torch.trapezoid(det_M, dx =1/N_steps)
	return integral

#######################

flow_2d = GW_Flow(transform=Std2DTransform(), distribution=StandardNormal(shape=[2]))
flow_3d = GW_Flow(transform=Std3DTransform(), distribution=StandardNormal(shape=[3]))

flow_2d.load_state_dict(torch.load('standard_flow_2D/weights'))
flow_3d.load_state_dict(torch.load('standard_flow_3D/weights'))


###
N_steps = 100

theta1 = torch.tensor([[np.log10(20),3, 0.4], [np.log10(16), 3.3, -0.14]], dtype = torch.float32)
theta2 = torch.tensor([[np.log10(21), 1.3, -0.4], [np.log10(26), 3.3, -0.4]], dtype = torch.float32)

#center = torch.tensor([[np.log10(15), 2.3, -0.14], [np.log10(20), 2.3, -0.4]], dtype = torch.float32)
center = torch.tensor([np.log10(15), 2.3, -0.14], dtype = torch.float32)

res = flow_3d.integrate_flow(theta1, theta2, center, N_steps = 100)
print(res)

theta1 = torch.tensor([np.log10(20),3, 0.4], dtype = torch.float32)
theta1 = torch.tensor([[np.log10(20),3, 0.4], [np.log10(16), 3.3, -0.14], [np.log10(20),3, 0.4]], dtype = torch.float32)
theta2 = torch.tensor([np.log10(21), 1.3, -0.4], dtype = torch.float32)

res = flow_3d.integrate_flow(theta1, theta2, center, N_steps = 100)
print(res)
#res = integrate_1d(theta1, theta2, flow_3d, N_steps = 100)
#print(res)

