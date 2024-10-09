from mbank.flow import TanhTransform

import numpy as np

import torch

import matplotlib.pyplot as plt

####################################

data = 'data/samples_mcmc_2D.dat'

data = np.loadtxt(data)
data = np.delete(data, np.where(data[:,1]<1.3), axis =0)


transf = TanhTransform(low=[10,1.3], high=[40,7])

transformed_data, _ = transf.forward(torch.tensor(data))
transformed_data = transformed_data.detach().numpy()


plt.figure()
plt.scatter(*data.T, s = 2)
plt.savefig('/home/stefano/Dropbox/Stefano/PhD/presentations/norm_flows_glasgow/real_data.png')
plt.figure()
plt.scatter(*transformed_data.T, s = 2)
plt.savefig('/home/stefano/Dropbox/Stefano/PhD/presentations/norm_flows_glasgow/transformed_data.png')
plt.show()
