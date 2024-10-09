import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
from ligo.lw.utils import load_filename

import torch
from torch import optim

from mbank import variable_handler, cbc_metric
from mbank.flow.flowmodel import tau0tau3Transform, TanhTransform, powerlawTransform

from glasflow.nflows.transforms.base import Transform, CompositeTransform


#dataset = np.loadtxt('datasets_high_dimensional/dataset_high_dimensional.dat')
#plt.scatter(dataset[:,0], dataset[:,-1])


#inputs = torch.Tensor([[1, 0.24, 0.9],[10, 0.1, 0.9]])

udist = torch.distributions.uniform.Uniform(torch.tensor([1, 0.01, 0.01]), torch.tensor([100, 0.25, 0.99]))
inputs = udist.sample(sample_shape = [10000])

#trans = tau0tau3Transform()
#trans = powerlawTransform([-10/3., -8/5., -1], [1, 0.01, -0.99], [100, 0.25, 0.99])
#trans_plaw = powerlawTransform([-10/3., -8/5., 2], [1, 0.01, 0.01], [100, 0.25, 0.99])
trans_plaw = powerlawTransform(3, [10/3., 8/5., 2], [1, 0.01, 0.01], [100, 0.25, 0.99])

extrema = torch.stack([trans_plaw(torch.tensor([1, 0.01, 0.01], requires_grad = False))[0],
	trans_plaw(torch.tensor([100, 0.25, 0.99], requires_grad = False))[0]], dim = 0)
low, high = torch.min(extrema, dim = 0).values, torch.max(extrema, dim = 0).values
diff = high-low
low = low - diff*1e-6
high = high + diff*1e-6
trans_tanh = TanhTransform(3, low, high)
print('low, high: ', trans_tanh.low, trans_tanh.high)

trans = CompositeTransform([trans_plaw, trans_tanh])
trans = trans_plaw


tau0tau3, logabsdet = trans(inputs)
#inside = torch.logical_and(torch.prod(tau0tau3>trans_tanh.low, dim = -1), torch.prod(tau0tau3<trans_tanh.high, dim = -1))
#print(torch.all(inside))
mceta, logabsdet_inv = trans.inverse(tau0tau3)

print(inputs, trans.inverse(trans(inputs)[0])[0])
#assert torch.allclose(inputs, trans.inverse(trans(inputs)[0])[0])
#assert torch.allclose(tau0tau3, trans(trans.inverse(tau0tau3)[0])[0])

with torch.no_grad():
	plt.figure()
	plt.scatter(inputs[:,0], logabsdet)
	plt.xlabel("mchirp")
	plt.ylabel("LL")

	plt.figure()
	plt.scatter(*inputs[:,:2].T)
	plt.xlabel("mchirp")
	plt.ylabel("eta")
	
	plt.figure()
	plt.scatter(*tau0tau3[:,:2].T, c = inputs[:,0])
	#plt.scatter(*extrema[:,:2].T)
	plt.xlabel("mchirp transformed")
	plt.ylabel("eta transformed")
	plt.colorbar()

	plt.figure()
	plt.scatter(*tau0tau3[:,[0,2]].T, c = inputs[:,0])
	#plt.scatter(*extrema[:,[0,2]].T)
	plt.ylabel('spin transformed')
	plt.xlabel('mchirp transformed')
	plt.colorbar()

	plt.show()
quit()
with torch.no_grad():
	c = 0
	for i, d in zip(inputs, trans(inputs)[1]):
		print(i.numpy()[:2], d.numpy())
		c+=1
		if c>=100: break
