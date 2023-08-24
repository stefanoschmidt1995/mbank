from torch import optim
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
	from mbank.flow import STD_GW_Flow
	from mbank import tiling_handler
	from mbank.utils import plot_tiles_templates
	flow = True
except (ImportError, RuntimeError):
	flow = None
	from figaro.mixture import DPGMM


samples = np.loadtxt('out_mcmc_analytical_test/samples_paper_precessing.dat')
samples, ll = samples[:,:4], samples[:,4]

	#Splitting training and validation
train_fraction = 0.8
train_data, validation_data = samples[:int(train_fraction*len(samples))], samples[int(train_fraction*len(samples)):]
train_ll, validation_ll = ll[:int(train_fraction*len(samples))], ll[int(train_fraction*len(samples)):]

print("Train | Validation data: ", len(train_data), len(validation_data))

if flow:
	if not True:
		flow = STD_GW_Flow(samples.shape[-1], n_layers = 4, hidden_features = 6)

		optimizer = optim.Adam(flow.parameters(), lr=0.001)

		history = flow.train_flow_forward_KL(N_epochs = 10000, train_data = train_data, validation_data = validation_data,
			optimizer = optimizer, batch_size = 10000, validation_step = 20, verbose = True)
		flow.save_weigths('out_mcmc_analytical_test/flow_analytical_test.zip')
	else:
		flow = STD_GW_Flow.load_flow('out_mcmc_analytical_test/flow_analytical_test.zip')

	with torch.no_grad():
		train_ll_flow = flow.log_prob(torch.Tensor(train_data[:100])).numpy()
		validation_ll_flow = flow.log_prob(torch.Tensor(validation_data[:10000])).numpy()

		#Loading tiling
	if True:
		t_obj = tiling_handler('out_mcmc_analytical_test/tiling_mcmc_analytical_test.npy')
		validation_ll_tiling = 0.5*np.log(np.abs(np.linalg.det(t_obj.get_metric(validation_data[:10000]))))

if flow is None:
	#TAKE HOME MESSAGE:
	#	Flow and DPGMM do the same job. If the accuracy is not satisfactory, the problem is due to the samples...
	#	Maybe you need more samples?
	
	boundaries = np.array([[ 25., 1., 0., 0.  ], [100., 5., 0.99, 3.  ]])
	mix = DPGMM(boundaries.T)
	for s in tqdm(train_data, desc = 'Computing density estimation with DPGMM'):
	    mix.add_new_point(s)
	rec = mix.build_mixture()
	validation_ll_tiling = rec.logpdf(validation_data)

shift = 0# np.log10(train_ll/train_ll_flow).mean()

plt.figure()
if flow: plt.hist(np.log10(validation_ll_flow/validation_ll[:len(validation_ll_flow)])+shift, histtype = 'step', bins = 100, label = 'flow')
if flow: plt.hist(np.log10(validation_ll_tiling/validation_ll[:len(validation_ll_tiling)]), histtype = 'step', bins = 100, label = 'tiling')
if flow is None: plt.hist(np.log10(validation_ll_tiling/validation_ll), histtype = 'step', bins = 100, label = 'DPGMM')
plt.legend()

if flow:
	plot_tiles_templates(validation_data, 'Mq_s1xz',
		tiling = None, dist_ellipse = None, save_folder = None, show = None)

plt.show()




