import torch
from torch import optim
import numpy as np
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

folder = 'out_mcmc_analytical_test/'
samples_file = folder+'samples_paper_precessing.dat'
flow_file = folder+'flow_analytical_test.zip'
tiling_file = folder+'tiling_mcmc_analytical_test.npy'

folder = 'out_mcmc_paper_precessing/'
#samples_file = folder+'samples_paper_precessing.dat'
#flow_file = folder+'flow_paper_precessing.zip'
samples_file = 'out_flow_paper_precessing/dataset_paper_precessing.dat'
flow_file = 'out_flow_paper_precessing/flow_paper_precessing.zip'
tiling_file = folder+'tiling_paper_precessing.npy'

samples = np.loadtxt(samples_file)
samples, ll = samples[:,:4], samples[:,-1]

	#Splitting training and validation
train_fraction = 0.9
train_data, validation_data = samples[:int(train_fraction*len(samples))], samples[int(train_fraction*len(samples)):]
train_ll, validation_ll = ll[:int(train_fraction*len(samples))], ll[int(train_fraction*len(samples)):]

print("Train | Validation data: ", len(train_data), len(validation_data))

if flow:
	if not True:
		flow = STD_GW_Flow(samples.shape[-1], n_layers = 3, hidden_features = 30)

		optimizer = optim.Adam(flow.parameters(), lr=0.0005)

		#history = flow.train_flow_forward_KL(N_epochs = 10000, train_data = train_data, validation_data = validation_data,
		#	optimizer = optimizer, batch_size = 10000, validation_step = 20, verbose = True)
		history = flow.train_flow_importance_sampling(N_epochs = 10000,
			train_data = train_data, train_weights = train_ll,
			validation_data = validation_data, validation_weights = validation_ll,
			optimizer = optimizer, batch_size = 20000, validation_step = 20, verbose = True)
		flow.save_weigths(flow_file)
	else:
		flow = STD_GW_Flow.load_flow(flow_file)

	with torch.no_grad():
		train_ll_flow = flow.log_prob(torch.Tensor(train_data[:10000])).numpy()
		validation_ll_flow = flow.log_prob(torch.Tensor(validation_data[:2000])).numpy()

		#Loading tiling
	if True:
		t_obj = tiling_handler(tiling_file)
		validation_ll_tiling = 0.5*np.log(np.abs(np.linalg.det(t_obj.get_metric(validation_data[:1000], kdtree = True))))

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

shift = np.nanmedian(train_ll[:len(train_ll_flow)]-train_ll_flow)
#SBOOOOM: maybe you can use the sample with log(p/p_true))**2 divergence to train the flow and you will be super fine :D
plt.figure()
if flow: plt.hist((validation_ll_flow-validation_ll[:len(validation_ll_flow)]+shift)/np.log(10), histtype = 'step', bins = 100, label = 'flow', density = True)
if flow: plt.hist((validation_ll_tiling - validation_ll[:len(validation_ll_tiling)])/np.log(10), histtype = 'step', bins = 100, label = 'tiling', density = True)
if flow is None: plt.hist(np.log10(validation_ll_tiling/validation_ll)/np.log(10), histtype = 'step', bins = 100, label = 'DPGMM')
plt.legend()
plt.xlabel(r"$\log_{10}(M_{pred}/M_{true})$")

plt.figure()
b_list = [10**i for i in range(-3,4)]
b_list.insert(0, -np.inf)
b_list.append(np.inf)
log_M_diff = (validation_ll_flow/validation_ll[:len(validation_ll_flow)])+shift
for start, stop in zip(b_list[:-1], b_list[1:]):
	ids_, = np.where(np.logical_and(validation_ll[:len(validation_ll_flow)]>start, validation_ll[:len(validation_ll_flow)]<stop))
	print('[{}, {}]: {}'.format(np.log10(start), np.log10(stop), len(ids_)))
	#print(log_M_diff[ids_], '\n', validation_ll_flow[ids_],  '\n', validation_ll[ids_])
	if len(ids_)>10:
		plt.hist(log_M_diff[ids_],
			histtype = 'step', bins = int(np.sqrt(len(ids_))), label = '[{}, {}]'.format(np.log10(start), np.log10(stop)), density = True)
plt.legend()
plt.show()

if flow:
	plot_tiles_templates(validation_data, 'Mq_s1xz',
		tiling = None, dist_ellipse = None, save_folder = None, show = None)

plt.show()




