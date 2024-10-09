import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mbank.flow import STD_GW_Flow
from mbank import tiling_handler
from mbank.utils import plot_tiles_templates

########################

samples_file = 'out_flow_paper_precessing/dataset_paper_precessing.dat'
flow_file = 'out_flow_paper_precessing/flow_paper_precessing.zip'
tiling_file = '../../paper/plots/precessing_bank//tiling_paper_precessing.npy'

D = 4
samples = np.loadtxt(samples_file)
samples, ll = samples[:,:D], samples[:,-1]

	#Splitting training and validation
train_fraction = 0.85
train_data, validation_data = samples[:int(train_fraction*len(samples))], samples[int(train_fraction*len(samples)):]
train_ll, validation_ll = ll[:int(train_fraction*len(samples))], ll[int(train_fraction*len(samples)):]

print("Train | Validation data: ", len(train_data), len(validation_data))

flow = STD_GW_Flow.load_flow(flow_file)

with torch.no_grad():
	train_ll_flow = flow.log_prob(torch.Tensor(train_data[:10000])).numpy()
	validation_ll_flow = flow.log_prob(torch.Tensor(validation_data[:3000])).numpy()

		#Loading tiling
t_obj = tiling_handler(tiling_file)
validation_ll_tiling = 0.5*np.log(np.abs(np.linalg.det(t_obj.get_metric(validation_data[:3000], kdtree = True))))
t_vol, _ = t_obj.compute_volume()

t_obj.flow = flow
validation_ll_tiling_flow = 0.5*np.log(np.abs(np.linalg.det(t_obj.get_metric(validation_data[:3000], kdtree = True, flow = True))))

shift = np.nanmedian(train_ll[:len(train_ll_flow)]-train_ll_flow)

print('Volume of the space (flow): ', np.exp(-shift))
print('Volume of the space (tiling): ', t_vol)

plt.figure()
plt.hist((validation_ll_flow - validation_ll[:len(validation_ll_flow)]+shift)/np.log(10),
	histtype = 'step', bins = 100, label = 'flow', density = True)
plt.hist((validation_ll_tiling - validation_ll[:len(validation_ll_tiling)])/np.log(10),
	histtype = 'step', bins = 100, label = 'tiling', density = True)
plt.hist((validation_ll_tiling_flow - validation_ll[:len(validation_ll_tiling_flow)])/np.log(10),
	histtype = 'step', bins = 100, label = 'tiling + flow', density = True)
plt.legend()
plt.yscale('log')
plt.xlabel(r"$\log_{10}(M_{pred}/M_{true})$")

plt.figure()
b_list = [i for i in range(-10,2)]
b_list.insert(0, -np.inf)
b_list.append(np.inf)
log_M_diff = (validation_ll_flow - validation_ll[:len(validation_ll_flow)])+shift

for start, stop in zip(b_list[:-1], b_list[1:]):
	ids_, = np.where(np.logical_and(validation_ll[:len(validation_ll_flow)]/np.log(10)>start, validation_ll[:len(validation_ll_flow)]/np.log(10)<stop))
	print('[{}, {}]: {}'.format(start, stop, len(ids_)))
	#print(log_M_diff[ids_], '\n', validation_ll_flow[ids_],  '\n', validation_ll[ids_])
	if len(ids_)>10:
		plt.hist(log_M_diff[ids_],
			histtype = 'step', bins = int(np.sqrt(len(ids_))), label = r'[{}, {}]'.format(start, stop), density = True)
plt.xlabel(r"$\log_{10}(M_{pred}/M_{true})$")
plt.legend()
plt.show()

