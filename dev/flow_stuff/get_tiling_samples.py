import numpy as np

import sys
sys.path.insert(0,'..')

from mbank import cbc_bank, cbc_metric
from mbank.utils import load_PSD, get_boundaries_from_ranges, plot_tiles_templates, split_boundaries
from mbank.handlers import tiling_handler
from mbank.flow.utils import compare_probability_distribution, plot_loss_functions

import matplotlib.pyplot as plt

#########################################

generate_tiling = not True
train_flow = False
variable_format =  'Mq_s1xz_s2z'
tiling_file = 'tilings/tiling_{}.npy'.format(variable_format)
flow_file = 'tilings/flow_{}'.format(variable_format)
dataset_file = 'data/samples_tiling_5D.dat'

N_dataset = 30000

if generate_tiling:

	approximant = 'IMRPhenomPv2'
	M_range = (10, 40)
	q_range = (1, 4)
	s_range = (-0.9, 0.9)
	e_range = (0., 0.5)
	
	epsilon = 1.
	
	boundaries = get_boundaries_from_ranges(variable_format, M_range, q_range, s_range, s_range, e_range = e_range)
	boundaries_list = split_boundaries(boundaries, [1,1,3,3,3], use_plawspace = True)
	
	psd = '../aligo_O3actual_H1.txt' 
	ifo = 'H1'
	f_min, f_max = 10., 1024.
	N_injs, N_neigh_templates = 5000, 75
		
	m_obj = cbc_metric(variable_format,
			PSD = load_PSD(psd, True, ifo),
			approx = approximant,
			f_min = f_min, f_max = f_max)

	t = tiling_handler() #emptying the handler... If the split is not volume based, you should start again with the tiling
	t.create_tiling_from_list(boundaries_list, epsilon, m_obj.get_metric, max_depth = 400, verbose = True)
	
	t.save(tiling_file)
else:
	t = tiling_handler(tiling_file)

samples = t.sample_from_tiling(N_dataset)
np.random.shuffle(samples)

np.savetxt(dataset_file, samples)

centers = t.sample_from_tiling(10)

if train_flow:
	history = t.train_flow(N_epochs =1000, N_train_data = 30000, n_layers = 10, hidden_features = 3, lr = 0.001, verbose = True)
	t.save(tiling_file, flow_file)
else:
	t.load_flow(flow_file, n_layers = 10, hidden_features = 3)

print(np.allclose(t.get_metric(t[3].center, True),t.get_metric(t[3].center, False)))

quit()
#plot_tiles_templates(samples, variable_format, show = True)

compare_probability_distribution(t.flow.sample(1000).detach().numpy(), data_true = t.sample_from_tiling(1000),
	variable_format = variable_format,
	title = None, hue_labels = ['flow', 'tiling'],
	savefile = 'tilings/train_plot_{}.png'.format(variable_format), show = True)
plot_loss_functions(history, savefolder = 'tilings')
plt.show()




