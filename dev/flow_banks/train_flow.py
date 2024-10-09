import numpy as np
from mbank.handlers import variable_handler, tiling_handler
from mbank.flow.utils import compare_probability_distribution

	#precessing bank
if False:
	variable_format = 'Mq_s1xz'
	folder = 'precessing_bank'
	tiling_file = 'tiling_paper_precessing.npy'
	flow_file = 'flow_paper_precessing.zip'

	#HM bank
if False:
	variable_format = 'logMq_chi_iota'
	folder = 'HM_bank'
	tiling_file = 'tiling_paper_HM.npy'
	flow_file = 'flow_paper_HM.zip'

	#eccentric bank
if True:
	variable_format = 'Mq_nonspinning_e'
	folder = 'eccentric_bank'
	tiling_file = 'tiling_paper_eccentric.npy'
	flow_file = 'flow_paper_eccentric.zip'

t_obj = tiling_handler('{}/{}'.format(folder,tiling_file))
t_obj.train_flow(N_epochs = 1000, verbose = True)

t_obj.save('{}/{}'.format(folder,tiling_file), '{}/{}'.format(folder, flow_file))

compare_probability_distribution(t_obj.flow.sample(5000).detach().numpy(),
	data_true = t_obj.sample_from_tiling(5000),
	variable_format = variable_format,
	title = None, hue_labels = ['flow', 'tiling'],
	savefile = '{}/flow.png'.format(folder), show = False)
