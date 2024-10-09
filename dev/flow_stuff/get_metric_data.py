"Given a set of samples, it computes the metric at the given samples and stores it to file"

import numpy as np
from mbank import cbc_metric, cbc_bank
from mbank.utils import load_PSD, plot_tiles_templates

from tqdm import tqdm

######################

variable_format = 'logMq_nonspinning'
f, PSD = load_PSD('../aligo_O3actual_H1.txt', False, 'H1')
metric = cbc_metric(variable_format,
			PSD = (f,PSD),
			approx = 'IMRPhenomD',
			f_min = 15, f_max = 1024)

samples_file = 'data/samples_mcmc_2D.dat'
metric_file = 'data/metric_mcmc_2D.dat'

metric_list = []

f_samples = open(samples_file, 'r')
f_metric = open(metric_file, 'w')

print("Saving shit to ", metric_file)

for l in tqdm(f_samples.readlines(), desc='Computing metric'):
	try:
		theta = l.split(' ')
		theta = np.array(theta, dtype = float)
		theta[0] = np.log10(theta[0])
		metric_theta = metric.get_metric(theta)
		to_save = [[*theta, *metric_theta.flatten()]]
		np.savetxt(f_metric, to_save)
	except KeyboardInterrupt:
		break

f_metric.close()
f_samples.close()


