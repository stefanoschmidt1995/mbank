import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from mbank.metric import cbc_metric
from mbank.utils import load_PSD

import pickle
import os, sys

#########################################

def get_metric_accuracy_data(metric_obj, MM_list, boundaries, N_points, overlap = False):
	"Given a metric object and some boundaries, it draws random points and for each point it draws a random point at constant metric distance. The metric distance will be then compared with the actual distance. The output is stored on a dict"
	
		#initializing out_dict
	out_dict = {'theta1': np.zeros((N_points,metric_obj.D)),
				'theta2': np.zeros((N_points,metric_obj.D, len(MM_list))),
				'MM_list': MM_list,
				'overlap': overlap,
				'variable_format': metric_obj.variable_format,
				'boundaries': boundaries,
				'approximant': metric_obj.approx,
				'f_range': (metric_obj.f_min, metric_obj.f_max)
				}
	for MM in MM_list:
		out_dict[MM] = np.zeros((N_points,))
	
	for i in tqdm(range(N_points), desc='Drawing points'):
		theta1 = np.random.uniform(*boundaries)
		out_dict['theta1'][i,:] = theta1

		for j, MM in enumerate(MM_list):
			theta2 = None
			for k in range(100):
				theta2 = np.squeeze(metric_obj.get_points_atmatch(1, theta1, MM , overlap))
				if np.all(np.logical_and(theta2>boundaries[0,:], theta2<boundaries[1,:])): break
				else: theta2 = None
			if theta2 is None:
				out_dict['theta2'][i,:,j] = np.nan
				out_dict[MM][i] = np.nan
			else:
				out_dict['theta2'][i,:,j] = theta2
				out_dict[MM][i] = metric_obj.match(theta1, theta2, overlap=overlap)
				#print(theta1[0], MM, metric_obj.metric_match(theta1, theta2, overlap = overlap), out_dict[MM][i])
	
	return out_dict

def plot_metric_accuracy_data(out_dict, savefile = None):
	"Given a dict produced by get_metric_accuracy_data, it plots the histograms for each of the match points"
	
	plt.figure()
	for MM in out_dict['MM_list']:
		bins = np.logspace(np.log10(np.percentile(out_dict[MM], 15)), 0, 10)
		plt.hist(out_dict[MM], bins = bins, histtype='step')
		plt.axvline(MM, c = 'r')
	plt.xscale('log')

	if savefile is not None: plt.savefig(savefile)	
	plt.show()


#########################################

if __name__ == '__main__':

		#definition
	N_points = 20
	variable_format = 'Mq_nonspinning'
	psd = 'H1L1-REFERENCE_PSD-1164556817-1187740818.xml.gz'
	ifo = 'H1'
	approximant = 'IMRPhenomPv2'
	f_min, f_max = 10., 1024.
	if len(sys.argv)>1: run_name = sys.argv[1]
	else: run_name = 'test_overlap'
	load = True
	overlap = (run_name.find('overlap')>-1)
	
	MM_list = [0.999, 0.99, 0.97, 0.95]
	
	boundaries = np.array([[10, 1.],[30, 5.]]) #Mq_nonspinning
	boundaries = np.array([[10, 1., 0., 0.],[30, 5., 0.9, np.pi]]) #Mq_s1xz

	filename = '{}_{}.pkl'.format(run_name, variable_format)
	print("Working with file {}".format(filename))
	
		#metric and calling the function
	m_obj = cbc_metric(variable_format,
			PSD = load_PSD(psd, False, ifo),
			approx = approximant,
			f_min = f_min, f_max = f_max)
	
	if not load:
		out_dict = get_metric_accuracy_data(m_obj, MM_list, boundaries, N_points, overlap = True)
		with open(filename, 'wb') as filehandler:
			pickle.dump(out_dict, filehandler)
	else:
		with open(filename, 'rb') as filehandler:
			out_dict = pickle.load(filehandler)
	
	plot_metric_accuracy_data(out_dict)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
