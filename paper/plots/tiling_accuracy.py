"Computes the metric change inside each tile"

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from mbank.metric import cbc_metric
from mbank.utils import load_PSD
from mbank.handlers import tiling_handler, variable_handler

from mbank.flow.utils import compare_probability_distribution

import pickle, json
import os, sys

def get_deltaM_data(t_obj, m_obj, N_points, filename = 'tiling_accuracy/test.pkl' ):
	
			#Checking if the metric object loaded is the correct one!
	for i in range(5):
		rect, metric_tiling = t_obj[np.random.randint(len(t_obj))]
		center = (rect.maxes+rect.mins)/2.
		metric_true = m_obj.get_metric(center)
		metric_flow = t_obj.get_metric(center, flow = True, kdtree = False)
		#print("Dets: true | tiling | flow ", np.linalg.det(metric_true), np.linalg.det(metric_tiling), np.linalg.det(metric_flow))
		
		assert np.allclose(np.linalg.det(metric_true), np.linalg.det(metric_tiling))
		assert np.allclose(np.linalg.det(metric_true), np.linalg.det(metric_flow))
	
	out_dict = {
		'variable_format': m_obj.variable_format,
	}
	
	points, t_id = t_obj.sample_from_tiling(N_points, seed = None, tile_id = True)
	deltaM_tiling_hist, deltaM_flow_hist = [], []
	logMratio_tiling_hist, logMratio_flow_hist = [], []
	Mdet_hist = []
	metric_flow_list = []
	
	for j in tqdm(range(len(points)), desc = "Loop on points - {}".format(out_dict['variable_format'])):
		
		try:
			p, id_ = points[j], t_id[j]
			
				#Computing all the different metric
				#TODO: do that in batches...
			metric_tiling = t_obj[id_].metric
			metric_true = m_obj.get_metric(p)
			
				#Creating a buffer of metric values for the flow
			if len(metric_flow_list)==0:
				metric_flow_list = t_obj.get_metric(points[j:j+1000], flow = True, kdtree = False)
				metric_flow_list = [m for m in metric_flow_list]
				
			metric_flow = metric_flow_list.pop(0)

				#Computing interesting quantities
			det_true = np.linalg.det(metric_true)
			det_tiling = np.linalg.det(metric_tiling)
			det_flow = np.linalg.det(metric_flow)

			deltaM = lambda det_true, det_other: (det_true - det_other)/det_true
			logMratio = lambda det_true, det_other: 0.5*np.log10(det_other/det_true)

			deltaM_tiling = deltaM(det_true, det_tiling)
			deltaM_flow = deltaM(det_true, det_flow)

			logMratio_tiling = logMratio(det_true, det_tiling)
			logMratio_flow = logMratio(det_true, det_flow)

			#print("Dets: true | tiling | flow ", det_true, det_tiling, det_flow)
			#print("logM_ratio: tiling | flow", logMratio_tiling, logMratio_flow)

				#Appending to the hists
			deltaM_tiling_hist.append(deltaM_tiling)
			deltaM_flow_hist.append(deltaM_flow)

			logMratio_tiling_hist.append(logMratio_tiling)
			logMratio_flow_hist.append(logMratio_flow)
			
			Mdet_hist.append([det_true, det_tiling, det_flow])
			
		except KeyboardInterrupt:
			break
		
	out_dict['deltaM_tiling'] = deltaM_tiling_hist
	out_dict['deltaM_flow'] = deltaM_flow_hist
	out_dict['logMratio_tiling'] = logMratio_tiling_hist
	out_dict['logMratio_flow'] = logMratio_flow_hist
	out_dict['Mdet'] = Mdet_hist
	
		#saving to file		
	with open(filename, 'w') as filehandler:
		json.dump(out_dict, filehandler)

def plot_deltaM_data(filename):

	with open(filename, 'r') as filehandler:
		out_dict = json.load(filehandler)
	N = len(out_dict['deltaM_tiling'])
	nbins = int(np.sqrt(N))+1

	out_dict['deltaM_tiling'] = np.abs(out_dict['deltaM_tiling'])
	out_dict['deltaM_flow'] = np.abs(out_dict['deltaM_flow'])


	hist_args = {
		'bins': np.logspace(np.log10(np.min(out_dict['deltaM_tiling'])), np.log10(np.max(out_dict['deltaM_tiling'])), nbins),
		'density': True,
		'histtype': 'step'
	}

	plt.figure()
	plt.title(out_dict['variable_format'])
	#plt.hist((out_dict['deltaM_tiling']), label = 'tiling', **hist_args)
	#plt.hist((out_dict['deltaM_flow']), label = 'flow', **hist_args)
	plt.hist(out_dict['deltaM_tiling'], label = 'tiling', **hist_args)
	plt.hist(out_dict['deltaM_flow'], label = 'flow', **hist_args)
	
	plt.legend()
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel(r'$\frac{|M|-|M_{true}|}{|M_{true}|}$', size = 15)
	plt.tight_layout()
	plt.show()
	
	
	plt.show()
		
	
		

if __name__ == '__main__':
	
	folder = 'tiling_accuracy/'
	
	assert len(sys.argv) >1, "Variable format must be given"
	
	load = (len(sys.argv) == 3 and sys.argv[-1].lower() == 'load')
	
	variable_format = sys.argv[1]
	
	N_points = 50_000
	
	psd = 'aligo_O3actual_H1.txt'; asd = True
	ifo = 'H1'
	f_min, f_max = 10., 1024.

	if variable_format == 'Mq_chi':
		approximant = 'IMRPhenomD'; tiling_file = 'placing_methods_accuracy/paper_Mq_chi/files/tiling_Mq_chi_8.npy'
	elif variable_format == 'Mq_s1xz':
		approximant = 'IMRPhenomPv2'; tiling_file = 'placing_methods_accuracy/paper_Mq_s1xz/files/tiling_Mq_s1xz_10.npy'
	elif variable_format == 'Mq_s1xz_s2z_iota':
		approximant = 'IMRPhenomPv2'; tiling_file = 'placing_methods_accuracy/paper_Mq_s1xz_s2z_iota/files/tiling_Mq_s1xz_s2z_iota_10.npy'
	else:
		raise ValueError("Variable format {} not configured".format(variable_format))
	
	m_obj = cbc_metric(variable_format,
			PSD = load_PSD(psd, asd, ifo),
			approx = approximant,
			f_min = f_min, f_max = f_max)
	
	flow_file = folder+'flow_{}.zip'.format(variable_format)
	t_obj = tiling_handler(tiling_file)
	
	if os.path.isfile(flow_file):
		t_obj.load_flow(flow_file)
	else:
		history = t_obj.train_flow(N_epochs=2500, N_train_data= 20_000,
				#n_layers=2, hidden_features=4,	#Mq_chi
				n_layers=5, hidden_features=4,	#Mq_s1xz
				#n_layers=6, hidden_features=4,	#Mq_s1xz_s2z_iota
				batch_size=None, lr=0.001, verbose=True)
		t_obj.flow.save_weigths(flow_file)
	
		with open(folder+'history_{}.pkl'.format(variable_format), 'wb') as f:
			pickle.dump(history, f)
	
		N_plot_points = 10000
		compare_probability_distribution(t_obj.sample_from_flow(N_plot_points), data_true=t_obj.sample_from_tiling(N_plot_points), 
				variable_format=variable_format,
				title=None, hue_labels=['flow', 'train'],
				savefile=folder+'flow_{}.png'.format(variable_format), show=False)
	
	filename = folder+'out_dict_{}.json'.format(variable_format)
	print("Working with {}".format(filename))

	if load:
		plot_deltaM_data(filename)
	else:
		get_deltaM_data(t_obj, m_obj, N_points = N_points, filename = filename)
	
	
	
