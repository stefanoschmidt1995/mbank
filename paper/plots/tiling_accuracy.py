"Computes the metric change inside each tile"

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from mbank.metric import cbc_metric
from mbank.utils import load_PSD
from mbank.handlers import tiling_handler, variable_handler

from mbank.flow.utils import compare_probability_distribution

import pickle, json
import os, sys, glob

def load_result_dict(input_folder, N_points = 100):
	vf_list = []
	res_dict = {}
	
		#Initializing result dict
	for f in glob.glob(input_folder+'paper_*/*pkl'):
		with open(f, 'rb') as f_obj:
			data_dict = pickle.load(f_obj)
		vf = data_dict['variable_format']
		if vf in vf_list: continue
		else: vf_list.append(vf)
		
		save_dict = {
		'f_max': 1024.,
		'f_min': 10. if vf != 'Mq_s1xz_s2z_iota' else 15.,
		'approximant': 'IMRPhenomD' if vf == 'Mq_chi' else 'IMRPhenomXP'
		}
		
		for md in data_dict['max_depth_list']:
			tiling_file = "{}/files/tiling_{}_{}.npy".format(os.path.dirname(f), vf, md)
			save_dict['max_depth_list'] = data_dict['max_depth_list']
			save_dict[md] = tiling_file
			
		#Filling save_dict with interesting values 
		psd = 'aligo_O3actual_H1.txt' 
		ifo = 'H1'
		
		m_obj = cbc_metric(vf,
			PSD = load_PSD(psd, True, ifo),
			approx = save_dict['approximant'],
			f_min = save_dict['f_min'], f_max = save_dict['f_max'])

		#loading all the tiling objs
		tiling_objs = {}
		for md in save_dict['max_depth_list']:
			tiling_objs[md] = tiling_handler(save_dict[md])
			print(vf, md, len(tiling_objs[md]))
		#extracting points to test the tiling accuracy at
		save_dict['test_points'] = tiling_objs[md].sample_from_tiling(N_points)
		
		metric_from_tiling = tiling_objs[md][0].metric
		metric_from_obj = m_obj.get_metric(tiling_objs[md][0].center, metric_type = 'symphony')

		if not np.allclose(metric_from_tiling, metric_from_obj, atol = 0., rtol = 1e-6):
			print("Something wonky going on with the metric obj: are you sure you set all the params right?")
			print('\t',save_dict['approximant'], save_dict['f_min'], save_dict['f_max'])
		
		det_true = np.linalg.det(m_obj.get_metric(save_dict['test_points'], metric_type = 'symphony'))
		for md in save_dict['max_depth_list']:
			det_tiling = np.linalg.det(tiling_objs[md].get_metric(save_dict['test_points']))
			save_dict['hist_{}'.format(md)] = 0.5 * np.log10(det_tiling/det_true)
			save_dict['det_tiling_{}'.format(md)] = det_tiling
			save_dict['det_true_{}'.format(md)] = det_true

		del tiling_objs

		res_dict[vf] = save_dict
	return res_dict

def plot_tiling_accuracy_study(res_dict):
	
	plt.figure()
	for k, v in res_dict.items():
		y_axis = []
		for md in v['max_depth_list']:
			y_axis.append(np.percentile(np.abs(v['hist_{}'.format(md)]), 50))
		plt.scatter(v['max_depth_list'], y_axis, label = k)
	plt.axhline(0.1, ls = '--', c = 'k')
	plt.legend()
	
	vf = 'Mq_s1xz_s2z_iota'
	plt.figure()
	plt.hist(res_dict[vf]['hist_{}'.format(8)], bins = 100, histtype = 'step')
	plt.yscale('log')
	plt.xlabel(r"$0.5 \log_{10}\left(\frac{M}{M_{true}}\right)$")
	plt.axvline(-0.1, ls = '--', c = 'k')
	
	
	vh = variable_handler()
	ids_, = np.where(res_dict[vf]['hist_{}'.format(8)]>4.)
	for i, l in enumerate(vh.labels(vf, latex = True)):

		bins = int(np.sqrt(len(ids_)))
		hist_kwargs = {
			'density': True,
			'bins': 100
		}
		plt.figure()
		plt.hist(res_dict[vf]['test_points'][:,i], label = 'all points',  histtype ='stepfilled', alpha = 0.3, 	**hist_kwargs)
		plt.hist(res_dict[vf]['test_points'][ids_,i], label = 'bad points', histtype = 'step', **hist_kwargs)
		plt.xlabel(l)
		plt.legend()
	
	plt.show()


if __name__ == '__main__':
	
	tiling_folder = 'tiling_accuracy/'
	
	input_folder = 'placing_methods_accuracy/'
	
	save_file = tiling_folder+'tiling_accuracy_study.pkl'
	
	if len(sys.argv)>1:
		if sys.argv[1].lower() != 'run': raise ValueError("If an arg is given, it must be 'run'")
		if len(sys.argv) > 2: N_points = int(sys.argv[2])
		else: N_points = 10
		res_dict = load_result_dict(input_folder, N_points = N_points)
		with open(save_file, 'wb') as f:
			pickle.dump(res_dict, f)

	with open(save_file, 'rb') as f:
		res_dict = pickle.load(f)
	
	#Plotting shit
	plot_tiling_accuracy_study(res_dict)
	
	










	
	
	
