"Computes the metric change inside each tile"

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','ieee', 'bright'])

from tqdm import tqdm

from mbank.metric import cbc_metric
from mbank.utils import load_PSD, get_boundaries_from_ranges
from mbank.handlers import tiling_handler, variable_handler

import pickle
import os, sys


def get_deltaM(metric1, metric2):
	det1, det2 = np.linalg.det(metric1), np.linalg.det(metric2)
	return np.abs(det1-det2)/np.maximum(det1,det2)

def get_deltaM_data(file_list, m_obj, N_points, filename = 'tiling_accuracy/test.pkl' ):
	
	for i, f in enumerate(file_list):
		t = tiling_handler(f['file'])
		
		m_obj.set_variable_format( f['var_format'])
		
		points, t_id = t.sample_from_tiling(N_points, seed = None, tile_id = True)
		deltaM_hist = []
		
		for j in tqdm(range(len(points)), desc = "Loop on points - {}".format(f['var_format'])):
			p, id_ = points[j], t_id[j]
			if j==0: p = t[id_].center
			metric = m_obj.get_metric(p)
			
			deltaM = get_deltaM(metric, t[id_].metric)
			if j ==0 and not np.allclose(deltaM, 0.): warnings.warn("The metric in the center of the tile does not match: are you sure you set things right?")
			deltaM_hist.append(deltaM)
	
		file_list[i]['deltaM_hist'] = np.array(deltaM_hist)
		
	with open(filename, 'wb') as filehandler:
		pickle.dump(file_list, filehandler)

def plot_deltaM_data(filename):

	with open(filename, 'rb') as filehandler:
		file_list = pickle.load(filehandler)

	fig, axes = plt.subplots(len(file_list), 1, sharex = True)
	if not isinstance(axes, np.ndarray): axes = [axes]
	
	for i, (f, ax) in enumerate(zip(file_list, axes)):
		ax.hist(f['deltaM_hist'])
		ax.axvline(f['epsilon'], c= 'r')
	
	plt.show()
		
	
		

if __name__ == '__main__':
	
	if len(sys.argv)>1: run_name = sys.argv[1]
	else: raise ValueError("Run name must be given!")
	filename = 'tiling_accuracy/{}.pkl'.format(run_name)
	
	load = not True
	if len(sys.argv)>2:
		if sys.argv[2] == 'plot': load = True
	N_points = 3000
	
	psd = 'aligo_O3actual_H1.txt'; asd = True
	#psd = 'H1L1-REFERENCE_PSD-1164556817-1187740818.xml.gz'; asd = False
	ifo = 'H1'
	approximant = 'IMRPhenomD'
	f_min, f_max = 15., 1024.
	
	m_obj = cbc_metric('Mq_nonspinning',
			PSD = load_PSD(psd, asd, ifo),
			approx = approximant,
			f_min = f_min, f_max = f_max)
	
	file_nonspinning = {'epsilon':0.5, 'var_format': 'Mq_nonspinning',
			'file': 'placing_methods_accuracy/paper_nonspinning/files/tiling_Mq_nonspinning_0.5.npy'}
	file_chi = {'epsilon':0.5, 'var_format': 'Mq_chi',
			'file': 'placing_methods_accuracy/paper_chi/files/tiling_Mq_chi_0.5.npy'}
	file_chi = {'epsilon':0.1, 'var_format': 'Mq_chi',
			'file': '/home/stefano/Dropbox/Stefano/PhD/mbank/runs/out_test/tiling_test.npy'}

	file_list = [file_chi]

	print("Working with {}".format(filename))

	if load:
		plot_deltaM_data(filename)
	else:
		get_deltaM_data(file_list, m_obj, N_points = N_points, filename = filename)
	
	
	
