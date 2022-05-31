import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import pickle

import sys
sys.path.insert(0, '../..')

from mbank.bank import cbc_bank
from mbank.metric import cbc_metric
from mbank.utils import load_PSD, get_boundaries_from_ranges, compute_injections_match, compute_metric_injections_match, ray_compute_injections_match
from mbank.handlers import tiling_handler, variable_handler

###########################################################################################

def get_N_templates_data(variable_format, placing_method, MM_list, epsilon_list, N_injs, N_neigh_templates, m_obj, boundaries, load_tiling, load_bank, full_match, folder_name):
	"Computes the number of templates for each MM and for each V_tile"

	epsilon_list.sort(reverse=True)
	out_dict = {'variable_format':variable_format,
			'placing_method':placing_method,
			'MM_list':MM_list, 'epsilon_list': epsilon_list,
			'boundaries':boundaries,
			'N_templates': np.zeros((len(epsilon_list), len(MM_list)), int),
			'N_tiles': np.zeros((len(epsilon_list), ), int),
			'volume_tiles': np.zeros((len(epsilon_list), )),
			'MM_metric': np.zeros((len(epsilon_list), N_injs), float),
			'MM_full': np.zeros((len(epsilon_list), N_injs), float),
			'N_injs': N_injs, 'N_neigh_templates': N_neigh_templates, 'MM_inj': 0.97
		}

	t = tiling_handler()
	for i, epsilon in enumerate(epsilon_list):
		filename = "{}/files/tiling_{}_{}.npy".format(folder_name, variable_format, epsilon)
			#getting the tiling
		#print("Tiling file: ",filename)
		if load_tiling and os.path.exists(filename):
			del t
			t = tiling_handler(filename)
		else:
			t.create_tiling(boundaries, epsilon, m_obj.get_metric, verbose = True)
			t.save(filename)
		out_dict['N_tiles'][i]= len(t)
		out_dict['volume_tiles'][i]= t.compute_volume()[0]
		
		for j, MM in enumerate(MM_list):
				#generating the bank
			bank_name = "{}/files/bank_{}_{}_{}.dat".format(folder_name, placing_method, epsilon, MM)
			b = cbc_bank(variable_format)
			if load_bank and os.path.exists(bank_name):
				b.load(bank_name)
			else: 
				b.place_templates(t, MM, placing_method = placing_method, verbose = True)
				b.save_bank(bank_name)
			out_dict['N_templates'][i,j] = b.templates.shape[0]
			print("MM, N_tiles, N_templates\t:", MM, out_dict['N_tiles'][i], out_dict['N_templates'][i,j])
		
				#Throwing injections
			if MM != out_dict['MM_inj']: continue #injections only for MM = 0.97
			injs = t.sample_from_tiling(N_injs, seed = 210795)
					#metric injections
			inj_dict = compute_metric_injections_match(injs, b, t, N_neigh_templates = N_neigh_templates, verbose = True)
			out_dict['MM_metric'][i,:] = inj_dict['match']
			print('\t\tMetric match: ', np.percentile(inj_dict['match'], [1, 5,50,95])) 
					#full match injections
			if full_match:
				inj_dict = ray_compute_injections_match(inj_dict, b.BBH_components(), m_obj, N_neigh_templates = N_neigh_templates,
							symphony_match = False, cache = True)
				out_dict['MM_full'][i,:] = inj_dict['match']
				print('\t\tFull match: ', np.percentile(inj_dict['match'], [1, 5,50,95]))
		
		
	return out_dict


def plot(out_dict, run_name, folder_name = None):
	"Plot the gathered data"

	print("Placing method: ", out_dict['placing_method'])
	print("Variable format: ", out_dict['variable_format'])
	print("Boundaries:\n", out_dict['boundaries'])
	
	N_templates=out_dict['N_templates']
	
		#N_templates
	plt.figure()
	plt.title("{}\n{} - {}".format(run_name, out_dict['variable_format'],out_dict['placing_method']))
	
	for i in range(N_templates.shape[1]):
		#plt.loglog(out_dict['epsilon_list'], N_templates[:,i], label = out_dict['MM_list'][i])
		plt.loglog(out_dict['N_tiles'], N_templates[:,i], label = out_dict['MM_list'][i])
	
	plt.legend()
	plt.xlabel(r"$\epsilon_{tile}$")
	plt.xlabel(r"$N_{tiles}$")
	plt.ylabel(r"$N_{templates}$")
	if isinstance(folder_name, str):
		plt.savefig('{}/template_number_{}_{}.png'.format(folder_name,
			out_dict['variable_format'], out_dict['placing_method']))
	
		#MM study
	plt.figure()
	plt.title("Injection study {}\n{} - {}".format(run_name, out_dict['variable_format'],out_dict['placing_method']))
	for i, N_t in enumerate(out_dict['N_tiles']):
		perc = np.percentile(out_dict['MM_metric'][i,:], 1) if np.all(out_dict['MM_full'][i,:]==0.) else np.minimum(np.percentile(out_dict['MM_full'][i,:], 1), np.percentile(out_dict['MM_metric'][i,:], 1))
		perc = np.array([perc, 1])
		MM_grid = np.linspace(*perc, 30)
		bw = np.diff(perc)/10
		if False and N_t>1000: #check KDE
			plt.figure()
			kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(out_dict['MM_metric'][i,:, None])
			plt.plot(MM_grid, np.exp(kde.score_samples(MM_grid[:,None])))
			plt.hist(out_dict['MM_metric'][i,:], density = True, bins =20)
			plt.show()
		
		plt.plot(np.repeat(N_t, 2), perc, '--', lw = 1, c='k')
			#creating a KDE for the plots
		scale_factor = 0.3
		kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(out_dict['MM_metric'][i,:, None])
		pdf_metric = np.exp(kde.score_samples(MM_grid[:,None]))
		plt.plot(N_t*(1-scale_factor*(pdf_metric-np.min(pdf_metric))/np.max(pdf_metric-pdf_metric[0])), MM_grid,
						c= 'b', label = 'Metric Match' if i==0 else None)

		if not np.all(out_dict['MM_full'][i,:]==0.):
			kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(out_dict['MM_full'][i,:, None])
			pdf_full = np.exp(kde.score_samples(MM_grid[:,None]))
			plt.plot(N_t*(1+scale_factor*(pdf_full-np.min(pdf_full))/np.max(pdf_full-pdf_full[0])), MM_grid,
						c= 'orange', label = 'Full Match' if i==0 else None)
		
	#plt.yscale('log')
	plt.xscale('log')
	plt.axhline(out_dict['MM_inj'], c = 'r')
	plt.xlabel(r"$N_{tiles}$")
	plt.ylabel(r"${Match}$")
	plt.ylim((0.94,1.001))
	plt.legend(loc = 'lower right')
	if isinstance(folder_name, str):
		plt.savefig('{}/injection_recovery_{}_{}.png'.format(folder_name,
			out_dict['variable_format'], out_dict['placing_method']))
	
	
		#Volume as a function of V_tile
	plt.figure()
	plt.title("Volume {} - {}".format(run_name, out_dict['variable_format']))
	plt.loglog(out_dict['N_tiles'], out_dict['volume_tiles']); plt.xlabel(r"$N_{tiles}$")
	#plt.loglog(out_dict['V_tile_list'], out_dict['volume_tiles'])
	#plt.xlabel(r"$V_{tile}$")
	plt.ylabel(r"$V_{tot}$")
	if isinstance(folder_name, str):
		plt.savefig('{}/volume_{}.png'.format(folder_name,
			out_dict['variable_format']))
	
	plt.show()


###########################################################################################

if __name__ == '__main__':
	
	load = False
	load_tiling = True
	load_bank = False
	full_match = False

	MM_list = [0.92, 0.95, 0.97, 0.99]
	
	#V_tile_list = [5, 10, 50, 100, 200, 500, 1000]; variable_format = 'Mq_s1xz_s2z' #for precessing
	#V_tile_list = [100, 200, 500, 1000]; variable_format = 'Mq_s1xz_s2z' #for precessing (with reduced tiling for random method)
	#V_tile_list = [10000, 5000, 1000, 500, 100, 70]; variable_format = 'Mq_s1z_s2z' #for aligned_spin
	#V_tile_list = [120, 100, 10, 5, 1]; variable_format =  'Mq_nonspinning' #for nonspinning
	#V_tile_list = [120, 100, 10]; variable_format =  'Mq_nonspinning' #for test
	
	epsilon_list = [1, 0.9, 0.8, 0.5, 0.3, 0.1]; variable_format =  'Mq_nonspinning'
	epsilon_list = [1, 0.9, 0.8, 0.5, 0.3, 0.2]; variable_format =  'Mq_chi'
	#epsilon_list = [1, 0.9, 0.8]; variable_format =  'Mq_s1xz_iota'
	
			#setting ranges
	M_range = (30, 50)
	q_range = (1,5)
	s_range = (-0.99, 0.99)
	e_range = (0., 0.5)
	boundaries = get_boundaries_from_ranges(variable_handler().format_info[variable_format], M_range, q_range, s_range, s_range, e_range = e_range)
	
	psd = 'H1L1-REFERENCE_PSD-1164556817-1187740818.xml.gz'
	ifo = 'H1'
	approximant = 'IMRPhenomD'
	f_min, f_max = 10., 1024.
	N_injs, N_neigh_templates = 1000, 30
	
	m_obj = cbc_metric(variable_format,
			PSD = load_PSD(psd, False, ifo),
			approx = approximant,
			f_min = f_min, f_max = f_max)
	

		#dealing with files
	if len(sys.argv)>1: run_name, placing_method = sys.argv[1], sys.argv[2]
	else: raise ValueError("Run name must be given!")
	
	folder_name = 'placing_methods_accuracy/{}'.format(run_name)	
	filename = '{}/data_{}_{}.pkl'.format(folder_name, variable_format, placing_method)
	if not os.path.isdir(folder_name): os.mkdir(folder_name)
	if not os.path.isdir(folder_name+'/files'): os.mkdir(folder_name+'/files')
	
	print("Working with folder: ", folder_name)
	
	if not load:

		out_dict = get_N_templates_data(variable_format, placing_method, MM_list, epsilon_list, N_injs, N_neigh_templates,
					m_obj, boundaries, load_tiling, load_bank, full_match, folder_name)
		
		with open(filename, 'wb') as filehandler:
			pickle.dump(out_dict, filehandler)
	else:
		with open(filename, 'rb') as filehandler:
			out_dict = pickle.load(filehandler)
	
	plot(out_dict, run_name, folder_name)
