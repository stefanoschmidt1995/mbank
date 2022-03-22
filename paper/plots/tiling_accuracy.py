import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','ieee', 'bright'])

from tqdm import tqdm

from mbank.metric import cbc_metric
from mbank.utils import load_PSD, get_boundaries_from_ranges
from mbank.handlers import tiling_handler, variable_handler

import pickle
import os, sys

#########################################

def get_volume_list(metric_obj, V_tile_list, variable_format, boundaries):
	"Given a metric object and a variable_format it computes the tiling for different values of `V_tile` and computes the overall volume with metric approximation."
	vol_list = []
	t_vols_list = []
	
	for V_tile in tqdm(V_tile_list, desc = 'Variable format: {}'.format(variable_format)):
		t = tiling_handler()
			#generating the tiling
		t.create_tiling(boundaries, V_tile, m_obj.get_metric, verbose = True)
		vol, t_vols = t.compute_volume()
		vol_list.append(vol)
		t_vols_list.append(t_vols)
	
	return vol_list, t_vols_list

def get_tiling_accuracy_data(metric_obj, variable_format_list, V_tile_list, M_range, q_range, s_range, e_range):
	"It computes the volume of the space as a function of V_tilelates and variable_format. It stores the results in an handy dictionary"
	out_dict = {'V_tile': V_tile_list, 'var_formats': variable_format_list, 'M_range':M_range,'q_range':q_range, 's_range':s_range, 'e_range':e_range}
	
	
	for var_format in variable_format_list:
		boundaries = get_boundaries_from_ranges(variable_handler().format_info[var_format], M_range, q_range, s_range, s_range, e_range = e_range)		
		metric_obj.set_variable_format(var_format)
		vol_list, t_vols_list = get_volume_list(metric_obj, V_tile_list, var_format, boundaries)
		out_dict[var_format] = vol_list

	return out_dict

def plot_tiling_accuracy_data(out_dict, savefile = None):
	"Plot stuff"
	fig_tot = plt.figure()
	ax_tot = plt.gca()
	
	fig_sub, ax = plt.subplots(len(out_dict['var_formats']), 1, sharex = True)
	
	
	for ax_, v_f in zip(ax, out_dict['var_formats']):
		ax_.set_title(v_f)
		x, y = out_dict['V_tile'], out_dict[v_f]
			#scaling volumes
		y /= y[np.argmin(out_dict['V_tile'])]
		
		p = ax_.scatter(x, y, marker = 'x', label = v_f)
		ax_tot.scatter(x, y, marker = 'x', label = v_f)
	
		ax_.set_ylabel(r"$Vol$")
		ax_.set_xlabel(r"$N_{templates-in-tile}$")
		ax_.set_xscale('log')
		#ax_.set_yscale('log')
	
	ax_tot.set_ylabel(r"$Vol$")
	ax_tot.set_xlabel(r"$N_{templates-in-tile}$")
	ax_tot.set_xscale('log')
	#ax_tot.set_yscale('log')
	ax_tot.legend()
	
	plt.tight_layout()
	#if isinstance(savefile, str): fig_sub.savefig(savefile)
	if isinstance(savefile, str): fig_tot.savefig(savefile)
	
	plt.show()
	
	return

#########################################

if __name__ == '__main__':
	
	if len(sys.argv)>1: run_name = sys.argv[1]
	else: run_name = 'test'
	
	load = True

	folder_name = 'tiling_accuracy'	
	filename = '{}/tiling_accuracy_{}.pkl'.format(folder_name, run_name)
	if not os.path.isdir(folder_name): os.mkdir(folder_name)

	print("Working with file: {}".format(filename))

	psd = 'H1L1-REFERENCE_PSD-1164556817-1187740818.xml.gz'
	ifo = 'H1'
	approximant = 'IMRPhenomPv2'
	f_min, f_max = 10., 1024.

	variable_format_list = ['Mq_s1xz_s2z_iota', 'Mq_s1xz_s2z', 'Mq_s1xz', 'Mq_nonspinning', 'Mq_s1z_s2z']
	variable_format_list = ['Mq_s1xz',  'Mq_s1z_s2z', 'Mq_nonspinning']
	variable_format_list = ['Mq_s1z_s2z', 'Mq_nonspinning']
	V_tile_list = [.5, 1, 5, 50, 100, 1000, 2000, 10000, 15000]
	#V_tile_list = [1000, 2000, 5000, 10000, 15000]
	
	m_obj = cbc_metric(variable_format_list[0],
			PSD = load_PSD(psd, False, ifo),
			approx = approximant,
			f_min = f_min, f_max = f_max)
	
		#setting ranges
	M_range = (50, 100)
	q_range = (1,5)
	s_range = (-0.99, 0.99)
	e_range = (0., 0.5)
	
	if not load:

		out_dict = get_tiling_accuracy_data(m_obj, variable_format_list, V_tile_list, M_range, q_range, s_range, e_range)
		
		with open(filename, 'wb') as filehandler:
			pickle.dump(out_dict, filehandler)
	else:
		with open(filename, 'rb') as filehandler:
			out_dict = pickle.load(filehandler)

	savefile = folder_name+'/tiling_accuracy_{}.png'.format(run_name)
	#savefile = '../tex/img/tiling_accuracy_{}.pdf'.format(run_name)
	print("Saving to file {}".format(savefile))
	plot_tiling_accuracy_data(out_dict, savefile = savefile)









