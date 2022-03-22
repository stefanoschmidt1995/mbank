import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
import pickle

import sys
sys.path.insert(0, '../..')

from mbank.bank import cbc_bank
from mbank.metric import cbc_metric
from mbank.utils import load_PSD, get_boundaries_from_ranges
from mbank.handlers import tiling_handler, variable_handler

###########################################################################################

def get_N_templates_data(variable_format, placing_method, MM_list, V_tile_list, m_obj, boundaries, load_tiling, folder_name):
	"Computes the number of templates for each MM and for each V_tile"

	V_tile_list.sort(reverse=True)
	out_dict = {'variable_format':variable_format,
			'placing_method':placing_method,
			'MM_list':MM_list, 'V_tile_list': V_tile_list,
			'boundaries':boundaries,
			'N_templates': np.zeros((len(V_tile_list), len(MM_list)), int),
			'N_tiles': np.zeros((len(V_tile_list), ), int),
			'volume_tiles': np.zeros((len(V_tile_list), ))
		}

	t = tiling_handler()
	for i, V_tile in enumerate(V_tile_list):
		filename = "{}/tiling_{}_{}.npy".format(folder_name, variable_format, V_tile)
			#getting the tiling
		if load_tiling:
			del t
			t = tiling_handler(filename)
		else: 
			t.create_tiling(boundaries, V_tile, m_obj.get_metric, verbose = True)
			t.save(filename)
		out_dict['N_tiles'][i]= len(t)
		out_dict['volume_tiles'][i]= t.compute_volume()[0]
		
		for j, MM in enumerate(MM_list):
			b = cbc_bank(variable_format)
			b.place_templates(t, MM, placing_method = placing_method, verbose = True)
			out_dict['N_templates'][i,j] = b.templates.shape[0]
			print("MM, N_tiles, N_templates\t:", MM, out_dict['N_tiles'][i], out_dict['N_templates'][i,j])
		
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
		#plt.loglog(out_dict['V_tile_list'], N_templates[:,i], label = out_dict['MM_list'][i])
		plt.loglog(out_dict['N_tiles'], N_templates[:,i], label = out_dict['MM_list'][i])
	
	plt.legend()
	plt.xlabel(r"$V_{tile}$")
	plt.xlabel(r"$N_{tiles}$")
	plt.ylabel(r"$N_{templates}$")
	if isinstance(folder_name, str):
		plt.savefig('{}/template_number_{}_{}.png'.format(folder_name,
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
	load_tiling = False
	
	V_tile_list = [1, 2, 5, 10, 100, 1000]
	MM_list = [0.92, 0.95, 0.97, 0.99]
	
	variable_format = 'Mq_s1xz_s2z'
	
			#setting ranges
	M_range = (50, 100)
	q_range = (1,5)
	s_range = (-0.99, 0.99)
	e_range = (0., 0.5)
	boundaries = get_boundaries_from_ranges(variable_handler().format_info[variable_format], M_range, q_range, s_range, s_range, e_range = e_range)
	
	psd = 'H1L1-REFERENCE_PSD-1164556817-1187740818.xml.gz'
	ifo = 'H1'
	approximant = 'IMRPhenomPv2'
	f_min, f_max = 10., 1024.
	
	m_obj = cbc_metric(variable_format,
			PSD = load_PSD(psd, False, ifo),
			approx = approximant,
			f_min = f_min, f_max = f_max)
	

		#dealing with files
	if len(sys.argv)>1: run_name, placing_method = sys.argv[1], sys.argv[2]
	else: raise ValueError("Run name must be given!")
	
	folder_name = 'N_templates/{}'.format(run_name)	
	filename = '{}/N_templates_{}_{}.pkl'.format(folder_name, variable_format, placing_method)
	if not os.path.isdir(folder_name): os.mkdir(folder_name)
	
	print("Working with folder: ", folder_name)
	
	if not load:

		out_dict = get_N_templates_data(variable_format, placing_method, MM_list, V_tile_list, m_obj, boundaries, load_tiling, folder_name)
		
		with open(filename, 'wb') as filehandler:
			pickle.dump(out_dict, filehandler)
	else:
		with open(filename, 'rb') as filehandler:
			out_dict = pickle.load(filehandler)
	
	plot(out_dict, run_name, folder_name)
