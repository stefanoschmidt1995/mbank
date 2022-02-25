import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from mbank.bank import cbc_bank
from mbank.metric import cbc_metric
from mbank.utils import load_PSD, get_boundaries_from_ranges
from mbank.handlers import tiling_handler, variable_handler

import os, sys

import pickle

from scipy.stats import linregress
from scipy.optimize import curve_fit

###########################################

def get_scaling_relation(variable_format, boundaries, MM_list, metric_object, templates_in_tile = 10, placing_method = 'geometric', save_folder = None, load_tiling = False, verbose = True):
	"Compute the relation N_templates-MM for the given variable_format and boundaries"
	b = cbc_bank(variable_format)
	out_list = []
	
		#generating the tiling
	if isinstance(save_folder, str):
		if not save_folder.endswith('/'): save_folder = save_folder+'/'
	filename = '{}tiling_{}.npy'.format(save_folder, variable_format) #it's garbage if save_folder is None

	if not load_tiling:
		t_obj = b.generate_tiling(m_obj, boundaries, templates_in_tile, use_ray = False, verbose = verbose)

		if isinstance(save_folder, str):
			t_obj.save(filename)
	else:
		if save_folder is None: raise ValueError("If you want to load, you need to give a folder!")
		t_obj = tiling_handler(filename)
	
		#generating the bank
	for MM in MM_list:
		b.place_templates(t_obj, MM, placing_method = placing_method, verbose = verbose)
		out_list.append((MM, b.templates.shape[0]))

		#plt.figure()
		#plt.title('{}- MM = {}'.format(variable_format, MM))
		#plt.scatter(*b.templates[:,[0,1]].T)

	return out_list

def get_scaling_data(m_obj, variable_format_list, MM_list, m_range, q_range, s_range, e_range, templates_in_tile = 10, placing_method = 'geometric', save_folder = None, load_tiling = False, verbose = True):
	"Calls multiple times get_scaling_relation for different variable formats. Saves the results in a nice dict"
	if isinstance(save_folder, str):
		if not save_folder.endswith('/'): save_folder = save_folder+'/'
	
	out_dict = {'MM_list': np.array(MM_list), 'var_formats': variable_format_list,
			'm_range':m_range,'q_range':q_range, 's_range':s_range, 'e_range':e_range,
			'templates_in_tile': templates_in_tile, 'placing_method':placing_method			
			}
	
	for variable_format in variable_format_list:
		info_dict = variable_handler().format_info[variable_format]
		boundaries = get_boundaries_from_ranges(info_dict, m_range, q_range, s_range, s_range, e_range = e_range)
		m_obj.set_variable_format(variable_format)

		out_list = get_scaling_relation(variable_format, boundaries, MM_list, m_obj,
				templates_in_tile = templates_in_tile, placing_method = placing_method,
				save_folder = folder_name, load_tiling = load_tiling, verbose = verbose)

		out_dict[variable_format] = np.array([N for _, N in out_list], dtype = int)
		print(variable_format, out_dict[variable_format])

	return out_dict

def get_predictions(x, y, D, vol = None):
	"""
	Given x= 1-MM and y = N_templates it computes the:
	
	- Fitted scaling relation
	- Fitted q for the Owen's law
	- Owen's law (if vol is not None)
	
	They are evaluated a the given x
	"""
	res = linregress(np.log(x), np.log(y))
	m, q = res.slope, res.intercept
	m_owen = -D/2

	y_ = lambda x_: x_*m+q
	y_owen_log_ = lambda x_log_, q_: q_ -0.5*D*x_log_
		
	q_owen, _ = curve_fit(y_owen_log_, np.log(x), np.log(y), q)
	
			#predictions
	y_pred = np.exp( y_(np.log(x)))
	y_pred_owen_fit = np.exp( y_owen_log_(np.log(x), q_owen))	

	if vol is not None:
		q_owen_teo = np.log(vol)-D*np.log(2/np.sqrt(D))
		y_pred_owen = np.exp( y_owen_log_(np.log(x), q_owen_teo))
		return y_pred, y_pred_owen_fit, y_pred_owen
	else:
		return y_pred, y_pred_owen_fit, None
	

def plot_scaling_data(out_dict, sbank_dict = None, savefile = None, title = None):
	"""
	Plot the content of the out_dict created by `get_scaling_data`
	`sbank_dict` is a dict with
		{`spin_format`: filename}
	where filename is a 2D array with MM N_templates entries
	"""
	v_h = variable_handler()
	
	plt.figure(figsize = (15,15))
	if isinstance(title, str): plt.title(title)
	
	for v_f in out_dict['var_formats']:
		x, y = 1-out_dict['MM_list'], out_dict[v_f]
		p = plt.scatter(x, y, label = v_f)
		color = p.get_facecolors()[0]

			#load tiling, and computing the volume (if it is the case)
		try:
			tiling_file = '{}/{}'.format(savefile.split('/')[0], 'tiling_{}.npy'.format(v_f))
			vol, _ = tiling_handler(tiling_file).compute_volume()
		except FileNotFoundError:
			vol = None
		
		D = v_h.format_info[v_f]['D']
		y_pred, y_pred_owen_fit, y_pred_owen = get_predictions(x, y, D, vol)

			#plotting regressions
		plt.plot(x, y_pred, '--', c= color) #linear fit
		plt.plot(x, y_pred_owen_fit, ls = 'dotted', c= color) #owen slope and fitted q
		#if vol is not None: plt.plot(x, y_pred_owen, ls = '-.', c= color) #owen law
		
		if isinstance(sbank_dict, dict):
			if v_f in sbank_dict.keys():
				x_sbank, y_sbank = np.loadtxt(sbank_dict[v_f]).T
				x_sbank = 1-x_sbank #x = 1-MM
				p = plt.scatter(x_sbank, y_sbank, label = 'sbank-{}'.format(v_f), marker = 'x')
				color = p.get_facecolors()[0]
				
				y_pred_sbank, y_pred_owen_fit_sbank, _ = get_predictions(x_sbank, y_sbank, D, None)
				plt.plot(x_sbank, y_pred_sbank, '--', c= color) #linear fit
				plt.plot(x_sbank, y_pred_owen_fit_sbank, ls = 'dotted', c= color) #owen slope and fitted q
		
	plt.legend()
	plt.xlabel(r"$1-MM$")
	plt.ylabel(r"$N_{templates}$")
	plt.xscale('log')
	plt.yscale('log')
	
	if isinstance(savefile, str): plt.savefig(savefile, transparent = False)
	
	plt.show()

################################################################################################

if __name__ == '__main__':

		#definition
	psd = 'H1L1-REFERENCE_PSD-1164556817-1187740818.xml.gz'
	ifo = 'L1'
	approximant = 'IMRPhenomPv2'
	f_min, f_max = 10., 1024.
	if len(sys.argv)>1: run_name = sys.argv[1]
	else: run_name = 'geometric_nonprecessing'
	
	placing_method = 'random'
	
	load_dict = False
	load_tiling = True
	
	folder_name = 'scaling_{}/'.format(run_name)
	if not os.path.isdir(folder_name): os.mkdir(folder_name)
	
	MM_list = [0.6, 0.7, 0.8, 0.9]#, 0.95, 0.97]#, 0.99]
	variable_format_list = ['Mq_s1xz_s2z_iota', 'Mq_s1xz_s2z','Mq_s1xz', 'Mq_nonspinning', 'Mq_s1z_s2z']
	variable_format_list = ['Mq_nonspinning', 'Mq_s1z_s2z']

	M_range = (10,30)
	q_range = (1,5)
	s_range = (-0.99, 0.99)
	e_range = (0., 0.5)
	
		#metric and calling the function
	m_obj = cbc_metric(variable_format_list[0],
			PSD = load_PSD(psd, False, ifo),
			approx = approximant,
			f_min = f_min, f_max = f_max)

	if not load_dict:

		out_dict = get_scaling_data(m_obj, variable_format_list, MM_list, M_range, q_range, s_range, e_range,
					templates_in_tile = 100, placing_method = placing_method,
					save_folder = folder_name, load_tiling = load_tiling, verbose = True)
		
		with open(folder_name+'out_dict.pkl', 'wb') as filehandler:
			pickle.dump(out_dict, filehandler)
	else:
		with open(folder_name+'out_dict.pkl', 'rb') as filehandler:
			out_dict = pickle.load(filehandler)

	sbank_dict = None
	sbank_dict = dict(Mq_nonspinning = folder_name+'sbank_Mq_nonspinning.dat', Mq_s1z_s2z = folder_name+'sbank_Mq_s1z_s2z.dat')
	plot_scaling_data(out_dict, sbank_dict = sbank_dict, savefile = folder_name+'scaling.png', title = run_name)


















