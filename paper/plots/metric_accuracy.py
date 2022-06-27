import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm

from mbank.metric import cbc_metric
from mbank.utils import load_PSD, avg_dist, get_ellipse, project_metric
from scipy import stats
from itertools import combinations, permutations, product

import pickle
import os, sys

import pycbc.types
import pycbc.filter

#########################################

class psd_pycbc():

	def __init__(self, metric_obj):
		self.f, self.PSD = metric_obj.f_grid, metric_obj.PSD
		self.df = self.f[1]-self.f[0]
		self.PSD_pycbc = pycbc.types.timeseries.FrequencySeries(self.PSD, self.df)

	def get_match_pycbc(self, theta1, theta2, mmetric):
		WF1 = mmetric.get_WF(theta1, approximant)
		WF2 = mmetric.get_WF(theta2, approximant)

		WF1 = pycbc.types.timeseries.FrequencySeries(WF1, self.df) 
		WF2 = pycbc.types.timeseries.FrequencySeries(WF2, self.df) 
		
		
		match = WF1.match(WF2, self.PSD_pycbc)[0]
		
		return match


def get_metric_accuracy_data(metric_obj, MM_list, boundaries, N_points, overlap = False):
	"Given a metric object and some boundaries, it draws random points and for each point it draws a random point at constant metric distance. The metric distance will be then compared with the actual distance. The output is stored on a dict"
	
	pycbc_match_obj = psd_pycbc(metric_obj)
	
	#TODO: store the metric or the eigenvalues
		#initializing out_dict
	out_dict = {'center': np.zeros((N_points,metric_obj.D)),
				'theta': np.zeros((N_points,metric_obj.D, len(MM_list))),
				'metric': np.zeros((N_points,metric_obj.D, metric_obj.D)), #this can be pretty heavy?
				'metric_type': 'parabolic_fit_hessian',
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
		center = np.random.uniform(*boundaries)
		WF_center = metric_obj.get_WF(center)
		out_dict['center'][i,:] = center
		
			#Computing the metric
		try:
			metric = metric_obj.get_metric(center, overlap, out_dict['metric_type'])
			out_dict['metric'][i,...] = metric
		except KeyboardInterrupt:
				quit()
		except:
			out_dict['metric'][i,...] = np.nan
		
		for j, MM in enumerate(MM_list):
				#extracting random points
			if np.any(np.isnan(out_dict['metric'][i,...])):
				print("Failed @ center = ",center)
				ids_ = [False, False, False]
			else:
				theta = metric_obj.get_points_on_ellipse(500, center, MM , metric, overlap)
				ids_ = metric_obj.var_handler.is_theta_ok(theta, metric_obj.variable_format)

			if np.any(ids_)>0:
				theta = theta[ids_,:]
				
				id_ok = 0 #this amounts to take things at random
					
				theta = theta[id_ok,:]
				WF_theta = metric_obj.get_WF(theta)

					#storing to out_dict
				out_dict['theta'][i,:,j] = theta
				out_dict[MM][i] = metric_obj.WF_match(WF_center, WF_theta, overlap=overlap) #mbank
				
			else:
				out_dict['theta'][i,:,j] = np.nan
				out_dict[MM][i] = np.nan
				#print('nan!!! ', center)
			
			#print(center, MM, out_dict[MM][i])
	
	return out_dict

def plot_metric_accuracy_data(out_dict, savefile = None):
	"Given a dict produced by get_metric_accuracy_data, it plots the histograms for each of the match points"
	
	fig = plt.figure()
	ax = fig.gca()
	#plt.title('{} - overlap = {}'.format(out_dict['variable_format'], out_dict['overlap']))
	plt.title('{}'.format(out_dict['variable_format']))
	next(ax._get_lines.prop_cycler)
	for MM in out_dict['MM_list']:
		bins = np.logspace(np.log10(np.nanpercentile(out_dict[MM], .5)), 0, 50)
		plt.hist(out_dict[MM], bins = bins, histtype='step')
		plt.axvline(MM, c = 'k', ls = '-')
	plt.xscale('log')
	plt.xlabel('$1-MM$')

	ax.set_xticks(out_dict['MM_list'], labels = [str(MM) for MM in out_dict['MM_list']])
	#ax.set_xticks([0.9+0.01*i for i in range(10)], labels = [], minor = True)
	ax.set_xticks([0.94+0.01*i for i in range(6)], labels = [], minor = True)
#	ax.set_xticklabels([str(MM) for MM in out_dict['MM_list']])

	if savefile is not None: plt.savefig(savefile)	
	plt.show()

def plot_hist(out_dict):
	MM = 0.97
	id_ = np.where(np.array(out_dict['MM_list'])==MM)[0][0]

	for i in range(out_dict['center'].shape[1]):
		plt.figure()
		plt.title(i)
		plt.scatter(out_dict['center'][:,i], out_dict[MM][:])
		plt.axhline(MM, c= 'r')

	plt.figure()
	plt.title("norm")
	
	eigvals, eigvec = np.linalg.eig(out_dict['metric'])
	dist = out_dict['theta'][...,id_]-out_dict['center'] #(N,D)
	dist_prime = np.einsum('ijk,ik->ij', eigvec, dist) #(N,D)
	dist_prime = dist_prime/eigvals

	plt.scatter(np.linalg.norm(dist, axis = 1)/np.sqrt(np.linalg.det(out_dict['metric'])), out_dict[MM])
	#plt.scatter(np.linalg.norm(dist_prime, axis = 1), out_dict[MM])
	
	plt.xscale('log')
	plt.axhline(MM, c= 'r')

	plt.show()

def plot_ellipse(center, MM, metric_obj, boundaries = None):
	"Plots the constant match of the ellipse"
	center = np.asarray(center)
	metric = -metric_obj.get_metric(center)
	var_handler = metric_obj.var_handler
	
	fig, axes = plt.subplots(metric_obj.D-1, metric_obj.D-1, figsize = (15,15))
	fs = 15
	plt.suptitle('Center: {}'.format(center), fontsize = fs+10)
	if metric_obj.D-1 == 1:
		axes = np.array([[axes]])
	for i,j in permutations(range(metric_obj.D-1), 2):
		if i<j:	axes[i,j].remove()

		#Plot the templates
	for ax_ in combinations(range(metric_obj.D), 2):
		currentAxis = axes[ax_[1]-1, ax_[0]]
		ax_ = list(ax_)
		center_2d = (center[ax_[0]], center[ax_[1]])
		
		currentAxis.scatter(*center_2d, s = 10, marker = 'x', c= 'r', alpha = 1)
		if boundaries is not None:
			d = boundaries[1,:]- boundaries[0,:]
			currentAxis.add_patch(matplotlib.patches.Rectangle(boundaries[0,ax_], d[ax_[0]], d[ax_[1]], fill = None, alpha =1))
		
			#plotting the projected metric 
		dist = avg_dist(MM, metric_obj.D) #1 -MM 
		metric_projected = project_metric(metric, ax_)
		currentAxis.add_patch(get_ellipse(metric_projected, center_2d, dist))
		#currentAxis.add_patch(get_ellipse(metric_projected, center_2d, np.sqrt(1-MM), color= 'r', ls = '--'))
		
			#other option for the ellipse (same as get_points_atmatch)
		if False:
			dist = np.sqrt(1-MM)
			N_points = 10000
			
			L = np.linalg.cholesky(metric_projected).T
			L_inv = np.linalg.inv(L)
			
			theta_prime = np.matmul(L, center[ax_])
			
				#generating points on the unit sphere
			v = np.random.normal(0, 1, (N_points, 2))
			norm = 1.0 / np.linalg.norm(v, axis = 1) #(N_points,)
			
			points_prime = theta_prime + dist*(v.T*norm).T
			points = np.matmul(points_prime, L_inv.T)
			
			currentAxis.scatter(*points.T, s = 1)
		
		
		if ax_[0] == 0:
			currentAxis.set_ylabel(var_handler.labels(variable_format, latex = True)[ax_[1]], fontsize = fs)
		else:
			currentAxis.set_yticks([])
		if ax_[1] == metric_obj.D-1:
			currentAxis.set_xlabel(var_handler.labels(variable_format, latex = True)[ax_[0]], fontsize = fs)
		else:
			currentAxis.set_xticks([])
		currentAxis.tick_params(axis='x', labelsize=fs)
		currentAxis.tick_params(axis='y', labelsize=fs)
	plt.show()
	
	return

#########################################

if __name__ == '__main__':

		#definition
	N_points = 15000
	#psd = 'H1L1-REFERENCE_PSD-1164556817-1187740818.xml.gz'
	psd = 'aligo_O3actual_H1.txt'
	ifo = 'H1'
	f_min, f_max = 10., 1024.
	if len(sys.argv)>1: run_name = sys.argv[1]
	else: run_name = 'test'
	load = False
	overlap = (run_name.find('overlap')>-1)
	print('overlap: ', overlap)
	
	MM_list = [0.999, 0.99, 0.97, 0.95]

	boundaries = np.array([[20, 1.],[50, 5.]]); variable_format = 'Mq_nonspinning'; approximant = 'IMRPhenomD'
	#boundaries = np.array([[20, 1., -0.99],[50., 5., 0.99]]) ; variable_format = 'Mq_chi'; approximant = 'IMRPhenomD'
	#boundaries = np.array([[20, 1., 0.1, 0.03, 0.],[50, 5., 0.99, np.pi, np.pi]]); variable_format = 'Mq_s1xz_iota'; approximant = 'IMRPhenomPv2'
	#boundaries = np.array([[20, 1., -0.99, 0.],[50, 5., 0.99, np.pi]]); variable_format = 'Mq_chi_iota'; approximant = 'IMRPhenomXPHM'

	filename = 'metric_accuracy/{}_{}.pkl'.format(run_name, variable_format)
	print("Working with file {}".format(filename))
	print("boundaries: ", *boundaries)
	
		#metric and calling the function
	m_obj = cbc_metric(variable_format,
			PSD = load_PSD(psd, True, ifo),
			approx = approximant,
			f_min = f_min, f_max = f_max)
	
	#plot_ellipse([25, 3, 0., 0.], 0.95, m_obj, boundaries)
	#quit()
	
	if not load:
		out_dict = get_metric_accuracy_data(m_obj, MM_list, boundaries, N_points, overlap = overlap)
		with open(filename, 'wb') as filehandler:
			pickle.dump(out_dict, filehandler)
	else:
		with open(filename, 'rb') as filehandler:
			out_dict = pickle.load(filehandler)
	
	plot_hist(out_dict)
	quit()
	
	savefile = None #'../tex/img/metric_accuracy_{}.pdf'.format(variable_format)
	plot_metric_accuracy_data(out_dict, savefile)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
