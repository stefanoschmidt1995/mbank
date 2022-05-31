"""
Script to make all the plots for the paper. It loads several files around and plot the results
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','ieee', 'bright']) #https://github.com/garrettj403/SciencePlots
import matplotlib
from matplotlib.lines import Line2D


from tqdm import tqdm

from sklearn.neighbors import KernelDensity

import warnings
import pickle
import os, sys

########################################################################################################

def plot_metric_accuracy(filenames, savefile = None):
	"Plot the metric accuracy plots"
		#loading files
	dict_list = []
	for filename in filenames:
		try:
			with open(filename, 'rb') as filehandler:
				dict_list.append( pickle.load(filehandler) )
		except:
			warnings.warn("No file {} found. Skipping".format(filename) )
	
	
		#creating the figures
	nbins = 100
	fig, axes = plt.subplots(len(dict_list), 1, sharex = True)
	
	for i, (ax, out_dict) in enumerate(zip(axes, dict_list)):
	
		#ax.title('{}'.format(out_dict['variable_format']))
		next(ax._get_lines.prop_cycler)
		for MM in out_dict['MM_list']:
			bins = np.logspace(np.log10(np.nanpercentile(out_dict[MM], .1)), 0, nbins)
			ax.hist(out_dict[MM], bins = bins, histtype='step', label = MM if i ==1 else None)
			ax.axvline(MM, c = 'k', ls = 'dotted')
		ax.set_xscale('log')
		ax.annotate(out_dict['variable_format'], xy = (.03,0.7), xycoords = 'axes fraction')
		if i==1:
			handles, labels = ax.get_legend_handles_labels()
			new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
			ax.legend(handles=new_handles, labels=labels, handlelength = 1, labelspacing = .1, loc = 'center right')
	axes[-1].set_xlabel('$1-MM$')

	axes[-1].set_xticks(out_dict['MM_list'], labels = [str(MM) for MM in out_dict['MM_list']])
	axes[-1].tick_params(axis = 'x', labelleft = True)
	axes[-1].set_xticks([0.93+0.01*i for i in range(7)], labels = [], minor = True)
	axes[-1].set_xlim([0.93,1.02])

	if savefile is not None: plt.savefig(savefile, transparent = True)	
	#plt.show()

def plot_MM_study(ax, out_dict):
	for i, N_t in enumerate(out_dict['N_tiles']):
		perc = np.percentile(out_dict['MM_metric'][i,:], 1) if np.all(out_dict['MM_full'][i,:]==0.) else np.minimum(np.percentile(out_dict['MM_full'][i,:], 1), np.percentile(out_dict['MM_metric'][i,:], 1))
		perc = np.array([perc, 1])
		MM_grid = np.linspace(*perc, 30)
		bw = np.diff(perc)/10
		
		ax.plot(np.repeat(N_t, 2), perc, '--', lw = 1, c='k')
			#creating a KDE for the plots
		scale_factor = 0.3
		kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(out_dict['MM_metric'][i,:, None])
		pdf_metric = np.exp(kde.score_samples(MM_grid[:,None]))
		ax.plot(N_t*(1-scale_factor*(pdf_metric-np.min(pdf_metric))/np.max(pdf_metric-pdf_metric[0])), MM_grid,
						c= 'b', label = 'Metric Match' if i==0 else None)

		if not np.all(out_dict['MM_full'][i,:]==0.):
			kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(out_dict['MM_full'][i,:, None])
			pdf_full = np.exp(kde.score_samples(MM_grid[:,None]))
			ax.plot(N_t*(1+scale_factor*(pdf_full-np.min(pdf_full))/np.max(pdf_full-pdf_full[0])), MM_grid,
						c= 'orange', label = 'Full Match' if i==0 else None)
		
	#plt.yscale('log')
	ax.set_xscale('log')
	ax.axhline(out_dict['MM_inj'], c = 'r')
	ax.set_xlabel(r"$N_{tiles}$")
	ax.set_ylabel(r"${Match}$")
	ax.set_ylim((0.94,1.001))
	ax.legend(loc = 'lower right')

def plot_placing_validation(format_files, placing_methods, savefile = None):

	size = plt.rcParams.get('figure.figsize')
	size = (size[0]*2, size[1]*len(placing_methods)/1.5)
	fig, axes = plt.subplots(len(placing_methods), len(format_files), sharex = False, figsize = size)
	
	for i, variable_format in enumerate(format_files):
		for j, method in enumerate(placing_methods):
			print(variable_format, method)
			text_dict = {'rotation':'horizontal', 'ha':'center', 'va':'center', 'fontsize':13, 'fontweight':'extra bold'}
			if j==0: axes[j,i].set_title(variable_format, pad = 20, **text_dict)
			text_dict['rotation'] = 'vertical'
			if i==0: axes[j,i].text(.8, 0.97, method, text_dict )
				#load file
			try:
				with open(format_files[variable_format].format(method), 'rb') as filehandler:
					out_dict = pickle.load(filehandler)
			except FileNotFoundError:
				axes[j,i].text(0.05, 0.5, 'Work in progress :)', {'fontsize':11})
				continue
				#plot
			plot_MM_study(axes[j,i], out_dict)
			
	
	plt.tight_layout()
	if savefile is not None: plt.savefig(savefile, transparent = True)	

	#plt.show()


########################################################################################################
if __name__ == '__main__':
	img_folder = '../tex/img/'

		###
		#metric accuracy plots
	metric_accuracy_filenames = ['metric_accuracy/paper_Mq_nonspinning.pkl',
				'metric_accuracy/paper_Mq_chi.pkl', 'metric_accuracy/paper_Mq_s1xz_iota.pkl']
				#'metric_accuracy/paper_Mq_s1z_s2z.pkl', 'metric_accuracy/paper_Mq_s1xz_s2z.pkl']
	plot_metric_accuracy(metric_accuracy_filenames, img_folder+'metric_accuracy.pdf')

	#plt.show(); quit()
		###
		#validation of placing methods
	variable_format_files = {'Mq_nonspinning': 'placing_methods_accuracy/paper_nonspinning/data_Mq_nonspinning_{}.pkl',
							'Mq_chi': 'placing_methods_accuracy/paper_chi/data_Mq_chi_{}.pkl',
							'Mq_s1xz_iota': 'placing_methods_accuracy/paper_precessing/data_Mq_s1xz_iota_{}.pkl'}
	placing_methods = ['uniform', 'random', 'stochastic']
	plot_placing_validation(variable_format_files, placing_methods, savefile = img_folder+'placing_validation.pdf')
	
	
	










