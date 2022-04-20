"""
Script to make all the plots for the paper. It loads several files around and plot the results
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','ieee', 'bright']) #https://github.com/garrettj403/SciencePlots
import matplotlib

from tqdm import tqdm

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
	
	for ax, out_dict in zip(axes, dict_list):
	
		#ax.title('{}'.format(out_dict['variable_format']))
		next(ax._get_lines.prop_cycler)
		for MM in out_dict['MM_list']:
			bins = np.logspace(np.log10(np.nanpercentile(out_dict[MM], .1)), 0, nbins)
			ax.hist(out_dict[MM], bins = bins, histtype='step')
			ax.axvline(MM, c = 'k', ls = 'dotted')
		ax.set_xscale('log')
		ax.annotate(out_dict['variable_format'], xy = (.03,0.7), xycoords = 'axes fraction')
	axes[-1].set_xlabel('$1-MM$')

	axes[-1].set_xticks(out_dict['MM_list'], labels = [str(MM) for MM in out_dict['MM_list']])
	axes[-1].tick_params(axis = 'x', labelleft = True)
	axes[-1].set_xticks([0.94+0.01*i for i in range(6)], labels = [], minor = True)
	axes[-1].set_xlim([0.94,1.01])

	if savefile is not None: plt.savefig(savefile, transparent = True)	
	plt.show()


########################################################################################################
if __name__ == '__main__':
	img_folder = '../tex/img/'


		#metric accuracy plots
	metric_accuracy_filenames = ['metric_accuracy/paper_Mq_nonspinning.pkl',
				'metric_accuracy/paper_Mq_s1z_s2z.pkl', 'metric_accuracy/paper_Mq_s1xz_s2z.pkl']
	plot_metric_accuracy(metric_accuracy_filenames, img_folder+'metric_accuracy.pdf')













