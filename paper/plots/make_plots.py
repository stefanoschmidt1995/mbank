"""
Script to make all the plots for the paper. It loads several files around and plot the results
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use(['science','ieee', 'bright']) #https://github.com/garrettj403/SciencePlots
import matplotlib
from matplotlib.lines import Line2D

from mbank import cbc_bank, variable_handler


from tqdm import tqdm

from sklearn.neighbors import KernelDensity
import scipy.stats as sts

import warnings
import shutil
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

		#out_dict['MM_list'].remove(0.95) #this is to remove the 0.95 part

		#ax.title('{}'.format(out_dict['variable_format']))
		next(ax._get_lines.prop_cycler)
		#print("N datapoints: ",len(out_dict[0.999]))
		
		for MM in out_dict['MM_list']:
			bins = np.logspace(np.log10(np.nanpercentile(out_dict[MM], .1)), 0, nbins)
			bins = np.logspace(np.log10(0.93), 0, nbins)
			ax.hist(out_dict[MM], bins = bins, histtype='step', density = True, label = MM if i ==1 else None)
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
	axes[-1].set_xticks([0.94+0.01*i for i in range(6)], labels = [], minor = True)
	axes[-1].set_xlim([0.94,1.02])

	if savefile is not None: plt.savefig(savefile, transparent = True)	
	#plt.show()

def plot_MM_study(ax, out_dict, set_labels = 'both'):
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
	if set_labels in ['x', 'both']: ax.set_xlabel(r"$N_{tiles}$")
	if set_labels in ['y', 'both']: ax.set_ylabel(r"${Match}$")
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
			if i==0: axes[j,i].text(0.05, 0.97, method, text_dict )
				#load file
			try:
				with open(format_files[variable_format].format(method), 'rb') as filehandler:
					out_dict = pickle.load(filehandler)
			except FileNotFoundError:
				axes[j,i].text(0.05, 0.5, 'Work in progress :)', {'fontsize':11})
				continue
				#plot
			plot_MM_study(axes[j,i], out_dict, 'x' if i==0 else 'both')
			
	
	plt.tight_layout()
	if savefile is not None: plt.savefig(savefile, transparent = True)	

	#plt.show()

def plot_comparison_injections(list_A, list_b, labels, keys, title = None, MM = None, savefile = None):
	label_A, label_B = labels
	key_A, key_B = keys

	size = plt.rcParams.get('figure.figsize')
	size = (size[0], size[1]*len(list_A)*0.5)
	fig, axes = plt.subplots(len(list_A), 1, sharex = True, figsize = size)

	if title is None: title = [None for _ in list_A]

	for sbank_pkl, mbank_pkl, ax, t in zip(list_A, list_b, axes, title):
		with open(sbank_pkl, 'rb') as filehandler:
			A_inj = pickle.load(filehandler)
		with open(mbank_pkl, 'rb') as filehandler:
			B_inj = pickle.load(filehandler)

			#making the KDE with scipy
		min_x = np.percentile(A_inj[key_A], .1)
		x = np.linspace(min_x, 1, 1000)

		kde_B = sts.gaussian_kde(B_inj[key_B])
		kde_A = sts.gaussian_kde(A_inj[key_A])

		ax.plot(x, kde_B.pdf(x), lw=1, label=label_B)
		ax.plot(x, kde_A.pdf(x), lw=1, label=label_A)
		#ax.hist(B_inj[key_B], label = label_B, density = True, histtype = 'step', bins = 1000)
		#ax.hist(A_inj[key_A], label = label_A, density = True, histtype = 'step', bins = 1000)
		
		if isinstance(MM, float): ax.axvline(MM, c = 'k', ls = '--')
		if isinstance(t, str): ax.set_title(t)
	axes[-1].legend(loc = 'upper left')
	axes[-1].set_xlim([0.94,1.005])
		
	axes[-1].set_xlabel(r"$\mathcal{M}$")
	plt.tight_layout()	

	if savefile is not None: plt.savefig(savefile, transparent = True)	
	#plt.show()


def plot_bank_hist(bank_list, format_list, title = None, savefile = None):
	"Plot several histogram of different banks"
	
	vh = variable_handler()
	N_colums = np.max([vh.D(f) for f in format_list])
	size = plt.rcParams.get('figure.figsize')
	size = (size[0], size[1]*0.5)
	
	if title is None: title = [None for _ in bank_list]

	for bank_file, var_format, t in zip(bank_list, format_list, title):
		print(var_format, bank_file)
		bank = cbc_bank(var_format, bank_file)
		templates = bank.templates

		fig, axes = plt.subplots(1, N_colums, figsize = size, sharey = True)	
		if isinstance(t,str): plt.suptitle(t)
		
		hist_kwargs = {'bins': min(50, int(len(templates)/50 +1)), 'histtype':'step', 'color':'orange'}
		fs = 10
		ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}K'.format(int(x/1e3)) if x >2 else '') #formatter
		
		for i, ax_ in enumerate(axes):

			if i>= templates.shape[1]:
				ax_.axis('off')
				continue
			
			ax_.yaxis.set_major_formatter(ticks_y)
			ax_.yaxis.label.set_visible(False)
			ax_.hist(templates[:,i], **hist_kwargs)
			if i==0: ax_.set_ylabel("templates", fontsize = fs)
			ax_.set_xlabel(vh.labels(var_format, latex = True)[i])#, fontsize = fs)
			
				#removing the spines
			ax_.spines.right.set_visible(False)
			ax_.spines.top.set_visible(False)
			ax_.xaxis.set_ticks_position('bottom')
			ax_.yaxis.set_ticks_position('left')
			if i !=0: ax_.tick_params(labelleft=False)

			min_, max_ = np.min(templates[:,i]), np.max(templates[:,i])
			d_ = 0.1*(max_-min_)
			ax_.set_xlim((min_-d_, max_+d_ ))
			ax_.tick_params(axis='x', labelsize=fs)
			ax_.tick_params(axis='y', labelsize=fs)
		
		if isinstance(savefile, str):
			plt.savefig(savefile.format(t.replace(' ', '_')))
	plt.show()	

########################################################################################################
if __name__ == '__main__':
	img_folder = '../tex/img/'

		###
		#metric accuracy plots
	metric_accuracy_filenames = ['metric_accuracy/paper_Mq_nonspinning.pkl',
				'metric_accuracy/paper_Mq_chi.pkl', 'metric_accuracy/paper_Mq_s1xz_iota.pkl']
	#plot_metric_accuracy(metric_accuracy_filenames, img_folder+'metric_accuracy.pdf')

		###
		#validation of placing methods
	variable_format_files = {'Mq_nonspinning': 'placing_methods_accuracy/paper_nonspinning/data_Mq_nonspinning_{}.pkl',
							'Mq_chi': 'placing_methods_accuracy/paper_chi/data_Mq_chi_{}.pkl',
							'Mq_s1xz_s2z': 'placing_methods_accuracy/paper_precessing/data_Mq_s1xz_s2z_{}.pkl'}
	placing_methods = ['uniform', 'random', 'stochastic']
	#plot_placing_validation(variable_format_files, placing_methods, savefile = img_folder+'placing_validation.pdf')


		###
		#Comparison with sbank - injections
	sbank_list_injs = []
	mbank_list_injs = []
	for ct in ['nonspinning', 'alignedspin', 'alignedspin_lowmass', 'gstlal']:
		sbank_list_injs.append('comparison_sbank_{}/injections_stat_dict_sbank.pkl'.format(ct))
		mbank_list_injs.append('comparison_sbank_{}/injections_stat_dict_mbank.pkl'.format(ct))
	savefile = img_folder+'sbank_comparison.pdf'
	title = ['Nonspinning', 'Aligned Spins', 'Aligned Spins Lowmass', 'Gstlal O3 bank']
	#plot_comparison_injections(sbank_list_injs, mbank_list_injs, ('sbank', 'mbank'), ('match','match'), MM = 0.97, title = title, savefile = savefile)
	
	
		###
		#Bank case studies
	format_list = ['Mq_s1xz', 'Mq_s1xz', 'Mq_nonspinning_e']
	bank_list = ['precessing_bank/bank_paper_precessing.dat', 'precessing_bank/bank_paper_precessing.dat',
		'eccentric_bank/bank_paper_eccentric.dat']
	title_list = ['Precessing', 'Aligned spins HM', 'Nonspinning eccentric']
	injs_list = ['precessing_bank/bank_paper_precessing-injections_stat_dict.pkl', 'precessing_bank/bank_paper_precessing-injections_stat_dict.pkl',
		'eccentric_bank/bank_paper_eccentric-injections_stat_dict.pkl']

		#plotting bank histograms
	plot_bank_hist(bank_list, format_list, title = title_list, savefile = img_folder+'bank_hist_{}.pdf')
		#Plotting injection recovery
	savefile = img_folder+'bank_injections.pdf'
	#plot_comparison_injections(injs_list, injs_list, ('metric match', 'match'), ('metric_match','match'), MM = 0.97, title = title_list, savefile = savefile)
	
	quit()
	
	










