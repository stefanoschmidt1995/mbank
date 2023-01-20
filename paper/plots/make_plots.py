"""
Script to make all the plots for the paper. It loads several files around and plot the results
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use(["seaborn-colorblind",])
#plt.style.use(['science','ieee', 'bright']) #https://github.com/garrettj403/SciencePlots
#plt.rcParams['axes.spines.right'] = False
#plt.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['cornflowerblue', 'tomato', 'mediumseagreen', 'orchid',  'darkorange'])
plt.rcParams['figure.figsize']=(3.29,2.8)
plt.rcParams['figure.dpi']= 100
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False
mpl.rcParams.update({
	"text.usetex": True,
	"font.family": "Times new Roman"
})

import matplotlib
from matplotlib.lines import Line2D

from mbank import cbc_bank, variable_handler

from scipy.integrate import quad

from tqdm import tqdm

from sklearn.neighbors import KernelDensity
import scipy.stats as sts

import warnings
import shutil
import pickle, json
import os, sys
import itertools

########################################################################################################

def corner_plot(bank_file, variable_format, title = None, savefile = None):
	#TODO: save the bank to pdf
	#import corner
	bank = cbc_bank(variable_format, bank_file)
	vh = variable_handler()
	bank_labels = vh.labels(variable_format, latex = True)
	D = bank.D

	np.random.shuffle(bank.templates)
	#bank.templates = bank.templates[:2000]

	print("Bank size: ", bank.templates.shape[0])
	
	size = plt.rcParams.get('figure.figsize')
	#size = (1.5**2*size[0]*bank.D/4., 1.5*size[1]*1.5/4*bank.D)
	ysize = size[0]*1.5/4*bank.D + 0.3#if bank.D>3 else size[1]*1.6/4*bank.D
	size = (size[0]*bank.D/4 + 1.5, ysize)
	fig, axes = plt.subplots(D,D, figsize = size)
	if isinstance(title, str): plt.suptitle(title)
	
	for i,j in itertools.product(range(D), range(D)):
		ax = axes[i,j]
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		
			#above diagonal elements
		if i<j: ax.axis('off')
	
			#diagonal elements
		if i==j:
			hist = bank.templates[:,i]
			#ax.hist(bank.templates[:,i])
			kde = sts.gaussian_kde(hist)
			x = np.linspace(np.min(hist), np.max(hist), 2000)
			ax.plot(x, kde.pdf(x), lw=1)

			ax.yaxis.set_ticklabels([])
			ax.spines.right.set_visible(False)
			ax.spines.top.set_visible(False)
		
		if i>j:
			#ax.scatter(bank.templates[:,j], bank.templates[:,i], s = .09, edgecolors='none', alpha = 0.7)

			scatter_fig = plt.figure()
			plt.gca().scatter(bank.templates[:,j], bank.templates[:,i], s = .21, edgecolors='none', alpha = 0.8)
			plt.axis('off')
			plt.tight_layout()
			plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
			plt.margins(0,0)
			plt.savefig('temp.png', bbox_inches='tight')
			plt.close(scatter_fig)

			im = plt.imread('temp.png') # insert local path of the image.
			#lims = (np.min(bank.templates[:,j]), np.max(bank.templates[:,j]), np.min(bank.templates[:,i]), np.max(bank.templates[:,i]))
			lims = (np.min(bank.templates[:,j]), np.max(bank.templates[:,j]), np.min(bank.templates[:,i]), np.max(bank.templates[:,i]))
			ax.imshow(im, extent = lims,  aspect = 'auto')
			#ax.set_xlim(np.min(bank.templates[:,j]), np.max(bank.templates[:,j]))
			#ax.set_ylim(np.min(bank.templates[:,i]), np.max(bank.templates[:,i]))
			
			os.remove('temp.png')

				
				#setting labels
		if j==0 and i!=0:
			ax.set_ylabel(bank_labels[i])
		else:
			empty_string_labels = ['' for item in ax.get_yticklabels()]
			ax.set_yticklabels(empty_string_labels)
		if i==D-1:
			ax.set_xlabel(bank_labels[j])
				#dirty but ok
			ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(x) if x!=0.5 else ''))
		else:
			empty_string_labels = ['' for item in ax.get_xticklabels()]
			ax.set_xticklabels(empty_string_labels)
		ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(3))
		ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(3))
		ax.tick_params(axis='both', which='minor', length = 1.5)

	#plt.tight_layout()
	if isinstance(savefile, str): plt.savefig(savefile)
	del bank
	#plt.show()
	#quit()

def plot_distance_vs_match(filenames, savefile = None):
	"Makes the coordinate distance vs match plots"

	dict_list = []
	for filename in filenames:
		try:
			with open(filename, 'rb') as filehandler:
				dict_list.append( pickle.load(filehandler) )
		except:
			warnings.warn("No file {} found. Skipping".format(filename) )

		#creating the figures
	MM = 0.97
	size = plt.rcParams.get('figure.figsize')
	size = (size[0], size[1]*1.5)
	fig, axes = plt.subplots(len(dict_list), 1, sharex = False, figsize = size)
	
	arg_dict = {'s':.2, 'edgecolors':'none'}
	
	for i, (ax, out_dict) in enumerate(zip(axes, dict_list)):
	
		id_ = np.where(np.array(out_dict['MM_list'])==MM)[0][0]	
	
		try:
			dist_vector = np.linalg.norm(out_dict['theta'][...,id_]-out_dict['center'], axis = 1)
			#dist_vector = np.sqrt(np.linalg.det(out_dict['metric']))
			#dist_vector = out_dict['center'][:,0]
		except:
				#old format for out_dict
			dist_vector = np.linalg.norm(out_dict['theta1'][...,id_]-out_dict['theta2'][...,id_], axis = 1)
		
		ids_plot = np.where(dist_vector<np.nanpercentile(dist_vector, 99))[0]
	
		next(ax._get_lines.prop_cycler)
		ax.scatter(dist_vector[ids_plot], 1-out_dict[0.97][ids_plot], **arg_dict)
	
		ax.axhline(1-MM, ls ='--', c = 'k', alpha = 0.5, lw = 1)
		#ax.axvline(1., ls ='--', c = 'k', alpha = 0.5, lw = 1)
		ax.set_ylabel(r'$1-\mathcal{M}$')
		ax.set_yscale('log')
		#ax.set_xscale('log')
		ax.set_ylim((1e-3, 1.))
		
		ticks_y_formatter = ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x) if (x in [1, 0.1, 1e-2]) else '') #formatter
		ax.yaxis.set_major_formatter(ticks_y_formatter)
		ax.annotate(r'$\texttt{'+out_dict['variable_format']+'}$', xy = (.97,0.2), xycoords = 'axes fraction', ha = 'right')
		
	axes[-1].set_xlabel(r'$||\Delta\theta||$')
	#axes[-1].set_xlabel(r'$\sqrt{|M|}$')
	#axes[-1].set_xlabel(r'$q$')


	plt.tight_layout()
	#plt.show()
	if savefile is not None: plt.savefig(savefile, transparent = True)	

	del dict_list

def plot_metric_accuracy(filenames, savefile = None, title = None, dist_cutoff = np.inf):
	"Plot the metric accuracy plots"
		#creating the figures
	nbins = 50
	size = plt.rcParams.get('figure.figsize')
	size = (size[0], size[1]*1.3)
	fig, axes = plt.subplots(len(filenames), 1, figsize = size,sharex = True)
	if isinstance(title, str): plt.suptitle(title)
	
	for i, (ax, filename) in enumerate(zip(axes, filenames)):
		try:
			with open(filename, 'rb') as filehandler:
				out_dict = pickle.load(filehandler)
		except:
			warnings.warn("No file {} found. Skipping".format(filename) )
			continue

		#out_dict['MM_list'].remove(0.95) #this is to remove the 0.95 part

		#ax.title('{}'.format(out_dict['variable_format']))
		next(ax._get_lines.prop_cycler)

		for MM in out_dict['MM_list']:
				#KDE
			id_MM = np.where(np.array(out_dict['MM_list'])==MM)[0][0]
			hist_values = np.delete(out_dict[MM], np.where(np.logical_or(
										np.isnan(out_dict[MM]),
										np.linalg.norm(out_dict['theta'][...,id_MM]-out_dict['center'], axis = 1)>dist_cutoff
										)))
			#print(out_dict['variable_format'], MM, len(out_dict[MM]), len(hist_values))
			
			kde = sts.gaussian_kde(hist_values)
			x = np.logspace(np.log10(np.nanpercentile(out_dict[MM], .01)) ,#if MM<0.999 else np.log10(0.99),
							np.log10(np.nanpercentile(out_dict[MM], 100)) if MM<0.999 else 0.,
							1000)
			label = None
			if i==1 and MM<=0.97: label = MM
			if i==2 and MM>0.97: label = MM
			ax.plot(x, kde.pdf(x)/np.max(kde.pdf(x)), lw=1, label = label)
			
			#bins = np.logspace(np.log10(np.nanpercentile(out_dict[MM], .1)), 0, nbins)
			#bins = np.logspace(np.log10(0.93), 0, nbins)			
			#ax.hist(out_dict[MM], bins = bins, histtype='step', density = True, label = MM if i ==1 else None)

			ax.axvline(MM, c = 'k', ls = 'dashed', alpha = 0.5)
		
		str_label = r'$\texttt{'+out_dict['variable_format']+'}$'
		ax.annotate(str_label, xy = (.03,0.2), xycoords = 'axes fraction')
		if i ==1 or i ==2: ax.legend(loc = 'center right', handlelength = 1, labelspacing = .1, fontsize = 7)
		
	axes[-1].set_xlabel('$\mathcal{M}$', fontsize = 10)

	axes[-1].set_xticks(out_dict['MM_list'], labels = [str(MM) for MM in out_dict['MM_list']], fontsize = 8)
	axes[-1].tick_params(axis = 'x', labelleft = True)
	min_MM_val = 0.9
	axes[-1].set_xticks([min_MM_val+0.01*i for i in range(int((1-min_MM_val)*100))], labels = [], minor = True)
	axes[-1].set_xlim([min_MM_val,1.02])

	axes[-1].get_xticklabels()[0].set_horizontalalignment('left')
	
	plt.tight_layout()

	if savefile is not None: plt.savefig(savefile, transparent = True)

	del out_dict	
	
	
def plot_MM_study(ax, out_dict, set_labels = 'both', set_legend = True):
	id_N_templates = np.where(np.array(out_dict['MM_list'])==out_dict['MM_inj'])[0]
	#out_dict['N_templates'] = np.log10(out_dict['N_templates'])
	max_N_templates, min_N_templates = np.max(out_dict['N_templates'][:,id_N_templates]), np.min(out_dict['N_templates'][:,id_N_templates])
	
	for i, N_t in enumerate(out_dict['N_tiles']):
		perc = np.percentile(out_dict['MM_metric'][i,:], 1) if np.all(out_dict['MM_full'][i,:]==0.) else np.minimum(np.percentile(out_dict['MM_full'][i,:], 1), np.percentile(out_dict['MM_metric'][i,:], 1))
		perc = np.array([perc, 1])
		MM_grid = np.sort([*np.linspace(*perc, 30), out_dict['MM_inj']])
		bw = np.diff(perc)/10
		
			#creating a KDE for the plots
		scale_factor = 0.5
		kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(out_dict['MM_metric'][i,:, None])
		pdf_metric = np.exp(kde.score_samples(MM_grid[:,None]))

		N_templates = out_dict['N_templates'][i, id_N_templates]
		print(out_dict['placing_method'], N_templates)
		x_hist = N_t*(1-scale_factor*(pdf_metric-np.min(pdf_metric))/np.max(pdf_metric-pdf_metric[0]))
		
			#dealing with the grid
				#this control the length of the support for the MM hist
				#TODO: play with this stretch factor...
		if out_dict['variable_format']=='Mq_chi':
			y_strecth = 1.2 #0.0008
			y_min, y_max = 10,2000
				#do not plot the second to last point
			if i == len(out_dict['N_tiles'])-1: continue
		elif out_dict['variable_format']=='Mq_s1xz':
			y_strecth = 1.9 #4e-5
			y_min, y_max = 2,50_000
		elif out_dict['variable_format']=='Mq_s1xz_s2z_iota':
			y_strecth = 1.9 #1e-7
			y_min, y_max = 2000, 2_000_000
		y_min, y_max = y_min*0.95, y_max*1.5
		ax.set_ylim([y_min, y_max])

		#transform_grid = lambda x: (max_N_templates - min_N_templates)*y_strecth*(x-1)/(1-out_dict['MM_inj'])
		transform_grid = lambda x: y_strecth*((y_max - y_min)/(y_max))*(x-1)/(1-out_dict['MM_inj'])
		id_MM = np.where(MM_grid ==out_dict['MM_inj'])[0][0]
		MM_grid_transformed = transform_grid(MM_grid)
		tick_location = MM_grid_transformed[[id_MM,-1]]
			
			#support of the histogram
		ax.plot(np.repeat(N_t, 2), np.exp(MM_grid_transformed[[0,-1]])*N_templates, '--', lw = 1, c='k', alpha = 0.5) 
		ax.plot(x_hist, np.exp(MM_grid_transformed)* N_templates,
						c= 'cornflowerblue', label = 'Metric Match' if i==0 else None)
		ax.scatter(N_t, N_templates, marker ='x', c='k', s = 6) #data point
		len_red_tick = np.abs(N_t - np.min(x_hist))/2.
		ax.plot([N_t-len_red_tick,  N_t+len_red_tick], np.full(2, N_templates *np.exp( MM_grid_transformed[id_MM])), '-', c= 'r', lw =1) #MM ticks
		
		if not np.all(out_dict['MM_full'][i,:]==0.):
			#TODO: you should tackle the case here...
			kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(out_dict['MM_full'][i,:, None])
			pdf_full = np.exp(kde.score_samples(MM_grid[:,None]))
			x_hist = N_t*(1+scale_factor*(pdf_full-np.min(pdf_full))/np.max(pdf_full-pdf_full[0]))
			ax.plot(x_hist, np.exp(MM_grid_transformed) * N_templates,
						c= 'darkorange', label = 'Match' if i==0 else None)
		
	ax.set_yscale('log')
	ax.set_xscale('log')
	#ax.axhline(out_dict['MM_inj'], c = 'r')
	if set_labels in ['x', 'both']: ax.set_xlabel(r"$N_\mathrm{tiles}$", fontsize = 12)
	if set_labels in ['y', 'both']: label = ax.set_ylabel(r"$N_\mathrm{templates}$", rotation = 270, fontsize = 12, labelpad=17)
	#ax.set_ylim((0.94,1.001))
	if set_legend: ax.legend(loc = 'lower right', fontsize = 10)

		#This will make sure that all the axis will come with minor ticks
	ax.yaxis.get_minor_locator().set_params(numticks = 100000)
	ax.xaxis.get_minor_locator().set_params(numticks = 100000)
	ax.yaxis.get_major_locator().set_params(numticks = 30)
	ax.xaxis.get_major_locator().set_params(numticks = 4)

def plot_placing_validation(format_files, placing_methods, savefile = None):

	size = plt.rcParams.get('figure.figsize')
	size = (size[0]*2.2, size[1]*len(placing_methods)/1.5)
	fig, axes = plt.subplots(len(placing_methods), len(format_files), sharex = False, figsize = size)
	
	for i, variable_format in enumerate(format_files):
		for j, method in enumerate(placing_methods):
				#load file
			try:
				with open(format_files[variable_format].format(method), 'rb') as filehandler:
					out_dict = pickle.load(filehandler)
			except FileNotFoundError:
				axes[j,i].text(0.05, 0.5, 'Work in progress :)', {'fontsize':11})
				axes[j,i].set_yticks([])
				axes[j,i].set_xticks([])
				continue
			print(variable_format, method, out_dict['MM_metric'].shape)
				#plot
			axes[j,i].yaxis.set_label_position("right")
			axes[j,i].yaxis.tick_right()
			if i == len(axes[0])-1:
				label_position = 'y'
				if j == len(axes)-1: label_position = 'both'
			elif j == len(axes)-1:
				label_position = 'x'
			else:
				label_position = None
			plot_MM_study(axes[j,i], out_dict, label_position, False if (i,j)!=(0,1) else True)
			text_dict = {'rotation':'horizontal', 'ha':'center', 'va':'center', 'fontsize':13, 'fontweight':'extra bold'}
			if j==0: axes[j,i].set_title(r'$\texttt{'+variable_format+'}$', pad = 20, **text_dict)
			text_dict['rotation'] = 'vertical'
			y_center = 10**np.mean(np.log10(axes[j,i].get_ylim()))
			if i==0: axes[j,i].text(0.1, y_center, method, text_dict )
			del out_dict

	plt.tight_layout()
	if savefile is not None: plt.savefig(savefile, transparent = True)	

	#plt.show()

def plot_delta_M(file_list, savefile = None):
	
	D = len(file_list)
	
	size = plt.rcParams.get('figure.figsize')
	size = (size[0], size[1]*D*0.45)
	fig, axes = plt.subplots(D, 1, sharex = not True, figsize = size)
	
	for f, ax in zip(file_list, axes):
		with open(f, 'r') as filehandler:
			out_dict = json.load(filehandler)

		x = np.logspace(-4, 2, 1000)
		kde_flow = sts.gaussian_kde(out_dict['logMratio_flow'])
		kde_tiling = sts.gaussian_kde(out_dict['logMratio_tiling'])
		#ax.plot(x, kde_flow.pdf(x), lw=1, label='flow')
		#ax.plot(x, kde_tiling.pdf(x), lw=1, label='tiling')
		
		nbins = int(np.sqrt(len(out_dict['deltaM_tiling'])))
		perc = 1.
		bins = np.linspace(*np.percentile(out_dict['logMratio_tiling'], [perc,100 -perc]), nbins)
		hist_args = {
			'bins': bins,
			'density': True,
			'histtype': 'step'
		}
		
		#next(ax._get_lines.prop_cycler)		
		ax.hist(out_dict['logMratio_flow'], label = 'flow', **hist_args)
		ax.hist(out_dict['logMratio_tiling'], label = 'no flow', **hist_args)
		
		ax.set_yscale('log')

		ax.tick_params(axis='both', which='major', labelsize=8)
		ax.tick_params(axis='both', which='minor', labelsize=7)
		
		#ax.yaxis.get_minor_locator().set_params(numticks = 100000)
		
		ax.set_title(r'$\texttt{'+out_dict['variable_format']+'}$', fontsize = 10)
		#ax.set_xlim(np.percentile(out_dict['logMratio_tiling'], [perc,100 -perc]))
		
	axes[1].legend(loc = 'upper right', fontsize = 8)
	#axes[-1].set_xlim([x_low_lim,1.001])
		
	axes[-1].set_xlabel(r"$\frac{1}{2}\log_{10} \frac{\mathrm{det}M^\mathrm{tiling}}{\mathrm{det} M}$", fontsize = 10)

	plt.tight_layout()	

	if savefile is not None: plt.savefig(savefile, transparent = True)
		

def plot_comparison_injections(files, labels, keys, title = None, c_list = None, MM = None, x_low_lim = 0.9, savefile = None):

	size = plt.rcParams.get('figure.figsize')
	size = (size[0], size[1]*len(files[0])*0.45)
	fig, axes = plt.subplots(len(files[0]), 1, sharex = True, figsize = size)
	
	if not c_list:
		c_list = [None for _ in files]

	if title is None: title = [None for _ in list_A]

	for i, (ax, t) in enumerate(zip(axes, title)):
		injs_dicts = []
	
		for pkl_list in files:
			with open(pkl_list[i], 'rb') as filehandler:
				injs_dicts.append(pickle.load(filehandler))
			injs_dicts[-1].pop('match_list', None)
			injs_dicts[-1].pop('id_match_list', None)
			print(pkl_list[i], len(injs_dicts[-1]['theta_inj']))

			#making the KDE with scipy
		min_x = np.percentile(injs_dicts[0][keys[0]], .01)
		x = np.linspace(min_x, 1, 1000)

		for inj_dict, key, label, color in zip(injs_dicts, keys, labels, c_list):
			kde_ = sts.gaussian_kde(inj_dict[key])
			ax.plot(x, np.cumsum(kde_.pdf(x))*np.diff(x)[0], lw=1, label=label, c=color)


		next(ax._get_lines.prop_cycler)
		
		ax.set_ylim([1e-4,1.0])
		ax.set_yscale('log')

		ax.tick_params(axis='both', which='major', labelsize=8)
		ax.tick_params(axis='both', which='minor', labelsize=7)
		
		ax.yaxis.get_minor_locator().set_params(numticks = 100000)
		#ax.yaxis.get_major_locator().set_params(numticks = 30)
		
		if isinstance(MM, float): ax.axvline(MM, c = 'k', ls = '--')
		if isinstance(t, str): ax.set_title(t, fontsize = 10)
	axes[0].legend(loc = 'upper left', fontsize = 8)
	axes[-1].set_xlim([x_low_lim,1.001])
		
	axes[-1].set_xlabel(r"$\mathcal{M}$", fontsize = 10)

	plt.tight_layout()	

	if savefile is not None: plt.savefig(savefile, transparent = True)
	#plt.show()
	del injs_dicts


def plot_bank_hist(bank_list, format_list, title = None, savefile = None):
	"Plot several histogram of different banks"
	
	vh = variable_handler()
	N_colums = np.max([vh.D(f) for f in format_list])
	size = plt.rcParams.get('figure.figsize')
	#size = (size[0], size[1]*0.5)
	size = (size[0]*4, size[1]*0.5)
	
	if title is None: title = [None for _ in bank_list]

	for bank_file, var_format, t in zip(bank_list, format_list, title):
		print(var_format, bank_file)
		bank = cbc_bank(var_format, bank_file)
		templates = bank.templates

		fig, axes = plt.subplots(1, N_colums, figsize = size, sharey = True)	
		if isinstance(t,str): plt.suptitle(t)
		
		hist_kwargs = {'bins': min(50, int(len(templates)/50 +1)), 'histtype':'step', 'color':'darkorange'}
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

def plot_injection_distance_hist(inj_pkl_list, variable_format_list, title = None, savefile = None):

	raise NotImplementedError("This does not work as promised!")
	
	for (inj_pkl, bank_file), variable_format in zip(inj_pkl_list, variable_format_list):
		with open(inj_pkl, 'rb') as filehandler:
			inj_dict = pickle.load(filehandler)
		bank = cbc_bank(variable_format, bank_file)
		h = bank.var_handler
		
		inj_theta = h.get_theta(inj_dict['theta_inj'], variable_format)
		id_match = inj_dict['id_match']
		id_metric_match = np.zeros(id_match.shape, dtype = int) # inj_dict['id_metric_match']
		delta_theta_metric_match =  np.linalg.norm(bank.templates[id_metric_match, :]-inj_theta, axis = 1)
		delta_theta_match = np.linalg.norm(bank.templates[id_match, :]-inj_theta, axis = 1)
		dist_templates = np.linalg.norm(bank.templates[id_metric_match, :]-bank.templates[id_match, :], axis = 1)

	if isinstance(savefile, str):
		plt.savefig(savefile.format(t.replace(' ', '_')))

	del inj_dict

########################################################################################################
if __name__ == '__main__':
	img_folder = '../tex/img/'

		###
		#metric accuracy plots
	metric_accuracy_filenames = ['metric_accuracy/paper_hessian_Mq_nonspinning.pkl',
				'metric_accuracy/paper_hessian_Mq_chi.pkl', 'metric_accuracy/paper_hessian_Mq_s1xz_iota.pkl',
				'metric_accuracy/paper_hessian_Mq_chi_iota.pkl']
	metric_accuracy_parabolic_filenames = [m.replace('paper_hessian', 'paper_parabolic') for m in metric_accuracy_filenames]
	#plot_metric_accuracy(metric_accuracy_filenames, img_folder+'metric_accuracy_hessian.pdf', None, np.inf)
	
		#old garbage
	#plot_distance_vs_match(metric_accuracy_filenames, img_folder+'metric_accuracy_hessian_distance.pdf')
	#plot_metric_accuracy(metric_accuracy_parabolic_filenames, img_folder+'metric_accuracy_parabolic.pdf', None, np.inf)
	#plot_distance_vs_match(metric_accuracy_parabolic_filenames, img_folder+'metric_accuracy_parabolic_distance.png')

		###
		#validation of placing methods
	variable_format_files = {#'Mq_nonspinning': 'placing_methods_accuracy/paper_Mq_nonspinning/data_Mq_nonspinning_{}.pkl',
							'Mq_chi': 'placing_methods_accuracy/paper_Mq_chi/data_Mq_chi_{}.pkl',
							'Mq_s1xz': 'placing_methods_accuracy/paper_Mq_s1xz/data_Mq_s1xz_{}.pkl',
							'Mq_s1xz_s2z_iota': 'placing_methods_accuracy/paper_Mq_s1xz_s2z_iota/data_Mq_s1xz_s2z_iota_{}.pkl',}
	placing_methods = ['uniform', 'random', 'stochastic']
	#plot_placing_validation(variable_format_files, placing_methods, savefile = img_folder+'placing_validation.pdf')

		###
		# Validation of the tiling
	tiling_validation_list = ['tiling_accuracy/out_dict_Mq_chi.json',
			'tiling_accuracy/out_dict_Mq_s1xz.json', 'tiling_accuracy/out_dict_Mq_s1xz_s2z_iota.json']
	#plot_delta_M(tiling_validation_list, img_folder+'tiling_validation.pdf')

		###
		#Comparison with sbank - injections
	sbank_list_injs = []
	mbank_list_injs = []
	for ct in ['nonspinning', 'alignedspin', 'alignedspin_lowmass']:#, 'gstlal']:
		sbank_list_injs.append('comparison_sbank_{}/injections_stat_dict_sbank.pkl'.format(ct))
		mbank_list_injs.append('comparison_sbank_{}/injections_stat_dict_mbank.pkl'.format(ct))
	savefile = img_folder+'sbank_comparison.pdf'
	title = ['Nonspinning', 'Aligned spins high mass', 'Aligned spins low mass']#, 'Gstlal O3 bank']
	
	plot_comparison_injections( (sbank_list_injs, mbank_list_injs), ('sbank', 'mbank'), ('match','match'), MM = 0.97, title = title, savefile = savefile)
	
		###
		#Bank case studies
	format_list = ['Mq_s1xz', 'logMq_chi_iota', 'Mq_nonspinning_e']
	bank_list = ['precessing_bank/bank_paper_precessing.dat', 'HM_bank/bank_paper_HM.dat',
		'eccentric_bank/bank_paper_eccentric.dat']
	title_list = ['Precessing', 'IMBH HM', 'Nonspinning eccentric']
	injs_list = ['precessing_bank/bank_paper_precessing-injections_stat_dict-OLD.pkl', 'HM_bank/bank_paper_HM-injections_stat_dict.pkl',
		'eccentric_bank/bank_paper_eccentric-injections_stat_dict.pkl']

		#plotting bank histograms
	for b, f, t in zip(bank_list, format_list, title_list):
		filename = img_folder+'bank_scatter_{}.pdf'.format(t.replace(' ', '_'))
		#corner_plot(b,f,t, savefile = filename)
		#plt.show()
		
		###
		#Bank case studies flow
	format_list = ['Mq_s1xz', 'logMq_chi_iota', 'Mq_nonspinning_e']
	bank_list = [	'../../flow_banks/precessing_bank/bank_paper_precessing_flow.dat',
					'../../flow_banks/HM_bank/bank_paper_HM_flow.dat',
					'../../flow_banks/eccentric_bank/bank_paper_eccentric_flow.dat']
	title_list = ['Precessing', 'IMBH HM', 'Nonspinning eccentric']

		#plotting bank histograms
	for b, f, t in zip(bank_list, format_list, title_list):
		filename = img_folder+'bank_scatter_{}_flow.pdf'.format(t.replace(' ', '_'))
		#corner_plot(b,f,t, savefile = filename)
		#plt.show()
		
	#plot_bank_hist(bank_list, format_list, title = title_list, savefile = img_folder+'bank_hist_{}.pdf')
	
		#Flow injection recovery
	injs_list_noflow = ['precessing_bank/bank_paper_precessing-injections_stat_dict.pkl', 'HM_bank/bank_paper_HM-injections_stat_dict.pkl',
		'eccentric_bank/bank_paper_eccentric-injections_stat_dict.pkl']
	injs_list_flow = ['precessing_bank/bank_paper_precessing_flow-injections_stat_dict.pkl', 'HM_bank/bank_paper_HM_flow-injections_stat_dict.pkl',
		'eccentric_bank/bank_paper_eccentric_flow-injections_stat_dict.pkl']
	
	#plot_comparison_injections( (injs_list_noflow, injs_list_noflow, injs_list_flow), ('metric match', 'match no flow', 'match flow'), ('metric_match','match', 'match'), c_list = ('darkorange', 'cornflowerblue', 'purple'), MM = 0.97, title = title_list, savefile = img_folder+'bank_injections_flow.pdf')
	
	
	quit()
	
	










