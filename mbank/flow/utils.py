"""
mbank.flow.utils
================

		Plotting & validation utilities for the `mbank.flow`
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import scipy.stats
import numpy as np

import os
import re
import imageio.v2 as iio
import torch

########################################################################

class validation_metric():
	"Base class for validation metric: the method get_dist should be implemented by subclasses"
	def __init__(self, data, flow, N_estimation = 1000):
		
		self.data = data
		self.flow = flow
		self.N_samples = len(data)
		
		dist_list = []
		for n in range(N_estimation):
			true_noise = flow._distribution.sample(self.N_samples).detach().numpy()
			dist_list.append(self.get_dist(true_noise))
		dist_list = np.array(dist_list)
		
		self.metric_mean = np.mean(dist_list)
		self.metric_std = np.std(dist_list)
		self.metric_std_of_mean = np.std(dist_list)/np.sqrt(N_estimation)
		
		return

	def get_dist(self, data):
		"Measures the distance between data and the noise distribution (a standard normal)"
		raise NotImplementedError("The distance between two dataset must be implemented by sub-classes!")

	def get_validation_metric(self):
		"Check if the data transformed into noise are consistent with the random normal distribution"
		
		noise_data = self.flow.transform_to_noise(self.data).detach().numpy()
		dist = self.get_dist(noise_data)
		
		return dist	

class ks_metric(validation_metric):
	"Class to compute the validation metric using the Kolmogorov-Smirnov test"
	def get_dist(self, data):
		true_noise = self.flow._distribution.sample(self.N_samples).detach().numpy()
		pval = 1.
		for d in range(data.shape[1]):
			_, pvalue = scipy.stats.kstest(data[:,d], true_noise[:,d])
			pval *= pvalue

		return np.log10(pval+1e-300)

class cross_entropy_metric(validation_metric):
	"Class to compute the validation metric using the Cross Entropy distance"
	def get_dist(self, data):
		return self.flow._distribution.log_prob(data).mean()
	
########################################################################
	
def plot_loss_functions(history, savefolder = None):
	"Makes some basic plots of the validation"

	if isinstance(savefolder, str):
		if not savefolder.endswith('/'): savefolder = savefolder+'/'
	
	train_loss = history['train_loss']
	validation_loss = history['validation_loss']
	metric_mean, metric_std = history['log_pvalue_mean'], history['log_pvalue_std']
	metric = history['log_pvalue']
	validation_epoch = range(0, len(train_loss), history['validation_step'])
	
	plt.figure()
	plt.plot(range(len(train_loss)), train_loss, label = 'train')
	plt.plot(validation_epoch, validation_loss, label = 'validation')
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	if isinstance(savefolder, str): plt.savefig(savefolder+"loss.png")
	
	plt.figure()
	plt.plot(validation_epoch, metric, c= 'b', label = 'validation metric')
	plt.gca().fill_between(validation_epoch, metric_mean - metric_std, metric_mean + metric_std, alpha = 0.5, color='orange')
	plt.axhline(metric_mean, c = 'r', label = 'expected value')
	plt.xlabel("Epoch")
	plt.ylabel(r"$\log(p_{value})$")
	plt.legend()
	if isinstance(savefolder, str): plt.savefig(savefolder+"p_value_metric.png")
	
	return

def create_gif(folder, savefile, fps = 1):
	"Given a folder of plots generated by a callback, it creates a gif summarizing the training history"
	#https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
	
	if not folder.endswith('/'): folder = folder+'/'
	filenames = os.listdir(folder)
	
	good_files, epochs_list = [], []
	for f in filenames:
		num_regex = re.findall(r'\d+', f)
		if len(num_regex)==0: continue
		epochs_list.append(int(num_regex[0]))
		good_files.append(f)
	
	ids_ = np.argsort(epochs_list)
	
	with iio.get_writer(savefile, mode='I', fps=fps) as writer:
		for id_ in ids_:
		    image = iio.imread(folder+good_files[id_])
		    writer.append_data(image)
	return


def plotting_callback(model, epoch, dirname, data_to_plot, variable_format, basefilename = None):
	"An example callback for plotting the KDE pairplots."

	if not os.path.isdir(dirname): os.mkdir(dirname)
	if not dirname.endswith('/'): dirname= dirname+'/'
	
	if isinstance(basefilename, str):
		savefile= '{}/{}_{}.png'.format(dirname, basefilename, epoch)
	else:
		savefile= '{}/{}.png'.format(dirname, epoch)
	
	
	data_flow = model.sample(data_to_plot.shape[0]).detach().numpy()
	compare_probability_distribution(data_flow, data_true = data_to_plot, variable_format = variable_format, title = 'epoch = {}'.format(epoch), savefile = savefile )
	return

def compare_probability_distribution(data_flow, data_true = None, variable_format = None, title = None, hue_labels = ['flow', 'train'], savefile = None, show = False):
	"""
	Make a nice contour plot for visualizing the 2D slices of a multidimensional PDF.
		
	Parameters
	----------
		writeme

	"""
		#FIXME: very likely this will result in circular imports at some point. The best way to fix this is to separate variable_handlers and tiling_handlers in separate files (but takes time).
	from mbank.handlers import variable_handler
	var_handler = variable_handler()
	labels = var_handler.labels(variable_format, latex = False) if isinstance(variable_format, str) else None
	hue_labels = list(hue_labels)
	
	plot_data = pd.DataFrame(data_flow, columns = labels)
	if data_true is not None:
		temp_plot_data = pd.DataFrame(data_true, columns = labels)
		plot_data = pd.concat([plot_data, temp_plot_data], axis=0, ignore_index = True)
		
	plot_data['distribution'] = hue_labels[0]
	if data_true is not None:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			plot_data['distribution'][len(data_true):] = hue_labels[1]
	
	bins_dat = 40
	if False:
		kdeplot_div = sns.jointplot(
			data=plot_data,
			x=labels[0],
			y=labels[1],
			kind="kde",
			hue="distribution",
			ratio=3, 
			marginal_ticks=True,
			levels=8
		)

	g = sns.PairGrid(plot_data, hue="distribution", hue_order = hue_labels[::-1] if data_true is not None else hue_labels[:1])
	g.map_upper(sns.scatterplot, s = 1)
	g.map_lower(sns.kdeplot, levels=8)
	#g.map_diag(sns.kdeplot, lw=2, legend=False)
	g.map_diag(sns.histplot, element = 'step')
	g.add_legend()

	if isinstance(title, str): plt.suptitle(title)
	if isinstance(savefile, str):plt.savefig(savefile)
	if show: plt.show()

	plt.close('all')
	
	return















