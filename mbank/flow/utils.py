"""
mbank.flow.utils
================
		Plotting utilities for the `mbank.flow`
"""

from mbank.handlers import variable_handler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

def compare_probability_distribution(data_flow, data_true = None, variable_format = None, title = None, savefile = None):
	"""
	Make a nice contour plot for visualizing the 2D slices of a multidimensional PDF.
		
	Parameters
	----------
		writeme

	"""
	var_handler = variable_handler()
	labels = var_handler.labels(variable_format, latex = False) if isinstance(variable_format, str) else None
	
	plot_data = pd.DataFrame(data_flow, columns = labels)
	if data_true is not None:
		temp_plot_data = pd.DataFrame(data_true, columns = labels)
		plot_data = pd.concat([plot_data, temp_plot_data], axis=0, ignore_index = True)
		
	plot_data['distribution'] = 'flow'
	if data_true is not None:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			plot_data['distribution'][len(data_true):] = 'train'
	
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

	g = sns.PairGrid(plot_data, hue="distribution", hue_order = [ 'train','flow'])
	g.map_upper(sns.scatterplot, s = 1)
	g.map_lower(sns.kdeplot)
	g.map_diag(sns.kdeplot, lw=2, legend=False)
	g.add_legend()

	if isinstance(title, str): plt.suptitle(title)
	if isinstance(savefile, str):plt.savefig(savefile)

	plt.close('all')
	
	return
