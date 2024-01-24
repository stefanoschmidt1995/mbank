"""
Example on how to perform injections on a template bank
"""
	##
	# Imports
	##
from mbank import variable_handler, cbc_metric, cbc_bank
from mbank.utils import compute_injections_match, get_random_sky_loc, initialize_inj_stat_dict, save_inj_stat_dict
from mbank.utils import load_PSD, plot_tiles_templates
from mbank.flow import STD_GW_Flow
import numpy as np

	##
	# Loading bank, flow and generating the metric
	##
bank = cbc_bank('Mq_chi', 'bank.dat')
flow = STD_GW_Flow.load_flow('flow.zip')

metric = cbc_metric(bank.variable_format,
	PSD = load_PSD('aligo_O3actual_H1.txt', True, 'H1', df = 1),
	approx = 'IMRPhenomD',
	f_min = 10, f_max = 1024)

	##
	# Sampling injections from the flow & initializing stat dictionary
	##
n_injs = 100
injs_3D = flow.sample(n_injs)
injs_12D = bank.var_handler.get_BBH_components(bank.templates, bank.variable_format)
sky_locs = np.column_stack(get_random_sky_loc(n_injs))
stat_dict = initialize_inj_stat_dict(injs_12D, sky_locs = sky_locs)

	##
	# Computing the injection match
	##
inj_stat_dict = compute_injections_match(stat_dict, bank,
	metric_obj = metric, mchirp_window = 0.1, symphony_match = True)
save_inj_stat_dict('injections.json', inj_stat_dict)

	##
	# Plotting
	##
plot_tiles_templates(bank.templates, bank.variable_format,
	injections = injs_3D, inj_cmap = stat_dict['match'], show = True)
