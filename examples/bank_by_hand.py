"""
Example on how to generate a template bank by hand
"""
	##
	# Imports
	##
from mbank import variable_handler, cbc_metric, cbc_bank
from mbank.utils import load_PSD, plot_tiles_templates, get_boundaries_from_ranges
from mbank.placement import place_random_flow
from mbank.flow import STD_GW_Flow
from mbank.flow.utils import early_stopper, plot_loss_functions
from tqdm import tqdm
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

	##
	# Initializing the metric & boundaries
	##

variable_format = 'Mq_chi'
boundaries = np.array([[30,1,-0.99],[50,5,0.99]])
	#Another option using get_boundaries_from_ranges
boundaries = get_boundaries_from_ranges(variable_format,
		(30, 50), (1, 5), chi_range = (-0.99, 0.99))

metric = cbc_metric(variable_format,
			PSD = load_PSD('aligo_O3actual_H1.txt', True, 'H1'),
			approx = 'IMRPhenomD',
			f_min = 10, f_max = 1024)

	##
	# Generating training data
	##

train_data = np.random.uniform(*boundaries, (10000, 3))
validation_data = np.random.uniform(*boundaries, (300, 3))
train_ll = np.array([metric.log_pdf(s) for s in tqdm(train_data)])
validation_ll = np.array([metric.log_pdf(s) for s in tqdm(validation_data)])

	##
	# Initializing, training and testing the normalizing flow
	##

flow = STD_GW_Flow(3, n_layers = 2, hidden_features = 30)

early_stopper_callback = early_stopper(patience=20, min_delta=1e-3)
optimizer = optim.Adam(flow.parameters(), lr=5e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold = .02, factor = 0.5, patience = 4)
	
history = flow.train_flow('ll_mse', N_epochs = 10000,
	train_data = train_data, train_weights = train_ll,
	validation_data = validation_data, validation_weights = validation_ll,
	optimizer = optimizer, batch_size = 500, validation_step = 100,
	callback = early_stopper_callback, lr_scheduler = scheduler,
	boundaries = boundaries, verbose = True)

residuals = np.squeeze(validation_ll) - flow.log_volume_element(validation_data)

	##
	# Placing the templates
	##

new_templates = place_random_flow(0.97, flow, metric,
	n_livepoints = 500, covering_fraction = 0.9,
	boundaries_checker = boundaries,
	metric_type = 'symphony', verbose = True)
bank = cbc_bank(variable_format)
bank.add_templates(new_templates)

	##
	# Saving the template banks and the flow
	##

flow.save_weigths('flow.zip')
bank.save_bank('bank.dat')

	##
	# Doing some plots
	##

	#Training data
plt.figure()
plt.scatter(train_data[:,0], train_data[:,1], c = train_ll, s = 5)
plt.colorbar()

	#Loss function and flow accuracy
plot_loss_functions(history)

plt.figure()
plt.hist(residuals/np.log(10),
	histtype = 'step', bins = 30, density = True)
plt.xlabel(r"$\log_{10}(M_{flow}/M_{true})$")

	#Template bank
plt.figure()
plt.scatter(bank.M, bank.q, s = 5)
plt.xlabel('M')
plt.ylabel('q')

plt.figure()
plt.scatter(bank.q, bank.chi, s = 5)
plt.xlabel('q')
plt.ylabel('chi')

plt.show()
