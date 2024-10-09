import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
from ligo.lw.utils import load_filename

import torch
from torch import optim

from mbank import variable_handler, cbc_metric
from mbank.utils import load_PSD, avg_dist, plot_tiles_templates, get_boundaries_from_ranges
from mbank.parser import get_boundary_box_from_args, boundary_keeper
import mbank.parser

from mbank.flow import STD_GW_Flow, GW_Flow
from mbank.flow.utils import early_stopper
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.linear import NaiveLinear
from nflows.transforms.base import Transform, CompositeTransform
from nflows.distributions.normal import StandardNormal

import emcee
from tqdm import tqdm
import argparse
import os
from lal import MTSUN_SI

#######################

parser = argparse.ArgumentParser(__doc__)

mbank.parser.add_general_options(parser)
mbank.parser.add_range_options(parser)
mbank.parser.add_flow_options(parser)

	#Options specific to this program
parser.add_argument(
	"--variable-format", required = False,
	help="Choose which variables to include in the bank. Valid formats are those of `mbank.handlers.variable_format`")
parser.add_argument(
	"--n-datapoints", default = None, type = int,
	help="Number of datapoints to be loaded from the dataset")
parser.add_argument(
	"--dataset", default = ['dataset.dat'], type = str, nargs = '+',
	help="Files with the datasets (more than one are accpted). If the file does not exist, the dataset will be created.")
parser.add_argument(
	"--ignore-boundaries", default = False, action = 'store_true',
	help="Whether to ignore the ranges given by command line. If set, the boudaries will be extracted from the dataset")

args, filenames = parser.parse_known_args()
for f in filenames:
	args = mbank.parser.updates_args_from_ini(f, args, parser)

var_handler = variable_handler()
D = var_handler.D(args.variable_format)

args.run_dir = 'out_hierarchical/'

args.flow_file = args.run_dir + args.flow_file
base_flow_file = 'out_high_dimensional_bank_flow_large/flow_high_dimensional.zip'
dataset_file = 'datasets_high_dimensional/dataset_high_dimensional.dat'

train_fraction = 0.85

if not False:
	dataset = np.loadtxt(dataset_file)
	N_train = int(train_fraction*len(dataset))
	train_data, validation_data = dataset[:N_train,:D], dataset[N_train:,:D]
	train_ll, validation_ll = dataset[:N_train, -1], dataset[N_train:, -1]

boundaries = None if args.ignore_boundaries else get_boundary_box_from_args(args)

base_flow = STD_GW_Flow.load_flow(base_flow_file)


transform_list = []

for transform in base_flow._transform._transforms:
	for m in transform.modules():
		for mm in m.parameters():
			mm.requires_grad = False
	transform_list.append(transform)

for _ in range(args.n_layers):
	transform_list.append(NaiveLinear(features=D))
	transform_list.append(MaskedAffineAutoregressiveTransform(features=D, hidden_features=args.hidden_features))
transform = CompositeTransform(transform_list)

flow = GW_Flow(transform, StandardNormal(shape=[D]), has_constant = True)
#flow._distribution = base_flow

rg = []
for transform in flow._transform._transforms:
	for m in transform.modules():
		for mm in m.parameters():
			rg.append(mm.requires_grad)

early_stopper_callback = early_stopper(patience = args.patience, min_delta = args.min_delta, temp_file = args.flow_file+'.checkpoint', verbose = args.verbose)
optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold = .02, factor = 0.5, patience = 4)
	
#print(flow._transform._transforms[1]._weight[0,1].item()) #This weight here has the grad deactivated!!

history = flow.train_flow(args.loss_function, N_epochs = args.n_epochs,
		train_data = train_data, train_weights = train_ll,
		validation_data = validation_data, validation_weights = validation_ll,
		optimizer = optimizer, batch_size = args.batch_size, validation_step = 100,
		callback = early_stopper_callback, lr_scheduler = scheduler,
		boundaries = boundaries, verbose = args.verbose)

if os.path.isfile(args.flow_file+'.checkpoint'): os.remove(args.flow_file+'.checkpoint')
flow.save_weigths(args.flow_file)
