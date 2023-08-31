"""
mbank.parser
============

Gathers group of options useful for the many executables that make mbank handy
"""

import configparser
import numpy as np
import os
from .flow.flowmodel import STD_GW_Flow
from .bank import cbc_bank
from .handlers import variable_handler
from .utils import get_boundaries_from_ranges

####################################################################################################################
#Parser stuff

def int_tuple_type(strings): #type for the grid-size parser argument
	strings = strings.replace("(", "").replace(")", "")
	mapped_int = map(int, strings.split(","))
	return tuple(mapped_int)

def updates_args_from_ini(ini_file, args, parser):
	"""	
	Updates the arguments of Namespace args according to the given `ini_file`.

	Parameters
	----------
	
	ini_file: str
		Filename of the ini file to load. It must readable by :class:`configparser.ConfigParser`
			
	args: argparse.Namespace
		A parser namespace object to be updated
			
	parser: argparse.ArgumentParser
		A parser object (compatibile with the given namespace)
	
	Returns
	-------
		args: argparse.Namespace
			Updated parser namespace object
	"""
	if not os.path.exists(ini_file):
		raise FileNotFoundError("The given ini file '{}' doesn't exist".format(ini_file))
		
		#reading the ini file
	config = configparser.ConfigParser()
	config.read(ini_file)
	assert len(config.sections()) ==1, "The ini file must have only one section"
	
		#casting to a dict and adding name entry
		#in principle this is not required, but makes things more handy
	ini_info = dict(config[config.sections()[0]])
	ini_info['run-name'] = config.sections()[0]

		#formatting the ini-file args
	args_to_read = []
	for k, v in ini_info.items():
		if v.lower() != 'false': #if it's a bool var, it is always store action
				#transforming 'pi' into an actual number and evaluating the expression
			if (k.find('range')>=0 or k.find('fixed-sky-loc-polarization')>=0) and v.find('pi')>=0:
				v = v.replace('pi', 'np.pi')
				v = ' '.join([str(eval(v_)) for v_ in v.split(' ')])
			args_to_read.extend('--{} {}'.format(k,v).split(' '))

	#args, _ = parser.parse_known_args(args_to_read, namespace = args) #this will update the existing namespace with the new values...
	
		#adding the new args to the namespace (if the values are not the default)
	new_data, _ = parser.parse_known_args(args_to_read, namespace = None)
	for k, v in vars(new_data).items():
		# set arguments in the args if they haven’t been set yet (i.e. they are not their default value)
		if getattr(args, k, None) == parser.get_default(k):
			setattr(args, k, v)
	return args

def add_metric_options(parser):
	"""
	Adds to the parser a set of arguments defining the metric
	
	Parameters
	----------
		
	parser: argparse.ArgumentParser
		The parser object to be updated
	"""
	parser.add_argument(
		"--variable-format", required = False,
		help="Choose which variables to include in the bank. Valid formats are those of `mbank.handlers.variable_format`")
	parser.add_argument(
		"--psd",  required = False,
		help="The input file for the PSD: it can be either a txt either a ligolw xml file")
	parser.add_argument(
		"--asd",  default = False, action='store_true',
		help="Whether the input file has an ASD (sqrt of the PSD)")
	parser.add_argument(
		"--ifo", default = 'L1', type=str, choices = ['L1', 'H1', 'V1'],
		help="Interferometer name: it can be L1, H1, V1. This is a field for the xml files for the PSD and the bank")
	parser.add_argument(
		"--f-min",  default = 10., type=float,
		help="Minium frequency for the scalar product")
	parser.add_argument(
		"--f-max",  default = 1024., type=float,
		help="Maximum frequency for the scalar product")
	parser.add_argument(
		"--df",  default = None, type=float,
		help="Spacing of the frequency grid where the PSD is evaluated")
	parser.add_argument(
		"--approximant", default = 'IMRPhenomPv2',
		help="LAL approximant for the bank generation")
	parser.add_argument(
		"--metric-type", default = 'symphony', type = str, choices = ['hessian', 'parabolic_fit_hessian', 'symphony'],
		help="Method to use to compute the metric.")


def add_flow_options(parser):
	"""
	Adds to the parser a set of arguments relevant to the training of a normalizing flow
	
	Parameters
	----------
		
	parser: argparse.ArgumentParser
		The parser object to be updated
	"""
	parser.add_argument(
		"--flow-file", default = 'flow.zip', type = str,
		help="File where the normalizing flow is saved/loaded (in zip format).")
	parser.add_argument(
		"--loss-function", default = 'll_mse', type = str, choices = STD_GW_Flow(1, 1).available_losses,
		help="Method to use to compute the metric.")
	parser.add_argument(
		"--n-layers", default = 2, type = int,
		help="Number of layers for the flow model to train")
	parser.add_argument(
		"--hidden-features", default = 4, type = int,
		help="Number of hidden features for the masked autoregressive flow to train.")
	parser.add_argument(
		"--n-epochs", default = 10000, type = int,
		help="Number of training epochs for the flow")
	parser.add_argument(
		"--learning-rate", default = 1e-3, type = float,
		help="Learning rate for the flow")
	parser.add_argument(
		"--train-fraction", default = 0.85, type = float,
		help="Fraction of the dataset to use for training")
	parser.add_argument(
		"--batch-size", default = 10000, type = int,
		help="Batch size for the training")
	parser.add_argument(
		"--min-delta", default = 0., type = float,
		help="Parameter for early stopping. If the validation loss doesn't decrease of more than min-delta for --patience times, the training will stop")
	parser.add_argument(
		"--patience", default = 5, type = int,
		help="Patience for early stopping. If the validation loss does not decrease enough for patience times, the training will stop")
	parser.add_argument(
		"--load-flow", default = False, action = 'store_true',
		help="Whether to load the flow from file (useful for plotting purposes)")

def add_general_options(parser):
	"""
	Adds to the parser some general options
	
	Parameters
	----------
		
	parser: argparse.ArgumentParser
		The parser object to be updated
	"""
	parser.add_argument(
		"--plot", action='store_true',
		help="Whether to make some plots. They will be store in run-dir")
	parser.add_argument(
		"--show", action='store_true',
		help="Whether to show the plots.")
	parser.add_argument(
		"--verbose", default = False, action = 'store_true',
		help="Whether to be verbose")
	parser.add_argument(
		"--run-dir", default = None,
		help="Output directory in which the outputs will be saved. If default is used, the run name will be appended.")
	parser.add_argument(
		"--run-name", default = 'flow_training',
		help="Name for the current run: it will set the name of some outputs")

def add_range_options(parser):
	"""
	Adds to the parser a set of arguments describing ranges on the physical quantities of a BBH
	
	Parameters
	----------
		
	parser: argparse.ArgumentParser
		The parser object to be updated
	"""
			#ranges for physical parameters
	parser.add_argument(
		"--m1-range", default = None, type=float, nargs = 2,
		help="Range values for the mass 1 (in solar masses)")
	parser.add_argument(
		"--m2-range", default = None, type=float, nargs = 2,
		help="Range values for the mass 2 (in solar masses)")
	parser.add_argument(
		"--mtot-range", default = None, type=float, nargs = 2,
		help="Range values for the total masses (in solar masses).")
	parser.add_argument(
		"--q-range", default = None, type=float, nargs = 2,
		help="Range values for the mass ratio.")
	parser.add_argument(
		"--mc-range", default = None, type=float, nargs = 2,
		help="Range values for the total masses (in solar masses).")
	parser.add_argument(
		"--eta-range", default = None, type=float, nargs = 2,
		help="Range values for the mass ratio.")
	parser.add_argument(
		"--s1-range", default = [-0.99,0.99], type=float, nargs = 2,
		help="Range values for magnitude of spin 1 (if applicable)")
	parser.add_argument(
		"--s2-range", default = [-0.99,0.99], type=float, nargs = 2,
		help="Range values for magnitude of spin 1 (if applicable)")
	parser.add_argument(
		"--chi-range", default = [-0.99,0.99], type=float, nargs = 2,
		help="Range values for effective spin parameter (if applicable)")
	parser.add_argument(
		"--theta-range", default = [-np.pi, np.pi], type=float, nargs = 2,
		help="Range values for theta angles of spins (if applicable)")
	parser.add_argument(
		"--phi-range", default = [-np.pi/2, np.pi/2], type=float, nargs = 2,
		help="Range values for phi angles of spins (if applicable)")
	parser.add_argument(
		"--e-range", default = [0., 0.5], type=float, nargs = 2,
		help="Range values for the eccentricity (if applicable)")
	parser.add_argument(
		"--meanano-range", default = [0., 1], type=float, nargs = 2,
		help="Range values for the mean anomaly (if applicable). TODO: find a nice default...")
	parser.add_argument(
		"--iota-range", default = [0., np.pi], type=float, nargs = 2,
		help="Range values for iota (if applicable)")
	parser.add_argument(
		"--ref-phase-range", default = [-np.pi, np.pi], type=float, nargs = 2,
		help="Range values for reference phase (if applicable)")

def add_tiling_generation_options(parser):
	"""
	Adds to the parser a set of arguments relevant to the generation of a tiling
	
	Parameters
	----------
		
	parser: argparse.ArgumentParser
		The parser object to be updated
	"""
	parser.add_argument(
		"--grid-size", default = None, type=int_tuple_type,
		help="Number of grid points for each dimension. The number of grid must match the number extra dimensions. If None, the grid size will be a set of ones")
	parser.add_argument(
		"--tile-tolerance", default = 0.1, type = float,
		help="Maximum tolerated variation of the relative difference of the metric determinant between parent and child in the iterative splitting procedure")
	parser.add_argument(
		"--max-depth", default = 6, type = int,
		help="Maximum number of iterative splitting before terminating.")

def add_template_placement_options(parser):
	"""
	Adds to the parser a set of arguments relevant to template placing
	
	Parameters
	----------
		
	parser: argparse.ArgumentParser
		The parser object to be updated
	"""
	parser.add_argument(
		"--mm", required = False, type = float, default = 0.97,
		help="Minimum match for the bank (a.k.a. average distance between templates)")
	parser.add_argument(
		"--placing-method", default = 'geometric', type = str, choices = cbc_bank('Mq_nonspinning').placing_methods,
		help="Which placing method to use for each tile")
	parser.add_argument(
		"--n-livepoints", default = 10000, type = int,
		help="Parameter to control the number of livepoints to use in the `random` and `pruning` placing method. For `random` (or related), it represents the number of livepoints to use for the estimation of the coverage fraction. For `pruning`, it amounts to the the ratio between the number of livepoints and the number of templates placed by ``uniform`` placing method.")
	parser.add_argument(
		"--covering-fraction", default = 0.01, type = float,
		help="Parameter to control the maximum fraction of livepoints alive before terminating the bank generation with the `random` and `pruning` placing method. The smaller the threshold, the higher the nuber of templates in the final bank.")
	parser.add_argument(
		"--empty-iterations", default = 100, type = float,
		help="Number of consecutive rejected proposal after which the `stochastic` placing method stops.")

def add_injections_options(parser):
	"""
	Adds to the parser a set of arguments relevant to injection recovery
	
	Parameters
	----------
		
	parser: argparse.ArgumentParser
		The parser object to be updated
	"""
	parser.add_argument(
		"--n-injs", type = int, default = None,
		help="Number of injections. If inj-file is specified, they will be read from it; otherwise they will be randomly drawn from the tiling and saved to file. If None, all the injections will be read from inj-file and it will throw an error if such file is not provided.")
	parser.add_argument(
		"--fixed-sky-loc-polarization", type = float, nargs = 3, default = None,
		help="Sky localization and polarization angles for the signal injections. They must be a tuple of float in the format (longitude,latitude,polarization). If None, the angles will be loaded from the injection file, if given, or uniformly drawn from the sky otherwise.")
	parser.add_argument(
		"--inj-file", type = str, default = None,
		help="An xml injection file to load the injections from. If not provided, the injections will be performed at random in each tile (injs-per-tile). If no path to the file is provided, it is understood it is located in run-dir.")
	parser.add_argument(
		"--stat-dict", type = str, default = None,
		help="The name of the file in which the results of the injection study will be saved (either json or pkl). If None, a suitable default will be provided.")
	parser.add_argument(
		"--mchirp-window", type = float, default = 0.1,
		help="The window in relative chirp mass inside which the templates are considered for full match (if --full-match is specified)")
	parser.add_argument(
		"--full-match", action='store_true', default = False,
		help="Whether to perform the full standard match computation. If False, a metric approximation to the match will be used")
	parser.add_argument(
		"--full-symphony-match", action='store_true', default = False,
		help="Whether to perform the full symphony match computation. If False, a metric approximation to the match will be used")
	parser.add_argument(
		"--seed", type = int, default = None,
		help="Random seed for extracting the random injections (if it applies)")
	parser.add_argument(
		"--use-ray", action='store_true', default = False,
		help="Whether to use ray package to parallelize the match computation")


def get_boundary_box_from_args(args):
	"""
	Given the arguments stored in a parser, it returns a rectangular boundary box for the parameter space defined by the arguments
	
	Parameters
	----------
		
	args: argparse.Namespace
		Arguments encoded in a parser Namespace object
	"""
	var_handler = variable_handler()
	format_info = var_handler.format_info[args.variable_format]
	if format_info['mass_format'] == 'm1m2':
		assert args.m1_range and args.m2_range, "If mass format is m1m2, --m1-range and --m2-range must be given"
		var1_min, var1_max = args.m1_range
		var2_min, var2_max = args.m2_range
	elif format_info['mass_format'] in ['Mq', 'logMq']:
		assert args.mtot_range and args.q_range, "If mass format is Mq, --mtot-range and --q-range must be given"
		var1_min, var1_max = args.mtot_range
		var2_min, var2_max = args.q_range
	elif format_info['mass_format'] == 'mceta':
		assert args.mc_range and args.eta_range, "If mass format is mceta, --mc-range and --eta-range must be given"
		var1_min, var1_max = args.mc_range
		var2_min, var2_max = args.eta_range

		#Ranges for quantities other than masses
	s1_min, s1_max = args.s1_range
	s2_min, s2_max = args.s2_range
	chi_min, chi_max = args.chi_range
	theta_min, theta_max = args.theta_range
	e_min, e_max = args.e_range
	meanano_min, meanano_max = args.meanano_range
	phi_min, phi_max = args.phi_range
	iota_min, iota_max = args.iota_range
	ref_phase_min, ref_phase_max = args.ref_phase_range

	return get_boundaries_from_ranges(args.variable_format,
				(var1_min, var1_max), (var2_min, var2_max),
				chi_range = (chi_min, chi_max), s1_range = (s1_min, s1_max), s2_range = (s2_min, s2_max),
				theta_range = (theta_min, theta_max), phi_range = (phi_min, phi_max),
				iota_range = (iota_min, iota_max), ref_phase_range = (ref_phase_min, ref_phase_max),
				e_range = (e_min, e_max), meanano_range = (meanano_min, meanano_max))
