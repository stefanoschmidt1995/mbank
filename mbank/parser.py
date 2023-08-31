"""
mbank.parser
============

Gathers group of options useful for the many executables that make mbank handy
"""

import configparser
import numpy as np
import os

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
