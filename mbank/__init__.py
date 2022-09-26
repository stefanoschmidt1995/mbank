"""
mbank
=====
	A `package <https://github.com/stefanoschmidt1995/mbank>`_ for generating a template bank of compact binary coalescence with metric placement.
	
	Generating a template bank is a crucial step for searching a gravitational wave signal in gravitational waves (GW) interferometric data. A template bank is a collection of signals that will be compared with the data, in order to look for a match (with a procedure called matched filtering). As the noise in the data is very high, it is very hard to detect a signal without looking for it: this is why a bank should cover the space of physical parameter with a great accuracy.
	
	Each template is characterized by the parameters of the compact binary system. They usually are the two masses and the spins. In addition, for precessing or eccentric searches, one can specify two orientation angles (inclination and reference phase) and two eccentric paramters (eccentricity and mean periastron anomaly).
	Ideally the templates should cover the paramter space as evenly as possible, so that

	- The distance between templates is approximately constant
	- Each point in the space is not further than a given distance from its nearest template

	The (squared) distance between templates is called match and it is very standard in GW data analysis.
	Standard tecniques to compute such distance requires the generation of the full signal and they are costly. Here a different approach is used, replacing the distance between templates by a second order approximation, called metric approximation.
	The metric approximation is then used to place the templates in a fast and efficient way.	
"""

from . import bank, metric, handlers
from mbank.metric import cbc_metric
from mbank.bank import cbc_bank
from mbank.handlers import variable_handler, tiling_handler

	#Removing annoying TqdmWarning warnings
import warnings
from tqdm import TqdmWarning
warnings.filterwarnings('ignore', message = 'clamping frac to range', category = TqdmWarning )

