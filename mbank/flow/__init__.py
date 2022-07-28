"""
mbank.flow
==========
	A module to train and use a normalizing flow model to sample from the probability distribution induced by the metric.
	The probability distribution is given by:
	
	.. math::

		p(\\theta) \propto \sqrt{\|M(\\theta)\|}
	
	A trained flow will learn how to sample form this model, enabling cool data analysis application.
	
	`IMPROVE THE DOCSTRINGS BETTER FOR THE FLOW!`
"""
from .flowmodel import TanhTransform, GW_Flow, STD_GW_Flow
