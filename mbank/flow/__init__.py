"""
mbank.flow
==========

A module to train and use a `normalizing flow model <https://arxiv.org/abs/1912.02762>`_ to sample from the probability distribution induced by the metric.
The probability distribution is given by:
	
.. math::

	p(\\theta) \propto \sqrt{|\\text{det}M(\\theta)|}
	
A trained flow will learn how to sample from this model.

Although the module can work as a stand-alone, the model is specialized to the bank generation and it is employed extensively in by the tiling handler (:class:`mbank.handlers.tiling_handler`) to:

- interpolate the metric
- sample from the tiling

For most standard application, the user can interface with the normalizing flow through the tiling, without using :mod:`mbank.flow`

The module has two submodules:

- :mod:`mbank.flow.flowmodel`: defines the normalizing flow model and the functions to train it
- :mod:`mbank.flow.utils`: gathers some utitilies to plot and validate the performance of the flow

The module relies on `pytorch <https://pytorch.org/>`_ for a general Machine Learning library and on `glasflow <https://github.com/uofgravity/glasflow>`_ for the flow implementation.

The module :mod:`mbank.flow.utils` depends on `seaborn <https://seaborn.pydata.org/>`_, `pandas <https://pandas.pydata.org/>`_ and `imageio <https://imageio.readthedocs.io/en/stable/>`_. To keep the distribution light, they are not among the dependencies of the package. If you want to use the full functionalities, you will need to install them manually.

.. code-block:: Bash

	pip install pandas, seaborn, imageio

Future realease of the code may remove the dependencies on this pacakges.

"""
from .flowmodel import TanhTransform, GW_Flow, STD_GW_Flow
