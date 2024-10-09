"Attempt to interpolate the metric with a GMM"
#https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
import numpy as np

from mbank import tiling_handler

tiling_file = '../runs/out_test/tiling_test.zip'

N_samples = 30000

t = tiling_handler(tiling_file)
samples = t.sample_from_tiling(N_dataset)
