import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
import itertools
import collections
import torch

import lal 
import lalsimulation as lalsim

from tqdm import tqdm

import ray

import mbank
from mbank.flow.flowmodel import STD_GW_Flow
from mbank.flow.utils import compare_probability_distribution

from torch import optim

import os
import subprocess

import scipy.spatial

from line_profiler import LineProfiler

def do_profile(follow=[]):
	def inner(func):
		def profiled_func(*args, **kwargs):
			try:
				profiler = LineProfiler()
				profiler.add_function(func)
				for f in follow:
					profiler.add_function(f)
				profiler.enable_by_count()
				return func(*args, **kwargs)
			finally:
				profiler.print_stats()
		return profiled_func
	return inner

#pip install line_profiler
#add decorator @do_profile(follow=[]) before any function you need to track

###########################################
class new_tiling():
	def __init__(self, tiling_file, flow):
		tiling_data = np.load(tiling_file)
		self.centers = (tiling_data[:,0,:] + tiling_data[:,1,:])/2.
		self.maxes = tiling_data[:,1,:]
		self.mins = tiling_data[:,0,:]
		self.metric = tiling_data[:,2:,:]
		self.flow = flow
		self.flow_centers = self.flow.log_prob(self.centers.astype(np.float32)).detach().numpy()
		self.tree = scipy.spatial.KDTree(self.centers)
	
	def get_metric(self, p):
		pdf_vals = self.flow.log_prob(p.astype(np.float32)).detach().numpy()
		_, id_ = self.tree.query(p)
		#id_, = np.where(np.prod(np.logical_and(p>self.mins, p<self.maxes), axis = 1))
		#print(id_, self.flow_centers.shape, self.metric.shape, p.shape)
		#m = np.einsum('ijk,i->ijk', self.metric[id_], np.exp((2/p.shape[1])*(pdf_vals-self.flow_centers[id_])))
		m = self.metric[id_] * np.exp((2/p.shape[-1])*(pdf_vals-self.flow_centers[id_]))
		return m
###########################################

#TODO: kdtree tiling!!!

@do_profile(follow=[])
def run():
	#points = t_obj.sample_from_flow(10_000)
	t_obj = mbank.tiling_handler('../../paper/plots/precessing_bank/tiling_paper_precessing.npy')
	t_obj.load_flow('../../paper/plots/precessing_bank/flow_paper_precessing.zip')
	
	t_new = new_tiling('../../paper/plots/precessing_bank/tiling_paper_precessing.npy',#None)
		t_obj.flow)

	p_tot = t_obj.sample_from_flow(100)
	for p in tqdm(p_tot):
		m = t_new.get_metric(p)
			## Usual stuff
		m_tiling = t_obj.get_metric(p, kdtree = True, flow = True)
		if not np.allclose(m, m_tiling): print("Metrics are not the same!")

#t_obj = mbank.tiling_handler('files/tiling.npy')
#t_obj.load_flow('files/flow.zip')

run()




