import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import subprocess

def test_imports():
	import mbank
	import mbank.utils
	import mbank.flow
	print("'test_imports' passed")

def test_psd(verbose = False):
	import mbank.utils
	psd_file = 'aligo_O3actual_H1.txt'
	if not os.path.isfile(psd_file):
		subprocess.run('wget https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt', shell = True)
	for df in [0.1, 1, 4, 10]:
		f, PSD = mbank.utils.load_PSD(psd_file, True, 'H1', df = df)
		assert np.allclose(f[1]-f[0], df)
	print("'test_psd' passed")
	
def test_metric(verbose = False):
	import mbank
	import mbank.utils
	psd_file = 'aligo_O3actual_H1.txt'
	if not os.path.isfile(psd_file):
		subprocess.run('wget https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt', shell = True)
	f, PSD = mbank.utils.load_PSD(psd_file, True, 'H1', df = 4)
	variable_format = 'Mq_s1xz_iota'
	f_min, f_max = 15, 1024.

	metric = mbank.cbc_metric(variable_format, (f,PSD), 'IMRPhenomXP', f_min = f_min, f_max = f_max)

	theta = [[25., 4.5, 0.4075, np.pi/2., 2.4], [75., 4.5, 0.9075, 1.38656713, 0.9]]
	overlap = not True

	hpc_consistent = (metric.get_hpc([10, 3, 0.3, 0., 0.]) == 0.)
	if not hpc_consistent:
		print("h_pc test failed!")
	hpc = metric.get_hpc(theta)
	if verbose:
		print('hpc = ', hpc)

	metric_std = metric.get_hessian(theta, overlap = overlap)
	metric_symphony = metric.get_hessian_symphony(theta, overlap = overlap, order = None, epsilon = 1e-5)
	metric_std_numerical = metric.get_numerical_hessian(theta, overlap = overlap, symphony = False, antenna_patterns = None)
	metric_symphony_numerical = metric.get_numerical_hessian(theta, overlap = overlap, symphony = True, antenna_patterns = None)
	#metric_std_Fc = metric.get_numerical_hessian(theta, overlap = overlap, symphony = False, antenna_patterns = (0.5,0.5))

	metric_std_parabolic =  [[0.]] #metric.get_parabolic_fit_hessian(theta, overlap = overlap, symphony = False, antenna_patterns = None)
	metric_std_Fc_parabolic =  [[0.]] #metric.get_parabolic_fit_hessian(theta, overlap = overlap, symphony = False, antenna_patterns = (0.5,0.5))
	metric_symphony_parabolic =  [[0.]] #metric.get_parabolic_fit_hessian(theta, overlap = overlap, symphony = True, antenna_patterns = (0.5,0.5))

	if verbose:
		print('##')
		print('Format: ', variable_format)
		print("Center: ", theta)
		print('f range: [{}, {}]'.format(f_min, f_max))
		print("overlap: ", overlap)
		print('##')
		print("Metric std:\t\t", np.linalg.det(metric_std))
		print("Metric std numerical:\t", np.linalg.det(metric_std_numerical))
		#print("Metric std F_x = 1:\t", np.linalg.det(metric_std_Fc))
		print("Metric symphony:\t", np.linalg.det(metric_symphony))
		print("Metric symphony numerical:\t", np.linalg.det(metric_symphony_numerical))
		#print('##')
		#print("Metric std parabolic:\t", np.linalg.det(metric_std_parabolic))
		#print("Metric std F_x = 1 parabolic:\t", np.linalg.det(metric_std_Fc_parabolic))
		#print("Metric symphony parabolic:\t", np.linalg.det(metric_symphony_parabolic))

		print('##')
		print("eigs std", *np.linalg.eig(metric_std)[0])
		print("eigs numerical std", *np.linalg.eig(metric_std_numerical)[0])
		print("eigs symphony", *np.linalg.eig(metric_symphony)[0])
		print("eigs numerical symphony", *np.linalg.eig(metric_symphony_numerical)[0])
	
	atol, rtol = 0., 1e-2
	metric_consistent = np.allclose(np.linalg.eig(metric_std)[0], np.linalg.eig(metric_std_numerical)[0],
		atol =atol, rtol = rtol)
	symphony_consistent = np.allclose(np.linalg.eig(metric_symphony)[0], np.linalg.eig(metric_symphony_numerical)[0],
		atol =atol, rtol = rtol)
	
	if not metric_consistent:
		print("std metric test failed!")
	if not symphony_consistent:
		print("symphony metric test failed!")
	if symphony_consistent and metric_consistent:
		print("'test_metric' passed")

def test_variable_format():
	import mbank
	import mbank.utils
	vh = mbank.variable_handler()
	
	assert np.allclose(vh.convert_theta([12, 2, 0.2], 'Mq_chi', 'm1m2_chi'), [8,4,0.2])
	
	M_range = (5, 10)
	for vf in vh.valid_formats:
		#vf = 'mceta_s1xz_s2z_iota'
		if vf.startswith('mceta'): q_range = (0.08, 0.25)
		elif vf.startswith('Mq'): q_range = (1,10)
		elif vf.startswith('m1m2') or vf.startswith('logm1logm2'): q_range = M_range
		elif vf.startswith('logMq'):
			q_range = (1,10)
			M_range = (.1, 2)
		else: continue
		
		boundaries = mbank.utils.get_boundaries_from_ranges(vf, M_range, q_range)
		theta = np.random.uniform(*boundaries, (10000, boundaries.shape[1]) )
		assert np.allclose(theta, vh.convert_theta(theta, vf, vf)), "Something wrong with theta conversion"
		BBH_comps = vh.get_BBH_components(theta, vf)
		theta_rec = vh.get_theta(BBH_comps, vf)
		BBH_comps_rec = vh.get_BBH_components(theta_rec, vf)
		#print(vf, np.allclose(theta, theta_rec), np.allclose(BBH_comps, BBH_comps_rec))

		assert not np.any(np.isnan(theta_rec)), "Some weird nan appeared"
		assert np.allclose(theta, theta_rec) and np.allclose(BBH_comps, BBH_comps_rec), "Some problem with reconstruction"
		#print(vf, np.allclose(theta, theta_rec), np.allclose(BBH_comps, BBH_comps_rec))
		#print(theta[0])
		#print(theta_rec[0])
	print("'test_variable_format' passed")

def test_bank_conversion():
	import mbank
	import mbank.utils
	
		#Checking properties stored
	b = mbank.cbc_bank('Mq_s1xz_iota')
	assert b.M is None
	b.add_templates([[10, 2, 0.9, 3, 1], [10, 20, 0.19, 3, 0], [100, 20, 0.1, 1.3, 2]])
	assert np.allclose(b.M, [10, 10, 100])
	assert np.allclose(b.s1, [0.9, 0.19, 0.1])
	b.add_templates([[10, 2, 0.9, 3, 1], [10, 20, 0.19, 3, 0], [100, 20, 0.1, 1.3, 2]])
	assert np.allclose(b.s1, [0.9, 0.19, 0.1, 0.9, 0.19, 0.1])
	
	if not os.path.isdir('tmp_test'): os.mkdir('tmp_test')
	vh = mbank.variable_handler()
	M_range = (5, 10)
	for vf in vh.valid_formats:
		if np.random.choice([0,1], p = [0.2, 0.8]): continue
		if vh.format_info[vf]['e']: continue
		#vf = 'mceta_s1xz_s2z_iota'
		if vf.startswith('mceta'):q_range = (0.08, 0.25)
		elif vf.startswith('Mq'): q_range = (1,10)
		elif vf.startswith('m1m2'): q_range = M_range
		else: continue
		
		boundaries = mbank.utils.get_boundaries_from_ranges(vf, M_range, q_range)
		theta = np.random.uniform(*boundaries, (1000, boundaries.shape[1]) )
		#generating a bank
		bank = mbank.cbc_bank(vf)
		bank.add_templates(theta)
		bank.save_bank('tmp_test/bank.dat')
		bank.save_bank('tmp_test/bank.xml.gz')
		bank_dat = mbank.cbc_bank(vf, 'tmp_test/bank.dat')
		bank_xml = mbank.cbc_bank(vf, 'tmp_test/bank.xml.gz')

		assert np.allclose(bank.templates, vh.get_theta(bank.BBH_components, vf))
		assert np.allclose(bank.templates, bank_dat.templates)
		assert np.allclose(bank.templates, bank_xml.templates)
		assert np.allclose(bank_dat.templates, bank_xml.templates)
		
		#assert np.allclose(bank.templates, vh.get_theta(bank.BBH_components, vf))
	print("'test_bank_conversion' passed")

def test_flow_transformations():
	import torch
	from mbank.flow.flowmodel import tau0tau3Transform, TanhTransform
	
	udist = torch.distributions.uniform.Uniform(torch.tensor([1, 0.01, -0.99]), torch.tensor([10, 0.25, 0.99]))
	for trans in [TanhTransform(3, [1, 0.01, -0.99], [10, 0.25, 0.99]), tau0tau3Transform()]:
		inputs = udist.sample(sample_shape = [10])
		tvals, _ = trans(inputs)

		assert torch.allclose(inputs, trans.inverse(trans(inputs)[0])[0]), "Transformation {} doesn't work".format(type(trans))
		assert torch.allclose(tvals, trans(trans.inverse(tvals)[0])[0]), "Transformation {} doesn't work".format(type(trans))
	print("'test_flow_transformations' passed")

def test_flow_IO():
	from mbank.flow import STD_GW_Flow
	
	flow = STD_GW_Flow(4, n_layers = 2, hidden_features = [10,3])
	
	flow.save_weigths('.flow_test_42735628347252763182.zip')
	
	flow_loaded = STD_GW_Flow.load_flow('.flow_test_42735628347252763182.zip')
	
	for (k1,v1), (k2,v2) in zip(flow.state_dict().items(), flow_loaded.state_dict().items()):
		assert k1 == k2, "The loaded flow doesn't have the same entries in state_dict"
		assert torch.allclose(v1, v2), "The loaded flow doesn't have the same values in state_dict"
	
	os.remove('.flow_test_42735628347252763182.zip')

	print("'test_flow_IO' passed")
	
def test_reference_phase():
	import mbank
	psd_file = 'aligo_O3actual_H1.txt'
	if not os.path.isfile(psd_file):
		subprocess.run('wget https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt', shell = True)
	f, PSD = mbank.utils.load_PSD(psd_file, True, 'H1', df = .1)
	variable_format = 'Mq_s1xz_iotaphi'
	f_min, f_max = 15, 1024.

	metric = mbank.cbc_metric(variable_format, (f,PSD), 'IMRPhenomXP', f_min = f_min, f_max = f_max)
	
	WFs = metric.get_WF([[20, 3, 0.8, 1, 1.98, 0.], [20, 3, 0.8, 1, 1.98, 2.]])
	
	assert not np.allclose(1e19*WFs[0], 1e19*WFs[1]), "The metric doesn't take into account the reference phase properly"
	
	print("'test_reference_phase' passed")

def test_sampling_from_boundaries():
	import argparse
	from argparse import Namespace
	from mbank.parser import boundary_keeper
	
	args = argparse.Namespace()
	args.m1_range = (1, 100)
	args.m2_range = (1, 100)
	args.q_range = (1, 10)
	args.chi_range = (0, 1)
	
	vf = 'm1m2_chi'
	a = boundary_keeper(args)
	
	assert np.all(a([[20, 10, 0.5], [50, 2, 0.1], [50, 20, -0.1]], vf) == [True, False, False])
	
	samples = a.sample(100000, vf)
	
	#plt.scatter(*samples[:,:2].T, s = 4)
	#plt.show()
	
	vol, std_err = a.volume(100000, vf)
	true_vol = 99*99/2 - 90*9/2
	assert np.allclose(vol, true_vol, rtol = 0, atol = 3*std_err)
	
	print("'test_sampling_from_boundaries' passed")

def test_match():
	import mbank
	from mbank.utils import load_PSD, get_antenna_patterns
	import pycbc.filter
	from pycbc.types.frequencyseries import FrequencySeries
	
	approximant = 'IMRPhenomPv2'
	f_min, f_max = 15., 1024.
	
	variable_format = 'Mq_s1xz_s2z_iotaphi'
	
	f, PSD = load_PSD('aligo_O3actual_H1.txt', asd = True, ifo = 'L1')
	df = f[1]-f[0]

	f = np.linspace(0, len(PSD)*df,len(PSD))

	#mbank stuff
	vh = mbank.variable_handler()
	m_obj = mbank.cbc_metric(variable_format, (f, PSD),
		approx = approximant, f_min = f_min, f_max = f_max)


	theta2 = [21, 4.2, .6, 2., -0.5, 2, 0.]
	theta1 = [21, 4.2, .6, 2., -0.5, 3, 0.]

	psi = np.random.uniform(0, 2*np.pi)
	delta = np.arccos(np.random.uniform(-1,1))
	alpha = np.random.uniform(-np.pi, np.pi)

	theta, phi, psi_sbank = np.pi/2 - delta, alpha+np.pi/2, np.pi-psi


	F_p, F_c = get_antenna_patterns(alpha, delta, psi)

		#mbank match
	mbank_match_sym = m_obj.match(theta2, theta1, symphony = True, antenna_patterns = (F_p, F_c))
	mbank_match_std = m_obj.match(theta2, theta1, symphony = False, antenna_patterns = (F_p, F_c))

		#pycbc match
	WF1_plus = FrequencySeries(m_obj.get_WF(theta1), delta_f = df)
	WF1_cross = FrequencySeries(m_obj.get_WF(theta1, plus_cross = True)[1], delta_f = df)
	h2 = m_obj.get_WF(theta2, plus_cross = True)
	h2 = F_p*h2[0] + F_c*h2[1]
	WF2 = FrequencySeries(h2, delta_f = df)
	
	args_pycbc = {'psd': FrequencySeries(m_obj.PSD, delta_f = df),
			'low_frequency_cutoff': f_min, 'high_frequency_cutoff': f_max}

	def norm(a):
		return np.sqrt(pycbc.filter.sigmasq(a, **args_pycbc))

	pycbc_match_std, _ = pycbc.filter.match(WF2, WF1_plus, **args_pycbc)
	
	hp_ts = pycbc.filter.matched_filter(WF1_plus/norm(WF1_plus), WF2/norm(WF2), **args_pycbc)
	hc_ts = pycbc.filter.matched_filter(WF1_cross/norm(WF1_cross), WF2/norm(WF2), **args_pycbc)
	
	#hp_ts /= np.sqrt(pycbc.filter.sigmasq(hp_ts, **args_pycbc))
	#hc_ts /= np.sqrt(pycbc.filter.sigmasq(hc_ts, **args_pycbc))
	
	hpc_pycbc = 1 - pycbc.filter.match(WF1_cross, WF1_plus, **args_pycbc)[0]
	hpc_mbank = m_obj.get_hpc(theta1)
	
	pycbc_match_sym = pycbc.filter.compute_max_snr_over_sky_loc_stat_no_phase(np.array(hp_ts), np.array(hc_ts),
		hpc_pycbc, hpnorm = 1, hcnorm = 1).max()
	
	print('hpc\n\tpycbc: {}\n\tmbank: {}'.format(hpc_pycbc, hpc_mbank))
	print('Match\n\tpycbc: {}\n\tmbank: {}\n\tpycbc symphony: {}\n\tmbank symphony: {}'.format(pycbc_match_std, mbank_match_std, pycbc_match_sym, mbank_match_sym))
	
	assert np.allclose(pycbc_match_std, mbank_match_std, rtol = 0, atol = 1e-3), "STD match does not agree with pycbc"
	assert np.allclose(pycbc_match_sym, mbank_match_sym, rtol = 0, atol = 1e-3), "SYM match does not agree with pycbc"
	
	print("'test_match' passed")


if __name__ == '__main__':
	test_match()
	test_match()
	test_imports()
	test_flow_IO()
	test_psd()
	test_variable_format()
	test_flow_transformations()
	test_metric(True)
	test_bank_conversion()
	test_reference_phase()
	test_sampling_from_boundaries()
