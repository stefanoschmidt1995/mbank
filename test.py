import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

def test_imports():
	import mbank
	import mbank.utils
	print("'test_imports' passed")
	return True

def test_metric(verbose = False):
	import mbank
	import mbank.utils
	psd_file = 'aligo_O3actual_H1.txt'
	if not os.path.isfile(psd_file):
		subprocess.run('wget https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt', shell = True)
	f, PSD = mbank.utils.load_PSD(psd_file, True, 'H1')
	variable_format = 'Mq_s1xz_iota'
	f_min, f_max = 15, 1024.

	metric = mbank.cbc_metric(variable_format, (f,PSD), 'IMRPhenomXP', f_min = f_min, f_max = f_max)

	theta = [[25., 4.5, 0.4075, np.pi/2., 2.4], [75., 4.5, 0.9075, 1.38656713, 0.9]]
	overlap = not True

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
	return metric_consistent and symphony_consistent

def test_variable_format():
	import mbank
	import mbank.utils
	vh = mbank.variable_handler()
	M_range = (5, 10)
	to_return = True
	for vf in vh.valid_formats:
		#vf = 'mceta_s1xz_s2z_iota'
		if vf.startswith('mceta'):q_range = (0.08, 0.25)
		elif vf.startswith('Mq'): q_range = (1,10)
		else: continue
		
		boundaries = mbank.utils.get_boundaries_from_ranges(vf, M_range, q_range)
		theta = np.random.uniform(*boundaries, (10000, boundaries.shape[1]) )
		BBH_comps = vh.get_BBH_components(theta, vf)
		theta_rec = vh.get_theta(BBH_comps, vf)
		BBH_comps_rec = vh.get_BBH_components(theta_rec, vf)
		#print(vf, np.allclose(theta, theta_rec), np.allclose(BBH_comps, BBH_comps_rec))

		assert not np.any(np.isnan(theta_rec)), "Some weird nan appeared"
		to_return = to_return and np.allclose(theta, theta_rec) and np.allclose(BBH_comps, BBH_comps_rec)
		#print(vf, np.allclose(theta, theta_rec), np.allclose(BBH_comps, BBH_comps_rec))
		#print(theta[0])
		#print(theta_rec[0])
	if to_return:
		print("'test_variable_format' passed")
	else:
		print("'test_variable_format' failed!")
	return to_return

def test_bank_conversion():
	import mbank
	import mbank.utils
	if not os.path.isdir('tmp_test'): os.mkdir('tmp_test')
	vh = mbank.variable_handler()
	M_range = (5, 10)
	to_return = True
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
	return True

if __name__ == '__main__':
	import mbank
	import mbank.utils
	vh = mbank.variable_handler()
	test_imports()
	test_metric(True)
	test_variable_format()
	test_bank_conversion()