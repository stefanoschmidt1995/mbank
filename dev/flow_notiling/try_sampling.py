import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm

target_pdf = norm(loc = 1, scale = 10)
x_target = np.linspace(-3*target_pdf.std() + target_pdf.mean(), 3*target_pdf.std() + target_pdf.mean(), 1000)

sample_pdf = norm(loc = 20.2, scale = 30)
x_sample = np.linspace(-3*sample_pdf.std() + sample_pdf.mean(), 3*sample_pdf.std() + sample_pdf.mean(), 1000)

if False:
	samples = sample_pdf.rvs(1000000)
	samples_old = sample_pdf.rvs(1000000)

	#weights = target_pdf.pdf(samples)/target_pdf.pdf(samples_old) * sample_pdf.pdf(samples_old)/sample_pdf.pdf(samples)
	weights = target_pdf.pdf(samples)/sample_pdf.pdf(samples)
	weights = np.minimum(weights, 1)
	ids_ = np.random.uniform(0,1, weights.shape)<=weights

	samples = samples[ids_]
else:
	n_walkers = 10000
	chain = sample_pdf.rvs(n_walkers)
	chain_list = []
	for i in tqdm(range(10000)):
		samples_new = sample_pdf.rvs(n_walkers)

		weights = target_pdf.pdf(samples_new)/target_pdf.pdf(chain) * sample_pdf.pdf(chain)/sample_pdf.pdf(samples_new)
		weights = np.minimum(weights, 1)
		ids_, = np.where(np.random.uniform(0,1, weights.shape)<=weights)
		
		chain[ids_] = samples_new[ids_]
		
		chain_list.append(np.array(chain))
		
	samples = np.array(chain_list[::10])
	samples = np.unique(samples)
	print(samples.shape)

plt.figure()
#plt.hist(samples, bins = 100, label = 'samples from sampling pdf', density = True, alpha = 0.5)
#plt.hist(samples[ids_], bins = 100, label = 'samples', density = True, histtype = 'step')
plt.hist(samples, bins = 100, label = 'samples', density = True, histtype = 'step')
plt.plot(x_target, target_pdf.pdf(x_target), label = 'target pdf')
plt.plot(x_sample, sample_pdf.pdf(x_sample), label = 'sample pdf')
plt.legend()
plt.show()
