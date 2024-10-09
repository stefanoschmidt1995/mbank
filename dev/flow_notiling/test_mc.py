import numpy as np

import scipy.stats
import matplotlib.pyplot as plt


def var(samples, mean, p = None):
	if p is None:
		return np.mean(np.square(samples-mean))
	else:
		return 	np.sum(np.square(samples-mean)*p)/np.sum(p)


dist = scipy.stats.norm(loc = 1., scale = 10)

dist_2 = scipy.stats.norm(loc = 4., scale = 20)

samples = dist.rvs(10000)
samples_2 = dist_2.rvs(10000)
x = np.linspace(-50, 50,1000)

print('true var', 10**2)
print('MC var ', var(samples, 1))
print('pdf var ', var(x, 1, dist.pdf(x)))

print('pdf var ', var(samples_2, 1, dist.pdf(samples_2)/dist_2.pdf(samples_2)))

plt.hist(samples, bins = 100, density = True, histtype = 'step')
plt.hist(samples_2, bins = 100, density = True, histtype = 'step')
plt.plot(x, dist.pdf(x))
plt.yscale('log')
plt.show()
