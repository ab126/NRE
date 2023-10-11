import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr

from src.network_connectivity import fit_mvn_to_samples

n = 8
mu = nr.rand(n)
A = nr.rand(n, n)
cov = np.dot(A, A.T)

fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
mat1 = ax1.matshow(mu.reshape((-1, 1)))
plt.colorbar(mat1)
ax1.set_xticks([], [])
ax2 = plt.subplot(1, 2, 2)
mat2 = ax2.matshow(cov)
plt.colorbar(mat2)
fig.suptitle("The Ground Truth")

n_sample = 1000
samples = nr.multivariate_normal(mu, cov, size=n_sample)

mu_est, cov_est = fit_mvn_to_samples(samples)

fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
mat1 = ax1.matshow(mu_est.reshape((-1, 1)))
plt.colorbar(mat1)
ax1.set_xticks([], [])
ax2 = plt.subplot(1, 2, 2)
mat2 = ax2.matshow(cov_est)
plt.colorbar(mat2)
fig.suptitle("Fitted Estimate")
plt.show()
