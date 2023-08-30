import numpy as np
import numpy.random as nr
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.safe_routing import naive_mc_max, calc_mean_max

seed = 232
np.random.seed(seed)

n_max = 8
mu_all = nr.rand(n_max)  # np.zeros(n)  # nr.rand(n)
A_all = A = nr.rand(n_max, n_max)
cov_all = np.dot(A_all, A_all.T)

ax1 = plt.subplot(121)
mat1 = ax1.matshow(mu_all.reshape((-1, 1)))
plt.colorbar(mat1)
ax2 = plt.subplot(122)
mat2 = ax2.matshow(cov_all)
plt.colorbar(mat2)

#plt.show()
plt.figure()

res_df = pd.DataFrame()
all_true_mean = []
n_dims = list(np.arange(3, n_max, 1))
for n_dim in n_dims:
    print(n_dim)
    mu = mu_all[:n_dim]
    cov = cov_all[:n_dim, :n_dim]
    temp_df = naive_mc_max(mu, cov, 10000, 10)
    temp_df['n_dim'] = n_dim
    res_df = pd.concat((res_df, temp_df))
    all_true_mean.append(calc_mean_max(cov, mu))

sns.lineplot(data=res_df, x='n_dim', y='max', label='Monte-Carlo')
plt.plot(n_dims, all_true_mean, label='Afonja (1972)')
print(all_true_mean)
plt.legend()
plt.show()
