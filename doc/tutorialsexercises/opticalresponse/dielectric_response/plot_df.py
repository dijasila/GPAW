import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

df_tetra = np.loadtxt('df_tetra.csv', delimiter=',')
df_point = np.loadtxt('df_point.csv', delimiter=',')

# convolve with gaussian to smooth the curve
sigma = 0.05
df2_wreal_result = gaussian_filter1d(df_point[:, 3], sigma)
df2_wimag_result = gaussian_filter1d(df_point[:, 4], sigma)

# plot
plt.figure(figsize=(6, 6))
plt.plot(df_tetra[:, 0], df_tetra[:, 3], 'blue', label='tetra Re')
plt.plot(df_tetra[:, 0], df_tetra[:, 4], 'red', label='tetra Im')
plt.plot(df_point[:, 0], df_point[:, 3], 'green', label='Re')
plt.plot(df_point[:, 0], df_point[:, 4], 'orange', label='Im')
plt.plot(df_point[:, 0], df2_wreal_result, 'pink', label='Inter Re')
plt.plot(df_point[:, 0], df2_wimag_result, 'yellow', label='Inter Im')
plt.xlabel('Frequency (eV)')
plt.ylabel('$\\varepsilon$')
plt.xlim(0, 10)
plt.ylim(-20, 20)
plt.legend()
plt.tight_layout()
plt.savefig('na_eps.png', dpi=600)