'''
File Created: Saturday, 12th October 2019 2:54:46 pm
Author: zhsh
'''
# %%
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation

import pandas

normal_data = sio.loadmat('../data/CaseWesternReserveUniversityData/normal_0_97.mat')
print(normal_data)
abnormal_data = sio.loadmat('../data/CaseWesternReserveUniversityData/48k_Drive_End_B007_0_122.mat')
X097_DE_time = normal_data['X097_DE_time']
X097_FE_time = normal_data['X097_FE_time']
print('abnormal', X097_FE_time)

X122_FE_time = abnormal_data['X122_FE_time']

point_number = 1000  # 数据点数量

x = np.arange(point_number)
normal_y0 = X097_FE_time.flatten()[:point_number]
normal_y1 = X097_DE_time.flatten()[:point_number]
abnormal_y0 = X122_FE_time.flatten()[:point_number]

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(x, normal_y0)
plt.subplot(3, 1, 2)
plt.plot(x, normal_y1)
plt.subplot(3, 1, 3)
plt.plot(x, abnormal_y0)
plt.show()

# %%
emd = EMD()
IMFs = emd(abnormal_y0)
imfs, res = emd.get_imfs_and_residue()
vis = Visualisation()
t = [i / 12000 for i in range(point_number)]
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
vis.plot_instant_freq(t, imfs=imfs)
vis.show()

# %%
