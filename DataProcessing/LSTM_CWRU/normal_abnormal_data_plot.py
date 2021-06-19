'''
File Created: Tuesday, 10th December 2019 5:37:37 pm
Author: zhsh
'''
# %%
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation
import pandas
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


filenames_12k = [
    'normal_0_97',
    '12k_Drive_End_B007_0_118',
    '12k_Drive_End_IR007_0_105',
    '12k_Drive_End_OR007@3_0_144'
]

filenames_48k = [
    'normal_0_97',
    '48k_Drive_End_B007_0_122',
    '48k_Drive_End_IR007_0_109',
    '48k_Drive_End_OR007@3_0_148'
]


def get_data(filename):
    filename = '../data/CaseWesternReserveUniversityData/{}.mat'.format(filename)
    data = sio.loadmat(filename)
    return data


def get_driver_end_data(filename, point_number=10000):
    data = get_data(filename)
    file_number = filename.split('_')[-1]
    key = 'X{}_DE_time'.format(file_number.zfill(3))
    return data[key].flatten()[:point_number]


point_number = 2000

driver_end_data = list(map(lambda filename: get_driver_end_data(filename, point_number), filenames_48k))

x = range(point_number)
# %%

fault_list = ['正常', '滚动体故障', '内圈故障', '外圈故障']
# fault_list = ['Normal', 'Ball Fault', 'Inner Raceway Fault', 'Out Raceway Fault']
color_list = ['dodgerblue', 'red', 'lightcoral', 'mediumslateblue']

fig = plt.figure()
fig.tight_layout()
# fig.set_size_inches(6, 8)

for index, data in enumerate(driver_end_data):
    plt.subplot(len(driver_end_data), 1, index + 1)
    plt.plot(x, data, color=color_list[index])
    # plt.title('Normal')
    plt.title(fault_list[index])
'''
plt.subplot(4, 1, 1)
plt.plot(x, driver_end_data[0], color='palegoldenrod')
# plt.title('Normal')
plt.title('正常')
plt.ylabel('Acceleration')

plt.subplot(4, 1, 2)
plt.plot(x, driver_end_data[1], color='palegoldenrod')
# plt.title('Ball Fault')
plt.title('滚动体故障')

plt.subplot(4, 1, 3)
plt.plot(x, driver_end_data[2], color='palegoldenrod')
# plt.title('Inner Raceway Fault')
plt.title('内圈故障')

plt.subplot(4, 1, 4)
plt.plot(x, driver_end_data[3], color='palegoldenrod')
# plt.title(u'Outer Raceway Fault')
plt.title(u'外圈故障')
'''
savePath = r'./model_file/result_picture/normal_abnormal_data_plot.png'
# plt.ylabel('Acceleration')
# plt.xlabel('Time Step')
plt.ylabel('加速度')
plt.xlabel('时间步长')

plt.tight_layout()  # 调整画布大小防止子图重叠
# plt.figure(figsize=(16, 12))  # 调整画布大小防止子图重叠
plt.savefig(savePath, bbox_inches='tight')  # 后面那句话是让保存的图片变得紧凑
plt.show()

# %%
