'''
File Created: Tuesday, 10th December 2019 3:07:29 pm
Author: zhsh
'''
# %%
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# from PyEMD import EMD, Visualisation
import pandas


# 获取文件
def get_data(filename):
    filename = '../../data/CaseWesternReserveUniversityData/{}.mat'.format(filename)
    data = sio.loadmat(filename)
    return data


# 获取驱动端数据
def get_driver_end_data(filename, point_number=10000):
    data = get_data(filename)
    file_number = filename.split('_')[-1]  # 按照_切分，第四个数据就是文件名
    key = 'X{}_DE_time'.format(file_number.zfill(3))
    print("key:", key)
    print("key_process:", data[key].flatten()[:point_number])
    return data[key].flatten()[:point_number]  # 变成一维数组


point_number = 500

filenames = ['normal_0_97', 'normal_1_98', 'normal_2_99', 'normal_3_100']
driver_end_data = map(lambda filename: get_driver_end_data(filename, point_number), filenames)

x = range(point_number)
print("x:", x)
plt.figure()
for index, data in enumerate(driver_end_data):
    plt.subplot(4, 1, index + 1)
    plt.plot(x, data)
    # plt.title(filenames[index])
    # print("1")
    # plt.boxplot(data)
    # plt.title(filenames[index])
    # print("2")

plt.show()

# %%
