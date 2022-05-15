# 读取文件并且可视化 可以是文件 也可以是SQL数据库 并且还有emd分解

import pymysql
import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation
from pandas import np
from tensorflow.python.keras.backend import flatten
db = pymysql.connect(host="localhost", user='root', passwd="306636", port=3306, db="analog_data", charset='utf8')
cursor = db.cursor()  # 获取一个游标
# sql = "select city,need from citys"
# sql = "select Accelerometer-X-Axis-SenEvent, Accelerometer-X-Axis-OAVelocity from vibrating_sensor"
sql = "select " \
      "Accelerometer_X_Axis_SenEvent, Accelerometer_X_Axis_OAVelocity, Accelerometer_X_Axis_Peakmg," \
      "Accelerometer_X_Axis_RMSmg,Accelerometer_X_Axis_Kurtosis,Accelerometer_X_Axis_CrestFactor,Accelerometer_X_Axis_Skewness," \
      "Accelerometer_X_Axis_Deviation,Accelerometer_X_Axis_Peak_to_Peak_Displacement  from vibrating_sensor"
cursor.execute(sql)
result = cursor.fetchall()  # result为元组

# 将元组数据存进列表中
a = []
b = []
c = []
d = []
e = []
f = []
g = []
h = []
i = []
for x in result:
    a.append(x[0])
    b.append(x[1])
    c.append(x[2])
    d.append(x[3])
    e.append(x[4])
    f.append(x[5])
    g.append(x[6])
    h.append(x[7])
    i.append(x[8])

data_set = [a, b, c, d, e, f, g, h, i]
nameList = ['Accelerometer_X_Axis_SenEvent', 'Accelerometer_X_Axis_OAVelocity', 'Accelerometer_X_Axis_Peakmg',
            'Accelerometer_X_Axis_RMSmg', 'Accelerometer_X_Axis_Kurtosis', 'Accelerometer_X_Axis_CrestFactor',
            'Accelerometer_X_Axis_Skewness', 'Accelerometer_X_Axis_Deviation',
            'Accelerometer_X_Axis_Peak_to_Peak_Displacement']

colorList = ['coral', 'olive', 'lightcoral', 'dodgerblue', 'red', 'lightpink', 'palegoldenrod', 'salmon',
             'mediumslateblue']
# print(data)

point_number = 1323  # x轴大小
x = range(point_number)
# %%
fig = plt.figure()
fig.tight_layout()
# 画图时调整子图的间距
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=1.5)
# plt.subplot(2, 1, 1)
# plt.plot(x, a, color='coral')
# plt.title('Normal')
# plt.ylabel('Acceleration')
#
# plt.subplot(2, 1, 2)
# plt.plot(x, b, color='lightcoral')
# plt.title('Attention')

# 绘图
for index, data in enumerate(data_set):
    plt.subplot(len(data_set), 1, index + 1)
    plt.plot(x, data, color=colorList[index])
    plt.title(nameList[index])
plt.show()

# #  emd分解
# f = np.array(f)
# f = f.flatten()[:point_number]
# test = np.array(0)
# print(f)
# # t = np.arange(0, 1, 0.01)
# # S = 2 * np.sin(2 * np.pi * 15 * t) + 4 * np.sin(2 * np.pi * 10 * t) * np.sin(2 * np.pi * t) + np.sin(2 * np.pi * 5 * t)
# print(test)
# emd = EMD()
# IMFs = emd(f)
# imfs, res = emd.get_imfs_and_residue()
# vis = Visualisation()
# t = [i / 140 for i in range(point_number)]
# # t = np.arange(0, 1400, 140)
# vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)  # 绘制 IMF
# vis.plot_instant_freq(t, imfs=imfs)  # 绘制并显示所有提供的IMF的瞬时频率
# vis.show()

'''
# 读取指定目录文件，并进行可视化

import pymysql
import matplotlib.pyplot as plt

path = r'D:/Download/2019-08-26_080000.txt'
f = open(path, 'r')
f = f.read()
item = f.strip().split(",")
data = []
data.append(item)
print(len(data[0]))

point_number = len(data[0])  # x轴大小
x = range(point_number)
plt.subplot(1, 1, 1)
plt.plot(x, data[0], color='coral')
plt.title('2019-08-26_080000')

plt.show()
'''