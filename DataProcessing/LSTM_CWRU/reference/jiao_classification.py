# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, LSTM, CuDNNLSTM, CuDNNGRU, RNN
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint

import os
import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import pickle

# 防止输出省略号，1000以上才省略
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
TIMESERIES_LENGTH = 100
# 打开文件列表，12k频率表
f = open("../data/12kdata.txt", "r")
# f = open("../data/DE&level.txt", "r", encoding='UTF-8')
# 读每一行
lines = f.readlines()
# data列表
datas = {}
# 错误类型初始化为未定义
fault_type = 'undefined'
# 描述为空
description = ''
# 错误类型：驱动——滚动体，驱动——内圈，驱动——外圈，风扇——滚动体，风扇——内圈，风扇-外圈
# fault_types = ['DE_B', 'DE_IR', 'DE_OR', 'FE_B', 'FE_IR', 'FE_OR']
# 统计数据 DataFrame是Pandas库中的一种数据结构，类似excel，是一种二维表
# 其中第一个参数是存放在DataFrame里的数据，第二个参数index就是之前说的行名，第三个参数columns是之前说的列名
# 教学：https://blog.csdn.net/tefuirnever/article/details/93708964
# min：最小值 max：最大值 mean：均值 std：方差 var：均方差
# 每一份数据都是五个混合，五个通道，前三个分别是驱动、风扇、基座震动数据，第四个为时序，第五个为转速
statistics_df = pd.DataFrame(
    columns=['filename', 'fault_type', 'description', 'length', 'DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min',
             'FE_max', 'FE_mean', 'FE_std', 'BA_min', 'BA_max', 'BA_mean', 'BA_std'])
# 生成一个3*0的数组
features = np.empty(shape=(3, 0))


# 定义函数：返回数据的min,max,mean,std
def npstatistics(data):
    return [data.min(), data.max(), data.mean(), data.std()]


# 之前 lines = f.readlines()
for line in lines:
    # 消除首尾空格、换行
    line = line.strip()
    # 0行或者开头为#，就continue继续循环，即跳过带#的文件
    if len(line) == 0 or line.startswith('#'):
        continue
    # 录入表格列名
    if line.startswith('faultType'):
        # 将line按照空格分隔存到comments中，有多个列表
        comments = line.split(' ')
        # comments[0]是 faultTpe 这几个字
        fault_type = comments[1]
        description = comments[2]
        continue
    # 通过split将读入的行通过.分隔成文件名和后缀名mat，分别存在filename和suffix中
    filename, suffix = line.split('.')
    # 输出loading data 同时format是格式化函数，能将括号中的三个filename等代替前方的 {0} 等
    # print('Loading data {0} {1} {2}'.format(filename, fault_type, description))
    # 此时的filename长这样 12k_Drive_End_B007_0_118  以_进行分割放入params列表（或者叫数组）
    params = filename.split('_')
    # [-1] 数组倒数最后一位 118 119这些编号
    data_no = params[-1]
    # scipy包中的io，loadmat是载入数据，data中的西储大学，并加上文件名
    mat_data = sio.loadmat('../data/CaseWesternReserveUniversityData/' + filename)
    # https://blog.csdn.net/gaozhanfire/article/details/95664379
    # lambda 是隐含函数 x:x+2（x为参数，：右边为函数） map 是遍历处理函数，接受一个f和一个list，对list中的所有元素做f函数。再返回新list
    # lambda和map结合，即可省略定义新函数，直接在map时编写函数
    # 考虑的特征 此处是将后面列表中的{0}，批量更换为data_no 只考虑以下三个特征
    features_considered = map(lambda key_str: key_str.format(data_no), ["X{0}_DE_time", "X{0}_FE_time", "X{0}_BA_time"])
    # print('feature_considered', features_considered)
    # 这句话不太好理解
    # 当前特征 flatten是变一维数组 将
    current_features = np.array([mat_data[feature].flatten() for feature in features_considered])
    # print('current_features', current_features)
    # features是上文定义的（3,0）数组 np.concatenate是数组拼接 axis=1是表示行，即所有待拼接的数组第一行与第一行拼接，第二行与第二行拼接
    # 如果axis=0则表示列，即把第二个数组直接拼在第一个后面，后面依次相接
    # 目前features有三列，分别是
    features = np.concatenate((features, current_features), axis=1)
    # print('feature', features)
    # multidimensional_timeseries = np.hstack(current_features) # 此处不知为何注释掉了，下文中有再次使用
    data_size = len(mat_data["X{0}_DE_time".format(data_no)])  # current file timeseries length
    # 数据列表定义为 文件名，错误类型，描述，数据大小
    statistics = [filename, fault_type, description, data_size]
    statistics += npstatistics(current_features[0])  # 加上DE端的四个属性 min max等
    statistics += npstatistics(current_features[1])  # 加上FE端的四个属性
    statistics += npstatistics(current_features[2])  # 加上BA端的四个属性
    # statistics_df是上面定义的DataFrame数组
    # DataFrame.loc[]获取指定行列的元素  .size获取共多少行
    statistics_df.loc[statistics_df.size] = statistics

f.close()
# print("\nStatistics:")
# 查看前五行数据。此处为DataFrame和head()函数，查看前几行，默认为5  后几行为tail()
# print(statistics_df.head())
print(statistics_df)


# 正则化
def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


# 012分别是DE端，FE端，BA端
features[0] = normalize(features[0])
features[1] = normalize(features[1])
features[2] = normalize(features[2])

start_index = 0
# iterrows在循环读数据时使用 index为索引，row为读取的每一行元素  row带列名，可以通过列名直接访问一整列数据
for index, row in statistics_df.iterrows():
    # 分别赋值，得到两个分离的列表，一个错误类型，一个长度
    fault_type, length = row['fault_type'], row['length']
    # a[:, x:y] 表示取所有数据的第x列到第y列，含左不含有  current_features表示中间过渡用的变量
    current_features = features[:, start_index:start_index + length]
    # 多维时间序列 .T是取转置
    multidimensional_timeseries = current_features.T
    start_index += length
    # TIMESERIES_LENGTH 是定义的时序长度，为100 划分样本为100长度一个的小数据
    data = [multidimensional_timeseries[i:i + TIMESERIES_LENGTH] for i in range(0, length - TIMESERIES_LENGTH, 100)]
    if fault_type not in datas:
        datas[fault_type] = {
            'fault_type': fault_type,
            'description': description,
            'X': np.empty(shape=(0, TIMESERIES_LENGTH, 3))
        }
    datas[fault_type]['X'] = np.concatenate((datas[fault_type]['X'], data))
# print(datas)

# numpy.random.choice(a, size=None, replace=True, p=None)
# 从a(只要是array都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
# replace:True表示可以取相同数字，False表示不可以取相同数字
# 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

# random choice
def choice(dataset, size):
    return dataset[np.random.choice(dataset.shape[0], size, replace=False), :]


# make data balance
datas['DE_OR']['X'] = choice(datas['DE_OR']['X'], 14500)
datas['FE_OR']['X'] = choice(datas['FE_OR']['X'], 14500)


label_placeholder = np.zeros(6, dtype=int)
x_data = np.empty(shape=(0, TIMESERIES_LENGTH, 3))
y_data = np.empty(shape=(0, 6), dtype=int)
DATASET_SIZE = 0
BATCH_SIZE = 16
BUFFER_SIZE = 10000
for index, (key, value) in enumerate(datas.items()):
    sample_size = len(value['X'])
    DATASET_SIZE += sample_size
    print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))
    label = np.copy(label_placeholder)
    label[index] = 1  # one-hot encode 独热编码手动添加
    x_data = np.concatenate((x_data, value['X']))
    labels = np.repeat([label], sample_size, axis=0)
    y_data = np.concatenate((y_data, labels))

training_data = [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
np.random.shuffle(training_data)


# %%
# full_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
# full_dataset = full_dataset.cache().shuffle(buffer_size=BUFFER_SIZE)
# train_size = int(0.7 * DATASET_SIZE)
# test_size = int(0.3 * DATASET_SIZE)
# train_dataset = full_dataset.take(train_size).batch(BATCH_SIZE).repeat()
# test_dataset = full_dataset.skip(train_size).batch(BATCH_SIZE).repeat()

model = Sequential()
# model.add(LSTM(30, input_shape=(TIMESERIES_LENGTH, 3))) 单向，在此改双向
model.add(LSTM(30, input_shape=(TIMESERIES_LENGTH, 3)))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint('classification1_10.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(x_data, y_data,
                    validation_split=0.25,
                    # epochs=50, batch_size=16,
                    epochs=10, batch_size=16,
                    verbose=1,
                    callbacks=callbacks_list)

with open('./classificationTrainHistoryDict1_10', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# plot_accuracy()
plot_accuracy(history)


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    plt.legend(['Train', '验证'], loc='upper left')
    plt.show()


# plot_loss()

plot_loss(history)
