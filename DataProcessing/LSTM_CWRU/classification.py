# 主要的分类文件 有读取CWRU数据文件，也有CNN + Bi-LSTM + attention模型
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization, GRU, Input, Conv1D, pooling, Lambda, RepeatVector, \
    Permute, merge
from sklearn.model_selection import train_test_split
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, LSTM, CuDNNLSTM, CuDNNGRU, RNN, Bidirectional
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

# FEATURE_NUM = 83  # 故障类型数量

# 防止输出省略号，1000以上才省略
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
TIMESERIES_LENGTH = 100  # 时序片段大小


def data_processing(path):
    # 打开文件列表，12k频率表
    # f = open("../data/12kdata.txt", "r")  # 12k所有
    # f = open("../data/DE&level.txt", "r", encoding='UTF-8')  # 12k驱动端 共9个类型
    # f = open("../data/FE&level.txt", "r", encoding='UTF-8')  # 12k风扇端 共6个类型
    # f = open("../data/12k&DE&FE&level&15.txt", "r", encoding='UTF-8')  # 12k风扇端+驱动端 共15个类型
    # f = open("../data/21.txt", "r", encoding='UTF-8')  # 21个类型
    # f = open("../data/83.txt", "r", encoding='UTF-8')  # 83个类型
    # f = open("../data/train_data_index/end_location_5.txt", "r", encoding='UTF-8')
    f = open(path, "r", encoding='UTF-8')
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

        # ！！！！！！！！！！！！！！！！
        print('Loading data {0} {1} {2}'.format(filename, fault_type, description))  # 暂时注释

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
        features_considered = map(lambda key_str: key_str.format(data_no),
                                  ["X{0}_DE_time", "X{0}_FE_time", "X{0}_BA_time"])
        # print('feature_considered', features_considered)
        # 这句话不太好理解
        # 当前特征 flatten是变一维数组 将
        current_features = np.array([mat_data[feature].flatten() for feature in features_considered])
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
    # print("\nStatistics:")
    # 查看前五行数据。此处为DataFrame和head()函数，查看前几行，默认为5  后几行为tail()
    # print(statistics_df.head())
    # print(statistics_df)

    f.close()

    # 012分别是DE端，FE端，BA端
    features[0] = normalize(features[0])
    features[1] = normalize(features[1])
    features[2] = normalize(features[2])

    start_index = 0
    # iterrows在循环读数据时使用 index为索引，row为读取的每一行元素  row带列名，可以通过列名直接访问一整列数据
    for index, row in statistics_df.iterrows():
        # 分别赋值，得到两个分离的列表，一个错误类型，一个长度
        fault_type, length, description = row['fault_type'], row['length'], row['description']
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
        # print('错误类型:', datas[fault_type]['description'], len((datas[fault_type]['X'])))  # 输出每个类型的数据个数

    # make data balance 数据平衡
    datas = data_balance(datas)
    # datas['DE_OR_7']['X'] = choice(datas['DE_OR_7']['X'], 4870)
    # datas['DE_OR_21']['X'] = choice(datas['DE_OR_21']['X'], 4870)
    # datas['FE_OR_7']['X'] = choice(datas['FE_OR_7']['X'], 4870)

    FEATURE_NUM = len(datas)  # 故障类型数量

    label_placeholder = np.zeros(FEATURE_NUM, dtype=int)  # 返回来一个给定形状和类型的用0填充的数组
    x_data = np.empty(shape=(0, TIMESERIES_LENGTH, 3))
    y_data = np.empty(shape=(0, FEATURE_NUM), dtype=int)
    DATASET_SIZE = 0
    BATCH_SIZE = 160  # --------------------------------修改16为160
    BUFFER_SIZE = 10000

    for index, (key, value) in enumerate(datas.items()):  # 字典 items()方法以列表返回可遍历的(键, 值) 元组数组。
        sample_size = len(value['X'])
        DATASET_SIZE += sample_size
        # ！！！！！！！！！！！！！！！！！！！！！！
        print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))  # 输出每个类型的数据个数
        label = np.copy(label_placeholder)
        label[index] = 1  # one-hot encode 独热编码手动添加
        x_data = np.concatenate((x_data, value['X']))
        labels = np.repeat([label], sample_size, axis=0)  # axis=0代表列
        y_data = np.concatenate((y_data, labels))
    # 划分测试集
    # training_data = [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
    # np.random.seed(200)  # 如果不设置seed，则每次生成的随机数都会不一样
    # np.random.shuffle(training_data)

    total_data = [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
    np.random.shuffle(total_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    return x_train, x_test, y_train, y_test, FEATURE_NUM


# data balance
def data_balance(data):  # 把各个分类的数据量都变为其中最小值的大小
    value_arr = []
    for index, (key, value) in enumerate(data.items()):
        # print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))  # 输出每个类型的数据个数
        sample_size = len(value['X'])
        value_arr.append(sample_size)
    min_value = min(value_arr)
    for index, (key, value) in enumerate(data.items()):
        data[value['fault_type']]['X'] = choice(data[value['fault_type']]['X'], min_value)
    return data


# 定义函数：返回数据的min,max,mean,std
def npstatistics(data):
    return [data.min(), data.max(), data.mean(), data.std()]


# 正则化 振动信号标准化
def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


# random choice
def choice(dataset, size):
    return dataset[np.random.choice(dataset.shape[0], size, replace=False), :]
    # numpy.random.choice(a, size=None, replace=True, p=None)
    # 从a(只要是array都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
    # replace:True表示可以取相同数字，False表示不可以取相同数字
    # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。


# 在这里在输入维度方向上（即3个特征时序性数据：驱动端、风扇端、基座）添加了注意力机制，即不同重要性的维度权值不同
SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    AveragePooling = pooling.GlobalAveragePooling1D(data_format='channels_last')(a)
    den1 = Dense(input_dim, activation='relu')(AveragePooling)
    den2 = Dense(input_dim, activation='hard_sigmoid')(den1)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(den2)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge.multiply([inputs, a_probs], name='attention_mul')
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')  # 旧版本
    return output_attention_mul


# %%
# full_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
# full_dataset = full_dataset.cache().shuffle(buffer_size=BUFFER_SIZE)
# train_size = int(0.7 * DATASET_SIZE)
# test_size = int(0.3 * DATASET_SIZE)
# train_dataset = full_dataset.take(train_size).batch(BATCH_SIZE).repeat()
# test_dataset = full_dataset.skip(train_size).batch(BATCH_SIZE).repeat()

def model_build(FEATURE_NUM):  # 模型构建层
    '''
    model = Sequential()  # 序贯模型（Sequential）单输入单输出，一条路通到底，层与层之间只有相邻关系，没有跨层连接
    # model.add(LSTM(30, input_shape=(TIMESERIES_LENGTH, 3)))  # 单向，在此改双向
    model.add(Bidirectional(GRU(30, return_sequences=True, input_shape=(TIMESERIES_LENGTH, 3)), merge_mode='concat'))  # 双向GRU 第一层要指定数据输入的形状
    # model.add(Bidirectional(LSTM(30, return_sequences=True, input_shape=(TIMESERIES_LENGTH, 3)),
    #                         merge_mode='concat'))  # 双向LSTM 第一层要指定数据输入的形状，改双层此处多加一句return返回ndim=3的序列
    model.add(BatchNormalization())  # 加入正则化
    model.add(Dropout(0.2))  # 随机选取，防止过拟合
    # model.add(Bidirectional(LSTM(30)))  # 第二层双向LSTM
    model.add(Bidirectional(GRU(30)))  # 第二层双向GRU
    model.add(BatchNormalization())  # 加入正则化
    model.add(Dropout(0.2))  # 随机选取，防止过拟合
    model.add(Dense(FEATURE_NUM, activation='softmax'))  # 是Keras定义网络层的基本方法，有FEATURE_NUM个节点，激活函数是softmax
    model.add(Flatten())  # 比单向多出来一句
    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.build((None, 90, 3))  # 告诉模型输入的格式  其中90是lstm的time_step ，3是input_dim，none这里个人认为代表样本数
    model.summary()
    '''



# 完整版 CNN-Att-BiLSTM
    input = Input(shape=(TIMESERIES_LENGTH, 3))  # 1. 输入shape
    conv1 = Conv1D(filters=16, kernel_size=10, activation='relu')(input)  # 2. for input1  主要用的卷积层
    LSTM1 = Bidirectional(LSTM(30, return_sequences=True, input_shape=(TIMESERIES_LENGTH, 3)), merge_mode='concat')(conv1)  # 3. 第一层双向GRU
    NormalOut = BatchNormalization()(LSTM1)  # 4. 正则化，防止过拟合
    DropOut = Dropout(0.2)(NormalOut)  # 5. dropout，防止过拟合
    attention_mul = attention_3d_block(DropOut)  # 6. 注意力机制
    BiLSTM2 = Bidirectional(LSTM(30))(attention_mul)  # 7. 第二层双向GRU
    NormalOut = BatchNormalization()(BiLSTM2)  # 8. 正则化，防止过拟合
    DropOut = Dropout(0.2)(NormalOut)  # 9. dropout，防止过拟合
    '''

    input = Input(shape=(TIMESERIES_LENGTH, 3))  # 1. 输入shape
    # BiGRU1 = LSTM(30, input_shape=(TIMESERIES_LENGTH, 3))(input)  # 单层LSTM
    # conv1 = Conv1D(filters=48, kernel_size=6, strides=1, activation='relu')(input)  # for input1

    # conv1 = Conv1D(filters=16, kernel_size=10, activation='relu')(input)  # 2. for input1  主要用的卷积层

    # attention_mul = attention_3d_block(conv1)

    # BiLSTM1 = Bidirectional(LSTM(30, return_sequences=True, input_shape=(TIMESERIES_LENGTH, 3)), merge_mode='concat')(
    #     input)  # 3. 第一层双向LSTM
    # BiLSTM1 = Bidirectional(LSTM(30, return_sequences=True, input_shape=(TIMESERIES_LENGTH, 3)), merge_mode='concat')(
    #     conv1)  # 第一层双向LSTM

    # attention_mul = attention_3d_block(conv1)  # 6. 注意力机制
    BiLSTM1 = LSTM(30, input_shape=(TIMESERIES_LENGTH, 3))(input)  # 单层LSTM
    # BiLSTM1 = Bidirectional(LSTM(30, input_shape=(TIMESERIES_LENGTH, 3)), merge_mode='concat')(attention_mul)  # 单层双向GRU
    # BiGRU1 = LSTM(30, input_shape=(TIMESERIES_LENGTH, 3))(input)  # 单层LSTM
    NormalOut = BatchNormalization()(BiLSTM1)  # 4. 正则化，防止过拟合
    DropOut = Dropout(0.2)(NormalOut)  # 5. dropout，防止过拟合
    

    # attention_mul = attention_3d_block(DropOut)  # 6. 注意力机制

    # # BiGRU2 = Bidirectional(GRU(30))(attention_mul)  # 7. 第二层双向GRU
    # BiLSTM12 = Bidirectional(LSTM(30))(attention_mul)  # 第二层双向LSTM
    # # LSTMOut = LSTM(30, input_shape=(TIMESERIES_LENGTH, 3))(conv1)
    # NormalOut = BatchNormalization()(BiLSTM12)  # 8. 正则化，防止过拟合
    # # NormalOut = BatchNormalization()(BiGRU1)  # 正则化，防止过拟合
    # DropOut = Dropout(0.2)(NormalOut)  # 9. dropout，防止过拟合
    '''


# 分界线
    DenseOut = Dense(FEATURE_NUM, activation='softmax')(DropOut)  # 10. 是Keras定义网络层的基本方法，有FEATURE_NUM个节点，激活函数是softmax
    model = Model(inputs=input, outputs=DenseOut)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])  # 交叉熵损失函数
    model.build((None, 90, 3))
    model.summary()

    return model


def model_fit(model, x_train, y_train, file):
    checkpoint = ModelCheckpoint('./model_file/classification_model.h5', monitor='val_acc', verbose=1,
                                 save_best_only=True,  # 保存最优模型
                                 mode='max')
    callbacks_list = [checkpoint]

    # history = model.fit(X_train, Y_train,
    #                     validation_split=0.25,
    #                     # epochs=50, batch_size=16,
    #                     epochs=10, batch_size=16,
    #                     verbose=1,
    #                     callbacks=callbacks_list)

    history = model.fit(x_train, y_train,
                        validation_split=0.1,
                        epochs=71, batch_size=128,  # --------------------------------修改50为51，修改16为160
                        # epochs=50, batch_size=160,  # --------------------------------修改50为5，修改16为160
                        verbose=1,  # verbose = 1 显示进度条
                        callbacks=callbacks_list)

    # 以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
    # with open('./model_file/model_log.txtodel_log.txt', 'wb') as file_pi:
    with open('./model_file/model_result_log/' + file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # plot_accuracy()
    plot_accuracy(history)

    # plot_loss()
    plot_loss(history)


def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'verify'], loc='upper left')
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    plt.legend(['train', 'verify'], loc='upper left')
    plt.show()


# 结果可视化
def result_visualization():
    labels = ['DE_BO_07', 'DE_BO_14', 'DE_BO_21', 'DE_IR_07', 'DE_IR_14', 'DE_IR_21', 'DE_O', 'R_07', 'DE_OR_14',
              'DE_OR_21']
    tick_marks = np.array(range(len(labels))) + 0.5


def model_test(x_test, y_test):
    final_model = load_model('./model_file/classification_model.h5')
    # 测试集
    # y_true = x_data.values.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    # y_pred = y_data.argmax(axis=1)
    y_pred = final_model.predict(x_test).argmax(axis=1)
    print(y_pred)
    print(y_pred.shape)

    test_loss, test_acc = final_model.evaluate(x_test, y_test)
    print('识别准确度为：', test_acc)


def explore_filename(dir):  # 获取目录下所有文件名
    filename = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            filename.append(file)
    return filename


# 主要执行代码

# data_processing()


filepath = r"../data/train_data_index"  # 指定训练集路径
filename = explore_filename(filepath)
for file in filename:
    X_train, X_test, Y_train, Y_test, FEATURE_NUM = data_processing(filepath + '/' + file)  # 数据处理 FEATURE_NUM是故障数量
    model = model_build(FEATURE_NUM)  # 模型构建
    model_fit(model, X_train, Y_train, file)  # 模型训练
    model_test(X_test, Y_test)  # 模型测试

# X_train, X_test, Y_train, Y_test = data_processing()  # 数据处理
# model = model_build()  # 模型构建
# model_fit(model, X_train, Y_train, file)  # 模型训练
# model_test(X_test, Y_test)  # 模型测试
