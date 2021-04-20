# 分类对比实验1：将注意力机制层放置CNN通道层

# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, LSTM, CuDNNLSTM, CuDNNGRU, RNN, Activation, \
    Bidirectional, Lambda, RepeatVector, Permute, Conv1D, Multiply, multiply, merge, Concatenate, TimeDistributed, \
    MaxPooling1D, BatchNormalization, pooling
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint

import pandas as pd
import scipy.io as sio
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

TIMESERIES_LENGTH = 100

# f = open("F:/data/datas/generalization.txt", "r", encoding='UTF-8')
f = open("../data/DE&level.txt", "r", encoding='UTF-8')
lines = f.readlines()
datas = {}
fault_type = 'undefined'
description = ''
# fault_types = ['DE_B', 'DE_IR', 'DE_OR', 'FE_B', 'FE_IR', 'FE_B']
statistics_df = pd.DataFrame(
    columns=['filename', 'fault_type', 'description', 'length', 'DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min',
             'FE_max', 'FE_mean', 'FE_std', 'BA_min', 'BA_max', 'BA_mean', 'BA_std'])
features = np.empty(shape=(3, 0))


def npstatistics(data):
    return [data.min(), data.max(), data.mean(), data.std()]


for line in lines:
    line = line.strip()  # strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    if len(line) == 0 or line.startswith('#'):
        continue
    if line.startswith('faultType'):
        comments = line.split(' ')
        fault_type = comments[1]
        description = comments[2]
        continue
    filename, suffix = line.split('.')
    print('Loading data {0} {1} {2}'.format(filename, fault_type, description))

    params = filename.split('_')
    data_no = params[-1]
    # mat_data = sio.loadmat('F:/data/datas/CaseWesternReserveUniversityData/' + filename)
    mat_data = sio.loadmat('../data/CaseWesternReserveUniversityData/' + filename)
    features_considered = map(lambda key_str: key_str.format(data_no), ["X{0}_DE_time", "X{0}_FE_time", "X{0}_BA_time"])
    current_features = np.array([mat_data[feature].flatten() for feature in features_considered])
    features = np.concatenate((features, current_features), axis=1)  # axis=1代表行，0代表列
    # multidimensional_timeseries = np.hstack(current_features)
    data_size = len(mat_data["X{0}_FE_time".format(data_no)])  # current file timeseries length
    statistics = [filename, fault_type, description, data_size]
    statistics += npstatistics(current_features[0])  # 驱动端
    statistics += npstatistics(current_features[1])  # 风扇端
    statistics += npstatistics(current_features[2])  # 基座加速度数据（正常）
    statistics_df.loc[statistics_df.size] = statistics

f.close()
print("\nStatistics:")
# print(statistics_df.head())    # 打印前5行
print(statistics_df)


# 振动信号标准化
def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


features[0] = normalize(features[0])
features[1] = normalize(features[1])
features[2] = normalize(features[2])

start_index = 0
for index, row in statistics_df.iterrows():  # iterrows()对表格进行遍历
    fault_type, length = row['fault_type'], row['length']
    current_features = features[:, start_index:start_index + length]
    multidimensional_timeseries = current_features.T
    start_index += length
    data = [multidimensional_timeseries[i:i + TIMESERIES_LENGTH] for i in range(0, length - TIMESERIES_LENGTH, 100)]
    if fault_type not in datas:
        datas[fault_type] = {
            'fault_type': fault_type,
            'description': description,
            'X': np.empty(shape=(0, TIMESERIES_LENGTH, 3))
        }
    datas[fault_type]['X'] = np.concatenate((datas[fault_type]['X'], data))


# %%
# random choice
def choice(dataset, size):
    return dataset[np.random.choice(dataset.shape[0], size, replace=False), :]


# make data balance
datas['DE_OR_7']['X'] = choice(datas['DE_OR_7']['X'], 4870)
datas['DE_OR_21']['X'] = choice(datas['DE_OR_21']['X'], 4870)

label_placeholder = np.zeros(9, dtype=int)  # 返回来一个给定形状和类型的用0填充的数组
x_data = np.empty(shape=(0, TIMESERIES_LENGTH, 3))
y_data = np.empty(shape=(0, 9), dtype=int)
DATASET_SIZE = 0
BATCH_SIZE = 160  # --------------------------------修改16为160
BUFFER_SIZE = 10000  # 缓冲区大小
for index, (key, value) in enumerate(datas.items()):  # 字典 items()方法以列表返回可遍历的(键, 值) 元组数组。
    sample_size = len(value['X'])
    DATASET_SIZE += sample_size
    # print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))
    print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))
    label = np.copy(label_placeholder)
    label[index] = 1  # one-hot encode
    x_data = np.concatenate((x_data, value['X']))
    labels = np.repeat([label], sample_size, axis=0)  # axis=0代表列
    y_data = np.concatenate((y_data, labels))

training_data = [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
np.random.seed(200)
np.random.shuffle(training_data)

total_data = [(x_data[i], y_data[i]) for i in range(0, len(x_data))]
np.random.shuffle(total_data)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)


# 定义IC层（批归一化与Dropout层相结合）
def IC(inputs, p):
    x = BatchNormalization(inputs)  # replace ZCA
    x = Dropout(p)(x)  # replace Rotation
    return x


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


def attention_model():
    inputs = Input(shape=(TIMESERIES_LENGTH, 3))
    x = Conv1D(filters=16, kernel_size=10, activation='relu')(inputs)  # , padding = 'same'
    x = Dropout(0.3)(x)
    # x = IC(x, 0.2)
    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # lstm_out = Bidirectional(LSTM(30, return_sequences=True))(x)
    attention_mul = attention_3d_block(x)
    attention_fla = Flatten()(attention_mul)
    # output = Dense(3, activation='softmax')(attention_mul)
    # first_model = Model(inputs=[inputs], outputs=output)
    first_model = Model(inputs=[inputs], outputs=attention_fla)
    return first_model


# first_model = attention_model()

# first_model = Sequential()
# first_model.add(Conv1D(filters=16, kernel_size=10, activation='relu', input_shape=(TIMESERIES_LENGTH, 3)))
# first_model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid'))
# first_model.add(Bidirectional(LSTM(30, return_sequences=True)))
# first_model.add(attention_3d_block())
# first_model.add(Dropout(0.3))
# first_model.add(Flatten())
# first_model.add(Dense(6, activation='softmax'))
def model2():
    inp2 = Input(shape=(TIMESERIES_LENGTH, 3))
    x2 = Bidirectional(LSTM(30, return_sequences=True, input_shape=(TIMESERIES_LENGTH, 3)), merge_mode='concat')(inp2)
    bn = BatchNormalization()(x2)
    x3 = Dropout(0.2)(bn)
    x4 = Bidirectional(LSTM(30))(x3)
    x5 = Dropout(0.2)(x4)
    x6 = Flatten()(x5)
    # x6 = Dense(3, activation='softmax')(x5)
    model = Model(inputs=[inp2], outputs=x6)
    return model


def merge_model():
    model_1 = attention_model()
    model_2 = model2()
    # model_1.load_weights('model_1_weight.h5')#这里可以加载各自权重
    # model_2.load_weights('model_2_weight.h5')#可以是预训练好的模型权重(迁移学习)
    inp1 = model_1.input  # 参数在这里定义
    inp2 = model_2.input  # 第二个模型的参数
    r1 = model_1.output
    r2 = model_2.output
    x = Concatenate(axis=1)([r1, r2])  # 融合模型
    output = Dense(9, activation='softmax')(x)
    model = Model(inputs=[inp1, inp2], outputs=output)
    return model


model4 = merge_model()
model4.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])  # 交叉熵损失函数
model4.build((None, 90, 3))
model4.summary()

checkpoint = ModelCheckpoint('./model_file/test.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
history = model4.fit([X_train, X_train], y_train,
                     validation_split=0.1,
                     # epochs=80, batch_size=160,  # --------------------------------修改50为5，修改16为160
                     epochs=2, batch_size=160,  # --------------------------------修改50为5，修改16为160
                     verbose=1,  # verbose = 1 显示进度条
                     callbacks=callbacks_list)

with open('./model_file/model_log.txt', 'wb') as file_pi:  # 以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
    pickle.dump(history.history,
                file_pi)  # pickle.dump(obj, file, [,protocol])序列化对象，将对象obj保存到文件file中去。参数protocol是序列化模式，默认是0


def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


plot_accuracy(history)


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


plot_loss(history)

# 混淆矩阵绘制
# labels表示你不同类别的代号，比如这里的demo中有6个类别
# labels = ['DE_B', 'DE_IR', 'DE_OR', 'FE_B', 'FE_IR', 'FE_OR']
labels = ['DE_BO_07', 'DE_BO_14', 'DE_BO_21', 'DE_IR_07', 'DE_IR_14', 'DE_IR_21', 'DE_OR_07', 'DE_OR_14', 'DE_OR_21']
'''
具体解释一下re_label.txt和pr_label.txt这两个文件，比如你有100个样本
去做预测，这100个样本中一共有10类，那么首先这100个样本的真实label你一定
是知道的，一共有10个类别，用[0,9]表示，则re_label.txt文件中应该有100
个数字，第n个数字代表的是第n个样本的真实label（100个样本自然就有100个
数字）。
同理，pr_label.txt里面也应该有1--个数字，第n个数字代表的是第n个样本经过
你训练好的网络预测出来的预测label。
这样，re_label.txt和pr_label.txt这两个文件分别代表了你样本的真实label和预测label，然后读到y_true和y_pred这两个变量中计算后面的混淆矩阵。当然，不一定非要使用这种txt格式的文件读入的方式，只要你最后将你的真实
label和预测label分别保存到y_true和y_pred这两个变量中即可。
'''

final_model = load_model('./model_file/test.h5')

# y_true = x_data.values.argmax(axis=1)
y_true = y_test.argmax(axis=1)
# y_pred = y_data.argmax(axis=1)
y_pred = final_model.predict([X_test, X_test]).argmax(axis=1)
print(y_pred)
print(y_pred.shape)
tick_marks = np.array(range(len(labels))) + 0.5

test_loss, test_acc = final_model.evaluate([X_test, X_test], y_test)
print('识别准确度为：', test_acc)


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=30)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=4)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.0001 and x_val == y_val:
        plt.text(x_val, y_val, "%0.4f" % (c,), color='white', fontsize=10, va='center', ha='center')
    if c > 0.0001 and x_val != y_val:
        plt.text(x_val, y_val, "%0.4f" % (c,), color='green', fontsize=9, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig('./model_file/matrix_result.png', format='png')
plt.show()
