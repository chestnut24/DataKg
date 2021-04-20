from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras

#from tensorflow.python.keras.models import Sequential   #TensorFlow1.13.0
from tensorflow.keras.models import Sequential
#from keras import models
#from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, CuDNNLSTM, CuDNNGRU, RNN
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, RNN, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
import time#需导入包
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn import manifold, datasets
from sklearn.manifold import TSNE

import os
import gc
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import scipy.io as sio
import pickle

TIMESERIES_LENGTH = 100

#f = open("12kdata.txt","r")
f = open("E:/Kerasliuyankaicode/sample/12kdata.txt","r",encoding='UTF-8')
#f = open("E:/Kerasliuyankaicode/sample/12kdata.txt","r").read().decode(‘gb18030’,’ignore’)
lines = f.readlines()
datas = {}
fault_type = 'undefined'
description = ''
fault_typeList = []
descriptionList = []
# fault_types = ['DE_B', 'DE_IR', 'DE_OR', 'FE_B', 'FE_IR', 'FE_B']
statistics_df = pd.DataFrame(columns=['filename','fault_type', 'description','length','DE_min','DE_max','DE_mean','DE_std','FE_min','FE_max','FE_mean','FE_std','BA_min','BA_max','BA_mean','BA_std'])
features = np.empty(shape=(3,0))
def npstatistics(data):
    return [data.min(), data.max(), data.mean(), data.std()]
for line in lines:
    line = line.strip()
    if len(line) == 0 or line.startswith('#'):
        continue
    if line.startswith('faultType'):
        comments = line.split(' ')
        fault_type = comments[1]
        description = comments[2]
        print("-------------------")
        print(description)
        descriptionList.append(description)
        continue
    filename, suffix = line.split('.')
    print('Loading data {0} {1} {2}'.format(filename,fault_type,description))
    params = filename.split('_')
    data_no = params[-1]
    #print("///////////////////")
    print(data_no)
    #mat_data = sio.loadmat('./CaseWesternReserveUniversityData/'+filename)
    mat_data = sio.loadmat('E:/Kerasliuyankaicode/sample/CaseWesternReserveUniversityData/'+filename)
    features_considered = map(lambda key_str:key_str.format(data_no), ["X{0}_DE_time", "X{0}_FE_time", "X{0}_BA_time"])
    #print("------------------------")
    #print(features_considered)
    current_features = np.array([mat_data[feature].flatten() for feature in features_considered])
    #print("***************************")
    #print(current_features)
    features = np.concatenate((features, current_features), axis=1)  #列拼接
    #print("+++++++++++++++++++++")
    #print(features)
    # multidimensional_timeseries = np.hstack(current_features)
    data_size = len(mat_data["X{0}_DE_time".format(data_no)]) # current file timeseries length
    statistics = [filename, fault_type, description, data_size]
    statistics += npstatistics(current_features[0])
    statistics += npstatistics(current_features[1])
    statistics += npstatistics(current_features[2])
    #print("..........................")
    #print(current_features[0])
    #print(current_features[1])
    #print(current_features[2])
    statistics_df.loc[statistics_df.size] = statistics

f.close()
print("\nStatistics:")
print(statistics_df.head())

def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std
features[0] = normalize(features[0])
features[1] = normalize(features[1])
features[2] = normalize(features[2])

start_index = 0
for index, row in statistics_df.iterrows():
    fault_type, length = row['fault_type'], row['length']
    current_features = features[:,start_index:start_index+length]
    multidimensional_timeseries = current_features.T
    start_index += length
    data = [multidimensional_timeseries[i:i+TIMESERIES_LENGTH] for i in range(0, length - TIMESERIES_LENGTH, 100)]
    if fault_type not in datas:
        fault_typeList.append(fault_type)
        description=descriptionList[fault_typeList.index(fault_type)]
		#descriptionList.append(description)
        datas[fault_type] = {
            'fault_type':fault_type,
            'description':description,
            'X':np.empty(shape=(0, TIMESERIES_LENGTH, 3))
        }
        
    datas[fault_type]['X'] = np.concatenate((datas[fault_type]['X'], data))

# %%
# random choice
def choice(dataset, size):
    return dataset[np.random.choice(dataset.shape[0], size, replace=False), :]
# make data balance
datas['DE_B']['X'] = choice(datas['DE_B']['X'], 145)
datas['DE_IR']['X'] = choice(datas['DE_IR']['X'], 145)
datas['DE_OR']['X'] = choice(datas['DE_OR']['X'], 145)
datas['FE_B']['X'] = choice(datas['FE_B']['X'], 145)
datas['FE_IR']['X'] = choice(datas['FE_IR']['X'], 145)
datas['FE_OR']['X'] = choice(datas['FE_OR']['X'], 145)
#datas['DE_OR']['X'] = choice(datas['DE_OR']['X'], 14500)
#datas['FE_OR']['X'] = choice(datas['FE_OR']['X'], 14500)

cloud_num=len(fault_typeList)
label_placeholder = np.zeros(cloud_num, dtype=int)
x_data = np.empty(shape=(0,TIMESERIES_LENGTH,3))
y_data = np.empty(shape=(0,cloud_num),dtype=int)
DATASET_SIZE = 0
#BATCH_SIZE = 16
BATCH_SIZE = 32
BUFFER_SIZE = 10000
for index, (key, value) in enumerate(datas.items()):
    sample_size = len(value['X'])
    DATASET_SIZE += sample_size
    print("{0} {1} {2}".format(value['fault_type'], value['description'], sample_size))
    label = np.copy(label_placeholder)
    label[index] = 1 # one-hot encode
    #print("========================")
    #print(value['X'])
    #print(value)
    x_data = np.concatenate((x_data, value['X']))
    labels = np.repeat([label], sample_size, axis=0)
    y_data = np.concatenate((y_data, labels))
#print(datas.items())
total_data = [(x_data[i], y_data[i]) for i in range(0,len(x_data))]
np.random.shuffle(total_data)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

#tsne = TSNE(n_components=3, init='pca', random_state=0)
t0 = time
X_data_p = np.array(x_data)
y_data_p = np.array(y_data)
print(":::::::::::::::::::::::")
print(X_data_p)
print("]]]]]]]]]]]]]]]]]]")
print(y_data_p)
#nsamples, nx, ny = X_data_p[0:1000].shape
nsamples, nx, ny = X_data_p.shape
nyx,nyy=y_data_p.shape
#d2_train_dataset = X_data_p[0:1000].reshape((nsamples,nx*ny))
d2_train_dataset = X_data_p.reshape(nsamples,nx*ny)
#d2_label_dataset = y_data_p.reshape(nyx*nyy)
d2_label_dataset=np.argmax(y_data_p,axis=1)
print(d2_train_dataset)
#X_tsne = tsne.fit_transform(d2_train_dataset)

tsne3 = TSNE(n_components=3, init='pca', random_state=0)
tsne = TSNE(n_components=2, init='pca', random_state=501)
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(d2_train_dataset)
X3_tsne = tsne3.fit_transform(d2_train_dataset)
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

#plot_embedding(X_tsne,y_data_p[0:1000],"t-SNE 2D")
plot_embedding(X_tsne,d2_label_dataset,"t-SNE 2D1")
plt.show()
def plot_embedding_3d(X, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)
    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],X[i, 2],str(d2_label_dataset[i]),
                 color=plt.cm.Set1(d2_label_dataset[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)

plot_embedding_3d(X3_tsne,"t-SNE 3D1 " )
plt.show()
#training_data=total_data[:len(x_data)*0.8]

#将training_data改为total_data,尝试拆分出测试集



# %%
# full_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
# full_dataset = full_dataset.cache().shuffle(buffer_size=BUFFER_SIZE)
# train_size = int(0.7 * DATASET_SIZE)
# test_size = int(0.3 * DATASET_SIZE)
# train_dataset = full_dataset.take(train_size).batch(BATCH_SIZE).repeat()
# test_dataset = full_dataset.skip(train_size).batch(BATCH_SIZE).repeat()

model = Sequential()
#model.add(LSTM(30, input_shape=(TIMESERIES_LENGTH,3)))
model.add(LSTM(30, input_shape=(TIMESERIES_LENGTH,3),return_sequences=True,name="lstm_1"))
#model.add(LSTM(30, input_shape=(TIMESERIES_LENGTH,3)))
model.add(LSTM(units=100,return_sequences=True,name="lstm_2"))
model.add(Dropout(0.2,name="drop_1"))
plot_embedding(X_tsne,d2_label_dataset,"t-SNE 2D")
plt.show()
model.add(LSTM(units=100,name="lstm_3"))
model.add(Dropout(0.2,name="drop_2"))
#plot_embedding(X3_tsne,d2_label_dataset,"t-SNE 2D")
model.add(Dense(cloud_num, activation='softmax',name="dense_1"))
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',  metrics=['accuracy'])
model.summary()



'''history = model.fit(x_data, y_data,
    #validation_split=0.25, 
	validation_split=0.10,
    #epochs=50, batch_size=16,
    epochs=20, batch_size=16,	
    verbose=1,
    callbacks=callbacks_list)
	'''
history = model.fit(X_train, y_train,
    #validation_split=0.25, 
	validation_split=0.20,
    #epochs=50, batch_size=16,
    epochs=20, batch_size=16,	
    verbose=1)
model.save_weights('fname1.h5')
plot_embedding(X_tsne,d2_label_dataset,"t-SNE 2D")
plt.show()
plot_embedding_3d(X3_tsne,"t-SNE 3D " )
plt.show()
#model = load_model('E:/Kerasliuyankaicode/my_model.h5')

'''checkpoint = ModelCheckpoint('classification1_10.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]'''

#model = tf.keras.models.load_model('./my_model.h5')
score = model.evaluate(X_test, y_test, verbose=1	)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(fault_typeList)
print(descriptionList)
#np.argmax(model.predict(x_data[104005:104015]))
indexArray=np.argmax(model.predict(X_test[:10]),axis=1)
print(indexArray)
for idx, val in enumerate(indexArray, 1):
   print(idx, val)
   print ('故障种类 :', fault_typeList[val])
   print ('故障种类名称:', descriptionList[val])   
#for index in range(0,len(indexArray)):
   #print(index)
   #print ('当前水果 :', fault_typeList[index])
   #print ('当前水果 :', descriptionList[index])
#print(fault_typeList[np.argmax(model.predict(x_data[104005:104015]),axis=1])
#print(descriptionList[np.argmax(model.predict(x_data[104005:104015]),axis=1]) 
print('test after load: ', model.predict(X_test[:10]))
print('test label: ',y_test[:10])
#print('test after load: ', model.predict(X_test[0:2]))


	

#with open('./classificationTrainHistoryDict1_10', 'wb') as file_pi:
'''with open('./classificationTrainHistoryDict1_10', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)'''

def plot_accuracy(history):
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
plot_accuracy(history)
#plot_accuracy()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
plot_loss(history)

'''
labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
y_true=np.argmax(model.predict(X_test[:4]),axis=1)
y_pred=np.argmax(y_test[:4],axis=4)
#y_true = np.loadtxt('../Data/re_label.txt')
#y_pred = np.loadtxt('../Data/pr_label.txt')
 
tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig('../Data/confusion_matrix.png', format='png')
plt.show()'''
#https://www.jb51.net/article/165319.htm