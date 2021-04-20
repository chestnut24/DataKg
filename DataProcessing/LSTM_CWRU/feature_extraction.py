# 用来提取最大值最小值等特征
import numpy as np
import pandas as pd
import scipy.io as sio
import os

# 防止输出省略号，1000以上才省略
from DataKg.py2neo_search import search

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def data_feature():
    # 打开文件列表，12k频率表
    # f = open("../data/12kdata.txt", "r")  # 12k所有
    # f = open("../data/DE&level.txt", "r", encoding='UTF-8')  # 12k驱动端 共9个类型
    # f = open("../data/FE&level.txt", "r", encoding='UTF-8')  # 12k风扇端 共6个类型
    # f = open("../data/12k&DE&FE&level&15.txt", "r", encoding='UTF-8')  # 12k风扇端+驱动端 共15个类型
    f = open("../data/83.txt", "r", encoding='UTF-8')  # 83个类型

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
        features_considered = map(lambda key_str: key_str.format(data_no),
                                  ["X{0}_DE_time", "X{0}_FE_time", "X{0}_BA_time"])
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
        # return statistics_df

    # print("\nStatistics:")
    # 查看前五行数据。此处为DataFrame和head()函数，查看前几行，默认为5  后几行为tail()
    # print(statistics_df.head())
    # print('训练集', statistics_df)

    f.close()
    # return test_df
    return statistics_df


def test_feature():  # 划分测试集
    # filename = os.path.abspath(r'../data/83.txt')  # 为了防止在其他文件引用该python文件时，相对路径更改，因此在此处获取绝对路径
    # print(filename)
    # filename = r'D:\code\KG\Datakg\DataProcessing\data\83.txt'  # 注意此处绝对路径是通过上一句获得的，解除注释获得路径后，再注释掉
    # 打开文件列表，12k频率表
    # f = open("../data/12kdata.txt", "r")  # 12k所有
    # f = open("../data/DE&level.txt", "r", encoding='UTF-8')  # 12k驱动端 共9个类型
    # f = open("../data/FE&level.txt", "r", encoding='UTF-8')  # 12k风扇端 共6个类型
    # f = open("../data/12k&DE&FE&level&15.txt", "r", encoding='UTF-8')  # 12k风扇端+驱动端 共15个类型
    f = open(r"../data/83.txt", "r", encoding='UTF-8')  # 83个类型
    # 读每一行
    lines = f.readlines()
    # data列表
    datas = {}
    fault_type = 'undefined'
    description = ''
    test_df = pd.DataFrame(
        columns=['filename', 'fault_type', 'description', 'length', 'DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min',
                 'FE_max', 'FE_mean', 'FE_std', 'BA_min', 'BA_max', 'BA_mean', 'BA_std'])
    # 生成一个3*0的数组

    # 之前 lines = f.readlines()

    test_size = 36000  # 定义每段测试小单元长度
    for line in lines:

        DE_min_list = []
        # 消除首尾空格、换行
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        if line.startswith('faultType'):
            comments = line.split(' ')
            fault_type = comments[1]
            description = comments[2]
            continue
        filename, suffix = line.split('.')
        params = filename.split('_')
        data_no = params[-1]
        mat_data = sio.loadmat('../data/CaseWesternReserveUniversityData/' + filename)
        features_considered = map(lambda key_str: key_str.format(data_no),
                                  ["X{0}_DE_time", "X{0}_FE_time", "X{0}_BA_time"])
        current_features = np.array([mat_data[feature].flatten() for feature in features_considered])
        len1 = int(len(current_features[0]) * 0.7)
        feature = [current_features[0][len1:], current_features[1][len1:], current_features[2][len1:]]

        feature_DE = list_split(feature[0], test_size)
        feature_FE = list_split(feature[1], test_size)
        feature_BA = list_split(feature[2], test_size)
        # print('fearue_DE:', feature_DE)

        # print('feature', feature)
        data_size = len(mat_data["X{0}_DE_time".format(data_no)])  # current file timeseries length
        # 数据列表定义为 文件名，错误类型，描述，数据大小

        test_data_len = int(data_size * 0.3)  # 总的测试集长度
        for i in range(test_data_len // test_size):
            # print(i)
            test_data = [filename, fault_type, description, len(feature_DE[i])]
            test_data += npstatistics(feature_DE[i])
            test_data += npstatistics(feature_FE[i])
            test_data += npstatistics(feature_BA[i])
            # print(test_data)
            test_df.loc[test_df.size] = test_data
            DE_min_list.append(test_data[9])  # 记录每个测试小单元的min，max等值

        # print(max(DE_min_list) - min(DE_min_list))


        # test_data = [filename, fault_type, description, len(feature[0])]
        # test_data += npstatistics(feature[0])
        # test_data += npstatistics(feature[1])
        # test_data += npstatistics(feature[2])
        # test_df.loc[test_df.size] = test_data

    f.close()
    return test_df


# 定义函数：返回数据的min,max,mean,std
def npstatistics(data):
    return [data.min(), data.max(), data.mean(), data.std()]


# 切分数组
def list_split(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]

#
# data_feature()
# data1 = data_feature().drop_duplicates('fault_type')  # 按照错误类型去重
# data1 = data_feature()
# print('训练集', data1)


data = test_feature()
# print(data)
data.reset_index(drop=True, inplace=True)
print('测试集', data)
# print(search(data))
'''
# 测试搜索图谱
search(data)
'''

# '''

# 将得到的特征数据导入csv的过程
# feature_list = ['DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min', 'FE_max', 'FE_mean', 'FE_std', 'BA_min', 'BA_max',
#                 'BA_mean', 'BA_std']  # 输出指定列
feature_list = ['DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min', 'FE_max', 'FE_mean', 'FE_std', 'BA_min', 'BA_max',
                'BA_mean', 'BA_std']  # 输出指定列
fault_list = [
    'DE_BO_7', 'DE_BO_14', 'DE_BO_21', 'DE_IR_7', 'DE_IR_14', 'DE_IR_21', 'DE_OR_7', 'DE_OR_14', 'DE_OR_21', 'FE_IR_7',
    'FE_IR_14', 'FE_IR_21', 'FE_OR_7', 'FE_OR_14', 'FE_OR_21'
]
data = data[feature_list]  # 只留下特征值，并且索引重新排序 为012
data.reset_index(drop=True, inplace=True)
print(data)

os.system('python change_encode.py')  # 更改文件编码

filename = r"../.././import/feature.csv"  # 文件地址
file = pd.read_csv(filename)  # 读取csv文件
# 仅进行简单的“拼接”而不是合并呢，要使用concat函数 参数axis=1表示列拼接，axis=0表示行拼接。多次运行会造成重复，且需要file中没有data中的列
# file = pd.concat([file, data], axis=1, join='outer')  # 可以在没有定义BA_min的时候使用，使用一次后得到列名，之后再改用update进行更新
file.update(data)  # 后来发现用update更好用，能够更新file，而不会造成重复。但是需要在file中提前写好相应列才会更新
# file = pd.merge(file, data)  # 仅进

# mode=a，以追加模式写入,header表示列名，默认为true,index表示行名，默认为true，再次写入不需要行名
file.to_csv(filename, index=False)  # 写入csv文件
print('write success')
# '''
