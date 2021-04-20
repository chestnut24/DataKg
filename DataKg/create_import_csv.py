# 自动化构建图谱第一步  根据输入规则的表格转化为知识图谱
# 根据输入要求（必有有哪些特征之类的）生成csv文件，然后在调用 py2neo_initialize 文件将csv文件转化为知识图谱
# faultType      DE_OR_size7_clock12_load0       驱动端外圈故障_深度7_点钟12_载荷0
import pandas as pd
import numpy as np
import os
import csv


def read_feature_csv(filename):  # 读取提供的总特征表csv文件
    file = pd.read_csv(filename)  #
    file.fillna('', inplace=True)  # 以''填充nan
    data = np.array(file.loc[:, :])  # 将读取的数据转为数组格式，不保留列名
    # labels = list(data.columns.values)  # 获取列名
    return data


def create_save_csv(path, filename, column_name):  # 创建保存用的csv 输入分别是 文件夹名，csv名，列名
    # column_name = ['nodeID', 'name', 'type', 'describe']
    filepath = path + filename + ".csv"
    # file = open(filepath, 'w', newline='')
    # 1. 创建文件对象
    file = open(filepath, 'w', encoding='utf-8', newline='')
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(file)
    # 3. 构建列表头
    csv_writer.writerow(column_name)
    # 5. 关闭文件
    file.close()


def write_csv_row(path, row):  # 在指定的csv文件写入指定的一行
    file = open(path, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(row)
    file.close()


# 此处递归调用 pre是前面的短语 arr是数组 layer是当前层数 path是写入路径 后面三个分别是 故障 特征 关系 文件所在位置
def build_feature_name(pre, arr, layer, fault_path, feature_path, relation_path):
    if layer == 0:  # 第一行
        for item in arr[0][1:]:  # 加[1:]是为了去除第一列
            if item != '':
                write_csv_row(fault_path, [pre + item, 'fault', pre + item])  # 写入故障节点
                write_csv_row(relation_path, [pre + item, 'fault', 'FAULT', 'fault', 'belongsto'])  # 写入关系
                build_feature_name(pre + item, arr, layer+1, fault_path, feature_path, relation_path)
    elif len(arr) == layer+1:  # 最后一行
        for item in arr[layer][1:]:
            if item != '':
                write_csv_row(feature_path, [pre + '_' + item, 'feature', pre + '_' + item])
                write_csv_row(relation_path, [pre + '_' + item, 'feature', pre, 'fault', 'belongsto'])  # 写入关系
    else:  # 其他行
        for item in arr[layer][1:]:  # 加[1:]是为了去除第一列
            if item != '':
                write_csv_row(fault_path, [pre + '_' + item, 'fault', pre + '_' + item])
                write_csv_row(relation_path, [pre + '_' + item, 'fault', pre, 'fault', 'belongsto'])  # 写入关系
                build_feature_name(pre + '_' + item, arr, layer+1, fault_path, feature_path, relation_path)


path = r".././import/"
all_feature_filename = r".././import/data_all_feature.csv"

data = read_feature_csv(all_feature_filename)

create_save_csv(path, 'fault', ['name', 'type', 'describe'])
create_save_csv(path, 'feature',
                ['nodeID', 'name', 'type', 'describe', 'DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min', 'FE_max',
                 'FE_mean', 'FE_std', 'BA_min', 'BA_max', 'BA_mean', 'BA_std'])
create_save_csv(path, 'belongs', ['start_entity', 'label', 'end_entity', 'label', 'relationship'])

write_csv_row(path+'fault.csv', ['FAULT', 'fault', 'FAULT'])  # 先写入一个总结点，叫做 FAULT

build_feature_name('', data, 0, path+'fault.csv', path+'feature.csv', path+'belongs.csv')
print('import csv build (fault, feature, relation)')

