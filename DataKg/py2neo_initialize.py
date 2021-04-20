# 从本项目的import文件夹中读取数据，通过create和merge的方式写入数据库
import DataKg.neo4j_model as neo
import pandas as pd
import numpy as np


def read_csv(name):  # 读取文件
    filename = r".././import/" + name + ".csv"  # 文件地址读取csv文件
    file = pd.read_csv(filename)  #
    file.fillna(0, inplace=True)  # 以''填充nan
    data = np.array(file.loc[:, :])  # 将读取的数据转为数组格式，不保留列名
    # labels = list(data.columns.values)  # 获取列名
    return data


def create_graph():  # 初始化图谱
    graph = neo.Neo4j_Operate()
    graph.connect_db()
    data = read_csv('fault')
    for line in data:
        # graph.add_fault_node('fault', line[0], line[1], line[2], line[3])  # 改为了三列
        graph.add_fault_node('fault', line[0], line[1], line[2])  # 改为了三列
    print('导入fault节点')

    data = read_csv('feature')
    for line in data:
        # graph.add_feature_node('feature', line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7],
        #                        line[8], line[9], line[10], line[11], line[12], line[13], line[14], line[15])
        graph.add_feature_node('feature', line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7],
                               line[8], line[9], line[10], line[11], line[12], line[13], line[14])
    print('导入feature节点')

    data = read_csv('belongs')
    for line in data:
        graph.add_single_relation('belongsto', line[0], line[1], line[2], line[3], line[4])
    print('导入belongs关系')


def update_graph():  # 更新图谱（重新导入）
    graph = neo.Neo4j_Operate()
    graph.connect_db()
    graph.delete_all()
    create_graph()


update_graph()
