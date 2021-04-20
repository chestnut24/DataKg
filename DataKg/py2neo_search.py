# 本文件对知识图谱进行查询并展示，有展示路径有展示详细信息
import DataKg.neo4j_model as neo
import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


# datas = graph.search_all()
# for node in datas:
#     print(node[0]['describe'])
#     # print(node)


# node = datas[0]  # 循环打印数组里的所有项
# for item in node[0]:
#     print(node[0][item])

def return_path(graph, first_name):  # 返回指定节点的故障路径
    # first_describe = '风扇端外圈故障_size21'
    fault_name_list = [first_name]  # 故障名字列表
    fault_describe_list = []  # 故障描述列表
    first_data = graph.search_node_name(first_name)  # 查询第一个节点的描述
    data = graph.search_father_node(first_name)
    print(data)
    describe = first_data[0][0]['describe']
    fault_describe_list.append(describe)
    while len(data) != 0:
        name = data[0][0]['name']
        describe = data[0][0]['describe']
        fault_name_list.append(name)
        fault_describe_list.append(describe)
        data = graph.search_father_node(data[0][0]['name'])  # 查询下一个节点
    # print('故障名子图：\n', ' ----> '.join(fault_name_list), '\n')
    # print('故障描述子图：\n', ' ----> '.join(fault_describe_list), '\n')
    print('\n\033[7;31;40m 故障名子图：\033[0m\n\n', ' ----> '.join(fault_name_list[:2]), ' ----> ',
          ' ----> '.join(fault_name_list[3:]), '\n')
    print('\n\033[7;31;40m 故障描述子图：\033[0m\n\n', ' ----> '.join(fault_describe_list[:2]), ' ----> ',
          ' ----> '.join(fault_describe_list[3:]), '\n')


def search(datas):  # 此处是自己编写的根据最大值最小值进行分类查询，效果一般，准备弃用
    acc = 0
    graph = neo.Neo4j_Operate()
    graph.connect_db()
    # thresholdList是阈值列表（min, max, mean, std有着不同的阈值），valueList是传递进来的查询值列表
    thresholdList = [0.4, 0.4, 0.005, 0.01]
    # thresholdList = [0.4, 0.4, 0.005, 0.05]

    right = 0
    return_name = ''  # 查询结果返回值
    feature_list = ['DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min', 'FE_max', 'FE_mean', 'FE_std', 'BA_min',
                    'BA_max', 'BA_mean', 'BA_std']  # 输出指定列
    # print(datas['fault_type'][50])
    data = pd.DataFrame(datas)
    data_len = len(data)
    # print(data)
    for index, row in data.iterrows():
        label = row['fault_type']  # 标签
        print('label:', label)
        valueList = row[feature_list].tolist()  # 值列表
        result = graph.search_feature_node(thresholdList, valueList)
        if len(result) == 0:
            print('未找到节点')
            return_name = 'DE_BO_size7_load0'
            # return_path('DE_BO_size7_load0')
        else:
            # print('label:', label)
            print('result_num:', len(result))
            return_name = result[0][0]['name']
            # return_path(result[0][0]['describe'])
        if label == return_name:
            right += 1
    if data_len != 0:
        acc = right / data_len
    else:
        print('输入数组长度为0')
    print('acc:', acc)

    return 'hello'


def display_node_information(node_name):  # 进行节点信息的查询与展示，输入的是name，比如 FE_OR_size7_clock3_load0
    graph = neo.Neo4j_Operate()
    graph.connect_db()

    # data = graph.search_node_name(node_name)[0][0]  # 返回节点信息
    # print('节点信息', data)

    # print("\033[7;31;40m FBI Warning \033[0m")
    print('\n\033[7;31;40m 查询节点：\033[0m\n\n', node_name, '\n')
    return_path(graph, node_name)  # 返回节点路径

    print('\033[7;31;40m 该节点所选特征：\033[0m\n')
    features = node_name.split('_')
    for index, feature in enumerate(features):
        # print(index + ': ' + feature + '-' + name_list[feature])
        print(str(index + 1) + ': ' + str(name_list[feature]) + ' - ' + str(feature))

    print('\n\033[7;31;40m 同级节点：\033[0m\n')
    parent = graph.search_father_node(node_name)  # 父结点
    children = graph.search_child_node(parent[0][0]['name'])
    for index, child in enumerate(children):
        print(str(index + 1) + ': ' + child[0]['name'])

    print('\n\033[7;31;40m 子节点：\033[0m\n')
    children = graph.search_child_node(node_name)
    for index, child in enumerate(children):
        print(str(index + 1) + ': ' + child[0]['name'])

    # title_list = ['nodeID', 'name', 'type', 'describe', 'DE_min', 'DE_max', 'DE_mean', 'DE_std', 'FE_min', 'FE_max',
    #               'FE_mean', 'FE_std', 'BA_min', 'BA_max', 'BA_mean', 'BA_std']
    # tplt = "{0:<10}\t{1:>11.8f}\t  "  # format格式化输出，{0}代表第0个数据， :<是左对齐， :>是右对齐 :^是居中对齐， 10和11代表有相应个数的空格，  .8f代表输出八位浮点数小数
    # tplt2 = "{0:<11}\t {1:<15}\t  "
    #
    # for index, item in enumerate(title_list):
    #     # print(item, data[item], end='   ')
    #     if index < 4:
    #         # print(item, '\t', data[item], end='   ')
    #         print(tplt2.format(item, data[item], chr(12288)), end='   ')
    #         print('')
    #     else:
    #
    #         print(tplt.format(item, data[item], chr(12288)), end='   ')
    #         if index != 4 and index % 3 == 0:
    #             print('')
    # print('')


name_list = {
    'DE': '驱动端',
    'FE': '风扇端',
    'BO': '滚动体故障',
    'IR': '内圈故障',
    'OR': '外圈故障',
    'size7': '深度7',
    'size14': '深度14',
    'size21': '深度21',
    'load0': '载荷0',
    'load1': '载荷1',
    'load2': '载荷2',
    'load3': '载荷3',
    'clock3': '点钟3',
    'clock6': '点钟6',
    'clock12': '点钟12',
}

node_name = 'FE_IR_size21_load0'
display_node_information(node_name)
