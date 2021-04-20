# 用来读取txt文件中 以更新feature节点csv 具体的故障名字以及描述 之前半自动是读取数据集标签文本来改csv文件
import pandas as pd

f = open("../data/83.txt", "r", encoding='UTF-8')  # 83个类型
lines = f.readlines()
fault_type = 'undefined'
description = ''
name = []
describe = []

for line in lines:
    line = line.strip()
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
    name.append(fault_type)
    describe.append(description)
print('type: ', name)
print('describe: ', describe)

filename = r"../.././import/feature.csv"  # 文件地址
file = pd.read_csv(filename)  # 读取csv文件

file['name'] = pd.DataFrame(name)
file['describe'] = pd.DataFrame(describe)

file.to_csv(filename, index=False)  # 写入csv文件

