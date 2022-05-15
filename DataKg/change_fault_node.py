#  将fault和feature格式的relation转换为fault1-5格式
import pandas as pd
import numpy as np

name = 'belongs'
write_name = 'belongs_new'
filename = r".././import/level/" + name + ".csv"  # 文件地址读取csv文件
write_path = r".././import/level/" + write_name + ".csv"

# for i in range(len(data)):
list = []
f = open(filename, 'r', encoding='utf-8')
tmp_arr = f.readlines()
f.close()
tmp_arr = tmp_arr[1:]
for item in tmp_arr:
    item = item.split(',')
    item1 = item[0].split('_')
    len1 = str(len(item1))
    item[1] = 'fault_level_' + len1
    item2 = item[2].split('_')
    len2 = str(len(item2))
    item[3] = 'fault_level_' + len2
    item = ','.join(item)
    list.append(item)
print(list)
writer = open(write_path, 'w', encoding='utf-8')
for item in list:
    writer.write(item)
writer.close()