# 统计各个特征出现数量
filePath = '../../../import/origin/feature.csv'
list = []
f = open(filePath, 'r', encoding='utf-8')
tmp_arr = f.readlines()
f.close()
tmp_arr = tmp_arr[1:]
for item in tmp_arr:
    list.append(item.split(',')[0])
# print(list)
count = []
