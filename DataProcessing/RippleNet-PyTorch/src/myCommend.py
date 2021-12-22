# 获取高分 进行推荐
# 1. 运行 kgDataCreate 创建打分文件
# 2. 运行 compare_pre_test 创建名称表、预测错误文件
# 3. 运行 myCommend 根据打分对预测错误的文件重新推荐
import random

rating_path = '../data/kgData/myRatings.txt'
fault_path = '../../LSTM_CWRU/model_file/result_pred/diff_label.txt'

test = 'DE_OR_size21_load3_clock3'

fault_list = []
rating_list = []


def mysort(array):
    random.shuffle(array)
    for i in range(1, len(array)):
        for j in range(0, len(array) - i):
            if array[j][1] < array[j + 1][1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    return array


def rateAndSort(test, pred, count):
    res = []
    for item in rating_list:
        if pred == item[0]:
            tmp = [item[1], float(item[2])]
            res.append(tmp)
    res = mysort(res)
    for index in range(k_num):
        if test == res[index][0]:
            k_res[index] += 1
    if count % 100 == 0:
        print('count', count, 'k_res', k_res)


def calSelect():
    print(1)


f = open(fault_path, 'r', encoding='utf-8')
tmp_arr = f.readlines()
f.close()
for item in tmp_arr:
    item = item.split('\n')[0].split(' ')
    fault_list.append(item)

f = open(rating_path, 'r', encoding='utf-8')
tmp_arr = f.readlines()
f.close()
for item in tmp_arr:
    item = item.split('\n')[0].split(' ')
    rating_list.append(item)
# print(rating_list)

# t1 = rateAndSort(test)
k_num = 10  # top k 个数
k_res = []  # 每个位数上正确的次数
k_add = []  # 累计次数
count = 0
for i in range(k_num):  # 用于统计在第几个推荐正确
    k_res.append(0)
    k_add.append(0)
for index, item in enumerate(fault_list):
    rateAndSort(item[0], item[1], index)
print(k_res)
add = 0
for i in range(k_num):
    add += k_res[i]
    k_add[i] = add
diff_len = len(fault_list)
print('diff_len', diff_len)
print('k_res', k_res)
print('k_add', k_add)
for i in range(k_num):
    c = k_add[i] / diff_len
    print('k为%d时：%.4f\t' % (i + 1, c))
