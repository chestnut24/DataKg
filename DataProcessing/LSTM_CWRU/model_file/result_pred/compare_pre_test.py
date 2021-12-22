pred_path = 'y_pred.txt'
test_path = 'y_test.txt'
test_label_path = 'y_test_label.txt'
diff_path = 'diff.txt'
diff_label_path = 'diff_label.txt'
data_origin = 'data_index_origin.txt'
data_new = 'data_index.txt'

# 读取y_pred
f = open(pred_path, "r", encoding='UTF-8')
pred = f.readlines()
f.close()

# 读取y_test
f = open(test_path, "r", encoding='UTF-8')
test = f.readlines()
f.close()
test_len = len(test)
index = 1
test_res = []
str_tmp = ''
# 去掉其中多余的空格和[]还有回车
for i in range(test_len):
    tmp = test[i]
    tmp = tmp[1:]
    tmp = tmp.split('\n')[0]
    if index == 3:
        tmp = tmp.split(']')[0]
    str_tmp += tmp + ' '
    if index == 3:
        test_res.append(str_tmp.strip())
        str_tmp = ''
        index = 0
    index += 1

test = []
# 将独热编码变为数字格式
for item in test_res:
     item = item.split(' ')
     for i in range(len(item)):
         if item[i] == '1':
             test.append(str(i) + '\n')
writer = open(test_label_path, 'w', encoding='utf-8')
for item in test:
    writer.write(item)

# 判断两者区别
res_len = len(test)
diff_res = []
for i in range(res_len):
    if test[i] != pred[i]:
        test_tmp = test[i].split('\n')[0]
        pred_tmp = pred[i].split('\n')[0]
        tmp = test_tmp + ' ' + pred_tmp
        diff_res.append(tmp)
writer = open(diff_path, 'w', encoding='utf-8')
for item in diff_res:
    writer.write(item + '\n')
print(diff_res)

# 读取原来index并处理
f = open(data_origin, "r", encoding='UTF-8')
index_origin = f.readlines()
f.close()
index_new = []
for item in index_origin:
    item = item.split(' ')
    if item[0] == 'faultType':
        index_new.append(item[1])
writer = open(data_new, 'w', encoding='utf-8')
for item in index_new:
    writer.write(item + '\n')
# print(index_new)
print(index_new)

# 将diff从数字形式变为标签形式
diff_label = []
# diff_res 是先 test 再 pred

index_len = len(index_new)
for item in diff_res:
    item = item.split(' ')
    t = item[0]
    p = item[1]
    # print('item', item)
    if len(item) != 0:
        for i in range(index_len):
            if str(i) == t:
                t = index_new[i]
            if str(i) == p:
                p = index_new[i]
        diff_label.append(t + ' ' + p)
writer = open(diff_label_path, 'w', encoding='utf-8')
for item in diff_label:
    writer.write(item + '\n')
# print(diff_label)

# 比较哪里出问题多
diff_num = [0, 0, 0, 0, 0]
len_diff_num = 0
len_diff_arr = []
for item in diff_label:
    item_save = item
    item = item.split(' ')
    t = item[0].split('_')
    p = item[1].split('_')
    t_len = len(t)
    p_len = len(p)
    if t_len != p_len:
        len_diff_num += 1
        len_diff_arr.append(item_save)
    else:
        for i in range(t_len):
            if t[i] != p[i]:
                diff_num[i] += 1
print('diff_num', diff_num)
print('len_diff_num', len_diff_num)
print('len_diff_arr', len_diff_arr)
