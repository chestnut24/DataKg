# 用于生成 推荐算法 所需的是三个文件 item-id对应表 user-item打分表 item关系表
import random

# 1. 构建故障节点及id，并写入 idPath 路径下的文件中
def handle_feature(s, list, index, id):  # 通过递归的形式组合各个故障特征
    arr = list[index]
    for i, item in enumerate(arr):
        tmp = s + item
        nowId = id + str(i + 1)
        diff = fe_len - len(nowId)
        for j in range(diff):
            nowId += '0'  # 空缺的地方补0
        result.append(tmp + ':' + nowId)  # 给每个节点后加上id
        nameList.append(tmp)  # 只保存名称
        idList.append(nowId)  # 只保存id
        if index + 1 < fe_len:
            handle_feature(tmp + '_', list, index + 1, id + str(i + 1))


# 2. 构建图谱文件kgList (id，关系，id)，加入图谱关系，并写入kgPath路径下的文件中
def handle_rel(name1, name2):  # 用于返回两个节点之间的关系，如果无关返回FALSE
    name1 = name1.split('_')
    name2 = name2.split('_')
    len1 = len(name1)
    len2 = len(name2)
    if abs(len1 - len2) == 1:  # 层级相差为1时才会有关系
        flag = 0  # 用以判断是否相同，不同时变为1
        if len1 - len2 < 0:  # len1比较小说明层级较小，就要判断第一个是不是包含第二个了
            for index in range(len1):
                if name1[index] != name2[index]:
                    flag = 1
                    break
        else:
            for index in range(len2):
                if name1[index] != name2[index]:
                    flag = 1
                    break
        if flag == 0:
            if len1 - len2 > 0:
                return 'belongsTo'
            else:
                return 'contains'
    return 'FALSE'


# 3. 创建打分表ratings，此处两个model，model1是权值相同，model2是权值不同且按照等差数列排列
def createRating(model, Smax):
    if model == 1:
        rating_model1(Smax)
    else:
        rating_model2(Smax)


def rating_model1(Smax):  # model1 权值相等
    print('model1')
    listLen = len(nameList)
    avg = Smax / fe_len  # 平均权值为 总值/特征数
    for i in range(listLen):
        for j in range(listLen):
            if nameList[i] != nameList[j]:  # 不为同一个节点时才进行打分
                diffRes = judgeDiff(nameList[i], nameList[j])
                # print(nameList[i], nameList[j], diffRes)
                k = diffRes.pop()  # k为层级差
                m = len(diffRes)  # m为特征不同的个数
                SimScore = Smax - (m + k) * avg
                if SimScore < 0:
                    SimScore = 0  # 当小于0时置为零
                # ratings构造要求 第一项为 节点1的id 第二项为 节点2的name 第三项为节点1和节点2的相似度
                ratingList.append('"%s";"%s";"%d"\n' % (idList[i], nameList[j], SimScore))  #


def rating_model2(Smax):  # model2 权值不等
    print('model2')
    listLen = len(nameList)
    n = fe_len
    avg = Smax / n  # 平均权值为 总值/特征数
    LevelScore = avg
    diff = -1 * Smax / (n * (n - 1))  # 本文是 -10/(4*5)=-0.5 另外注意diff是正负值，本文采用降序，则diff为-0.5
    paraArr = []  # 权重参数列表，本文原本是 [3, 2.5, 2, 1.5, 1]
    # 根据错误频率[1, 68, 192, 1851, 113] 改为 [3, 2.5, 1.5, 1, 2]
    a1 = Smax / n - (n - 1) * diff / 2  # 计算首项
    for i in range(n):
        if i == 0:
            paraArr.append(a1)
        else:
            a = a1 + i * diff  # 计算通项
            paraArr.append(a)
    # 根据错误频率[1, 68, 192, 1851, 113] 改为 [3, 2.5, 1.5, 1, 2]
    paraArr = [3, 2.5, 1.5, 1, 2]
    for i in range(listLen):
        for j in range(listLen):
            if nameList[i] != nameList[j]:  # 不为同一个节点时才进行打分
                diffRes = judgeDiff(nameList[i], nameList[j])
                # print(nameList[i], nameList[j], diffRes)
                k = diffRes.pop()  # k为层级差
                aTotal = 0  # 不同特征权重的累加和
                for item in diffRes:
                    aTotal += paraArr[item]
                SimScore = Smax - aTotal - k * LevelScore
                if SimScore < 0:
                    SimScore = 0  # 当小于0时置为零
                # ratings构造要求 第一项为 节点1的id 第二项为 节点2的name 第三项为节点1和节点2的相似度
                # ratingList.append('"%s";"%s";"%.1f"\n' % (idList[i], nameList[j], SimScore))
                ratingList.append('"%s";"%s";"%d"\n' % (idList[i], nameList[j], SimScore))
                myRatingList.append('%s %s %.1f\n' % (nameList[i], nameList[j], SimScore))


def judgeDiff(name1, name2):  # 用来判断两个节点是否相同，返回一个数组，前几项为不同项的下标，最后一项为层级差
    diffRes = []
    name1 = name1.split('_')
    name2 = name2.split('_')
    len1 = len(name1)
    len2 = len(name2)
    if len1 < len2 or len1 == len2:  # 当len1比较短或者节点长度相同时
        calLen = len1
    else:
        calLen = len2
    for index in range(calLen):
        if name1[index] != name2[index]:
            diffRes.append(index)
    diffRes.append(abs(len1 - len2))  # 最后加一项层级差
    return diffRes


# 特征、路径等参数确定
feature = ['end', 'location', 'size', 'load', 'clock']
feature_list = [
    ['DE', 'FE'],
    ['BO', 'IR', 'OR'],
    ['size7', 'size14', 'size21'],
    ['load0', 'load1', 'load2', 'load3'],
    ['clock3', 'clock6', 'clock12']
]
idPath = '../data/kgData/item_index2entity_id_rehashed.txt'
kgPath = '../data/kgData/kg_rehashed.txt'
ratingPath = '../data/kgData/ratings.txt'
myRatingPath = '../data/kgData/myRatings.txt'
# print(feature_list)
result = []  # 保存故障节点名称+id
nameList = []
idList = []
kgList = []
ratingList = []
myRatingList = []
fe_len = len(feature)

# 1. 构建故障节点及id，并写入 idPath 路径下的文件中
handle_feature('', feature_list, 0, '')
writer = open(idPath, 'w', encoding='utf-8')
for i in range(len(nameList)):
    writer.write('%s\t%s\n' % (nameList[i], idList[i]))
print('id file done')

# 2. 构建图谱文件kgList (id，关系，id)，加入图谱关系，并写入kgPath路径下的文件中

for i in range(len(idList)):
    for j in range(len(idList)):
        rel = handle_rel(nameList[i], nameList[j])
        if rel != 'FALSE':
            if rel == 'belongsTo':  # 此处写入双向关系，正常只有第一个
                kgList.append('%s\t%s\t%s\n' % (idList[i], rel, idList[j]))
                kgList.append('%s\t%s\t%s\n' % (idList[j], 'contains', idList[i]))  # 可不写
            if rel == 'contains':
                kgList.append('%s\t%s\t%s\n' % (idList[i], rel, idList[j]))
                kgList.append('%s\t%s\t%s\n' % (idList[j], 'belongsTo', idList[i]))  # 可不写
writer = open(kgPath, 'w', encoding='utf-8')
for i in range(len(kgList)):
    writer.write(kgList[i])
print('kg file done')

# 3. 创建打分表ratings，此处两个model，model1是权值相同，model2是权值不同且按照等差数列排列
createRating(2, 10)  # 参数为model和总分数
# print(ratingList)
writer = open(ratingPath, 'w', encoding='utf-8')
for item in ratingList:
    if random.random() > 0.4:  # 随机筛选加入的评分，剩下一半不评分用于做空白训练集
        writer.write(item)
print('rating file done')

writer = open(myRatingPath, 'w', encoding='utf-8')
for item in myRatingList:
    # writer.write(item)
    if random.random() > 0.0:
        writer.write(item)
print('my rating file done')
