taskList = [6856, 2499, 624, 8939, 10145, 6953, 9528, 1671, 1417, 8497, 5841, 8632, 6405, 1065, 9805, 9580, 8378, 3216, 7138, 8605]
nodeList = [5000, 2500, 2500, 1500, 1000]
timeSum = [0, 0, 0, 0, 0]
taskArr = ['', '', '', '', '']


for i in range(len(taskList)):
    if i==3:
        continue
    tmp = []
    for j in range(len(nodeList)):
        val = timeSum[j] * 0.8 + taskList[i] / nodeList[j] * 0.2
        tmp.append(val)
    min = 10000
    minIndex = 0
    for x in range(len(tmp)):
        if tmp[x] < min:
            min = tmp[x]
            minIndex = x
    timeSum[minIndex] += taskList[i] / nodeList[minIndex]
    taskArr[minIndex] = taskArr[minIndex] + str(i) + ','

# print('节点1已分配任务:', taskArr[0], '时间', timeSum[0])
# print('节点2已分配任务:', taskArr[1])
# print('节点3已分配任务:', taskArr[2])
# print('节点4已分配任务:', taskArr[3])
# print('节点5已分配任务:', taskArr[4])
for i in range(len(taskArr)):
    print('节点{0}已分配任务:'.format(i), taskArr[i], '时间：', timeSum[i])