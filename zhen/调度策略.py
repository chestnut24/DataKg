weightList = [0.90,  0.80, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
taskList = [6856, 2499, 624, 8939, 10145, 6953, 9528, 1671, 1417, 8497, 5841, 8632, 6405, 1065, 9805, 9580, 8378, 3216, 7138, 8605]
nodeList = [5000, 2500, 2500, 1500, 1000]
timeSum = [0, 0, 0, 0, 0]
taskArr = ['', '', '', '', '']
for k in range(len(weightList)):
    timeSum = [0, 0, 0, 0, 0]
    taskArr = ['', '', '', '', '']
    w1 = weightList[k]
    w2 = 1 - weightList[k]
    for i in range(len(taskList)):
        tmp = []
        for j in range(len(nodeList)):
            val = timeSum[j] * w1 + taskList[i] / nodeList[j] * w2
            tmp.append(val)
        min = 10000
        minIndex = 0
        for x in range(len(tmp)):
            if tmp[x] < min:
                min = tmp[x]
                minIndex = x
        timeSum[minIndex] += taskList[i] / nodeList[minIndex]
        taskArr[minIndex] = taskArr[minIndex] + str(i) + ','
    print('权重w1', weightList[k])
    print('调度队列', taskArr)
    print('每个耗时', timeSum)