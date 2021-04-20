# 用于将保存的模型结果进行可视化，并进行对比，将文件地址添加到pathList，将图例名称添加到nameList
# plot_accuracy是对比准确率，acc是train，val_acc是verify；plot_loss是对比损失
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

color_list = ['red', 'cyan', 'lime', 'mediumslateblue', 'lightcoral', 'lightcoral', 'palegoldenrod']  # 曲线颜色列表
bot_list = ['s', 'o', '^', 'x', '*', 's']  # 点形状列表
type_list = ['-', '--', '-.', ':', '--', '-', '-.']  # 折线形状列表


# 折线图形状
# 's' : 方块状
# 'o' : 实心圆
# '^' : 正三角形
# 'v' : 反正三角形
# '+' : 加好
# '*' : 星号
# 'x' : x号
# 'p' : 五角星
# '1' : 三脚架标记
# '2' : 三脚架标记

# 折线图类型
# -      实线(solid)
# --     短线(dashed)
# -.     短点相间线(dashdot)
# ：    虚点线(dotted)
# '', ' ', None


def plot_accuracy(history, nameList, savePath):
    step = 4

    y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为0.1的倍数

    for index, item in enumerate(history):
        # 折线图
        acc_choose = []  # 从中挑选出十个点
        x = []  # x轴坐标值
        print('acc:', item['val_acc'])
        for i in range(0, 70, step):  # 为4时效果看起来最好
            acc_choose.append(item['val_acc'][i])
            x.append(i)
        # print(acc_choose)
        # for i in x:  # 曲线上加数字
        #     # print(acc_choose[int(i / 5)])
        #     plt.text(i, acc_choose[int(i / step)] + 0.01, '%.4f' % acc_choose[int(i / step)], ha='center', va='bottom', fontsize=9)
        plt.plot(x, acc_choose, marker=bot_list[index], ls=type_list[index], color=color_list[index])

        # 曲线图
        # plt.plot(item['val_acc'], color=color_list[index])

    # CNN-Att-BiGRU
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    # plt.legend(['train', 'verify'], loc='upper left')
    plt.legend(nameList, loc='lower right')
    plt.savefig(savePath, bbox_inches='tight')  # 后面那句话是让保存的图片变得紧凑
    plt.show()


def plot_loss(history, nameList, savePath):
# def plot_loss(history, nameList):
    for index, item in enumerate(history):
        step = 4
        # 折线图
        loss_choose = []  # 从中挑选出十个点
        x = []  # x轴坐标值
        print('loss:', item['val_loss'])
        for i in range(0, 70, step):
            loss_choose.append(item['val_loss'][i])
            x.append(i)
        plt.plot(x, loss_choose, marker=bot_list[index], ls=type_list[index], color=color_list[index])
        # # 曲线图
        # plt.plot(item['loss'], color=color_list[index])
    # plt.plot(history['val_loss'])


    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    plt.legend(nameList, loc='upper right')  # legend是图例
    plt.savefig(savePath, bbox_inches='tight')  # 后面那句话是让保存的图片变得紧凑
    plt.show()


'''
path1 = r'./model_file/history_model/cnn_双层BiGRU_attention.txt'
path2 = r'./model_file/history_model/cnn_双层BiLSTM_attention.txt'
path3 = r'./model_file/history_model/双层BiGRU_attention.txt'
# path4 = r'./model_file/history_model/cnn_双层BiGRU.txt'
path5 = r'./model_file/history_model/cnn_attention_单层BiGRU.txt'
path6 = r'./model_file/history_model/单层LSTM.txt'
# 对比图
pathList = [path1, path2, path3, path5, path6]
nameList = ['CNN-Att-BiGRU', 'CNN + 双层Bi-LSTM + attention', '双层BiGRU + attention', 'CNN + 单层Bi-GRU + attention', '单层LSTM']
'''


def explore_filename(dir):  # 获取目录下所有文件名
    filename = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            filename.append(file[:-4])
    return filename


# 遍历文件夹所有文件进行展示
pathList = []
# filepath = r'./model_file/result_model_by_lstm'  # 5种分类 LSTM 标签 LSTM
# filepath = r'./model_file/result_model_by_proLSTM'  # 5种分类 OBiLSTM 标签 OBiLSTM
filepath = r'./model_file/history_model_97kinds_70epochs'  # 97分=类 OBiLSTM 标签 OBiLSTM
# filepath = r'./model_file/70epoch'
nameList = explore_filename(filepath)
for name in nameList:
    pathList.append(filepath + '/' + name + '.txt')

print(pathList)


# 单个曲线
# pathList = [path1]
# nameList = ['CNN-Att-BiGRU']
'''
# 整个文件夹遍历
history = []
for path in pathList:
    with open(path, 'rb') as file_pi:
        # history = pickle.load(file_pi)
        history.append(pickle.load(file_pi))

# plot_accuracy()

# accSavePath = r'./model_file/result_picture/prolstm_70epochs_4kinds.png'
accSavePath = r'./model_file/result_picture/lstm_70epochs_4kinds.png'
plot_accuracy(history, nameList, accSavePath)
plot_loss(history, nameList)

# plot_loss()
# plot_loss(history)
'''
# 指定地址进行展示
path1 = r'./model_file/history_model_97kinds_70epochs/OBiLSTM.txt'
path2 = r'./model_file/history_model_97kinds_70epochs/CNN+LSTM.txt'
path3 = r'./model_file/history_model_97kinds_70epochs/LSTM.txt'
# 对比图
pathList = [path1, path2, path3]
nameList = ['OBiLSTM', 'CNN+LSTM', 'LSTM']
history = []
for path in pathList:
    with open(path, 'rb') as file_pi:
        # history = pickle.load(file_pi)
        history.append(pickle.load(file_pi))
accSavePath = r'./model_file/result_picture/acc_70epochs_97.png'
lossSavePath = r'./model_file/result_picture/loss_70epochs_97.png'
plot_accuracy(history, nameList, accSavePath)
plot_loss(history, nameList, lossSavePath)

