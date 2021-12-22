# 这是用来看看预测和真实的区别
# compare_pre_test是用来处理y_pred和y_test
# y_pred是预测文件，输出的是预测的标签数字
# y_test是真实测试数据，是独热编码，处理后变成y_test_label，输出的是真实标签数字
# diff是两者的差距，是数字形式
# data_index_origin 是载入数据原始状态，经过处理后变为data_index，其中是标签名称，用来跟标签数字对应
