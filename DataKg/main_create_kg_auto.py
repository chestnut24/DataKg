# 自动创建kg，不需要挨个点调用

import os

os.system('python create_import_csv.py')  # 初始化csv文件，构建图谱的节点、关系等
os.system('python py2neo_initialize.py')  # 初始化图谱数据库，将原始图谱清空，并新建一个
# os.system('python ../DataProcessing/LeePY/print.py')  # 不在同一个文件夹，使用相对路径（或者绝对路径）

