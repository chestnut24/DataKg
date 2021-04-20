# 从neo4j的import中通过load函数导入数据，缺点是导入的值均为字符，没有浮点数无法表达数值，因此弃用，改用create和merge创造节点

import DataKg.neo4j_model as neo
# 节点列表 带有三个属性 包含nodeID,name,type
# nodeList = ['device', 'class', 'degree', 'feature', 'position', 'state']
nodeList = ['fault']
# feature节点列表 带有DE_min等属性
featureList = ['feature']
# 关系列表 只有关系名 无属性
relationshipList = ['belongs']
# 函数关系列表 有属性名以及属性
# functionList = ['function']
# 连接数据库
graph = neo.Neo4j_Operate()
graph.connect_db()
# load节点
for file in nodeList:
    graph.add_node_csv(file)
# load feature
for file in featureList:
    graph.add_feature_csv(file)
# load单个关系
for file in relationshipList:
    graph.add_relationship_csv(file)
# # load函数关系
# for file in functionList:
#     graph.add_function_csv(file)
print("all node:", graph.search_all())
