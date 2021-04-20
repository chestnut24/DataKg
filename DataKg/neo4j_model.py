from py2neo import Graph, Node, Relationship


# 版本说明：Py2neo v4
class Neo4j_Operate:
    graph = None
    matcher = None

    def _init_(self):
        print("Neo4j Init...")

    # 连接数据库，需打开neo4j网页
    def connect_db(self):
        self.graph = Graph("bolt://localhost:7687", username="neo4j", password="306636")
        print("success")

    # 删除所有节点及关系
    def delete_all(self):
        self.graph.run("match (a)-[r]-(b) delete r")
        self.graph.run("match (n) delete n")
        print("delete success")

    # 按照节点名字进行查询
    def search_node_name(self, value):
        answer = self.graph.run(cypher="match (n{name:'" + value + "'}) return n").to_ndarray()
        return answer

    # 查询所有节点及属性
    def search_all(self):
        answer = self.graph.run(cypher="match (n) return n").to_ndarray()
        return answer

    # 通过name查询某个节点的belonsto关系 的 父结点
    def search_father_node(self, name):
        answer = self.graph.run("match (n1{name: '" + name + "'})-[r:belongsto]->(n2) return n2 ").to_ndarray()
        return answer

    # 通过name查询某个节点的belonsto关系 的 所有子节点
    def search_child_node(self, name):
        answer = self.graph.run("match (n1)-[r:belongsto]->(n2{name: '" + name + "'}) return n1 ").to_ndarray()
        return answer

    # 从csv文件load 普通节点
    def add_node_csv(self, file):
        answer = self.graph.run("LOAD CSV WITH HEADERS  FROM 'file:///" + file + ".csv' AS line "
                                                                                 "MERGE (node:" + file + "{ "
                                                                                                         "nodeID:line.nodeID, "
                                                                                                         "name:line.name, "
                                                                                                         "type:line.type, "
                                                                                                         "describe:line.describe })"
                                                                                                         "return node").data()
        return answer

    # 从csv文件load feature节点
    def add_feature_csv(self, file):
        answer = self.graph.run("LOAD CSV WITH HEADERS  FROM 'file:///" + file + ".csv' AS line "
                                                                                 "MERGE (node:" + file + "{ "
                                                                                                         "nodeID:line.nodeID, "
                                                                                                         "name:line.name, "
                                                                                                         "type:line.type, "
                                                                                                         "describe:line.describe,"
                                                                                                         "DE_min: line.DE_min,"
                                                                                                         "DE_max: line.DE_max,"
                                                                                                         "DE_mean: line.DE_mean,"
                                                                                                         "DE_std: line.DE_std,"
                                                                                                         "FE_min: line.FE_min,"
                                                                                                         "FE_max: line.FE_max,"
                                                                                                         "FE_mean: line.FE_mean,"
                                                                                                         "FE_std: line.FE_std,"
                                                                                                         "BA_min: line.BA_min,"
                                                                                                         "BA_max: line.BA_max,"
                                                                                                         "BA_mean: line.BA_mean,"
                                                                                                         "BA_std: line.BA_std })"
                                                                                                         "return node").data()
        return answer

    # 从csv文件load单个关系
    def add_relationship_csv(self, file):
        answer = self.graph.run("LOAD CSV WITH HEADERS FROM 'file:///" + file + ".csv' AS line "
                                                                                "match (start{nodeID:line.START_ID}),(end{nodeID:line.END_ID}) "
                                                                                "merge (start)-[r:" + file + "{relationship:line.relationship}]->(end)").data()
        return answer

    # 从csv文件load函数关系
    def add_function_cs(self, file):
        answer = self.graph.run("LOAD CSV WITH HEADERS FROM 'file:///" + file + ".csv' AS line "
                                                                                "match (start{nodeID:line.START_ID}),(end{nodeID:line.END_ID}) "
                                                                                "merge (start)-[r:" + file + "{relationship:line.relationship,down:line.down,up:line.up}]->(end)").data()
        return answer

    # 插入普通故障节点
    # def add_fault_node(self, label, nodeID, name, type, describe):
    def add_fault_node(self, label, name, type, describe):
        answer = self.graph.run(
            "MERGE (n: " + str(label) + " {"
            # "nodeID: " + str(nodeID) + ","  # 数值类型不需要单引号
                                        "name: '" + str(name) + "',"  # 字符串类型多加一个单引号
                                                                "type: '" + str(type) + "',"
                                                                                        "describe: '" + str(
                describe) + "'"
                            "})"
        ).data()
        return answer

    # 插入带有BA_max等特征值的特征节点
    # def add_feature_node(self, label, nodeID, name, type, describe, DE_min, DE_max, DE_mean, DE_std, FE_min, FE_max,
    #                      FE_mean, FE_std, BA_min, BA_max, BA_mean, BA_std):
    def add_feature_node(self, label, name, type, describe, DE_min, DE_max, DE_mean, DE_std, FE_min, FE_max,
                         FE_mean, FE_std, BA_min, BA_max, BA_mean, BA_std):
        answer = self.graph.run(
            "MERGE (n: " + str(label) + " {"
            # "nodeID: " + str(nodeID) + ","  # 数值类型不需要单引号
                                        "name: '" + str(name) + "',"  # 字符串类型多加一个单引号
                                                                "type: '" + str(type) + "',"
                                                                                        "describe: '" + str(
                describe) + "',"
                            "DE_min: " + str(DE_min) + ","
                                                       "DE_max: " + str(DE_max) + ","
                                                                                  "DE_mean: " + str(DE_mean) + ","
                                                                                                               "DE_std: " + str(
                DE_std) + ","
                          "FE_min: " + str(FE_min) + ","
                                                     "FE_max: " + str(FE_max) + ","
                                                                                "FE_mean: " + str(FE_mean) + ","
                                                                                                             "FE_std: " + str(
                FE_std) + ","
                          "BA_min: " + str(BA_min) + ","
                                                     "BA_max: " + str(BA_max) + ","
                                                                                "BA_mean: " + str(BA_mean) + ","
                                                                                                             "BA_std: " + str(
                BA_std) + ""
                          "})"
        ).data()
        return answer

    # 插入单一关系
    def add_single_relation(self, relationship, startName, startLabel, endName, endLabel, type):
        answer = self.graph.run(
            "match (n1: " + str(startLabel) + "), (n2: " + str(endLabel) + ") "
                                                                           "where n1.name = '" + str(
                startName) + "' and n2.name = '" + str(endName) + "' "
                                                                  "merge (n1)-[r1: " + str(
                relationship) + " {type: '" + str(type) + "'}]->(n2) "
        ).data()
        return answer

    # 按条件查询feature节点
    def search_feature_node(self, thresholdList, valueList):
        # thresholdList是阈值列表（min, max, mean, std有着不同的阈值），valueList是传递进来的查询值列表
        # DE_min, DE_max, DE_mean, DE_std, FE_min, FE_max, FE_mean, FE_std, BA_min, BA_max, BA_mean, BA_std
        answer = self.graph.run(
            " match (n:feature)"
            " where "
            " n.DE_min > " + str(valueList[0] - thresholdList[0]) + " and n.DE_min < " + str(
                valueList[0] + thresholdList[0]) + ""
                                                   " and n.DE_max > " + str(
                valueList[1] - thresholdList[1]) + " and n.DE_max < " + str(valueList[1] + thresholdList[1]) + ""
                                                                                                               " and n.DE_mean > " + str(
                valueList[2] - thresholdList[2]) + " and n.DE_mean < " + str(valueList[2] + thresholdList[2]) + ""
                                                                                                                " and n.DE_std > " + str(
                valueList[3] - thresholdList[3]) + " and n.DE_std < " + str(valueList[3] + thresholdList[3]) + ""
                                                                                                               " and n.FE_min > " + str(
                valueList[4] - thresholdList[0]) + " and n.FE_min < " + str(valueList[4] + thresholdList[0]) + ""
                                                                                                               " and n.FE_max > " + str(
                valueList[5] - thresholdList[1]) + " and n.FE_max < " + str(valueList[5] + thresholdList[1]) + ""
                                                                                                               " and n.FE_mean > " + str(
                valueList[6] - thresholdList[2]) + " and n.FE_mean < " + str(valueList[6] + thresholdList[2]) + ""
                                                                                                                " and n.FE_std > " + str(
                valueList[7] - thresholdList[3]) + " and n.FE_std < " + str(valueList[7] + thresholdList[3]) + ""
                                                                                                               " and n.BA_min > " + str(
                valueList[8] - thresholdList[0]) + " and n.BA_min < " + str(valueList[8] + thresholdList[0]) + ""
                                                                                                               " and n.BA_max > " + str(
                valueList[9] - thresholdList[1]) + " and n.BA_max < " + str(valueList[9] + thresholdList[1]) + ""
                                                                                                               " and n.BA_mean > " + str(
                valueList[10] - thresholdList[2]) + " and n.BA_mean < " + str(valueList[10] + thresholdList[2]) + ""
                                                                                                                  " and n.BA_std > " + str(
                valueList[11] - thresholdList[3]) + " and n.BA_std < " + str(valueList[11] + thresholdList[3]) + ""
                                                                                                                 " return n"
        ).to_ndarray()
        return answer
