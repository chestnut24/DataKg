# from DataKg.neo4j_model import Neo4j_Operate
import DataKg.neo4j_model as neo

# 连接
neo = neo.Neo4j_Operate()
neo.connect_db()
print('--Neo4j connecting--')
# 清空数据库
neo.delete_all()
