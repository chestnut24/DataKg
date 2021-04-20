**本方法使用规则：**

order_by:   

end_ location_size_load_clock

端口-故障圈位置-故障深度-载荷-故障点钟



**规则：**

① 提供使用的特征

② 特征按照特定顺序排列，次级结点继承上级结点，并使用fault结点作为第零级进行汇总，举例：

假设共三个特征，分别是end，location，size

* 零级节点：fault

* 一级结点：end

* 二级结点：end_ location

* 三级结点：end_ location_size

③ 最后一级结点作为【特征节点（feature）】，之前层级的节点作为【故障节点（fault）】，相邻层级节点的关系是【属于（belong）】，举例：

* 故障结点：(end_ location, fault)

* 特征节点：(end_ location_size, feature)
* 关系：
  * (end_ location_size, belong, end_ location)
  
  * (end_ location belong, end)
  
  * (end, belong, fault)
  
    

**后期改进设想：**

将 fault 节点划分更细，fault_level1, fault_level2等