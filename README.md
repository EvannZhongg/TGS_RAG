# TGS_RAG

## 依赖安装
```
pip install 'docling[pdf]'    
docling-tools models download -o D:\Personal_Project\BiRAG\model
```


## 第一步：初始化数据库
在运行主程序 TGSRAG.py 之前，运行一次初始化命令：
```
python manage_db.py init
```
## 第二步：运行主程序
现在你可以正常运行主程序了，它会自动检测到数据库并在其中建立表结构：
```
python TGSRAG.py
```

## 第三步：(如果需要) 删除某个知识库
如果你想清空 my_electronics_kb 的数据重新跑，可以使用：
```
python manage_db.py delete my_kb
```
这会执行 DROP SCHEMA my_electronics_kb CASCADE，彻底清除该知识库下的所有表和数据，但不影响其他知识库。

## 数据库搜索
1. 基础用法（搜索所有表）
直接在命令行输入关键词：
```
python search_kb.py search "刘备"
```

2. 指定搜索范围
如果你只想查包含该词的实体：
```
python search_kb.py search "刘备" --scope entities
```
3. 如果想查关系和文本块（不查实体）：
```
python search_kb.py search "刘备" --scope relations,chunks
```

4. 查看有哪些知识库
```
python search_kb.py list
```

5. 帮助菜单 ，查看可用参数：
```
python search_kb.py -h
```
