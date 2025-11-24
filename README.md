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
python search_kb.py "刘备"
```
输出示例：
```
🔍 正在知识库 '四大名著' (Schema: ...) 中搜索: '刘备'
...
[🧩 Entities / 实体]
  • ID: ent-xxxx
    名称: 刘备 (Person)
    描述: 字玄德，汉中山靖王刘胜之后...
...
[📄 Chunks / 文本块]
  • ID: chunk_123
    来源: 三国演义.txt
    内容摘要: ...桃园三结义，刘备、关羽、张飞三人...
```

2. 指定搜索范围
如果你只想查包含该词的实体：
```
python search_kb.py "关羽" --scope entities
```
如果你想查关系和文本块（不查实体）：
```
python search_kb.py "结义" --scope relations,chunks
```
3. 帮助菜单 ，查看可用参数：
```
python search_kb.py -h

```
