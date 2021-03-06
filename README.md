# RecommenderSystem_userCF-itemCF-CB

## 1、项目目标
基于 Movielens 1M 数据集分别实现了 User Based Collaborative Filtering（以下简称UserCF）和 Item Based Collaborative Filtering（以下简称ItemCF）两个算法。

## 2、主要技术
基于用户的协同过滤（UserCF）/基于物品的协同过滤（ItemCF）

## 3、数据准备
下载 Movielens 1M 数据集 [ml-1m.zip](http://files.grouplens.org/datasets/movielens/ml-1m.zip)，并解压到项目 RecommenderSystem_userCF-itemCF-CB 文件夹下，即可得到。

## 4、环境
Python3

## 5、模块说明

    ├─data ：运行 CB.ipynb 需要的数据
    ├─ml-1m ：运行 itemCF.ipynb 和 userCF.ipynb 需要的数据
    ├─CB.ipynb ：基于内容的推荐算法的实现
    ├─itemCF.ipynb ：基于物品的协同过滤的推荐算法的实现
    ├─userCF.ipynb ：基于用户的协同过滤的推荐算法的实现
    └─README.md ：基本描述

## 6、算法对比

**基于用户的协同过滤算法：**

    步骤：
    1) 找到和目标用户兴趣相似的用户集合。 
    2) 找到这个集合中的用户喜欢的，且目标用户没有听说过的物品推荐给目标用户。 

    缺点：
    1) UserCF需要维护一个用户相似度的矩阵，随着用户数目增多，维护用户兴趣相似度矩阵的代价越大。计算用户兴趣相似度矩阵的运算时间复杂度和空间复杂度的增长和用户数的增长近似于平方关系。
    2) 基于用户的协同过滤很难对推荐结果作出解释。

    实用场景 —— 新闻推荐：
    1）个性化新闻推荐更加强调抓住新闻热点，热门程度和时效性是个性化新闻推荐的重点，而个性化相对于这两点略显次要。
    2）新闻的更新非常快，物品相关度的表也需要很快更新，虽UserCF对于新用户也需要更新相似度表，但在新闻网站中，物品的更新速度远远快于新用户的加入速度，而且对于新用户，完全可以给他推荐最热门的新闻。


**基于物品的协同过滤算法：**

    步骤：
    1) 计算物品之间的相似度。 
    2) 根据物品的相似度和用户的历史行为给用户生成推荐列表。 

    缺点：
    1) ItemCF需要维护一个物品相似度矩阵，随着物品数目增多，维护物品相似度矩阵的代价越大。

    实用场景 —— 图书、电子商务和电影网站：
    1）在这些网站中，用户的兴趣是比较固定和持久的。这些系统中的用户大都不太需要流行度来辅助他们判断一个物品的好坏，而是可以通过自己熟悉领域的知识自己判断物品的质量。
    2）这些网站的物品更新速度不会特别快，一天一次更新物品相似度矩阵对它们来说不会造成太大的损失，是可以接受的。 

**本案例：**

    UserCF算法中，由于用户数量多，生成的相似性矩阵也大，会占用比较多的内存，不过一般电脑都没问题。
    ItemCF算法中，每次推荐都需要找出一个用户的所有电影，再为每一部电影找出最相似的电影，运算量比UserCF大，因此推荐的过程比较慢。

## 7、温馨提示
python 基础不是很好的话，建议先看 itemcf.ipynb ，其中有详细注释。
