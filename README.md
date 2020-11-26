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

    ├─.ipynb_checkpoints ：Jupyter Notebook 中运行代码时自动生成的文件夹
    ├─data ：运行 CB.ipynb 需要的数据
    ├─ml-1m ：运行 itemCF.ipynb 和 userCF.ipynb 需要的数据
    ├─CB.ipynb ：基于内容的推荐算法的实现
    ├─itemCF.ipynb ：基于物品的协同过滤的推荐算法的实现
    ├─userCF.ipynb ：基于用户的协同过滤的推荐算法的实现
    └─README.md ：基本描述

## 6、算法对比
UserCF算法中，由于用户数量多，生成的相似性矩阵也大，会占用比较多的内存，不过一般电脑都没问题。

ItemCF算法中，每次推荐都需要找出一个用户的所有电影，再为每一部电影找出最相似的电影，运算量比UserCF大，因此推荐的过程比较慢。

## 7、温馨提示
python 基础不是很好的话，建议先看 itemcf.ipynb ，其中有详细注释。
