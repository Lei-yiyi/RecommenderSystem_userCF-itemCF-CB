#!/usr/bin/env python
# coding: utf-8

# 基于内容的推荐算法

# 这里的物品指节目

import pandas as pd
import numpy as np
import math


# -------------------------------------- 用户看过的物品的 物品画像 -------------------------------------- #   
def createLabelMatrix():
    '''
    创建用户看过的物品的标签矩阵、对应的物品id
    
    saw_itemsLabel：用户看过的物品的标签矩阵，[[0 0 0 ...] [...] ...]
    saw_itemsID_itemsLabel：对应的物品id，['某item', '某item', '某item', ...]
    labels：对应的标签，['label1', 'label2', ...]
    '''
    
    df = pd.read_csv("data/所有用户看过的节目及所属类型的01矩阵.csv")
    
    saw_itemsLabel = np.array(df.iloc[:, 1:])
    saw_itemsID_itemsLabel = np.array(df.iloc[:, 0]).tolist()
    labels = df.columns[1:].tolist()
    
    return saw_itemsLabel, saw_itemsID_itemsLabel, labels

 
def createItemsProfiles(saw_itemsLabel, saw_itemsID_itemsLabel, labels) :
    '''
    创建用户看过的物品的物品画像
    
    saw_itemsLabel: 用户看过的物品的标签矩阵，[[0 0 0 ...] [...] ...]
    saw_itemsID_itemsLabel：对应的物品id，['某item', '某item', '某item', ...]
    labels：对应的标签，['label1', 'label2', ...]
    
    saw_itemsProfiles：用户看过的物品的物品画像，{'某item':{'label1':frequency, 'label2': frequency, ...}, '某item':{...}...}
    '''
    
    saw_itemsProfiles = {}
    
    for i in range(len(saw_itemsID_itemsLabel)):

        saw_itemsProfiles[saw_itemsID_itemsLabel[i]] = {}

        for j in range(len(labels)):
            saw_itemsProfiles[saw_itemsID_itemsLabel[i]][labels[j]] = saw_itemsLabel[i][j]

    return saw_itemsProfiles


# --------------------------------------- 看过物品的用户的 用户画像 ------------------------------------- # 
def createRatingMatrix():
    '''
    创建用户对其看过的物品的评分矩阵、对应的用户id、对应的物品id
    
    saw_itemsRating：用户对其看过的物品的评分矩阵，[[score score score ...] [...] ...]
    usersID：对应的用户id，['某user', '某user', ...]
    saw_itemsID_itemsRating：对应的物品id，['某item', '某item', '某item', ...]
    '''  

    df = pd.read_csv("data/所有用户对其看过的节目的评分矩阵.csv")

    saw_itemsRating = np.array(df.iloc[:, 1:])
    usersID = np.array(df.iloc[:, 0]).tolist()
    saw_itemsID_itemsRating = df.columns[1:].tolist()
    
    return saw_itemsRating, usersID, saw_itemsID_itemsRating


def createUsersProfiles(saw_itemsRating, usersID, saw_itemsID_itemsRating, saw_itemsProfiles, labels):
    '''
    创建用户画像、用户看过的物品
    
    saw_itemsRating：用户对其看过的物品的评分矩阵，[[score score score ...] [...] ...]
    usersID：对应的用户id，['某user', '某user', ...]
    saw_itemsID_itemsRating：对应的物品id，['某item', '某item', '某item', ...]
    saw_itemsProfiles：用户看过的物品的物品画像，{'某item':{'label1':frequency, 'label2': frequency, ...}, '某item':{...}...}
    labels：对应的标签，['label1', 'label2', ...]
    
    saw_usersProfiles：看过物品的用户的用户画像，{'某user':{'label1': frequency, 'label2': frequency, ...}, '某user':{...}...}
    per_saw_itemsID：用户看过的物品（不加入隐性评分信息），{'某user': ['某item', '某item' ...], '某user':[...]...}
    '''

    saw_usersProfiles = {}

    # 计算用户对其看过的物品的平均隐性评分，[score, score, ...]
    # 统计用户看过的物品（不加入隐性评分信息），{'某user': ['某item', '某item' ...], '某user':[...]...}
    # 统计用户对其看过的物品的评分，{'某user':[[某item, score], [某item, score], ...], '某user':[[...], ...], ...}
    users_average_scores_list = []
    per_saw_itemsID = {}
    per_saw_itemsID_scores = {}

    for i in range(len(usersID)):

        per_saw_itemsID_scores[usersID[i]] = []
        per_saw_itemsID[usersID[i]] = []
        count = 0
        sum = 0.0

        for j in range(len(saw_itemsID_itemsRating)):

            # 用户对该物品隐性评分为正，表示真正看过该物品
            if saw_itemsRating[i][j] > 0:
                per_saw_itemsID[usersID[i]].append(saw_itemsID_itemsRating[j])
                per_saw_itemsID_scores[usersID[i]].append([saw_itemsID_itemsRating[j], saw_itemsRating[i][j]])
                count += 1
                sum += saw_itemsRating[i][j]

        if count == 0:
            users_average_scores_list.append(0)
        else:
            users_average_scores_list.append(sum / count)


    # 看过物品的用户的用户画像，{'某user':{'label1': frequency, 'label2': frequency, ...}, '某user':{...}...}
    for i in range(len(usersID)):

        saw_usersProfiles[usersID[i]] = {}

        for j in range(len(labels)):
            count = 0
            score = 0.0

            for item in per_saw_itemsID_scores[usersID[i]]:
                '''
                公式：
                user1_score_to_label1 = Sigma(score_to_itemi - user1_average_score)/items_count
                
                参数：
                user1_score_to_label1：用户user1对于标签label1的隐性评分
                score_to_itemi：用户user1对于其看过的含有标签label1的物品itemi的评分
                user1_average_score：用户user1对其所看过的所有物品的平均评分
                items_count：用户user1看过的物品总数
                '''

                # 该物品含有特定标签labels[j]
                if saw_itemsProfiles[item[0]][labels[j]] > 0:
                    score += (item[1] - users_average_scores_list[i])
                    count += 1

            # 如果求出的值太小，直接置0
            if abs(score) < 1e-6:
                score = 0.0
            if count == 0:
                result = 0.0
            else:
                result = score / count

            saw_usersProfiles[usersID[i]][labels[j]] = result

    return saw_usersProfiles, per_saw_itemsID


# -------------------------------------- 备选物品的 物品画像 -------------------------------------------- #    
def optional_createLabelMatrix():
    '''
    创建备选物品的标签矩阵、对应的物品id
    
    optional_itemsLabel： 备选物品的标签矩阵，[[0 0 0 ...] [...] ...]
    optional_itemsID：对应的物品id，['某item', '某item', '某item', ...]
    '''
    
    df = pd.read_csv("data/备选推荐节目集及所属类型01矩阵.csv")
    
    optional_itemsLabel = np.array(df.iloc[:, 1:])
    optional_itemsID = np.array(df.iloc[:, 0]).tolist()
    
    return optional_itemsLabel, optional_itemsID

 
def optional_createItemsProfiles(optional_itemsLabel, optional_itemsID, labels):
    '''
    创建备选物品的物品画像
    
    optional_itemsLabel： 备选物品的标签矩阵，[[0 0 0 ...] [...] ...]
    optional_itemsID：对应的物品id，['某item', '某item', '某item', ...]
    labels：对应的标签，['label1', 'label2', ...]
    
    optional_itemsProfiles：备选物品的物品画像，{'某item':{'label1':frequency, 'label2': frequency, ...}, '某item':{...}...}
    '''
    
    optional_itemsProfiles = {}
    
    for i in range(len(optional_itemsID)):

        optional_itemsProfiles[optional_itemsID[i]] = {}

        for j in range(len(labels)):
            optional_itemsProfiles[optional_itemsID[i]][labels[j]] = optional_itemsLabel[i][j]

    return optional_itemsProfiles


# ---------------------------------------------- 排序 --------------------------------------------------- # 
def calCosDistance(user, item, labels):
    '''
    计算用户画像向量与物品画像向量的距离（相似度），向量相似度计算公式：cos(user, item) = sigma_ui/sqrt(sigma_u * sigma_i)
    
    user；某一用户的画像，{'label1': frequency, 'label2': frequency, ...}
    item：某一物品的画像，{'label1': frequency, 'label2': frequency, ...}
    labels：对应的标签，['label1', 'label2', ...]
    
    sigma_ui/math.sqrt(sigma_u * sigma_i)：用户画像向量与物品画像向量的距离（相似度）
    '''

    sigma_ui = 0.0
    sigma_u = 0.0
    sigma_i = 0.0

    for label in labels:
        sigma_ui += user[label] * item[label]
        sigma_u += (user[label] * user[label])
        sigma_i += (item[label] * item[label])

    if sigma_u == 0.0 or sigma_i == 0.0:  # 若分母为0，相似度为0
        return 0

    return sigma_ui/math.sqrt(sigma_u * sigma_i)


def contentBased(user_profile, optional_itemsProfiles, optional_itemsID, labels, user_saw_itemsID):
    '''
    排序，借助某个用户的画像和备选推荐物品集的画像，通过计算向量之间的相似度得出备选推荐物品集的排序

    user_profile: 某一用户的画像，{'label1': frequency, 'label2': frequency, ...}
    optional_itemsProfiles: 备选物品的物品画像，{'某item':{'label1':frequency, 'label2': frequency, ...}, '某item':{...}...}
    optional_itemsID: 对应的物品id，['某item', '某item', '某item', ...]
    labels：对应的标签，['label1', 'label2', ...]
    user_saw_itemsID: 某一用户看过的物品, ['某item', '某item' ...]
    
    sorted_results：按相似度降序排列的推荐物品集，[['某item', similarity], [...], ...]
    '''
    
    sorted_results = []

    for i in range(len(optional_itemsID)):
        # 从备选推荐物品集中选择用户没有看过的物品
        if optional_itemsID[i] not in user_saw_itemsID:
            sorted_results.append([optional_itemsID[i], calCosDistance(user_profile, optional_itemsProfiles[optional_itemsID[i]], labels)])

    # 将推荐物品集按相似度降序排列
    sorted_results.sort(key=lambda item: item[1], reverse=True)

    return sorted_results


# ------------------------------------------------ Top N ------------------------------------------------ # 
def printRecommendedItems(sorted_results, max_num):
    '''
    Top N，输出推荐给该用户的物品列表
    
    sorted_results：按相似度降序排列的推荐物品集，[['某item', similarity], [...], ...]
    max_num：最多输出的推荐物品数，3
    '''
    
    count = 0
    for item, degree in sorted_results:
        print("物品id：%s， 推荐指数：%f" % (item, degree))
        count += 1
        if count == max_num:
            break

            
# ----------------------------------------------- 主程序 ------------------------------------------------ # 
if __name__ == '__main__':   

    # ==================================== 用户看过的物品的 物品画像 ==================================== #   

    # 创建用户看过的物品的标签矩阵、对应的物品id
    saw_itemsLabel, saw_itemsID_itemsLabel, labels = createLabelMatrix()
    # 创建用户看过的物品的物品画像
    saw_itemsProfiles = createItemsProfiles(saw_itemsLabel, saw_itemsID_itemsLabel, labels)   

    # ==================================== 看过物品的用户的 用户画像 ==================================== # 

    # 创建用户对其看过的物品的评分矩阵、对应的用户id、对应的物品id
    saw_itemsRating, usersID, saw_itemsID_itemsRating = createRatingMatrix()
    # 创建用户画像、用户看过的物品
    saw_usersProfiles, per_saw_itemsID = createUsersProfiles(saw_itemsRating, usersID, 
                                                             saw_itemsID_itemsRating, saw_itemsProfiles, labels)

    # ======================================= 备选物品的 物品画像 ======================================= #    

    # 创建备选物品的标签矩阵、对应的物品id
    optional_itemsLabel, optional_itemsID = optional_createLabelMatrix()
    # 创建备选物品的物品画像
    optional_itemsProfiles = optional_createItemsProfiles(optional_itemsLabel, optional_itemsID, labels)

    # ============================================== 排序 =============================================== # 

    # 被推荐的用户id
    user = 'A'
    print("对于用户 %s 的推荐物品如下：" % user)
    # 排序
    sorted_results = contentBased(saw_usersProfiles[user], optional_itemsProfiles, 
                                  optional_itemsID, labels, per_saw_itemsID[user])

    # ============================================= Top N =============================================== # 

    # Top N
    printRecommendedItems(sorted_results, 3)
