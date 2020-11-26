#!/usr/bin/env python
# coding: utf-8

# User Based Collaborative Filtering

# 这里的物品指电影

import sys
import random
import math
import os
from operator import itemgetter
from collections import defaultdict

# 使得随机数据可预测，即只要seed的值一样，后续生成的随机数都一样
random.seed(0)


class UserBasedCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_user = 20  # n_sim_user: top 20个用户
        self.n_rec_movie = 10  # n_rec_movie: top 10个推荐结果

        self.user_sim_mat = {}  # user_sim_mat: 用户之间的相似度
        self.movie_popular = {}  # movie_popular: 电影的出现次数
        self.movie_count = 0  # movie_count: 总电影数量

        print ('Similar user number = %d' % self.n_sim_user, file=sys.stderr)
        print ('recommended movie number = %d' % self.n_rec_movie, file=sys.stderr)
        
    # ============================================================================================================================ #
    # 某用户对某物品的喜欢程度（训练集、测试集），即：
    # {'某user': {'某item': score, '某item': score, ...}, '某user': {'某item': score, '某item': score, ...}}
    # ============================================================================================================================ #
    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print ('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print ('load %s succ' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating, _ = line.split('::')
            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print ('split training set and test set succ', file=sys.stderr)
        print ('train set = %s' % trainset_len, file=sys.stderr)
        print ('test set = %s' % testset_len, file=sys.stderr)

    # ============================================================================================================================ #
    # 物品的流行度（某物品被多少用户观看），即：
    # {'某item': popularity, '某item': popularity, ...}
    
    # 某物品被哪些用户观看，即：
    # {'某item': {'某user', '某user', ...}, '某item': {'某user', '某user', ...}, ...}
    
    # 用户间的相关矩阵（俩用户共同看过多少个物品），即：
    # {'某user': {'某user': frequency, '某user': frequency, ...}, '某user': {'某user': frequency, '某user': frequency, ...}, ...}
    
    # 用户间的相似矩阵（俩用户共同看过多少个物品 / math.sqrt(一个用户看过多少个物品 * 另一个用户看过多少个物品)），即：
    # {'某user': {'某user': similarity, '某user': similarity, ...}, '某user': {'某user': similarity, '某user': similarity, ...}, ...}
    # ============================================================================================================================ #
    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users（key=itemID, value=list of userIDs who have seen this movie）
        print ('building movie-users inverse table...', file=sys.stderr)
        movie2users = dict()

        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)
                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        print ('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print ('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print ('building user co-rated movies matrix...', file=sys.stderr)

        for movie, users in movie2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1
        print ('build user co-rated movies matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print ('calculating user similarity factor(%d)' %
                           simfactor_count, file=sys.stderr)

        print ('calculate user similarity matrix(similarity factor) succ',
               file=sys.stderr)
        print ('Total similarity factor number = %d' %
               simfactor_count, file=sys.stderr)

    # ============================================================================================================================ #
    # 相似用户（相似度高的前 K 个用户），即：
    # {'某user': similarity, '某user': similarity, ...}
    
    # 推荐前 N 个物品（用户对某物品的喜欢程度 = 所有看过该物品的相似用户的 相似度 之和），即：
    # [('某item', score), ('某item', score), ...]
    # ============================================================================================================================ #
    def recommend(self, user):
        ''' Find K similar users and recommend N movies '''
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = dict()
        watched_movies = self.trainset[user]

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainset[similar_user]:
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += similarity_factor
                
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    # ============================================================================================================================ #
    # 评估 准确率、召回率、覆盖率、流行度，即：
    # 准确率 = 所有用户的测试集和推荐集相同数 / 所有用户的推荐数
    # 召回率 = 所有用户的测试集和推荐集相同数 / 所有用户的测试数
    # 覆盖率 = 所有用户的推荐物品的种类 / 总物品种类
    # 流行度 = 所有(math.log(1 + 某物品的流行度)) / 所有用户的推荐数
    # ============================================================================================================================ #
    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print ('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        # varables for precision and recall
        hit = 0  # hit 命中(所有用户的测试集和推荐集相同数)
        rec_count = 0  # rec_count 所有用户的推荐数
        test_count = 0  # test_count 所有用户测试数
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            # compare the differences between test set and recommendation set
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity), file=sys.stderr)
        

if __name__ == '__main__':
    
    # 创建UserCF对象
    usercf = UserBasedCF()
    
    # 将数据按照 7:3的比例，拆分成：训练集和测试集，存储在usercf的trainset和testset中
    ratingfile = os.path.join('ml-1m', 'ratings.dat')
    usercf.generate_dataset(ratingfile)
    
    # 计算用户之间的相似度
    usercf.calc_user_sim()
    
    # 评估推荐效果
    usercf.evaluate()

    # 查看某用户的推荐结果和测试结果
    user = "2"
    print("推荐结果", usercf.recommend(user))
    print("测试结果", usercf.testset.get(user, {}))
