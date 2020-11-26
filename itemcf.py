#!/usr/bin/env python
# coding: utf-8

# Item Based Collaborative Filtering

# 这里的物品指电影

import sys
import random
import math
import os
from operator import itemgetter
from collections import defaultdict

# 使得随机数据可预测，即只要seed的值一样，后续生成的随机数都一样
random.seed(0)


class ItemBasedCF(object):
    ''' TopN recommendation - Item Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_movie = 20  # n_sim_movie: top 20个电影
        self.n_rec_movie = 10  # n_rec_movie: top 10个推荐结果

        self.movie_sim_mat = {}  # movie_sim_mat: 电影之间的相似度
        self.movie_popular = {}  # movie_popular: 电影的出现次数
        self.movie_count = 0  # movie_count: 总电影数量

        print('Similar movie number = %d' % self.n_sim_movie, file=sys.stderr)  # sys.stderr 目的就是返回错误信息
        print('Recommended movie number = %d' % self.n_rec_movie, file=sys.stderr)

    # ============================================================================================================================ #
    # 某用户对某物品的喜欢程度（训练集、测试集），即：
    # {'某user': {'某item': score, '某item': score, ...}, '某user': {'某item': score, '某item': score, ...}}
    # ============================================================================================================================ #
    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for i, line in enumerate(fp):
            # yield和return的关系和区别：
            # return在程序中返回某个值，返回之后程序就不再往下运行了。
            # 带yield的函数是一个生成器，而不是一个函数，这个生成器有一个函数就是next函数，next就相当于“下一步”生成哪个数，
            # 这一次的next开始的地方是接着上一次的next停止的地方执行的，然后遇到yield后，return出要生成的数，此步就结束。
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
                # dict.setdefault(key, default=None), 如果键不存在于字典中，将会添加键并将值设为默认值
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
    
    # 物品间的相关矩阵（两物品被多少个用户同时看过），即：
    # {'某item': {'某item': frequency, '某item': frequency, ...}, '某item': {'某item': frequency, '某item': frequency, ...}, ...}
    
    # 物品间的相似矩阵（两物品被多少个用户同时看过 / math.sqrt(一个物品被多少用户观看 * 另一个物品被多少用户观看)），即：
    # {'某item': {'某item': similarity, '某item': similarity, ...}, '某item': {'某item': similarity, '某item': similarity, ...}, ...}
    # ============================================================================================================================ #
    def calc_movie_sim(self):
        ''' calculate movie similarity matrix '''
        print('counting movies number and popularity...', file=sys.stderr)

        for user, movies in self.trainset.items():  # dict.items() 以列表返回可遍历的(键, 值) 元组数组
            for movie in movies:
                # count item popularity
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        print('count movies number and popularity succ', file=sys.stderr)

        # save the total number of movies
        self.movie_count = len(self.movie_popular)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated users between items
        itemsim_mat = self.movie_sim_mat
        print('building co-rated users matrix...', file=sys.stderr)

        for user, movies in self.trainset.items():
            for m1 in movies:
                itemsim_mat.setdefault(m1, defaultdict(int))
                for m2 in movies:
                    if m1 == m2:
                        continue
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print('calculating movie similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.items():
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating movie similarity factor(%d)' %
                          simfactor_count, file=sys.stderr)

        print('calculate movie similarity matrix(similarity factor) succ',
              file=sys.stderr)
        print('Total similarity factor number = %d' %
              simfactor_count, file=sys.stderr)

    # ============================================================================================================================ #
    # 相似物品（相似度高的前 K 个物品），即：
    # {'某item': similarity, '某item': similarity, ...}
    
    # 推荐前 N 个物品（用户对某物品的喜欢程度 = 所有相似用户中每个看过该物品的用户的相似度之和），即：
    # [('某item', score), ('某item', score), ...]
    # 推荐前 N 个物品（用户对某物品的喜欢程度 = 所有相似物品的 相似度*喜欢程度 之和），即：
    # [('某item', score), ('某item', score), ...]
    # ============================================================================================================================ #
    def recommend(self, user):
        ''' Find K similar movies and recommend N movies '''
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainset[user]

        for movie, rating in watched_movies.items():
            # itemgetter函数用于获取对象的哪些维的数据
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating
                
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    # ============================================================================================================================ #
    # 评估 准确率、召回率、覆盖率、流行度，即：
    # 准确率 = 所有用户的测试集和推荐集相同数 / 所有用户的推荐数
    # 召回率 = 所有用户的测试集和推荐集相同数 / 所有用户的测试数
    # 覆盖率 = 所有用户的推荐物品的种类 / 总物品种类
    # 流行度 = 所有(math.log(1 + 某物品的流行度)) / 所有用户的推荐数
    # ============================================================================================================================ #
    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0  # hit 命中(所有用户的测试集和推荐集相同数)
        rec_count = 0  # rec_count 所有用户的推荐数
        test_count = 0  # test_count 所有用户测试数
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        # compare the differences between test set and recommendation set
        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
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

    # 创建ItemCF对象
    itemcf = ItemBasedCF()
    
    # 将数据按照 7:3的比例，拆分成：训练集和测试集，存储在itemcf的trainset和testset中
    ratingfile = os.path.join('ml-1m', 'ratings.dat')
    itemcf.generate_dataset(ratingfile)

    # 计算电影之间的相似度
    itemcf.calc_movie_sim()
    
    # 评估推荐效果
    itemcf.evaluate()

    # 查看某用户的推荐结果和测试结果
    user = "2"
    print("推荐结果", itemcf.recommend(user))
    print("测试结果", itemcf.testset.get(user, {}))
