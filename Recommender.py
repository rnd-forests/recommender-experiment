from __future__ import absolute_import, division, print_function, unicode_literals

import os
import math
import pprint
import datetime
import numpy as np
from timeit import default_timer
from collections import defaultdict
from surprise import GridSearch, Dataset, accuracy, dump


class Recommender:
    def __init__(self, algorithm, param_grid={}, bsl_options={}, sim_options={},
                 data = None, rating_scale=(1, 5), perf_measure='rmse', n_folds=5,
                 trainset_size=0.8, dump_model=True, dump_file_name='knn'):
        self.algorithm = algorithm
        self.param_grid = param_grid
        self.bsl_options = bsl_options
        self.sim_options = sim_options
        self.data = data or self.load_data()
        self.rating_scale = rating_scale
        self.perf_measure = perf_measure
        self.n_folds = n_folds
        self.trainset_size = trainset_size
        self.dump_model = dump_model
        self.dump_file_name = dump_file_name

    def recommend(self, uids, n_items=10, verbose=False):
        if verbose:
            print('■ ■ ■ {} ■ ■ ■'.format(self.algorithm.__name__))

        # Assign original data to a temporary variable
        data = self.data

        # Path to the serialized model
        trained_model = os.path.expanduser(self.dump_file_name)

        # Load the serialized model or perform training again
        try:
            _, algo = dump.load(trained_model)
        except FileNotFoundError:
            # Perform random sampling on the raw ratings
            if verbose:
                print('■ Performing random sampling on the dataset')
            raw_ratings = data.raw_ratings
            np.random.shuffle(raw_ratings)
            threshold = int(self.trainset_size * len(raw_ratings))
            trainset_raw_ratings = raw_ratings[:threshold]
            testset_raw_ratings = raw_ratings[threshold:]

            # Assign new ratings to the original data to construct the trainset
            data.raw_ratings = trainset_raw_ratings

            # Perform Grid Search
            # TODO: implement Random Search or Bayesian Optimization to increase performance
            #       and accurary of the hyperparameter tuning process
            if self.perf_measure not in ['rmse', 'mae']:
                raise ValueError('■ Invalid accuracy measurement provided')

            if verbose:
                print('■ Performing Grid Search')

            data.split(n_folds=self.n_folds)
            grid_search = GridSearch(self.algorithm, param_grid=self.param_grid,
                                     measures=[self.perf_measure], verbose=verbose)
            grid_search.evaluate(data)
            algo = grid_search.best_estimator[self.perf_measure]
            algo.sim_options = self.sim_options
            algo.bsl_options = self.bsl_options
            algo.verbose = verbose

            if verbose:
                print('■ Grid Search completed')
                pp = pprint.PrettyPrinter()
                pp.pprint(vars(algo))

            if verbose:
                # Retrain on the whole train set
                print('■ Training using trainset')
                trainset = data.build_full_trainset()
                algo.train(trainset)

                # Test on the testset
                print('■ Evaluating using testset')
                testset = data.construct_testset(testset_raw_ratings)
                predictions = algo.test(testset)
                accuracy.rmse(predictions)

        # Generate top recommendations
        if verbose:
            print('■ Using the best estimator on full dataset')
        start = default_timer()
        data = self.data
        trainset = data.build_full_trainset()
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)

        # Dump the trained model to a file
        if self.dump_model:
            if verbose:
                print('■ Saving the trained model')
            dump.dump(trained_model, predictions, algo, verbose)

        # Display some accuracy scores
        accuracy.mae(predictions)
        accuracy.rmse(predictions)

        # Calculate execution time
        duration = default_timer() - start
        duration = datetime.timedelta(seconds=math.ceil(duration))
        print('■ Time elapsed:', duration)

        return self.get_top_predictions(uids, predictions, n_items)

    def get_top_predictions(self, uids, predictions, n_items):
        if not uids:
            raise ValueError('■ Invalid users provided')
        try:
            predictions = self.get_top_n(predictions, n_items)
            return {str(uid): predictions[str(uid)] for uid in list(uids)}
        except KeyError:
            print('■ Cannot find the given user')

    def load_data(self):
        data = Dataset.load_builtin('ml-100k')
        return data

    def get_top_n(self, predictions, n):
        top_n = defaultdict(list)
        for uid, iid, r_ui, est, _ in predictions:
            info = {'iid': iid, 'r_ui': "%.2f" % r_ui, 'est': "%.2f" % est}
            top_n[uid].append(info.copy())

        for uid, ratings in top_n.items():
            ratings.sort(key=lambda x: x['est'], reverse=True)
            top_n[uid] = ratings[:n]

        return top_n


# Testing users
uids = [1, 2, 3]
pp = pprint.PrettyPrinter()

def print_recommendations(results):
    print('■ Recommendations:')
    pp.pprint(results)


"""Neighborhood-based collaborative filtering (kNN-basic)
"""
from surprise import KNNBasic

param_grid = {'k': [20, 30, 40]}
sim_options = {'name': 'pearson_baseline', 'user_based': True}
recommender = Recommender(algorithm=KNNBasic,
                          param_grid=param_grid,
                          sim_options=sim_options,
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='knn_basic')

print_recommendations(recommender.recommend(uids=uids, verbose=True))


"""Neighborhood-based collaborative filtering (kNN-baseline)
"""
from surprise import KNNBaseline

param_grid = {'k': [20, 40, 60]}
bsl_options = {'method': 'sgd', 'learning_rate': 0.0007}
sim_options = {'name': 'pearson_baseline', 'user_based': True}
recommender = Recommender(algorithm=KNNBaseline,
                          param_grid=param_grid,
                          bsl_options=bsl_options,
                          sim_options=sim_options,
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='knn_baseline')

print_recommendations(recommender.recommend(uids=uids, verbose=True))


"""Matrix factorization - SVD using Stochastic Gradient Descent
"""
from surprise import SVD

bsl_options = {'method': 'sgd'}
param_grid = {'n_factors': [20, 50], 'lr_all': [0.0003, 0.0007]}
recommender = Recommender(algorithm=SVD,
                          param_grid=param_grid,
                          bsl_options=bsl_options,
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='svd')

print_recommendations(recommender.recommend(uids=uids, verbose=True))


"""Matrix factorization - SVD++ using Alternating Least Squares
"""
from surprise import SVDpp

bsl_options = {'method': 'als'}
param_grid = {'n_epochs': [20, 30], 'reg_all': [0.01, 0.02]}
recommender = Recommender(algorithm=SVDpp,
                          param_grid=param_grid,
                          bsl_options=bsl_options,
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='svdpp')

print_recommendations(recommender.recommend(uids=uids, verbose=True))


"""Slope One
"""
from surprise import SlopeOne

recommender = Recommender(algorithm=SlopeOne,
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='slope_one')

print_recommendations(recommender.recommend(uids=uids, verbose=True))


"""Co-Clustering
"""
from surprise import CoClustering

param_grid = {'n_epochs': [20, 40], 'n_cltr_u': [3, 5], 'n_cltr_i': [3, 5]}
recommender = Recommender(algorithm=CoClustering,
                          param_grid=param_grid,
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='co_clustering')

print_recommendations(recommender.recommend(uids=uids, verbose=True))


"""VIBLO
"""
from surprise import SVDpp, Reader

def load_viblo_data(path, rating_scale):
    file_path = os.path.expanduser(path)
    if not os.path.exists(file_path):
        raise RuntimError('Cannot find the given dataset')
    reader = Reader(line_format='user item rating', sep=',', rating_scale=rating_scale, skip_lines=1)
    data = Dataset.load_from_file(file_path=file_path, reader=reader)
    return data

votes = load_viblo_data('./data/votes.csv', (-1, 1))
clips = load_viblo_data('./data/clips.csv', (0, 1))

bsl_options = {'method': 'als'}
param_grid = {'n_epochs': [20, 30], 'n_factors': [20, 50], 'reg_all': [0.01, 0.02]}
recommender = Recommender(algorithm=SVDpp,
                          param_grid=param_grid,
                          bsl_options=bsl_options,
                          data=votes,
                          perf_measure='rmse',
                          dump_model=False)

print_recommendations(recommender.recommend(uids=[2, 9, 21, 86, 14239, 14300], verbose=True))

bsl_options = {'method': 'sgd'}
param_grid = {'n_epochs': [20, 30], 'n_factors': [20, 50], 'reg_all': [0.01, 0.02]}
recommender = Recommender(algorithm=SVDpp,
                          param_grid=param_grid,
                          bsl_options=bsl_options,
                          data=clips,
                          perf_measure='rmse',
                          dump_model=False)

print_recommendations(recommender.recommend(uids=[2, 5010, 5081, 12758, 12825, 13072], verbose=True))
