
# coding: utf-8

# In[9]:

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
    def __init__(self, algorithm, param_grid, bsl_options, sim_options,
                 data = None, rating_scale=(1, 5), perf_measure='rmse', n_folds=5,
                 trainset_size=0.8, dump_model=True, dump_file_name='knn'):
        self.algorithm = algorithm
        self.param_grid = param_grid
        self.bsl_options = bsl_options
        self.sim_options = sim_options
        self.rating_scale = rating_scale
        self.perf_measure = perf_measure
        self.n_folds = n_folds
        self.trainset_size = trainset_size
        self.dump_model = dump_model
        self.dump_file_name = dump_file_name
        self.data = data or self.load_data()

    def recommend(self, uids, n_items=10, verbose=False):
        # Assign original data to a temporary variable
        data = self.data
        
        # Path to the serialized model
        trained_model = os.path.expanduser(self.dump_file_name)

        # Load the serialized model or performing training again
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
            test_raw_ratings = raw_ratings[threshold:]

            # Assign new ratings to the original data
            data.raw_ratings = trainset_raw_ratings

            """Perform Grid Search
            TODO: implement Random Search or Bayesian Optimization to increase performance
                  and accurary of the hyperparameter tuning process
            """
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

            # Retrain on the whole train set
            if verbose:
                print('■ Training using trainset')
            trainset = data.build_full_trainset()
            algo.train(trainset)
            algo.verbose = verbose

            if verbose:
                # Test on the testset
                print('■ Evaluating using testset')
                testset = data.construct_testset(test_raw_ratings)
                predictions = algo.test(testset)
                accuracy.rmse(predictions)

        # Generate top-N recommendations
        if verbose:
            print('■ Using the best estimator on full dataset')
        start = default_timer()
        data = self.data
        trainset = data.build_full_trainset()
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        
        if self.dump_model:
            if verbose:
                print('■ Saving the trained model')
            dump.dump(trained_model, predictions, algo, verbose)
                
        accuracy.mae(predictions)
        accuracy.rmse(predictions)

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
        for uid, iid, _, est, details in predictions:
            top_n[uid].append((iid, est, details))

        for uid, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = ratings[:n]

        return top_n


# In[12]:

uids = [1, 2, 3]
pp = pprint.PrettyPrinter()


# In[23]:

from surprise import KNNBasic

# Neighborhood-based collaborative filtering (kNN-basic)
param_grid = {'k': [20, 40, 60]}
sim_options = {'name': 'pearson_baseline', 'user_based': True}
recommender = Recommender(algorithm=KNNBasic,
                          param_grid=param_grid,
                          bsl_options={},
                          sim_options=sim_options,
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='knn_basic')

recommendations = recommender.recommend(uids=uids, verbose=True)
print('■ Recommendations:')
pp.pprint(recommendations)


# In[24]:

from surprise import KNNBaseline

# Neighborhood-based collaborative filtering (kNN-baseline)
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

recommendations = recommender.recommend(uids=uids, verbose=True)
print('■ Recommendations:')
pp.pprint(recommendations)


# In[25]:

from surprise import SVD

# Matrix factorization - SVD using Stochastic Gradient Descent
bsl_options = {'method': 'sgd'}
param_grid = {'n_factors': [20, 50], 'lr_all': [0.0003, 0.0007]}
recommender = Recommender(algorithm=SVD,
                          param_grid=param_grid,
                          bsl_options=bsl_options,
                          sim_options={},
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='svd')

recommendations = recommender.recommend(uids=uids, verbose=True)
print('■ Recommendations:')
pp.pprint(recommendations)


# In[37]:

from surprise import SVDpp

# Matrix factorization - SVD++ using Alternating Least Squares
bsl_options = {'method': 'als'}
param_grid = {'n_epochs': [20, 30], 'reg_all': [0.01, 0.02]}
recommender = Recommender(algorithm=SVDpp,
                          param_grid=param_grid,
                          bsl_options=bsl_options,
                          sim_options={},
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='svdpp')

recommendations = recommender.recommend(uids=uids, verbose=True)
print('■ Recommendations:')
pp.pprint(recommendations)


# In[26]:

from surprise import SlopeOne

# Slope One
recommender = Recommender(algorithm=SlopeOne,
                          param_grid={},
                          bsl_options={},
                          sim_options={},
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='slope_one')

recommendations = recommender.recommend(uids=uids, verbose=True)
pp.pprint(recommendations)


# In[27]:

from surprise import CoClustering

# Co-Clustering
param_grid = {'n_epochs': [20, 40], 'n_cltr_u': [3, 5], 'n_cltr_i': [3, 5]}
recommender = Recommender(algorithm=CoClustering,
                          param_grid=param_grid,
                          bsl_options={},
                          sim_options={},
                          perf_measure='rmse',
                          dump_model=True,
                          dump_file_name='co_clustering')

recommendations = recommender.recommend(uids=uids, verbose=True)
print('■ Recommendations:')
pp.pprint(recommendations)


# In[28]:

import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def load_data_into_dataframe(path):
    if not os.path.exists(path):
        raise RuntimeError('Cannot find the given dataset!')
    df = pd.read_csv(path, sep='\t', names=['UserID', 'ItemID', 'Rating', 'Timestamp'])
    return df

path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
df = load_data_into_dataframe(path)


# In[29]:

print(df.shape)


# In[30]:

print(df.columns)


# In[31]:

print(df.head())


# In[32]:

print(df.info())


# In[33]:

print(df.describe())


# In[34]:

plt.hist(df['Rating'])


# In[35]:

df.groupby(['Rating'])['UserID'].count()


# In[36]:

plt.hist(df.groupby(['ItemID'])['ItemID'].count())


# In[19]:

from surprise import Reader

def load_viblo_data(path, rating_scale):
    file_path = os.path.expanduser(path)
    if not os.path.exists(file_path):
        raise RuntimError('Cannot find the given dataset')
    reader = Reader(line_format='user item rating', sep=',', rating_scale=rating_scale, skip_lines=1)
    data = Dataset.load_from_file(file_path=file_path, reader=reader)
    return data


# In[20]:

from surprise import SVDpp
votes = load_viblo_data('./data/votes.csv', (-1, 1))
clips = load_viblo_data('./data/clips.csv', (0, 1))


# In[21]:

bsl_options = {'method': 'als'}
param_grid = {'n_epochs': [20, 30], 'n_factors': [20, 50], 'reg_all': [0.01, 0.02]}
recommender = Recommender(algorithm=SVDpp,
                          param_grid=param_grid,
                          bsl_options=bsl_options,
                          sim_options={},
                          data=votes,
                          perf_measure='rmse',
                          dump_model=False)

recommendations = recommender.recommend(uids=[2, 9, 21, 86, 14239, 14300], verbose=True)
print('■ Recommendations:')
pp.pprint(recommendations)


# In[22]:

bsl_options = {'method': 'sgd'}
param_grid = {'n_epochs': [20, 30], 'n_factors': [20, 50], 'reg_all': [0.01, 0.02]}
recommender = Recommender(algorithm=SVDpp,
                          param_grid=param_grid,
                          bsl_options=bsl_options,
                          sim_options={},
                          data=clips,
                          perf_measure='rmse',
                          dump_model=False)

recommendations = recommender.recommend(uids=[2, 5010, 5081, 12758, 12825, 13072], verbose=True)
print('■ Recommendations:')
pp.pprint(recommendations)

