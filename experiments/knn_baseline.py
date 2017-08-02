"""
Neighborhood-based collaborative filtering (kNN-baseline)
"""

from surprise import KNNBaseline
from rs import Recommender, pretty_print

uids = [1, 2, 3]
param_grid = {'k': [20, 40],
              'bsl_options': [{'method': 'als', 'n_epochs': 30},
                              {'method': 'sgd', 'n_epochs': 30, 'learning_rate': 0.0007}],
              'sim_options': [{'name': 'cosine'},
                              {'name': 'pearson_baseline', 'shrinkage': 150}]}

recommender = Recommender(algorithm=KNNBaseline,
                          param_grid=param_grid,
                          dump_model=True,
                          dump_file_name='./trained_models/knn_baseline')

pretty_print(recommender.recommend(uids=uids, verbose=True))
