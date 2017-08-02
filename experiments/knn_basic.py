"""
Neighborhood-based collaborative filtering (kNN-basic)
"""

from surprise import KNNBasic
from rs import Recommender, pretty_print

uids = [1, 2, 3]
param_grid = {'k': [20, 40],
              'sim_options': [{'name': 'msd'},
                              {'name': 'cosine'},
                              {'name': 'pearson'},
                              {'name': 'pearson_baseline'},
                              {'name': 'pearson_baseline', 'shrinkage': 150}]}

recommender = Recommender(algorithm=KNNBasic,
                          param_grid=param_grid,
                          dump_model=True,
                          dump_file_name='./trained_models/knn_basic')

pretty_print(recommender.recommend(uids=uids, verbose=True))
