"""
Neighborhood-based collaborative filtering (kNN-baseline)
"""

from surprise import KNNBaseline
from rs import Recommender, get_dump_path

uids = [1, 2, 3]
param_grid = {'k': [20, 40],
              'bsl_options': [{'method': 'als'},
                              {'method': 'sgd', 'learning_rate': 0.0007}],
              'sim_options': [{'name': 'cosine'},
                              {'name': 'pearson_baseline'}]}

recommender = Recommender(algorithm=KNNBaseline,
                          param_grid=param_grid,
                          dump_model=True,
                          dump_file_name=get_dump_path('knn_baseline'))

recommender.recommend(uids=uids, verbose=True)
