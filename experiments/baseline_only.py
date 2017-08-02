"""
Baseline only
"""

from surprise import BaselineOnly
from rs import Recommender, pretty_print, get_dump_path

uids = [1, 2, 3]
param_grid = {'bsl_options': [{'method': 'als', 'n_epochs': 30},
                              {'method': 'sgd', 'n_epochs': 30, 'learning_rate': 0.0007}]}

recommender = Recommender(algorithm=BaselineOnly,
                          param_grid=param_grid,
                          dump_model=True,
                          dump_file_name=get_dump_path('baseline_only'))

pretty_print(recommender.recommend(uids=uids, verbose=True))
