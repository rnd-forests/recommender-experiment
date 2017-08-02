"""
Co-Clustering
"""

from surprise import CoClustering
from rs import Recommender, pretty_print

uids = [1, 2, 3]
param_grid = {'n_epochs': [20, 60], 'n_cltr_u': [3, 5], 'n_cltr_i': [3, 5]}

recommender = Recommender(algorithm=CoClustering,
                          param_grid=param_grid,
                          dump_model=True,
                          dump_file_name='./trained_models/co_clustering')

pretty_print(recommender.recommend(uids=uids, verbose=True))
