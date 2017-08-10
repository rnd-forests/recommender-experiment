"""
Co-Clustering
"""

from surprise import CoClustering
from rs import Recommender, get_dump_path

uids = [1, 2, 3]
param_grid = {'n_epochs': [20, 60], 'n_cltr_u': [3, 5], 'n_cltr_i': [3, 5]}

recommender = Recommender(algorithm=CoClustering,
                          param_grid=param_grid,
                          dump_model=True,
                          dump_file_name=get_dump_path('co_clustering'))

recommender.recommend(uids=uids, verbose=True)
