"""
Matrix factorization - Non-negative Matrix Factorization
"""

from surprise import NMF
from rs import Recommender, pretty_print, get_dump_path

uids = [1, 2, 3]
param_grid = {'n_epochs': [50, 100], 'n_factors': [15, 20]}

recommender = Recommender(algorithm=NMF,
                          param_grid=param_grid,
                          dump_model=True,
                          dump_file_name=get_dump_path('nmf'))

pretty_print(recommender.recommend(uids=uids, verbose=True))
