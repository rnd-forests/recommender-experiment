"""
Matrix factorization - SVD++ using Alternating Least Squares
"""

from surprise import SVDpp
from rs import Recommender, get_dump_path

uids = [1, 2, 3]
param_grid = {'n_epochs': [10, 20], 'reg_all': [0.01, 0.02]}

recommender = Recommender(algorithm=SVDpp,
                          param_grid=param_grid,
                          dump_model=True,
                          n_folds=3,
                          dump_file_name=get_dump_path('svdpp'))

recommender.recommend(uids=uids, verbose=True)
