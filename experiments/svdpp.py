"""
Matrix factorization - SVD++ using Alternating Least Squares
"""

from surprise import SVDpp
from rs import Recommender, pretty_print, get_dump_path

uids = [1, 2, 3]
param_grid = {'n_epochs': [10, 20], 'reg_all': [0.01, 0.02]}

recommender = Recommender(algorithm=SVDpp,
                          param_grid=param_grid,
                          dump_model=True,
                          dump_file_name=get_dump_path('svdpp'))

pretty_print(recommender.recommend(uids=uids, verbose=True))
