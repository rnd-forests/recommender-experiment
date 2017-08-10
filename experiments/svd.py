"""
Matrix factorization - SVD using Stochastic Gradient Descent
"""

from surprise import SVD
from rs import Recommender, get_dump_path

uids = [1, 2, 3]
param_grid = {'n_factors': [100, 150], 'reg_all': [0.01, 0.02]}

recommender = Recommender(algorithm=SVD,
                          param_grid=param_grid,
                          dump_model=True,
                          dump_file_name=get_dump_path('svd'))

recommender.recommend(uids=uids, verbose=True)
