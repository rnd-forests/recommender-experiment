"""
Slope One
"""

from surprise import SlopeOne
from rs import Recommender, pretty_print

uids = [1, 2, 3]

recommender = Recommender(algorithm=SlopeOne,
                          dump_model=True,
                          dump_file_name='./trained_models/slope_one')

pretty_print(recommender.recommend(uids=uids, verbose=True))
