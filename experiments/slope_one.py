"""
Slope One
"""

from surprise import SlopeOne
from rs import Recommender, pretty_print, get_dump_path

uids = [1, 2, 3]

recommender = Recommender(algorithm=SlopeOne,
                          dump_model=True,
                          dump_file_name=get_dump_path('slope_one'))

pretty_print(recommender.recommend(uids=uids, verbose=True))
