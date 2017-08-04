import os
from surprise import SVD, Reader, Dataset
from rs import Recommender, pretty_print, parse_config, get_dump_path

def load_viblo_data(path, rating_scale):
    file_path = os.path.expanduser(path)
    if not os.path.exists(file_path):
        raise RuntimeError('Cannot find the given dataset')
    reader = Reader(line_format='user item rating',
                    sep=',',
                    rating_scale=rating_scale,
                    skip_lines=1)
    data = Dataset.load_from_file(file_path=file_path, reader=reader)
    return data

data_path = parse_config(section='Path', key='data')
votes = load_viblo_data(data_path + '/votes.csv', (-1, 1))
clips = load_viblo_data(data_path + '/clips.csv', (0, 1))

print('■ Voting data')
param_grid = {'n_epochs': [20, 30, 50], 'n_factors': [20, 50], 'reg_all': [0.01, 0.02]}
recommender = Recommender(algorithm=SVD,
                          param_grid=param_grid,
                          data=votes,
                          dump_model=True,
                          dump_file_name=get_dump_path('viblo_votes'))

pretty_print(recommender.recommend(uids=[2, 9, 21], verbose=True))

print()
print('■ Clipping data')
param_grid = {'n_epochs': [20, 30], 'n_factors': [20, 50]}
recommender = Recommender(algorithm=SVD,
                          param_grid=param_grid,
                          data=clips,
                          dump_model=True,
                          dump_file_name=get_dump_path('viblo_clips'))

pretty_print(recommender.recommend(uids=[2, 12825, 13072], verbose=True))
