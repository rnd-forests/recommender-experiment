from surprise import SVD
from rs import Recommender, load_data_from_file, parse_config, get_dump_path


data_path = parse_config(section='Path', key='data')
votes = load_data_from_file(data_path + '/votes.csv', (-1, 1))
clips = load_data_from_file(data_path + '/clips.csv', (0, 1))

print('■ Voting data')
param_grid = {'n_epochs': [20, 30], 'n_factors': [20, 50]}
recommender = Recommender(algorithm=SVD,
                          param_grid=param_grid,
                          data=votes,
                          rating_threshold=0.5,
                          dump_model=True,
                          dump_file_name=get_dump_path('viblo_votes'))

recommender.recommend(uids=[2, 9, 21], verbose=True)

print()
print('■ Clipping data')
param_grid = {'n_epochs': [20, 30], 'n_factors': [20, 50]}
recommender = Recommender(algorithm=SVD,
                          param_grid=param_grid,
                          data=clips,
                          rating_threshold=1,
                          dump_model=True,
                          dump_file_name=get_dump_path('viblo_clips'))

recommender.recommend(uids=[2, 12825, 13072], verbose=True)
