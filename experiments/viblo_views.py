from surprise import SVD
from rs import Recommender, load_data_from_file, parse_config, get_dump_path


path = parse_config(section='Path', key='data') + '/views.csv'
views = load_data_from_file(path, (1, 240))

print('■ Views data')
param_grid = {'n_epochs': [20, 30], 'n_factors': [20, 50]}
recommender = Recommender(algorithm=SVD,
                          param_grid=param_grid,
                          data=views,
                          rating_threshold=15,
                          dump_model=True,
                          dump_file_name=get_dump_path('viblo_views'))

recommendations = recommender.recommend(uids=[1087])
print(recommendations)
