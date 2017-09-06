import json
from surprise import SVD
from rs import Recommender, load_data_from_file, parse_config


path = parse_config(section='Path', key='data') + '/views_without_ips.csv'
views = load_data_from_file(path, (1, 50))

print('■ Views data')
param_grid = {'n_factors': [100, 150]}
recommender = Recommender(algorithm=SVD,
                          param_grid=param_grid,
                          data=views,
                          rating_threshold=10)

data = recommender.recommend(uids=[1087], n_items=10, verbose=True)

with open('data.json', 'w') as file:
    json.dump(data, file)
