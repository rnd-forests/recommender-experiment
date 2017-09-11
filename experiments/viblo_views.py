import json
from rs import Recommender, load_data_from_file, parse_config


path = parse_config(section='Path', key='data') + '/views_without_ips.csv'
views = load_data_from_file(path, (1, 238))

recommender = Recommender(data=views, rating_threshold=10, anti_testset=False)
data = recommender.recommend(uids=[1087], n_items=15, verbose=True)

with open('data.json', 'w') as file:
    json.dump(data, file)
