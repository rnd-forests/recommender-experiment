from __future__ import absolute_import, division, print_function, unicode_literals

import os
import math
import datetime
import numpy as np
import pandas as pd
from .utils import print_object
from timeit import default_timer
from collections import defaultdict
from surprise import GridSearch, Dataset, accuracy, dump

class Recommender:
    def __init__(self, algorithm,
                 param_grid=None, bsl_options=None, sim_options=None, perf_measure='rmse',
                 data=None, rating_scale=(1, 5), trainset_size=0.8, n_folds=5,
                 dump_model=False, dump_file_name=''):

        self.algorithm = algorithm
        self.param_grid = param_grid if param_grid is not None else {}
        self.bsl_options = bsl_options
        self.sim_options = sim_options
        self.perf_measure = perf_measure

        self.data = data if data is not None else self.load_data()
        self.rating_scale = rating_scale
        self.trainset_size = trainset_size
        self.n_folds = n_folds

        self.dump_model = dump_model
        self.dump_file_name = dump_file_name

    def recommend(self, uids, n_items=10, verbose=False):
        if verbose:
            print('■ ■ ■ {} ■ ■ ■'.format(self.algorithm.__name__))

        # Assign original data to a temporary variable
        data = self.data

        # Path to the serialized model
        trained_model = os.path.expanduser(self.dump_file_name)

        try:
            # Load the serialized model or perform training again
            _, algo = dump.load(trained_model)
        except FileNotFoundError:
            # Perform random sampling on the raw ratings
            if verbose:
                print('■ Performing random sampling on the dataset')
            raw_ratings = data.raw_ratings
            np.random.shuffle(raw_ratings)
            threshold = int(self.trainset_size * len(raw_ratings))
            trainset_raw_ratings = raw_ratings[:threshold]
            testset_raw_ratings = raw_ratings[threshold:]

            # Assign new ratings to the original data
            data.raw_ratings = trainset_raw_ratings

            # Perform Grid Search
            if self.perf_measure not in ['rmse', 'mae', 'fcp']:
                raise ValueError('■ Invalid accuracy measurement provided')

            if verbose:
                print('■ Performing Grid Search')

            data.split(n_folds=self.n_folds)
            grid_search = GridSearch(self.algorithm,
                                     param_grid=self.param_grid,
                                     measures=[self.perf_measure],
                                     verbose=verbose)
            grid_search.evaluate(data)
            algo = grid_search.best_estimator[self.perf_measure]
            if self.sim_options is not None:
                algo.sim_options = self.sim_options
            if self.bsl_options is not None:
                algo.bsl_options = self.bsl_options
            algo.verbose = verbose

            if verbose:
                print('■ Grid Search completed. Here is the summary:')
                df = pd.DataFrame.from_dict(grid_search.cv_results)
                sort_column = self.perf_measure.upper()
                if df.columns.contains(sort_column):
                    df = df.sort_values([sort_column], ascending=True)
                print(df)

                print('■ Algorithm properties:')
                print_object(algo)

            if verbose:
                # Retrain on the whole train set
                print('■ Training using trainset')
                trainset = data.build_full_trainset()
                algo.train(trainset)

                # Test on the testset
                print('■ Evaluating using testset')
                testset = data.construct_testset(testset_raw_ratings)
                predictions = algo.test(testset)
                accuracy.rmse(predictions)

        # Generate top recommendations
        if verbose:
            print('■ Using the best estimator on full dataset')
        start = default_timer()
        data = self.data
        trainset = data.build_full_trainset()
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)

        # Dump the trained model to a file
        if self.dump_model:
            if verbose:
                print('■ Saving the trained model')
            dump.dump(trained_model, predictions, algo, verbose)

        # Display some accuracy scores
        print('■ Accuracy scores:')
        accuracy.mae(predictions)
        accuracy.rmse(predictions)

        # Calculate execution time
        duration = default_timer() - start
        duration = datetime.timedelta(seconds=math.ceil(duration))
        print('■ Time elapsed:', duration)

        return self.get_recommendations_for_users(uids, predictions, n_items)

    def get_recommendations_for_users(self, uids, predictions, n_items):
        if not uids:
            raise ValueError('■ Invalid users provided')
        try:
            predictions = self.get_top_n(predictions, n_items)
            return {str(uid): predictions[str(uid)] for uid in list(uids)}
        except KeyError:
            print('■ Cannot find the given user')

    @staticmethod
    def load_data():
        data = Dataset.load_builtin('ml-100k')
        return data

    @staticmethod
    def get_top_n(predictions, n):
        top_n = defaultdict(list)
        for uid, iid, _, est, _ in predictions:
            info = {'iid': iid, 'est': "%.2f" % est}
            top_n[uid].append(info.copy())

        for uid, ratings in top_n.items():
            ratings.sort(key=lambda x: x['est'], reverse=True)
            top_n[uid] = ratings[:n]

        return top_n
