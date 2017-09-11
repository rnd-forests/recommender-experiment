from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import math
import datetime
import numpy as np
import pandas as pd
from timeit import default_timer
from collections import defaultdict
from surprise import SVD, GridSearch, Dataset, accuracy, dump
from .utils import precision_recall_at_k, print_object, pretty_print


class Recommender:
    def __init__(self, algorithm=SVD,
                 param_grid=None, bsl_options=None, sim_options=None, perf_measure='rmse',
                 data=None, rating_threshold=3.5, trainset_size=0.8, n_folds=5,
                 dump_model=False, dump_file_name=''):

        self.algorithm = algorithm  # The estimator
        self.param_grid = param_grid if param_grid is not None else {}  # Parameter list used for grid search
        self.bsl_options = bsl_options  # Baseline options
        self.sim_options = sim_options  # Similarity options
        self.perf_measure = perf_measure  # Performance measure (RMSE, MAE, FCP)

        self.data = data if data is not None else self.load_data()  # The data used for algorithm
        self.rating_threshold = rating_threshold  # Used for calculate precision-recall scores
        self.trainset_size = trainset_size  # The ratio to split original dataset
        self.n_folds = n_folds  # The number of folds in cross-validation

        self.dump_model = dump_model
        self.dump_file_name = dump_file_name

    def recommend(self, uids, n_items=100, verbose=False):
        if verbose:
            print('■ ■ ■ {} ■ ■ ■'.format(self.algorithm.__name__))

        data = self.data  # Cache the original data
        trained_model = os.path.expanduser(self.dump_file_name)  # Path to the serialized model

        try:
            _, algo = dump.load(trained_model)  # Load the serialized model
        except FileNotFoundError:
            if verbose:
                print('■ Performing random sampling on the dataset')
            raw_ratings = data.raw_ratings
            np.random.shuffle(raw_ratings)
            threshold = int(self.trainset_size * len(raw_ratings))
            trainset_raw_ratings = raw_ratings[:threshold]
            testset_raw_ratings = raw_ratings[threshold:]

            data.raw_ratings = trainset_raw_ratings  # Assign new ratings to the original data

            if any(self.param_grid):
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
            else:
                algo = self.algorithm()  # Use the default settings for the algorithm

            algo.verbose = verbose

            if verbose and any(self.param_grid):
                print('■ Grid Search completed. Here is the summary:')
                cv_results = grid_search.cv_results
                del cv_results['scores']
                df = pd.DataFrame.from_dict(cv_results)
                sort_column = self.perf_measure.upper()
                if df.columns.contains(sort_column):
                    df = df.sort_values([sort_column], ascending=True)
                pretty_print(df)

                print('■ Algorithm properties:')
                print_object(algo)

            if verbose:
                print('■ Training using trainset')
                trainset = data.build_full_trainset()
                algo.train(trainset)

                print('■ Evaluating using testset')
                testset = data.construct_testset(testset_raw_ratings)
                predictions = algo.test(testset)
                accuracy.rmse(predictions)

        # Generate top recommendations
        if verbose:
            print('■ Using the best estimator on full dataset')
        data = self.data
        trainset = data.build_full_trainset()
        # testset = trainset.build_anti_testset()  # May cause problem when dataset is large
        testset = trainset.build_testset()

        start = default_timer()

        # Train and make predictions using best estimator
        algo.train(trainset)
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

        # Print Precision and Recall scores
        self.print_precision_call(predictions, uids, n_items)

        recommendations = self.get_recommendations_for_users(uids, predictions, n_items)

        # Calculate execution time
        duration = default_timer() - start
        duration = datetime.timedelta(seconds=math.ceil(duration))
        print('■ Time elapsed:', duration)

        if verbose:
            print('■ Recommendations:')
            pretty_print(recommendations)

        return recommendations

    def get_recommendations_for_users(self, uids, predictions, n_items):
        if not uids:
            raise ValueError('■ Invalid users provided')
        try:
            predictions = self.get_top_n(predictions, n_items)
            return {str(uid): predictions[str(uid)] for uid in list(uids)}
        except KeyError:
            print('■ Cannot find the given user')

    def print_precision_call(self, predictions, uids, n_items):
        if not uids:
            raise ValueError('■ Invalid users provided')
        try:
            all_precision_recall = precision_recall_at_k(predictions, k=n_items, threshold=self.rating_threshold)
            precision_recall = [('', ('Precision', 'Recall'))]
            precision_recall.extend(item for item in all_precision_recall.items() if int(item[0]) in uids)
            print('■ Precision and Recall:')
            pretty_print(dict(precision_recall))
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
