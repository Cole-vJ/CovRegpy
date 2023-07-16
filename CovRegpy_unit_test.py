

#     ________
#            /
#      \    /
#       \  /
#        \/

import time

import pandas as pd
import pytest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF

from CovRegpy_DCC import covregpy_dcc, dcc_loglike
from CovRegpy_forecasting import gp_forecast

sns.set(style='darkgrid')


class RCRUnitTests:

    def __init__(self, covregpy_dcc, dcc_loglike, gp_forecast):

        self.covregpy_dcc = covregpy_dcc
        self.dcc_loglike = dcc_loglike
        self.gp_forecast = gp_forecast

    def test_all(self, print_all=False):

        test_1 = self.test_returns_suitable(test_all=True)
        test_2 = self.test_returns_nans(test_all=True)
        test_3 = self.test_returns_floats(test_all=True)
        test_4 = self.test_dcc_loglike_params_nans(test_all=True)
        test_5 = self.test_dcc_loglike_params_floats(test_all=True)
        test_6 = self.test_dcc_loglike_returns_nans(test_all=True)
        test_7 = self.test_dcc_loglike_returns_floats(test_all=True)
        test_8 = self.test_dcc_loglike_covariance_shape(test_all=True)
        test_9 = self.test_dcc_loglike_covariance_nans(test_all=True)
        test_10 = self.test_dcc_loglike_covariance_floats(test_all=True)
        test_11 = self.test_gp_forecast_independent_suitable(test_all=True)
        test_12 = self.test_gp_forecast_independent_nans(test_all=True)
        test_13 = self.test_gp_forecast_independent_floats(test_all=True)
        test_14 = self.test_gp_forecast_dependent_suitable(test_all=True)
        test_15 = self.test_gp_forecast_dependent_nans(test_all=True)
        test_16 = self.test_gp_forecast_dependent_floats(test_all=True)

        tests = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10, test_11, test_12,
                 test_13, test_14, test_15, test_16]

        if print_all:
            print(tests)
            print('Number of tests: {}'.format(len(tests)))

        if all(tests):
            print('ALL TESTS PASSED.')
        else:
            print('SOME TESTS FAILED.')
            for i, test in enumerate(tests):
                if not test:
                    print(f'TEST {int(i + 1)} FAILED')

    def test_returns_suitable(self, test_all=False):

        returns_matrix_test = 1.0

        with pytest.raises(TypeError) as error_info:
            self.covregpy_dcc(returns_matrix_test, p=3, q=3, days=10, print_correlation=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Returns must be of type np.ndarray and pd.Dataframe.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Returns must be of type np.ndarray and pd.Dataframe.'

    def test_returns_nans(self, test_all=False):

        returns_matrix_test = pd.DataFrame(np.asarray([[1, 1], [1, 1]]))
        returns_matrix_test[0, 0] = np.nan

        with pytest.raises(TypeError) as error_info:
            self.covregpy_dcc(returns_matrix_test, p=3, q=3, days=10, print_correlation=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Returns must not contain nans.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Returns must not contain nans.'

    def test_returns_floats(self, test_all=False):

        returns_matrix_test = pd.DataFrame(np.asarray([[1, 1], [1, 1]]))
        returns_matrix_test.iloc[0] = True

        with pytest.raises(TypeError) as error_info:
            self.covregpy_dcc(returns_matrix_test, p=3, q=3, days=10, print_correlation=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Returns must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Returns must only contain floats.'

    def test_dcc_loglike_params_nans(self, test_all=False):

        params_test = (np.nan, 1.0)

        with pytest.raises(ValueError) as error_info:
            self.dcc_loglike(params=params_test,
                             returns_matrix=np.random.normal(0., 1., (100, 5)),
                             modelled_variance=np.random.normal(0., 1., (5, 5)))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == 'Parameters must not contain nans.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == 'Parameters must not contain nans.'

    def test_dcc_loglike_params_floats(self, test_all=False):

        params_test = ('A', 1.0)

        with pytest.raises(ValueError) as error_info:
            self.dcc_loglike(params=params_test,
                             returns_matrix=np.random.normal(0., 1., (5, 100)),
                             modelled_variance=np.random.normal(0., 1., (5, 5)))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == 'Parameters must only contain floats.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == 'Parameters must only contain floats.'

    def test_dcc_loglike_returns_nans(self, test_all=False):

        returns_test = returns_matrix=np.random.normal(0., 1., (100, 5))
        returns_test[0, 0] = np.nan

        with pytest.raises(ValueError) as error_info:
            self.dcc_loglike(params=(0.2, 0.8),
                             returns_matrix=returns_test,
                             modelled_variance=np.random.normal(0., 1., (5, 5)))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == 'Returns must not contain nans.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == 'Returns must not contain nans.'

    def test_dcc_loglike_returns_floats(self, test_all=False):

        returns_test = np.array(np.random.normal(0., 1., (100, 5)), dtype=object)
        returns_test[0, 0] = 'A'

        with pytest.raises(ValueError) as error_info:
            self.dcc_loglike(params=(0.2, 0.8),
                             returns_matrix=returns_test,
                             modelled_variance=np.random.normal(0., 1., (5, 5)))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == 'Returns must only contain floats.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == 'Returns must only contain floats.'

    def test_dcc_loglike_covariance_shape(self, test_all=False):

        covariance_test = np.random.normal(0., 1., (5, 6))
        covariance_test[0, 0] = np.nan

        with pytest.raises(ValueError) as error_info:
            self.dcc_loglike(params=(0.2, 0.8),
                             returns_matrix=np.random.normal(0., 1., (5, 100)),
                             modelled_variance=covariance_test)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == 'Covariance must be square matrix.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == 'Covariance must be square matrix.'

    def test_dcc_loglike_covariance_nans(self, test_all=False):

        covariance_test = np.random.normal(0., 1., (5, 5))
        covariance_test[0, 0] = np.nan

        with pytest.raises(ValueError) as error_info:
            self.dcc_loglike(params=(0.2, 0.8),
                             returns_matrix=np.random.normal(0., 1., (5, 100)),
                             modelled_variance=covariance_test)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == 'Covariance must not contain nans.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == 'Covariance must not contain nans.'

    def test_dcc_loglike_covariance_floats(self, test_all=False):

        covariance_test = np.array(np.random.normal(0., 1., (5, 5)), dtype=object)
        covariance_test[0, 0] = 'A'

        with pytest.raises(ValueError) as error_info:
            self.dcc_loglike(params=(0.2, 0.8),
                             returns_matrix=np.random.normal(0., 1., (5, 100)),
                             modelled_variance=covariance_test)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == 'Covariance must only contain floats.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == 'Covariance must only contain floats.'


    def test_gp_forecast_independent_suitable(self, test_all=False):

        gp_x = 1.0
        example_kernel = RBF(length_scale=1.0)

        with pytest.raises(TypeError) as error_info:
            self.gp_forecast(x_fit=gp_x, y_fit=np.asarray(np.arange(0.0, 11.0, 1.0) + np.random.normal(0, 1, 11)),
                             x_forecast=np.asarray([11.0, 12.0, 13.0]), kernel=example_kernel, confidence_level=0.95,
                             plot=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Independent variable for fitting must '
                                                                               'be of type np.ndarray and pd.Dataframe.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Independent variable for fitting must ' \
                                                                                'be of type np.ndarray and pd.Dataframe.'

    def test_gp_forecast_independent_nans(self, test_all=False):

        gp_x = pd.DataFrame(np.asarray(np.arange(11)))
        gp_x[3] = np.nan

        example_kernel = RBF(length_scale=1.0)

        with pytest.raises(TypeError) as error_info:
            self.gp_forecast(x_fit=gp_x, y_fit=np.asarray(np.arange(0.0, 11.0, 1.0) + np.random.normal(0, 1, 11)),
                             x_forecast=np.asarray([11.0, 12.0, 13.0]), kernel=example_kernel, confidence_level=0.95,
                             plot=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Independent variable for fitting '
                                                                               'must not contain nans.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Independent variable for fitting ' \
                                                                                'must not contain nans.'

    def test_gp_forecast_independent_floats(self, test_all=False):

        gp_x = np.array(np.random.normal(0., 1., 10), dtype=object)
        gp_x[3] = True

        example_kernel = RBF(length_scale=1.0)

        with pytest.raises(TypeError) as error_info:
            self.gp_forecast(x_fit=gp_x, y_fit=np.asarray(np.arange(0.0, 11.0, 1.0) + np.random.normal(0, 1, 11)),
                             x_forecast=np.asarray([11.0, 12.0, 13.0]), kernel=example_kernel, confidence_level=0.95,
                             plot=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Independent variable for fitting must '
                                                                               'only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Independent variable for fitting must ' \
                                                                                'only contain floats.'


    def test_gp_forecast_dependent_suitable(self, test_all=False):

        gp_y = 1.0
        example_kernel = RBF(length_scale=1.0)

        with pytest.raises(TypeError) as error_info:
            self.gp_forecast(x_fit=np.asarray(np.arange(0.0, 11.0, 1.0)), y_fit=gp_y,
                             x_forecast=np.asarray([11.0, 12.0, 13.0]), kernel=example_kernel, confidence_level=0.95,
                             plot=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Dependent variable for fitting must '
                                                                               'be of type np.ndarray and pd.Dataframe.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Dependent variable for fitting must ' \
                                                                                'be of type np.ndarray and pd.Dataframe.'

    def test_gp_forecast_dependent_nans(self, test_all=False):

        gp_y = pd.DataFrame(np.asarray(np.arange(11)))
        gp_y[3] = np.nan

        example_kernel = RBF(length_scale=1.0)

        with pytest.raises(TypeError) as error_info:
            self.gp_forecast(x_fit=np.asarray(np.arange(0.0, 11.0, 1.0)), y_fit=gp_y,
                             x_forecast=np.asarray([11.0, 12.0, 13.0]), kernel=example_kernel, confidence_level=0.95,
                             plot=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Dependent variable for fitting '
                                                                               'must not contain nans.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Dependent variable for fitting ' \
                                                                                'must not contain nans.'

    def test_gp_forecast_dependent_floats(self, test_all=False):

        gp_y = np.array(np.random.normal(0., 1., 10), dtype=object)
        gp_y[3] = True

        example_kernel = RBF(length_scale=1.0)

        with pytest.raises(TypeError) as error_info:
            self.gp_forecast(x_fit=np.asarray(np.arange(0.0, 11.0, 1.0)), y_fit=gp_y,
                             x_forecast=np.asarray([11.0, 12.0, 13.0]), kernel=example_kernel, confidence_level=0.95,
                             plot=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Dependent variable for fitting must '
                                                                               'only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Dependent variable for fitting must ' \
                                                                                'only contain floats.'


if __name__ == "__main__":

    test_rcr = RCRUnitTests(covregpy_dcc, dcc_loglike, gp_forecast)
    test_rcr.test_returns_suitable()
    test_rcr.test_returns_nans()
    test_rcr.test_returns_floats()
    test_rcr.test_dcc_loglike_params_nans()
    test_rcr.test_dcc_loglike_params_floats()
    test_rcr.test_dcc_loglike_returns_nans()
    test_rcr.test_dcc_loglike_returns_floats()
    test_rcr.test_dcc_loglike_covariance_shape()
    test_rcr.test_dcc_loglike_covariance_nans()
    test_rcr.test_dcc_loglike_covariance_floats()
    test_rcr.test_gp_forecast_independent_suitable()
    test_rcr.test_gp_forecast_independent_nans()
    test_rcr.test_gp_forecast_independent_floats()
    test_rcr.test_gp_forecast_dependent_suitable()
    test_rcr.test_gp_forecast_dependent_nans()
    test_rcr.test_gp_forecast_dependent_floats()
    test_rcr.test_all(print_all=True)

