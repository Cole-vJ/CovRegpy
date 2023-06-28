

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

# from AdvEMDpy import EMD
# from emd_basis import Basis
# from emd_hilbert import Hilbert, theta, omega, hilbert_spectrum, morlet_window, morlet_window_adjust
# from emd_mean import Fluctuation
# from emd_preprocess import Preprocess
# from emd_utils import Utility, time_extension
# from PyEMD import EMD as pyemd0215
# import emd as emd040

from CovRegpy_DCC import covregpy_dcc

sns.set(style='darkgrid')


class RCRUnitTests:

    def __init__(self, covregpy_dcc):

        self.covregpy_dcc = covregpy_dcc

    def test_all(self, print_all=False):

        test_1 = self.test_returns_suitable(test_all=True)
        test_2 = self.test_returns_nans(test_all=True)
        test_3 = self.test_returns_floats(test_all=True)

        tests = [test_1, test_2, test_3]

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




if __name__ == "__main__":

    test_rcr = RCRUnitTests(covregpy_dcc)
    test_rcr.test_returns_suitable()
    test_rcr.test_returns_nans()
    test_rcr.test_returns_floats()
    test_rcr.test_all()
    # test_emd.test_emd_basis(plot=True)

