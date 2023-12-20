

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
from sklearn.gaussian_process.kernels import RBF

from CovRegpy import calc_B_Psi, gamma_v_m_error, cov_reg_given_mean, subgrad_opt
from CovRegpy_DCC import covregpy_dcc, dcc_loglike
from CovRegpy_forecasting import gp_forecast

sns.set(style='darkgrid')


class RCRUnitTests:

    def __init__(self, calc_B_Psi, gamma_v_m_error, cov_reg_given_mean, subgrad_opt, covregpy_dcc, dcc_loglike,
                 gp_forecast):

        self.calc_B_Psi = calc_B_Psi
        self.gamma_v_m_error = gamma_v_m_error
        self.cov_reg_given_mean = cov_reg_given_mean
        self.subgrad_opt = subgrad_opt

        self.covregpy_dcc = covregpy_dcc
        self.dcc_loglike = dcc_loglike
        self.gp_forecast = gp_forecast

    def test_all(self, print_all=False):

        test_covreg_1 = self.test_m_array(test_all=True)
        test_covreg_2 = self.test_m_only_floats(test_all=True)
        test_covreg_3 = self.test_m_shape(test_all=True)

        test_covreg_4 = self.test_v_array(test_all=True)
        test_covreg_5 = self.test_v_only_floats(test_all=True)
        test_covreg_6 = self.test_v_shape(test_all=True)

        test_covreg_7 = self.test_m_and_v(test_all=True)

        test_covreg_8 = self.test_x_array(test_all=True)
        test_covreg_9 = self.test_x_only_floats(test_all=True)

        test_covreg_10 = self.test_y_array(test_all=True)
        test_covreg_11 = self.test_y_only_floats(test_all=True)

        test_covreg_12 = self.test_basis_array(test_all=True)
        test_covreg_13 = self.test_basis_only_floats(test_all=True)

        test_covreg_14 = self.test_A_est_array(test_all=True)
        test_covreg_15 = self.test_A_est_only_floats(test_all=True)

        test_covreg_16 = self.test_technique_type(test_all=True)

        test_covreg_17 = self.test_technique_type(test_all=True)
        test_covreg_18 = self.test_alpha_type(test_all=True)
        test_covreg_19 = self.test_l1_ratio_or_reg_type(test_all=True)
        test_covreg_20 = self.test_group_reg_type(test_all=True)
        test_covreg_21 = self.test_max_iter_type(test_all=True)

        test_covreg_22 = self.test_group_array(test_all=True)
        test_covreg_23 = self.test_groups_only_integers(test_all=True)

        test_covreg_24 = self.test_gamma_v_m_error_errors_array(test_all=True)
        test_covreg_25 = self.test_gamma_v_m_error_errors_only_floats(test_all=True)
        test_covreg_26 = self.test_gamma_v_m_error_x_array(test_all=True)
        test_covreg_27 = self.test_gamma_v_m_error_x_only_floats(test_all=True)
        test_covreg_28 = self.test_gamma_v_m_error_Psi_array(test_all=True)
        test_covreg_29 = self.test_gamma_v_m_error_Psi_only_floats(test_all=True)
        test_covreg_30 = self.test_gamma_v_m_error_B_array(test_all=True)
        test_covreg_31 = self.test_gamma_v_m_error_B_only_floats(test_all=True)

        test_covreg_32 = self.test_cov_reg_given_mean_A_est_array(test_all=True)
        test_covreg_33 = self.test_cov_reg_given_mean_A_est_only_floats(test_all=True)
        test_covreg_34 = self.test_cov_reg_given_mean_basis_array(test_all=True)
        test_covreg_35 = self.test_cov_reg_given_mean_basis_only_floats(test_all=True)
        test_covreg_36 = self.test_cov_reg_given_mean_x_array(test_all=True)
        test_covreg_37 = self.test_cov_reg_given_mean_x_only_floats(test_all=True)
        test_covreg_38 = self.test_cov_reg_given_mean_y_array(test_all=True)
        test_covreg_39 = self.test_cov_reg_given_mean_y_only_floats(test_all=True)
        test_covreg_40 = self.test_cov_reg_given_mean_iterations_type(test_all=True)
        test_covreg_41 = self.test_cov_reg_given_mean_technique_type(test_all=True)
        test_covreg_42 = self.test_cov_reg_given_mean_alpha_type(test_all=True)
        test_covreg_43 = self.test_cov_reg_given_mean_l1_ratio_or_reg_type(test_all=True)
        test_covreg_44 = self.test_cov_reg_given_mean_group_reg_type(test_all=True)
        test_covreg_45 = self.test_cov_reg_given_mean_max_iter_type(test_all=True)
        test_covreg_46 = self.test_cov_reg_given_mean_groups_array(test_all=True)
        test_covreg_47 = self.test_cov_reg_given_mean_groups_only_integers(test_all=True)

        test_covreg_48 = self.test_subgrad_opt_x_tilda_array(test_all=True)
        test_covreg_49 = self.test_subgrad_opt_x_tilda_only_floats(test_all=True)
        test_covreg_50 = self.test_subgrad_opt_y_tilda_array(test_all=True)
        test_covreg_51 = self.test_subgrad_opt_y_tilda_only_floats(test_all=True)
        test_covreg_52 = self.test_subgrad_opt_max_iter_type(test_all=True)
        test_covreg_53 = self.test_subgrad_opt_alpha_type(test_all=True)

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

        tests = [test_covreg_1, test_covreg_2, test_covreg_3, test_covreg_4, test_covreg_5, test_covreg_6,
                 test_covreg_7, test_covreg_8, test_covreg_9, test_covreg_10, test_covreg_11, test_covreg_12,
                 test_covreg_13, test_covreg_14, test_covreg_15, test_covreg_16, test_covreg_17, test_covreg_18,
                 test_covreg_19, test_covreg_20, test_covreg_21, test_covreg_22, test_covreg_23, test_covreg_24,
                 test_covreg_25, test_covreg_26, test_covreg_27, test_covreg_28, test_covreg_29, test_covreg_30,
                 test_covreg_31, test_covreg_32, test_covreg_33, test_covreg_34, test_covreg_35, test_covreg_36,
                 test_covreg_37, test_covreg_38, test_covreg_39, test_covreg_40, test_covreg_41, test_covreg_42,
                 test_covreg_43, test_covreg_44, test_covreg_45, test_covreg_46, test_covreg_47, test_covreg_48,
                 test_covreg_49, test_covreg_50, test_covreg_51, test_covreg_52, test_covreg_53,

                 test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10, test_11, test_12,
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

    def test_m_array(self, test_all=False):

        m = 1.0

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m, v=np.arange(5), x=np.arange(5), y=np.arange(5),
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'm must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'm must be of type np.ndarray.'

    def test_m_only_floats(self, test_all=False):

        m = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m, v=np.arange(5), x=np.arange(5), y=np.arange(5),
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'm must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'm must only contain floats.'


    def test_m_shape(self, test_all=False):

        m = np.arange(5.0)

        with pytest.raises(ValueError) as error_info:
            self.calc_B_Psi(m, v=np.arange(5), x=np.arange(5), y=np.arange(5),
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is ValueError and ((error_info.value.args[0] == 'm must be column vector.') or
                  error_info.value.args[0] == 'm must be column vector. Suggest: m.reshape(-1, 1)'))
        else:
            return error_info.type is ValueError and \
                ((error_info.value.args[0] == 'm must be column vector.') or
                 error_info.value.args[0] == 'm must be column vector. Suggest: m.reshape(-1, 1)')


    def test_v_array(self, test_all=False):

        v = 1.0

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(0.0, 5.0, 1.0).reshape(-1, 1), v=v, x=np.arange(5), y=np.arange(5),
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'v must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'v must be of type np.ndarray.'

    def test_v_only_floats(self, test_all=False):

        v = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(0.0, 5.0, 1.0).reshape(-1, 1), v=np.arange(5), x=np.arange(5), y=np.arange(5),
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'v must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'v must only contain floats.'


    def test_v_shape(self, test_all=False):

        v = np.arange(5.0)

        with pytest.raises(ValueError) as error_info:
            self.calc_B_Psi(np.arange(0.0, 5.0, 1.0).reshape(-1, 1), v=v, x=np.arange(5), y=np.arange(5),
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is ValueError and ((error_info.value.args[0] == 'v must be column vector.') or
                  error_info.value.args[0] == 'v must be column vector. Suggest: v.reshape(-1, 1)'))
        else:
            return error_info.type is ValueError and \
                ((error_info.value.args[0] == 'v must be column vector.') or
                 error_info.value.args[0] == 'v must be column vector. Suggest: v.reshape(-1, 1)')


    def test_m_and_v(self, test_all=False):

        m = np.arange(5.0).reshape(-1, 1)
        v = np.arange(4.0).reshape(-1, 1)

        with pytest.raises(ValueError) as error_info:
            self.calc_B_Psi(m=m, v=v, x=np.arange(5), y=np.arange(5),
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == 'm and v are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == 'm and v are incompatible lengths.'


    def test_x_array(self, test_all=False):

        x = 5

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=x, y=np.arange(5),
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'x must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'x must be of type np.ndarray.'

    def test_x_only_floats(self, test_all=False):

        x = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=x, y=np.arange(5),
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'x must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'x must only contain floats.'


    def test_y_array(self, test_all=False):

        y = 5

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=y,
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'y must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'y must be of type np.ndarray.'

    def test_y_only_floats(self, test_all=False):

        y = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=y,
                            basis=np.arange(5), A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'y must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'y must only contain floats.'


    def test_basis_array(self, test_all=False):

        basis = 5

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=basis, A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'basis must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'basis must be of type np.ndarray.'

    def test_basis_only_floats(self, test_all=False):

        basis = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=basis, A_est=np.arange(5), technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'basis must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'basis must only contain floats.'


    def test_A_est_array(self, test_all=False):

        A_est = 5

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=np.arange(5.0).reshape(-1, 1), A_est=A_est, technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'A_est must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'A_est must be of type np.ndarray.'

    def test_A_est_only_floats(self, test_all=False):

        A_est = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=np.arange(5.0).reshape(-1, 1), A_est=A_est, technique='direct',
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'A_est must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'A_est must only contain floats.'


    def test_technique_type(self, test_all=False):

        technique = 'not-direct'

        with pytest.raises(ValueError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=np.arange(5.0).reshape(-1, 1), A_est=np.arange(5.0).reshape(-1, 1),
                            technique=technique,
                            alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'technique\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'technique\' not an acceptable value.'


    def test_alpha_type(self, test_all=False):

        alpha = -0.1

        with pytest.raises(ValueError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=np.arange(5.0).reshape(-1, 1), A_est=np.arange(5.0).reshape(-1, 1),
                            technique='direct', alpha=alpha, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1.0,
                            groups=1.0)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'alpha\' must be a non-negative float.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'alpha\' must be a non-negative float.'


    def test_l1_ratio_or_reg_type(self, test_all=False):

        l1_ratio_or_reg = -0.1

        with pytest.raises(ValueError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=np.arange(5.0).reshape(-1, 1), A_est=np.arange(5.0).reshape(-1, 1),
                            technique='direct', alpha=1.0, l1_ratio_or_reg=l1_ratio_or_reg, group_reg=1.0,
                            max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'l1_ratio_or_reg\' must be a non-negative float.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'l1_ratio_or_reg\' must be a non-negative float.'


    def test_group_reg_type(self, test_all=False):

        group_reg = -0.1

        with pytest.raises(ValueError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=np.arange(5.0).reshape(-1, 1), A_est=np.arange(5.0).reshape(-1, 1),
                            technique='direct', alpha=1.0, l1_ratio_or_reg=1.0, group_reg=group_reg,
                            max_iter=1.0, groups=1.0)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'group_reg\' must be a non-negative float.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'group_reg\' must be a non-negative float.'


    def test_max_iter_type(self, test_all=False):

        max_iter = 0

        with pytest.raises(ValueError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=np.arange(5.0).reshape(-1, 1), A_est=np.arange(5.0).reshape(-1, 1),
                            technique='direct', alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0,
                            max_iter=max_iter, groups=1.0)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'max_iter\' must be a positive integer.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'max_iter\' must be a positive integer.'


    def test_group_array(self, test_all=False):

        groups = 1.0

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=np.arange(5.0).reshape(-1, 1), A_est=np.arange(5.0).reshape(-1, 1),
                            technique='direct', alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1,
                            groups=groups)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'groups must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'groups must be of type np.ndarray.'

    def test_groups_only_integers(self, test_all=False):

        groups = np.arange(5.0)

        with pytest.raises(TypeError) as error_info:
            self.calc_B_Psi(m=np.arange(5.0).reshape(-1, 1), v=np.arange(5.0).reshape(-1, 1),
                            x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                            basis=np.arange(5.0).reshape(-1, 1), A_est=np.arange(5.0).reshape(-1, 1),
                            technique='direct', alpha=1.0, l1_ratio_or_reg=1.0, group_reg=1.0, max_iter=1,
                            groups=groups)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'groups must only contain integers.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'groups must only contain integers.'


    def test_gamma_v_m_error_errors_array(self, test_all=False):

        errors = 1

        with pytest.raises(TypeError) as error_info:
            self.gamma_v_m_error(errors=errors, x=np.arange(5.0).reshape(-1, 1),
                                 Psi=np.arange(5.0).reshape(-1, 1), B=np.arange(5.0).reshape(-1, 1))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'errors must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'errors must be of type np.ndarray.'

    def test_gamma_v_m_error_errors_only_floats(self, test_all=False):

        errors = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.gamma_v_m_error(errors=errors, x=np.arange(5.0).reshape(-1, 1),
                                 Psi=np.arange(5.0).reshape(-1, 1), B=np.arange(5.0).reshape(-1, 1))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'errors must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'errors must only contain floats.'


    def test_gamma_v_m_error_x_array(self, test_all=False):

        x = 1

        with pytest.raises(TypeError) as error_info:
            self.gamma_v_m_error(errors=np.arange(5.0).reshape(-1, 1), x=x,
                                 Psi=np.arange(5.0).reshape(-1, 1), B=np.arange(5.0).reshape(-1, 1))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'x must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'x must be of type np.ndarray.'

    def test_gamma_v_m_error_x_only_floats(self, test_all=False):

        x = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.gamma_v_m_error(errors=np.arange(5.0).reshape(-1, 1), x=x,
                                 Psi=np.arange(5.0).reshape(-1, 1), B=np.arange(5.0).reshape(-1, 1))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'x must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'x must only contain floats.'


    def test_gamma_v_m_error_Psi_array(self, test_all=False):

        Psi = 1

        with pytest.raises(TypeError) as error_info:
            self.gamma_v_m_error(errors=np.arange(5.0).reshape(-1, 1), x=np.arange(5.0).reshape(-1, 1),
                                 Psi=Psi, B=np.arange(5.0).reshape(-1, 1))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Psi must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Psi must be of type np.ndarray.'

    def test_gamma_v_m_error_Psi_only_floats(self, test_all=False):

        Psi = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.gamma_v_m_error(errors=np.arange(5.0).reshape(-1, 1), x=np.arange(5.0).reshape(-1, 1),
                                 Psi=Psi, B=np.arange(5.0).reshape(-1, 1))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Psi must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Psi must only contain floats.'


    def test_gamma_v_m_error_B_array(self, test_all=False):

        B = 1

        with pytest.raises(TypeError) as error_info:
            self.gamma_v_m_error(errors=np.arange(5.0).reshape(-1, 1), x=np.arange(5.0).reshape(-1, 1),
                                 Psi=np.arange(5.0).reshape(-1, 1), B=B)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'B must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'B must be of type np.ndarray.'

    def test_gamma_v_m_error_B_only_floats(self, test_all=False):

        B = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.gamma_v_m_error(errors=np.arange(5.0).reshape(-1, 1), x=np.arange(5.0).reshape(-1, 1),
                                 Psi=np.arange(5.0).reshape(-1, 1), B=B)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'B must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'B must only contain floats.'


    def test_cov_reg_given_mean_A_est_array(self, test_all=False):

        A_est = 5

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=A_est, basis=np.arange(5.0).reshape(-1, 1), x=np.arange(5.0).reshape(-1, 1),
                                    y=np.arange(5.0).reshape(-1, 1), iterations=10, technique='direct',
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'A_est must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'A_est must be of type np.ndarray.'


    def test_cov_reg_given_mean_A_est_only_floats(self, test_all=False):

        A_est = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=A_est, basis=np.arange(5.0).reshape(-1, 1), x=np.arange(5.0).reshape(-1, 1),
                                    y=np.arange(5.0).reshape(-1, 1), iterations=10, technique='direct',
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'A_est must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'A_est must only contain floats.'


    def test_cov_reg_given_mean_basis_array(self, test_all=False):

        basis = 5

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=basis, x=np.arange(5.0).reshape(-1, 1),
                                    y=np.arange(5.0).reshape(-1, 1), iterations=10, technique='direct',
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'basis must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'basis must be of type np.ndarray.'


    def test_cov_reg_given_mean_basis_only_floats(self, test_all=False):

        basis = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=basis, x=np.arange(5.0).reshape(-1, 1),
                                    y=np.arange(5.0).reshape(-1, 1), iterations=10, technique='direct',
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'basis must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'basis must only contain floats.'


    def test_cov_reg_given_mean_x_array(self, test_all=False):

        x = 5

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1), x=x,
                                    y=np.arange(5.0).reshape(-1, 1), iterations=10, technique='direct',
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'x must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'x must be of type np.ndarray.'


    def test_cov_reg_given_mean_x_only_floats(self, test_all=False):

        x = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1), x=x,
                                    y=np.arange(5.0).reshape(-1, 1), iterations=10, technique='direct',
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'x must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'x must only contain floats.'


    def test_cov_reg_given_mean_y_array(self, test_all=False):

        y = 5

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1),
                                    y=y, iterations=10, technique='direct',
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'y must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'y must be of type np.ndarray.'


    def test_cov_reg_given_mean_y_only_floats(self, test_all=False):

        y = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1), y=y, iterations=10, technique='direct',
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'y must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'y must only contain floats.'


    def test_cov_reg_given_mean_iterations_type(self, test_all=False):

        iterations = 0

        with pytest.raises(ValueError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                                    iterations=iterations, technique='direct',
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'iterations\' must be a positive integer.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'iterations\' must be a positive integer.'


    def test_cov_reg_given_mean_technique_type(self, test_all=False):

        technique = 'not-direct'

        with pytest.raises(ValueError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                                    iterations=10, technique=technique,
                                    alpha=1.0, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'technique\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'technique\' not an acceptable value.'

    def test_cov_reg_given_mean_alpha_type(self, test_all=False):

        alpha = -0.1

        with pytest.raises(ValueError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                                    iterations=10, technique='direct',
                                    alpha=alpha, l1_ratio_or_reg=0.1, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'alpha\' must be a non-negative float.')
        else:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'alpha\' must be a non-negative float.'

    def test_cov_reg_given_mean_l1_ratio_or_reg_type(self, test_all=False):

        l1_ratio_or_reg = -0.1

        with pytest.raises(ValueError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                                    iterations=10, technique='direct',
                                    alpha=0.1, l1_ratio_or_reg=l1_ratio_or_reg, group_reg=1e-6, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'l1_ratio_or_reg\' must be a non-negative float.')
        else:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'l1_ratio_or_reg\' must be a non-negative float.'

    def test_cov_reg_given_mean_group_reg_type(self, test_all=False):

        group_reg = -0.1

        with pytest.raises(ValueError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                                    iterations=10, technique='direct',
                                    alpha=0.1, l1_ratio_or_reg=0.1, group_reg=group_reg, max_iter=10000,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'group_reg\' must be a non-negative float.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'group_reg\' must be a non-negative float.'


    def test_cov_reg_given_mean_max_iter_type(self, test_all=False):

        max_iter = 0

        with pytest.raises(ValueError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                                    iterations=10, technique='direct',
                                    alpha=0.1, l1_ratio_or_reg=0.1, group_reg=0.1, max_iter=max_iter,
                                    groups=np.arange(5))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'max_iter\' must be a positive integer.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'max_iter\' must be a positive integer.'


    def test_cov_reg_given_mean_groups_array(self, test_all=False):

        groups = 1.0

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                                    iterations=10, technique='direct',
                                    alpha=0.1, l1_ratio_or_reg=0.1, group_reg=0.1, max_iter=100,
                                    groups=groups)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'groups must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'groups must be of type np.ndarray.'

    def test_cov_reg_given_mean_groups_only_integers(self, test_all=False):

        groups = np.arange(5.0)

        with pytest.raises(TypeError) as error_info:
            self.cov_reg_given_mean(A_est=np.arange(5.0).reshape(-1, 1), basis=np.arange(5.0).reshape(-1, 1),
                                    x=np.arange(5.0).reshape(-1, 1), y=np.arange(5.0).reshape(-1, 1),
                                    iterations=10, technique='direct',
                                    alpha=0.1, l1_ratio_or_reg=0.1, group_reg=0.1, max_iter=100,
                                    groups=groups)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'groups must only contain integers.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'groups must only contain integers.'


    def test_subgrad_opt_x_tilda_array(self, test_all=False):

        x_tilda = 5

        with pytest.raises(TypeError) as error_info:
            self.subgrad_opt(x_tilda=x_tilda, y_tilda=np.arange(5.0).reshape(-1, 1), max_iter=10, alpha=0.1)

        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'x_tilda must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'x_tilda must be of type np.ndarray.'

    def test_subgrad_opt_x_tilda_only_floats(self, test_all=False):

        x_tilda = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.subgrad_opt(x_tilda=x_tilda, y_tilda=np.arange(5.0).reshape(-1, 1), max_iter=10, alpha=0.1)

        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'x_tilda must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'x_tilda must only contain floats.'


    def test_subgrad_opt_y_tilda_array(self, test_all=False):

        y_tilda = 5

        with pytest.raises(TypeError) as error_info:
            self.subgrad_opt(x_tilda=np.arange(5.0).reshape(-1, 1), y_tilda=y_tilda, max_iter=10, alpha=0.1)

        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'y_tilda must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'y_tilda must be of type np.ndarray.'

    def test_subgrad_opt_y_tilda_only_floats(self, test_all=False):

        y_tilda = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            self.subgrad_opt(x_tilda=np.arange(5.0).reshape(-1, 1), y_tilda=y_tilda, max_iter=10, alpha=0.1)

        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'y_tilda must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'y_tilda must only contain floats.'


    def test_subgrad_opt_max_iter_type(self, test_all=False):

        max_iter = 0

        with pytest.raises(ValueError) as error_info:
            self.subgrad_opt(x_tilda=np.arange(5.0).reshape(-1, 1), y_tilda=np.arange(5.0).reshape(-1, 1),
                             max_iter=max_iter, alpha=0.1)

        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'max_iter\' must be a positive integer.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'max_iter\' must be a positive integer.'


    def test_subgrad_opt_alpha_type(self, test_all=False):

        alpha = -0.1

        with pytest.raises(ValueError) as error_info:
            self.subgrad_opt(x_tilda=np.arange(5.0).reshape(-1, 1), y_tilda=np.arange(5.0).reshape(-1, 1),
                             max_iter=100, alpha=alpha)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'alpha\' must be a non-negative float.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'alpha\' must be a non-negative float.'


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

    test_rcr = RCRUnitTests(calc_B_Psi, gamma_v_m_error, cov_reg_given_mean, subgrad_opt, covregpy_dcc, dcc_loglike,
                            gp_forecast)

    test_rcr.test_m_array()
    test_rcr.test_m_only_floats()
    test_rcr.test_m_shape()

    test_rcr.test_v_array()
    test_rcr.test_v_only_floats()
    test_rcr.test_v_shape()

    test_rcr.test_m_and_v()

    test_rcr.test_x_array()
    test_rcr.test_x_only_floats()

    test_rcr.test_y_array()
    test_rcr.test_y_only_floats()

    test_rcr.test_basis_array()
    test_rcr.test_basis_only_floats()

    test_rcr.test_A_est_array()
    test_rcr.test_A_est_only_floats()

    test_rcr.test_technique_type()

    test_rcr.test_technique_type()
    test_rcr.test_alpha_type()
    test_rcr.test_l1_ratio_or_reg_type()
    test_rcr.test_group_reg_type()
    test_rcr.test_max_iter_type()

    test_rcr.test_group_array()
    test_rcr.test_groups_only_integers()

    test_rcr.test_gamma_v_m_error_errors_array()
    test_rcr.test_gamma_v_m_error_errors_only_floats()
    test_rcr.test_gamma_v_m_error_x_array()
    test_rcr.test_gamma_v_m_error_x_only_floats()
    test_rcr.test_gamma_v_m_error_Psi_array()
    test_rcr.test_gamma_v_m_error_Psi_only_floats()
    test_rcr.test_gamma_v_m_error_B_array()
    test_rcr.test_gamma_v_m_error_B_only_floats()

    test_rcr.test_cov_reg_given_mean_A_est_array()
    test_rcr.test_cov_reg_given_mean_A_est_only_floats()
    test_rcr.test_cov_reg_given_mean_basis_array()
    test_rcr.test_cov_reg_given_mean_basis_only_floats()
    test_rcr.test_cov_reg_given_mean_x_array()
    test_rcr.test_cov_reg_given_mean_x_only_floats()
    test_rcr.test_cov_reg_given_mean_y_array()
    test_rcr.test_cov_reg_given_mean_y_only_floats()
    test_rcr.test_cov_reg_given_mean_iterations_type()
    test_rcr.test_cov_reg_given_mean_technique_type()
    test_rcr.test_cov_reg_given_mean_alpha_type()
    test_rcr.test_cov_reg_given_mean_l1_ratio_or_reg_type()
    test_rcr.test_cov_reg_given_mean_group_reg_type()
    test_rcr.test_cov_reg_given_mean_max_iter_type()
    test_rcr.test_cov_reg_given_mean_groups_array()
    test_rcr.test_cov_reg_given_mean_groups_only_integers()

    test_rcr.test_subgrad_opt_x_tilda_array()
    test_rcr.test_subgrad_opt_x_tilda_only_floats()
    test_rcr.test_subgrad_opt_y_tilda_array()
    test_rcr.test_subgrad_opt_y_tilda_only_floats()
    test_rcr.test_subgrad_opt_max_iter_type()
    test_rcr.test_subgrad_opt_alpha_type()

    # test_rcr.test_returns_suitable()
    # test_rcr.test_returns_nans()
    # test_rcr.test_returns_floats()
    # test_rcr.test_dcc_loglike_params_nans()
    # test_rcr.test_dcc_loglike_params_floats()
    # test_rcr.test_dcc_loglike_returns_nans()
    # test_rcr.test_dcc_loglike_returns_floats()
    # test_rcr.test_dcc_loglike_covariance_shape()
    # test_rcr.test_dcc_loglike_covariance_nans()
    # test_rcr.test_dcc_loglike_covariance_floats()
    # test_rcr.test_gp_forecast_independent_suitable()
    # test_rcr.test_gp_forecast_independent_nans()
    # test_rcr.test_gp_forecast_independent_floats()
    # test_rcr.test_gp_forecast_dependent_suitable()
    # test_rcr.test_gp_forecast_dependent_nans()
    # test_rcr.test_gp_forecast_dependent_floats()
    test_rcr.test_all(print_all=True)

