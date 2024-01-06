

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

from CovRegpy import calc_B_Psi, gamma_v_m_error, cov_reg_given_mean, subgrad_opt
from CovRegpy_DCC import covregpy_dcc, dcc_loglike
from CovRegpy_RPP import (risk_parity_obj_fun, equal_risk_parity_weights_long_restriction,
                          equal_risk_parity_weights_short_restriction, equal_risk_parity_weights_summation_restriction,
                          global_obj_fun, global_weights, global_weights_long, global_weights_short_and_long_restrict,
                          equal_risk_parity_weights_individual_restriction)
from CovRegpy_SSA import CovRegpy_ssa
from CovRegpy_SSD import (gaussian, max_bool, spectral_obj_func_l1, spectral_obj_func_l2, gaus_param,
                          scaling_factor_obj_func, scaling_factor, CovRegpy_ssd)
from CovRegpy_X11 import henderson_kernel, henderson_weights, henderson_ma, seasonal_ma, CovRegpy_X11

sns.set(style='darkgrid')


class RCRUnitTests:

    def __init__(self, calc_B_Psi, gamma_v_m_error, cov_reg_given_mean, subgrad_opt, covregpy_dcc, dcc_loglike,
                 risk_parity_obj_fun, equal_risk_parity_weights_long_restriction,
                 equal_risk_parity_weights_short_restriction, equal_risk_parity_weights_summation_restriction,
                 global_obj_fun, global_weights, global_weights_long, global_weights_short_and_long_restrict,
                 equal_risk_parity_weights_individual_restriction, CovRegpy_ssa, gaussian, max_bool,
                 spectral_obj_func_l1, spectral_obj_func_l2, gaus_param, scaling_factor_obj_func, scaling_factor,
                 CovRegpy_ssd, henderson_kernel, henderson_weights, henderson_ma, seasonal_ma, CovRegpy_X11):

        self.calc_B_Psi = calc_B_Psi
        self.gamma_v_m_error = gamma_v_m_error
        self.cov_reg_given_mean = cov_reg_given_mean
        self.subgrad_opt = subgrad_opt

        self.covregpy_dcc = covregpy_dcc
        self.dcc_loglike = dcc_loglike

        self.risk_parity_obj_fun = risk_parity_obj_fun
        self.equal_risk_parity_weights_long_restriction = equal_risk_parity_weights_long_restriction
        self.equal_risk_parity_weights_short_restriction = equal_risk_parity_weights_short_restriction
        self.equal_risk_parity_weights_summation_restriction = equal_risk_parity_weights_summation_restriction
        self.global_obj_fun = global_obj_fun
        self.global_weights = global_weights
        self.global_weights_long = global_weights_long
        self.global_weights_short_and_long_restrict = global_weights_short_and_long_restrict
        self.equal_risk_parity_weights_individual_restriction = equal_risk_parity_weights_individual_restriction

        self.CovRegpy_ssa = CovRegpy_ssa

        self.gaussian = gaussian
        self.max_bool = max_bool
        self.spectral_obj_func_l1 = spectral_obj_func_l1
        self.spectral_obj_func_l2 = spectral_obj_func_l2
        self.gaus_param = gaus_param
        self.scaling_factor_obj_func = scaling_factor_obj_func
        self.scaling_factor = scaling_factor
        self.CovRegpy_ssd = CovRegpy_ssd

        self.henderson_kernel = henderson_kernel
        self.henderson_weights = henderson_weights
        self.henderson_ma = henderson_ma
        self.seasonal_ma = seasonal_ma
        self.CovRegpy_X11 = CovRegpy_X11

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

        test_dcc_1 = self.test_returns_suitable(test_all=True)
        test_dcc_2 = self.test_returns_nans(test_all=True)
        test_dcc_3 = self.test_returns_floats(test_all=True)
        test_dcc_4 = self.test_dcc_p_value(test_all=True)
        test_dcc_5 = self.test_dcc_q_value(test_all=True)
        test_dcc_6 = self.test_dcc_days_value(test_all=True)
        test_dcc_7 = self.test_dcc_print_correlation_bool(test_all=True)
        test_dcc_8 = self.test_dcc_rescale_bool(test_all=True)

        test_dcc_9 = self.test_dcc_loglike_params_nans(test_all=True)
        test_dcc_10 = self.test_dcc_loglike_params_floats(test_all=True)
        test_dcc_11 = self.test_dcc_loglike_returns_nans(test_all=True)
        test_dcc_12 = self.test_dcc_loglike_returns_floats(test_all=True)
        test_dcc_13 = self.test_dcc_loglike_covariance_nans(test_all=True)
        test_dcc_14 = self.test_dcc_loglike_covariance_floats(test_all=True)

        test_rpp_1 = self.test_RPP_risk_parity_obj_fun_x_array(test_all=True)
        test_rpp_2 = self.test_RPP_risk_parity_obj_fun_x_only_floats(test_all=True)
        test_rpp_3 = self.test_RPP_risk_parity_obj_fun_p_cov_array(test_all=True)
        test_rpp_4 = self.test_RPP_risk_parity_obj_fun_p_cov_only_floats(test_all=True)
        test_rpp_5 = self.test_RPP_risk_parity_obj_fun_p_cov_psd(test_all=True)
        test_rpp_6 = self.test_RPP_risk_parity_obj_fun_x_p_cov_shape(test_all=True)
        test_rpp_7 = self.test_RPP_risk_parity_obj_fun_rb_array(test_all=True)
        test_rpp_8 = self.test_RPP_risk_parity_obj_fun_rb_only_floats(test_all=True)
        test_rpp_9 = self.test_RPP_risk_parity_obj_fun_x_rb_shape(test_all=True)

        test_rpp_10 = self.test_RPP_equal_risk_parity_weights_long_restriction_cov_array(test_all=True)
        test_rpp_11 = self.test_RPP_equal_risk_parity_weights_long_restriction_cov_only_floats(test_all=True)
        test_rpp_12 = self.test_RPP_equal_risk_parity_weights_long_restriction_cov_psd(test_all=True)

        test_rpp_13 = self.test_RPP_equal_risk_parity_weights_short_restriction_cov_array(test_all=True)
        test_rpp_14 = self.test_RPP_equal_risk_parity_weights_short_restriction_cov_only_floats(test_all=True)
        test_rpp_15 = self.test_RPP_equal_risk_parity_weights_short_restriction_cov_psd(test_all=True)
        test_rpp_16 = self.test_RPP_equal_risk_parity_weights_short_restriction_short_limit_value(test_all=True)

        test_rpp_17 = self.test_RPP_equal_risk_parity_weights_summation_restriction_cov_array(test_all=True)
        test_rpp_18 = self.test_RPP_equal_risk_parity_weights_summation_restriction_cov_only_floats(test_all=True)
        test_rpp_19 = self.test_RPP_equal_risk_parity_weights_summation_restriction_cov_psd(test_all=True)
        test_rpp_20 = self.test_RPP_equal_risk_parity_weights_summation_restriction_short_limit_value(test_all=True)
        test_rpp_21 = self.test_RPP_equal_risk_parity_weights_summation_restriction_long_limit_value(test_all=True)

        test_rpp_22 = self.test_RPP_global_obj_fun_x_array(test_all=True)
        test_rpp_23 = self.test_RPP_global_obj_fun_x_only_floats(test_all=True)
        test_rpp_24 = self.test_RPP_global_obj_fun_p_cov_array(test_all=True)
        test_rpp_25 = self.test_RPP_global_obj_fun_p_cov_only_floats(test_all=True)
        test_rpp_26 = self.test_RPP_global_obj_fun_p_cov_psd(test_all=True)
        test_rpp_27 = self.test_RPP_global_obj_fun_x_p_cov_shape(test_all=True)

        test_rpp_28 = self.test_RPP_global_weights_cov_array(test_all=True)
        test_rpp_29 = self.test_RPP_global_weights_cov_only_floats(test_all=True)
        test_rpp_30 = self.test_RPP_global_weights_cov_psd(test_all=True)

        test_rpp_31 = self.test_RPP_global_weights_long_cov_array(test_all=True)
        test_rpp_32 = self.test_RPP_global_weights_long_cov_only_floats(test_all=True)
        test_rpp_33 = self.test_RPP_global_weights_long_cov_psd(test_all=True)

        test_rpp_34 = self.test_RPP_global_weights_short_and_long_restrict_cov_array(test_all=True)
        test_rpp_35 = self.test_RPP_global_weights_short_and_long_restrict_cov_only_floats(test_all=True)
        test_rpp_36 = self.test_RPP_global_weights_short_and_long_restrict_cov_psd(test_all=True)
        test_rpp_37 = self.test_RPP_global_weights_short_and_long_restrict_b_value(test_all=True)
        test_rpp_38 = self.test_RPP_global_weights_short_and_long_restrict_a_value(test_all=True)

        test_rpp_39 = self.test_RPP_equal_risk_parity_weights_individual_restriction_cov_array(test_all=True)
        test_rpp_40 = self.test_RPP_equal_risk_parity_weights_individual_restriction_cov_only_floats(test_all=True)
        test_rpp_41 = self.test_RPP_equal_risk_parity_weights_individual_restriction_cov_psd(test_all=True)
        test_rpp_42 = self.test_RPP_equal_risk_parity_weights_individual_restriction_short_limit_value(test_all=True)
        test_rpp_43 = self.test_RPP_equal_risk_parity_weights_individual_restriction_long_limit_value(test_all=True)

        test_ssa_1 = self.test_SSA_time_series_array(test_all=True)
        test_ssa_2 = self.test_SSA_time_series_only_floats(test_all=True)
        test_ssa_3 = self.test_SSA_L_value(test_all=True)
        test_ssa_4 = self.test_SSA_est_value(test_all=True)
        test_ssa_5 = self.test_SSA_plot_bool(test_all=True)
        test_ssa_6 = self.test_SSA_KS_test_bool(test_all=True)
        test_ssa_7 = self.test_SSA_plot_KS_test_bool(test_all=True)
        test_ssa_8 = self.test_SSA_KS_scale_limit_value(test_all=True)
        test_ssa_9 = self.test_SSA_max_eig_ratio_value(test_all=True)
        test_ssa_10 = self.test_SSA_KS_start_value(test_all=True)
        test_ssa_11 = self.test_SSA_KS_end_value(test_all=True)
        test_ssa_12 = self.test_SSA_KS_interval_value(test_all=True)

        test_ssd_1 = self.test_SSD_gaussian_f_array(test_all=True)
        test_ssd_2 = self.test_SSD_gaussian_f_only_floats(test_all=True)
        test_ssd_3 = self.test_SSD_gaussian_A_value(test_all=True)
        test_ssd_4 = self.test_SSD_gaussian_mu_value(test_all=True)
        test_ssd_5 = self.test_SSD_gaussian_sigma_value(test_all=True)

        test_ssd_6 = self.test_SSD_max_bool_time_series_array(test_all=True)
        test_ssd_7 = self.test_SSD_max_bool_time_series_only_floats(test_all=True)

        test_ssd_8 = self.test_SSD_spectral_obj_func_l1_theta_array(test_all=True)
        test_ssd_9 = self.test_SSD_spectral_obj_func_l1_theta_only_floats(test_all=True)
        test_ssd_10 = self.test_SSD_spectral_obj_func_l1_theta_length(test_all=True)
        test_ssd_11 = self.test_SSD_spectral_obj_func_l1_f_array(test_all=True)
        test_ssd_12 = self.test_SSD_spectral_obj_func_l1_f_only_floats(test_all=True)
        test_ssd_13 = self.test_SSD_spectral_obj_func_l1_mu_1_value(test_all=True)
        test_ssd_14 = self.test_SSD_spectral_obj_func_l1_mu_2_value(test_all=True)
        test_ssd_15 = self.test_SSD_spectral_obj_func_l1_mu_3_value(test_all=True)
        test_ssd_16 = self.test_SSD_spectral_obj_func_l1_spectrum_array(test_all=True)
        test_ssd_17 = self.test_SSD_spectral_obj_func_l1_spectrum_only_floats(test_all=True)
        test_ssd_18 = self.test_SSD_spectral_obj_func_l1_f_and_spectrum_lengths(test_all=True)

        test_ssd_19 = self.test_SSD_spectral_obj_func_l2_theta_array(test_all=True)
        test_ssd_20 = self.test_SSD_spectral_obj_func_l2_theta_only_floats(test_all=True)
        test_ssd_21 = self.test_SSD_spectral_obj_func_l2_theta_length(test_all=True)
        test_ssd_22 = self.test_SSD_spectral_obj_func_l2_f_array(test_all=True)
        test_ssd_23 = self.test_SSD_spectral_obj_func_l2_f_only_floats(test_all=True)
        test_ssd_24 = self.test_SSD_spectral_obj_func_l2_mu_1_value(test_all=True)
        test_ssd_25 = self.test_SSD_spectral_obj_func_l2_mu_2_value(test_all=True)
        test_ssd_26 = self.test_SSD_spectral_obj_func_l2_mu_3_value(test_all=True)
        test_ssd_27 = self.test_SSD_spectral_obj_func_l2_spectrum_array(test_all=True)
        test_ssd_28 = self.test_SSD_spectral_obj_func_l2_spectrum_only_floats(test_all=True)
        test_ssd_29 = self.test_SSD_spectral_obj_func_l2_f_and_spectrum_lengths(test_all=True)

        test_ssd_30 = self.test_SSD_gaus_param_w0_array(test_all=True)
        test_ssd_31 = self.test_SSD_gaus_param_w0_only_floats(test_all=True)
        test_ssd_32 = self.test_SSD_gaus_param_w0_length(test_all=True)
        test_ssd_33 = self.test_SSD_gaus_param_f_array(test_all=True)
        test_ssd_34 = self.test_SSD_gaus_param_f_only_floats(test_all=True)
        test_ssd_35 = self.test_SSD_gaus_param_mu_1_value(test_all=True)
        test_ssd_36 = self.test_SSD_gaus_param_mu_2_value(test_all=True)
        test_ssd_37 = self.test_SSD_gaus_param_mu_3_value(test_all=True)
        test_ssd_38 = self.test_SSD_gaus_param_spectrum_array(test_all=True)
        test_ssd_39 = self.test_SSD_gaus_param_spectrum_only_floats(test_all=True)
        test_ssd_40 = self.test_SSD_gaus_param_f_and_spectrum_lengths(test_all=True)
        test_ssd_41 = self.test_SSD_gaus_param_method_value(test_all=True)

        test_ssd_42 = self.test_SSD_scaling_factor_obj_func_a_value(test_all=True)
        test_ssd_43 = self.test_SSD_scaling_factor_obj_func_residual_time_series_array(test_all=True)
        test_ssd_44 = self.test_SSD_scaling_factor_obj_func_residual_time_series_only_floats(test_all=True)
        test_ssd_45 = self.test_SSD_scaling_factor_obj_func_trend_estimate_array(test_all=True)
        test_ssd_46 = self.test_SSD_scaling_factor_obj_func_trend_estimate_only_floats(test_all=True)
        test_ssd_47 = self.test_SSD_scaling_factor_obj_func_residual_time_series_and_trend_estimate_lengths(
            test_all=True)

        test_ssd_48 = self.test_SSD_scaling_factor_residual_time_series_array(test_all=True)
        test_ssd_49 = self.test_SSD_scaling_factor_residual_time_series_only_floats(test_all=True)
        test_ssd_50 = self.test_SSD_scaling_factor_trend_estimate_array(test_all=True)
        test_ssd_51 = self.test_SSD_scaling_factor_trend_estimate_only_floats(test_all=True)
        test_ssd_52 = self.test_SSD_scaling_factor_residual_time_series_and_trend_estimate_lengths(test_all=True)

        test_ssd_53 = self.test_SSD_CovRegpy_ssd_time_series_array(test_all=True)
        test_ssd_54 = self.test_SSD_CovRegpy_ssd_time_series_only_floats(test_all=True)
        test_ssd_55 = self.test_SSD_CovRegpy_ssd_initial_trend_ratio_value(test_all=True)
        test_ssd_56 = self.test_SSD_CovRegpy_ssd_nmse_threshold_value(test_all=True)
        test_ssd_57 = self.test_SSD_CovRegpy_ssd_plot_bool(test_all=True)
        test_ssd_58 = self.test_SSD_CovRegpy_ssd_debug_bool(test_all=True)
        test_ssd_59 = self.test_SSD_CovRegpy_ssd_method_type(test_all=True)

        test_x11_1 = self.test_X11_henderson_kernel_order_value(test_all=True)
        test_x11_2 = self.test_X11_henderson_kernel_start_value(test_all=True)
        test_x11_3 = self.test_X11_henderson_kernel_end_value(test_all=True)

        test_x11_4 = self.test_X11_henderson_weights_order_value(test_all=True)
        test_x11_5 = self.test_X11_henderson_weights_start_value(test_all=True)
        test_x11_6 = self.test_X11_henderson_weights_end_value(test_all=True)
        test_x11_7 = self.test_X11_henderson_weights_start_end_value(test_all=True)

        test_x11_8 = self.test_X11_henderson_ma_time_series_array(test_all=True)
        test_x11_9 = self.test_X11_henderson_ma_time_series_only_floats(test_all=True)
        test_x11_10 = self.test_X11_henderson_ma_order_value(test_all=True)
        test_x11_11 = self.test_X11_henderson_ma_method_type(test_all=True)

        test_x11_12 = self.test_X11_seasonal_ma_time_series_array(test_all=True)
        test_x11_13 = self.test_X11_seasonal_ma_time_series_only_floats(test_all=True)
        test_x11_14 = self.test_X11_seasonal_ma_factors_type(test_all=True)
        test_x11_15 = self.test_X11_seasonal_ma_seasonality_type(test_all=True)

        test_x11_16 = self.test_X11_time_array(test_all=True)
        test_x11_17 = self.test_X11_time_only_floats(test_all=True)
        test_x11_18 = self.test_X11_time_series_array(test_all=True)
        test_x11_19 = self.test_X11_time_series_only_floats(test_all=True)
        test_x11_20 = self.test_X11_time_series_error_length(test_all=True)
        test_x11_21 = self.test_X11_seasonality_value(test_all=True)
        test_x11_22 = self.test_X11_seasonal_factor_value(test_all=True)
        test_x11_23 = self.test_X11_trend_window_width_1_value(test_all=True)
        test_x11_24 = self.test_X11_trend_window_width_2_value(test_all=True)
        test_x11_25 = self.test_X11_trend_window_width_3_value(test_all=True)

        tests = [test_covreg_1, test_covreg_2, test_covreg_3, test_covreg_4, test_covreg_5, test_covreg_6,
                 test_covreg_7, test_covreg_8, test_covreg_9, test_covreg_10, test_covreg_11, test_covreg_12,
                 test_covreg_13, test_covreg_14, test_covreg_15, test_covreg_16, test_covreg_17, test_covreg_18,
                 test_covreg_19, test_covreg_20, test_covreg_21, test_covreg_22, test_covreg_23, test_covreg_24,
                 test_covreg_25, test_covreg_26, test_covreg_27, test_covreg_28, test_covreg_29, test_covreg_30,
                 test_covreg_31, test_covreg_32, test_covreg_33, test_covreg_34, test_covreg_35, test_covreg_36,
                 test_covreg_37, test_covreg_38, test_covreg_39, test_covreg_40, test_covreg_41, test_covreg_42,
                 test_covreg_43, test_covreg_44, test_covreg_45, test_covreg_46, test_covreg_47, test_covreg_48,
                 test_covreg_49, test_covreg_50, test_covreg_51, test_covreg_52, test_covreg_53,

                 test_dcc_1, test_dcc_2, test_dcc_3, test_dcc_4, test_dcc_5, test_dcc_6, test_dcc_7, test_dcc_8,
                 test_dcc_9, test_dcc_10, test_dcc_11, test_dcc_12, test_dcc_13, test_dcc_14,

                 test_rpp_1, test_rpp_2, test_rpp_3, test_rpp_4, test_rpp_5, test_rpp_6, test_rpp_7, test_rpp_8,
                 test_rpp_9, test_rpp_10, test_rpp_11, test_rpp_12, test_rpp_13, test_rpp_14, test_rpp_15, test_rpp_16,
                 test_rpp_17, test_rpp_18, test_rpp_19, test_rpp_20, test_rpp_21, test_rpp_22, test_rpp_23, test_rpp_24,
                 test_rpp_25, test_rpp_26, test_rpp_27, test_rpp_28, test_rpp_29, test_rpp_30, test_rpp_31, test_rpp_32,
                 test_rpp_33, test_rpp_34, test_rpp_35, test_rpp_36, test_rpp_37, test_rpp_38, test_rpp_39, test_rpp_40,
                 test_rpp_41, test_rpp_42, test_rpp_43,

                 test_ssa_1, test_ssa_2, test_ssa_3, test_ssa_4, test_ssa_5, test_ssa_6, test_ssa_7, test_ssa_8,
                 test_ssa_9, test_ssa_10, test_ssa_11, test_ssa_12,

                 test_ssd_1, test_ssd_2, test_ssd_3, test_ssd_4, test_ssd_5, test_ssd_6, test_ssd_7, test_ssd_8,
                 test_ssd_9, test_ssd_10, test_ssd_11, test_ssd_12, test_ssd_13, test_ssd_14, test_ssd_15, test_ssd_16,
                 test_ssd_17, test_ssd_18, test_ssd_19, test_ssd_20, test_ssd_21, test_ssd_22, test_ssd_23, test_ssd_24,
                 test_ssd_25, test_ssd_26, test_ssd_27, test_ssd_28, test_ssd_29, test_ssd_30, test_ssd_31, test_ssd_32,
                 test_ssd_33, test_ssd_34, test_ssd_35, test_ssd_36, test_ssd_37, test_ssd_38, test_ssd_39, test_ssd_40,
                 test_ssd_41, test_ssd_42, test_ssd_43, test_ssd_44, test_ssd_45, test_ssd_46, test_ssd_47, test_ssd_48,
                 test_ssd_49, test_ssd_50, test_ssd_51, test_ssd_52, test_ssd_53, test_ssd_54, test_ssd_55, test_ssd_56,
                 test_ssd_57, test_ssd_58, test_ssd_59,

                 test_x11_1, test_x11_2, test_x11_3, test_x11_4, test_x11_5, test_x11_6, test_x11_7, test_x11_8,
                 test_x11_9, test_x11_10, test_x11_11, test_x11_12, test_x11_13, test_x11_14, test_x11_15, test_x11_16,
                 test_x11_17, test_x11_18, test_x11_19, test_x11_20, test_x11_21, test_x11_22, test_x11_23, test_x11_24,
                 test_x11_25]

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
            self.covregpy_dcc(returns_matrix_test, p=3, q=3, days=10, print_correlation=False, rescale=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Returns must be of type np.ndarray and pd.Dataframe.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Returns must be of type np.ndarray and pd.Dataframe.'

    def test_returns_nans(self, test_all=False):

        returns_matrix_test = np.asarray([[1.0, 1.0], [1.0, 1.0]])
        returns_matrix_test[0, 0] = np.nan

        with pytest.raises(TypeError) as error_info:
            self.covregpy_dcc(returns_matrix_test, p=3, q=3, days=10, print_correlation=False, rescale=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Returns must not contain nans.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Returns must not contain nans.'

    def test_returns_floats(self, test_all=False):

        returns_matrix_test = np.asarray([[1, 1], [1, 1]])
        returns_matrix_test[0, 0] = True

        with pytest.raises(TypeError) as error_info:
            self.covregpy_dcc(returns_matrix_test, p=3, q=3, days=10, print_correlation=False, rescale=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'Returns must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'Returns must only contain floats.'


    def test_dcc_p_value(self, test_all=False):

        p_bool = True
        for p in [6.0, -1]:
            with pytest.raises(ValueError) as error_info:
                self.covregpy_dcc(returns_matrix=np.asarray([[1., 1.], [1., 1.]]), p=p, q=3, days=10,
                                  print_correlation=False, rescale=False)
            p_bool = \
                p_bool and (error_info.type is ValueError and
                              error_info.value.args[0] ==
                              '\'p\' must be a positive integer.')

        if (not test_all) and p_bool:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'p\' must be a positive integer.')
        elif p_bool:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'p\' must be a positive integer.'


    def test_dcc_q_value(self, test_all=False):

        q_bool = True
        for q in [6.0, -1]:
            with pytest.raises(ValueError) as error_info:
                self.covregpy_dcc(returns_matrix=np.asarray([[1., 1.], [1., 1.]]), p=3, q=q, days=10,
                                  print_correlation=False, rescale=False)
            q_bool = \
                q_bool and (error_info.type is ValueError and
                              error_info.value.args[0] ==
                              '\'q\' must be a positive integer.')

        if (not test_all) and q_bool:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'q\' must be a positive integer.')
        elif q_bool:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'q\' must be a positive integer.'


    def test_dcc_days_value(self, test_all=False):

        days_bool = True
        for days in [6.0, -1]:
            with pytest.raises(ValueError) as error_info:
                self.covregpy_dcc(returns_matrix=np.asarray([[1., 1.], [1., 1.]]), p=3, q=3, days=days,
                                  print_correlation=False, rescale=False)
            days_bool = \
                days_bool and (error_info.type is ValueError and
                              error_info.value.args[0] ==
                              '\'days\' must be a positive integer.')

        if (not test_all) and days_bool:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'days\' must be a positive integer.')
        elif days_bool:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'days\' must be a positive integer.'


    def test_dcc_print_correlation_bool(self, test_all=False):

        print_correlation = 'not_bool'

        with pytest.raises(TypeError) as error_info:
            self.covregpy_dcc(returns_matrix=np.asarray([[1., 1.], [1., 1.]]), p=3, q=3, days=10,
                              print_correlation=print_correlation, rescale=False)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'print_correlation\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'print_correlation\' must be boolean.'


    def test_dcc_rescale_bool(self, test_all=False):

        rescale = 'not_bool'

        with pytest.raises(TypeError) as error_info:
            self.covregpy_dcc(returns_matrix=np.asarray([[1., 1.], [1., 1.]]), p=3, q=3, days=10,
                              print_correlation=False, rescale=rescale)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'rescale\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'rescale\' must be boolean.'


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


    def test_X11_henderson_kernel_order_value(self, test_all=False):

        order_bool = True
        for order in [6.0, -1, 8]:
            with pytest.raises(ValueError) as error_info:
                henderson_kernel(order=order, start=None, end=None)
            order_bool = order_bool and (error_info.type is ValueError and
                                         error_info.value.args[0] ==
                                         '\'order\' must be a positive odd integer.')

        if (not test_all) and order_bool:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'order\' must be a positive odd integer.')
        elif order_bool:
            return error_info.type is ValueError and error_info.value.args[0] == '\'order\' must be a positive odd integer.'


    def test_X11_henderson_kernel_start_value(self, test_all=False):

        start_bool = True
        for start in [6.0, 1, -8]:
            with pytest.raises(ValueError) as error_info:
                henderson_kernel(order=13, start=start, end=None)
            start_bool = \
                start_bool and (error_info.type is ValueError and
                                error_info.value.args[0] ==
                                '\'start\' must be non-positive integer of correct magnitude.')

        if (not test_all) and start_bool:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'start\' must be non-positive integer of correct magnitude.')
        elif start_bool:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'start\' must be non-positive integer of correct magnitude.'


    def test_X11_henderson_kernel_end_value(self, test_all=False):

        end_bool = True
        for end in [6.0, -1, 8]:
            with pytest.raises(ValueError) as error_info:
                henderson_kernel(order=13, start=None, end=end)
            end_bool = \
                end_bool and (error_info.type is ValueError and
                              error_info.value.args[0] ==
                              '\'end\' must be non-negative integer of correct magnitude.')

        if (not test_all) and end_bool:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'end\' must be non-negative integer of correct magnitude.')
        elif end_bool:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'end\' must be non-negative integer of correct magnitude.'


    def test_X11_henderson_weights_order_value(self, test_all=False):

        order_bool = True
        for order in [6.0, -1, 8]:
            with pytest.raises(ValueError) as error_info:
                henderson_weights(order=order, start=None, end=None)
            order_bool = order_bool and (error_info.type is ValueError and
                                         error_info.value.args[0] ==
                                         '\'order\' must be a positive odd integer.')

        if (not test_all) and order_bool:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'order\' must be a positive odd integer.')
        elif order_bool:
            return error_info.type is ValueError and error_info.value.args[0] == '\'order\' must be a positive odd integer.'


    def test_X11_henderson_weights_start_value(self, test_all=False):

        start_bool = True
        for start in [6.0, 1, -8]:
            with pytest.raises(ValueError) as error_info:
                henderson_weights(order=13, start=start, end=None)
            start_bool = \
                start_bool and (error_info.type is ValueError and
                                error_info.value.args[0] ==
                                '\'start\' must be non-positive integer of correct magnitude.')

        if (not test_all) and start_bool:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'start\' must be non-positive integer of correct magnitude.')
        elif start_bool:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'start\' must be non-positive integer of correct magnitude.'


    def test_X11_henderson_weights_end_value(self, test_all=False):

        end_bool = True
        for end in [6.0, -1, 8]:
            with pytest.raises(ValueError) as error_info:
                henderson_weights(order=13, start=None, end=end)
            end_bool = \
                end_bool and (error_info.type is ValueError and
                              error_info.value.args[0] ==
                              '\'end\' must be non-negative integer of correct magnitude.')

        if (not test_all) and end_bool:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'end\' must be non-negative integer of correct magnitude.')
        elif end_bool:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'end\' must be non-negative integer of correct magnitude.'


    def test_X11_henderson_weights_start_end_value(self, test_all=False):

        start = -6
        end = 4
        with pytest.raises(ValueError) as error_info:
            henderson_weights(order=13, start=start, end=end)

        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[
                0] == '\'start\' and \'end\' must be equal and opposite.')
        else:
            return error_info.type is ValueError and error_info.value.args[
                0] == '\'start\' and \'end\' must be equal and opposite.'


    def test_X11_henderson_ma_time_series_array(self, test_all=False):

        time_series = 5

        with pytest.raises(TypeError) as error_info:
            henderson_ma(time_series, order=13, method='kernel')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time_series must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time_series must be of type np.ndarray.'


    def test_X11_henderson_ma_time_series_only_floats(self, test_all=False):

        time_series = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            henderson_ma(time_series, order=13, method='kernel')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time_series must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time_series must only contain floats.'


    def test_X11_henderson_ma_order_value(self, test_all=False):

        order_bool = True
        for order in [6.0, -1, 8]:
            with pytest.raises(ValueError) as error_info:
                henderson_ma(time_series=np.array(1000.0), order=order, method='kernel')
            order_bool = order_bool and (error_info.type is ValueError and
                                         error_info.value.args[0] ==
                                         '\'order\' must be a positive odd integer.')

        if (not test_all) and order_bool:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'order\' must be a positive odd integer.')
        elif order_bool:
            return error_info.type is ValueError and error_info.value.args[0] == '\'order\' must be a positive odd integer.'


    def test_X11_henderson_ma_method_type(self, test_all=False):

        method = 'not-direct'

        with pytest.raises(ValueError) as error_info:
            henderson_ma(time_series=np.array(1000.0), order=13, method=method)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'method\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'method\' not an acceptable value.'


    def test_X11_seasonal_ma_time_series_array(self, test_all=False):

        time_series = 5

        with pytest.raises(TypeError) as error_info:
            seasonal_ma(time_series, factors='3x3', seasonality='annual')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time_series must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time_series must be of type np.ndarray.'


    def test_X11_seasonal_ma_time_series_only_floats(self, test_all=False):

        time_series = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            seasonal_ma(time_series, factors='3x3', seasonality='annual')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time_series must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time_series must only contain floats.'


    def test_X11_seasonal_ma_factors_type(self, test_all=False):

        factors = '3x11'

        with pytest.raises(ValueError) as error_info:
            seasonal_ma(time_series=np.arange(1000.0), factors=factors, seasonality='annual')
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'factors\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'factors\' not an acceptable value.'


    def test_X11_seasonal_ma_seasonality_type(self, test_all=False):

        seasonality = 'monthly'

        with pytest.raises(ValueError) as error_info:
            seasonal_ma(time_series=np.arange(1000.0), factors='3x9', seasonality=seasonality)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'seasonality\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'seasonality\' not an acceptable value.'


    def test_X11_time_array(self, test_all=False):

        time = 5

        with pytest.raises(TypeError) as error_info:
            CovRegpy_X11(time, time_series=np.arange(100.0), seasonality='annual', seasonal_factor='3x3',
                         trend_window_width_1=13, trend_window_width_2=13, trend_window_width_3=13)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time must be of type np.ndarray.'


    def test_X11_time_only_floats(self, test_all=False):

        time = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            CovRegpy_X11(time, time_series=np.arange(100.0), seasonality='annual', seasonal_factor='3x3',
                         trend_window_width_1=13, trend_window_width_2=13, trend_window_width_3=13)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time must only contain floats.'


    def test_X11_time_series_array(self, test_all=False):

        time_series = 5

        with pytest.raises(TypeError) as error_info:
            CovRegpy_X11(time=np.arange(100.0), time_series=time_series, seasonality='annual', seasonal_factor='3x3',
                         trend_window_width_1=13, trend_window_width_2=13, trend_window_width_3=13)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time_series must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time_series must be of type np.ndarray.'


    def test_X11_time_series_only_floats(self, test_all=False):

        time_series = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            CovRegpy_X11(time=np.arange(100.0), time_series=time_series, seasonality='annual', seasonal_factor='3x3',
                         trend_window_width_1=13, trend_window_width_2=13, trend_window_width_3=13)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time_series must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time_series must only contain floats.'


    def test_X11_time_series_error_length(self, test_all=False):

        time = np.arange(101.0)
        time_series = np.arange(100.0)

        with pytest.raises(ValueError) as error_info:
            CovRegpy_X11(time=time, time_series=time_series, seasonality='annual', seasonal_factor='3x3',
                         trend_window_width_1=13, trend_window_width_2=13, trend_window_width_3=13)
        if not test_all:
            print(error_info.type is ValueError and
                  error_info.value.args[0] == 'time_series and time are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   'time_series and time are incompatible lengths.'


    def test_X11_seasonality_value(self, test_all=False):

        seasonality = 'monthly'

        with pytest.raises(ValueError) as error_info:
            CovRegpy_X11(time=np.arange(100.0), time_series=np.arange(100.0), seasonality=seasonality,
                         seasonal_factor='3x3', trend_window_width_1=13, trend_window_width_2=13,
                         trend_window_width_3=13)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'seasonality\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'seasonality\' not an acceptable value.'


    def test_X11_seasonal_factor_value(self, test_all=False):

        seasonal_factor='3x11'

        with pytest.raises(ValueError) as error_info:
            CovRegpy_X11(time=np.arange(100.0), time_series=np.arange(100.0), seasonality='quarterly',
                         seasonal_factor=seasonal_factor, trend_window_width_1=13, trend_window_width_2=13,
                         trend_window_width_3=13)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'seasonal_factor\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'seasonal_factor\' not an acceptable value.'


    def test_X11_trend_window_width_1_value(self, test_all=False):

        trend_window_width_1_bool = True
        for trend_window_width_1 in [6.0, -1, 8]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_X11(time=np.arange(100.0), time_series=np.arange(100.0), seasonality='quarterly',
                             seasonal_factor='3x9', trend_window_width_1=trend_window_width_1, trend_window_width_2=13,
                             trend_window_width_3=13)
            trend_window_width_1_bool = \
                trend_window_width_1_bool and (error_info.type is ValueError and
                                               error_info.value.args[0] ==
                                               '\'trend_window_width_1\' must be a positive odd integer.')

        if (not test_all) and trend_window_width_1_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'trend_window_width_1\' must be a positive odd integer.')
        elif trend_window_width_1_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'trend_window_width_1\' must be a positive odd integer.'


    def test_X11_trend_window_width_2_value(self, test_all=False):

        trend_window_width_2_bool = True
        for trend_window_width_2 in [6.0, -1, 8]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_X11(time=np.arange(100.0), time_series=np.arange(100.0), seasonality='quarterly',
                             seasonal_factor='3x9', trend_window_width_1=13, trend_window_width_2=trend_window_width_2,
                             trend_window_width_3=13)
            trend_window_width_2_bool = \
                trend_window_width_2_bool and (error_info.type is ValueError and
                                               error_info.value.args[0] ==
                                               '\'trend_window_width_2\' must be a positive odd integer.')

        if (not test_all) and trend_window_width_2_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'trend_window_width_2\' must be a positive odd integer.')
        elif trend_window_width_2_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'trend_window_width_2\' must be a positive odd integer.'


    def test_X11_trend_window_width_3_value(self, test_all=False):

        trend_window_width_3_bool = True
        for trend_window_width_3 in [6.0, -1, 8]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_X11(time=np.arange(100.0), time_series=np.arange(100.0), seasonality='quarterly',
                             seasonal_factor='3x9', trend_window_width_1=13, trend_window_width_2=13,
                             trend_window_width_3=trend_window_width_3)
            trend_window_width_3_bool = \
                trend_window_width_3_bool and (error_info.type is ValueError and
                                               error_info.value.args[0] ==
                                               '\'trend_window_width_3\' must be a positive odd integer.')

        if (not test_all) and trend_window_width_3_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'trend_window_width_3\' must be a positive odd integer.')
        elif trend_window_width_3_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'trend_window_width_3\' must be a positive odd integer.'


    def test_SSA_time_series_array(self, test_all=False):

        time_series = 5

        with pytest.raises(TypeError) as error_info:
            CovRegpy_ssa(time_series, L=10, est=3, plot=False, KS_test=False, plot_KS_test=False, KS_scale_limit=1.0,
                         max_eig_ratio=0.0001, KS_start=10, KS_end=100, KS_interval=10)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time_series must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time_series must be of type np.ndarray.'


    def test_SSA_time_series_only_floats(self, test_all=False):

        time_series = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            CovRegpy_ssa(time_series, L=10, est=3, plot=False, KS_test=False, plot_KS_test=False, KS_scale_limit=1.0,
                         max_eig_ratio=0.0001, KS_start=10, KS_end=100, KS_interval=10)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == 'time_series must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == 'time_series must only contain floats.'


    def test_SSA_L_value(self, test_all=False):

        L_bool = True
        for L in [6.0, -1, 120]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_ssa(time_series=np.arange(100.0), L=L, est=3, plot=False, KS_test=False, plot_KS_test=False,
                             KS_scale_limit=1.0,
                             max_eig_ratio=0.0001, KS_start=10, KS_end=100, KS_interval=10)
            L_bool = \
                L_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'L\' must be a positive integer of appropriate magnitude: L < len(time_series).')

        if (not test_all) and L_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'L\' must be a positive integer of appropriate magnitude: L < len(time_series).')
        elif L_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'L\' must be a positive integer of appropriate magnitude: L < len(time_series).'


    def test_SSA_est_value(self, test_all=False):

        est_bool = True
        for est in [6.0, -1, 30]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_ssa(time_series=np.arange(100.0), L=20, est=est, plot=False, KS_test=False,
                             plot_KS_test=False, KS_scale_limit=1.0,
                             max_eig_ratio=0.0001, KS_start=10, KS_end=100, KS_interval=10)
            est_bool = \
                est_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'est\' must be a positive integer of appropriate magnitude: est <= L.')

        if (not test_all) and est_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'est\' must be a positive integer of appropriate magnitude: est <= L.')
        elif est_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'est\' must be a positive integer of appropriate magnitude: est <= L.'


    def test_SSA_plot_bool(self, test_all=False):

        plot = 'not_bool'

        with pytest.raises(TypeError) as error_info:
            CovRegpy_ssa(time_series=np.arange(100.0), L=20, est=5, plot=plot, KS_test=False,
                         plot_KS_test=False, KS_scale_limit=1.0,
                         max_eig_ratio=0.0001, KS_start=10, KS_end=100, KS_interval=10)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'plot\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'plot\' must be boolean.'


    def test_SSA_KS_test_bool(self, test_all=False):

        KS_test = 'not_bool'

        with pytest.raises(TypeError) as error_info:
            CovRegpy_ssa(time_series=np.arange(100.0), L=20, est=5, plot=False, KS_test=KS_test,
                         plot_KS_test=False, KS_scale_limit=1.0,
                         max_eig_ratio=0.0001, KS_start=10, KS_end=100, KS_interval=10)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'KS_test\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'KS_test\' must be boolean.'


    def test_SSA_plot_KS_test_bool(self, test_all=False):

        plot_KS_test = 'not_bool'

        with pytest.raises(TypeError) as error_info:
            CovRegpy_ssa(time_series=np.arange(100.0), L=20, est=5, plot=False, KS_test=False,
                         plot_KS_test=plot_KS_test, KS_scale_limit=1.0,
                         max_eig_ratio=0.0001, KS_start=10, KS_end=100, KS_interval=10)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'plot_KS_test\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'plot_KS_test\' must be boolean.'


    def test_SSA_KS_scale_limit_value(self, test_all=False):

        KS_scale_limit_bool = True
        for KS_scale_limit in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_ssa(time_series=np.arange(100.0), L=20, est=10, plot=False, KS_test=False,
                             plot_KS_test=False, KS_scale_limit=KS_scale_limit,
                             max_eig_ratio=0.0001, KS_start=10, KS_end=100, KS_interval=10)
            KS_scale_limit_bool = \
                KS_scale_limit_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'KS_scale_limit\' must be a positive float.')

        if (not test_all) and KS_scale_limit_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'KS_scale_limit\' must be a positive float.')
        elif KS_scale_limit_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'KS_scale_limit\' must be a positive float.'


    def test_SSA_max_eig_ratio_value(self, test_all=False):

        max_eig_ratio_bool = True
        for max_eig_ratio in [1, -0.1, 1.1]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_ssa(time_series=np.arange(100.0), L=20, est=10, plot=False, KS_test=False,
                             plot_KS_test=False, KS_scale_limit=1.0,
                             max_eig_ratio=max_eig_ratio, KS_start=10, KS_end=100, KS_interval=10)
            max_eig_ratio_bool = \
                max_eig_ratio_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'max_eig_ratio\' must be a float percentage between 0 and 1.')

        if (not test_all) and max_eig_ratio_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'max_eig_ratio\' must be a float percentage between 0 and 1.')
        elif max_eig_ratio_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'max_eig_ratio\' must be a float percentage between 0 and 1.'


    def test_SSA_KS_start_value(self, test_all=False):

        KS_start_bool = True
        for KS_start in [10.0, -10, 350]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_ssa(time_series=np.arange(1000.0), L=200, est=10, plot=False, KS_test=False,
                             plot_KS_test=False, KS_scale_limit=1.0,
                             max_eig_ratio=0.5, KS_start=KS_start, KS_end=100, KS_interval=10)
            KS_start_bool = \
                KS_start_bool and (error_info.type is ValueError) and (error_info.value.args[0] ==
                            '\'KS_start\' must be a positive integer of appropriate magnitude: '
                            'KS_start <= len(time_series) / 3.')

        if (not test_all) and KS_start_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'KS_start\' must be a positive integer of appropriate magnitude: '
                  'KS_start <= len(time_series) / 3.')
        elif KS_start_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                ('\'KS_start\' must be a positive integer of appropriate magnitude: '
                 'KS_start <= len(time_series) / 3.')


    def test_SSA_KS_end_value(self, test_all=False):

        KS_end_bool = True
        for KS_end in [50.0, -10, 15]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_ssa(time_series=np.arange(1000.0), L=200, est=10, plot=False, KS_test=False,
                             plot_KS_test=False, KS_scale_limit=1.0,
                             max_eig_ratio=0.5, KS_start=20, KS_end=KS_end, KS_interval=10)
            KS_end_bool = \
                KS_end_bool and (error_info.type is ValueError) and (error_info.value.args[0] ==
                            '\'KS_end\' must be a positive integer of appropriate magnitude: '
                            'KS_end > KS_start and KS_end <= len(time_series) / 3.')

        if (not test_all) and KS_end_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'KS_end\' must be a positive integer of appropriate magnitude: '
                  'KS_end > KS_start and KS_end <= len(time_series) / 3.')
        elif KS_end_bool:
            return error_info.type is ValueError and (error_info.value.args[0] == \
                '\'KS_end\' must be a positive integer of appropriate magnitude: '
                'KS_end > KS_start and KS_end <= len(time_series) / 3.')


    def test_SSA_KS_interval_value(self, test_all=False):

        KS_interval_bool = True
        for KS_interval in [50.0, -10, 90]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_ssa(time_series=np.arange(1000.0), L=200, est=10, plot=False, KS_test=False,
                             plot_KS_test=False, KS_scale_limit=1.0,
                             max_eig_ratio=0.5, KS_start=20, KS_end=100, KS_interval=KS_interval)
            KS_interval_bool = \
                KS_interval_bool and (error_info.type is ValueError) and (error_info.value.args[0] ==
                            '\'KS_interval\' must be a positive integer of appropriate magnitude: '
                            'KS_interval < (KS_end - KS_start).')

        if (not test_all) and KS_interval_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'KS_interval\' must be a positive integer of appropriate magnitude: '
                  'KS_interval < (KS_end - KS_start).')
        elif KS_interval_bool:
            return error_info.type is ValueError and (error_info.value.args[0] ==
                                                      '\'KS_interval\' must be a positive integer of appropriate magnitude: '
                                                      'KS_interval < (KS_end - KS_start).')


    def test_SSD_gaussian_f_array(self, test_all=False):

        f = 5.0

        with pytest.raises(TypeError) as error_info:
            gaussian(f=f, A=1.0, mu=1.0, sigma=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'f\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'f\' must be of type np.ndarray.'


    def test_SSD_gaussian_f_only_floats(self, test_all=False):

        f = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            gaussian(f=f, A=1.0, mu=1.0, sigma=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'f\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'f\' must only contain floats.'


    def test_SSD_gaussian_A_value(self, test_all=False):

        A_bool = True
        for A in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                gaussian(f=np.arange(100.0), A=A, mu=1.0, sigma=1.0)
            A_bool = \
                A_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'A\' must be a non-negative float.')

        if (not test_all) and A_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'A\' must be a non-negative float.')
        elif A_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'A\' must be a non-negative float.'


    def test_SSD_gaussian_mu_value(self, test_all=False):

        mu_bool = True
        for mu in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                gaussian(f=np.arange(100.0), A=1.0, mu=mu, sigma=1.0)
            mu_bool = \
                mu_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu\' must be a positive float.')

        if (not test_all) and mu_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu\' must be a positive float.')
        elif mu_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu\' must be a positive float.'


    def test_SSD_gaussian_sigma_value(self, test_all=False):

        sigma_bool = True
        for sigma in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                gaussian(f=np.arange(100.0), A=1.0, mu=1.0, sigma=sigma)
            sigma_bool = \
                sigma_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'sigma\' must be a positive float.')

        if (not test_all) and sigma_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'sigma\' must be a positive float.')
        elif sigma_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'sigma\' must be a positive float.'


    def test_SSD_max_bool_time_series_array(self, test_all=False):

        time_series = 5.0

        with pytest.raises(TypeError) as error_info:
            max_bool(time_series=time_series)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'time_series\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'time_series\' must be of type np.ndarray.'


    def test_SSD_max_bool_time_series_only_floats(self, test_all=False):

        time_series = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            max_bool(time_series=time_series)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'time_series\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'time_series\' must only contain floats.'


    def test_SSD_spectral_obj_func_l1_theta_array(self, test_all=False):

        theta = 5.0

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l1(theta, f=np.asarray(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'theta\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'theta\' must be of type np.ndarray.'


    def test_SSD_spectral_obj_func_l1_theta_only_floats(self, test_all=False):

        theta = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l1(theta, f=np.asarray(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'theta\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'theta\' must only contain floats.'


    def test_SSD_spectral_obj_func_l1_theta_length(self, test_all=False):

        theta = np.arange(7.0)

        with pytest.raises(ValueError) as error_info:
            spectral_obj_func_l1(theta, f=np.asarray(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'theta\' must be of length 6.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'theta\' must be of length 6.'


    def test_SSD_spectral_obj_func_l1_f_array(self, test_all=False):

        f = 5.0

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l1(theta=np.arange(6.0), f=f, mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'f\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'f\' must be of type np.ndarray.'


    def test_SSD_spectral_obj_func_l1_f_only_floats(self, test_all=False):

        f = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l1(theta=np.arange(6.0), f=f, mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'f\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'f\' must only contain floats.'


    def test_SSD_spectral_obj_func_l1_mu_1_value(self, test_all=False):

        mu_1_bool = True
        for mu_1 in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                spectral_obj_func_l1(theta=np.arange(6.0), f=np.asarray(1000.0),
                                     mu_1=mu_1, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
            mu_1_bool = \
                mu_1_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu_1\' must be a positive float.')

        if (not test_all) and mu_1_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu_1\' must be a positive float.')
        elif mu_1_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu_1\' must be a positive float.'


    def test_SSD_spectral_obj_func_l1_mu_2_value(self, test_all=False):

        mu_2_bool = True
        for mu_2 in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                spectral_obj_func_l1(theta=np.arange(6.0), f=np.asarray(1000.0),
                                     mu_1=1.0, mu_2=mu_2, mu_3=1.0, spectrum=np.asarray(1000.0))
            mu_2_bool = \
                mu_2_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu_2\' must be a positive float.')

        if (not test_all) and mu_2_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu_2\' must be a positive float.')
        elif mu_2_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu_2\' must be a positive float.'


    def test_SSD_spectral_obj_func_l1_mu_3_value(self, test_all=False):

        mu_3_bool = True
        for mu_3 in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                spectral_obj_func_l1(theta=np.arange(6.0), f=np.asarray(1000.0),
                                     mu_1=1.0, mu_2=1.0, mu_3=mu_3, spectrum=np.asarray(1000.0))
            mu_3_bool = \
                mu_3_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu_3\' must be a positive float.')

        if (not test_all) and mu_3_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu_3\' must be a positive float.')
        elif mu_3_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu_3\' must be a positive float.'


    def test_SSD_spectral_obj_func_l1_spectrum_array(self, test_all=False):

        spectrum = 5.0

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l1(theta=np.arange(6.0), f=np.asarray(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                                 spectrum=spectrum)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must be of type np.ndarray.'


    def test_SSD_spectral_obj_func_l1_spectrum_only_floats(self, test_all=False):

        spectrum = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l1(theta=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                                 spectrum=spectrum)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must only contain floats.'


    def test_SSD_spectral_obj_func_l1_f_and_spectrum_lengths(self, test_all=False):

        spectrum = np.arange(70.0)

        with pytest.raises(ValueError) as error_info:
            spectral_obj_func_l1(theta=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                                 spectrum=spectrum)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'f\' and \'spectrum\' are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'f\' and \'spectrum\' are incompatible lengths.'


    def test_SSD_spectral_obj_func_l2_theta_array(self, test_all=False):

        theta = 5.0

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l2(theta, f=np.asarray(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'theta\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'theta\' must be of type np.ndarray.'


    def test_SSD_spectral_obj_func_l2_theta_only_floats(self, test_all=False):

        theta = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l2(theta, f=np.asarray(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'theta\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'theta\' must only contain floats.'


    def test_SSD_spectral_obj_func_l2_theta_length(self, test_all=False):

        theta = np.arange(7.0)

        with pytest.raises(ValueError) as error_info:
            spectral_obj_func_l2(theta, f=np.asarray(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'theta\' must be of length 6.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'theta\' must be of length 6.'


    def test_SSD_spectral_obj_func_l2_f_array(self, test_all=False):

        f = 5.0

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l2(theta=np.arange(6.0), f=f, mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'f\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'f\' must be of type np.ndarray.'


    def test_SSD_spectral_obj_func_l2_f_only_floats(self, test_all=False):

        f = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l2(theta=np.arange(6.0), f=f, mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'f\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'f\' must only contain floats.'


    def test_SSD_spectral_obj_func_l2_mu_1_value(self, test_all=False):

        mu_1_bool = True
        for mu_1 in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                spectral_obj_func_l2(theta=np.arange(6.0), f=np.asarray(1000.0),
                                     mu_1=mu_1, mu_2=1.0, mu_3=1.0, spectrum=np.asarray(1000.0))
            mu_1_bool = \
                mu_1_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu_1\' must be a positive float.')

        if (not test_all) and mu_1_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu_1\' must be a positive float.')
        elif mu_1_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu_1\' must be a positive float.'


    def test_SSD_spectral_obj_func_l2_mu_2_value(self, test_all=False):

        mu_2_bool = True
        for mu_2 in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                spectral_obj_func_l2(theta=np.arange(6.0), f=np.asarray(1000.0),
                                     mu_1=1.0, mu_2=mu_2, mu_3=1.0, spectrum=np.asarray(1000.0))
            mu_2_bool = \
                mu_2_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu_2\' must be a positive float.')

        if (not test_all) and mu_2_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu_2\' must be a positive float.')
        elif mu_2_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu_2\' must be a positive float.'


    def test_SSD_spectral_obj_func_l2_mu_3_value(self, test_all=False):

        mu_3_bool = True
        for mu_3 in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                spectral_obj_func_l2(theta=np.arange(6.0), f=np.asarray(1000.0),
                                     mu_1=1.0, mu_2=1.0, mu_3=mu_3, spectrum=np.asarray(1000.0))
            mu_3_bool = \
                mu_3_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu_3\' must be a positive float.')

        if (not test_all) and mu_3_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu_3\' must be a positive float.')
        elif mu_3_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu_3\' must be a positive float.'


    def test_SSD_spectral_obj_func_l2_spectrum_array(self, test_all=False):

        spectrum = 5.0

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l2(theta=np.arange(6.0), f=np.asarray(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                                 spectrum=spectrum)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must be of type np.ndarray.'


    def test_SSD_spectral_obj_func_l2_spectrum_only_floats(self, test_all=False):

        spectrum = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            spectral_obj_func_l2(theta=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                                 spectrum=spectrum)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must only contain floats.'


    def test_SSD_spectral_obj_func_l2_f_and_spectrum_lengths(self, test_all=False):

        spectrum = np.arange(70.0)

        with pytest.raises(ValueError) as error_info:
            spectral_obj_func_l2(theta=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                                 spectrum=spectrum)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'f\' and \'spectrum\' are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'f\' and \'spectrum\' are incompatible lengths.'


    def test_SSD_gaus_param_w0_array(self, test_all=False):

        w0 = 5.0

        with pytest.raises(TypeError) as error_info:
            gaus_param(w0=w0, f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.arange(1000.0),
                       method='l1')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'w0\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'w0\' must be of type np.ndarray.'


    def test_SSD_gaus_param_w0_only_floats(self, test_all=False):

        w0 = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            gaus_param(w0=w0, f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.arange(1000.0),
                       method='l1')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'w0\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'w0\' must only contain floats.'


    def test_SSD_gaus_param_w0_length(self, test_all=False):

        w0 = np.arange(7.0)

        with pytest.raises(ValueError) as error_info:
            gaus_param(w0=w0, f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.arange(1000.0),
                       method='l1')
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'w0\' must be of length 6.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'w0\' must be of length 6.'


    def test_SSD_gaus_param_f_array(self, test_all=False):

        f = 5.0

        with pytest.raises(TypeError) as error_info:
            gaus_param(w0=np.arange(6.0), f=f, mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.arange(1000.0),
                       method='l1')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'f\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'f\' must be of type np.ndarray.'


    def test_SSD_gaus_param_f_only_floats(self, test_all=False):

        f = np.arange(5)

        with pytest.raises(TypeError) as error_info:
            gaus_param(w0=np.arange(6.0), f=f, mu_1=1.0, mu_2=1.0, mu_3=1.0, spectrum=np.arange(1000.0),
                       method='l1')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'f\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'f\' must only contain floats.'


    def test_SSD_gaus_param_mu_1_value(self, test_all=False):

        mu_1_bool = True
        for mu_1 in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                gaus_param(w0=np.arange(6.0), f=np.arange(1000.0), mu_1=mu_1, mu_2=1.0, mu_3=1.0, spectrum=np.arange(1000.0),
                           method='l1')
            mu_1_bool = \
                mu_1_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu_1\' must be a positive float.')

        if (not test_all) and mu_1_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu_1\' must be a positive float.')
        elif mu_1_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu_1\' must be a positive float.'


    def test_SSD_gaus_param_mu_2_value(self, test_all=False):

        mu_2_bool = True
        for mu_2 in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                gaus_param(w0=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=mu_2, mu_3=1.0,
                           spectrum=np.arange(1000.0),
                           method='l1')
            mu_2_bool = \
                mu_2_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu_2\' must be a positive float.')

        if (not test_all) and mu_2_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu_2\' must be a positive float.')
        elif mu_2_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu_2\' must be a positive float.'


    def test_SSD_gaus_param_mu_3_value(self, test_all=False):

        mu_3_bool = True
        for mu_3 in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                gaus_param(w0=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=mu_3,
                           spectrum=np.arange(1000.0),
                           method='l1')
            mu_3_bool = \
                mu_3_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'mu_3\' must be a positive float.')

        if (not test_all) and mu_3_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mu_3\' must be a positive float.')
        elif mu_3_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'mu_3\' must be a positive float.'


    def test_SSD_gaus_param_spectrum_array(self, test_all=False):

        spectrum = 5.0

        with pytest.raises(TypeError) as error_info:
            gaus_param(w0=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                       spectrum=spectrum,
                       method='l1')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must be of type np.ndarray.'


    def test_SSD_gaus_param_spectrum_only_floats(self, test_all=False):

        spectrum = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            gaus_param(w0=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                       spectrum=spectrum,
                       method='l1')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'spectrum\' must only contain floats.'


    def test_SSD_gaus_param_f_and_spectrum_lengths(self, test_all=False):

        spectrum = np.arange(70.0)

        with pytest.raises(ValueError) as error_info:
            gaus_param(w0=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                       spectrum=spectrum,
                       method='l1')
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'f\' and \'spectrum\' are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'f\' and \'spectrum\' are incompatible lengths.'


    def test_SSD_gaus_param_method_value(self, test_all=False):

        method = 'l3'

        with pytest.raises(ValueError) as error_info:
            gaus_param(w0=np.arange(6.0), f=np.arange(1000.0), mu_1=1.0, mu_2=1.0, mu_3=1.0,
                       spectrum=np.arange(1000.0),
                       method=method)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'method\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'method\' not an acceptable value.'


    def test_SSD_scaling_factor_obj_func_a_value(self, test_all=False):

        a_bool = True
        for a in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                scaling_factor_obj_func(a=a, residual_time_series=np.arange(1000.0), trend_estimate=np.arange(1000.0))
            a_bool = \
                a_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'a\' must be a positive float.')

        if (not test_all) and a_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'a\' must be a positive float.')
        elif a_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'a\' must be a positive float.'


    def test_SSD_scaling_factor_obj_func_residual_time_series_array(self, test_all=False):

        residual_time_series = 5.0

        with pytest.raises(TypeError) as error_info:
            scaling_factor_obj_func(a=1.0, residual_time_series=residual_time_series,
                                    trend_estimate=np.arange(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'residual_time_series\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'residual_time_series\' must be of type np.ndarray.'


    def test_SSD_scaling_factor_obj_func_residual_time_series_only_floats(self, test_all=False):

        residual_time_series = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            scaling_factor_obj_func(a=1.0, residual_time_series=residual_time_series,
                                    trend_estimate=np.arange(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'residual_time_series\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'residual_time_series\' must only contain floats.'


    def test_SSD_scaling_factor_obj_func_trend_estimate_array(self, test_all=False):

        trend_estimate = 5.0

        with pytest.raises(TypeError) as error_info:
            scaling_factor_obj_func(a=1.0, residual_time_series=np.arange(1000.0),
                                    trend_estimate=trend_estimate)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'trend_estimate\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'trend_estimate\' must be of type np.ndarray.'


    def test_SSD_scaling_factor_obj_func_trend_estimate_only_floats(self, test_all=False):

        trend_estimate = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            scaling_factor_obj_func(a=1.0, residual_time_series=np.arange(1000.0),
                                    trend_estimate=trend_estimate)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'trend_estimate\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'trend_estimate\' must only contain floats.'


    def test_SSD_scaling_factor_obj_func_residual_time_series_and_trend_estimate_lengths(self, test_all=False):

        trend_estimate = np.arange(6.0)

        with pytest.raises(ValueError) as error_info:
            scaling_factor_obj_func(a=1.0, residual_time_series=np.arange(1000.0),
                                    trend_estimate=trend_estimate)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'residual_time_series\' and \'trend_estimate\' are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'residual_time_series\' and \'trend_estimate\' are incompatible lengths.'


    def test_SSD_scaling_factor_residual_time_series_array(self, test_all=False):

        residual_time_series = 5.0

        with pytest.raises(TypeError) as error_info:
            scaling_factor(residual_time_series=residual_time_series,
                           trend_estimate=np.arange(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'residual_time_series\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'residual_time_series\' must be of type np.ndarray.'


    def test_SSD_scaling_factor_residual_time_series_only_floats(self, test_all=False):

        residual_time_series = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            scaling_factor(residual_time_series=residual_time_series,
                           trend_estimate=np.arange(1000.0))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'residual_time_series\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'residual_time_series\' must only contain floats.'


    def test_SSD_scaling_factor_trend_estimate_array(self, test_all=False):

        trend_estimate = 5.0

        with pytest.raises(TypeError) as error_info:
            scaling_factor(residual_time_series=np.arange(1000.0),
                           trend_estimate=trend_estimate)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'trend_estimate\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'trend_estimate\' must be of type np.ndarray.'


    def test_SSD_scaling_factor_trend_estimate_only_floats(self, test_all=False):

        trend_estimate = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            scaling_factor(residual_time_series=np.arange(1000.0),
                           trend_estimate=trend_estimate)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'trend_estimate\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'trend_estimate\' must only contain floats.'


    def test_SSD_scaling_factor_residual_time_series_and_trend_estimate_lengths(self, test_all=False):

        trend_estimate = np.arange(6.0)

        with pytest.raises(ValueError) as error_info:
            scaling_factor(residual_time_series=np.arange(1000.0),
                           trend_estimate=trend_estimate)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'residual_time_series\' and \'trend_estimate\' are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'residual_time_series\' and \'trend_estimate\' are incompatible lengths.'


    def test_SSD_CovRegpy_ssd_time_series_array(self, test_all=False):

        time_series = 5.0

        with pytest.raises(TypeError) as error_info:
            CovRegpy_ssd(time_series, initial_trend_ratio=3.0, nmse_threshold=0.01, plot=False,
                         debug=False, method='l2')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'time_series\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'time_series\' must be of type np.ndarray.'


    def test_SSD_CovRegpy_ssd_time_series_only_floats(self, test_all=False):

        time_series = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            CovRegpy_ssd(time_series, initial_trend_ratio=3.0, nmse_threshold=0.01, plot=False,
                         debug=False, method='l2')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'time_series\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'time_series\' must only contain floats.'


    def test_SSD_CovRegpy_ssd_initial_trend_ratio_value(self, test_all=False):

        initial_trend_ratio_bool = True
        for initial_trend_ratio in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_ssd(time_series=np.arange(100.0), initial_trend_ratio=initial_trend_ratio, nmse_threshold=0.01,
                             plot=False, debug=False, method='l2')
            initial_trend_ratio_bool = \
                initial_trend_ratio_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'initial_trend_ratio\' must be a positive float.')

        if (not test_all) and initial_trend_ratio_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'initial_trend_ratio\' must be a positive float.')
        elif initial_trend_ratio_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'initial_trend_ratio\' must be a positive float.'


    def test_SSD_CovRegpy_ssd_nmse_threshold_value(self, test_all=False):

        nmse_threshold_bool = True
        for nmse_threshold in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                CovRegpy_ssd(time_series=np.arange(100.0), initial_trend_ratio=3.0, nmse_threshold=nmse_threshold,
                             plot=False, debug=False, method='l2')
            nmse_threshold_bool = \
                nmse_threshold_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'nmse_threshold\' must be a positive float.')

        if (not test_all) and nmse_threshold_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'nmse_threshold\' must be a positive float.')
        elif nmse_threshold_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'nmse_threshold\' must be a positive float.'


    def test_SSD_CovRegpy_ssd_plot_bool(self, test_all=False):

        plot = 'not_bool'

        with pytest.raises(TypeError) as error_info:
            CovRegpy_ssd(time_series=np.arange(100.0), initial_trend_ratio=3.0, nmse_threshold=0.01,
                         plot=plot, debug=False, method='l2')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'plot\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'plot\' must be boolean.'


    def test_SSD_CovRegpy_ssd_debug_bool(self, test_all=False):

        debug = 'not_bool'

        with pytest.raises(TypeError) as error_info:
            CovRegpy_ssd(time_series=np.arange(100.0), initial_trend_ratio=3.0, nmse_threshold=0.01,
                         plot=False, debug=debug, method='l2')
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'debug\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'debug\' must be boolean.'


    def test_SSD_CovRegpy_ssd_method_type(self, test_all=False):

        method = 'l3'

        with pytest.raises(ValueError) as error_info:
            CovRegpy_ssd(time_series=np.arange(100.0), initial_trend_ratio=3.0, nmse_threshold=0.01,
                         plot=False, debug=False, method=method)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'method\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'method\' not an acceptable value.'


    def test_RPP_risk_parity_obj_fun_x_array(self, test_all=False):

        x = 5.0

        with pytest.raises(TypeError) as error_info:
            risk_parity_obj_fun(x=x, p_cov=np.asarray([[1, 0.5], [0.5, 1]]), rb=np.asarray([1/2, 1/2]))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'x\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'x\' must be of type np.ndarray.'


    def test_RPP_risk_parity_obj_fun_x_only_floats(self, test_all=False):

        x = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            risk_parity_obj_fun(x=x, p_cov=np.asarray([[1, 0.5], [0.5, 1]]), rb=np.asarray([1/2, 1/2]))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'x\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'x\' must only contain floats.'


    def test_RPP_risk_parity_obj_fun_p_cov_array(self, test_all=False):

        p_cov = 5.0

        with pytest.raises(TypeError) as error_info:
            risk_parity_obj_fun(x=np.asarray([1/2, 1/2]), p_cov=p_cov, rb=np.asarray([1/2, 1/2]))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'p_cov\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'p_cov\' must be of type np.ndarray.'


    def test_RPP_risk_parity_obj_fun_p_cov_only_floats(self, test_all=False):

        p_cov = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            risk_parity_obj_fun(x=np.asarray([1/2, 1/2]), p_cov=p_cov, rb=np.asarray([1/2, 1/2]))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'p_cov\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'p_cov\' must only contain floats.'


    def test_RPP_risk_parity_obj_fun_p_cov_psd(self, test_all=False):

        p_cov = np.asarray([[1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])

        with pytest.raises(ValueError) as error_info:
            risk_parity_obj_fun(x=np.asarray([1/2, 1/2]), p_cov=p_cov, rb=np.asarray([1/2, 1/2]))
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'p_cov\' must be PSD.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'p_cov\' must be PSD.'


    def test_RPP_risk_parity_obj_fun_x_p_cov_shape(self, test_all=False):

        x = np.ones(3) / 3
        p_cov = np.eye(5)

        with pytest.raises(ValueError) as error_info:
            risk_parity_obj_fun(x=x, p_cov=p_cov, rb=np.ones(3) / 3)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'x\' and \'p_cov\' are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'x\' and \'p_cov\' are incompatible lengths.'


    def test_RPP_risk_parity_obj_fun_rb_array(self, test_all=False):

        rb = 5.0

        with pytest.raises(TypeError) as error_info:
            risk_parity_obj_fun(x=np.asarray([1/2, 1/2]), p_cov=np.eye(2), rb=rb)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'rb\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'rb\' must be of type np.ndarray.'


    def test_RPP_risk_parity_obj_fun_rb_only_floats(self, test_all=False):

        rb = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            risk_parity_obj_fun(x=np.asarray([1/2, 1/2]), p_cov=np.eye(2), rb=rb)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'rb\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'rb\' must only contain floats.'


    def test_RPP_risk_parity_obj_fun_x_rb_shape(self, test_all=False):

        x = np.ones(3) / 3
        rb = np.ones(4) / 4

        with pytest.raises(ValueError) as error_info:
            risk_parity_obj_fun(x=x, p_cov=np.eye(3), rb=rb)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'x\' and \'rb\' are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'x\' and \'rb\' are incompatible lengths.'


    def test_RPP_equal_risk_parity_weights_long_restriction_cov_array(self, test_all=False):

        cov = 5.0

        with pytest.raises(TypeError) as error_info:
            equal_risk_parity_weights_long_restriction(cov)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.'


    def test_RPP_equal_risk_parity_weights_long_restriction_cov_only_floats(self, test_all=False):

        cov = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            equal_risk_parity_weights_long_restriction(cov)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.'


    def test_RPP_equal_risk_parity_weights_long_restriction_cov_psd(self, test_all=False):

        cov = np.asarray([[1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])

        with pytest.raises(ValueError) as error_info:
            equal_risk_parity_weights_long_restriction(cov)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.'


    def test_RPP_equal_risk_parity_weights_short_restriction_cov_array(self, test_all=False):

        cov = 5.0

        with pytest.raises(TypeError) as error_info:
            equal_risk_parity_weights_short_restriction(cov, short_limit=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.'


    def test_RPP_equal_risk_parity_weights_short_restriction_cov_only_floats(self, test_all=False):

        cov = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            equal_risk_parity_weights_short_restriction(cov, short_limit=1.0)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.'


    def test_RPP_equal_risk_parity_weights_short_restriction_cov_psd(self, test_all=False):

        cov = np.asarray([[1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])

        with pytest.raises(ValueError) as error_info:
            equal_risk_parity_weights_short_restriction(cov, short_limit=1.0)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.'


    def test_RPP_equal_risk_parity_weights_short_restriction_short_limit_value(self, test_all=False):

        short_limit_bool = True
        for short_limit in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                equal_risk_parity_weights_short_restriction(cov=np.asarray([[1., -1.], [1., -1.]]),
                                                            short_limit=short_limit)
            short_limit_bool = \
                short_limit_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'short_limit\' must be a positive float.')

        if (not test_all) and short_limit_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'short_limit\' must be a positive float.')
        elif short_limit_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'short_limit\' must be a positive float.'


    def test_RPP_equal_risk_parity_weights_summation_restriction_cov_array(self, test_all=False):

        cov = 5.0

        with pytest.raises(TypeError) as error_info:
            equal_risk_parity_weights_summation_restriction(cov, short_limit=0.3, long_limit=1.3)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.'


    def test_RPP_equal_risk_parity_weights_summation_restriction_cov_only_floats(self, test_all=False):

        cov = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            equal_risk_parity_weights_summation_restriction(cov, short_limit=0.3, long_limit=1.3)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.'


    def test_RPP_equal_risk_parity_weights_summation_restriction_cov_psd(self, test_all=False):

        cov = np.asarray([[1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])

        with pytest.raises(ValueError) as error_info:
            equal_risk_parity_weights_summation_restriction(cov, short_limit=0.3, long_limit=1.3)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.'


    def test_RPP_equal_risk_parity_weights_summation_restriction_short_limit_value(self, test_all=False):

        short_limit_bool = True
        for short_limit in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                equal_risk_parity_weights_summation_restriction(cov=np.asarray([[1., -1.], [1., -1.]]),
                                                                short_limit=short_limit, long_limit=1.3)
            short_limit_bool = \
                short_limit_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'short_limit\' must be a positive float.')

        if (not test_all) and short_limit_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'short_limit\' must be a positive float.')
        elif short_limit_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'short_limit\' must be a positive float.'


    def test_RPP_equal_risk_parity_weights_summation_restriction_long_limit_value(self, test_all=False):

        long_limit_bool = True
        for long_limit in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                equal_risk_parity_weights_summation_restriction(cov=np.asarray([[1., -1.], [1., -1.]]),
                                                                short_limit=0.3, long_limit=long_limit)
            long_limit_bool = \
                long_limit_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'long_limit\' must be a positive float.')

        if (not test_all) and long_limit_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'long_limit\' must be a positive float.')
        elif long_limit_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'long_limit\' must be a positive float.'


    def test_RPP_global_obj_fun_x_array(self, test_all=False):

        x = 5.0

        with pytest.raises(TypeError) as error_info:
            global_obj_fun(x=x, p_cov=np.asarray([[1., -1.], [1., -1.]]))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'x\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'x\' must be of type np.ndarray.'


    def test_RPP_global_obj_fun_x_only_floats(self, test_all=False):

        x = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            global_obj_fun(x=x, p_cov=np.asarray([[1., -1.], [1., -1.]]))
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'x\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'x\' must only contain floats.'


    def test_RPP_global_obj_fun_p_cov_array(self, test_all=False):

        p_cov = 5.0

        with pytest.raises(TypeError) as error_info:
            global_obj_fun(x=np.asarray([1/2, 1/2]), p_cov=p_cov)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'p_cov\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'p_cov\' must be of type np.ndarray.'


    def test_RPP_global_obj_fun_p_cov_only_floats(self, test_all=False):

        p_cov = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            global_obj_fun(x=np.asarray([1/2, 1/2]), p_cov=p_cov)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'p_cov\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'p_cov\' must only contain floats.'


    def test_RPP_global_obj_fun_p_cov_psd(self, test_all=False):

        p_cov = np.asarray([[1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])

        with pytest.raises(ValueError) as error_info:
            global_obj_fun(x=np.asarray([1/3, 1/3, 1/3]), p_cov=p_cov)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'p_cov\' must be PSD.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'p_cov\' must be PSD.'


    def test_RPP_global_obj_fun_x_p_cov_shape(self, test_all=False):

        x = np.ones(3) / 3
        p_cov = np.eye(5)

        with pytest.raises(ValueError) as error_info:
            global_obj_fun(x=x, p_cov=p_cov)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'x\' and \'p_cov\' are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'x\' and \'p_cov\' are incompatible lengths.'


    def test_RPP_global_weights_cov_array(self, test_all=False):

        cov = 5.0

        with pytest.raises(TypeError) as error_info:
            global_weights(cov)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.'


    def test_RPP_global_weights_cov_only_floats(self, test_all=False):

        cov = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            global_weights(cov)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.'


    def test_RPP_global_weights_cov_psd(self, test_all=False):

        cov = np.asarray([[1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])

        with pytest.raises(ValueError) as error_info:
            global_weights(cov)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.'


    def test_RPP_global_weights_long_cov_array(self, test_all=False):

        cov = 5.0

        with pytest.raises(TypeError) as error_info:
            global_weights_long(cov)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.'


    def test_RPP_global_weights_long_cov_only_floats(self, test_all=False):

        cov = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            global_weights_long(cov)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.'


    def test_RPP_global_weights_long_cov_psd(self, test_all=False):

        cov = np.asarray([[1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])

        with pytest.raises(ValueError) as error_info:
            global_weights_long(cov)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.'


    def test_RPP_global_weights_short_and_long_restrict_cov_array(self, test_all=False):

        cov = 5.0

        with pytest.raises(TypeError) as error_info:
            global_weights_short_and_long_restrict(cov, b=0.3, a=1.3)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.'


    def test_RPP_global_weights_short_and_long_restrict_cov_only_floats(self, test_all=False):

        cov = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            global_weights_short_and_long_restrict(cov, b=0.3, a=1.3)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.'


    def test_RPP_global_weights_short_and_long_restrict_cov_psd(self, test_all=False):

        cov = np.asarray([[1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])

        with pytest.raises(ValueError) as error_info:
            global_weights_short_and_long_restrict(cov, b=0.3, a=1.3)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.'


    def test_RPP_global_weights_short_and_long_restrict_b_value(self, test_all=False):

        b_bool = True
        for b in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                global_weights_short_and_long_restrict(cov=np.asarray([[1., -1.], [1., -1.]]), b=b, a=1.3)
            b_bool = \
                b_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'b\' must be a positive float.')

        if (not test_all) and b_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'b\' must be a positive float.')
        elif b_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'b\' must be a positive float.'


    def test_RPP_global_weights_short_and_long_restrict_a_value(self, test_all=False):

        a_bool = True
        for a in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                global_weights_short_and_long_restrict(cov=np.asarray([[1., -1.], [1., -1.]]), b=0.3, a=a)
            a_bool = \
                a_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'a\' must be a positive float.')

        if (not test_all) and a_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'a\' must be a positive float.')
        elif a_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'a\' must be a positive float.'


    def test_RPP_equal_risk_parity_weights_individual_restriction_cov_array(self, test_all=False):

        cov = 5.0

        with pytest.raises(TypeError) as error_info:
            equal_risk_parity_weights_individual_restriction(cov, short_limit=0.3, long_limit=1.3)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must be of type np.ndarray.'


    def test_RPP_equal_risk_parity_weights_individual_restriction_cov_only_floats(self, test_all=False):

        cov = np.arange(6)

        with pytest.raises(TypeError) as error_info:
            equal_risk_parity_weights_individual_restriction(cov, short_limit=0.3, long_limit=1.3)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'cov\' must only contain floats.'


    def test_RPP_equal_risk_parity_weights_individual_restriction_cov_psd(self, test_all=False):

        cov = np.asarray([[1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])

        with pytest.raises(ValueError) as error_info:
            equal_risk_parity_weights_individual_restriction(cov, short_limit=0.3, long_limit=1.3)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == '\'cov\' must be PSD.'


    def test_RPP_equal_risk_parity_weights_individual_restriction_short_limit_value(self, test_all=False):

        short_limit_bool = True
        for short_limit in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                equal_risk_parity_weights_individual_restriction(cov=np.asarray([[1., -1.], [1., -1.]]),
                                                                short_limit=short_limit, long_limit=1.3)
            short_limit_bool = \
                short_limit_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'short_limit\' must be a positive float.')

        if (not test_all) and short_limit_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'short_limit\' must be a positive float.')
        elif short_limit_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'short_limit\' must be a positive float.'


    def test_RPP_equal_risk_parity_weights_individual_restriction_long_limit_value(self, test_all=False):

        long_limit_bool = True
        for long_limit in [6, -1.0]:
            with pytest.raises(ValueError) as error_info:
                equal_risk_parity_weights_summation_restriction(cov=np.asarray([[1., -1.], [1., -1.]]),
                                                                short_limit=0.3, long_limit=long_limit)
            long_limit_bool = \
                long_limit_bool and (error_info.type is ValueError and error_info.value.args[0] ==
                            '\'long_limit\' must be a positive float.')

        if (not test_all) and long_limit_bool:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'long_limit\' must be a positive float.')
        elif long_limit_bool:
            return error_info.type is ValueError and error_info.value.args[0] == \
                '\'long_limit\' must be a positive float.'


if __name__ == "__main__":

    test_rcr = RCRUnitTests(calc_B_Psi, gamma_v_m_error, cov_reg_given_mean, subgrad_opt, covregpy_dcc, dcc_loglike,
                 risk_parity_obj_fun, equal_risk_parity_weights_long_restriction,
                 equal_risk_parity_weights_short_restriction, equal_risk_parity_weights_summation_restriction,
                 global_obj_fun, global_weights, global_weights_long, global_weights_short_and_long_restrict,
                 equal_risk_parity_weights_individual_restriction, CovRegpy_ssa, gaussian, max_bool,
                 spectral_obj_func_l1, spectral_obj_func_l2, gaus_param, scaling_factor_obj_func, scaling_factor,
                 CovRegpy_ssd, henderson_kernel, henderson_weights, henderson_ma, seasonal_ma, CovRegpy_X11)

    test_rcr.test_all(print_all=True)

