import numpy as np
import pandas as pd
from datetime import date
from datetime import datetime
import feature_functions_ar as feature_functions
import data_pipeline
import matplotlib.pyplot as plt
import os
import pickle as pkl
import arviz as az

# today = 'test'
# include_test_2 = True
# today_2 = ''
# if include_test_2:
#     today_2 = "_test_2"
#
# long_string = "trace/trace_ar_{}_generation{}".format(str(today), today_2)
# print(long_string)

# data = [165, 162, 120, 128, 50, 104, 218, 388, 249, 169, 119, 308, 217, 357]
# last_week = data[-7:]
# day_last_week = data[-7]
# previous_week = data[-14:-7]
# phi_0 = -0.7
# phi_1 = 1.12
# phi_2 = 0.0011
# phi_0_n = -0.335
# phi_1_n = 0.156
# phi_2_n = -0.035
# phi_self = 0.995
# data_n = [307, 274, 202, 220, 92, 174, 321, 693, 662, 358, 394, 581, 471, 796]
# expected_mu = 467
# expected_y = 208
# expected_y_tred = 163
# calculated_mean_y = (data[-7] - np.mean(data[-14:-7])) * phi_0 + np.mean(data[-7:]) * phi_1 + (np.mean(data[-7:]) ** 2) * phi_2
# calculated_mean_y_n = (data_n[-7] - np.mean(data_n[-14:-7])) * phi_0_n + np.mean(data_n[-7:]) * phi_1_n + (np.mean(data_n[-7:]) ** 2) * phi_2_n
# calcualted_y = phi_self * calculated_mean_y + (1-phi_self) * calculated_mean_y_n
# print(calcualted_y)

# --------------------------------------------------------------------------------------------
# Start plotting parameters
# --------------------------------------------------------------------------------------------
def plot_parameters():
    dates = [date(2020,9,10), date(2020,11,1), date(2020,12,15), date(2021,5,15), date(2021,9,25), date(2022,2,1)]
    for today in dates:
        with open("model/model_ar_{}".format(str(today)), "rb") as f:
            model = pkl.load(f)
        with open("trace/trace_ar_{}".format(str(today)), "rb") as f:
            trace = pkl.load(f)
        with model:
            az.plot_trace(trace)
            # print(az.summary(trace, round_to=2))
        cd = os.getcwd()
        plots = cd + "\\Plots\\"
        plots = plots + 'posterior_plots'
        try:
            os.mkdir(plots)
        except OSError:
            print("Creation of the directory %s failed" % plots)
        else:
            print("Successfully created the directory %s " % plots)
        plt.savefig(plots + "/posteriors_{}".format(today))
# --------------------------------------------------------------------------------------------
# End plotting parameters
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Start plotting interactions
# --------------------------------------------------------------------------------------------

def plot_interaction_proportions():
    dates = [date(2020,9,10), date(2020,11,1), date(2020,12,15), date(2021,5,15), date(2021,9,25), date(2022,2,1)]
    fig, ax = plt.subplots(figsize=(10,8))
    # ax.set_title('Comparison of Interaction weights')
    ax.xaxis.set_tick_params(rotation=30, labelsize=10)
    add_info_pd = pd.read_csv("ID_to_name_demographic.csv")
    additional_info = add_info_pd.to_dict("records")
    demographic = add_info_pd["demographic"].to_numpy()
    for fixed_params in [False]:
        phi_neighbour = []
        phi_selves = []
        file_change = ''
        if fixed_params:
            file_change ='_fixed'

        for today in dates:
            start_day = datetime.combine(today - pd.Timedelta(days=42), datetime.strptime("0000", "%H%M").time())
            data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv", start_day, pad=5)
            data[data < 0] = 0  # to get rid of the negative values

            data_train, target_train, data_test, target_test = data_pipeline.split_data(
                data,
                train_start=start_day,
                test_start=start_day + pd.Timedelta(days=6 * 7),
                post_test=start_day + pd.Timedelta(days=6 * 7 + 5),  # *7 + 5
            )
            features_for_model = feature_functions.get_features(target_train, demographic)
            feature_a_mean = features_for_model['34_weekly_mean']
            feature_a_mean_squared = features_for_model['34_weekly_mean_squared']
            feature_b_mean = features_for_model['35_weekly_mean']
            feature_b_mean_squared = features_for_model['35_weekly_mean_squared']
            feature_a = features_for_model['34'][-1] - feature_a_mean[-7]
            feature_b = features_for_model['35'][-1] - feature_b_mean[-7]

            with open("predictions/predictions_trend_ar_{}{}".format(str(today), file_change), "rb") as f:
                pred = pkl.load(f)
            phi_self = pred["phi_self"]
            mean_a = feature_a * pred['phi'][0] + feature_a_mean[-1] * pred['phi'][1] + feature_a_mean_squared[-1] * pred['phi'][2]
            mean_b = feature_b * pred['phi_neighbour'][0] + feature_b_mean[-1] * pred['phi_neighbour'][1] + feature_b_mean_squared[-1] * pred['phi_neighbour'][2]
            mean = phi_self * mean_a + (1 - phi_self) * mean_b
            phi_neighbour.append(((1 - phi_self) * mean_b) / mean)
            phi_selves.append((phi_self * mean_a) / mean)
            print(f'proportion self: {(phi_self * mean_a) / mean}, proportion interaction: {((1 - phi_self) * mean_b) / mean}')

        x_addition = 0
        color1= 'limegreen'
        color2 = 'tab:blue'
        if fixed_params:
            x_addition = 0.25
            color1 = 'tab:blue'
            color2 = 'firebrick'

        ax.bar(np.arange(6) + x_addition, phi_neighbour, label=f'Proportion_b{file_change}', width=0.25, color=color2)
        ax.bar(np.arange(6) + x_addition, phi_selves, bottom=phi_neighbour, label=f'Proportion_a{file_change}', width=0.25, color=color1)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(dates)
    ax.legend(loc="lower right")
    cd = os.getcwd()
    plots = cd + "\\Plots\\"
    plots = plots + 'correlation_analysis'
    try:
        os.mkdir(plots)
    except OSError:
        print("Creation of the directory %s failed" % plots)
    else:
        print("Successfully created the directory %s " % plots)
    plt.savefig(plots + "/interaction_plot")
# --------------------------------------------------------------------------------------------
# End plotting interactions
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Start Correlation
# --------------------------------------------------------------------------------------------
def plot_correlation_interaction_comparison():
    dates = [date(2020,9,10), date(2020,11,1), date(2020,12,15), date(2021,5,15), date(2021,9,25), date(2022,2,1)]
    correlations = []
    weighted_correlations = []
    phi_neighbour = []
    fixed_params = True

    file_change = ''
    if fixed_params:
        file_change ='_fixed'

    for today in dates:
        with open("predictions/predictions_trend_ar_{}{}".format(str(today), file_change), "rb") as f:
            pred = pkl.load(f)
        phi_self = pred["phi_self"]
        start_day = datetime.combine(today - pd.Timedelta(days=35), datetime.strptime("0000", "%H%M").time())
        data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv", start_day, pad=0)
        data[data < 0] = 0  # to get rid of the negative values
        today_long = datetime.combine(today, datetime.strptime("0000", "%H%M").time())
        data_self = data['34']
        data_interaction = data['35']
        data_interval = data_self.loc[(start_day <= data.index)
                                & (data.index < today_long)].to_numpy()
        data_interval_interaction = data_interaction.loc[(start_day <= data.index)
                                      & (data.index < today_long)].to_numpy()
        correlation = np.corrcoef(data_interval, data_interval_interaction)
        print(f'Correlation: {correlation[0, 1]}, weighted correlation: {correlation[0, 1] / (1 + correlation[0, 1])}, phi_b: {1 - phi_self}')
        correlations.append(correlation[0, 1])
        weighted_correlations.append(correlation[0, 1] / (1 + correlation[0, 1]))
        phi_neighbour.append(1 - phi_self)

    fig, ax = plt.subplots()
    ax.set_title('Comparison of correlation and model weight')
    ax.plot(dates, correlations, label='correlation')
    ax.plot(dates, weighted_correlations, label='weighted correlation')
    ax.plot(dates, phi_neighbour, label='phi_b')
    ax.xaxis.set_tick_params(rotation=30, labelsize=10)
    ax.legend()

    cd = os.getcwd()
    plots = cd + "\\Plots\\"
    plots = plots + 'correlation_analysis'
    try:
        os.mkdir(plots)
    except OSError:
        print("Creation of the directory %s failed" % plots)
    else:
        print("Successfully created the directory %s " % plots)
    plt.savefig(plots + "/correlation_analysis_fixed_params")
    print(np.corrcoef([correlations, weighted_correlations, phi_neighbour]))
# --------------------------------------------------------------------------------------------
# End correlation
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Begin Plotting selection dates
# --------------------------------------------------------------------------------------------
def plot_date_selection():
    cd = os.getcwd()
    plots = cd + "\\Plots\\"

    fig, ax = plt.subplots()
    data = data_pipeline.preprocess_LKOS_data(filename="Uni_OS_Fallzahlen 21.02.22.xlsx")
    data = data.loc[34]
    ax.scatter(data.index, data, marker='.', alpha=0.6, label='Counts of infections')
    ax.axvline(date(2020, 9, 10), c='orange', label='Days for analysis')
    ax.axvline(date(2020, 11, 1), c='orange')
    ax.axvline(date(2020, 12, 15), c='orange')
    ax.axvline(date(2021, 5, 15), c='orange')
    ax.axvline(date(2021, 9, 25), c='orange')
    ax.axvline(date(2022, 2, 1), c='orange')
    ax.set_title('COVID-19 infections in Osnabrück')
    ax.set_ylabel('Infections per day')
    ax.xaxis.set_tick_params(rotation=30, labelsize=10)
    ax.legend()

    plots = plots + 'overall_distribution'
    try:
        os.mkdir(plots)
    except OSError:
        print("Creation of the directory %s failed" % plots)
    else:
        print("Successfully created the directory %s " % plots)
    plt.savefig(plots + "/overall_distribution")
# -------------------------------------------------------------------------
# End plotting selection dates
# -------------------------------------------------------------------------




# print(datetime.date(2021,12,15), datetime.date(2021,12,15)+pd.Timedelta(days=30))

# data = data_pipeline.preprocess_LKOS_data(filename="Fallzahlen 28.04.21.xlsx")
# days_into_future = 5
# number_of_weeks = 6
#
# add_info_pd = pd.read_csv("ID_to_name_demographic.csv")
# additional_info = add_info_pd.to_dict("records")
# demographic = add_info_pd["demographic"].to_numpy()
# # nl_names = add_info_pd["NL Name"].to_numpy()
#
# data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv", pad=days_into_future)
# start_day = data.index[-1] - pd.Timedelta(days=46)
# data[data < 0] = 0  # to get rid of the negative values
#
# today = datetime.date.today()  # - pd.Timedelta(days=2) #- pd.Timedelta(days=1) # change after 15.04
#
# data_train, target_train, data_test, target_test = data_pipeline.split_data(
#     data,
#     train_start=start_day,
#     test_start=start_day + pd.Timedelta(days=number_of_weeks * 7),
#     post_test=start_day + pd.Timedelta(days=number_of_weeks * 7 + days_into_future),  # *7 + 5
# )
#
# features_for_model = feature_functions.get_features(target_train, demographic)
#
# data_os = features_for_model['34']
# data_lk = features_for_model['35']
#
# data_os_mean = np.array(features_for_model['34_weekly_mean'])
# data_lk_mean = np.array(features_for_model['35_weekly_mean'])
#
# print('Covariance and correlation matrix between os and surrounding area:')
# print(np.cov(np.array([data_os, data_lk])))
# print(np.corrcoef(np.array([data_os, data_lk])))
#
# print('Covariance and correlation matrix between os and surrounding area looking at weekly mean:')
# print(np.cov(np.array([data_os_mean, data_lk_mean])))
# print(np.corrcoef(np.array([data_os_mean, data_lk_mean])))
#
# print('Mean os: ', np.mean(data_os), '; std os: ', np.std(data_os))
# print('Mean os last 10 days: ', np.mean(data_os[-10:]), '; std os last 10 days: ', np.std(data_os[-10:]))
# print('Mean os first 10 days: ', np.mean(data_os[:10]), '; std os first 10 days: ', np.std(data_os[:10]))
