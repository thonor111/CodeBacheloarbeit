import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.stats import chisquare, normaltest
import arviz as az
import warnings
from scipy import special
import pymc3 as pm
from datetime import date


import data_pipeline

# Not used, some ideas for a Q-Q-plot
def q_q_plot(today):
    data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv")

    with open("predictions/predictions_ar_{}".format(str(today)), "rb") as f:
        pred = pkl.load(f)
    with open("predictions/predictions_trend_ar_{}".format(str(today)), "rb") as f:
        pred_trend = pkl.load(f)

    start_day = data.index[-1] - pd.Timedelta(days=34)

    day_0 = data.index[-1]
    day_p5 = day_0 + pd.Timedelta(days=5)

    _, target, _, _ = data_pipeline.split_data(
        data, train_start=start_day, test_start=day_0, post_test=day_p5
    )

    target = target["34"]

    prediction_samples = np.reshape(pred["y"], (pred["y"].shape[0], -1))

    target_sorted = np.sort(target.to_numpy())
    p_t = (np.arange(len(target_sorted)) + 0.5) / len(target_sorted)

    prediction_quantiles_trend = np.quantile(prediction_samples, p_t, axis=0)
    # averaging over all data points (not really sensible -> not used)
    number_predictions = prediction_quantiles_trend.shape[1]
    prediction_quantiles_trend = prediction_quantiles_trend.sum(axis=1)/ number_predictions

    plt.scatter(prediction_quantiles_trend, target_sorted)
    plt.plot([0, max(prediction_quantiles_trend.max(), target_sorted.max())], [0, max(prediction_quantiles_trend.max(), target_sorted.max())])
    plt.show()

# Not used, some ideas for a chi-squared-test
def chi_squared_test(today):
    data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv")

    with open("predictions/predictions_ar_{}".format(str(today)), "rb") as f:
        pred = pkl.load(f)
    with open("predictions/predictions_trend_ar_{}".format(str(today)), "rb") as f:
        pred_trend = pkl.load(f)

    start_day = data.index[-1] - pd.Timedelta(days=34)

    day_0 = data.index[-1]
    day_p5 = day_0 + pd.Timedelta(days=5)

    _, target, _, _ = data_pipeline.split_data(
        data, train_start=start_day, test_start=day_0, post_test=day_p5
    )

    target = target["34"]

    prediction_samples = np.reshape(pred["y"], (pred["y"].shape[0], -1))
    prob_borders = np.linspace(0,1,8)
    prediction_quantiles_trend = np.quantile(prediction_samples, prob_borders, axis=0)
    counts = np.zeros(7)
    expected_counts = np.zeros(7)
    for j, elem in enumerate(target):
        for i, border in enumerate(prediction_quantiles_trend[:,j]):
            if elem > border and elem <= prediction_quantiles_trend[i+1,j]:
                counts[i] += 1
    for bin in range(len(counts)):
        expected_counts[bin] = len(target) * (prob_borders[bin+1] - prob_borders[bin])
    ddof = len(counts)-1-2
    probability_regression = chisquare(f_obs=counts.astype(int), f_exp=expected_counts.astype(int))
    print(probability_regression)
    plt.bar(range(len(counts)), counts)
    plt.plot(range(len(counts)), expected_counts, 'g')
    plt.show()

# The negative binomial pdf
def NegBinom(a, m, x):
    pmf = special.binom(x + a - 1, x) * (a / (m + a)) ** a * (m / (m + a)) ** x
    return pmf

# Calculates the likelihood of the target given mu and alpha
def calculate_likelihood(mu, alpha, targets):
    if len(mu.shape) > 1:
        mu = np.mean(mu, axis=0)
    elif mu.shape[0] != 35:
        mu = np.ones_like(targets) * mu
    alpha = np.mean(alpha)
    targets = targets.to_numpy()
    log_likelihood = 0
    for i, target in enumerate(targets):
        log_likelihood += np.log(NegBinom(alpha, mu[i], target))
    return log_likelihood

# Reads the models given the parameters what to read and calculates the likelihoods
def likelihood(today, parallel_prediction, targets, fixed_params=False, features=None):
    file_change = ''
    if fixed_params:
        file_change = '_fixed'
    with open("trace/trace_ar_{}_baseline".format(str(today)), "rb") as f:
        trace_baseline = pkl.load(f)
    neg_log_likelihood_baseline = -1 * calculate_likelihood(np.ones((35)) * np.mean(targets.to_numpy()), trace_baseline["alpha"], targets)
    if parallel_prediction:
        with open("trace/trace_ar_{}".format(str(today)), "rb") as f:
            trace = pkl.load(f)
        neg_log_likelihood = -1 * calculate_likelihood(trace["mu"], trace["alpha"], targets)
        return neg_log_likelihood, neg_log_likelihood_baseline
    else:
        with open("trace/trace_ar_{}{}".format(str(today), file_change), "rb") as f:
            trace = pkl.load(f)
        print(f'Mean tree accept: {np.mean(trace.get_sampler_stats("mean_tree_accept"))}, diverging: {np.mean(trace.get_sampler_stats("diverging"))}')
        with open("predictions/predictions_trend_ar__only_trend{}".format(str(today)), "rb") as f:
            pred = pkl.load(f)
        with open("predictions/predictions_trend_ar_{}".format(str(today)), "rb") as f:
            pred2 = pkl.load(f)
        mu = trace['mu']


        phi_0 = np.mean(trace['phi_0'])
        phi_1 = np.mean(trace['phi_1'])
        phi_2 = np.mean(trace['phi_2'])
        phi_0_neighbour = np.mean(trace['phi_0_neighbour'])
        phi_1_neighbour = np.mean(trace['phi_1_neighbour'])
        phi_2_neighbour = np.mean(trace['phi_2_neighbour'])
        phi_self = np.mean(trace['phi_self'])

        current_county = '34'
        data = features[current_county][:-7]
        data_weekly_mean = features[current_county + '_weekly_mean'][6:-1]
        data_weekly_mean_squared = features[current_county + '_weekly_mean_squared'][6:-1]
        data_weekly_mean_previous = features[current_county + '_weekly_mean'][:-7]
        data = data_weekly_mean_previous - data
        interaction_county = '35'
        interaction_data = features[interaction_county][:-7]
        interaction_data_weekly_mean = features[interaction_county + '_weekly_mean'][6:-1]
        interaction_data_weekly_mean_squared = features[interaction_county + '_weekly_mean_squared'][6:-1]
        interaction_data_weekly_mean_previous = features[interaction_county + '_weekly_mean'][:-7]
        interaction_data = interaction_data_weekly_mean_previous - interaction_data

        prediction_self = phi_0 * data[0] + phi_1 * data_weekly_mean[0] + phi_2 * data_weekly_mean_squared[0]
        prediction_neighbour = phi_0_neighbour * interaction_data[0] + phi_1_neighbour * interaction_data_weekly_mean[0] + phi_2_neighbour * interaction_data_weekly_mean_squared[0]

        neg_log_likelihood = -1 * calculate_likelihood(mu, trace["alpha"], targets)
        with open("trace/trace_ar_{}_generation{}".format(str(today), file_change), "rb") as f:
            trace_no_interaction = pkl.load(f)
        neg_log_likelihood_no_interaction = -1 * calculate_likelihood(trace_no_interaction["mu"], trace_no_interaction["alpha"], targets)
        return neg_log_likelihood, neg_log_likelihood_baseline, neg_log_likelihood_no_interaction

# Saves the likelihoods and mse to a file
def analysis(today, parallel_prediction, targets, fixed_params=False, features=None):
    file_change = ''
    if fixed_params:
        file_change = '_fixed'
    with open("predictions/predictions_trend_ar_{}{}".format(str(today), file_change), "rb") as f:
        pred_trend = pkl.load(f)
        mse_interaction = pred_trend['mse']
    with open("predictions/predictions_baseline_{}".format(str(today)), "rb") as f:
        pred_baseline = pkl.load(f)
        mse_baseline = pred_baseline['mse']
    if parallel_prediction:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            neg_log_likelihood, neg_log_likelihood_baseline = likelihood(today, parallel_prediction, targets, fixed_params, features)
    if not parallel_prediction:
        with open("predictions/predictions_input_generation_{}{}".format(str(today), file_change), "rb") as f:
            pred_trend_no_interaction = pkl.load(f)
            mse_no_interaction = pred_trend_no_interaction['mse']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            neg_log_likelihood, neg_log_likelihood_baseline, neg_log_likelihood_no_interaction = likelihood(today, parallel_prediction, targets, fixed_params, features)
    try:
        cd = os.getcwd()  # has to be adapted for final data structure
        file = cd + "/" + "analysis_data.csv"
        data = pd.read_csv(file, index_col='date')  # , encoding='latin-1')
        try:
            data.at[today.strftime('%Y/%m/%d'), 'mse_interaction'] = mse_interaction
            data.at[today.strftime('%Y/%m/%d'), 'neg_log_likelihood_interaction'] = neg_log_likelihood
            data.at[today.strftime('%Y/%m/%d'), 'mse_baseline'] = mse_baseline
            data.at[today.strftime('%Y/%m/%d'), 'neg_log_likelihood_baseline'] = neg_log_likelihood_baseline
            if not parallel_prediction:
                data.at[today.strftime('%Y/%m/%d'), 'mse_no_interaction'] = mse_no_interaction
                data.at[today.strftime('%Y/%m/%d'), 'neg_log_likelihood_no_interaction'] = neg_log_likelihood_no_interaction
        except:
            if parallel_prediction:
                new_row = {'date': today.strftime('%Y/%m/%d'), 'mse_interaction': mse_interaction, 'mse_baseline': mse_baseline, 'neg_log_likelihood_interaction': neg_log_likelihood, 'neg_log_likelihood_baseline': neg_log_likelihood_baseline}
            else:
                new_row = {'date': today.strftime('%Y/%m/%d'), 'mse_interaction': mse_interaction, 'mse_baseline': mse_baseline, 'mse_no_interaction': mse_no_interaction, 'neg_log_likelihood_interaction': neg_log_likelihood, 'neg_log_likelihood_baseline': neg_log_likelihood_baseline, 'neg_log_likelihood_no_interaction': neg_log_likelihood_no_interaction}
            data=data.append(new_row, ignore_index=True)
            data.sort_values(by='date', inplace=True)
    except:
        if parallel_prediction:
            data = pd.DataFrame({'mse_interaction': [mse_interaction], 'mse_baseline': [mse_baseline], 'neg_log_likelihood_interaction': [neg_log_likelihood], 'neg_log_likelihood_baseline': [neg_log_likelihood_baseline]}, index=[today.strftime('%Y/%m/%d')])
        else:
            data = pd.DataFrame({'mse_interaction': [mse_interaction], 'mse_baseline': [mse_baseline], 'mse_no_interaction': [mse_no_interaction], 'neg_log_likelihood_interaction':[neg_log_likelihood], 'neg_log_likelihood_baseline': [neg_log_likelihood_baseline], 'neg_log_likelihood_no_interaction': [neg_log_likelihood_no_interaction]}, index=[today.strftime('%Y/%m/%d')])
        data.index.name = 'date'
    data.to_csv(cd + "/" + "analysis_data.csv")

# Reads the models given the parameters what to read and calculates the likelihoods for the future data
def likelihood_future(today, parallel_prediction, features, targets, fixed_params=False):
    file_change = ''
    if fixed_params:
        file_change = '_fixed'
    with open("trace/trace_ar_{}_baseline".format(str(today)), "rb") as f:
        trace_baseline = pkl.load(f)
    neg_log_likelihood_baseline = -1 * calculate_likelihood(np.ones((5)) * np.mean(features.to_numpy()) * 0.8, trace_baseline["alpha"], targets)
    if parallel_prediction:
        with open("trace/trace_ar_{}".format(str(today)), "rb") as f:
            trace = pkl.load(f)
        neg_log_likelihood = -1 * calculate_likelihood(trace["mu"], trace["alpha"], targets)
        return neg_log_likelihood, neg_log_likelihood_baseline
    else:
        with open("trace/trace_ar_{}{}".format(str(today), file_change), "rb") as f:
            trace = pkl.load(f)
        print(f'Mean tree accept: {np.mean(trace.get_sampler_stats("mean_tree_accept"))}, diverging: {np.mean(trace.get_sampler_stats("diverging"))}')
        with open("predictions/predictions_trend_ar_{}".format(str(today)), "rb") as f:
            pred = pkl.load(f)
        mu = pred['y'][-5:]


        neg_log_likelihood = -1 * calculate_likelihood(mu, trace["alpha"], targets)
        with open("trace/trace_ar_{}_generation{}".format(str(today), file_change), "rb") as f:
            trace_no_interaction = pkl.load(f)
        # pm_data_no_interaction = az.from_pymc3(trace_no_interaction)
        # neg_log_likelihood_no_interaction = -1 * pm_data_no_interaction.log_likelihood.Y_obs.sum().item()
        neg_log_likelihood_no_interaction = -1 * calculate_likelihood(trace_no_interaction["mu"], trace_no_interaction["alpha"], targets)
        return neg_log_likelihood, neg_log_likelihood_baseline, neg_log_likelihood_no_interaction

# Saves the likelihoods and mse to a file
def analysis_future(today, parallel_prediction, features, targets, fixed_params=False):
    file_change = ''
    if fixed_params:
        file_change = '_fixed'
    with open("predictions/predictions_trend_ar_{}{}".format(str(today), file_change), "rb") as f:
        pred_trend = pkl.load(f)
    with open("predictions/predictions_baseline_{}".format(str(today)), "rb") as f:
        pred_baseline = pkl.load(f)
    if parallel_prediction:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            neg_log_likelihood, neg_log_likelihood_baseline = likelihood_future(today, parallel_prediction, features, targets, fixed_params)
    if not parallel_prediction:
        with open("predictions/predictions_input_generation_{}{}".format(str(today), file_change), "rb") as f:
            pred_trend_no_interaction = pkl.load(f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            neg_log_likelihood, neg_log_likelihood_baseline, neg_log_likelihood_no_interaction = likelihood_future(today, parallel_prediction, features, targets, fixed_params)
    print(f'\n {today}')
    if not parallel_prediction:
        likelihood_ratio = np.exp((-1 * neg_log_likelihood) - (-1 * neg_log_likelihood_no_interaction))
        print(f"Likelihood ratio of Interaction over no Interaction for future data: {likelihood_ratio}")
        print(f'Likelihoods: Interaction: {neg_log_likelihood}, Baseline: {neg_log_likelihood_baseline}, No Interaction: {neg_log_likelihood_no_interaction}')
    likelihood_ratio = np.exp(
        (-1 * neg_log_likelihood) - (-1 * neg_log_likelihood_baseline))
    print(f"Likelihood ratio of Interaction over Baseline for future data: {likelihood_ratio}")
    if not parallel_prediction:
        likelihood_ratio = np.exp(
            (-1 * neg_log_likelihood_no_interaction) - (-1 * neg_log_likelihood_baseline))
        print(f"Likelihood ratio of no Interaction over Baseline for future data: {likelihood_ratio}")


# Plots the analysis that is given in the file
def show_mse():
    cd = os.getcwd()  # has to be adapted for final data structure
    file = cd + "/" + "analysis_data.csv"
    data = pd.read_csv(file, index_col='date')  # , encoding='latin-1')
    likelihood_ratio = np.exp((-1 * data['neg_log_likelihood_interaction']) - (-1 * data['neg_log_likelihood_no_interaction']))
    print(f"Likelihood ratio of Interaction over no Interaction: {likelihood_ratio}")
    print(f"Likelihood ratio averaged over all intervals: {np.prod(likelihood_ratio)}")
    likelihood_ratio = np.exp(
        (-1 * data['neg_log_likelihood_interaction']) - (-1 * data['neg_log_likelihood_baseline']))
    print(f"Likelihood ratio of Interaction over Baseline: {likelihood_ratio}")
    likelihood_ratio = np.exp(
        (-1 * data['neg_log_likelihood_no_interaction']) - (-1 * data['neg_log_likelihood_baseline']))
    print(f"Likelihood ratio of no Interaction over Baseline: {likelihood_ratio}")

def geweke_analysis():
    fig, axis = plt.subplots(figsize=(20, 10), nrows=2, ncols=3)
    for index, today in enumerate([date(2020, 9, 10), date(2020, 11, 1), date(2020, 12, 15), date(2021, 5, 15), date(2021, 9, 25), date(2022, 2, 1)]):
        ax = axis[index // 3, index % 3]
        with open("trace/trace_ar_{}".format(str(today)), "rb") as f:
            trace = pkl.load(f)
        obs_all = np.zeros(4)
        # print(f'Mean tree accept: {np.mean(trace.get_sampler_stats("mean_tree_accept"))}, diverging: {np.mean(trace.get_sampler_stats("diverging"))}')
        for param in ['phi_self', 'phi_0', 'phi_1', 'phi_2', 'phi_0_neighbour', 'phi_1_neighbour', 'phi_2_neighbour', 'alpha']:
            score = pm.geweke(trace[param])
            ax.scatter(np.arange(20)+1, score[:, 1], label=param, alpha=0.7)
            sigma_3 = np.sum(np.where(np.logical_or(score[:, 1] < -2, score[:, 1] > 2), True, False))
            sigma_2 = np.sum(np.where(np.logical_or(np.logical_and(score[:, 1]>-2, score[:, 1]<-1), np.logical_and(score[:, 1]>1, score[:, 1]<2)), True, False))
            sigma_1_minus = np.sum(np.where(np.logical_and(score[:, 1] > -1, score[:, 1] < 0), True, False))
            sigma_1_plus = np.sum(np.where(np.logical_and(score[:, 1] > 0, score[:, 1] < 1), True, False))
            obs_all = [obs_all[0] + sigma_3, obs_all[1] + sigma_2, obs_all[2] + sigma_1_plus, obs_all[3] + sigma_1_minus]
        k2, p = chisquare(f_obs=obs_all, f_exp=[8, 40, 56, 56])
        print("p = {:g}".format(p))
        # if p < 0.05:
        #     print(f'Evidence against convergence for param {param} at day {today.strftime("%Y/%m/%d")}')
        if p > 0.05:
            print(f'No Evidence against convergence at day {today.strftime("%Y/%m/%d")}')

        ax.axhline(-1.98, c='r')
        ax.axhline(1.98, c='r')
        ax.set_ylim(-3, 3)
        ax.legend(loc='lower left')
        ax.set_title(f't = {today.strftime("%Y/%m/%d")}', fontsize=16)
        # plt.xlim(0 - 10, .5 * trace['Mean of Data'].shape[0] / 2 + 10)
        # plt.title('Geweke Plot Comparing first 10% and Slices of the Last 50% of Chain\nDifference in Mean Z score')

    cd = os.getcwd()
    plots = cd + "\\Plots\\"
    currentplots = plots + 'convergence_analysis'

    try:
        os.mkdir(currentplots)
    except OSError:
        print("Creation of the directory %s failed" % currentplots)
    else:
        print("Successfully created the directory %s " % currentplots)
    plt.savefig(currentplots + "/Osnabrück_convergence_analysis.png")
