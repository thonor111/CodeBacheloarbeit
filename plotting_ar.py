import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from matplotlib import cm
from matplotlib.colors import ListedColormap
from datetime import date

import data_pipeline


def sample_x_days_incidence_by_county(samples, x):
    num_sample = len(samples)
    timesteps = len(samples[0])
    counties = len(samples[0][0])
    incidence = np.empty((num_sample, timesteps - x, counties), dtype="int64")
    for sample in range(num_sample):
        for week in range(timesteps - x):
            incidence[sample][week] = np.sum(samples[sample][week: week + x], axis=0)
    return incidence


def plot(today, additional_info, only_trend=True, fixed_params=False):
    """
    Plots the Regression for the given day
    :param today: For which day to plot the regression
    :param additional_info:
    :param only_trend: If only the trend component of the regression should be plotted
    :param fixed_params: If the params were fixed beforehand (which file to load the weights from)
    """
    file_change = ''
    if fixed_params:
        file_change = '_fixed'
    start_day = datetime.combine(today - pd.Timedelta(days=36), datetime.strptime("0000", "%H%M").time())
    data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv", start_day)
    data[data < 0] = 0  # to get rid of the negative values

    if only_trend:
        with open("predictions/predictions_ar_only_trend{}{}".format(str(today), file_change), "rb") as f:
            pred = pkl.load(f)
        with open("predictions/predictions_trend_ar__only_trend{}{}".format(str(today), file_change), "rb") as f:
            pred_trend = pkl.load(f)
    else:
        with open("predictions/predictions_ar_{}{}".format(str(today), file_change), "rb") as f:
            pred = pkl.load(f)
        with open("predictions/predictions_trend_ar_{}{}".format(str(today), file_change), "rb") as f:
            pred_trend = pkl.load(f)


    day_0 = datetime.combine(today - pd.Timedelta(days=1), datetime.strptime("0000", "%H%M").time())
    day_p5 = day_0 + pd.Timedelta(days=4)

    target_counties = data.columns

    _, target, _, _ = data_pipeline.split_data(
        data, train_start=start_day, test_start=day_0, post_test=day_p5
    )
    ext_index = pd.date_range(start_day, day_p5)

    prediction_samples = np.reshape(pred["y"], (pred["y"].shape[0], -1))
    prediction_samples_trend = np.reshape(
        pred_trend["y"], (pred_trend["y"].shape[0], -1)
    )
    prediction_quantiles_trend = np.quantile(prediction_samples, [0.05, 0.25, 0.75, 0.95], axis=0)

    IDNameDem = additional_info

    # colors for curves
    C1 = "#D55E00"
    C2 = "#E69F00"

    cd = os.getcwd()

    plots = cd + "\\Plots\\"
    currentplots = plots + str(today)
    try:
        os.mkdir(plots)
    except OSError:
        print("Creation of the directory %s failed" % plots)
    else:
        print("Successfully created the directory %s " % plots)

    try:
        os.mkdir(currentplots)
    except OSError:
        print("Creation of the directory %s failed" % currentplots)
    else:
        print("Successfully created the directory %s " % currentplots)

    dates = [pd.Timestamp(day) for day in ext_index]
    itsgettingworse = 0
    thesearegettingworse = []

    county_id = str(34)
    fig, ax = plt.subplots(figsize=(10, 7))

    if prediction_samples_trend[-5] - prediction_samples_trend[-6] > 0:
        itsgettingworse += 1
        thesearegettingworse.append(IDNameDem[int(county_id)].get("NL Name"))
    ax.plot_date(dates[:-5],
                 target[county_id],
                 color="k",
                 label="Bestaetigte Infektionen - {}".format(IDNameDem[int(county_id)].get("NL Name")))

    ax.plot_date(
        dates,
        prediction_samples_trend,
        "-",
        color=C1,
        label="Autoregressive Prediction - {}".format(IDNameDem[int(county_id)].get("NL Name")),
        linewidth=2.0,
        zorder=4)

    ax.fill_between(
        dates,
        prediction_quantiles_trend[1],
        prediction_quantiles_trend[2],
        facecolor=C2,
        alpha=0.5,
        zorder=1)
    ax.plot_date(
        dates,
        prediction_quantiles_trend[1],
        ":",
        color=C2,
        label="Q25",
        linewidth=2.0,
        zorder=3)
    ax.plot_date(dates,  # upper line
                 prediction_quantiles_trend[3], ":",
                 label="Q05-Q95",
                 color="green", alpha=0.5, linewidth=2.0, zorder=1)
    ax.axvline(day_0, label='Tag der Datenerhebung')
    ax.legend(loc="upper left")  # everything above will be included in legend
    ax.fill_between(
        dates,
        prediction_quantiles_trend[0],
        prediction_quantiles_trend[3],
        facecolor="green",
        alpha=0.25,
        zorder=0)
    ax.plot_date(dates,  # lower line
                 prediction_quantiles_trend[0],
                 ":",
                 color="green", alpha=0.5, linewidth=2.0, zorder=1)
    ax.plot_date(  # upper of q25
        dates,
        prediction_quantiles_trend[2],
        ":",
        color=C2,
        linewidth=2.0,
        zorder=3)

    ax.xaxis.set_tick_params(rotation=30, labelsize=10)
    ax.set_ylabel("Anzahl Infektionen")

    highest_value = target[county_id].max()
    ax.set_ylim(0, (int(highest_value) * 3))
    if only_trend:
        plt.savefig(currentplots + "/{}_{}_Prediction_trend_only{}.png".format(today, IDNameDem[int(county_id)].get("NL Name"), file_change))
    else:
        plt.savefig(currentplots + "/{}_{}_Prediction{}.png".format(today, IDNameDem[int(county_id)].get("NL Name"), file_change))

    print(
        "Fuer {} der {} Kreise wird ein Anstieg der Fallzahlen prognostiziert. Die betroffenen Landkreise sind: {}".format(
            itsgettingworse, len(data.columns), thesearegettingworse))

    file = open(currentplots + "Prognose_{}.txt".format(today), "w")
    file.write(
        "Fuer {} der {} Kreise wird ein Anstieg der Fallzahlen prognostiziert. Die betroffenen Landkreise sind: {}".format(
            itsgettingworse, len(data.columns), thesearegettingworse))
    file.close()

def plot_both(additional_info, fixed_params=False):
    """
    Plots the Regression for the given day
    :param today: For which day to plot the regression
    :param additional_info:
    :param fixed_params: If the params were fixed beforehand (which file to load the weights from)
    """
    file_change = ''
    if fixed_params:
        file_change = '_fixed'

    cd = os.getcwd()

    plots = cd + "\\Plots\\"
    currentplots = plots + 'all_plots'
    try:
        os.mkdir(plots)
    except OSError:
        print("Creation of the directory %s failed" % plots)
    else:
        print("Successfully created the directory %s " % plots)

    try:
        os.mkdir(currentplots)
    except OSError:
        print("Creation of the directory %s failed" % currentplots)
    else:
        print("Successfully created the directory %s " % currentplots)

    plt.close()
    fig, axis = plt.subplots(nrows=6, ncols=2, figsize=(20, 42))

    for index, today in enumerate([date(2020, 9, 10), date(2020, 11, 1), date(2020, 12, 15), date(2021, 5, 15), date(2021, 9, 25),
                  date(2022, 2, 1)]):

        start_day = datetime.combine(today - pd.Timedelta(days=36), datetime.strptime("0000", "%H%M").time())
        data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv", start_day)
        data[data < 0] = 0  # to get rid of the negative values

        for plot_nr in [0, 1]:
            if plot_nr == 1:
                with open("predictions/predictions_ar_only_trend{}{}".format(str(today), file_change), "rb") as f:
                    pred = pkl.load(f)
                with open("predictions/predictions_trend_ar__only_trend{}{}".format(str(today), file_change), "rb") as f:
                    pred_trend = pkl.load(f)
            else:
                with open("predictions/predictions_ar_{}{}".format(str(today), file_change), "rb") as f:
                    pred = pkl.load(f)
                with open("predictions/predictions_trend_ar_{}{}".format(str(today), file_change), "rb") as f:
                    pred_trend = pkl.load(f)


            day_0 = datetime.combine(today - pd.Timedelta(days=1), datetime.strptime("0000", "%H%M").time())
            day_p5 = day_0 + pd.Timedelta(days=4)

            target_counties = data.columns

            _, target, _, _ = data_pipeline.split_data(
                data, train_start=start_day, test_start=day_0, post_test=day_p5
            )
            ext_index = pd.date_range(start_day, day_p5)

            prediction_samples = np.reshape(pred["y"], (pred["y"].shape[0], -1))
            prediction_samples_trend = np.reshape(
                pred_trend["y"], (pred_trend["y"].shape[0], -1)
            )
            prediction_quantiles_trend = np.quantile(prediction_samples, [0.05, 0.25, 0.75, 0.95], axis=0)

            IDNameDem = additional_info

            # colors for curves
            C1 = "#D55E00"
            C2 = "#E69F00"

            dates = [pd.Timestamp(day) for day in ext_index]
            itsgettingworse = 0
            thesearegettingworse = []

            county_id = str(34)

            ax = axis[index, plot_nr]

            if plot_nr == 0:
                ax.set_title(f'{today.strftime("%Y/%m/%d")}: Regression with periodic element', fontsize=16)
            else:
                ax.set_title(f'{today.strftime("%Y/%m/%d")}: Regression without periodic element', fontsize=16)

            if prediction_samples_trend[-5] - prediction_samples_trend[-6] > 0:
                itsgettingworse += 1
                thesearegettingworse.append(IDNameDem[int(county_id)].get("NL Name"))
            ax.plot_date(dates[:-5],
                         target[county_id],
                         color="k",
                         label="Infection counts - {}".format(IDNameDem[int(county_id)].get("NL Name")))

            ax.plot_date(
                dates,
                prediction_samples_trend,
                "-",
                color=C1,
                label="Autoregressive Prediction - {}".format(IDNameDem[int(county_id)].get("NL Name")),
                linewidth=2.0,
                zorder=4)

            ax.fill_between(
                dates,
                prediction_quantiles_trend[1],
                prediction_quantiles_trend[2],
                facecolor=C2,
                alpha=0.5,
                zorder=1)
            ax.plot_date(
                dates,
                prediction_quantiles_trend[1],
                ":",
                color=C2,
                label="Q25",
                linewidth=2.0,
                zorder=3)
            ax.plot_date(dates,  # upper line
                         prediction_quantiles_trend[3], ":",
                         label="Q05-Q95",
                         color="green", alpha=0.5, linewidth=2.0, zorder=1)
            ax.axvline(day_0, label='Current day')
            if plot_nr == 0:
                ax.legend(loc="upper left")  # everything above will be included in legend
            ax.fill_between(
                dates,
                prediction_quantiles_trend[0],
                prediction_quantiles_trend[3],
                facecolor="green",
                alpha=0.25,
                zorder=0)
            ax.plot_date(dates,  # lower line
                         prediction_quantiles_trend[0],
                         ":",
                         color="green", alpha=0.5, linewidth=2.0, zorder=1)
            ax.plot_date(  # upper of q25
                dates,
                prediction_quantiles_trend[2],
                ":",
                color=C2,
                linewidth=2.0,
                zorder=3)

            ax.xaxis.set_tick_params(rotation=30, labelsize=10)
            ax.set_ylabel("Counts of infections")

            highest_value = prediction_quantiles_trend.max()
            ax.set_ylim(0, (int(highest_value) * 1.3))
    plt.savefig(currentplots + "/{}_Prediction_all{}.png".format(IDNameDem[int(county_id)].get("NL Name"), file_change))


def plot_function(today, fixed_params=False):
    """
    Plots the transformation function of the ar model
    :param today: For which day to plot the transformation
    :param fixed_params: Which file to load
    """
    file_change = ''
    if fixed_params:
        file_change = '_fixed'
    with open("predictions/predictions_trend_ar_{}{}".format(str(today), file_change), "rb") as f:
        pred = pkl.load(f)
    phi_1 = pred["phi"][1]
    phi_2 = pred["phi"][2]
    values_n = np.linspace(0, np.max(pred["y"]) * 1.5, 500)
    values_n_plus_one = phi_1 * values_n + phi_2 * values_n ** 2
    critical_point = (1 - phi_1) / phi_2
    current_point = pred["y"][-1]
    # print('Distance of the latest value to the stable point:', pred['y'][-1] - critical_point)
    # plt.clf()
    # plt.plot(values_n, values_n_plus_one, label='transformation function')
    # plt.plot(values_n, values_n, label='no change')
    # plt.fill_between(values_n, values_n, values_n_plus_one, where=values_n > values_n_plus_one, facecolor='green', alpha=0.3)
    # plt.fill_between(values_n, values_n, values_n_plus_one, where=values_n < values_n_plus_one, facecolor='red', alpha=0.3)
    # # plt.plot(np.ones_like(values_n) * critical_point, values_n, label='predicted stable value')
    # plt.plot(np.ones_like(values_n) * current_point, values_n, label='latest mean', linestyle="-.")
    # plt.xlabel('mean of days [n-6, n]')
    # plt.ylabel('Estimated mean of days [n-5, n+1]')
    # plt.legend()
    # plt.xlim(0, np.max(pred["y"]) * 1.5)
    # plt.ylim(0, np.max(pred["y"]) * 1.5)
    # plt.show()

    try:
        phi_1_neighbour = pred["phi_neighbour"][1]
        phi_2_neighbour = pred["phi_neighbour"][2]
        phi_self = pred["phi_self"]
    except:
        return
    values_n = np.linspace(0, np.max(pred["y"]) * 1.5, 500)
    values_n_neighbour = np.linspace(0, np.max(pred["y"]) * 1.5, 500)
    pred_self = (phi_1 * values_n + phi_2 * values_n ** 2) * phi_self
    pred_neighbour = (phi_1_neighbour * values_n_neighbour + phi_2_neighbour * values_n_neighbour ** 2) * (1 - phi_self)
    values_n_plus_one = np.zeros((500,500))
    for row in range(500):
        for column in range(500):
            values_n_plus_one[row, column] = pred_self[row] + pred_neighbour[column]

    # TODO substract mean and weighted mean of os and lk
    for column in range(values_n_plus_one.shape[0]):
        values_n_plus_one[:, column] -= values_n
    values_n_plus_one[0, 0] = np.min(values_n_plus_one) * -1
    values_n_plus_one[0, 1] = np.max(values_n_plus_one) * -1

    c_map = cm.get_cmap('RdYlGn', 256)
    reversed_map = np.flip(c_map(np.linspace(0, 1, 256)), axis=0)
    new_cmap = ListedColormap(reversed_map)


    fig, ax = plt.subplots()
    im = ax.imshow(values_n_plus_one, cmap=new_cmap, origin='lower')
    if np.max(pred['y']) > 200:
        ax.set_xticks(np.arange(0, np.max(pred["y"]) * 1.5, 100) * (500 / (np.max(pred["y"]) * 1.5)))
        ax.set_yticks(np.arange(0, np.max(pred["y"]) * 1.5, 100) * (500 / (np.max(pred["y"]) * 1.5)))
        ax.set_xticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 100))
        ax.set_yticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 100))
    else:
        ax.set_xticks(np.arange(0, np.max(pred["y"]) * 1.5, 25) * (500 / (np.max(pred["y"]) * 1.5)))
        ax.set_yticks(np.arange(0, np.max(pred["y"]) * 1.5, 25) * (500 / (np.max(pred["y"]) * 1.5)))
        ax.set_xticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 25))
        ax.set_yticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 25))
    ax.set_xlabel("mean of days [n-6, n] for the surrounding area")
    ax.set_ylabel("mean of days [n-6, n] for Osnabrück")
    fig.colorbar(im, orientation='vertical')
    fig.suptitle("Estimated difference of mean of days [n-5, n+1] t0 [n-6, n] for Osnabrück")
    plt.show()


def plot_all_functions(fixed_params=False):
    """
    Plots the transformation function of the ar model
    :param today: For which day to plot the transformation
    :param fixed_params: Which file to load
    """
    file_change = ''
    if fixed_params:
        file_change = '_fixed'

    cd = os.getcwd()

    plots = cd + "\\Plots\\"
    currentplots = plots + 'all_plots'
    try:
        os.mkdir(plots)
    except OSError:
        print("Creation of the directory %s failed" % plots)
    else:
        print("Successfully created the directory %s " % plots)

    try:
        os.mkdir(currentplots)
    except OSError:
        print("Creation of the directory %s failed" % currentplots)
    else:
        print("Successfully created the directory %s " % currentplots)

    fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))

    for index, today in enumerate(
            [date(2020, 9, 10), date(2020, 11, 1), date(2020, 12, 15), date(2021, 5, 15), date(2021, 9, 25),
             date(2022, 2, 1)]):
        with open("predictions/predictions_trend_ar_{}{}".format(str(today), file_change), "rb") as f:
            pred = pkl.load(f)
        phi_1 = pred["phi"][1]
        phi_2 = pred["phi"][2]
        values_n = np.linspace(0, np.max(pred["y"]) * 1.5, 500)
        values_n_plus_one = phi_1 * values_n + phi_2 * values_n ** 2
        critical_point = (1 - phi_1) / phi_2
        current_point = pred["y"][-1]

        start_day = datetime.combine(today - pd.Timedelta(days=36), datetime.strptime("0000", "%H%M").time())
        data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv", start_day)
        data[data < 0] = 0  # to get rid of the negative values
        day_0 = datetime.combine(today - pd.Timedelta(days=1), datetime.strptime("0000", "%H%M").time())
        day_p5 = day_0 + pd.Timedelta(days=4)

        _, target, _, _ = data_pipeline.split_data(
            data, train_start=start_day, test_start=day_0, post_test=day_p5
        )

        try:
            phi_1_neighbour = pred["phi_neighbour"][1]
            phi_2_neighbour = pred["phi_neighbour"][2]
            phi_self = pred["phi_self"]
        except:
            return
        values_n = np.linspace(0, np.max(pred["y"]) * 1.5, 500)
        values_n_neighbour = np.linspace(0, np.max(pred["y"]) * 1.5, 500)
        pred_self = (phi_1 * values_n + phi_2 * values_n ** 2) * phi_self
        pred_neighbour = (phi_1_neighbour * values_n_neighbour + phi_2_neighbour * values_n_neighbour ** 2) * (1 - phi_self)
        values_n_plus_one = np.zeros((500,500))
        for row in range(500):
            for column in range(500):
                values_n_plus_one[row, column] = pred_self[row] + pred_neighbour[column]

        # TODO substract mean and weighted mean of os and lk
        for column in range(values_n_plus_one.shape[0]):
            values_n_plus_one[:, column] -= values_n
        values_n_plus_one[0, 0] = np.min(values_n_plus_one) * -1
        values_n_plus_one[0, 1] = np.max(values_n_plus_one) * -1
        # test = values_n_plus_one[:, target['34'][-1]]
        # test2 = [[0, 123, 255]] * 500
        # values_n_plus_one[:, target['34'][-1]] = [[0, 123, 255]] * 500

        c_map = cm.get_cmap('RdYlGn', 256)
        reversed_map = np.flip(c_map(np.linspace(0, 1, 256)), axis=0)
        new_cmap = ListedColormap(reversed_map)

        norm = plt.Normalize(values_n_plus_one.min(), values_n_plus_one.max())
        rgba = new_cmap(norm(values_n_plus_one))

        row_number = int(np.mean(target['34'][-7:]) * (500 / (np.max(pred["y"]) * 1.5)))
        rgba[row_number, :] = [[0, 0, 0, 1]] * 500
        col_number = int(np.mean(target['35'][-7:-1]) * (500 / (np.max(pred["y"]) * 1.5)))
        if col_number < 500:
            rgba[:, col_number] = [[0, 0, 0, 1]] * 500

        ax = axis[index // 3, index % 3]

        im = ax.imshow(rgba, origin='lower')
        if np.max(pred['y']) > 200:
            ax.set_xticks(np.arange(0, np.max(pred["y"]) * 1.5, 100) * (500 / (np.max(pred["y"]) * 1.5)))
            ax.set_yticks(np.arange(0, np.max(pred["y"]) * 1.5, 100) * (500 / (np.max(pred["y"]) * 1.5)))
            ax.set_xticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 100))
            ax.set_yticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 100))
        elif np.max(pred['y']) > 20:
            ax.set_xticks(np.arange(0, np.max(pred["y"]) * 1.5, 10) * (500 / (np.max(pred["y"]) * 1.5)))
            ax.set_yticks(np.arange(0, np.max(pred["y"]) * 1.5, 10) * (500 / (np.max(pred["y"]) * 1.5)))
            ax.set_xticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 10))
            ax.set_yticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 10))
        else:
            ax.set_xticks(np.arange(0, np.max(pred["y"]) * 1.5, 2) * (500 / (np.max(pred["y"]) * 1.5)))
            ax.set_yticks(np.arange(0, np.max(pred["y"]) * 1.5, 2) * (500 / (np.max(pred["y"]) * 1.5)))
            ax.set_xticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 2))
            ax.set_yticklabels(np.arange(0, np.max(pred["y"]) * 1.5, 2))
        ax.set_xlabel("mean of days [t-6, t] for the surrounding area", fontsize=14)
        ax.set_ylabel("mean of days [t-6, t] for Osnabrück", fontsize=14)
        ax.set_title(f't = {today.strftime("%Y/%m/%d")}', fontsize=16)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=new_cmap), ax=ax, orientation='vertical')
        # fig.suptitle("Estimated difference of mean of days [t-5, t+1] to [t-6, t] for Osnabrück")
    plt.savefig(currentplots + "/Osnabrück_Transformation_all{}.png".format(file_change))
