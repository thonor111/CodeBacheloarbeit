import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import timedelta

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


def plot(today, additional_info):
    data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv")

    with open("model_{}".format(str(today)), "rb") as f:
        model = pkl.load(f)
    with open("trace_{}".format(str(today)), "rb") as f:
        trace = pkl.load(f)
    with open("predictions_{}".format(str(today)), "rb") as f:
        pred = pkl.load(f)
    with open("predictions_trend_{}".format(str(today)), "rb") as f:
        pred_trend = pkl.load(f)

    start_day = data.index[-1] - pd.Timedelta(days=42)
    test = data.index[-1]

    day_0 = data.index[-1]
    day_m5 = day_0 - pd.Timedelta(days=5)
    day_p5 = day_0 + pd.Timedelta(days=5)

    # target_counties = data_selection.columns
    target_counties = data.columns
    num_counties = len(target_counties)

    _, target, _, _ = data_pipeline.split_data(
        data, train_start=start_day, test_start=day_0, post_test=day_p5
    )
    # print(target)
    ext_index = pd.date_range(start_day, day_p5 - timedelta(1))

    # changed res to pred and res_trend to pred_trend
    prediction_samples = np.reshape(pred["y"], (pred["y"].shape[0], -1, num_counties))
    prediction_samples_mu = np.reshape(pred["mu"], (pred["mu"].shape[0], -1, num_counties))
    prediction_samples_trend = np.reshape(
        pred_trend["y"], (pred_trend["y"].shape[0], -1, num_counties)
    )
    prediction_samples_trend_mu = np.reshape(
        pred_trend["mu"], (pred_trend["mu"].shape[0], -1, num_counties)
    )
    prediction_samples_mu.shape

    predictions_7day_inc = sample_x_days_incidence_by_county(
        prediction_samples_trend, 7
    )
    predictions_7day_inc_mu = sample_x_days_incidence_by_county(
        prediction_samples_trend_mu, 7
    )

    prediction_quantiles = np.quantile(prediction_samples, [0.05, 0.25, 0.75, 0.95], axis=0)
    prediction_quantiles_trend = np.quantile(prediction_samples_trend, [0.05, 0.25, 0.75, 0.95], axis=0)
    prediction_quantiles_7day_inc = np.quantile(predictions_7day_inc, [0.05, 0.25, 0.75, 0.95], axis=0)
    # print("Quantiles:\n", prediction_quantiles) # why are all values the same? will this change with proper training?

    prediction_mean = pd.DataFrame(
        data=np.mean(prediction_samples_mu, axis=0),
        index=ext_index,
        columns=target.columns,
    )
    # print("Prediction Mean:\n", prediction_mean)
    prediction_q25 = pd.DataFrame(
        data=prediction_quantiles[1], index=ext_index, columns=target.columns
    )
    prediction_q75 = pd.DataFrame(
        data=prediction_quantiles[2], index=ext_index, columns=target.columns
    )
    prediction_q5 = pd.DataFrame(
        data=prediction_quantiles[0], index=ext_index, columns=target.columns
    )
    prediction_q95 = pd.DataFrame(
        data=prediction_quantiles[3], index=ext_index, columns=target.columns
    )
    # print("Prediction q95:\n", prediction_q95)

    prediction_mean_trend = pd.DataFrame(
        data=np.mean(prediction_samples_trend_mu, axis=0),
        index=ext_index,
        columns=target.columns,
    )
    prediction_q25_trend = pd.DataFrame(
        data=prediction_quantiles_trend[1], index=ext_index, columns=target.columns
    )
    prediction_q75_trend = pd.DataFrame(
        data=prediction_quantiles_trend[2], index=ext_index, columns=target.columns
    )
    prediction_q5_trend = pd.DataFrame(
        data=prediction_quantiles_trend[0], index=ext_index, columns=target.columns
    )
    prediction_q95_trend = pd.DataFrame(
        data=prediction_quantiles_trend[3], index=ext_index, columns=target.columns
    )

    prediction_mean_7day = pd.DataFrame(
        data=np.pad(
            np.mean(predictions_7day_inc_mu, axis=0),
            ((7, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        ),
        index=ext_index,
        columns=target.columns,
    )

    prediction_q75_7day = pd.DataFrame(
        data=np.pad(
            prediction_quantiles_7day_inc[2].astype(float),
            ((7, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        ),
        index=ext_index,
        columns=target.columns,
    )
    prediction_q5_7day = pd.DataFrame(
        data=np.pad(
            prediction_quantiles_7day_inc[0].astype(float),
            ((7, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        ),
        index=ext_index,
        columns=target.columns,
    )
    prediction_q95_7day = pd.DataFrame(
        data=np.pad(
            prediction_quantiles_7day_inc[3].astype(float),
            ((7, 0), (0, 0)),
            "constant",
            constant_values=np.nan,
        ),
        index=ext_index,
        columns=target.columns,
    )

    rki_7day = target.rolling(7).sum()

    ref_date = target.iloc[-1].name
    nowcast_vals = prediction_mean.loc[prediction_mean.index == ref_date]
    nowcast7day_vals = prediction_mean_7day.loc[prediction_mean.index == ref_date]
    rki_vals = target.iloc[-1]
    rki_7day_vals = rki_7day.iloc[-1]

    IDNameDem = additional_info

    # colors for curves
    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = "#0073CF"

    # quantiles we want to plot
    qs = [0.25, 0.50, 0.75, 0.95]

    i_start_day = (start_day - data.index.min()).days

    county_ids = target.columns

    # Load our prediction samples
    res = pred
    n_days = (day_p5 - start_day).days

    prediction_samples = prediction_samples[:, i_start_day:i_start_day + n_days, :]

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
    days = [(day - min(dates)).days for day in dates]
    itsgettingworse = 0
    thesearegettingworse = []

    for county_id in range(len(data.columns)):
        county_id = str(county_id)
        fig, ax = plt.subplots(figsize=(10, 7))

        if prediction_mean_trend[county_id].iloc[-5] - prediction_mean_trend[county_id].iloc[-6] > 0:
            itsgettingworse += 1
            thesearegettingworse.append(IDNameDem[int(county_id)].get("NL Name"))
        ax.plot_date(dates[:-5],
                     target[county_id],
                     color="k",
                     label="Bestaetigte Infektionen - {}".format(IDNameDem[int(county_id)].get("NL Name")))

        ax.plot_date(
            dates,
            prediction_mean_trend[county_id],
            "-",
            color=C1,
            label="Prediction Mean Trend - {}".format(IDNameDem[int(county_id)].get("NL Name")),
            linewidth=2.0,
            zorder=4)

        ax.fill_between(
            dates,
            prediction_q25_trend[county_id],
            prediction_q75_trend[county_id],
            facecolor=C2,
            alpha=0.5,
            zorder=1)
        ax.plot_date(
            dates,
            prediction_q25_trend[county_id],
            ":",
            color=C2,
            label="Q25",
            linewidth=2.0,
            zorder=3)
        ax.plot_date(dates,  # upper line
                     prediction_q95_trend[county_id], ":",
                     label="Q05-Q95",
                     color="green", alpha=0.5, linewidth=2.0, zorder=1)
        ax.axvline(data.index[-1], label='Tag der Datenerhebung')
        ax.legend(loc="upper left")  # everything above will be included in legend
        ax.fill_between(
            dates,
            prediction_q5_trend[county_id],
            prediction_q95_trend[county_id],
            facecolor="green",
            alpha=0.25,
            zorder=0)
        ax.plot_date(dates,  # lower line
                     prediction_q5_trend[county_id],
                     ":",
                     color="green", alpha=0.5, linewidth=2.0, zorder=1)
        ax.plot_date(  # upper of q25
            dates,
            prediction_q75_trend[county_id],
            ":",
            color=C2,
            linewidth=2.0,
            zorder=3)

        ax.xaxis.set_tick_params(rotation=30, labelsize=10)
        ax.set_ylabel("Anzahl Infektionen")

        highest_value = target[county_id].max()
        ax.set_ylim(0, (int(highest_value) * 3))
        plt.savefig(currentplots + "/{}_{}_Prediction.png".format(today, IDNameDem[int(county_id)].get("NL Name")))

    print(
        "Fuer {} der {} Kreise wird ein Anstieg der Fallzahlen prognostiziert. Die betroffenen Landkreise sind: {}".format(
            itsgettingworse, len(data.columns), thesearegettingworse))

    file = open(currentplots + "Prognose_{}.txt".format(today), "w")
    file.write(
        "Fuer {} der {} Kreise wird ein Anstieg der Fallzahlen prognostiziert. Die betroffenen Landkreise sind: {}".format(
            itsgettingworse, len(data.columns), thesearegettingworse))
    file.close()
