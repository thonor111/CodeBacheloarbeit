import numpy as np
import pandas as pd
import scipy as sp
from datetime import datetime, timedelta, date


# def spatio_temporal_feature(times, locations):
#     _times = [datetime.strptime(d, "%Y-%m-%d") for d in times]
#     return np.asarray(_times).reshape((-1, 1)), np.asarray(locations).reshape((1, -1)).astype(np.float32)


def temporal_polynomial_feature(t0, t, tmax, order):
    # print("Aus report temporal polynomial feauture", t, t0, tmax, order)
    t = datetime.strptime(t, "%Y-%m-%d")
    t0 = datetime.strptime(t0, "%Y-%m-%d")
    tmax = datetime.strptime(tmax, "%Y-%m-%d")
    scale = (tmax - t0).days

    t_delta = (t - t0).days / scale
    # print(scale)
    # print(t_delta)
    # print(t_delta ** order)
    return t_delta ** order


# TemporalFourierFeature(SpatioTemporalFeature)

def temporal_periodic_polynomial_feature(t0, t, period, order):
    t = datetime.strptime(t, "%Y-%m-%d")
    t0 = datetime.strptime(t0, "%Y-%m-%d")
    tdelta = (t - t0).days % period

    return (tdelta / period) ** order


# def temporal_sigmoid_feature(t0, t, scale):
#     # what does scale do here?
#     t = datetime.strptime(t, "%Y-%m-%d")
#     t0 = datetime.strptime(t0, "%Y-%m-%d")
#     t_delta = (t - t0) / scale
#     return sp.special.expit(t_delta.days + (t_delta.seconds / (3600 * 24)))


# def report_delay_polynomial_feature(t0, t, t_max, order):
#     #print("Aus report delay polynomial feauture",t, t0, t_max, order)
#     t = datetime.strptime(t, "%Y-%m-%d")
#     t0 = datetime.strptime(t0, "%Y-%m-%d")
#     t_max = datetime.strptime(t_max, "%Y-%m-%d")
#     scale = (t_max - t0).days
#     _t = 0 if t <= t0 else (t - t0).days / scale
#     return _t ** order


def features(trange, order, demographic, periodic_poly_order=2, trend_poly_order=2, include_temporal=True,
             # trend_poly_order was 2
             include_periodic=True,
             include_demographics=True, include_report_delay=False):
    report_delay_order = order
    feature_collection = {
        "temporal_trend": {
            "temporal_polynomial_{}".format(i): temporal_polynomial_feature(
                trange[0], trange[1], trange[2], i
            )
            for i in range(trend_poly_order + 1)
        }
        if include_temporal
        else {},
        "temporal_seasonal": {
            "temporal_periodic_polynomial_{}".format(
                i
            ): temporal_periodic_polynomial_feature(trange[0], trange[1], 7, i)  # why 7
            for i in range(periodic_poly_order + 1)
        }
        if include_periodic
        else {},

        # "temporal_report_delay": {
        #     "report_delay": report_delay_polynomial_feature(
        #         trange[0], trange[1], trange[2], report_delay_order  #
        #     )
        # }
        # if include_report_delay
        # else {},
        "exposure": {
            "exposure": demographic * 1.0 / 100000
        }
    }

    return feature_collection

def datetimeadaptions(date):  # I don't like myself for doing this
    year = str(date)[:4]
    month = str(date)[5:7]
    day = str(date)[8:10]
    return year + "-" + month + "-" + day


def evaluate_features(days, counties, demographic, polynom_order):
    all_features = pd.DataFrame()

    first_day_train = datetimeadaptions(days[0])
    last_day_train = datetimeadaptions(days[-1])

    for day in days:
        trange = [first_day_train, datetimeadaptions(day), last_day_train]  # last days train oder last_day_forecast?

        for i, county in enumerate(counties):
            feature = features(trange, polynom_order, demographic[i],
                               include_temporal=True, include_periodic=True, include_demographics=True)
            # include_report_delay=True, )

            feature['date'] = datetimeadaptions(day)
            feature['ID'] = county
            feature_df = pd.DataFrame.from_dict(feature)
            all_features = all_features.append(feature_df)

    return all_features


def get_features(target, demographics, poly_order=3):
    days, counties = target.index, target.columns
    # extract features

    all_features = evaluate_features(days, counties, demographics, polynom_order=poly_order)

    all_features.astype(float, errors='ignore')

    Y_obs = target.stack().values.astype(np.float32)

    len_targets = len(days) * len(counties)

    T_S = all_features.filter(regex="temporal_periodic_polynomial_\d", axis=0).dropna(
        axis=1)  # .values.astype(np.float32) #features["temporal_seasonal"].values.astype(np.float32)
    T_S = T_S.sort_values(["date", "ID"])
    T_S = T_S['temporal_seasonal'].to_numpy()
    T_S = T_S.reshape(len_targets, -1)

    T_T = all_features.filter(regex="temporal_polynomial_\d", axis=0).dropna(
        axis=1)  # features["temporal_trend"].values.astype(np.float32)
    T_T = T_T.sort_values(["date", "ID"])
    T_T = T_T["temporal_trend"].to_numpy()
    T_T = T_T.reshape(len_targets, -1)

    # T_D = all_features.filter(regex="report_delay", axis=0).dropna(
    #     axis=1)  # features["temporal_report_delay"].values.astype(np.float32)
    # T_D = T_D.sort_values(["date", "ID"])
    # T_D = T_D["temporal_report_delay"].to_numpy()
    # T_D = T_D.reshape(len_targets, -1)

    exposure = all_features.filter(regex="exposure", axis=0).dropna(
        axis=1)  # features["spatiotemporal"].values.astype(np.float32)
    exposure = exposure.sort_values(["date", "ID"])
    exposure = exposure["exposure"].to_numpy()
    exposure = exposure.reshape(len_targets, -1)

    # has to be sorted I guess? order matches the one of Y_obs =)
    # return [Y_obs, T_S, T_T, T_D, exposure]
    # Y_obs are the actual data (target), so this should be the sum of the modellings by T_S and T_T
    # T_S are temporal_seasonal features (to model the periodic part lasting 7 days)
    # T_T are temporal_trend features (to model the underlying trend with a polynomial function)
    return [Y_obs, T_S, T_T, exposure]