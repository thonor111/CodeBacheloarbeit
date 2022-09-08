import numpy as np
import math


def datetimeadaptions(date):  # I don't like myself for doing this
    year = str(date)[:4]
    month = str(date)[5:7]
    day = str(date)[8:10]
    return year + "-" + month + "-" + day


def get_features(target, demographics=None, poly_order=2):
    """
    Calculates the weakly means and the squared weekly means
    :param target: targets to calculate the features from
    :param demographics: not used
    :param poly_order: not used
    :return: the features for training
    """
    days, counties = target.index, target.columns
    targets_weekly_mean = target.copy()
    for county in counties:
        current_data = np.zeros(7)
        for day in days:
            if math.isnan(targets_weekly_mean.at[day, county]):
                targets_weekly_mean.at[day, county] = 0
            if math.isnan(target.at[day, county]):
                target.at[day, county] = 0
            current_data = np.append(current_data, target.at[day, county])
            targets_weekly_mean.at[day, county] = np.mean(current_data[-7:])
    targets_weekly_mean_squared = targets_weekly_mean * targets_weekly_mean
    targets_weekly_mean.columns = [str(column) + '_weekly_mean' for column in targets_weekly_mean.columns]
    targets_weekly_mean_squared.columns = [str(column) + '_weekly_mean_squared' for column in targets_weekly_mean_squared.columns]
    features = target.join([targets_weekly_mean, targets_weekly_mean_squared])
    return features
