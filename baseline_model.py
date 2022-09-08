import pymc3 as pm
import numpy as np



def make_model(features, data, current_county='34'):
    '''
    Makes a baseline model that uses the mean of the data as the fixed mean of the distribution
    :param features: Features for training
    :param data: Data for training (not used)
    :param current_county: The index of the county to train it for
    :return: The baseline model
    '''
    data = features[current_county][:-7]
    data_weekly_mean = features[current_county + '_weekly_mean'][6:-1]
    data_weekly_mean_squared = features[current_county + '_weekly_mean_squared'][6:-1]
    data_weekly_mean_previous = features[current_county + '_weekly_mean'][:-7]
    data = data_weekly_mean_previous - data

    target = features[current_county][7:]
    target_weekly_mean = features[current_county + '_weekly_mean'][1:]
    target_weekly_mean_squared = features[current_county + '_weekly_mean_squared'][1:]


    with pm.Model() as ar_model:

        mu = np.mean(target.to_numpy())

        alpha = pm.HalfNormal("alpha", 2.5)

        Y_obs = pm.NegativeBinomial(
            mu=mu,
            alpha=alpha,
            observed=target,
            name='Y_obs'
        )

    return ar_model


def sample_parameters(
        model,
        target,
        n_init=100,
        samples=250,
        chains=2,
        init="advi",
        target_accept=0.8,
        max_treedepth=10,
        cores=2,
        **kwargs
):
    """
        sample_parameters(target, samples=1000, cores=8, init="auto", **kwargs)

    Samples from the posterior parameter distribution, given a training dataset.
    The basis functions are designed to be causal, i.e. only data points strictly
    predating the predicted time points are used (this implies "one-step-ahead"-predictions).
    """

    # self.init_model(target)

    with model:
        # run!
        nuts = pm.step_methods.NUTS(
            # vars= params,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
        )
        trace = pm.sample(
            samples, cores=1, chains=1
        )
    return trace


def sample_predictions(
        features,
        parameters,
        prediction_days,
        target,
        average_all=False,
        only_trend=False
):
    target_days = target.index
    current_county = '34'
    target = features[current_county][7:]

    # extract coefficient samples
    mu = parameters["mu"]
    alpha = parameters["alpha"]

    num_predictions = len(target_days) + len(prediction_days)
    num_parameter_samples = alpha.shape[0]

    mean_squared_error = 0
    mu = np.mean(mu)
    alpha = np.mean(alpha)

    y = np.ones((num_predictions,), dtype=np.float64) * mu
    for i in range(len(target)):
        mean_squared_error += (y[i] - target[i])
    mean_squared_error = mean_squared_error / len(target)
    # if average_all:
    #     y = np.zeros((num_predictions,), dtype=np.float64)
    #     for i in range(num_predictions):
    #         y[i] = mu
    #         if i < len(target_days):
    #             mean_squared_error += (y[i] - target[i]) ** 2
    #     if not only_trend:
    #         mean_squared_error = mean_squared_error / num_predictions
    #         print('Mean squared error of Baseline Regression: ', mean_squared_error)
    # else:
    #     y = np.zeros((num_parameter_samples, num_predictions), dtype=np.float64)
    #     for i in range(num_predictions):
    #         y[:, i] = pm.NegativeBinomial.dist(mu=mu, alpha=alpha).random(size=num_parameter_samples)


    return {"y": y, "mu": mu, alpha: alpha, "mse": mean_squared_error}