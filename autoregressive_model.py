import pymc3 as pm
import numpy as np
import theano.tensor as tt
import pandas as pd
import arviz as az

import feature_functions


def make_model(features, data):
    current_county = '34'
    target = data[current_county]

    order = 20
    with pm.Model() as ar_model:
        # assumes 95% of prob mass is between -2 and 2
        theta = pm.Normal("theta", 0.0, 1.0, shape=order)
        # precision of the innovation term
        tau = pm.Exponential("tau", 0.5)

        Y_obs = pm.AR("Y", rho=theta, tau=tau, constant=True, observed=target)

    return ar_model


def sample_parameters(
        model,
        target,
        n_init=100,
        samples=1000,
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
            samples, cores=1
        )
    return trace


def sample_predictions(
        features,
        parameters,
        prediction_days,
        target,
        average_all=False,
        only_trend=True
):
    current_county = '34'
    target = target[current_county]
    target_days = target.index

    # extract coefficient samples
    theta = parameters["theta"]
    tau = parameters["tau"]

    num_predictions = len(target_days) + len(prediction_days)
    num_parameter_samples = theta.shape[0]

    if average_all:
        y = np.zeros((num_predictions,), dtype=np.float64)
        mean_tau = np.mean(tau)
        mean_theta = np.mean(theta, axis=0)
        for i in range(num_predictions):
            if i > theta.shape[1] - 1:
                y[i] = mean_theta[0]
                for offset in range(1, theta.shape[1]):
                    if i - offset < target.shape[0]:
                        y[i] += target[i - offset] * mean_theta[offset]
                    else:
                        y[i] += y[i - offset] * mean_theta[offset]
    else:
        y = np.zeros((num_parameter_samples, num_predictions), dtype=np.float64)
        for sample in range(num_parameter_samples):
            if sample % 100 == 0: print(sample, "/", num_parameter_samples)
            for i in range(num_predictions):
                if i > theta.shape[1] - 1:
                    y[sample, i] = theta[sample, 0]
                    for offset in range(1, theta.shape[1]):
                        if i - offset < target.shape[0]:
                            y[sample, i] += target[i - offset] * theta[sample, offset]
                        else:
                            y[sample, i] += y[sample, i - offset] * theta[sample, offset]


    print("y", y, "theta", theta, "tau", tau)
    return {"y": y, "theta": theta, "tau": tau}