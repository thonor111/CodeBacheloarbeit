import pymc3 as pm
import numpy as np
import theano.tensor as tt
import pandas as pd

import feature_functions


def make_model(features, data):
    target = features[0]

    T_S = features[1]
    T_T = features[2]
    # T_D = features[3]
    exposure = features[3]  # was 4
    days, counties = data.index, data.columns

    log_exposure = np.log(exposure).astype(np.float64).ravel()
    num_obs = np.prod(target.shape)
    num_t_s = T_S.shape[1]
    num_t_t = T_T.shape[1]
    # num_t_d = T_D.shape[1]
    num_counties = len(counties)
    with pm.Model() as model:
        # priors
        # δ = 1/√α
        delta = pm.HalfCauchy("delta", 10, testval=1.0)
        alpha = pm.Deterministic("alpha", np.float64(1.0) / delta)

        # prior parameter distribution over the parameters for the periodic part of the distribution
        W_t_s = pm.Normal(
            "W_t_s", mu=0, sd=10, testval=np.zeros(num_t_s), shape=num_t_s
        )
        # prior parameter distribution over the parameters for the trend part of the distribution
        W_t_t = pm.Normal(
            "W_t_t",
            mu=0,
            sd=10,
            testval=np.zeros((num_counties, num_t_t)),
            shape=(num_counties, num_t_t),
        )


        expanded_Wtt = tt.tile(
            W_t_t.reshape(shape=(1, num_counties, -1)), reps=(42, 1, 1)

        )
        expanded_TT = np.reshape(T_T, newshape=(42, num_counties, -1))

        result_TT = tt.flatten(tt.sum(expanded_TT * expanded_Wtt, axis=-1))

        # mean of the sum of both parts of the distribution (periodic and trend) with applied priors and the logarithm of the exposure (idk what that is though)
        # calculate mean rates
        mu = pm.Deterministic(
            "mu",
            tt.exp(
                tt.dot(T_S, W_t_s)
                + result_TT
                # + tt.dot(T_D, W_t_d)
                # + tt.dot(TS, W_ts)
                + log_exposure
            ),
        )
        # constrain to observations
        pm.NegativeBinomial("Y", mu=mu, alpha=alpha, observed=target)
    return model


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
            samples,
            nuts,
            chains=chains,
            cores=cores,
            compute_convergence_checks=False,
            **kwargs
        )
    return trace


def sample_predictions(
        target_days_counties,
        demographics,
        parameters,
        prediction_days,
        average_periodic_feature=False,
        average_all=False,
        only_trend = True,
        init="auto",
):
    PPO = 2
    TPO = 2  # was 2
    target_days = target_days_counties.index

    target_counties = target_days_counties.columns
    num_counties = len(target_counties)

    all_days = target_days.append(prediction_days)

    all_days_counties = pd.DataFrame(index=all_days, columns=target_counties)

    # extract features
    features_ = feature_functions.get_features(all_days_counties, demographics)
    target = features_[0]
    T_S = features_[1]
    T_T = features_[2]
    # T_D = features_[3]
    exposure = features_[3]

    log_exposure = np.log(exposure).astype(np.float64).ravel()

    if average_periodic_feature:
        T_S = np.reshape(T_S, newshape=(-1, num_counties, 5))
        mean = np.mean(T_S, axis=0, keepdims=True)
        T_S = np.reshape(np.tile(mean, reps=(T_S.shape[0], 1, 1)), (-1, 5))

    if average_all:
        T_S = np.reshape(T_S, newshape=(47, num_counties, -1))
        mean = np.mean(T_S, axis=0, keepdims=True)
        T_S = np.reshape(np.tile(mean, reps=(47, 1, 1)), (-1, PPO + 1))  # periodic feature!!!

        # T_D = np.reshape(T_D, newshape=(47, num_counties, -1))
        # mean = np.mean(T_D, axis=0, keepdims=True)
        # T_D = np.reshape(np.tile(mean, reps=(47, 1)), newshape=(-1, 1))

        log_exposure = np.reshape(log_exposure, newshape=(47, num_counties))
        mean = np.mean(log_exposure, axis=0, keepdims=True)
        log_exposure = np.reshape(np.tile(mean, reps=(47, 1, 1)), (-1))

    # extract coefficient samples
    alpha = parameters["alpha"]
    W_t_s = parameters["W_t_s"]
    W_t_t = parameters["W_t_t"]
    # W_t_d = parameters["W_t_d"]
    # W_ts = parameters["W_ts"]
    # print("This is W-t-t", W_t_t)
    num_predictions = len(target_days) * len(target_counties) + len(
        prediction_days
    ) * len(target_counties)
    num_parameter_samples = alpha.size

    y = np.zeros((num_parameter_samples, num_predictions), dtype=np.float64)
    mu = np.zeros((num_parameter_samples, num_predictions), dtype=np.float64)

    expanded_Wtt = np.tile(
        np.reshape(W_t_t, newshape=(-1, 1, num_counties, TPO + 1)), reps=(1, 47, 1, 1)
    )

    expanded_TT = np.reshape(T_T, newshape=(1, 47, num_counties, TPO + 1))  # TT=1

    result_TT = np.reshape(
        np.sum(expanded_TT * expanded_Wtt, axis=-1), newshape=(-1, 47 * num_counties)
    )

    for i in range(num_parameter_samples):
        if i % 100 == 0: print(i, "/", num_parameter_samples)
        mu[i, :] = np.exp(
            np.dot(T_S, W_t_s[i])
            + result_TT[i]
            # + np.dot(T_D, W_t_d[i])
            + log_exposure
        )

        y[i, :] = pm.NegativeBinomial.dist(mu=mu[i, :], alpha=alpha[i]).random()
    print("y", y, "mu", mu, "alpha", alpha)
    return {"y": y, "mu": mu, "alpha": alpha}