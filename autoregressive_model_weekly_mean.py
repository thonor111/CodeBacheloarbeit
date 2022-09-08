import pymc3 as pm
import numpy as np



def make_model(features, data, current_county='34'):
    data = features[current_county][:-7]
    data_weekly_mean = features[current_county + '_weekly_mean'][6:-1]
    data_weekly_mean_squared = features[current_county + '_weekly_mean_squared'][6:-1]
    data_weekly_mean_previous = features[current_county + '_weekly_mean'][:-7]
    data = data_weekly_mean_previous - data

    target = features[current_county][7:]
    target_weekly_mean = features[current_county + '_weekly_mean'][1:]
    target_weekly_mean_squared = features[current_county + '_weekly_mean_squared'][1:]


    with pm.Model() as ar_model:

        phi_0 = pm.Normal('phi_0', 0.0, 1.0)

        phi_1 = pm.Normal('phi_1', 0.0, 1.0)

        phi_2 = pm.Normal('phi_2', 0.0, 1.0)

        mu_log = phi_0 * data + phi_1 * data_weekly_mean + phi_2 * data_weekly_mean_squared
        # mu_log = phi_0 * data

        mu = pm.Deterministic(
            "mu",
            mu_log,
        )

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
        samples=2000,
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
            samples, tune=1000, cores=1, chains=chains
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
    current_county = '34'
    target = features[current_county][7:]
    data_weekly_mean = features[current_county + '_weekly_mean'][6:-1]
    data_weekly_mean_previous = features[current_county + '_weekly_mean'][:-7]
    data_weekly_mean_squared = features[current_county + '_weekly_mean_squared'][6:-1]
    data = features[current_county][:-7]
    data = data_weekly_mean_previous - data

    target_days = target.index

    # extract coefficient samples
    phi_0 = parameters["phi_0"]
    phi_1 = parameters["phi_1"]
    phi_2 = parameters["phi_2"]
    mu = parameters["mu"]
    alpha = parameters["alpha"]

    num_predictions = len(target_days) + len(prediction_days)
    num_parameter_samples = phi_0.shape[0]

    mean_squared_error = 0

    if average_all:
        phi_0 = np.mean(phi_0)
        phi_1 = np.mean(phi_1)
        phi_2 = np.mean(phi_2)
        mu = np.mean(mu, axis=0)
        y = np.zeros((num_predictions,), dtype=np.float64)
        for i in range(num_predictions):
            if i < len(target_days):
                calculated_weekly_mean = data_weekly_mean[i]
            else:
                calculated_weekly_mean = np.mean(y[i - 7:i])
            if only_trend:
                difference_mean = 0
            else:
                if i < len(target_days):
                    difference_mean = data[i]
                else:
                    difference_mean = np.mean(y[i - 14: i - 7]) - y[i-7]
            y[i] = phi_0 * difference_mean + phi_1 * calculated_weekly_mean + phi_2 * calculated_weekly_mean ** 2
            if not only_trend:
                if i < len(target_days):
                    mean_squared_error += (y[i] - target[i]) ** 2
        if not only_trend:
            mean_squared_error = mean_squared_error / num_predictions
            print('Mean squared error of Regression: ', mean_squared_error)
    else:
        y = np.zeros((num_parameter_samples, num_predictions), dtype=np.float64)
        for i in range(num_predictions):
            print(i + 1, "/", num_predictions)
            if i < len(target_days):
                calculated_weekly_mean = data_weekly_mean[i]
            else:
                calculated_weekly_mean = np.mean(y[:, i - 7:i])
            if only_trend:
                difference_mean = 0
            else:
                if i < len(target_days):
                    difference_mean = data[i]
                else:
                    difference_mean = np.mean(y[:, i - 14: i - 7]) - np.mean(y[:, i-7])

            # beginning normal part
            # for sample in range(num_parameter_samples):
            #     calculated_mean = phi_0[sample] * difference_mean + phi_1[sample] * calculated_weekly_mean + phi_2[sample] * calculated_weekly_mean ** 2
            #     y[sample, i] = pm.NegativeBinomial.dist(mu=calculated_mean, alpha=alpha[sample]).random()
            # beginning part only for quicker testing
            phi_0 = np.mean(phi_0)
            phi_1 = np.mean(phi_1)
            phi_2 = np.mean(phi_2)
            alpha = np.mean(alpha)
            calculated_mean = phi_0 * difference_mean + phi_1 * calculated_weekly_mean + phi_2 * calculated_weekly_mean ** 2
            y[:, i] = pm.NegativeBinomial.dist(mu=calculated_mean, alpha=alpha).random(size=num_parameter_samples)
            # end quicker testing


    print("phi", [phi_0, phi_1, phi_2])
    return {"y": y, "phi": [phi_0, phi_1, phi_2], "mse": mean_squared_error}