import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt


def make_model(features, counties):
    '''
    Makes an Autoregressive model with an interaction component between the given counties
    :param features: Features for training
    :param counties: Counties to use for the interaction (0 is area a, 1 is area b)
    :return: the ar model
    '''
    current_county = counties[0]
    data = features[current_county][:-7]
    data_weekly_mean = features[current_county + '_weekly_mean'][6:-1]
    data_weekly_mean_squared = features[current_county + '_weekly_mean_squared'][6:-1]
    data_weekly_mean_previous = features[current_county + '_weekly_mean'][:-7]
    data = data_weekly_mean_previous - data

    target = features[current_county][7:]
    target_weekly_mean = features[current_county + '_weekly_mean'][1:]
    target_weekly_mean_squared = features[current_county + '_weekly_mean_squared'][1:]

    interaction_county = counties[1]
    interaction_data = features[interaction_county][:-7]
    interaction_data_weekly_mean = features[interaction_county + '_weekly_mean'][6:-1]
    interaction_data_weekly_mean_squared = features[interaction_county + '_weekly_mean_squared'][6:-1]
    interaction_data_weekly_mean_previous = features[interaction_county + '_weekly_mean'][:-7]
    interaction_data = interaction_data_weekly_mean_previous - interaction_data



    with pm.Model() as ar_model:

        phi_self = pm.Uniform('phi_self', lower=0.0, upper=1.0)

        phi_0 = pm.Normal('phi_0', 1.0, 1.5)

        phi_1 = pm.Normal('phi_1', 0.0, 1.5)

        phi_2 = pm.Normal('phi_2', 0.0, 1.0)

        # phi_neighbour = pm.Continuous('phi_neighbour', 1 - phi_self)

        phi_0_neighbour = pm.Normal('phi_0_neighbour', 1.0, 1.5)

        phi_1_neighbour = pm.Normal('phi_1_neighbour', 0.0, 1.5)

        phi_2_neighbour = pm.Normal('phi_2_neighbour', 0.0, 1.0)

        # prediction_self = phi_0 * data + phi_1 * data_weekly_mean + phi_2 * data_weekly_mean_squared
        #
        # prediction_neighbour = phi_0_neighbour * interaction_data + phi_1_neighbour * interaction_data_weekly_mean + phi_2_neighbour * interaction_data_weekly_mean_squared
        # prediction_neighbour = phi_0 * interaction_data + phi_1 * interaction_data_weekly_mean + phi_2 * interaction_data_weekly_mean_squared

        prediction_self = tt.add(tt.add(tt.mul(phi_0, data), tt.mul(phi_1, data_weekly_mean)), tt.mul(phi_2, data_weekly_mean_squared))

        prediction_neighbour = tt.add(tt.add(tt.mul(phi_0_neighbour, interaction_data), tt.mul(phi_1_neighbour, interaction_data_weekly_mean)), tt.mul(phi_2_neighbour, interaction_data_weekly_mean_squared))

        prediction_self_trend = tt.add(tt.mul(phi_1, data_weekly_mean), tt.mul(phi_2, data_weekly_mean_squared))

        prediction_neighbour_trend = tt.add(tt.mul(phi_1_neighbour, interaction_data_weekly_mean), tt.mul(phi_2_neighbour, interaction_data_weekly_mean_squared))

        # mu_log = phi_self * prediction_self + ((1 - phi_self) * prediction_neighbour)
        # mu_log = prediction_self

        # mu = pm.Deterministic(
        #     "mu",
        #     phi_self * prediction_self + (1 - phi_self) * prediction_neighbour,
        # )
        mu = pm.Deterministic(
            "mu",
            tt.add(tt.mul(phi_self, prediction_self), tt.mul(tt.sub(1, phi_self), prediction_neighbour))
        )

        mu_trend = pm.Deterministic(
            "mu_trend",
            tt.add(tt.mul(phi_self, prediction_self_trend), tt.mul(tt.sub(1, phi_self), prediction_neighbour_trend))
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
            # vars=params,
            target_accept=target_accept
            # max_treedepth=max_treedepth,
        )
        trace = pm.sample(
            samples, tune=1000, cores=1, chains=1
        )
    return trace


def sample_predictions(
        features,
        parameters,
        prediction_days,
        predicted_inputs=None,
        average_all=False,
        only_trend=False,
        parallel_prediction=True,
        parameters_neighbour=None
):

    if not parallel_prediction:
        predicted_data, predicted_data_interaction = predicted_inputs

    mean_squared_error = 0

    current_county = '34'
    target = features[current_county][7:]
    data_weekly_mean = features[current_county + '_weekly_mean'][6:-1]
    data_weekly_mean_previous = features[current_county + '_weekly_mean'][:-7]
    data_weekly_mean_squared = features[current_county + '_weekly_mean_squared'][6:-1]
    data = features[current_county][:-7]
    data = data_weekly_mean_previous - data
    best_known_data = np.concatenate((np.zeros(14), features[current_county].to_numpy()), axis=None)

    interaction_county = '35'
    target_neighbour = features[interaction_county][7:]
    data_weekly_mean_neighbour = features[interaction_county + '_weekly_mean'][6:-1]
    data_weekly_mean_previous_neighbour = features[interaction_county + '_weekly_mean'][:-7]
    data_weekly_mean_squared_neighbour = features[interaction_county + '_weekly_mean_squared'][6:-1]
    data_neighbour = features[interaction_county][:-7]
    data_neighbour = data_weekly_mean_previous_neighbour - data_neighbour
    best_known_data_neighbour = np.concatenate((np.zeros(14), features[interaction_county].to_numpy()), axis=None)

    target_days = target.index

    # extract coefficient samples
    phi_0 = parameters["phi_0"]
    phi_1 = parameters["phi_1"]
    phi_2 = parameters["phi_2"]
    phi_self = parameters["phi_self"]
    # phi_neighbour = parameters["phi_neighbour"]
    phi_0_neighbour = parameters["phi_0_neighbour"]
    phi_1_neighbour = parameters["phi_1_neighbour"]
    phi_2_neighbour = parameters["phi_2_neighbour"]
    # phi_0_neighbour = parameters["phi_0"]
    # phi_1_neighbour = parameters["phi_1"]
    # phi_2_neighbour = parameters["phi_2"]
    mu = parameters['mu']
    mu = np.mean(mu, axis=0)
    mu_trend = parameters['mu_trend']
    mu_trend = np.mean(mu_trend, axis=0)

    if parallel_prediction:
        phi_0_interaction = parameters_neighbour["phi_0"]
        phi_1_interaction = parameters_neighbour["phi_1"]
        phi_2_interaction = parameters_neighbour["phi_2"]
        phi_self_interaction = parameters_neighbour["phi_self"]
        phi_0_neighbour_interaction = parameters_neighbour["phi_0_neighbour"]
        phi_1_neighbour_interaction = parameters_neighbour["phi_1_neighbour"]
        phi_2_neighbour_interaction = parameters_neighbour["phi_2_neighbour"]

    num_predictions = len(target_days) + len(prediction_days)
    num_parameter_samples = phi_0.shape[0]

    if average_all:
        phi_0 = np.mean(phi_0)
        phi_1 = np.mean(phi_1)
        phi_2 = np.mean(phi_2)
        phi_0_neighbour = np.mean(phi_0_neighbour)
        phi_1_neighbour = np.mean(phi_1_neighbour)
        phi_2_neighbour = np.mean(phi_2_neighbour)
        phi_self = np.mean(phi_self)
        y = np.zeros((num_predictions,), dtype=np.float64)
        for i in range(num_predictions):
            if i < 14:
                calculated_weekly_mean = data_weekly_mean[i]
                calculated_weekly_mean_neighbour = data_weekly_mean_neighbour[i]
            else:
                calculated_weekly_mean = np.mean(best_known_data[i+14-7:i+14])
                calculated_weekly_mean_neighbour = np.mean(best_known_data_neighbour[i+14-7:i+14])
            if only_trend:
                difference_mean = 0
                difference_mean_neighbour = 0
            else:
                if i < 14:
                    difference_mean = data[i]
                    difference_mean_neighbour = data_neighbour[i]
                else:
                    difference_mean = np.mean(best_known_data[i+14-14: i+14-7]) - best_known_data[i+14-7]
                    difference_mean_neighbour = np.mean(best_known_data_neighbour[i+14-14: i+14-7]) - best_known_data_neighbour[i+14-7]
            prediction_self = phi_0 * difference_mean + phi_1 * calculated_weekly_mean + phi_2 * (calculated_weekly_mean ** 2)
            prediction_neighbour = phi_0_neighbour * difference_mean_neighbour + phi_1_neighbour * calculated_weekly_mean_neighbour + phi_2_neighbour * (calculated_weekly_mean_neighbour ** 2)
            y[i] = phi_self * prediction_self + (1 - phi_self) * prediction_neighbour
            if i < len(mu):
                if only_trend:
                    y[i] = mu_trend[i]
                else:
                    y[i] = mu[i]
            if i+14 >= len(best_known_data):
                if parallel_prediction:
                    best_known_data = np.append(best_known_data, phi_self * prediction_self + (1 - phi_self) * prediction_neighbour)
                    prediction_self_interaction = phi_0_interaction * difference_mean_neighbour + phi_1_interaction * calculated_weekly_mean_neighbour + phi_2_interaction * calculated_weekly_mean_neighbour ** 2
                    prediction_neighbour_interaction = phi_0_neighbour_interaction * difference_mean + phi_1_neighbour_interaction * calculated_weekly_mean + phi_2_neighbour_interaction * calculated_weekly_mean ** 2
                    best_known_data_neighbour = np.append(best_known_data_neighbour, phi_self_interaction * prediction_self_interaction + (1 - phi_self_interaction) * prediction_neighbour_interaction)
                else:
                    best_known_data = np.append(best_known_data, y[i])
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
            if i < 14:
                calculated_weekly_mean = data_weekly_mean[i]
                calculated_weekly_mean_neighbour = data_weekly_mean_neighbour[i]
            else:
                calculated_weekly_mean = np.mean(best_known_data[i + 14 - 7:i + 14])
                calculated_weekly_mean_neighbour = np.mean(best_known_data_neighbour[i + 14 - 7:i + 14])
            if only_trend:
                difference_mean = 0
                difference_mean_neighbour = 0
            else:
                if i < 14:
                    difference_mean = data[i]
                    difference_mean_neighbour = data_neighbour[i]
                else:
                    difference_mean = np.mean(best_known_data[i + 14 - 14: i + 14 - 7]) - best_known_data[i + 14 - 7]
                    difference_mean_neighbour = np.mean(best_known_data_neighbour[i + 14 - 14: i + 14 - 7]) - best_known_data_neighbour[i + 14 - 7]

            # beginning normal part
            # for sample in range(num_parameter_samples):
            #     calculated_mean = phi_self[sample] * (phi_0[sample] * difference_mean + phi_1[sample] * calculated_weekly_mean + phi_2[sample] * calculated_weekly_mean ** 2)
            #     calculated_mean_neighbour = (1 - phi_self[sample]) * (phi_0_neighbour[sample] * difference_mean_neighbour + phi_1_neighbour[sample] * calculated_weekly_mean_neighbour + phi_2_neighbour[sample] * calculated_weekly_mean_neighbour ** 2)
            #     y[sample, i] = pm.NegativeBinomial.dist(mu=calculated_mean + calculated_mean_neighbour, alpha=alpha[sample]).random()
            # beginning part only for quicker testing
            phi_0 = np.mean(phi_0)
            phi_1 = np.mean(phi_1)
            phi_2 = np.mean(phi_2)
            phi_0_neighbour = np.mean(phi_0_neighbour)
            phi_1_neighbour = np.mean(phi_1_neighbour)
            phi_2_neighbour = np.mean(phi_2_neighbour)
            phi_self = np.mean(phi_self)
            alpha = np.mean(parameters["alpha"])
            calculated_mean = phi_0 * difference_mean + phi_1 * calculated_weekly_mean + phi_2 * calculated_weekly_mean ** 2
            calculated_mean_neighbour = phi_0_neighbour * difference_mean_neighbour + phi_1_neighbour * calculated_weekly_mean_neighbour + phi_2_neighbour * calculated_weekly_mean_neighbour ** 2
            combined_mean = phi_self * calculated_mean + (1 - phi_self) * calculated_mean_neighbour
            if i < len(mu):
                if only_trend:
                    combined_mean = mu_trend[i]
                else:
                    combined_mean = mu[i]
            y[:, i] = pm.NegativeBinomial.dist(mu=np.max([0, combined_mean]), alpha=alpha).random(size=num_parameter_samples)
            if i + 14 >= len(best_known_data):
                if parallel_prediction:
                    best_known_data = np.append(best_known_data,
                        phi_self * calculated_mean + (1 - phi_self) * calculated_mean_neighbour)
                    prediction_self_interaction = phi_0_interaction * difference_mean_neighbour + phi_1_interaction * calculated_weekly_mean_neighbour + phi_2_interaction * calculated_weekly_mean_neighbour ** 2
                    prediction_neighbour_interaction = phi_0_neighbour_interaction * difference_mean + phi_1_neighbour_interaction * calculated_weekly_mean + phi_2_neighbour_interaction * calculated_weekly_mean ** 2
                    best_known_data_neighbour = np.append(best_known_data_neighbour,
                        phi_self_interaction * prediction_self_interaction + (
                                    1 - phi_self_interaction) * prediction_neighbour_interaction)

            # end quicker testing

    print("phi_self: ", phi_self, "phi_neighbour: ", 1 - phi_self, "phi: ", [phi_0, phi_1, phi_2], "interaction phi: ", [phi_0_neighbour, phi_1_neighbour, phi_2_neighbour])
    return {"y": y, "phi": [phi_0, phi_1, phi_2], "phi_neighbour": [phi_0_neighbour, phi_1_neighbour, phi_2_neighbour], "phi_self": phi_self, "mse": mean_squared_error}