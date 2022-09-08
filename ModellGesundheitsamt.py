import pandas as pd
import pymc3 as pm
from datetime import date
from datetime import datetime
import pickle as pkl


import data_pipeline
# import feature_functions
# import trend_model as prediction_model
# from plotting import plot
# import autorecurrent_model as prediction_model
import ar_model_weekly_mean_interaction as prediction_model
# import autorecurrent_model_weekly_mean as prediction_model
import autoregressive_model_weekly_mean as input_generator_model
import ar_model_interaction_fixed_params as fixed_prediction_model
from plotting_ar import plot, plot_function, plot_both, plot_all_functions
import feature_functions_ar as feature_functions
import model_analysis
import baseline_model

print(f"Running on PyMC3 v{pm.__version__}")

# Hyperparameters:
# Retrain the models?
training = False
# Plot the results?
plotting = False
# Use Autoregressive Models? (False = polynomial model by Laura)
ar = True
# predict for both regions in parallel for the future?
parallel_prediction = False
# Train the params for the individual regions first (and fix them)?
fixed_params = False
# Calculate Analysis of the models?
result_analysis = True

file_change = ''
if fixed_params:
    file_change ='_fixed'

# sampling multiple times to avoid one model not converging
def repeated_sampling(model, target_train):
    successful_sampling = False
    counter = 0
    while not successful_sampling:
        try:
            trace = prediction_model.sample_parameters(model, target_train, chains=2,
                                                       cores=1)  # , samples = 50) #for superficial testing
            successful_sampling = True
        except pm.exceptions.SamplingError as ex:
            successful_sampling = False
            counter += 1
            if counter > 20:
                raise Exception("Sampling went wrong repeatedly: ", str(ex))
    return trace

# get data
data = data_pipeline.preprocess_LKOS_data(filename="Uni_OS_Fallzahlen 21.02.22.xlsx")
days_into_future = 5
number_of_weeks = 6

add_info_pd = pd.read_csv("ID_to_name_demographic.csv")
additional_info = add_info_pd.to_dict("records")
demographic = add_info_pd["demographic"].to_numpy()
# nl_names = add_info_pd["NL Name"].to_numpy()

# repeat for interesting days
#today = date(2020, 4, 16) + pd.Timedelta(days=1)
for today in [date(2020,9,10), date(2020,11,1), date(2020,12,15), date(2021,5,15), date(2021,9,25), date(2022,2,1)]:
# for today in [date(2021,9,25), date(2022,2,1)]:
# for today in [date(2020,9,10)]:
    start_day = datetime.combine(today - pd.Timedelta(days=42), datetime.strptime("0000", "%H%M").time())
    data = data_pipeline.load_data_n_weeks("preprocessedLKOS.csv", start_day, pad=days_into_future)
    data[data < 0] = 0  # to get rid of the negative values


    data_train, target_train, data_test, target_test = data_pipeline.split_data(
        data,
        train_start=start_day,
        test_start=start_day + pd.Timedelta(days=number_of_weeks * 7),
        post_test=start_day + pd.Timedelta(days=number_of_weeks * 7 + days_into_future),  # *7 + 5
    )
    features_for_model = feature_functions.get_features(target_train, demographic)

    if training:
        model = prediction_model.make_model(features_for_model, ['34', '35'])
        print(model.check_test_point())
        if ar:
            if parallel_prediction:
                trace = repeated_sampling(model, target_train)
                model_interaction = prediction_model.make_model(features_for_model, ['35', '34'])
                trace_interaction = repeated_sampling(model_interaction, target_train)
                with open("trace/trace_ar_{}_interaction".format(str(today)), "wb") as f:
                    pkl.dump(trace_interaction, f)
                with open("model/model_ar_{}_interaction".format(str(today)), "wb") as f:
                    pkl.dump(model_interaction, f)
            else:
                if not fixed_params:
                    trace = repeated_sampling(model, target_train)
                model_generation = input_generator_model.make_model(features_for_model, target_train, current_county='34')
                trace_generation = repeated_sampling(model_generation, target_train)
                model_generation_neighbour = input_generator_model.make_model(features_for_model, target_train, current_county='35')
                trace_generation_neighbour = repeated_sampling(model_generation_neighbour, target_train)
                with open("trace/trace_ar_{}_generation{}".format(str(today), file_change), "wb") as f:
                    pkl.dump(trace_generation, f)
                with open("model/model_ar_{}_generation{}".format(str(today), file_change), "wb") as f:
                    pkl.dump(model_generation, f)
                with open("trace/trace_ar_{}_generation_neighbour{}".format(str(today), file_change), "wb") as f:
                    pkl.dump(trace_generation_neighbour, f)
                with open("model/model_ar_{}_generation_neighbour{}".format(str(today), file_change), "wb") as f:
                    pkl.dump(model_generation_neighbour, f)

            # sampling from the models predicting X and Y to be able to use the predictions for later sampling
            if ar and not parallel_prediction:
                pred_input = input_generator_model.sample_predictions(
                    features_for_model,
                    trace_generation,
                    target_test.index,
                    target_train,
                    average_all=True
                )
                with open("predictions/predictions_input_generation_{}{}".format(str(today), file_change), "wb") as f:
                    pkl.dump(pred_input, f)
                pred_input_neighbour = input_generator_model.sample_predictions(
                    features_for_model,
                    trace_generation_neighbour,
                    target_test.index,
                    target_train,
                    average_all=True
                )
                with open("predictions/predictions_input_generation_neighbour_{}{}".format(str(today), file_change), "wb") as f:
                    pkl.dump(pred_input_neighbour, f)

            if fixed_params:
                model = fixed_prediction_model.make_model(features_for_model, ['34', '35'], (pred_input, pred_input_neighbour))
                trace = repeated_sampling(model, target_train)


        if ar:
            with open("trace/trace_ar_{}{}".format(str(today), file_change), "wb") as f:
                pkl.dump(trace, f)
            with open("model/model_ar_{}{}".format(str(today), file_change), "wb") as f:
                pkl.dump(model, f)
        else:
            with open("trace/trace_{}".format(str(today)), "wb") as f:
                pkl.dump(trace, f)
            with open("model/model_{}".format(str(today)), "wb") as f:
                pkl.dump(model, f)

        if ar:
            if parallel_prediction:
                pred = prediction_model.sample_predictions(
                    features_for_model,
                    trace,
                    target_test.index,
                    average_all=False,
                    parameters_neighbour=trace_interaction
                )
                with open("predictions/predictions_ar_{}".format(str(today)), "wb") as f:
                    pkl.dump(pred, f)
                pred = prediction_model.sample_predictions(
                    features_for_model,
                    trace,
                    target_test.index,
                    average_all=False,
                    only_trend=True,
                    parameters_neighbour=trace_interaction
                )
                with open("predictions/predictions_ar_only_trend{}".format(str(today)), "wb") as f:
                    pkl.dump(pred, f)
            else:
                if fixed_params:
                    pred = fixed_prediction_model.sample_predictions(
                        features_for_model,
                        trace,
                        target_test.index,
                        (pred_input, pred_input_neighbour),
                        predicted_inputs=(pred_input, pred_input_neighbour),
                        average_all=False,
                        parallel_prediction=False
                    )
                    with open("predictions/predictions_ar_{}{}".format(str(today), file_change), "wb") as f:
                        pkl.dump(pred, f)
                    pred = fixed_prediction_model.sample_predictions(
                        features_for_model,
                        trace,
                        target_test.index,
                        (pred_input, pred_input_neighbour),
                        predicted_inputs=(pred_input, pred_input_neighbour),
                        average_all=False,
                        only_trend=True,
                        parallel_prediction=False
                    )
                    with open("predictions/predictions_ar_only_trend{}{}".format(str(today), file_change), "wb") as f:
                        pkl.dump(pred, f)
                else:
                    pred = prediction_model.sample_predictions(
                        features_for_model,
                        trace,
                        target_test.index,
                        predicted_inputs=(pred_input, pred_input_neighbour),
                        average_all=False,
                        parallel_prediction=False
                    )
                    with open("predictions/predictions_ar_{}".format(str(today)), "wb") as f:
                        pkl.dump(pred, f)
                    pred = prediction_model.sample_predictions(
                        features_for_model,
                        trace,
                        target_test.index,
                        predicted_inputs=(pred_input, pred_input_neighbour),
                        average_all=False,
                        only_trend=True,
                        parallel_prediction=False
                    )
                    with open("predictions/predictions_ar_only_trend{}".format(str(today)), "wb") as f:
                        pkl.dump(pred, f)
        else:
            pred = prediction_model.sample_predictions(
                features_for_model,
                trace,
                target_test.index,
                average_periodic_feature=False,
                average_all=False,
            )
            with open("predictions/predictions_{}".format(str(today)), "wb") as f:
                pkl.dump(pred, f)

        if ar:
            if parallel_prediction:
                pred_trend = prediction_model.sample_predictions(
                    features_for_model,
                    trace,
                    target_test.index,
                    average_all=True,
                    parameters_neighbour=trace_interaction
                )
                with open("predictions/predictions_trend_ar_{}".format(str(today)), "wb") as f:
                    pkl.dump(pred_trend, f)
                pred_trend = prediction_model.sample_predictions(
                    features_for_model,
                    trace,
                    target_test.index,
                    average_all=True,
                    only_trend=True,
                    parameters_neighbour=trace_interaction
                )
                with open("predictions/predictions_trend_ar__only_trend{}".format(str(today)), "wb") as f:
                    pkl.dump(pred_trend, f)
            else:
                if fixed_params:
                    pred_trend = fixed_prediction_model.sample_predictions(
                        features_for_model,
                        trace,
                        target_test.index,
                        (pred_input, pred_input_neighbour),
                        predicted_inputs=(pred_input, pred_input_neighbour),
                        average_all=True,
                        parallel_prediction=False
                    )
                    with open("predictions/predictions_trend_ar_{}{}".format(str(today), file_change), "wb") as f:
                        pkl.dump(pred_trend, f)
                    pred_trend = fixed_prediction_model.sample_predictions(
                        features_for_model,
                        trace,
                        target_test.index,
                        (pred_input, pred_input_neighbour),
                        predicted_inputs=(pred_input, pred_input_neighbour),
                        average_all=True,
                        only_trend=True,
                        parallel_prediction=False
                    )
                    with open("predictions/predictions_trend_ar__only_trend{}{}".format(str(today), file_change), "wb") as f:
                        pkl.dump(pred_trend, f)
                else:
                    pred_trend = prediction_model.sample_predictions(
                        features_for_model,
                        trace,
                        target_test.index,
                        predicted_inputs=(pred_input, pred_input_neighbour),
                        average_all=True,
                        parallel_prediction=False
                    )
                    with open("predictions/predictions_trend_ar_{}".format(str(today)), "wb") as f:
                        pkl.dump(pred_trend, f)
                    pred_trend = prediction_model.sample_predictions(
                        features_for_model,
                        trace,
                        target_test.index,
                        predicted_inputs=(pred_input, pred_input_neighbour),
                        average_all=True,
                        only_trend=True,
                        parallel_prediction=False
                    )
                    with open("predictions/predictions_trend_ar__only_trend{}".format(str(today)), "wb") as f:
                        pkl.dump(pred_trend, f)
        else:
            pred_trend = prediction_model.sample_predictions(
                target_train,
                demographic,
                trace,
                target_test.index,
                average_periodic_feature=True,
                average_all=True,
            )
            with open("predictions/predictions_trend_{}".format(str(today)), "wb") as f:
                pkl.dump(pred_trend, f)

        # with model:
        #     az.plot_trace(trace)
        #     # print(az.summary(trace, round_to=2))
        # plt.show()

    if plotting:
        plot(today, additional_info, only_trend=True, fixed_params=fixed_params)
        plot(today, additional_info, only_trend=False, fixed_params=fixed_params)

        plot_function(today)

    # model_analysis.q_q_plot(today)
    # model_analysis.chi_squared_test(today)
    if result_analysis:
        if training:
            model_baseline = baseline_model.make_model(features_for_model, target_train, current_county='34')
            trace_baseline = repeated_sampling(model_baseline, target_train)
            with open("trace/trace_ar_{}_baseline".format(str(today)), "wb") as f:
                pkl.dump(trace_baseline, f)
            with open("model/model_ar_{}_baseline".format(str(today)), "wb") as f:
                pkl.dump(model_baseline, f)
            pred_baseline = baseline_model.sample_predictions(
                features_for_model,
                trace_generation,
                target_test.index,
                target_train,
                average_all=True,
                only_trend=False
            )
            with open("predictions/predictions_baseline_{}".format(str(today)), "wb") as f:
                pkl.dump(pred_baseline, f)

        model_analysis.analysis(today, parallel_prediction, features_for_model['34'][7:], fixed_params=fixed_params, features=features_for_model)
        model_analysis.analysis_future(today, parallel_prediction,  features_for_model['34'], target_test['34'], fixed_params=fixed_params)
model_analysis.show_mse()
model_analysis.geweke_analysis()
plot_both(additional_info, fixed_params=fixed_params)
plot_all_functions(fixed_params)
