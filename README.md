# Bayesian Autoregressive Model for the Bachelors thesis of Thomas Nortmann

The "Modell Gesundheitsamt" written by Laura Krieger was given as it can be seen in the main branch of this repositiory.
The given model was extracted, three more models and ways to analyse them were added.

## ModellGesundheitsamt.py
This is the main script running the model. It entails all options how to run the model
(retrain it, plot results, analyse results, print results of analysing, use interaction, use the autoregressive model, ...)

## ar_model_weekly_mean_interaction.py
This is the main model, a Bayesian model based on an autoregressive design taking an interacting area into account

## ar_model_interaction_fixed_params.py
This is the equivalent to the main model, the orde of training is just different (this was not used in the thesis)

## autoregressive_model.py
This is the naive implementation of the autoregressive model as shown in the thesis

## autoregressive_model_weekly_mean
This is the main model without interaction

## Baseline model
This is the baseline model used for the analysis of the models

## data_pipeline
To prepare the data for training

## feature_functions_ar.py
The functions used to extract features from the data for the original model

## feature_functions_ar.py
The functions used to extract features from the data for the autoregressive models using the weekly mean

## model_analysis.py
A collection of functions used to analyse the models

## plotting.py
The script used to plot the regressions of the original model

## plotting_ar.py
The script used to plot the regressions and transformation functions of the autoregressive models

## trend_model.py
The original model written by Laura Krieger

## additional_functions.py
A collection of additional functions, the ones used for the thesis are the plotting of the selection of dates to test
and the plotting of the interaction proportions

## akaike_weights
The calculation of the Akaike weights to determine the best distribution