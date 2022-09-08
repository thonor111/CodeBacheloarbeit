import numpy as np
import arviz as az
import pickle as pkl

with open("trace/trace_ar_2022-06-22_poisson", "rb") as f:
    trace = pkl.load(f)

pm_data = az.from_pymc3(trace)
log_likelihood_poisson = pm_data.log_likelihood.Y_obs.sum().item()

aic_poisson = 2 * 3 - 2 * log_likelihood_poisson

with open("trace/trace_ar_2022-06-22_neg_binomial", "rb") as f:
    trace = pkl.load(f)

pm_data = az.from_pymc3(trace)
log_likelihood_neg_binomial = pm_data.log_likelihood.Y_obs.sum().item()

aic_neg_binomial = 2 * 4 - 2 * log_likelihood_neg_binomial

with open("trace/trace_2022-06-22", "rb") as f:
    trace = pkl.load(f)

pm_data = az.from_pymc3(trace)
log_likelihood_trend = pm_data.log_likelihood.Y.sum().item()

aic_trend = 2 * 3 - 2 * log_likelihood_trend

print(f"log-likelihoods:\nPoisson: {log_likelihood_poisson}\nNegative Binomial: {log_likelihood_neg_binomial}\nTrend: {log_likelihood_trend}")

print("Akaike Information Criterion:")
print(f"AIC of model with Poisson distribution: {aic_poisson}\nAIC of model with negative Binomial Distribution: {aic_neg_binomial}\nAIC of model with Trend model: {aic_trend}")

## Calculation of Akaike weights

min_aic = min([aic_poisson, aic_neg_binomial, aic_trend])
aic_weight_p = np.exp(-0.5 * (aic_poisson - min_aic)) / (np.exp(-0.5 * (aic_poisson - min_aic)) + np.exp(-0.5 * (aic_neg_binomial - min_aic)) + np.exp(-0.5 * (aic_trend - min_aic)))
aic_weight_nb = np.exp(-0.5 * (aic_neg_binomial - min_aic)) / (np.exp(-0.5 * (aic_poisson - min_aic)) + np.exp(-0.5 * (aic_neg_binomial - min_aic)) + np.exp(-0.5 * (aic_trend - min_aic)))
aic_weight_trend = np.exp(-0.5 * (aic_trend - min_aic)) / (np.exp(-0.5 * (aic_poisson - min_aic)) + np.exp(-0.5 * (aic_neg_binomial - min_aic)) + np.exp(-0.5 * (aic_trend - min_aic)))

print(f"AIC-weights:\nPoisson: {aic_weight_p}\nNegative Binomial: {aic_weight_nb}\nTrend Model: {aic_weight_trend}")

