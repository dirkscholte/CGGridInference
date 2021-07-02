import numpy as np
from scipy.special import erf


class CalculatePosterior:
    def __init__(self, model_flux_values, model_line_labels):
        self.model_flux_values = model_flux_values
        self.model_line_labels = dict(zip(model_line_labels, range(len(model_line_labels))))
        self.signal_to_noise_limit = 3.0
        self.detection_flux_values = np.ma.masked_invalid(np.ones((len(self.model_line_labels)))*np.nan)
        self.detection_flux_errors = np.ma.masked_invalid(np.ones((len(self.model_line_labels)))*np.nan)
        self.uplim_flux_values = np.ma.masked_invalid(np.ones((len(self.model_line_labels)))*np.nan)
        self.uplim_flux_errors = np.ma.masked_invalid(np.ones((len(self.model_line_labels)))*np.nan)
        self.prior = np.ones(self.model_flux_values.shape[:-1]) / np.sum(np.ones(self.model_flux_values.shape[:-1]))
        self.likelihood = np.ones(self.model_flux_values.shape[:-1]) * np.nan
        self.posterior = np.ones(self.model_flux_values.shape[:-1]) * np.nan

    def reset_likelihood(self):
        self.detection_flux_values = np.ma.masked_invalid(np.ones((len(self.model_line_labels))) * np.nan)
        self.detection_flux_errors = np.ma.masked_invalid(np.ones((len(self.model_line_labels))) * np.nan)
        self.uplim_flux_values = np.ma.masked_invalid(np.ones((len(self.model_line_labels))) * np.nan)
        self.uplim_flux_errors = np.ma.masked_invalid(np.ones((len(self.model_line_labels))) * np.nan)
        self.likelihood = np.ones(self.model_flux_values.shape[-1]) * np.nan
        self.posterior = np.ones(self.model_flux_values.shape[-1]) * np.nan

    def input_data(self, data_flux_values, data_flux_errors, data_line_labels):
        for i in range(len(data_line_labels)):
            idx = self.model_line_labels.get(data_line_labels[i])
            if data_flux_values[i]/data_flux_errors[i] >= self.signal_to_noise_limit:
                self.detection_flux_values[idx] = data_flux_values[i]
                self.detection_flux_errors[idx] = data_flux_errors[i]
            else:
                self.uplim_flux_values[idx] = self.signal_to_noise_limit * data_flux_errors[i]           #F. Masci 10/25/2011 Computing flux upper-limits for non-detections
                self.uplim_flux_errors[idx] = (self.signal_to_noise_limit + 2.054) * data_flux_errors[i]


    def normalize_model(self, line_label):
        self.model_flux_values = self.model_flux_values / np.expand_dims(np.take(self.model_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)

    def normalize_data(self, line_label):
        self.uplim_flux_errors = self.uplim_flux_errors / np.expand_dims(np.take(self.detection_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)
        self.uplim_flux_values = self.uplim_flux_values / np.expand_dims(np.take(self.detection_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)
        self.detection_flux_errors = self.detection_flux_errors / np.expand_dims(np.take(self.detection_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)
        self.detection_flux_values = self.detection_flux_values / np.expand_dims(np.take(self.detection_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)

    def calculate_likelihood(self):
        def calc_lnlikelihood_detections(model, data, data_errors):
            return -0.5 * np.sum( ((data - model) / data_errors)**2, axis=-1)
        def calc_lnlikelihood_uplims(model, data, data_errors):
            return np.sum(np.log(0.5 * (erf( (data - model) / data_errors)) + 1), axis=-1)

        lnlikelihood_detections = calc_lnlikelihood_detections(self.model_flux_values,
                                                               self.detection_flux_values,
                                                               self.detection_flux_errors)
        lnlikelihood_uplims = calc_lnlikelihood_uplims(self.model_flux_values,
                                                       self.uplim_flux_values,
                                                       self.uplim_flux_errors)
        if np.sum(~self.uplim_flux_values.mask)>0:
            lnlikelihood = lnlikelihood_detections + lnlikelihood_uplims
        else:
            lnlikelihood = lnlikelihood_detections
        self.likelihood = np.exp(lnlikelihood)
        self.likelihood_detections = np.exp(lnlikelihood_detections)
        self.likelihood_uplims = np.exp(lnlikelihood_uplims)
        return self.likelihood

    def calculate_posterior(self):
        self.posterior = self.prior*self.likelihood
        return self.posterior