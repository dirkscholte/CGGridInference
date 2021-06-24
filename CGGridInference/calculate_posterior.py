import numpy as np
from scipy.special import erf


class CalculatePosterior:
    def __init__(self, model_flux_values, model_line_labels):
        self.model_flux_values = model_flux_values
        self.model_line_labels = dict(zip(model_line_labels, range(len(model_line_labels))))
        self.data_flux_values = np.array([])
        self.data_flux_errors = np.array([])
        self.data_line_labels = dict()
        self.prior = np.ones(self.model_flux_values.shape[-1]) / np.sum(np.ones(self.model_flux_values.shape[-1]))
        self.likelihood = np.ones(self.model_flux_values.shape[-1]) * np.nan
        self.posterior = np.ones(self.model_flux_values.shape[-1]) * np.nan
        self.signal_to_noise_limit = 3.0

    def reset_likelihood(self):
        self.data_flux_values = np.array([])
        self.data_line_labels = np.array([])
        self.likelihood = np.ones(self.model_flux_values.shape[-1]) * np.nan
        self.posterior = np.ones(self.model_flux_values.shape[-1]) * np.nan

    def reshape_dims(self, dims, dim):
        dim_array = np.ones((1, dims), int).ravel()
        dim_array[dim] = -1
        return dim_array

    def input_data(self, data_flux_values, data_flux_errors, data_line_labels):
        self.data_line_labels = dict(zip(data_line_labels, range(len(data_line_labels))))
        self.data_flux_values = data_flux_values
        self.data_flux_errors = data_flux_errors


    def normalize_model(self, line_label):
        self.model_flux_values = self.model_flux_values / np.take(self.model_flux_values, self.model_line_labels.get(line_label), axis=-1).reshape(self.reshape_dims(self.model_flux_values.ndim, range(self.model_flux_values.ndim-1)))

    def normalize_data(self, line_label):
        self.data_flux_errors = self.data_flux_errors / np.take(self.data_flux_values, self.data_line_labels.get(line_label), axis=-1).reshape(self.reshape_dims(self.data_flux_values.ndim, range(self.data_flux_values.ndim-1)))
        self.data_flux_values = self.data_flux_values / np.take(self.data_flux_values, self.data_line_labels.get(line_label), axis=-1).reshape(self.reshape_dims(self.data_flux_values.ndim, range(self.data_flux_values.ndim-1)))

    def mask_unused_lines(self):
        line_indices = np.array([self.model_line_labels.get(key) for key in list(self.data_line_labels.keys())], dtype=int)
        mask_lines = np.array([np.any(value == line_indices) for value in list(self.model_line_labels.values())], dtype=int)
        mask_lines_model = np.ones(self.model_flux_values.shape, dtype=bool) * mask_lines.reshape(self.reshape_dims(self.model_flux_values.ndim, -1))
        return mask_lines, mask_lines_model

    def mask_uplims(self):
        mask_uplims_data = (self.data_flux_values >= self.signal_to_noise_limit * self.data_flux_errors)
        mask_uplims_model = np.ones(self.model_flux_values.shape[-1], dtype=bool) * mask_uplims_data.reshape(self.reshape_dims(self.model_flux_values.ndim, -1))
        return mask_uplims_data, mask_uplims_model

    def calculate_lnlikelihood(self):
        def calc_lnlikelihood_detections(model, data, data_errors):
            return -0.5 * np.sum( ((data - model) / data_errors)**2, axis=-1)
        def calc_lnlikelihood_uplims(model, data, data_errors):
            return np.sum( -np.log(2) + np.log( erf( (data - model) / data_errors) ) )

        mask_lines, mask_lines_model = self.mask_unused_lines()
        mask_uplims_data, mask_uplims_model = self.mask_uplims()
        lnlikelihood_detections = calc_lnlikelihood_detections(self.model_flux_values[mask_lines_model][mask_uplims_model],
                                                               self.data_flux_values[mask_uplims_data].reshape(self.reshape_dims(self.model_flux_values.ndim, -1)),
                                                               self.data_flux_errors[mask_uplims_data].reshape(self.reshape_dims(self.model_flux_values.ndim, -1)) )
        lnlikelihood_uplims = calc_lnlikelihood_uplims(self.model_flux_values[mask_lines_model][~mask_uplims_model],
                                                       self.data_flux_values[~mask_uplims_data].reshape(self.reshape_dims(self.model_flux_values.ndim, -1)),
                                                       self.data_flux_errors[~mask_uplims_data].reshape(self.reshape_dims(self.model_flux_values.ndim, -1)) )
        lnlikelihood = lnlikelihood_detections + lnlikelihood_uplims
        return lnlikelihood

    def calculate_posterior(self):
        self.likelihood = np.exp(self.calculate_lnlikelihood())
        self.posterior = self.prior*self.likelihood
        return self.posterior