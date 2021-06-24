import numpy as np

class CGGridInference:
    def __init__(self, parameter_values, model_luminosity_values, model_line_labels):
        self.parameter_values = parameter_values
        self.model_luminosity_values = model_luminosity_values
        self.model_line_labels = model_line_labels
        self.data_luminosity_values = np.array([])
        self.data_line_labels = np.array([])
        self.prior = np.ones(self.model_luminosity_values.shape[-1])/np.sum(np.ones(self.model_luminosity_values.shape[-1]))
        self.likelihood
        self.posterior

    def normalize_model(self, line_label):
        return normalized_model

    def normalize_data(self, line_label):
        return normalized_data

    def mask_ul(self, signal_to_noise_limit):
        return data_mask, model_mask



    def calculate_likelihood(self):
        def calculate_likelihood_detections():
            return likelihood_detections
        def calculate_likelihood_ul():
            return likelihood_ul
        self.likelihood = calculate_likelihood_detections()*calculate_likelihood_ul()
        return self.likelihood

    def calculate_posterior(self):
        self.posterior = self.prior*self.likelihood
        return self.posterior

    def calculate_goodness_of_fit(self):
        return goodness_of_fit


