import numpy as np


class MarginalizePosterior:
    def __init__(self, model_parameter_values, posterior):
        self.model_parameter_values = model_parameter_values
        self.posterior = posterior