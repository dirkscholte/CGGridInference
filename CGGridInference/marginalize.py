import numpy as np


class Marginalize:
    def __init__(self, parameter_values, parameter_labels, prob_dist):
        self.parameter_values = parameter_values
        self.n_parameters = len(parameter_values.shape) - 1
        self.parameter_labels = dict(zip(parameter_labels, range(len(parameter_labels))))
        self.prob_dist = prob_dist

    def get_parameter_index(self, parameter_label):
        if np.isscalar(parameter_label) and isinstance(parameter_label, str):
            return int(self.parameter_labels.get(parameter_label))
        elif np.isscalar(parameter_label) and ~isinstance(parameter_label, str):
            return int(parameter_label)
        else:
            if len(parameter_label)==1:
                if isinstance(parameter_label[0], str):
                    return int(self.parameter_labels.get(parameter_label[0]))
                else:
                    return int(parameter_label[0])
            else:
                index = np.ones((len(parameter_label))) * np.nan
                for i in range(len(parameter_label)):
                    if isinstance(parameter_label[i], str):
                        index[i] = self.parameter_labels.get(parameter_label)
                    elif ~isinstance(parameter_label[i], str):
                        index[i] = parameter_label[i]
                return index

    def parameter_bin_mids(self, parameter_label):
        parameter_index = self.get_parameter_index(parameter_label)
        return np.unique(np.take(self.parameter_values, parameter_index, axis=-1))

    def parameter_bin_edges(self, parameter_label):
        parameter_index = self.get_parameter_index(parameter_label)
        bin_mids = self.parameter_bin_mids(parameter_index)
        bin_edges = np.append(bin_mids - 0.5 * (bin_mids[1] - bin_mids[0]),
                              bin_mids[-1] + 0.5 * (bin_mids[1] - bin_mids[0]))
        return bin_edges

    def marginalize(self, parameter_labels):
        parameter_indices = self.get_parameter_index(parameter_labels)
        return np.sum(self.prob_dist, axis=parameter_indices)

    def marginalize_derived_parameter(self, derived_parameter_values, bin_edges):
        histogram, _ = np.histogram(derived_parameter_values, weights=self.prob_dist, bins=bin_edges)
        return histogram

    def histogram_cdf_inverse(self, percentile, bin_height, bin_edges):
        bin_width = bin_edges[1:] - bin_edges[:-1]
        bin_height = bin_height / np.sum(bin_height * bin_width)
        cumulative = np.cumsum(bin_height * bin_width)
        last_bin = np.sum([cumulative <= percentile])
        if last_bin == 0:
            remainder = percentile
        else:
            remainder = percentile - cumulative[last_bin - 1]
        frac_bin = remainder / (bin_height[last_bin] * (bin_width[last_bin]))
        cdf_inverse = bin_edges[0] + np.sum(bin_width[:last_bin]) + frac_bin * bin_width[last_bin]
        return cdf_inverse

    def parameter_percentile(self, parameter_label, percentile):
        parameter_index = self.get_parameter_index(parameter_label)
        bin_edges = self.parameter_bin_edges(parameter_index)
        parameters_to_marginalize = np.delete(np.arange(self.n_parameters), parameter_index)
        print()
        marginalized = self.marginalize(parameters_to_marginalize)
        print(marginalized.shape, bin_edges.shape)
        return self.histogram_cdf_inverse(percentile, marginalized, bin_edges)

    def derived_parameter_percentile(self, derived_parameter_values, bin_edges, percentile):
        marginalized = self.marginalize_derived_parameter(derived_parameter_values, bin_edges)
        return self.histogram_cdf_inverse(percentile, marginalized, bin_edges)
