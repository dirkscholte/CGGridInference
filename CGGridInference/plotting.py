import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from calculate_posterior import CalculatePosterior
from marginalize import Marginalize

class CornerPlot:
    def __init__(self,  parameter_values, parameter_labels, prob_dist, save=False, savename='cornerplot.pdf'):
        self.parameter_values = parameter_values
        self.n_parameters = len(parameter_values.shape) - 1
        self.parameter_labels = parameter_labels
        self.axis_labels = parameter_labels
        self.prob_dist = prob_dist
        self.save = save
        self.savename = savename

    def forceAspect(self, ax, aspect=1):
        im = ax.get_images()
        extent = im[0].get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

    def plot(self):
        marg = Marginalize(self.parameter_values, self.parameter_labels, self.prob_dist)
        plt.figure(figsize=(6, 6))
        gs1 = gridspec.GridSpec(self.n_parameters, self.n_parameters)
        gs1.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.

        for i in range(self.n_parameters ** 2):
            yi = int(i % self.n_parameters)
            xi = int(i / self.n_parameters)
            if xi == yi:
                ax1 = plt.subplot(gs1[i])
                plt.axis('on')
                # ax1.set_xticklabels([])
                # ax1.set_yticklabels([])
                ax1.tick_params(axis="x", direction="in")
                ax1.tick_params(axis="y", direction="in")
                ax1.set_xlim([marg.parameter_bin_edges(xi)[0], marg.parameter_bin_edges(xi)[-1]])
                if xi == 0:
                    ax1.axvline(0.0, linestyle='--', c='k')
                if xi == 3:
                    ax1.axvline(1.0, linestyle='--', c='k')

                parameters_to_marginalize = tuple(np.delete(np.arange(self.n_parameters), xi))
                ax1.step(marg.parameter_bin_edges(xi),
                         np.append(marg.marginalize(parameters_to_marginalize), marg.marginalize(parameters_to_marginalize)[-1]),
                         where='post')
            elif xi > yi:
                ax1 = plt.subplot(gs1[i])
                plt.axis('on')
                # ax1.set_xticklabels([])
                # ax1.set_yticklabels([])
                ax1.set_aspect(1)
                ax1.tick_params(axis="x", direction="in")
                ax1.tick_params(axis="y", direction="in")

                pars_to_marginalize = tuple(np.delete(np.arange(self.n_parameters), [xi, yi]))
                ax1.imshow(np.transpose(marg.marginalize(pars_to_marginalize)), origin='lower', cmap='Blues',
                           extent=[marg.parameter_bin_edges(yi)[0], marg.parameter_bin_edges(yi)[-1], marg.parameter_bin_edges(xi)[0],
                                   marg.parameter_bin_edges(xi)[-1]])
                self.forceAspect(ax1, aspect=1)

            else:
                ax1 = plt.subplot(gs1[i])
                plt.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.set_aspect(1)
                ax1.tick_params(axis="x", direction="in")
                ax1.tick_params(axis="y", direction="in")

            if yi == 0 and xi != 0:
                ax1 = plt.subplot(gs1[i])
                ax1.set_ylabel(self.axis_labels[xi])
            else:
                ax1.set_yticklabels([])

            if xi == self.n_parameters - 1:
                ax1 = plt.subplot(gs1[i])
                ax1.set_xlabel(self.axis_labels[yi])
            else:
                ax1.set_xticklabels([])
        if self.save:
            plt.savefig(self.savename)
        else:
            plt.show()

class LineFitPlot:
    def __init__(self, parameter_values, parameter_labels, model_flux_values, model_line_labels, data_flux_values, data_flux_errors, data_line_labels, show_line_labels, normalize_label, parameter_colorbar=0, save=False, savename='cornerplot.pdf'):
        self.parameter_values = parameter_values
        self.n_parameters = len(parameter_values.shape) - 1
        self.parameter_labels = parameter_labels
        self.model_flux_values = model_flux_values
        self.model_line_labels = dict(zip(model_line_labels, range(len(model_line_labels))))
        self.data_flux_values = data_flux_values
        self.data_flux_errors = data_flux_errors
        self.data_line_labels = data_line_labels
        self.show_line_labels = show_line_labels
        self.normalize_label = normalize_label
        self.parameter_colorbar = parameter_colorbar
        self.save = save
        self.savename = savename

        self.calc_post = CalculatePosterior(self.model_flux_values, model_line_labels)
        self.calc_post.normalize_model(self.normalize_label)
        self.calc_post.input_data(self.data_flux_values, self.data_flux_errors, self.data_line_labels)
        self.calc_post.normalize_data(self.normalize_label)
        self.calc_post.calculate_likelihood()
        self.calc_post.calculate_posterior()
        self.posterior = self.calc_post.posterior

        self.marg = Marginalize(self.parameter_values, self.parameter_labels, self.posterior)

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(len(self.show_line_labels)):
            model_flux = np.take(self.calc_post.model_flux_values, self.model_line_labels.get(self.show_line_labels[i]), axis=-1).reshape(-1)
            colorbar_values = np.take(self.marg.parameter_values, self.marg.get_parameter_index(self.parameter_colorbar), axis=-1).reshape(-1)
            pos = ax.scatter(np.random.uniform(low=0.1, high=0.9, size=model_flux.shape) + i,
                             model_flux,
                             alpha=0.4,
                             s=1000 * self.posterior.reshape(-1) / np.max(self.posterior.reshape(-1)) + 0.3,
                             c=colorbar_values,
                             edgecolor='none',
                             vmin=np.min(colorbar_values),
                             vmax=np.max(colorbar_values),
                             cmap='turbo')
            if i == 0:
                ymin = np.min(model_flux)
                ymax = np.min(model_flux)
            if ymin > np.min(model_flux):
                ymin = np.min(model_flux)
            if ymax < np.max(model_flux):
                ymax = np.max(model_flux)

        for i in range(len(self.show_line_labels)):
            idx = self.model_line_labels.get(self.show_line_labels[i])
            plt.plot([0.1 + i, 0.9 + i],
                     [self.calc_post.detection_flux_values[idx], self.calc_post.detection_flux_values[idx]], c='k')
            plt.plot([0.1 + i, 0.1 + i], [self.calc_post.detection_flux_values[idx] - self.calc_post.detection_flux_errors[idx],
                                          self.calc_post.detection_flux_values[idx] + self.calc_post.detection_flux_errors[idx]], c='k')
            plt.plot([0.9 + i, 0.9 + i], [self.calc_post.detection_flux_values[idx] - self.calc_post.detection_flux_errors[idx],
                                          self.calc_post.detection_flux_values[idx] + self.calc_post.detection_flux_errors[idx]], c='k')

        for i in range(len(self.show_line_labels)):
            idx = self.model_line_labels.get(self.show_line_labels[i])
            plt.plot([0.1 + i, 0.9 + i],
                     [self.calc_post.uplim_flux_values[idx], self.calc_post.uplim_flux_values[idx]], c='grey')
            plt.plot([0.1 + i, 0.1 + i], [-999.,
                                          self.calc_post.uplim_flux_values[idx]], c='grey')
            plt.plot([0.9 + i, 0.9 + i], [-999.,
                                          self.calc_post.uplim_flux_values[idx]], c='grey')

        plt.xlim(0., len(self.show_line_labels))
        plt.ylim(ymin, ymax)
        plt.yscale('log')

        plt.ylabel('Modelled flux and measured flux')
        plt.xticks(np.arange(len(self.show_line_labels)) + 0.5, labels=self.show_line_labels, fontsize=8, rotation='vertical')
        fig.colorbar(pos, label=self.parameter_colorbar)

        if self.save:
            plt.savefig(self.savename)
        else:
            plt.show()

