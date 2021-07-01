import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

            if yi == 0:
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
