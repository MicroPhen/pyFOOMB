import collections
import copy
import matplotlib
from matplotlib import pyplot
import numpy
import pandas
from typing import Dict, List
from scipy.stats import norm
import seaborn
import warnings

from .caretaker import Caretaker
from .datatypes import TimeSeries
from .datatypes import Measurement

from .utils import Helpers

pyplot.style.use('ggplot')


class VisualizationHelpers():

    @staticmethod
    def random_color() -> list:
        """
        Helper method for plotting functions.
        """
        
        return list(numpy.random.choice(range(256), size=3)/256)


    @staticmethod
    def get_n_colors(n:int) -> list:
        """
        Helper method for plotting functions.
        """

        _n = int(n)
        if _n <= 10:
            colors = matplotlib.cm.get_cmap('tab10').colors
        elif _n <= 20:
            colors = matplotlib.cm.get_cmap('tab20').colors
        else:
            colors = [VisualizationHelpers.random_color() for _ in range(_n)]
        return colors


class Visualization():
    """
    Collects several methods for creating summary figures of different pyFOOMB data types.
    """

    @staticmethod
    def show_kinetic_data_many(time_series:List[List[TimeSeries]], tight_layout:bool=True, savename=None, ncols:int=3, dpi:int=150) -> dict:
        """
        Plots many time series of different TimeSeries subclasses.

        Arguments
        ---------
            time_series : List[List[TimeSeries]]
                A list of lists of TimeSeries (including subclasses thereof) objects to be plotted.

        Keyword arguments
        -----------------
            tight_layout : bool
                Calls pyplot.tight_layout() for each figure. May be not wanted for manipulating return figures yourself.
                Default is True
            savename : str or pathlib.Path object
                Where the figure will be saved. File extension '.pdf' is often a good choice.
                Default is None, which implies not saving the figure.
            ncols : int
                How many columns for subplots the figure will have. 
                Default is 3.
            dpi : int
                The resolution of the figure. 
                Default is 150.

        Returns
        -------
            figs_axes : dict
                The figure and axes objects for each replicate_id.
                Useful for additional layouting.
        
        Raises
        ------
            TypeError
                A list containing not only Measurement objects is provided.
        """

        if all([isinstance(_item, TimeSeries) for _item in time_series]):
            multi_time_series = [time_series]
        elif all([[isinstance(_item, TimeSeries) for _item in _time_series] for _time_series in time_series]):
            multi_time_series = time_series
        else:
            raise ValueError('Argument `time_series` must be of type List[TimeSeries] or List[List[TimeSeries]]')

        replicate_ids =  sorted(list(numpy.unique([[_item.replicate_id for _item in _time_series] for _time_series in multi_time_series])), key=str.lower)
        names = sorted(list(numpy.unique([[_item.name for _item in _time_series] for _time_series in multi_time_series])), key=str.lower)

        # Creation of figures and axes for plotting
        n = len(
            [
                Helpers.extract_time_series(
                    multi_time_series[0], 
                    name=_name, 
                    replicate_id=replicate_ids[0],
                ) 
                for _name in sorted(names, key=str.lower)
            ]
        )
        nrows = int(numpy.ceil(n/ncols))
        figs_axs = {
            _replicate_id : pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3), dpi=dpi)
            for _replicate_id in sorted(replicate_ids, key=str.lower)
        }

        # Auto-select alpha for repeated plottings
        if len(multi_time_series) <= 10:
            _alpha = 0.25
        elif len(multi_time_series) <= 50:
            _alpha = 0.2
        else:
            _alpha = 0.1

        # Get a joint time vector for all time series
        t_all = numpy.unique(
            [
                _t 
                for _time_series in multi_time_series
                for _t in Helpers.get_unique_timepoints(_time_series)
            ]
        ).flatten()

        for _replicate_id in replicate_ids:

            fig, ax = figs_axs[_replicate_id]

            # Collect these for calculation medians, assuming that the time vectors are all the same for each name
            _values_vectors = {_name : [] for _name in names}

            for _time_series in multi_time_series:
                _single_replicate_time_series = [
                        Helpers.extract_time_series(_time_series, name=_name, replicate_id=_replicate_id) 
                        for _name in names
                    ]

                for _single_replicate_time_series, _name, _ax in zip(_single_replicate_time_series, names, ax.flat):
                    _t = _single_replicate_time_series.timepoints
                    _values = numpy.empty_like(t_all)*numpy.nan
                    _values[numpy.in1d(t_all, _t)] = _single_replicate_time_series.values.flatten()
                    _values_vectors[_name].append(_values)
                    _ax.plot(
                        t_all, 
                        _values_vectors[_name][-1], 
                        color='red', 
                        alpha=_alpha,
                        zorder=1,
                    )

            # Calculate and plot median for name
            for _name, _ax in zip(names, ax.flat):
                _ax.plot(
                    t_all, 
                    numpy.nanmedian(_values_vectors[_name], axis=0), 
                    color='grey', 
                    zorder=2,
                    label=f'Median for {_name}'
                )
                _ax.set_title(f'Replicate ID: {_replicate_id}')
                _ax.legend()
        
            n = len(_values_vectors)
            for i, _ax in zip(range(nrows*ncols), ax.flat):
                if i >= n:
                    _ax.axis('off')
            if tight_layout:
                fig.tight_layout()

        return figs_axs


    @staticmethod
    def show_kinetic_data(time_series:List[TimeSeries], tight_layout:bool=True, savename=None, ncols:int=3, dpi:int=150) -> dict:
        """
        Plots time series of different TimeSeries subclasses.

        Arguments
        ---------
            time_series : List[TimeSeries]
                A list of TimeSeries (including subclasses thereof) objects to be plotted.

        Keyword arguments
        -----------------
            tight_layout : bool
                Calls pyplot.tight_layout() for each figure. May be not wanted for manipulating return figures yourself.
                Default is True
            savename : str or pathlib.Path object
                Where the figure will be saved. File extension '.pdf' is often a good choice.
                Default is None, which implies not saving the figure.
            ncols : int
                How many columns for subplots the figure will have. 
                Default is 3.
            dpi : int
                The resolution of the figure. 
                Default is 150.

        Returns
        -------
            figs_axes : dict
                The figure and axes objects for each replicate_id.
                Useful for additional layouting.
        
        Raises
        ------
            TypeError
                A list containing not only Measurement objects is provided.
        """

        for _item in time_series:
            if not isinstance(_item, TimeSeries):
                raise TypeError('Must provide a list of TimeSeries objects')

        replicate_ids = list(set([_time_series.replicate_id for _time_series in time_series]))
        names = list(set([_time_series.name for _time_series in time_series]))

        if len(replicate_ids) > 1:
            sorted_replicate_ids = sorted(replicate_ids, key=str.lower) 
        else:
            sorted_replicate_ids = replicate_ids

        figs_axes = {}
        for _replicate_id in sorted_replicate_ids:

            single_replicate_time_series = [
                Helpers.extract_time_series(time_series, name=_name, replicate_id=_replicate_id) 
                for _name in sorted(names, key=str.lower)
            ]

            n = len(single_replicate_time_series)
            colors = VisualizationHelpers.get_n_colors(n)
            nrows = int(numpy.ceil(n/ncols))

            fig, ax = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3), dpi=dpi)
            figs_axes[_replicate_id] = (fig, ax)

            if not isinstance(ax, numpy.ndarray):
                ax = numpy.array([ax])

            for _single_replicate_time_series, color, _ax in zip(single_replicate_time_series, colors, ax.flat):
                _single_replicate_time_series.plot(color=color, ax=_ax)

            for i, _ax in zip(range(nrows*ncols), ax.flat):
                if i >= n:
                    _ax.axis('off')
            if tight_layout:
                fig.tight_layout()
            if savename is not None:
                fig.savefig(f'{_replicate_id}_{savename}')

        return figs_axes


    @staticmethod
    def compare_estimates(parameters:dict, measurements:List[Measurement], caretaker:Caretaker, truth:List[TimeSeries]=None, tight_layout:bool=True, savename=None, ncols:int=3, dpi:int=150) -> dict:
        """
        Plots time series of Measurement object together with the trajectory of estimates parameters.

        Arguments
        ---------
            parameters : dict
                A set of parameters (model parameters, initial values, observation parameters), 
                for which the trajectories shall be shown.
            measurements : List[Measurement]
                A list of Measurement objects.
            caretaker : Caretaker
                The Caretaker object that manages the current model under investigation.

        Keyword arguments
        -----------------
            truth : List[TimeSeries]
                List with ModelState and/or Observable objects from a simulation with "true" parameters.
                Useful for model development, wehn one is working with synthetic noisy data.
                Default is None.
            tight_layout : bool
                Calls pyplot.tight_layout() for each figure. May be not wanted for manipulating return figures yourself.
                Default is True
            savename : str or pathlib.Path object 
                Where the figure will be saved. File extension '.pdf' is often a good choice.
                Default is None, which implies not saving the figure.
            ncols : int
                How many columns for subplots the figure will have. 
                Default is 3.
            dpi : int
                The resolution of the figure. 
                Default is 150.

        Returns
        -------
            figs_axes : dict
                The figure and axes objects for each replicate_id.
                Useful for additional layouting.
        
        Raises
        ------
            TypeError
                A list containing not only Measurement objects is provided.
        """

        for _item in measurements:
            if not isinstance(_item, Measurement):
                raise TypeError('Must provide a list of Measurement objects')

        if truth is not None:
            t_max = numpy.max(Helpers.get_unique_timepoints([*measurements, *truth]))
        else:
            t_max = numpy.max(Helpers.get_unique_timepoints(measurements))
        predictions = caretaker.simulate(t=t_max, parameters=parameters, verbosity=50)

        replicate_ids = list(set([_prediction.replicate_id for _prediction in predictions]))
        names = list(set([_prediction.name for _prediction in predictions]))

        if len(replicate_ids) > 1:
            sorted_replicate_ids = sorted(replicate_ids, key=str.lower) 
        else:
            sorted_replicate_ids = replicate_ids

        figs_axes = {}
        for _replicate_id in sorted_replicate_ids:

            single_replicate_predictions = [
                Helpers.extract_time_series(predictions, name=_name, replicate_id=_replicate_id) 
                for _name in sorted(names, key=str.lower)
            ]

            n = len(single_replicate_predictions)
            colors = VisualizationHelpers.get_n_colors(n)
            nrows = int(numpy.ceil(n/ncols))
            fig, ax = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3), dpi=dpi)
            figs_axes[_replicate_id] = (fig, ax)

            if not isinstance(ax, numpy.ndarray):
                ax = numpy.array([ax])

            for _single_replicate_prediction, color, _ax in zip(single_replicate_predictions, colors, ax.flat):
                _single_replicate_prediction.plot(
                    ax=_ax, 
                    marker='', 
                    linestyle='-', 
                    color=color, 
                    label=f'Estimate for {_single_replicate_prediction.name}',
                )

                _measurement = Helpers.extract_time_series(measurements, name=_single_replicate_prediction.name, replicate_id=_replicate_id)
                if _measurement is not None:
                    _measurement.plot(
                        ax=_ax, 
                        marker='o', 
                        linestyle='', 
                        color=color, 
                        label=f'Measured {_single_replicate_prediction.name}',
                    )

                if truth is not None:
                    _truth = Helpers.extract_time_series(truth, name=_single_replicate_prediction.name, replicate_id=_replicate_id)
                    _truth.plot(
                        ax=_ax, 
                        marker='', 
                        linestyle='--', 
                        color=color, 
                        label=f'Ground truth for {_single_replicate_prediction.name}',
                    )
                _ax.set_title(f'Replicate ID: {_replicate_id}')

            for i, _ax in zip(range(nrows*ncols), ax.flat):
                if i >= n:
                    _ax.axis('off')
            if tight_layout:
                fig.tight_layout()
        if savename is not None:
            fig.savefig(savename)

        return figs_axes



    @staticmethod
    def show_parameter_distributions(parameter_collections:Dict[str, numpy.ndarray], estimates:dict=None, show_corr_coeffs:bool=False, tight_layout:bool=True, savename=None, dpi:int=150) -> tuple:
        """
        Creates a corner plot that compares the empirical distributions of parameter one-by-one as 2-D scatter plots left to the diagonal. 
        Also, the individual parameter distributions are shown with a kernel density estimate on the diagonal subplots.
        right to the diagonal, the linear correlation coefficients between the parameter pairs are shown. 

        NOTE: This correlation coefficient does not cover non-linear correlations, which are visible for the plots left to the diagonal.

        Arguments
        ---------
            parameter_collections : Dict[str, numpy.ndarray]
                Key-value pairs for the parameters, with values being a numpy.ndarray 
                representing the empirical distribution of this parameter

        Keyword arguments
        -----------------
            estimates : dict
                In case a maximum-likelihood estimate or other point estimate for the parameters is available, 
                these are also plotted for comparison.
                Default is None.
            show_corr_coeffs : bool
                Decides whether to calculate the (linear) correlation coefficients between to parameter distribution 
                and print them in the corresponding places in the upper right triangle of the corner plot.
                Default is False, which will not print the correlation coefficients. 
            tight_layout : bool
                Calls pyplot.tight_layout() for each figure. May be not wanted for manipulating return figures yourself.
                Default is True
            savename : str or pathlib.Path object 
                Where the figure will be saved. File extension '.pdf' is often a good choice.
                Default is None, which implies not saving the figure.
            dpi : int
                The resolution of the figure. 
                Default is 150.
            
        Returns
        -------
            fig, ax : tuple
                The figure and axes object.
                Useful for additional layouting.
        """

        if isinstance(parameter_collections, pandas.DataFrame):
            parameter_collections = {p : parameter_collections[p].to_numpy() for p in parameter_collections.columns}

        nrows, ncols = [len(parameter_collections)]*2
        fig, ax = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4), dpi=dpi)
        for i, par_i in zip(range(nrows), parameter_collections.keys()):
            for j, par_j in zip(range(ncols), parameter_collections.keys()):

                if i == j:
                # Draw histograms for each parameter distribution
                    seaborn.distplot(parameter_collections[par_i], color='black', ax=ax[i, j])
                    ax[i, j].set_title(par_i, size=24, fontweight='bold')
                    ax[i, j].set_xlabel('Parameter value', size=16)
                    ax[i, j].set_ylabel('Frequency', size=16)
                    ax[i, j].tick_params(axis='both', which='major', labelsize=14)
                    ax[i, j].grid(True)
                    if estimates is not None:
                        ax[i, j].plot([estimates[par_i]]*2, ax[i, j].get_ylim(), color='red', linestyle='--', label='Estimate', zorder=1)
                        ax[i, j].legend(frameon=False, fontsize=12)
                
                elif i > j:
                # Draw correlation plots for pairs of parameters
                    ax[i, j].scatter(parameter_collections[par_i], parameter_collections[par_j], marker='.', color='grey')
                    if estimates is not None:
                        ax[i, j].scatter(estimates[par_i], estimates[par_j], color='red', label='Estimate')
                    ax[i, j].scatter(numpy.nanmedian(parameter_collections[par_i]), numpy.nanmedian(parameter_collections[par_j]), color='black', label='Sampling\nmedian') 
                    ax[i, j].set_xlabel(par_i, size=16)
                    ax[i, j].set_ylabel(par_j, size=16)
                    ax[i, j].legend(frameon=False, fontsize=12)
                    ax[i, j].tick_params(axis='both', which='major', labelsize=14)
                    ax[i, j].grid(True)

                elif show_corr_coeffs:
                # Calculate linear correlation coefficients for pairs of parameters
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                    corr = numpy.ma.corrcoef(numpy.ma.masked_invalid(parameter_collections[par_i]), numpy.ma.masked_invalid(parameter_collections[par_j]))
                    corr_text = f'$\\rho$ ({par_i}, {par_j})\n= {corr[0, 1]:.3f}'
                    ax[i, j].text(
                        x=0.5, y=0.5, s=f'{corr_text}', 
                        fontsize=20, horizontalalignment='center', verticalalignment='center',
                    )

                else:
                # No linear correlation coefficients wanted
                    ax[i, j].set_visible(False)
        if tight_layout:
            fig.tight_layout()
        if savename is not None:
            fig.savefig(savename)

        return (fig, ax)


    @staticmethod
    def compare_estimates_many(parameter_collections:Dict[str, numpy.ndarray], 
                               measurements:List[Measurement], 
                               caretaker:Caretaker, 
                               show_measurements_only:bool=False, 
                               truth:List[TimeSeries]=None, 
                               tight_layout:bool=True, 
                               savename=None, 
                               ncols:int=3, 
                               dpi:int=150,
                               ) -> dict:
        """
        Plots time courses of model predictions for parameter estimation sets from Monte-Carlo simulations,
        togehter with corresponding medians and measurements.

        Arguments
        ---------
            parameter_collections : Dict[str, numpy.ndarray]
                Key-value pairs for the parameters, with values being a numpy.ndarray 
                representing the empirical distribution of this parameter.
            measurements : List[Measurement]
                A list of Measurement objects.
            caretaker : Caretaker
                The Caretaker object that manages the current model under investigation.

        Keyword arguments
        -----------------
            show_measurements_only : bool
                Shows only model predictions for which measurements are available.
                Default is False.
            truth : List[TimeSeries]
                List with ModelState and/or Observable objects from a simulation with "true" parameters.
                Useful for model development, wehn one is working with synthetic noisy data.
                Default is None.
            tight_layout : bool
                Calls pyplot.tight_layout() for each figure. May be not wanted for manipulating return figures yourself.
                Default is True
            savename : str or pathlib.Path object 
                Where the figure will be saved. File extension '.pdf' is often a good choice.
                Default is None, which implies not saving the figure.
            ncols : int
                How many columns for subplots the figure will have. 
                Default is 3.
            dpi : int
                The resolution of the figure. 
                Default is 150.

        Returns
        -------
            figs_axes : dict
                The figure and axes objects for each replicate_id.
                Useful for additional layouting.
        """

        if truth is not None:
            t = Helpers.get_unique_timepoints([*measurements, *truth])
        else:
            t = Helpers.get_unique_timepoints(measurements)
        if len(t) < 250:
            t = numpy.linspace(0, max(t), 250)

        parameters_splits = Helpers.split_parameters_distributions(parameter_collections)
        multi_predicitions = [caretaker.simulate(t=t, parameters=_parameters) for _parameters in parameters_splits]

        replicate_ids = sorted(list(set([_measurement.replicate_id for _measurement in measurements])), key=str.lower)
        measurement_names = sorted(list(set([_measurement.name for _measurement in measurements])), key=str.lower)
        names = sorted(list(set([_prediction.name for _prediction in multi_predicitions[0]])), key=str.lower)

        if show_measurements_only:
            multi_predicitions = [
                [
                    _prediction 
                    for _prediction in _predictions 
                    if _prediction.name in measurement_names
                ] 
                for _predictions in multi_predicitions
            ]

            if truth is not None:
                truth = [
                    _truth 
                    for _truth in truth 
                    if _truth.name in measurement_names
                ]

            names = measurement_names

        # Plot all forward simulations using another method
        figs_axes = Visualization.show_kinetic_data_many(time_series=multi_predicitions, ncols=ncols, dpi=dpi)

        # Plot additionally measurements and ground truth (if provided)
        for _replicate_id in replicate_ids:
            fig, ax = figs_axes[_replicate_id]

            for _name, _ax in zip(names, ax.flat):
                _single_replicate_measurement = Helpers.extract_time_series(
                    measurements, 
                    replicate_id=_replicate_id, 
                    name=_name,
                )
                if _single_replicate_measurement is not None:
                    _ax.errorbar(
                        _single_replicate_measurement.timepoints, 
                        _single_replicate_measurement.values, 
                        yerr=_single_replicate_measurement.errors, 
                        color='black', 
                        marker='.',
                        linestyle='',
                        zorder=3,
                        label=f'Measurement for {_single_replicate_measurement.name}',
                    )

                if truth is not None:
                    _truth = Helpers.extract_time_series(truth, name=_name, replicate_id=_replicate_id)
                    if _truth is not None:
                        _ax.plot(
                            _truth.timepoints,
                            _truth.values,
                            color='grey',
                            linestyle='--',
                            zorder=2,
                            label=f'Ground truth for {_truth.name}',
                        )
        
                _ax.legend()
            if tight_layout:
                fig.tight_layout()
        return figs_axes
