import abc
import collections
from matplotlib import pyplot
import numpy
import scipy
from scipy.stats import norm
from typing import Callable, Dict
import warnings

from .constants import Constants

SINGLE_ID = Constants.single_id


class DatatypesHelpers():

    @staticmethod
    def _nanify(value) -> float:
        """
        Converts any non-mueric value into an nan.
        """
        
        try:
            _value = float(value)
            if numpy.isinf(_value) or numpy.isnan(_value):
                _value = numpy.nan
        except:
            _value = numpy.nan
        return _value

    @staticmethod
    def nanify(value) -> numpy.ndarray:
        """
        Verctorized version of static method `_nanify`.
        """

        return numpy.vectorize(DatatypesHelpers._nanify)(value)


class TimeSeries(object):
    """
    The base class for pyFOOMB data objects, which describe time variant data. 
    """

    def __init__(self, name:str, timepoints:numpy.ndarray, values:numpy.ndarray, replicate_id:str=None, info:str=None):
        """
        Arguments
        ---------
            name : str
                The name of the TimeSeries object for meaningful identification.

            timepoints : numpy.ndarray
                The vector of timepoints.

            values : numpy.ndarray
                The corresponding vector of values.

        Keyword arguments
        -----------------
            replicate_id : str
                The replicate ID this TimeSeries object is connected with.
                Default is None, which relates this TimeSeries object to an implicit single-replicate Caretaker.

            info : str
                Some further meaningful info.
                Default is None.
        """
        
        self._is_init = True
        self.name = name
        self.timepoints = timepoints
        self.values = values
        self.replicate_id = replicate_id
        self.info = info
        self._is_init = False
        self._equal_lengths()


    @property
    def length(self) -> int:
        return numpy.sum(self.joint_mask)


    @property
    def joint_mask(self) -> numpy.ndarray:
        return self.mask_timepoints * self.mask_values


    @property
    def mask_timepoints(self) -> numpy.ndarray:
        return ~numpy.isnan(self._timepoints)


    @property
    def mask_values(self) -> numpy.ndarray:
        return ~numpy.isnan(self._values)


    @property
    def timepoints(self) -> numpy.ndarray:
        return self._timepoints[self.joint_mask]
    

    @timepoints.setter
    def timepoints(self, value):
        _timepoints = self._shape_checker(value)
        self._equal_lengths(_timepoints)
        self._timepoints = _timepoints


    @property
    def values(self) -> numpy.ndarray:
        return self._values[self.joint_mask]
    

    @values.setter
    def values(self, value):
        _values = self._shape_checker(value)
        self._equal_lengths(_values)
        self._values = _values


    def _shape_checker(self, in_array) -> numpy.array:
        """
        Ensures that the passed array is convertible into a numpy array of shape (n, ).
        """

        _in_array = DatatypesHelpers.nanify(in_array)
        if len(_in_array.shape) > 1:
            if len(_in_array.shape) > 2 or _in_array.shape[1] > 1:
                raise ValueError('Only arrays with one dimension are supported')
            else:
                out_array = _in_array.flatten()
        else:
            out_array =  _in_array.flatten()

        return out_array


    def _equal_lengths(self, in_array=None):
        """
        Ensures that all value properties (`timepoints`, `values`, `errors`) have the same length.
        """

        if not self._is_init:
            lengths = []
            if in_array is not None:
                lengths.append(len(in_array))
            lengths.append(len(self._timepoints))
            lengths.append(len(self._values))
            if hasattr(self, 'errors'):
                if self.errors is not None:
                    lengths.append(len(self._errors))
            all_same_length = not lengths or lengths.count(lengths[0]) == len(lengths)
            if not all_same_length:
                raise ValueError(f'Timepoints and corresponding values (and errors, if present) must be the same length {lengths}')


    def plot(self, ax=None, label=None, title=None, linestyle='-', marker='.', color=None):
        """
        Simple plotting method for rapid protoyping.
        """

        if ax is None:
            fig, ax = pyplot.subplots()
        if label is None:
            label = self.name
        ax.plot(self.timepoints, self.values, label=label, linestyle=linestyle, marker=marker, color='black' if color is None else color)
        if title is not None:
            ax.set_title(title)
        elif self.replicate_id is not None:
            ax.set_title(f'Replicate ID: {self.replicate_id}')
        ax.legend()
        ax.grid(True)
        return ax

    def __str__(self):
        if self.info is None:
            return f'{self.__class__.__name__} | Name: {self.name} | Replicate ID: {self.replicate_id}'
        else:
            return f'{self.__class__.__name__} | Name: {self.name} | Replicate ID: {self.replicate_id} | Info: {self.info}'


class ModelState(TimeSeries):
    """
    The dynamics in time for a certain model state, described by the governing system of differential equations.
    """

    def __init__(self, name:str, timepoints:numpy.ndarray, values:numpy.ndarray, replicate_id:str=None, info:str=None):
        """
        Arguments
        ---------
            name : str
                The name of this ModelState object for meaningful identification.

            timepoints : numpy.ndarray
                The vector of timepoints.

            values : numpy.ndarray
                The corresponding vector of values.

        Keyword arguments
        -----------------
            replicate_id : str
                The replicate ID this ModelState object is connected with.
                Default is None, which relates this ModelState object to an implicit single-replicate Caretaker.

            info : str
                Some further meaningful info.
                Default is None.
        """

        super().__init__(name=name, timepoints=timepoints, values=values, replicate_id=replicate_id, info=info)


class Observation(TimeSeries):
    """
    The dynamics in time for the observation of a certain model state. 
    An observation is typically described by an algebraic equation.
    """

    def __init__(self, name:str, observed_state:str, timepoints:numpy.ndarray, values:numpy.ndarray, replicate_id:str=None):
        
        """
        Arguments
        ---------
            name : str
                The name of this Observation object for meaningful identification.

            observed_state : str
                Determines which model state is observed.

            timepoints : numpy.ndarray
                The vector of timepoints.

            values : numpy.ndarray
                The corresponding vector of values.

        Keyword arguments
        -----------------
            replicate_id : str
                The replicate ID this ModelState object is connected with.
                Default is None, which relates this Observation object to an implicit single-replicate Caretaker.
        """

        self.observed_model_state = observed_state
        info = f'{name}, observes {observed_state}'
        super().__init__(name=name, timepoints=timepoints, values=values, replicate_id=replicate_id, info=info)


    def plot(self, ax=None, label=None, title=None, linestyle='-', marker='.', color=None):
        if label is None:
            label = self.info
        return super().plot(ax=ax, label=label, title=title, linestyle=linestyle, marker=marker, color=color)


class Measurement(TimeSeries):
    """
    The time-resolved measurements needed for model calibation. This class manages also necessary methods and properties related to this task, 
    including the statistical distribution of the measurement values, an error model, and the calculation of a loss for corresponding model predicitions.
    """

    def __init__(self, 
                 name:str, timepoints:numpy.ndarray, values:numpy.ndarray, errors:numpy.ndarray=None, 
                 replicate_id:str=None, info:str=None, 
                 error_model:Callable=None, error_model_parameters:dict=None,
                 error_distribution=norm, distribution_kwargs:dict=None,
                 ):
        """
        Arguments
        ---------
            name : str
                The name of this Measurement object for meaningful identification. 
                The name corresponds typically to a ModelState or Observation, 
                which is needed to calculate a loss from a model prediction.

            timepoints : numpy.ndarray
                The vector of timepoints.

            values : numpy.ndarray
                The corresponding vector of values.

        Keyword arguments
        -----------------
            errors : numpy.ndarray
                A positive vector of errors, corresponding to the `values` argument.
                Default is None, which implies there are no measurement errors available.

            replicate_id : str
                The replicate ID this ModelState object is connected with.
                Default is None, which relates this Measurement object to an implicit single-replicate Caretaker.

            info : str
                Some further meaningful info.
                Default is None.

            error_model : Callable
                An error model describing the measurement error as function of the measurement values 
                and further error model parameters. Is typically used when there are no measurement errors available.
                Often, the error is modeled as constant value or linearly dependent on the measurement values. 
                Default is None, which means there is no measurement error modeled.

            error_model_parameters : dict
                A set of parameter values needed to specify the `error_model`.
                In case an error model with out further parametrization is used, this argument should be an empty dictionary.
                Default is None. 

            error_distribution : scipy.stat.rv_continuous
                Describes the statistical distribution the measurement errors are assumed to follow. 
                This distribution is needed to calculate the log-likelihood for corresponding model predictions, 
                as well as to draw random samples for MC-based parameter estimation.
                Default is scipy.stat.norm, assuming that each measurement error follows a normal distribution.

            distribution_kwargs : 
                Any further parameters needed to evaluate the error distribution, e.g. the `df` for a student t-distribution.
        """

        super().__init__(name=name, timepoints=timepoints, values=values, replicate_id=replicate_id, info=info)
        self._is_init = True
        self.errors = errors     
        self.error_model = error_model
        self.error_model_parameters = error_model_parameters
        self.error_distribution = error_distribution
        if distribution_kwargs is None:
            self.distribution_kwargs = {}
        else:
            self.distribution_kwargs = distribution_kwargs
        if self.errors is None and self.error_model is not None:
            self.apply_error_model()
        self._is_init = False


    @property
    def joint_mask(self) -> numpy.ndarray:
        _mask_errors = self.mask_errors
        if _mask_errors is None:
            return self.mask_timepoints * self.mask_values
        else:
            return self.mask_timepoints * self.mask_values * _mask_errors


    @property
    def mask_errors(self) -> numpy.ndarray:
        if self._errors is None:
            return None
        else:
            return ~numpy.isnan(self._errors)


    @property
    def error_distribution(self) -> scipy.stats.rv_continuous:
        return self._error_distribution


    @error_distribution.setter
    def error_distribution(self, value:scipy.stats.rv_continuous):
        if not isinstance(value, scipy.stats.rv_continuous):
            raise ValueError('Distribution object must an instance of a subsclass of `scipy.stats.rv_continuous`')
        else:
            self._error_distribution = value


    @property
    def errors(self) -> numpy.ndarray:
        # this is the only vector allowed to be None
        if self._errors is None:
            return None
        else:
            return self._errors[self.joint_mask]
    

    @errors.setter
    def errors(self, value):
        if value is not None: 
            _errors = self._shape_checker(value)
            self._equal_lengths(_errors)
            if any(_errors[~numpy.isnan(_errors)]<=0):
                raise ValueError(f'Errors cannot be <= 0. {_errors}')
        else:
            _errors = None
        self._errors = _errors


    @property
    def error_model(self) -> Callable:
        return self._error_model


    @error_model.setter
    def error_model(self, value:Callable):
        self._error_model = value
        if not self._is_init and hasattr(self, '_error_model_parameters'):
            self.apply_error_model()


    @property
    def error_model_parameters(self) -> dict:
        return self._error_model_parameters


    @error_model_parameters.setter
    def error_model_parameters(self, value:dict):
        self._error_model_parameters = value
        if not self._is_init and hasattr(self, '_error_model'):
            self.apply_error_model()


    def apply_error_model(self, report_level:int=0):
        """
        Applies a current error model with its parametrization to this Measurement object.

        Keyword arguments
        -----------------
            report_level : int
                To raise a warning in case the existig `errors` property is not None.
                Default is 0, which suppresses a warning.
        Warns
        -----
            UserWarning
                The errors property will be overwritten and `report_level` > 0.
        """

        if self.errors is not None and report_level >= 1:
            warnings.warn('Applying the error model will overwrite current `errors` property', UserWarning)
        _errors = self.error_model(self._values, self.error_model_parameters)
        self._errors = _errors


    def update_error_model(self, error_model:Callable, error_model_parameters:dict):
        """
        Replaces an exisiting error model and its parametrization.

        Arguments
        ---------
            error_model : Callable
                The new error model.
            error_model_parameters : dict
                the corresponding parameter values.
        """

        self.errors = None
        del self._error_model_parameters
        del self._error_model
        self.error_model_parameters = error_model_parameters
        self.error_model = error_model
        

    def plot(self, ax=None, label=None, title=None, linestyle='--', marker='.', color=None):
        """
        Simple plotting method for rapid protoyping.
        """

        if ax is None:
            fig, ax = pyplot.subplots()
        if label is None:
            label = self.name
        ax.errorbar(x=self.timepoints, y=self.values, yerr=self.errors, label=label, linestyle=linestyle, marker=marker, color='black' if color is None else color)
        if title is not None:
            ax.set_title(title)
        elif self.replicate_id is not None:
            if self.info is not None:
                _info = f'\n{self.info}'
            else:
                _info = ''
            ax.set_title(f'Replicate ID: {self.replicate_id}{_info}')
        ax.legend()
        ax.grid(True)
        return ax


    def get_loss(self, metric:str, predictions:list, distribution_kwargs:dict=None) -> float:
        """
        Compares this Measurement object with some prediction and calculates a metric of distance (i.e., the loss) to the prediction.
        This metric is used typically for parameter estimation from given measurement data.
        Currently, valid metrics are negative log-likelihood, sum of squares, and weighted sum of squares.
        
        Arguments
        ---------
            metric : str
                The metric which describes the distance between this Measurement object and some prediction. 
                Can be one of 'negLL', 'negative-log-likelihood', or 'SS', 'sum-of-squares', or 'WSS', 'weighted-sum-of-squares'.

            predictions : list of ModelState and/or Observation objects
                Contains the prediction for which this Measurement object shall be compared with.
                If a corresponding prediction is missing, the resulting loss is computed to NaN.

        Keyword arguments
        -----------------
            distribution_kwargs : dict
                Any other arguments than `x`, `loc` and `scale` needed to evaluate the logpdf.
                Default is None, which implies the use of the `distribution_kwargs` property of this Measurement object.

        Returns
        -------
            loss : float
                The loss value calculated according to the chosen metric.

        Raises
        ------
            ValueError
                There are multiple predictions matching the same `name` and `replicate_id`.
            AttributeError
                The pdf is evaluated when `errors` property is None.
            Exception
                Evaluating the pdf fails for other reasons.
            AttributeError
                Weighted sum of squares is calculated when errors property is None.
            NotImplementedError
                An unknown metric for loss calculation is used.
        """

        _prediction = [
            prediction for prediction in predictions 
            if (
                (
                    prediction.name == self.name and prediction.replicate_id == self.replicate_id
                ) or (
                    prediction.name == self.name and (prediction.replicate_id is None and self.replicate_id is None)
                )
            )
        ]

        if len(_prediction) > 1:
            raise ValueError(f'There are multiple predictions matching {self.name} and {self.replicate_id}')
        elif len(_prediction) == 1:
            prediction = _prediction[0]
        else:
            return numpy.nan

        if distribution_kwargs is None:
            distribution_kwargs = self.distribution_kwargs

        prediction_mask = numpy.isin(prediction.timepoints, self.timepoints)
        y_pred = prediction.values[prediction_mask]
        if prediction_mask.sum() < len(self.timepoints):
            measurement_mask = numpy.isin(self.timepoints, prediction.timepoints)
            y_meas = self.values[measurement_mask]
            y_meas_err = self.errors[measurement_mask] if self.errors is not None else None
        else:
            y_meas = self.values
            y_meas_err = self.errors if self.errors is not None else None

        if metric in ['negLL', 'negative-log-likelihood']:
            if self.errors is None and isinstance(self.error_distribution, scipy.stats.rv_continuous):
                raise AttributeError('Could not calculate log-likelihood for continous random distribution because `errors` property is None.')
            try:
                if hasattr(self.error_distribution, 'logpdf'):
                    losses = -1*self.error_distribution.logpdf(x=y_pred, loc=y_meas, scale=y_meas_err, **distribution_kwargs)
            except Exception as e:
                warnings.warn(f'Could not evaluate logpdf. Maybe due to missing keyword arguments? {distribution_kwargs}')
                raise e
        
        elif metric in ['SS', 'sum-of-squares']:
            losses = numpy.square(y_pred-y_meas)
        
        elif metric in ['WSS', 'weighted-sum-of-squares']:
            if self.errors is None:
                raise AttributeError('Cannot evaluate weighted sum of squares because `errors` property is None.')
            losses = numpy.square((y_pred-y_meas)/y_meas_err)
        
        else:
            raise NotImplementedError('Unknown metric for calculating the loss')

        if not all(numpy.isnan(losses)):
            loss = numpy.nansum(losses)
        else:
            loss = numpy.nan
        
        return loss


    def _get_random_samples_values(self, distribution_kwargs:dict=None) -> numpy.ndarray:
        """
        Draws random samples from the measurement values, according to the specified statistical distribution.

        Keyword arguments
        -----------------
            distribution_kwargs : dict
                Values for distribution parameters.
                Default is None, which refers to the `distribution_kwargs` property of this Measurement object.
                
        """

        if distribution_kwargs is None:
            distribution_kwargs = self.distribution_kwargs
        if self.errors is None:
            raise AttributeError('Property `errors` is None, thus cannot sample random values.')
        # get rvs for masked values
        _rvs_unmasked = numpy.zeros_like(self.joint_mask)*numpy.nan
        # Make sure to get nonnan values only at the maks positions
        _rvs_masked = self.error_distribution.rvs(loc=self.values, scale=self.errors, **distribution_kwargs)
        _rvs_unmasked[self.joint_mask] = _rvs_masked 

        return _rvs_unmasked


class Sensitivity(TimeSeries):
    """
    Describes the time-varying sensitivity for a model state or observation w.r.t. to a model parameter, initial value or observation parameter.
    """

    def __init__(self, timepoints:numpy.ndarray, values:numpy.ndarray, response:str, parameter:str, info=None, h:float=None, replicate_id:str=None):
        """
        Arguments
        ---------
            timepoints : numpy.ndarray
                The vector of timepoints.

            values : numpy.ndarray
                The corresponding vector of values.

            response : str
                The model response (i.e., model state or observation) for this sensitivity.

            parameter : str
                The parameter (i.e., model parameter, initial value or observation parameter) for this sensitivity.

        Keyword arguments
        -----------------
            info : str
                Some further meaningful info.
                Default is None.

            h : float
                The perturbation value used for the calculation of sensitivity via central difference quotient.
                Default is None.

            replicate_id : str
                The replicate ID this Sensitivity object is connected with.
                Default is None, which relates this Sensitivity object to an implicit single-replicate Caretaker.
        """

        self.response = response
        self.parameter = parameter
        self.h = h
        name = f'd({response})/d({parameter})'
        super().__init__(name=name, timepoints=timepoints, values=values, replicate_id=replicate_id, info=info)


    def plot(self, ax=None, label=None, title=None, linestyle='-', marker='.', color=None):
        if label is None:
            label = self.name
        ext_label = '\n'
        if self.h is not None:
            ext_label = f'{ext_label}h={self.h:.2e} '
        if ext_label != '\n':
            label = f'{label}{ext_label}'
        if title is None:
            title = f'Replicate ID: {self.replicate_id}\nResponse: {self.response} | Parameter: {self.parameter}'
        return super().plot(ax=ax, label=label, title=title, linestyle=linestyle, marker=marker, color=color)
