import collections
import copy
import numpy
from typing import Dict, List
import warnings

from .constants import Messages
from .datatypes import Measurement
from .datatypes import TimeSeries
from .datatypes import Sensitivity
from .constants import Constants

SINGLE_ID = Constants.single_id

class OwnDict(collections.OrderedDict):
    """
    Extendeds OrderedDict with `to_numpy()` method
    """  

    def to_numpy(self) -> numpy.ndarray:
        return numpy.array(list((self.values())))


class Calculations():

    @staticmethod
    def cov_into_corr(Cov:numpy.ndarray) -> numpy.ndarray:
        """
        Calculates correlation matrix from variance-covariance matrix.

        Arguments
        ---------
            Cov : numpy.ndarray
                Variance-covariance matrix, must be sqaure and positive semi-definite.

        Returns
        -------
            Corr : numpy.ndarray
                Correlation matrix for Cov.

        Raises
        ------
            ValueError
                Cov is not square.
        """
        
        if Cov.shape[0] != Cov.shape[1]:
            raise ValueError('Cov must be square')

        Corr = numpy.zeros_like(Cov) * numpy.nan
        for i in range(Cov.shape[0]):
            for j in range(Cov.shape[0]):
                Corr[i, j] = Cov[i, j] / (numpy.sqrt(Cov[i, i]) * numpy.sqrt(Cov[j, j]))
        return Corr


class Helpers():

    @staticmethod
    def bounds_to_floats(bounds:List[tuple]) -> List[tuple]:
        """
        Casts bounds from int to float.
        """
        
        new_bounds = []
        for _bounds in bounds:
            lower, upper = _bounds
            new_bounds.append((float(lower), float(upper)))
        return new_bounds


    @staticmethod
    def has_unique_ids(values, report:bool=True) -> bool:
        """
        Verifies that a list or dict has only (case-insensitive) unique items or keys, respectively.

        Keyword arguments
        -----------------
            report : bool
                To show the non-unique ids.
        """

        success = True
        
        if isinstance(values, set):
            return success
        if len(values) == 1:
            return success

        _values = copy.deepcopy(values)

        if isinstance(_values, list):
            _values.sort(key=str.lower)
            values_str_lower = [_value.lower() for _value in _values]
            if len(_values) > len(set(values_str_lower)):
                success = False
        elif isinstance(_values, dict) or isinstance(_values, OwnDict):
            _values = list(_values.keys())
            values_str_lower = [_value.lower() for _value in _values]
            if len(_values) > len(set(values_str_lower)):
                success = False
        else:
            raise TypeError(f'Type {type(values)} cannot be handled.')
        if not success and report:
            print(f'Bad, non-unique (case-insensitive) ids: {_values}')

        return success


    @staticmethod
    def all_measurements_have_errors(measurements:List[Measurement]) -> bool:
        """
        Checks whether if Measurement objects have errors.
        """
        
        with_errors = []
        for measurement in measurements:
            if measurement.errors is None:
                with_errors.append(False)
            else:
                with_errors.append(True)
        return all(with_errors)


    @staticmethod
    def get_unique_timepoints(time_series:List[TimeSeries]) -> numpy.ndarray:
        """
        Creates a joint unique time vector from all timepoints of a list of TimeSeries objects.

        Arguments
        ---------
            time_series : List[TimeSeries]
                The list of TimeSeries (and subclasses thereof) for which a joint time vector is wanted.

        Returns
        -------
            t_all : numpy.ndarray
                The joint vector of time points.
        """

        t_all = numpy.array(0)
        for _time_series in time_series:
            t_all = numpy.append(t_all, _time_series.timepoints)
        return numpy.unique(t_all)


    @staticmethod
    def extract_time_series(time_series:List[TimeSeries], name:str, replicate_id:str, no_extraction_warning:bool=False) -> TimeSeries:
        """
        Extract a specific TimeSeries object, identified by its properties `name` and `replicate_id`.
        In case no match is found, None is returned.

        Arguments
        ---------
            time_series : List[TimeSeries]
                The list from which the specific TimeSeries object shall be extracted.
            name : str
                The identifying `name` property.
            replicate_id : str
                The identifying `replicate_id` property.

        Keyword arguments
        -----------------
            no_extraction_warning : bool
                Whether to raise a warning when no TimeSeries object can be extracted.
                Default is False

        Returns
        -------
            extracted_time_series : TimeSeries or None
        
        Raises
        ------
            ValueError
                Multiple TimeSeries objects have the same `name` and `replicate_id` property.

        Warns
        -----
            UserWarning
                No TimeSeries object match the criteria.
                Only raised for `no_extraction_warning` set to True.
        """

        _extracted_time_series = [
            _time_series for _time_series in time_series 
            if _time_series.name == name and _time_series.replicate_id == replicate_id
        ]

        if len(_extracted_time_series) > 1:
            raise ValueError('List of (subclassed) TimeSeries objects is ambigous. Found multiple occurences ')
        elif len(_extracted_time_series) == 0:
            extracted_time_series = None
            if no_extraction_warning:
                warnings.warn(f'Could not extract a TimeSeries object with replicate_id {replicate_id} and name {name}')
        else:
            extracted_time_series = _extracted_time_series[0]

        return extracted_time_series


    @staticmethod
    def get_parameters_length(parameter_collections:Dict[str, numpy.ndarray]) -> int:
        """
        Arguments
        ---------
            parameter_collections : Dict[str, numpy.ndarray]
                A set of parameters (model parameters, initial values, observation parameters).
        
        Returns
        -------
            length : int
                The number of values for each parameter

        Raises
        ------
            ValueError
                Parameters have different number of estimated values.
        """

        lengths = set([len(parameter_collections[_p]) for _p in parameter_collections])
        if len(lengths) > 1:
            raise ValueError('Parameters have different number of estimated values.')
        length = list(lengths)[0]
        return length


    @staticmethod
    def split_parameters_distributions(parameter_collections:Dict[str, numpy.ndarray]) -> List[Dict]:
        """
        Arguments
        ---------
            parameter_collections : Dict[str, numpy.ndarray]
                A set of parameters (model parameters, initial values, observation parameters).
        
        Returns
        -------
            splits : List[Dict]
                A list of separate parameter dictonaries for each slice of the parameter collections.
        """
    
        _length = Helpers.get_parameters_length(parameter_collections)
        splits = [
            {
                _p : parameter_collections[_p][i] 
                for _p in parameter_collections
            } 
            for i in range(_length)
        ]
        return splits