
import numpy as np

import pytest

from pyfoomb.datatypes import TimeSeries
from pyfoomb.datatypes import Measurement

from pyfoomb.utils import Calculations
from pyfoomb.utils import Helpers


class StaticData():
    measurements_wo_errors = Measurement(name='M1', timepoints=[1, 2, 3], values=[10, 20, 30], replicate_id='1st')
    measurements_w_errors = Measurement(name='M2', timepoints=[2, 3, 4], values=[20, 30, 40], errors=[1/20, 1/30, 1/40], replicate_id='1st')
    time_series_1 = TimeSeries(name='TS1', timepoints=[1, 2], values=[10, 20], replicate_id='1st')
    time_series_2 = TimeSeries(name='TS2', timepoints=[2, 3], values=[20, 30], replicate_id='2nd')


class TestCalculations():

    def test_corr(self):

        matrix = np.array(
            [
                [1, 2], [3, 4]
            ]
        )
        Calculations.cov_into_corr(matrix)

        # can use only square matrices
        non_square_matrix = np.array(
            [
                [1, 2, 3], [4, 5, 6]
            ]
        )
        with pytest.raises(ValueError):
            Calculations.cov_into_corr(non_square_matrix)


class TestHelpers():

    def test_bounds_to_floats(self):
        int_bounds = [(0, 1), (2, 3)]
        float_bounds = Helpers.bounds_to_floats(int_bounds)
        for _bounds in float_bounds:
            assert isinstance(_bounds[0], float)
            assert isinstance(_bounds[1], float)
    
    @pytest.mark.parametrize(
        'ok_ids',
        [
            ({'a01' : 1, 'b01' : 2}),
            ({'b01' : 1, 'a01' : 2}),
            (['a01', 'b01']),
            (['a01'])

        ]
    )
    def test_unique_ids_ok(self, ok_ids):
        """
        To ensure that ids (replicate_ids, states, etc) are case-insensitive unique
        """
        assert Helpers.has_unique_ids(ok_ids)

    @pytest.mark.parametrize(
        'not_ok_ids',
        [
            ({'a01' : 1, 'A01' : 2}),
            (['a01', 'b01', 'b01']),
            (['a01', 'B01', 'b01']),
        ]
    )
    def test_unique_ids_not_ok(self, not_ok_ids):
        """
        To ensure that ids (replicate_ids, states, etc) are case-insensitive unique
        """
        assert not Helpers.has_unique_ids(not_ok_ids)

    def test_unique_ids_must_be_list_or_dict(self):
        with pytest.raises(TypeError):
            Helpers.has_unique_ids(('a01', 'b01'))

    def test_utils_for_datatypes(self):
        
        # To check whether all measurements in a list of those hve errors or not
        assert not Helpers.all_measurements_have_errors([StaticData.measurements_wo_errors, StaticData.measurements_w_errors])
        assert Helpers.all_measurements_have_errors([StaticData.measurements_w_errors, StaticData.measurements_w_errors])
        assert not Helpers.all_measurements_have_errors([StaticData.measurements_wo_errors, StaticData.measurements_wo_errors]) 

        # Get the joint time vector of several TimeSeries objects
        actual = Helpers.get_unique_timepoints([StaticData.measurements_wo_errors, StaticData.measurements_w_errors])
        for _actual, _expected in zip(actual, np.array([1., 2., 3., 4.])):
            assert _actual == _expected

        # Extract a specific TimeSeries from a list
        timeseries_list = [StaticData.measurements_wo_errors, StaticData.measurements_w_errors]
        assert isinstance(Helpers.extract_time_series(timeseries_list, replicate_id='1st', name='M1'), TimeSeries)
        # In case not match is found, the method returns None
        with pytest.warns(UserWarning):
            assert Helpers.extract_time_series(timeseries_list, replicate_id='2nd', name='M1', no_extraction_warning=True) is None
        # More than one match is found
        with pytest.raises(ValueError):
             Helpers.extract_time_series(timeseries_list*2, replicate_id='1st', name='M1')

    def test_parameter_collections(self):
        """
        Methods related to parameter distributions from MC sampling or parameter scanning studies
        """

        parameter_collection_not_ok = {
            'p1' : [1, 2, 3],
            'p2' : [10, 20, 30, 40]
        }
        parameter_collection_ok = {
            'p1' : [1, 2, 3],
            'p2' : [10, 20, 30]
        }
        # The parameters shall all have the same length
        assert Helpers.get_parameters_length(parameter_collection_ok) == 3
        # The parameters are not allowed to have different lengths
        with pytest.raises(ValueError):
             Helpers.get_parameters_length(parameter_collection_not_ok)

        # Parameter collections can be sliced for, e.g. get predictions for a particular slice
        parameter_slices = Helpers.split_parameters_distributions(parameter_collection_ok)
        for parameter_slice in parameter_slices:
            assert list(parameter_slice.keys()) == list(parameter_collection_ok.keys())

    def test_unique_timepoints(self):
        t_all = Helpers.get_unique_timepoints(
            [
                StaticData.time_series_1, 
                StaticData.time_series_2, 
                StaticData.measurements_w_errors, 
                StaticData.measurements_wo_errors,
            ]
        )
        assert len(t_all) == 4
        assert all(np.equal(t_all, np.array([1, 2, 3, 4])))
