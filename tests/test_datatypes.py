
import numpy as np
import pytest
import scipy

from pyfoomb.datatypes import Measurement
from pyfoomb.datatypes import ModelState
from pyfoomb.datatypes import Observation
from pyfoomb.datatypes import Sensitivity
from pyfoomb.datatypes import TimeSeries


class StaticHelpers():

    name = 'TestName'
    timepoints = [1, 2, 3, 4, 5]
    values = [100, 200, 300, 400, 500]
    errors = [1/1, 1/2, 1/3, 1/4, 1/5]
    replicate_id = '1st'
    state = 'y1'
    parameter = 'p1'


class StaticDatatypes():

    timeseries = TimeSeries(
        name=StaticHelpers.name, 
        timepoints=StaticHelpers.timepoints, 
        values=StaticHelpers.values,
        replicate_id=StaticHelpers.replicate_id
    )

    modelstate = ModelState(
        name=StaticHelpers.name, 
        timepoints=StaticHelpers.timepoints, 
        values=StaticHelpers.values,
        replicate_id=StaticHelpers.replicate_id
    )

    measurement_wo_errs = Measurement(
        name=StaticHelpers.name, 
        timepoints=StaticHelpers.timepoints, 
        values=StaticHelpers.values,
        replicate_id=StaticHelpers.replicate_id
    )

    measurement_w_errs = Measurement(
        name=StaticHelpers.name, 
        timepoints=StaticHelpers.timepoints, 
        values=StaticHelpers.values,
        errors=StaticHelpers.errors,
        replicate_id=StaticHelpers.replicate_id
    )

    observation = Observation(
        name=StaticHelpers.name, 
        timepoints=StaticHelpers.timepoints, 
        values=StaticHelpers.values, 
        observed_state=StaticHelpers.state, 
        replicate_id=StaticHelpers.replicate_id
    )

    sensitivity = Sensitivity(
        timepoints=StaticHelpers.timepoints, 
        values=StaticHelpers.values, 
        response=StaticHelpers.state,
        parameter=StaticHelpers.parameter, 
        replicate_id=StaticHelpers.replicate_id
    )


class StaticErrorModelHelpers():

    constant_error_model_parameters = {
            'offset' : 0,
        }

    linear_error_model_parameters = {
                'offset' : 0,
                'slope' : 1,
    }

    squared_error_model_parameters = {
        'w0' : 1,
        'w1' : 0.1,
        'w2' : 0.02,
    }

    @staticmethod
    def constant_error_model(values, parameters):
            offset = parameters['offset']
            return np.ones_like(values)*offset

    @staticmethod
    def linear_error_model(values, parameters):
            offset = parameters['offset']
            slope = parameters['slope']
            return values * slope + offset

    @staticmethod
    def squared_error_model(values, parameters):
            w0 = parameters['w0']
            w1 = parameters['w1']
            w2 = parameters['w2']
            return w0 + values*w1 + np.square(values)*w2


#%% Actual tests

class TestInstantiationVariousDatatypes():

    @pytest.mark.parametrize(
        'values, errors, info, replicate_id', 
        [
            ([[10], [20], [30], [40], [50]], None, None, None),
            (StaticHelpers.values, None, None, None),
            (StaticHelpers.values, StaticHelpers.errors, None, None),
            (StaticHelpers.values, StaticHelpers.errors, 'TestInfo', None),
            (StaticHelpers.values, StaticHelpers.errors, 'TestInfo', '1st'),
        ]
    )
    def test_init_datatypes(self, values, errors, info, replicate_id):
        """
        To test typical instantiations of datatypes. 
        """
        
        # Standard instatiations
        TimeSeries(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=values, info=info, replicate_id=replicate_id)
        ModelState(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=values, info=info, replicate_id=replicate_id)
        Measurement(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=values, errors=errors, info=info, replicate_id=replicate_id)
        Observation(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=values, observed_state='y1', replicate_id=replicate_id)
        Sensitivity(timepoints=StaticHelpers.timepoints, values=values, response='y1', parameter='p1', replicate_id=replicate_id)
        Sensitivity(timepoints=StaticHelpers.timepoints, values=values, response='y1', parameter='p1', h=1e-8, replicate_id=replicate_id)

        # Must provide timepoints
        with pytest.raises(ValueError):
            TimeSeries(name=StaticHelpers.name, timepoints=None, values=values)

        # Must provide values
        with pytest.raises(ValueError):
            TimeSeries(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=None)

        # Measurements can be created with error_models
        Measurement(
            name=StaticHelpers.name, 
            timepoints=StaticHelpers.timepoints, 
            values=values, 
            error_model=StaticErrorModelHelpers.constant_error_model, 
            error_model_parameters=StaticErrorModelHelpers.constant_error_model_parameters,
        )

        # Must provide a subclass of rvs.continous as p.d.f.
        with pytest.raises(ValueError):
            Measurement(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=values, error_distribution=scipy.stats.bernoulli)

        # Error values must be >0
        with pytest.raises(ValueError):
            Measurement(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=values, errors=[0]*len(StaticHelpers.values))
        with pytest.raises(ValueError):
            Measurement(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=values, errors=[-1]*len(StaticHelpers.values))


    @pytest.mark.parametrize(
        'input_vector, masked_vector',  
        [
            ([10, None, 30, 40, 50], [10, 30, 40, 50]), 
            ([10, 20, np.nan, 40, 50], [10, 20, 40, 50]),
            ([10, 20, 30, np.inf, 50], [10, 20, 30, 50]),
            ([10, 20, 30, 40, -np.inf], [10, 20, 30, 40]),
            ([10, None, np.nan, np.inf, -np.inf], [10]),
        ]
    )
    def test_init_datatypes_masking_non_numeric(self, input_vector, masked_vector):
        """
        Non-numeric values implicitly define a mask, which is in turn applied to all vectors of the corresponding datatypes.
        """

        _timeseries = TimeSeries(name=StaticHelpers.name, timepoints=input_vector, values=StaticHelpers.values)
        assert all(_timeseries.timepoints) == all(masked_vector)
        assert _timeseries.length == len(masked_vector)

        _timeseries = TimeSeries(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=input_vector)
        assert all(_timeseries.timepoints) == all(masked_vector)
        assert _timeseries.length == len(masked_vector)

        _measurement = Measurement(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=StaticHelpers.values, errors=input_vector)
        assert all(_measurement.timepoints) == all(masked_vector)
        assert _measurement.length == len(masked_vector)


class TestSetters():
        
    @pytest.mark.parametrize(
        'bad_vector', 
        [
            ([10, 20, 30, 40]), # length does not match the length of the other vectors
            ([[10, 20, 30, 40, 50], [10, 20, 30, 40, 50]]), # Cannot be cast into 1D vector
            ([[10, 20, 30, 40, 50]]),
        ],
    )
    def test_setters_reject_bad_vectors(self, bad_vector):
        """
        Testing that the setters do not accept bad_vectors.
        """

        with pytest.raises(ValueError):
            StaticDatatypes.timeseries.values = bad_vector
        with pytest.raises(ValueError):
            StaticDatatypes.measurement_w_errs.errors = bad_vector


class TestPlot():

    @pytest.mark.parametrize(
        'datatype', [
            (StaticDatatypes.timeseries),
            (StaticDatatypes.modelstate),
            (StaticDatatypes.measurement_wo_errs),
            (StaticDatatypes.observation),
            (StaticDatatypes.sensitivity)
        ]
    )
    def test_plotting(self, datatype):
        """
        The pyfoomb datatypes come with an own plot method for rapid development.
        Based on some properties of the datatype, the plot will auto-generated legend and title in different ways.
        Some arguments are also tested.
        """

        datatype.plot()
        datatype.plot(title='Some title')
        datatype.replicate_id = StaticHelpers.replicate_id
        datatype.plot()
        datatype.info = 'Some info'
        datatype.plot()


class TestMeasurementErrorModels():

    @pytest.mark.parametrize(
        'error_model, error_model_parameters', 
        [
            (StaticErrorModelHelpers.constant_error_model, StaticErrorModelHelpers.constant_error_model_parameters), 
            (StaticErrorModelHelpers.linear_error_model, StaticErrorModelHelpers.linear_error_model_parameters),
            (StaticErrorModelHelpers.squared_error_model, StaticErrorModelHelpers.squared_error_model_parameters),
        ]
    )
    def test_update_error_models_parameters(self, error_model, error_model_parameters):
        """
        Updates error_models for existing Measurement objects
        """

        # create measurement first
        measurement = Measurement(name=StaticHelpers.name, timepoints=StaticHelpers.timepoints, values=StaticHelpers.values)

        # To use a different (new) error_model, it must be passed with its corresponding error_model_parameters
        measurement.update_error_model(error_model=error_model, error_model_parameters=error_model_parameters)

        # Parameter values can be updated, as long as all parameters are present in the new dictionary
        measurement.error_model_parameters = {_p : error_model_parameters[_p]*1.5 for _p in error_model_parameters}

        # Incase the error model is applied, a warning can be given for overwriting the error vector
        with pytest.warns(UserWarning):
            measurement.apply_error_model(report_level=1)

        # Setting new parameter values won't work
        with pytest.raises(KeyError):
            measurement.error_model_parameters = {'bad_parameter' : 1000}

    @pytest.mark.parametrize(
        'metric', 
        [
            ('negLL'),
            ('WSS'),
            ('SS'),
            ('bad_metric')
        ],
    )
    def test_metrics_and_loss_caluclation(self, metric):
        
        if metric == 'bad_metric':
            with pytest.raises(NotImplementedError):
                StaticDatatypes.measurement_wo_errs.get_loss(metric=metric, predictions=[StaticDatatypes.modelstate])
        # Only for metric SS (sum-of-squares) the loss can be calculated from Measurement objects without having errors
        elif metric == 'SS':
            assert not np.isnan(StaticDatatypes.measurement_wo_errs.get_loss(metric=metric, predictions=[StaticDatatypes.modelstate]))
            assert not np.isnan(StaticDatatypes.measurement_w_errs.get_loss(metric=metric, predictions=[StaticDatatypes.modelstate]))
        else:
            with pytest.raises(AttributeError):
                StaticDatatypes.measurement_wo_errs.get_loss(metric=metric, predictions=[StaticDatatypes.modelstate])
            assert not np.isnan(StaticDatatypes.measurement_w_errs.get_loss(metric=metric, predictions=[StaticDatatypes.modelstate]))

            # Fail fast for ambiguous, non-unique predictions: list of prediction with the same 'name' and 'replicate_id'
            with pytest.raises(ValueError):
                StaticDatatypes.measurement_w_errs.get_loss(metric=metric, predictions=[StaticDatatypes.modelstate]*2)

            modelstate = ModelState(
                name=StaticHelpers.name, 
                timepoints=StaticHelpers.timepoints[::2], # Use less timepoints
                values=StaticHelpers.values[::2], # Use less values, corresponding to using less timepoints
            )
            # Using predictions that have not matching replicate_ids returns nan loss
            assert np.isnan(StaticDatatypes.measurement_wo_errs.get_loss(metric='SS', predictions=[modelstate]))

            # When adding at least one matching prediction, a loss can be calculated
            assert not np.isnan(StaticDatatypes.measurement_w_errs.get_loss(metric=metric, predictions=[modelstate, StaticDatatypes.modelstate]))


class TestMiscellaneous():

    def test_str(self):
        print(StaticDatatypes.timeseries)

    def test_other_distributions(self):
        
        # Create a Measurement object having an error t-distribution 
        measurement_t = Measurement(
            name=StaticHelpers.name, 
            timepoints=StaticHelpers.timepoints, 
            values=StaticHelpers.values, 
            errors=StaticHelpers.values, 
            error_distribution=scipy.stats.t,
            distribution_kwargs={'df' : 1},
            replicate_id='1st'
        )

        # Get rvs values for, e.g. MC sampling
        measurement_t._get_random_samples_values()

        loss_1 = measurement_t.get_loss(metric='negLL', predictions=[StaticDatatypes.modelstate])
        assert not np.isnan(loss_1)
        loss_2 = measurement_t.get_loss(metric='negLL', predictions=[StaticDatatypes.modelstate], distribution_kwargs={'df' : 100})
        assert not np.isnan(loss_2)

        # Does not work in case there are no errors
        with pytest.raises(AttributeError):
            StaticDatatypes.measurement_wo_errs._get_random_samples_values()













