

import numpy as np

import pytest

from pyfoomb import BioprocessModel
from pyfoomb import Observation
from pyfoomb import ModelState

from modelling_library import ModelLibrary
from modelling_library import ObservationFunctionLibrary


class TestBioprocessModel():

    @pytest.mark.parametrize('name', ModelLibrary.modelnames)
    def test_init_model(self, name):
        # Collect required parts to create the model instance
        modelclass = ModelLibrary.modelclasses[name]
        states = ModelLibrary.states[name]
        model_parameters_list = list(ModelLibrary.model_parameters[name].keys())
        # Instantiate the model class, expected to work
        modelclass(states=states, model_parameters=model_parameters_list, model_name='my_model')
        model = modelclass(states=states, model_parameters=model_parameters_list)
        # The states argument for instatiation must must be a list as they have no values like parameters do
        with pytest.raises(TypeError):
            modelclass(states={_state : 0 for _state in states}, model_parameters=model_parameters_list)
        # The states can only be set during instatiation
        with pytest.raises(AttributeError):
            model.states = states
        # There is also a str method
        print(model)
        # States must be unique
        with pytest.raises(KeyError):
            modelclass(states=states*2, model_parameters=model_parameters_list)
        

    def test_init_model_with_events(self):
        # Model03 has events
        name = 'model03'
        modelclass = ModelLibrary.modelclasses[name]
        states = ModelLibrary.states[name]
        model_parameters = ModelLibrary.model_parameters[name]
        # The number of initial_switches can be autodetected
        model_v01 = modelclass(states=states, model_parameters=model_parameters)
        # Can also explicitly set the intial_switches
        model_v02 = modelclass(states=states, model_parameters=model_parameters, initial_switches=[False])
        assert model_v01.initial_switches == model_v02.initial_switches

    def test_set_parameters(self):
        # Get a model instance to work with
        name = 'model03'
        modelclass = ModelLibrary.modelclasses[name]
        states = ModelLibrary.states[name]
        model_parameters = ModelLibrary.model_parameters[name]
        initial_values = ModelLibrary.initial_values[name]
        model = modelclass(states=states, model_parameters=list(model_parameters.keys()))
        # The BioprocessModel object has a dedicated method to set parameters (model parameters & initial values)
        model.set_parameters(model_parameters)
        model.set_parameters(initial_values)
        # Parameter names must be case-insensitive unique
        with pytest.raises(KeyError):
            non_unique_params = {str.upper(_iv) : 1 for _iv in initial_values}
            non_unique_params.update(initial_values)
            model.set_parameters(non_unique_params)
        # Keys for initial values must match the state names extended by "0"
        model.initial_values = {f'{_state}0' : 1 for _state in states}
        with pytest.raises(KeyError):
            model.initial_values = {f'{_state}X' : 1 for _state in states}
        # Initial values and model parameters must be a dict
        with pytest.raises(TypeError):
            model.initial_values = [_iv for _iv in initial_values]
        with pytest.raises(TypeError):
            model.model_parameters = [_p for _p in model_parameters]
        # After init, no new model_parameters can be introduced
        with pytest.raises(KeyError):
            model.model_parameters = {**model_parameters, 'new_p' : 1}
        # Number of initial_switches cannot be changed after init
        initial_switches = ModelLibrary.initial_switches[name]
        with pytest.raises(ValueError):
            model.initial_switches = initial_switches*2
        # Initial switches must be a list of booleans
        with pytest.raises(ValueError):
            model.initial_switches = ['False' for _ in initial_switches]


class TestObservationFunction():

    @pytest.mark.parametrize('name', ObservationFunctionLibrary.names)
    def test_init(self, name):
        obsfun = ObservationFunctionLibrary.observation_functions[name]
        observed_state = ObservationFunctionLibrary.observed_states[name]
        observation_parameters = ObservationFunctionLibrary.observation_function_parameters[name]
        obsfun(observed_state=observed_state, observation_parameters=list(observation_parameters.keys()))

    def test_get_observations(self):
        # Create an ObservationFunction
        name = 'obsfun01'
        obsfun = ObservationFunctionLibrary.observation_functions[name]
        observed_state = ObservationFunctionLibrary.observed_states[name]
        observation_parameters = ObservationFunctionLibrary.observation_function_parameters[name]
        observation_function = obsfun(observed_state=observed_state, observation_parameters=list(observation_parameters.keys()))

        # After creating the ObservationFunction, all parameter values are None, regardless if a list of dictionary is used as argument for parameters
        for _p in observation_parameters:
            assert observation_function.observation_parameters[_p] is None

        # One must explicitly set the parameter values
        observation_function.set_parameters(observation_parameters)
        for _p in observation_parameters:
            assert observation_function.observation_parameters[_p] is not None

        # Create a ModelState that now can be observed observe
        modelstate = ModelState(
            name=ObservationFunctionLibrary.observed_states[name], 
            timepoints=[1, 2, 3], 
            values=[10, 20, 30],
        )
        observation_function.get_observation(modelstate)

        # The ModelState to be observed must match the ObservationsFunction's replicate_id
        modelstate.replicate_id = '1st'
        with pytest.raises(ValueError):
            observation_function.get_observation(modelstate)

        # Same for the state name
        modelstate.name = 'other_state'
        with pytest.raises(KeyError):
            observation_function.get_observation(modelstate)

    def test_properties(self):
        # Create an ObservationFunction
        name = 'obsfun01'
        obsfun = ObservationFunctionLibrary.observation_functions[name]
        observed_state = ObservationFunctionLibrary.observed_states[name]
        observation_parameters = ObservationFunctionLibrary.observation_function_parameters[name]
        observation_function = obsfun(observed_state=observed_state, observation_parameters=list(observation_parameters.keys()))
        observation_function.set_parameters(observation_parameters)

        # Can't change the observed state after instantiation
        with pytest.raises(AttributeError):
            observation_function.observed_state = 'new_state'

        # Cant't set unknown parameters
        with pytest.raises(KeyError):
            observation_function.observation_parameters = {'unknown_parameter' : 1000}

        # Observed state parameter must match the corresponding property of ObservationFunction
        with pytest.raises(ValueError):
            observation_function.observation_parameters = {**observation_parameters, 'observed_state' : 'unknown_state'}

        # Must use a dictionary to set the property
        with pytest.raises(ValueError):
            observation_function.observation_parameters = list(observation_parameters.keys())

        # There is a str method
        print(observation_function)


    




        














