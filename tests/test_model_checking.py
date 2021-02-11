
import pytest

from pyfoomb.simulation import ExtendedSimulator
from pyfoomb.model_checking import ModelChecker

import modelling_library
from modelling_library import ModelLibrary
from modelling_library import ObservationFunctionLibrary


@pytest.fixture
def model_checker():
    return ModelChecker()


class TestCheckBioprocessModel():

    @pytest.mark.parametrize('model', ModelLibrary.variants_model03)
    def test_bioprocess_model_checking(self, model_checker, model):
        name = 'model03'
        model_parameters = ModelLibrary.model_parameters[name]
        initial_values = ModelLibrary.initial_values[name]
        extended_simulator = ExtendedSimulator(bioprocess_model_class=model, model_parameters=model_parameters, initial_values=initial_values)
        # These models should not raise any warnings
        model_checker.check_model_consistency(extended_simulator)

    @pytest.mark.parametrize('bad_model', ModelLibrary.bad_variants_model03)
    def test_bioprocess_bad_model_checking(self, model_checker, bad_model):
        name = 'model03'
        model_parameters = ModelLibrary.model_parameters[name]
        initial_values = ModelLibrary.initial_values[name]
        extended_simulator = ExtendedSimulator(bioprocess_model_class=bad_model, model_parameters=model_parameters, initial_values=initial_values)
        # These models should raise warnings for different reasons (cf. the specific model in the modelling library for details)
        with pytest.warns(UserWarning):
            model_checker.check_model_consistency(extended_simulator)

    @pytest.mark.parametrize(
        'model_variant, initial_switches, expected_behavior', 
        [
            (modelling_library.Model06, None, 'pass'),    
            (modelling_library.Model06_V02, [False]*4, 'UserWarning'),
            (modelling_library.Model06_V02, None, 'pass'),
            (modelling_library.Model06_V03, None, 'UserWarning'),
            (modelling_library.Model06_Bad01, None, 'UserWarning'),
            (modelling_library.Model06_Bad02, None, 'UserWarning'),
            (modelling_library.Model06_Bad03, None, 'NameError'),
            (modelling_library.Model06_Bad04, None, 'NameError'),
            (modelling_library.Model06_Bad05, [False]*3, 'UserWarning'),
            (modelling_library.Model06_Bad06, None, 'UserWarning'),
            (modelling_library.Model06_Bad07, None, 'UserWarning'),
            (modelling_library.Model06_Bad08, None, 'UserWarning'),
        ]
    )
    def test_model06_variants(self, model_checker, model_variant, initial_switches, expected_behavior):
        name = 'model06'
        model_parameters = ModelLibrary.model_parameters[name]
        initial_values = ModelLibrary.initial_values[name]
        extended_simulator = ExtendedSimulator(
            bioprocess_model_class=model_variant, 
            model_parameters=model_parameters, 
            initial_values=initial_values,
            initial_switches=initial_switches,
        )
        if expected_behavior == 'UserWarning':
            with pytest.warns(UserWarning):
                model_checker.check_model_consistency(extended_simulator)
        elif expected_behavior == 'NameError':
            with pytest.raises(NameError):
                model_checker.check_model_consistency(extended_simulator)
        else:
            model_checker.check_model_consistency(extended_simulator)


class TestObservationFunction():

    @pytest.mark.parametrize('model', ModelLibrary.variants_model03)
    @pytest.mark.parametrize('obsfun', ObservationFunctionLibrary.variants_obsfun01)
    def test_observation_function_checking(self, model_checker, model, obsfun):
        # Get all building blocks for the bioprocess model
        name = 'model03'
        model_parameters = ModelLibrary.model_parameters[name]
        initial_values = ModelLibrary.initial_values[name]
        # Get all buidling blocks for the observation function
        name = 'obsfun01'
        observed_state = ObservationFunctionLibrary.observed_states[name]
        observation_parameters = ObservationFunctionLibrary.observation_function_parameters[name]
        # Create an extended simulator for checking
        extended_simulator = ExtendedSimulator(
            bioprocess_model_class=model, 
            model_parameters=model_parameters, 
            initial_values=initial_values,
            observation_functions_parameters=[
                (
                    obsfun, 
                    {
                        **observation_parameters, 
                        'observed_state' : observed_state,
                    }
                ),
            ]
        )
        # These checks should not raise any warnings
        model_checker.check_model_consistency(extended_simulator)

    @pytest.mark.parametrize('model', ModelLibrary.variants_model03)
    @pytest.mark.parametrize('bad_obsfun', ObservationFunctionLibrary.bad_variants_obsfun01)
    def test_bad_observation_function_checking(self, model_checker, model, bad_obsfun):
        # Get all building blocks for the bioprocess model
        name = 'model03'
        model_parameters = ModelLibrary.model_parameters[name]
        initial_values = ModelLibrary.initial_values[name]
        # Get all buidling blocks for the observation function
        name = 'obsfun01'
        observed_state = ObservationFunctionLibrary.observed_states[name]
        observation_parameters = ObservationFunctionLibrary.observation_function_parameters[name]
        # Create an extended simulator for checking
        extended_simulator = ExtendedSimulator(
            bioprocess_model_class=model, 
            model_parameters=model_parameters, 
            initial_values=initial_values,
            observation_functions_parameters=[
                (
                    bad_obsfun, 
                    {
                        **observation_parameters, 
                        'observed_state' : observed_state,
                    }
                ),
            ]
        )
        # These checks sould raise any warnings
        with pytest.warns(UserWarning):
            model_checker.check_model_consistency(extended_simulator)
            