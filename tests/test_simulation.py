
import numpy as np
import warnings

from assimulo.solvers.sundials import CVodeError

import pytest

from pyfoomb.datatypes import ModelState
from pyfoomb.datatypes import Measurement

from pyfoomb.simulation import Simulator
from pyfoomb.simulation import ExtendedSimulator
from pyfoomb.simulation import ModelObserver

import modelling_library
from modelling_library import ModelLibrary
from modelling_library import ObservationFunctionLibrary


@pytest.fixture(params=ModelLibrary.modelnames)
def simulator(request):
    modelclass = ModelLibrary.modelclasses[request.param]
    model_parameters = ModelLibrary.model_parameters[request.param]
    initial_values = ModelLibrary.initial_values[request.param]
    return Simulator(bioprocess_model_class=modelclass, model_parameters=model_parameters, initial_values=initial_values)


@pytest.fixture(params=ModelLibrary.modelnames)
def extended_simulator(request):
    modelclass = ModelLibrary.modelclasses[request.param]
    model_parameters = ModelLibrary.model_parameters[request.param]
    initial_values = ModelLibrary.initial_values[request.param]
    return ExtendedSimulator(bioprocess_model_class=modelclass, model_parameters=model_parameters, initial_values=initial_values)


class TestSimulator():

    @pytest.mark.parametrize('name', ModelLibrary.modelnames)
    def test_init(self, name):
        modelclass = ModelLibrary.modelclasses[name]
        states = ModelLibrary.states[name]
        initial_values = ModelLibrary.initial_values[name]
        model_parameters = ModelLibrary.model_parameters[name]
        initial_switches = ModelLibrary.initial_switches[name]
        # These inits will work
        Simulator(bioprocess_model_class=modelclass, model_parameters=model_parameters, states=states)
        simulator = Simulator(bioprocess_model_class=modelclass, model_parameters=model_parameters, initial_values=initial_values)
        # Check correct autodetection of initial switches
        if initial_switches is not None:
            for _actual, _expected in zip(simulator.bioprocess_model.initial_switches, initial_switches):
                assert _actual == _expected

        # Can also provide the model parameters as list
        Simulator(bioprocess_model_class=modelclass, model_parameters=list(model_parameters.keys()), initial_values=initial_values)
        # The model parameter cannot be of other types that list or dict
        with pytest.raises(ValueError):
            Simulator(bioprocess_model_class=modelclass, model_parameters=True, states=states)
        # Must provide at least either states list of initial values dict
        with pytest.raises(ValueError):
            Simulator(bioprocess_model_class=modelclass, model_parameters=model_parameters)

    @pytest.mark.parametrize('t', [24, [0, 1, 2, 3]])
    def test_simulate(self, simulator, t):
        simulator.simulate(t=t)
        # Using unknown parameters has no effect and passes silently
        simulator.simulate(t=t, parameters={'unknown' : np.nan})
        # Integrator warnings (non-critical) which are sent to stdout are suppred by default
        simulator.simulate(t=t, suppress_stdout=False)
        
    def test_integrator_kwargs(self, simulator):
        # Must be a dict        
        with pytest.raises(ValueError):
            simulator.integrator_kwargs = ('atol', 1e-8, 'rtol' , 1e-8)
        # Tighter tolerance can lead to increased number or integrations for models with high dynamic states
        simulator.integrator_kwargs = {'atol' : 1e-14, 'rtol' : 1e-14}
        sim_lower_tols = simulator.simulate(t=1000)
        simulator.integrator_kwargs = {'atol' : 1e-2, 'rtol' : 1e-2}
        sim_higher_tols = simulator.simulate(t=1000)
        assert sim_lower_tols[0].length >= sim_higher_tols[0].length

    def test_simulator_with_observations(self):
        # Get building blocks for BioprocessModel
        name = 'model01'
        modelclass = ModelLibrary.modelclasses[name]
        initial_values = ModelLibrary.initial_values[name]
        model_parameters = ModelLibrary.model_parameters[name]
        # Get building blocks for ObservationFunctions
        obsfun_name = 'obsfun01'
        obsfun = ObservationFunctionLibrary.observation_functions[obsfun_name]
        obsfun_parameters = ObservationFunctionLibrary.observation_function_parameters[obsfun_name]
        observed_state = ObservationFunctionLibrary.observed_states[obsfun_name]
        obsfuns_params = [
            (
                obsfun, 
                {**obsfun_parameters, 'observed_state' : observed_state}
            )
        ]
        simulator = Simulator(
            bioprocess_model_class=modelclass, 
            model_parameters=model_parameters, 
            initial_values=initial_values, 
            observation_functions_parameters=obsfuns_params,
        )
        simulator.simulate(t=24)
        simulator.simulate(t=24, reset_afterwards=True)

        # The observation cannot target a not define modelstate
        with pytest.raises(ValueError):
            simulator = Simulator(
                bioprocess_model_class=modelclass, 
                model_parameters=model_parameters, 
                initial_values=initial_values, 
                observation_functions_parameters=[(obsfun, {**obsfun_parameters, 'observed_state' : 'unknown_state'})],
            )


class TestExtendedSimulator():

    @pytest.mark.parametrize('name', ModelLibrary.modelnames)
    def test_init(self, name):
        modelclass = ModelLibrary.modelclasses[name]
        initial_values = ModelLibrary.initial_values[name]
        model_parameters = ModelLibrary.model_parameters[name]
        ExtendedSimulator(bioprocess_model_class=modelclass, model_parameters=model_parameters, initial_values=initial_values)

    @pytest.mark.parametrize('t', [24, [1, 2, 3]])
    @pytest.mark.parametrize('metric', ['SS', 'WSS', 'negLL'])
    @pytest.mark.parametrize('handle_CVodeError', [True, False])
    def test_get_loss(self, extended_simulator, t, metric, handle_CVodeError):
        # Create some measurement objects from predictions, i.e. create some artifical data
        predicitions = extended_simulator.simulate(t=t)
        measurements = [
            Measurement(name=_prediction.name, timepoints=_prediction.timepoints, values=_prediction.values, errors=np.ones_like(_prediction.values))
            for _prediction in predicitions
        ]
        extended_simulator._get_loss(metric=metric, measurements=measurements)
        # Loss will be nan in case no relevant measurements are provided, i.e. measurements for states that are not defined
        assert np.isnan(
            extended_simulator._get_loss(
                metric=metric, 
                measurements=[
                    Measurement(name='y1000', timepoints=[100, 200], values=[1, 2], errors=[10, 20]),
                ]
            )
        )
        # Get loss for other parameters, as a minimizer would do several times
        _params = extended_simulator.get_all_parameters()
        different_params = {_p : _params[_p]*0.95 for _p in _params}
        extended_simulator._get_loss_for_minimzer(
            metric=metric, 
            guess_dict=different_params, 
            measurements=measurements, 
            handle_CVodeError=handle_CVodeError, 
            verbosity_CVodeError=False,
        )

    def test_with_model_enforcing_CVodeError(self):
        name = 'model06'
        modelclass = ModelLibrary.modelclasses[name]
        model_parameters = ModelLibrary.model_parameters[name]
        initial_values = ModelLibrary.initial_values[name]
        extended_simulator = ExtendedSimulator(bioprocess_model_class=modelclass, model_parameters=model_parameters, initial_values=initial_values)
        # The chosen model will create an integration error for rate = 0. A RuntimeWarning is tehrefore raised before the CVodeError is raised
        with pytest.warns(RuntimeWarning):
            with pytest.raises(CVodeError):
                extended_simulator.simulate(t=24, parameters={'rate0' : 0})
        with pytest.warns(RuntimeWarning):
            with pytest.raises(CVodeError):
                extended_simulator._get_loss_for_minimzer(
                    metric='negLL', 
                    guess_dict={'rate0' : 0},
                    measurements=[
                        Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20, 30]),
                    ],
                    handle_CVodeError=False,
                    verbosity_CVodeError=False,
                )
        # For toxic parameters causing integration errors, CVodeError handling results in inf loss
        with pytest.warns(RuntimeWarning):
            assert np.isinf(
                extended_simulator._get_loss_for_minimzer(
                    metric='negLL', 
                    guess_dict={'rate0' : 0},
                    measurements=[
                        Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20, 30]),
                    ],
                    handle_CVodeError=True,
                    verbosity_CVodeError=True,
                )
            )

    def test_extended_simulator_with_observations(self):
        # Get building blocks for BioprocessModel
        name = 'model01'
        modelclass = ModelLibrary.modelclasses[name]
        initial_values = ModelLibrary.initial_values[name]
        model_parameters = ModelLibrary.model_parameters[name]
        # Get building blocks for ObservationFunctions
        obsfun_name = 'obsfun01'
        obsfun = ObservationFunctionLibrary.observation_functions[obsfun_name]
        obsfun_parameters = ObservationFunctionLibrary.observation_function_parameters[obsfun_name]
        observed_state = ObservationFunctionLibrary.observed_states[obsfun_name]
        obsfuns_params = [
            (
                obsfun, 
                {**obsfun_parameters, 'observed_state' : observed_state}
            )
        ]

        # Set new values for parameters, using an extended simulator
        extended_simulator = ExtendedSimulator(
            bioprocess_model_class=modelclass, 
            model_parameters=model_parameters, 
            initial_values=initial_values, 
            observation_functions_parameters=obsfuns_params,
        )
        params = extended_simulator.get_all_parameters()
        extended_simulator.set_parameters({_p : params[_p]*1.05 for _p in params})

        # Get some prediction to be used as artifical data
        predicitions = extended_simulator.simulate(t=24)
        measurements = [
            Measurement(name=_prediction.name, timepoints=_prediction.timepoints, values=_prediction.values, errors=np.ones_like(_prediction.values))
            for _prediction in predicitions
        ]
        extended_simulator._get_loss(metric='negLL', measurements=measurements)


class TestModelObserver():

    @pytest.mark.parametrize('obsfun_name', ObservationFunctionLibrary.names)
    def test_init_observe(self, simulator, obsfun_name):

        obsfun = ObservationFunctionLibrary.observation_functions[obsfun_name]
        obsfun_parameters = ObservationFunctionLibrary.observation_function_parameters[obsfun_name]
        observed_state = ObservationFunctionLibrary.observed_states[obsfun_name]
        obsfuns_params = [
            (
                obsfun, 
                {**obsfun_parameters, 'observed_state' : observed_state}
            )
        ]
        observer = ModelObserver(observation_functions_parameters=obsfuns_params)
        # The observed_state must be indicated in the dictionary with observation function parameters
        with pytest.raises(KeyError):
            ModelObserver(observation_functions_parameters=[(obsfun, obsfun_parameters)])
        # Create and observe a Modelstate
        modelstate = ModelState(name=observed_state, timepoints=[1, 2, 3], values=[10, 20, 30])
        observer.get_observations(model_states=[modelstate])
        # There is also a str method
        print(observer)
        