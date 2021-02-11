from contextlib import redirect_stdout
import copy
import io
import numpy
from typing import List

from assimulo.solvers.sundials import CVode
from assimulo.solvers.sundials import CVodeError

from .constants import Constants
from .constants import Messages

from .datatypes import Measurement
from .datatypes import ModelState
from .datatypes import Observation
from .datatypes import TimeSeries

from .modelling import BioprocessModel

from .utils import Helpers


HANDLED_CVODE_ERRORS = Constants.handled_CVodeErrors
OBSERVED_STATE_KEY = Constants.observed_state_key


class Simulator():
    """
    Manages a BioprocessModel class, as well as its corresponding ObservationFunctions, if specified. 
    Main responsibility is to carry out forward simulations for the model.
    """

    def __init__(self, 
                 bioprocess_model_class:BioprocessModel, model_parameters:dict, states:list=None, initial_values:dict=None, 
                 initial_switches:list=None, model_name:str=None,
                 observation_functions_parameters:List[tuple]=None,
                 replicate_id:str=None,
                 ):
        """
        Arguments
        ---------
            bioprocess_model_class : Subclass of BioprocessModel
                This class implements the bioprocess model.
            model_parameters : dict or list
                The model parameters, as specified in `bioprocess_model_class`.

        Keyword arguments
        -----------------
            states : list
                The model states, as specified in `bioprocess_model_class`.
                Default is none, which enforces `initial_values` not to be None.
            initial_values : dict
                Initial values to the model, keys must match the states with a trailing '0'. 
                Default is None, which enforces `states` not to be None.
            replicate_ids : list
                Unique ids of replicates, for which the full model applies. 
                The parameters, as specified for the model and observation functions are  considered as global ones, 
                which may have different names and values for each replicate.
                Default is None, which implies a single replicate model.
            initial_switches : 
                A list of booleans, indicating the initial state of switches. 
                Number of switches must correpond to the number of return events in method `state_events`,
                if this method is implemented by the inheriting class.
                Default is None, which enables auto-detection of initial switches, which all will be False.
            model_name : str
                A descriptive model name. 
                Default is None.
            observation_funtions_parameters : list of tuples
                Each tuple stores a subclass of ObservationFunction 
                and a dictionary of its correponding parametrization.
                Default is None, which implies that there are no ObservationFunctions.
        """

        # Input error checking
        if (states is None) and (initial_values is None):
            raise ValueError('Must provide at least either states or initial_values argument')

        if states is None:
            # remove the trailing 0 from initial value keys to generate the names of the states (by convention)
            self._states = [iv[0:-1] for iv in initial_values.keys()]
        else:
            self._states = states

        if isinstance(model_parameters, dict):
            self._model_parameters = model_parameters
            self._model_parameters_list = list(model_parameters.keys())
        elif isinstance(model_parameters, list):
            self._model_parameters = None
            self._model_parameters_list = model_parameters
        else:
            raise ValueError('Provided model_parameters argument must be either of type list or dict')

        # Collect further building blocks
        self._bioprocess_model_class = bioprocess_model_class
        self._initial_values = initial_values
        self._model_name = model_name
        self._initial_switches = initial_switches

        self.replicate_id = replicate_id

        # now instantiate the model
        self.bioprocess_model = self._get_bioprocess_model_instance()

        # Override with the auto-detected initial_switches property by the bioprocess_model. 
        # This is needed to correctly set this property when resetting the bioprocess_model without auto-detection of the initial switches.
        # Auto-detection is disabled when using the joblib parallel loky backend because source code inspection causes trouble then with IPython.
        self._initial_switches = self.bioprocess_model.initial_switches

        if observation_functions_parameters is None:
            self._has_observer = False
            self.observer = None
        else:
            self.observer = ModelObserver(observation_functions_parameters, replicate_id=replicate_id)
            self._has_observer = True
            # observed states must be a subset of the model states
            if not set(self.observer.observed_states).issubset(set(self.bioprocess_model.states)):
                _invalids = set(self.observer.observed_states).difference(set(self.bioprocess_model.states))
                raise ValueError(f'Observed states must be a subset of the model states. Invalid observed state detected: {_invalids}')

        self.integrator_kwargs = None


    #%% Properties

    @property
    def integrator_kwargs(self) -> dict:
        return self._integrator_kwargs


    @integrator_kwargs.setter
    def integrator_kwargs(self, value):
        if value is not None and not isinstance(value, dict):
            raise ValueError('Integrator kwargs must be dictionary or `None`')
        self._integrator_kwargs = value


    #%% Public methods

    def simulate(self, t:numpy.ndarray, parameters:dict=None, verbosity:int=30, reset_afterwards:bool=False, suppress_stdout:bool=True) -> List[TimeSeries]:
        """
        Runs a forward simulation for the fully specified model and its observation functions (if specified).

        Arguments
        ---------
            t : numpy.ndarray or float
                The time points for integration. In case a single time point is provided, 
                the solver will treat this as final integration time and chooses the intermediate steps on its own.

        Keyword arguments
        -----------------
            parameters : dict
                In case a simulation for specific parameter values is wanted. 
                Default is None.
            verbosity : int
                Prints solver statistics (quiet = 50, whisper = 40, normal = 30, loud = 20, scream = 10). 
                Default is 30.
            reset_afterwards : bool
                After simulation, reset the Simulator instance to its state directly after instantiation. 
                Useful, if parameters argument is used.
            suppress_stdout : bool
                No printouts of integrator warnings, which are directed to stdout by the assimulo package.
                Set to False for model debugging purposes.
                Default is True.

        Returns
        -------
            simulations : List[TimeSeries]
                The collection of simulations results as `ModelState` objects
                and `Observation` objects (if at least one `ObservationFunction` has been specified)

        Raises
        ------
            ValueError 
                Not all parameters have values.
        """

        if parameters is not None:
            self.set_parameters(parameters)

        # Create the solver from problem instance
        exp_sim = CVode(self.bioprocess_model)
        exp_sim.verbosity = verbosity # QUIET = 50 WHISPER = 40 NORMAL = 30 LOUD = 20 SCREAM = 10
        exp_sim.store_event_points = False
        exp_sim.discr = 'BDF'
        exp_sim.stablimit = True

        if self.integrator_kwargs is not None:
            for key in list(self.integrator_kwargs.keys()):
                exec(f'exp_sim.{key} = self.integrator_kwargs["{key}"]')

        # Do the actual forward simulation
        _t = numpy.array(t, dtype=numpy.float64).flatten()
        if len(_t) == 1:
            tfinal = float(_t)
            ncp_list = None
        elif len(_t) > 1:
            ncp_list = _t
            tfinal = numpy.max(_t)
        try:
            if suppress_stdout:
                f = io.StringIO()
                with redirect_stdout(f):
                    t, y = exp_sim.simulate(tfinal=tfinal, ncp_list=ncp_list)
            else:
                t, y = exp_sim.simulate(tfinal=tfinal, ncp_list=ncp_list)
        except CVodeError as e:
            print(f'CVodeError occured with flag {e.value}. CVodeError message was: {e}.')
            raise e

        # clean double entry artifacts due to event detection
        unq, unq_idx = numpy.unique(t, return_index=True)

        # build model prediction dictionary
        model_predictions = [
            ModelState(
                name=_name, 
                timepoints=numpy.array(t)[unq_idx], 
                values=numpy.array(_y)[unq_idx], 
                replicate_id=self.replicate_id,
                ) 
            for _name, _y in zip(self.bioprocess_model.states, y.T)
        ]

        if self.observer is not None:
            observations = self.observer.get_observations(model_predictions)
        else:
            observations = []

        if reset_afterwards:
            self.reset()

        simulations = []
        simulations.extend(model_predictions)
        simulations.extend(observations)

        return simulations


    def set_parameters(self, parameters:dict):
        """
        Assigns specfic values to initial values, model parameters and/or observation parameters.

        Arguments
        ---------
            values : dict
                Key-value pairs for parameters that are to be set.
                Keys must match the names of initial values, model parameters, observation parameters.
        """

        self.bioprocess_model.set_parameters(parameters)
        if self._has_observer:
            self.observer.set_parameters(parameters)


    def reset(self):
        """
        Resets the model and observer (if specified) objects to their states after instantiation of the Simulator class.
        """

        self.bioprocess_model = self._get_bioprocess_model_instance()
        if self._has_observer:
            self.observer.reset()


    #%% Private methods

    def _get_bioprocess_model_instance(self) -> BioprocessModel:
        """
        Creates an instance of the BioprocessModel class, needed to run forward simulations.
        """

        bioprocess_model = self._bioprocess_model_class(
            model_parameters=self._model_parameters_list, 
            states=self._states, 
            initial_switches=self._initial_switches, 
            model_name=self._model_name,
            replicate_id=self.replicate_id,
        )
        if self._initial_values is not None:
            bioprocess_model.initial_values = self._initial_values
        if self._model_parameters is not None:
            bioprocess_model.model_parameters = self._model_parameters
        return bioprocess_model


class ExtendedSimulator(Simulator):

    def __init__(self, 
                 bioprocess_model_class, model_parameters, states:list=None, initial_values:dict=None, 
                 initial_switches:list=None, model_name:str=None, 
                 observation_functions_parameters:List[tuple]=None,
                 replicate_id:str=None,
                 ):
        """
        Arguments
        ---------
            bioprocess_model_class : Subclass of BioprocessModel
                This class implements the bioprocess model.
            model_parameters : list or dict
                The model parameters, as specified in `bioprocess_model_class`.

        Keyword arguments
        -----------------
            states : list
                The model states, as specified in `bioprocess_model_class`.
                Default is None, which enforces `initial_values` not to be None.
            initial_values : dict
                Initial values to the model, keys must match the states with a trailing '0'. 
                Default is None, which enforces `states` not to be None.
            replicate_ids : list
                Unique ids of replicates, for which the full model applies. 
                The parameters, as specified for the model and observation functions are  considered as global ones, 
                which may have different names and values for each replicate.
                Default is None, which implies a single replicate model.
            initial_switches : 
                A list of booleans, indicating the initial state of switches. 
                Number of switches must correpond to the number of return events in method `state_events`,
                if this method is implemented by the inheriting class.
                Default is None, which enables auto-detection of initial switches, which all will be False.
            model_name : str
                A descriptive model name. 
                Default is None.
            observation_funtions_parameters : list of tuples
                Each tuple stores a subclass of ObservationFunction 
                and a dictionary of its correponding parametrization.
                Default is None, which implies that there are no ObservationFunctions.
        """

        super().__init__(
            bioprocess_model_class, 
            model_parameters, 
            states, 
            initial_values, 
            initial_switches, 
            model_name, 
            observation_functions_parameters, 
            replicate_id,
        )
        

    #%% Public methods

    def get_all_parameters(self) -> dict:
        """
        Retrieve all parameters (model parameters, initial values, observation parameters)
        """

        _parameters = {}
        _parameters.update(self.bioprocess_model.model_parameters)
        _parameters.update(self.bioprocess_model.initial_values)
        if self._has_observer:
            _parameters.update(self.observer.observation_parameters)
        return {p : _parameters[p] for p in sorted(_parameters.keys(), key=str.lower)}


    #%% Private methods

    def _get_loss(self, metric:str, measurements:List[Measurement], parameters:dict=None) -> float:
        """
        Arguments
        ---------
            metric : str
                The metric according to which the loss will be calculated.

            measurements : List[Measurement]
                The loss will be calculated for all given Measurement objects.

        Keyword arguments
        -----------------
            parameters : dict
                Model predictions are created using this set of parameter values.
                Default is None.

        Returns
        -------
            float
                The loss of all measurements, according to the specified metric.
        """

        # Collect all timepoints for any measurements
        t = Helpers.get_unique_timepoints(measurements)
        allowed_keys = self._get_allowed_measurement_keys()

        _relevant_measurements = [
            Helpers.extract_time_series(measurements, name=_name, replicate_id=self.replicate_id) 
            for _name in allowed_keys
        ]
        relevant_measurements = [
            _item for _item in _relevant_measurements 
            if _item is not None
        ]
        if relevant_measurements == []:
            return numpy.nan

        # Set parameters and run forward simulation
        simulations = self.simulate(t=t, parameters=parameters, reset_afterwards=False, verbosity=50)
        losses = [relevant_measurement.get_loss(metric, simulations) for relevant_measurement in relevant_measurements]

        return numpy.nansum(losses)


    def _get_loss_for_minimzer(self, 
                               metric:str, guess_dict:dict, measurements:List[Measurement], 
                               handle_CVodeError:bool, verbosity_CVodeError:bool,
                               ) -> float: 
        """
        The objective function for parameter estimation called by the optimizer. 
        The main purpose is to handle integrations errors that may arise form toxic parameter values set by the optimizer.

        Arguments
        ---------
            metric : str
                Defines how the loss is to be calculated. 
                Currently, valid metric are `negLL` (negative log likelihood), `SS` (sum of squares), `WSS` (weighted sum of squares).
            guess_dict : dict
                The current parameter values set by the minimizer.
            measurements : List[Measurement]
                The data, from which the parameter estimation is performed. 
                Can provide a Measurement object for any model state or observation.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
            verbosity_CVodeError : bool
                Enables informative output during handling CVodeErrors. 

        Returns
        -------
            loss : float
                This value is to be minimized by the optimizer, as the loss describes the distance between a model prediction and the given data.

        Raises
        ------
            CVodeError
                All of them, except for certain flags if these are handled.
        """

        if handle_CVodeError:
            try:
                loss = self._get_loss(
                    metric=metric,
                    measurements=measurements, 
                    parameters=guess_dict,
                )
            except CVodeError as e:
                if e.value in HANDLED_CVODE_ERRORS:
                    loss = numpy.inf
                    if verbosity_CVodeError:
                        print(f'CVodeError occured with flag {e.value}: Setting loss {loss}. Apparent toxic parameters are {guess_dict}')
                else:
                    print(f'CVodeError occured with flag {e.value}. CVodeError message was: {e}. Apparent toxic parameters are {guess_dict}')
                    raise e
        else:
            loss = self._get_loss(
                    metric=metric,
                    measurements=measurements, 
                    parameters=guess_dict,
            )
        return loss


    def _get_allowed_measurement_keys(self) -> list:
        """
        Returns the names of model states and observations (if specified), 
        to evaluate against the keys of the Measurement objects, which are used for parameter estimation.

        Returns
        -------
            list
        """

        _allowed_measurement_keys = []
        _allowed_measurement_keys.extend(self.bioprocess_model.states)
        if self._has_observer:
            _allowed_measurement_keys.extend(self.observer.observables)
        return _allowed_measurement_keys


class ModelObserver():
    """
    Manages a collection of ObservationFunction objects and their corresponding observation parameters.
    """
    
    def __init__(self, observation_functions_parameters:List[tuple], replicate_id:str=None):
        """
        Arguments
        ---------
            observation_functions_parameters : List[tuple])
                Tuples of the ObservationFunction subclasses 
                and dicts with corresponding observation parameters

        Keyword arguments
        -----------------
            replicate_id : str
                Makes this `ModelObserver` instance know about the `replicate_id` is is assigned to.
                Will be forwarded to all `ObervationFunction` instances.
                Default is None, which implies a single replicate model.

        Raises
        ------
            ValueError
                Items of `observation_function_parameters` are not tuples.
            TypeError
                The second item in any of the above tuples is not a dictionary.
            KeyError
                Any observation parameters dict does not indicate the observed state of the corresponding ObservationFunction object.
            KeyError
                There are non-unique keys among all observation parameters dictionaries.
        """

        self._first_init = True
        self.replicate_id = replicate_id
        self._observation_functions_parameters = observation_functions_parameters
        self.observation_functions = {}
        self._obs_pars_names = []
        self.observation_parameters = {}
        self.observed_states = []
        for _obs in self._observation_functions_parameters:
            if not isinstance(_obs, tuple):
                raise ValueError('Must provide a list of tuples containing ObservationFunction subclasses and dicts with corresponding observation parameters')
            if not isinstance(_obs[1], dict):
                raise TypeError(f'Second item of {_obs} must be a dictionary holding the observation parameters')
            if OBSERVED_STATE_KEY not in list(_obs[1].keys()):
                raise KeyError(f'Observation parameters dictionary for {_obs[0]} indicated not the observed state')

            _obs_fun = _obs[0]
            _obs_pars = _obs[1]
            self.observation_functions[f'{_obs_fun.__name__}'] = _obs_fun(
                observed_state=_obs_pars[OBSERVED_STATE_KEY], 
                observation_parameters=list(_obs_pars.keys()), 
                replicate_id=self.replicate_id,
            )
            self.observation_functions[f'{_obs_fun.__name__}'].set_parameters(_obs_pars)
            self.observed_states.append(self.observation_functions[f'{_obs_fun.__name__}'].observed_state)
            self._obs_pars_names.extend([_p for _p in _obs_pars.keys() if _p != OBSERVED_STATE_KEY])
            self.observation_parameters.update({_p : _obs_pars[_p] for _p in _obs_pars.keys() if _p != OBSERVED_STATE_KEY})
            # check that all observation parameters have unique names
            if not Helpers.has_unique_ids(self._obs_pars_names):
                raise KeyError(Messages.non_unique_ids)

        self._lookup = {}
        for observed_state in self.observed_states:
            self._lookup[observed_state] = []
            for obs_fun in self.observation_functions.keys():
                if self.observation_functions[obs_fun].observed_state == observed_state:
                    self._lookup[observed_state].append(self.observation_functions[obs_fun])

        self.observables = list(self.observation_functions.keys())
        self._first_init = False


    #%% Public methods

    def get_observations(self, model_states:List[ModelState]) -> List[Observation]:
        """
        Applies the observation functions on a dictionary of ModelState objects. 

        Arguments
        ---------
            model_states : List[ModelState]
                Several ModelState object to be observed. 
                Not all of them must be necessarily observed.

        Returns
        -------
            observations : List[Observation]
                A dictionary of observation_name:Observation for all observed model states.

        Raises
        ------
            KeyError
                There are non-unique model states to be observed 
        """

        state_list = set([model_state.name for model_state in model_states])

        if not Helpers.has_unique_ids(state_list):
            raise KeyError(Messages.non_unique_ids)

        observations = []
        for state in state_list:
            if state in self.observed_states:
                model_state = Helpers.extract_time_series(model_states, name=state, replicate_id=self.replicate_id)
                _obs_fun_list = self._lookup[state]
                for _obs_fun in _obs_fun_list:
                    observation = _obs_fun.get_observation(model_state, replicate_id=self.replicate_id)
                    observations.append(observation)
        return observations


    def reset(self):
        """
        Resets the observer object to its state after instantiation.
        """

        observation_functions_parameters = self._observation_functions_parameters
        self.__init__(observation_functions_parameters, replicate_id=self.replicate_id)


    def set_parameters(self, parameters:dict):
        """
        Assigns specfic values to observation parameters.

        Arguments:
            values (dict) : Key-value pairs for parameters that are to be set.
                Keys must match the names of observation parameters.
        """

        for obs_fun in self.observation_functions.keys():
            self.observation_functions[obs_fun].set_parameters(parameters)


    #%% Private methods

    def __str__(self):
        return self.__class__.__name__
