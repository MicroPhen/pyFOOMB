import abc
import copy
import inspect
import numpy
from typing import List
import warnings

from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode

from .constants import Constants
from .constants import Messages

from .datatypes import ModelState
from .datatypes import Observation

from .utils import Helpers
from .utils import OwnDict

OBSERVED_STATE_KEY = Constants.observed_state_key


class BioprocessModel(Explicit_Problem):
    """
    The abstract base class for bioprocess models, implemented as system of ODEs. 
    Supports event handling and corresponding modification of states, parameters or whole equations.
    For integration of the ODE system, the CVode solver by Sundials is used, which is made available via the assimulo package.
    The BioprocessModel class subclasses assimulos `Explicit_Problem`, as recommended, see https://jmodelica.org/assimulo/ODE_CVode.html.
    """

    def __init__(self, model_parameters:list, states:list, initial_switches:list=None, model_name:str=None, replicate_id:str=None):
        """
        Arguments
        ---------
            model_parameters : list
                The (time-invariant) model parameters.
            states : list
                The names of the model states.

        Keyword arguments
        -----------------
            initial_switches : 
                A list of booleans, indicating the initial state of switches. 
                Number of switches must correpond to the number of return events in method `state_events`,
                if this method is implemented by the inheriting class.
                Default is None, which enables auto-detection of initial switches, which all will be False.
            model_name : str
                A descriptive model name. 
                Default is None.
            replicate_id : str
                Makes this `BioprocessModel` instance know about the `replicate_id` is is assigned to.
                Default is None, which implies a single replicate model.
        """

        self._is_init = True
        self.replicate_id = replicate_id
        self.states = states
        self.initial_values = {f'{state}0' : numpy.nan for state in self.states}
        self.model_parameters = {model_parameter : numpy.nan for model_parameter in model_parameters} 

        if initial_switches is None:
            try:
                _no_of_events = len(self.state_events(t=0, y=self.initial_values.to_numpy(), sw=None))
            except Exception as e:
                print(f'Falling back to detect number of events: {e}')
                _no_of_events = self._auto_detect_no_of_events()
                print(f'Detected {_no_of_events} events')
            self.initial_switches = [False] * _no_of_events
        else:
            self.initial_switches = initial_switches

        if model_name is not None:
            self._name = model_name
        else:
            self._name = self.__class__.__name__

        super(BioprocessModel, self).__init__(
            y0=self.initial_values.to_numpy(), 
            sw0=self.initial_switches,
            name=self._name,
        )  

        self._is_init = False 


    #%% Methods that implement the actual model

    @abc.abstractmethod
    def rhs(self, t:float, y:numpy.ndarray, sw:List[bool]) -> List[float]:
        """
        Defines the right-hand-side of the explicit ODE formulation. This method will be integrated by the solver.
        
        Arguments
        ---------
            t : float
                The current time.
            y : numpy.ndarray
                The vector holding the current values of the model states.
            sw : List[bool]
                 The current switch states. A switch is turned after its corresponding event was hit.
                 Use this argument only if the model implements events.

            NOTE: An event equals not zero at one instant timepoint, while its corresponding switch is turned afterwards, 
                  and maintains its state until the event occurs again.
            
        Returns
        -------
            List[float] or numpy.array
                The corresponding vector of derivatives for argument y. Must be the same order as `y`.
        """


    def state_events(self, t:float, y:numpy.ndarray, sw:List[bool]) -> List[float]:
        """
        Defines the roots (events) of the model states.
        An event is defined as y_i = 0, detected by an change in sign of y_i.
        
        Arguments
        ---------
            t : float
                The current time.
            y : numpy.ndarray
                The vector holding the current values of the model states.
            sw : list of bool
                The current switch states. A switch is turned after its corresponding event was hit.
                Use this argument only if the model implements events

            NOTE: An event equals not zero at one instant timepoint, while its corresponding switch is turned afterwards, 
                  and maintains its state until the event occurs again.

        Returns
        -------
            List[float] or numpy.ndarray

        Example
        -------
            # unpack the state vector for more convenient reading
            P, S, X = y 

            X_ind = self.model_parameters['X_ind']

             # event is when y[2] - X_ind = 0
            event_X = X - X_ind
            # event is hit when integration time is 20
            event_t = 20 - t 

            return [event_X, event_t]
        """

        return numpy.array([])
    

    def change_states(self, t:float, y:numpy.ndarray, sw:List[bool]) -> List[float]:
        """
        Initialize the ODE system with the new conditions, i.e. change the values of state variables 
        depending on the value of an state_event_info list (can be 1, -1, or 0).

        NOTE: This method is only called in case ANY event is hit. 
              One can filter which event was hit by evaluating `solver.sw` and `state_event_info`.

        Arguments
        ---------
            t : float
                The current time.
            y : array or list
                The vector holding the current values of the model states.
            sw : list of bool
                The current switch states. A switch is turned after its corresponding event was hit.

        Returns
        -------
            List[float] or numpy.ndarray
                The updated state vector for restart of integration.

        Example
        -------
            # Unpacks the state vector. The states are alphabetically ordered.
            A, B, C = y
        
            # Change state A when the second event is hit.
            if sw[1]:
                A_add = self.model_parameters['A_add']
                A = A + A_add
            
            return [A, B, C]
        """
        
        return y


    #%% Helper methods for handling the model implementation, need normally not to be implemented by the subclass

    def handle_event(self, solver:CVode, event_info:list):
        """
        Handling events that are discovered during the integration process.
        Normally, this method does not need to be overridden by the subclass.
        """

        state_event_info = event_info[0] # Not the 'time events', has their own method (event_info is a list)
        
        while True:
            # turn event switches of the solver instance
            self.event_switch(solver, state_event_info)
            # Collect event values before changing states
            before_mode = self.state_events(solver.t, solver.y, solver.sw)
            # Can now change the states
            solver.y = numpy.array(self.change_states(solver.t, solver.y, solver.sw))
            # Collect event values after changing states
            after_mode = self.state_events(solver.t, solver.y, solver.sw)
            event_iter = self.check_event_iter(before_mode, after_mode)
            # Check if event values have been changes because the states were changed by the user
            if not True in event_iter: # Breaks the iteration loop
                break
    

    def event_switch(self, solver:CVode, state_event_info:List[int]):
        """
        Turns the switches if a correponding event was hit.
        Helper method for method `handle_event`.

        Arguments
        ---------
            solver : CVode
                The solver instance.
            state_event_info : List[int]
                Indicates for which state an event was hit (0: no event, -1 and 1 indicate a zero crossing)
        """

        for i in range(len(state_event_info)): #Loop across all event functions
            if state_event_info[i] != 0:
                solver.sw[i] = not solver.sw[i] #Turn the switch
               
                
    def check_event_iter(self, before:List[float], after:List[float]) -> List[bool]:
        """
        Helper method for method `handle_event` to change the states at an timpoint of event.

        Arguments
        ---------
            before : List[float]
                The list of event monitoring values BEFORE the solver states (may) have been changed.
            after : List[float]
                The list of event monitoring values AFTER the solver states (may) have been changed.

        Returns
        -------
            event_iter : List[bool]
                Indicates changes in state values at corresponding positions.
        """

        event_iter = [False]*len(before)
        
        for i in range(len(before)):
            if (before[i] <= 0.0 and after[i] > 0.0) or \
                (before[i] >= 0.0 and after[i] < 0.0) or \
                (before[i] < 0.0 and after[i] >= 0.0) or \
                (before[i] > 0.0 and after[i] <= 0.0):
                event_iter[i] = True

        return event_iter


    #%% Other public methods

    def set_parameters(self, values:dict):
        """
        Assigns specfic values to the models initial values and / or model parameters.

        Arguments
        ---------
            values : dict
                Key-value pairs for parameters that are to be set.
                Keys must match the names of initial values or model parameters.

        Raises
        ------
            KeyError 
                The parameters values to be set contain a key 
                that is neither an initial value, nor a model parameter.
        """

        if not Helpers.has_unique_ids(values):
            raise KeyError(Messages.non_unique_ids)

        existing_keys = []
        existing_keys.extend(self.initial_values.keys())
        existing_keys.extend(self.model_parameters.keys())

        _initial_values = copy.deepcopy(self.initial_values)
        _model_parameters = copy.deepcopy(self.model_parameters)

        for key in values.keys():
            if key in _initial_values.keys():
                _initial_values[key] = values[key]
            if key in _model_parameters.keys():
                _model_parameters[key] = values[key]
        
        self.initial_values = _initial_values
        self.model_parameters = _model_parameters


    #%% Private methods

    def __str__(self):
        return self.__class__.__name__


    def _auto_detect_no_of_events(self) -> int:
        """
        Convenient auto-detection of event to define initial switches. 

        Returns
        -------
            no_of_events : int
                The automatically detected number of events.
                Works only for explicitly states events in the return of methods `state_events`.

        NOTE: Does not work with joblib parallel loky backend and IPython.
        """

        _lines = inspect.getsourcelines(self.state_events)
        all_in_one = ''.join(_lines[0]).replace('\n', '').replace(' ', '').replace('\\', '')
        after_return = all_in_one.split('return')[-1]

        if '[]' in after_return: # there are no detectable events
            no_of_events = 0
        else:
            # detect automatically the number of returned events by the number of commas
            final_comma = after_return.count(',]') 
            no_of_events = after_return.count(',') - final_comma + 1 
        return no_of_events


    #%% Properties
    
    @property
    def states(self):
        return self._states
    

    @states.setter
    def states(self, value:list):
        if self._is_init:
            if not Helpers.has_unique_ids(value):
                raise KeyError(Messages.non_unique_ids)
            if not isinstance(value, list):
                raise TypeError('Model states must be a list.')
            self._states = sorted(value, key=str.lower)
        else:
            raise AttributeError(f'Cannot set states after instantiation of {self.__class__.__name__}')


    @property
    def initial_values(self):
        return self._initial_values
    

    @initial_values.setter
    def initial_values(self, value):
        if not isinstance(value, dict):
            raise TypeError(Messages.invalid_initial_values_type)

        if not Helpers.has_unique_ids(value):
            raise KeyError(Messages.non_unique_ids)

        _dict = OwnDict()
        for key, state in zip(sorted(value, key=str.lower), [f'{_state}0' for _state in self._states]):
            if key != state:
                raise KeyError(f'Initial value keys must match the state names {self.states}, extended by "0"')
            _dict[key] = value[key]    
        self._initial_values = _dict
            
        # updates y0 in case the property is set after model initialization
        if not self._is_init:
            self.y0 = self.initial_values.to_numpy()


    @property
    def model_parameters(self):
        return self._model_parameters
    

    @model_parameters.setter
    def model_parameters(self, value):
        if not isinstance(value, dict):
            raise TypeError('Model parameters must be provided as dictionary')  

        if not Helpers.has_unique_ids(value):
            raise KeyError(Messages.non_unique_ids)

        if not self._is_init:
            old_keys = sorted(self.model_parameters, key=str.lower)
            new_keys = sorted(value, key=str.lower)
            if old_keys != new_keys:
                raise KeyError(f'Cannot set values for unknown parameters: {new_keys} vs. {old_keys}')

        _dict = OwnDict()
        for key in sorted(value, key=str.lower):
             _dict[key] = value[key]
        self._model_parameters = _dict


    @property
    def initial_switches(self):
        return self._initial_switches
    

    @initial_switches.setter
    def initial_switches(self, value):

        if value == [] or value is None:
            self._initial_switches = None
        else:
            if not self._is_init and len(value) != len(self.initial_switches):
                raise ValueError(f'Invalid number of initial switches provided')
            for _value in value:
                if type(_value) != bool:
                    raise ValueError('Initial switch states must be of type boolean')
            self._initial_switches = value


class ObservationFunction(abc.ABC):
    """
    Base class for observation functions that observe model states. Each model state can be observed, 
    while the mapping is described by the specific observation function with its own parameters. 
    A model state can be observed by multiple observation functions.
    """

    def __init__(self, observed_state:str, observation_parameters:list, replicate_id:str=None):
        """
        Arguments
        ---------
            observed_state : str
                The name of the model state that is observed by this object.
            observation_parameters : list
                The names of observation parameters for this ObservationFunction.

        Keyword arguments
        -----------------
            replicate_id : str
                Makes this `ObservationFunction` instance know about the `replicate_id` is is assigned to.
                Default is None, which implies a single replicate model.
        """

        self._is_init = True
        self.replicate_id = replicate_id
        self.observed_state = observed_state
        self.observation_parameters = {p : None for p in observation_parameters if p != OBSERVED_STATE_KEY}
        self._is_init = False


    #%% Public methods

    @abc.abstractmethod
    def observe(self, state_values:numpy.ndarray) -> numpy.ndarray:
        """
        Describes the mapping of model state into observation.

        Arguments
        ---------
            state_values : numpy.ndarray

        Returns
        -------
            numpy.ndarray
        """

        raise NotImplementedError('Method must be implemented by the inheriting class.')
        

    def get_observation(self, model_state:ModelState, replicate_id:str=None):
        """
        Applies the observation function on a ModelState object.

        Arguments
        ---------
            model_state : ModelState
                An instance of ModelState, as returned by the simulate method of a Simulator object, 
                together with several other ModelState objects.

        Returns
        -------
            observation : Observation
                An instance of the Observation class.

        Raises
        ------
            KeyError
                The name of the model state to be observed and `observed_state` property of this observation function do not match. 
            ValueError
                The replicate ids of this ObservationFunction object and the ModelState object to be observed do not match.
        """

        if model_state.name != self.observed_state:
            raise KeyError(f'Model state and observed state do not match. {model_state.name} vs. {self.observed_state}')
        if model_state.replicate_id != self.replicate_id:
            raise ValueError(f'Replicate ids of model state and observation functions do not match: {model_state.replicate_id} vs. {self.replicate_id}')
        observation = Observation(
            name=self.name, 
            observed_state=self.observed_state,
            timepoints=model_state.timepoints, 
            values=self.observe(model_state.values),
            replicate_id=replicate_id,
        )
        return observation 


    def set_parameters(self, parameters:dict):
        """
        Assigns specfic values to observation parameters.

        Arguments
        ---------
            values : dict
                Key-value pairs for parameters that are to be set.
                Keys must match the names of observation parameters.
        """

        _observation_parameters = copy.deepcopy(self.observation_parameters)
        for key in parameters.keys():
            if key in self.observation_parameters.keys():
                _observation_parameters[key] = parameters[key]
        self.observation_parameters = _observation_parameters


    #%% Private methods

    def __str__(self):
        return self.__class__.__name__


    #%% Properties

    @property
    def name(self):
        return self.__class__.__name__


    @property
    def observed_state(self):
        return self._observed_state


    @observed_state.setter
    def observed_state(self, value):
        if self._is_init:
            if not isinstance(value, str):
                raise ValueError(f'Bad value: {value}. Must provide a str.')
            self._observed_state = value
        else:
            raise AttributeError(f'Cannot set observed_state after instantiation of {self.__class__.__name__}')


    @property
    def observation_parameters(self):
        return self._observation_parameters
    

    @observation_parameters.setter
    def observation_parameters(self, value):
        if not isinstance(value, dict):
            raise ValueError('Observation parameters must be provided as dictionary.')  

        if not Helpers.has_unique_ids(value):
            raise KeyError(Messages.non_unique_ids)

        if not self._is_init:
            old_keys = sorted(self.observation_parameters, key=str.lower)
            new_keys = sorted(value, key=str.lower)
            if OBSERVED_STATE_KEY in new_keys:
                new_keys.remove(OBSERVED_STATE_KEY)

            if old_keys != new_keys:
                raise KeyError(f'Cannot set values for unknown parameters: {new_keys} vs. {old_keys}')

        _dict = OwnDict()
        for key in sorted(value, key=str.lower):
            if key == OBSERVED_STATE_KEY:
                if value[key] != self.observed_state:
                    raise ValueError(f'Parameter observed_state {value[key]} does not match {self.observed_state}')
            else:
             _dict[key] = value[key]
        self._observation_parameters = _dict