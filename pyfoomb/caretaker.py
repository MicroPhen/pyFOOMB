import copy
import gc
import numpy
from numpy.linalg import LinAlgError
import joblib
import pandas
import psutil
import pygmo
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time
from typing import Dict, List, Tuple
import warnings

from .constants import Constants
from .constants import Messages

from .datatypes import Measurement
from .datatypes import Sensitivity

from .generalized_islands import ArchipelagoHelpers
from .generalized_islands import LossCalculator
from .generalized_islands import ParallelEstimationInfo

from .model_checking import ModelChecker

from .oed import CovOptimality

from .parameter import Parameter
from .parameter import ParameterMapper
from .parameter import ParameterManager

from .simulation import ExtendedSimulator

from .utils import Calculations
from .utils import OwnDict
from .utils import Helpers


EPS64 = Constants.eps_float64
PRETTY_METRICS = Constants.pretty_metrics
SINGLE_ID = Constants.single_id


class Caretaker():
    """
    Manages (takes care of) all major methods related to simulation, estimation, 
    and evaluation of dynamic bioprocess models and its observation functions.
    Exposes several convient methods of the individual classes in this module.
    """

    def __init__(self, 
                 bioprocess_model_class, model_parameters, states:list=None, initial_values:dict=None, replicate_ids:list=None,
                 initial_switches:list=None, model_name:str=None, observation_functions_parameters:List[tuple]=None,
                 model_checking_assistance:bool=True,
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
                Default is none, which enforces `initial_values` not to be None.
            initial_values : dict
                Initial values to the model, keys must match the states with a trailing '0'. 
                Default is None, which enforces `states` not to be None.
            replicate_ids : list
                Unique ids of replicates, for which the full model applies. 
                The parameters, as specified for the model and observation functions are  considered as global ones, 
                which may have different names and values for each replicate.
                Default is None, which implies a single replicate model.
            initial_switches : list
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
            model_checking_assistance : bool
                Runs a few sanity and call checks on the implemented model
        """

        # store arguements for later use
        self.__bioprocess_model_class = bioprocess_model_class
        self.__model_parameters = model_parameters
        self.__states = states
        self.__initial_values = initial_values
        self.__replicate_ids = replicate_ids
        self.__initial_switches = initial_switches
        self.__model_name = model_name
        self.__observation_functions_parameters = observation_functions_parameters

        self.replicate_ids = replicate_ids

        if model_name is None:
            self.model_name = bioprocess_model_class.__name__
        else:
            self.model_name = model_name

        # Create an ExtendendSimulator instance for each replicate id

        model_checker = ModelChecker()
        self.simulators = {}
        for _replicate_id in self.replicate_ids:
            if _replicate_id is None:
                _model_name = model_name
            else:
                _model_name = f'{model_name}_{_replicate_id}'

            _simulator = ExtendedSimulator(
                bioprocess_model_class,
                model_parameters, 
                states, 
                initial_values, 
                initial_switches, 
                _model_name, 
                observation_functions_parameters,
                _replicate_id,
            )
            if model_checking_assistance:
                if not model_checker.check_model_consistency(copy.deepcopy(_simulator)):
                    warnings.warn(f'There might by some issues for {_model_name} with replicate_id {_replicate_id}')
            self.simulators[_replicate_id] = _simulator

        # Create a ParameterManager object
        self._parameter_manager = ParameterManager(
            self.replicate_ids, 
            self.simulators[self.replicate_ids[0]].get_all_parameters(),
        )

        self.optimizer_kwargs = None


    #%% Properties

    @property
    def replicate_ids(self) -> list:
        return self._replicate_ids

    @replicate_ids.setter
    def replicate_ids(self, value):
        if value is None:
            self._replicate_ids = [SINGLE_ID]
        else:
            if not Helpers.has_unique_ids(value):
                raise ValueError(Messages.non_unique_ids)
            self._replicate_ids = value

    @property
    def parameter_mapping(self):
        return self._parameter_manager.parameter_mapping


    @property
    def optimizer_kwargs(self) -> dict:
        return self._optimizer_kwargs


    @optimizer_kwargs.setter
    def optimizer_kwargs(self, value):
        if value is not None and not isinstance(value, dict):
            raise ValueError('Optimizer kwargs must be either `None` or a dictionary')
        self._optimizer_kwargs = value


    #%% Public methods

    def add_replicate(self, replicate_id:str, mappings:List[ParameterMapper]=None):
        """
        Adds another replicate to the multi model Caretaker object.

        Arguments
        ---------
            replicate_id : str
                The new replicate_id to be added.

        Keyword arguments
        -----------------
            mappings : list of ParameterMapper or tuple
                A list parameter mappings that should be applied to the new replicate_id.
                Default is None, which implies that the local parameters names for the new replicate correspond to the global names.

        Raises
        ------
            AttributeError
                In case the Caretaker object was created without explicit `replicate_ids` argument.
            ValueError
                The new replicate_id is not unique including the extisting ones.
            KeyError
                Any of the `mappings` aims not for the new replicate_id.
        """
        
        # store current parameter mapping
        _parameter_mappers = self._parameter_manager.get_parameter_mappers()
        _parameters = self._get_all_parameters()
        if len(self.replicate_ids) == 1 and self.replicate_ids[0] is None:
            raise AttributeError('Cannot add replicate_id to implicitly defined single replicate Caretaker object')

        _updated_replicate_ids = copy.deepcopy(self.replicate_ids)
        _updated_replicate_ids.append(replicate_id)
        if not Helpers.has_unique_ids(_updated_replicate_ids):
                raise ValueError(Messages.non_unique_ids)

        if mappings is not None:
            for _mapping in mappings:
                if _mapping.replicate_id != replicate_id:
                    raise KeyError('The given mapping does not aim for the new replicate')

        self.__init__(
            bioprocess_model_class=self.__bioprocess_model_class, 
            model_parameters=self.__model_parameters, 
            states=self.__states, 
            initial_values=self.__initial_values, 
            replicate_ids=_updated_replicate_ids, 
            initial_switches=self.__initial_switches, 
            model_name=self.__model_name, 
            observation_functions_parameters=self.__observation_functions_parameters,
        )
        self.set_parameters(_parameters)
        self.apply_mappings(_parameter_mappers)
        if mappings is not None:
            self.apply_mappings(mappings)


    def simulate(self, t:numpy.ndarray, parameters:dict=None, verbosity:int=40, reset_afterwards:bool=False, suppress_stdout:bool=True) -> list:
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
            suppress_stdout : bool
                No printouts of integrator warnings, which are directed to stdout by the assimulo package.
                Set to False for model debugging purposes.
                Default is True.

        Returns
        -------
            simulations : list
                The collection of simulations results as list of ModelState or Observation objects. 
        """

        if parameters is not None:
            _original_parameters = self._get_all_parameters()
            self.set_parameters(parameters)
        
        simulations = []
        for _id in self.simulators.keys():
            _simulator = self.simulators[_id]
            simulations.extend(_simulator.simulate(t=t, verbosity=verbosity, reset_afterwards=reset_afterwards, suppress_stdout=suppress_stdout))

        if parameters is not None:
            self.set_parameters(_original_parameters)

        return simulations


    def estimate(self, 
                 unknowns:dict, measurements:List[Measurement], bounds:List[Tuple]=None, metric:str='negLL', use_global_optimizer:bool=None,
                 report_level:int=0, reset_afterwards:bool=False, handle_CVodeError:bool=True, optimizer_kwargs:dict=None,
                 ) -> Tuple[dict, dict]:
        """
        Estimates values for requested unknowns according to a specific metric, given some measurements.

        Arguments
        ---------
            unknowns : dict or list
                The parameters to be estimated. Can be any of the model parameters, initial values or observation parameters. 
                Providing a list of valid unknowns causes the use of scipy's differential evolution global optimizer. 
                A dictionary with parameter:guess as key-value pairs is needed to use the local but faster minimizer.
            measurements : List[Measurement]
                The data from which the parameter estimation is performed. 
                Can provide a Measurement object for any model state or observation.

        Keyword arguments
        -----------------
            bounds : list of tuples
                Bounds for for each unknown to be estimated.
                Must be provided for use with differential evolution optimizer. 
                Default is None.
            metric : str
                The metric according to which the loss to be minimized is calculated. 
                Can be one of, e.g. `negLL` (negative log-likelihood), 'SS' (sum of squares), or `WSS` (weighted sum of squares).
                Default is `negLL`, which implies that the corresponding Measurement object are accordingly specified.
            use_global_optimizer : bool
                Enforce the use of differential evolution optimizer. 
                Default is None, which makes this decision based on the the type of `unknowns` and `bounds`. 
            report_level : int
                Enables informative output about the estimation process. 
                Default is 0, which is no output.
                1 = prints estimated parameters and runtime of the estimation job.
                2 = prints additionally the `OptimizeResult` result object, as returned by the optimizer
                3 = prints additionally handled CVodeErrors, which arise from toxic parameters. 
                    This has only effect in case `handle_CVodeError` is True
                4 = prints additionally the progress of the optimizer.
            reset_afterwards : bool
                To reset the Caretaker object after the estimation has finished. 
                Default is False.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.

        Returns
        -------
            estimations : dict
                Key-value pairs of the unknowns and corresponding estimated values.
            estimation_info : dict
                Additional information about the estimation job.

        Raises
        ------
            KeyError
                Non-unique unknowns are provided.
            ValueError
                Invalid unknowns shall be estimated.
            ValueError
                No bounds are provided for use of differential evolution optimizer.
            TypeError
                A list containing not only Measurement objects is provided.

        Warns
        -----
            UserWarning
                The `optimizer_kwargs` argument is not None.
        """

        for _item in measurements:
            if not isinstance(_item, Measurement):
                raise TypeError('Must provide a list of Measurement objects')

        _start = time.time()

        if not Helpers.has_unique_ids(unknowns):
            raise KeyError(Messages.bad_unknowns)

        # test if parameters are estimated that are not known to the model
        _valid_unknowns = self._get_valid_parameter_names()
        for _unknown in unknowns:
            if _unknown not in _valid_unknowns:
                raise ValueError(f'Detected invalid unknown to be estimated: {_unknown} vs. {_valid_unknowns}')

        if use_global_optimizer == False and not isinstance(unknowns, dict):
            raise ValueError('Must provide initial guesses to use the local minimizer')

        # sort unknowns and corresponding bounds alphabetically and case-insensitive
        # Decide also whether to use the local or global minimizer
        _unknowns_names_sorted = sorted(unknowns, key=str.lower)
        if isinstance(unknowns, dict):
            _unknowns = {_unknown_name : unknowns[_unknown_name] for _unknown_name in _unknowns_names_sorted}
        elif isinstance(unknowns, list):
            _unknowns = {_unknown_name : None for _unknown_name in _unknowns_names_sorted}
            if use_global_optimizer is None:
                use_global_optimizer = True

        if use_global_optimizer and bounds is None:
            raise ValueError(Messages.missing_bounds)

        if bounds is not None:
            try:
                _bounds = [bounds[unknowns.index(_unknown_name)] for _unknown_name in _unknowns_names_sorted]
            except AttributeError:
                _bounds = [bounds[list(unknowns.keys()).index(_unknown_name)] for _unknown_name in _unknowns_names_sorted]
            # To protect for integration error when a bound is integer 0
            _bounds = Helpers.bounds_to_floats(_bounds)
        else:
            _bounds = None
        
        # Check for optimizer_kwargs to be used
        _optimizer_kwargs = {}
        if self.optimizer_kwargs is None:
            _warning_flag = False
        else:
            _optimizer_kwargs.update(self.optimizer_kwargs)
            _warning_flag = True

        # check if the keyword argument `optimizer_kwargs` was set, which has a higher priority over the corresponding Caretaker property
        if optimizer_kwargs is not None:
            _optimizer_kwargs = optimizer_kwargs
            if _warning_flag:
                warnings.warn('Using the `optimizer_kwargs` keyword argument overrides the Caretaker property `optimizer_kwargs`.', UserWarning)

        if report_level >= 3:
            verbosity_CVodeError = True
        else:
            verbosity_CVodeError = False

        if report_level >= 4 and 'disp' not in _optimizer_kwargs.keys():
            _optimizer_kwargs['disp'] = True

        if use_global_optimizer:
            minimizer_scope = 'differential evolution optimizer'
            if 'popsize' not in _optimizer_kwargs.keys():
                popsize = 5*len(_unknowns)
                _optimizer_kwargs['popsize'] = popsize

            opt = differential_evolution(self._loss_fun_scipy, 
                                         bounds=_bounds, 
                                         args=(_unknowns, 
                                               metric,
                                               measurements, 
                                               handle_CVodeError, 
                                               verbosity_CVodeError, 
                                               ), 
                                         **_optimizer_kwargs,
                                         )
        else:
            minimizer_scope = 'local minimizer'
            if 'disp' in _optimizer_kwargs.keys():
                options = {'disp' : _optimizer_kwargs['disp']}
                del _optimizer_kwargs['disp']
                _optimizer_kwargs['options'] = options

            opt = minimize(self._loss_fun_scipy, 
                           list(_unknowns.values()), 
                           args=(_unknowns, 
                                 metric,
                                 measurements, 
                                 handle_CVodeError, 
                                 verbosity_CVodeError, 
                                 ), 
                           bounds=_bounds,
                           **_optimizer_kwargs,
                           )

        # Preparing returns
        estimations = {_unknown : value for _unknown, value in zip(_unknowns, opt.x)}
        
        estimation_info = {}
        estimation_info['opt_info'] = opt
        if metric in list(PRETTY_METRICS.keys()):
            estimation_info['metric'] = PRETTY_METRICS[metric]
        else:
            estimation_info['metric'] = metric
        estimation_info['loss'] = opt.fun
        _end = time.time()
        estimation_info['runtime_min'] = (_end - _start)/60

        if report_level >= 1:
            print(f'\n----------Results from {minimizer_scope}')
            print('\nEstimated parameters:')
            for estimation in estimations.keys():
                print(f'{estimation}: {estimations[estimation]}')
            print(f'\nRuntime was {estimation_info["runtime_min"]:.2f} min')

        if report_level >= 2:
            print('\n----------')
            print(opt)

        return estimations, estimation_info


    def estimate_parallel(self, 
                          unknowns:list, measurements:List[Measurement], bounds:List[Tuple],
                          metric:str='negLL', report_level:int=0,
                          optimizers:List[str]='de1220', optimizers_kwargs:List[dict]={}, log_each_nth_gen:int=None,
                          rel_pop_size:float=10.0, evolutions:int=5, archipelago_kwargs:dict={},
                          atol_islands:float=None, rtol_islands:float=1e-6, 
                          max_runtime_min:float=None,
                          max_evotime_min:float=None,
                          max_memory_share:float=0.95,
                          handle_CVodeError:bool=True,
                          loss_calculator:LossCalculator=LossCalculator,
                          ) -> Tuple[dict, ParallelEstimationInfo]:
        """
        Estimates values for requested unknowns according to a specific metric, given some measurements, 
        using the generalized island model for parallelization that allows for global optimization. 
        This is provided by the pygmo package, which runs parallel evolutions of populations, 
        with migration of improved variants between the populations occuring. 
        For further info and use of pygmo, see https://github.com/esa/pygmo2, doi:10.5281/zenodo.3603747.

        Arguments
        ---------
            unknowns : dict or list
                The parameters to be estimated. Can be any of the model parameters, initial values or observation parameters. 
            measurements : List[Measurement]
                The data from which the parameter estimation is performed. 
                Can provide a Measurement object for any model state or observation.
            bounds : list of tuples
                Bounds for for each unknown to be estimated, in the following form [(lower, upper), ...]

        Keyword arguments
        ----------------
            metric : str
                The metric according to which the loss to be minimized is calculated. 
                Can be one of, e.g. `negLL` (negative log-likelihood), `SS` (sum of squares), or `WSS` (weighted sum of squares).
                Default is `negLL`, which implies that the corresponding Measurement object are accordingly specified.
            report_level : int
                Enables informative output about the estimation process. Information will be printed after each evolution.
                Default is 0, which is no output.
                1 = Prints the best loss, as well as information about archipelago creation and evolution. 
                    For each completed evolution, a dot is printed.
                2 = Prints additionally the best loss after each evolution
                3 = Prints additionally average loss among all islands, and the runtime of each evolution.
                4 = Prints additionally the parameter values for the best loss, and the average parameter values 
                    among the champions of all islands in the `archipelago` after the evolutions. 
            optimizers : List[str] or str
                A list of names for the pygmo optimization algorithms of choice. For a list of such to be conveniently used, 
                see `PygmoOptimizers` class of this module. 
                In case a list with one item is used, this optimizer is used for all explicitly 
                or implicitly (default None of `n_islands`) defined nunmber islands.
                In case a list with >1 optimizers is used, the corresponding number of islands will be created within the archipelago.
                The currently supported list of optimizer can be found at pyfoomb.generalized_islands.PygmoOptimizers.optimizers
                Default is `de1220`, which makes each island to use this algorithm.
            optimizers_kwargs : List[dict] or dict
                A list of optimizer_kwargs as dicts, corresponding to the list of optimizers. 
                In case more >1 optimizers are used, the 1-item list of optimizer_kwargs will be applied to all of the optimizers.
                Default is `[{}]`, i.e. no additional optimizer kwargs.
            log_each_nth_gen : int
                Specifies at which each n-th generation the algorithm stores logs. 
                Can be later extracted from the returned `archipelago` instance.
                Note that only the log from the last evolution is stored in the archipelago.
                Default is None, which disables logging.
            rel_pop_size : float
                Determines the population size on each island, relative to the number of unknown to be estimated, 
                i.e. pop_size = rel_pop_size * len(unknowns), rounded to the next integer.
                Default is 10, which creates population sizes 10 times the number of unknowns.
            evolutions : int
                Defines how often the populations on the islands are evolved. 
                Migrations between the populations of the islands occur after each finished evolution. 
                Migration depends of the topology of the archipelago, as well as the defined migration polices, 
                which are parts of `archipelago_kwargs`.
                Default is 5, which triggers five rounds of evolution.
            archipelago_kwargs : dict
                The keyword arguments for instantiation of the archipelago.
                In case `archipelago_kwargs` has no key `t`, the `pygmo.fully_connected()` topology will be used 
                Default is {}, i.e. an empty dictionary, which implies the use of `pygmo.fully_connected()` topology.
            atol_islands : float
                Defines a stopping criterion that is checked after each evolution.
                If the std of the islands' losses < atol_islands + rtol_islands * abs(mean(islands' losses)), then the optimization is stopped.
                Default is None, which implies no effect for this argument.
            rtol_islands : float
                Defines a stopping criterion that is checked after each evolution.
                If the std of the islands' losses < atol_islands + rtol_islands * abs(mean(islands' losses)), then the optimization is stopped.
                Default is 1e-6.
            max_runtime_min : float
                The maximun time in min the estimation process may take. The current runtimee is evaluated after each completion of an evolution.
                Default is None, which implies there is no maximum runtime for the estimation process.
            max_evotime_min : float
                The maximum cumulative pure evolution time the estimation process is allowed to take.
                In contrast to the `max_runtime_min` stopping criterion, only the evolution runtime is considered, 
                without runtime needed for checking stopping criteria, reporting prints outs between each evolution, etc.
                Default is None.
            max_memory_share : float
                Defines the allowed memory share in usage, for which no evolutions are run anymore.
                Default is 0.95, meaning that repeat are only run if used memory share is less than 95 %.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.
            loss_calculator : LossCalculator
                By subclassing `LossCalculator`, user-defined constraints can be implemented. The resulting subclass needs to be provided.
                Default is LossCalculator, which implements no additional constraints

        Returns
        -------
            best_estimates : dict
                The estimated parameters, according to the best champion among all populations of the archipelago, aftet the last evolution.
            estimation_result : ParallelEstimationInfo
                Stores the archipelago and the history of the previous evolutions.
                Needed to continue an estimation process.

        Raises
        ------
            KeyError
                Non-unique unknowns are provided.
            ValueError
                Invalid unknowns shall be estimated.
        """

        _start = time.time()

        if len(unknowns) != len(bounds):
            raise ValueError('Number of unkowns does not match number of pairs of upper and lower bounds.')

        if not isinstance(optimizers, list) and isinstance(optimizers, str):
            optimizers = [optimizers]
        if not isinstance(optimizers_kwargs, list) and isinstance(optimizers_kwargs, dict):
            optimizers_kwargs = [optimizers_kwargs]

        if not Helpers.has_unique_ids(unknowns):
            raise KeyError(Messages.bad_unknowns)

        # test if parameters are estimated that are not known to the model
        _valid_unknowns = self._get_valid_parameter_names()
        for _unknown in unknowns:
            if _unknown not in _valid_unknowns:
                raise ValueError(f'Detected invalid unknown to be estimated: {_unknown} vs. {_valid_unknowns}')

        # sort unknowns and corresponding bounds 
        _unknowns_names_sorted = sorted(unknowns, key=str.lower)
        _unknowns = [_unknown_name for _unknown_name in _unknowns_names_sorted]
        _bounds = [bounds[unknowns.index(_unknown_name)] for _unknown_name in _unknowns_names_sorted]

        if report_level >= 5:
            _verbosity_CVodeError = True
        else:
            _verbosity_CVodeError = False

        # get the problem
        pg_problem = pygmo.problem(
            loss_calculator(
                unknowns=_unknowns, 
                bounds=_bounds, 
                metric=metric, 
                measurements=measurements, 
                caretaker_loss_fun=self.loss_function, 
                handle_CVodeError=handle_CVodeError, 
                verbosity_CVodeError=_verbosity_CVodeError,
            ),
        )

        # get the archipelago
        archipelago = ArchipelagoHelpers.create_archipelago(
            _unknowns, 
            optimizers, 
            optimizers_kwargs, 
            pg_problem, 
            rel_pop_size, 
            archipelago_kwargs, 
            log_each_nth_gen, 
            report_level, 
            )
        archipelago.problem = loss_calculator

        estimation_result = ParallelEstimationInfo(archipelago=archipelago)

        return self.estimate_parallel_continued(
            estimation_result=estimation_result, 
            evolutions=evolutions,
            report_level=report_level, 
            atol_islands=atol_islands, 
            rtol_islands=rtol_islands, 
            max_runtime_min=max_runtime_min, 
            max_evotime_min=max_evotime_min,
            max_memory_share=max_memory_share,
            start_time=_start,
        )


    def estimate_parallel_continued(self, 
                                    estimation_result:ParallelEstimationInfo, evolutions:int=1, report_level:int=0, 
                                    atol_islands:float=None, rtol_islands:float=1e-6, 
                                    max_runtime_min:float=None, 
                                    max_evotime_min:float=None,
                                    max_memory_share:float=0.95,
                                    start_time:float=None,
                                    ) -> Tuple[dict, ParallelEstimationInfo]:
        """
        Continues a parallel parameter estimation job by running more evolutions on a corresponding archipelago object. 

        Arguments
        ---------
            estimation_result : ParallelEstimationInfo
                Stores the archipelago and the history of the previous evolutions as returned by method 'estimate_parallel'.
                Needed to continue an estimation process.

        Keyword arguments
        -----------------
            evolutions : int
                Defines how often the populations on the islands are evolved. 
                Migrations between the populations of the islands occur after an evolution.
            report_level : int
                Enables informative output about the estimation process. Information will be printed after each evolution.
                Default is 0, which is no output.
                1 = Prints the best loss, as well as information about archipelago creation and evolution. 
                    For each completed evolution, a dot is printed.
                2 = Prints additionally the best loss after each evolution
                3 = Prints additionally average loss among all islands, and the runtime of each evolution.
                4 = Prints additionally the parameter values for the best loss, and the average parameter values 
                    among the champions of all islands in the `archipelago` after the evolutions. 
            atol_islands : float
                Defines a stopping criterion that is checked after each evolution.
                If the std of the islands' losses < atol_islands + rtol_islands * abs(mean(islands' losses)), then the optimization is stopped.
                Default is None, which implies no effect for this argument.
            rtol_islands : float
                Defines a stopping criterion that is checked after each evolution.
                If the std of the islands' losses < atol_islands + rtol_islands * abs(mean(islands' losses)), then the optimization is stopped.
                Default is 1e-6.
            max_runtime_min : float
                The maximum runtime in min the estimation process is allowed to take. 
                The current runtime is evaluated after each completion of an evolution.
                Default is None, which implies that there is no time limit for the estimation process.
            max_evotime_min : float
                The maximum cumulative pure evolution time the estimation process is allowed to take.
                In contrast to the `max_runtime_min` stopping criterion, only the evolution runtime is considered, 
                without runtime needed for checking stopping criteria, reporting prints outs between each evolution, etc.
                Default is None.
            max_memory_share : float
                Defines the allowed memory share in usage, for which no evolutions are run anymore.
                Default is 0.95, meaning that repeat are only run if used memory share is less than 95 %.
            start_time : float
                In case a total runtime, started from another method, shall be reported.
                Default is None, which measured the total run time only within this method.

        Returns
        -------
            best_estimates : dict
                The estimated parameters, according to the best champion among all populations of the archipelago, aftet the last evolution.
            estimation_result : ParallelEstimationInfo
                Stores the archipelago and the history of the previous evolutions.
                Needed to continue an estimation process.

        Raises
        ------
            ValueError
                Not all island of the archipelago have the same unknowns.
        """

        if start_time is None:
            start_time = time.time()

        archipelago = estimation_result.archipelago
        evolutions_trail = estimation_result.evolutions_trail.copy()

        if report_level >= 1:
            if not evolutions_trail['evo_time_min']:
                _filler = ''
            else:
                _filler = 'additional '
            print(f'Running {_filler}{evolutions} evolutions for all {len(archipelago)} islands of the archipelago...\n')

        _current_evotime_min = 0
        for _evolution in range(1, evolutions+1):

            _evo_start = time.time()
            archipelago.evolve()
            archipelago.wait_check()
            _evo_end = time.time()
            _evo_time_min = (_evo_end - _evo_start)/60
            _current_evotime_min += _evo_time_min

            best_estimates, best_loss, estimates_info = ArchipelagoHelpers.extract_archipelago_results(archipelago)

            evolutions_trail['evo_time_min'].append(_evo_time_min)
            evolutions_trail['best_losses'].append(best_loss)
            evolutions_trail['best_estimates'].append(best_estimates)
            evolutions_trail['estimates_info'].append(estimates_info)

            if report_level == 1:
                if _evolution % 120 == 0:
                    end = '\n'
                else:
                    end = ''
                print('.', end=end)
            elif report_level >= 2:
                ArchipelagoHelpers.report_evolution_result(evolutions_trail, report_level)

            # evaluate stopping criteria after each evolution
            _current_runtime_min = (time.time() - start_time)/60
            stopping_criteria = ArchipelagoHelpers.check_evolution_stop(
                current_losses=estimates_info['losses'], 
                atol_islands=atol_islands, 
                rtol_islands=rtol_islands, 
                current_runtime_min=_current_runtime_min, 
                max_runtime_min=max_runtime_min,
                current_evotime_min=_current_evotime_min,
                max_evotime_min=max_evotime_min,
                max_memory_share=max_memory_share,
            )
            if any(stopping_criteria.values()):
                if report_level >= 1:
                    print(f'\nReached a stopping criterion after evolution {len(evolutions_trail["evo_time_min"])}:')
                    for _st in stopping_criteria:
                        print(f'{_st}: {stopping_criteria[_st]}')
                early_stop = True
                break
            else:
                early_stop = False
        
        if report_level >= 1:

            if not early_stop:
                print(f'\nCompleted {_evolution} {_filler}evolution runs.')

            print('\nEstimated parameters:')
            for p in best_estimates:
                print(f'{p}: {best_estimates[p]}')
            print('')

            ArchipelagoHelpers.report_evolution_result(evolutions_trail, report_level=3)

        _runtime_min = (time.time() - start_time)/60
        evolutions_trail['cum_runtime_min'].append(_runtime_min)
        if report_level >= 1:
            if _runtime_min/60 > 1:
                print(f'\nTotal runtime was {_runtime_min/60:.2f} h\n')
            else:
                print(f'\nTotal runtime was {_runtime_min:.2f} min\n')

        estimation_result = ParallelEstimationInfo(archipelago=archipelago, evolutions_trail=evolutions_trail)

        return best_estimates, estimation_result


    def estimate_parallel_MC_sampling(self,
                                      unknowns:list, 
                                      measurements:List[Measurement], 
                                      bounds:List[Tuple],
                                      mc_samples:int=25, 
                                      reuse_errors_as_weights:bool=True,
                                      metric:str='negLL', 
                                      report_level:int=0,
                                      optimizers:List[str]='de1220', 
                                      optimizers_kwargs:List[dict]={},
                                      rel_pop_size:float=10.0, 
                                      evolutions:int=25, 
                                      archipelago_kwargs:dict={},
                                      atol_islands:float=None, 
                                      rtol_islands:float=1e-6, 
                                      n_islands:int=4,
                                      handle_CVodeError:bool=True,
                                      loss_calculator:LossCalculator=LossCalculator,
                                      jobs_to_save:int=None,
                                      max_memory_share:float=0.95,
                                      ) -> pandas.DataFrame:
        """
        Performs Monte-Carlo sampling from measurements to create new measurements, according to the statitical distribution of the respective Measurement objects.
        For each newly created measurement, the requested unknowns (parameters) are estimated, resulting in an empirical distribution of parameter values. 
        these empirical distributions for the parameters can be assessed for uncertainties and correlations.
        For each MC sample, a parallel estimation procedure is carried out, for details see methods `estimate_parallel` and `estimate_parallel_continued`.
        Depending on the available number of CPUs on your machine, these estimation procedure are run in parallel.
        The selection of suitable hyperparameters, e.g. which optimizers, etc., use method `estimate_parallel` and refer to corresponding Jupyter notebooks.

        NOTE: To increase the number of MC samples to an arbiratry high number, set `repeats_to_save` argument. 
              Afterwards, the results saved to disk can be read and merged.

        NOTE: This method puts considerable computational load on your machine.

        Arguments
        ---------
            unknowns : dict or list
                The parameters to be estimated. Can be any of the model parameters, initial values or observation parameters. 
            measurements : List[Measurement]
                The measurements from which the parameters are to be estimated.
            bounds : List[tuple]
                List of tuples (lower, upper), one tuple for each parameter. Must be provided when using the global optimizer. 
                Default is None.

        Keyword arguments
        -----------------
            mc_samples : int
                The number of MC samples that shall be drawn from the measurement data. 
                Default is 25.
            reuse_errors_as_weights : bool
                Uses the measurement errors as weights for each set of measurement samples drawn. 
                Default is True.
            metric : str
                The metric according to which the loss to be minimized is calculated. 
                Can be one of, e.g. `negLL` (negative log-likelihood), 'SS' (sum of squares), or `WSS` (weighted sum of squares).
                Default is `negLL`, which implies that the corresponding Measurement objects are accordingly specified.
            report_level : int
                Enables informative output about the estimation process. 
                Default is 0, which is no output.
                1 = Prints a dot for each processing batch, the total runtime and the ratio of samples that reached convergence.
                2 = Prints likewise for 1, but with more information on each batch.
                3 = Prints additionally the runtime of each batch, as well as as summary for the obtained parameter distributions.
                4 = Prints additionally the reason for a MS sample to finish (i.e., 
                    whether convergence or the maximum number of evolutions was reached).
                5 = Prints additionally information on the creation of the archipelagos for each batch
                6 = Prints additionally the current evolution for each MC samples, and report any handled integration error.
            optimizers : List[str] or str
                A list of names for the pygmo optimization algorithms of choice. For a list of such to be conveniently used, 
                see `PygmoOptimizers` class of this module. 
                In case a list with one item is used, this optimizer is used for all explicitly 
                or implicitly (default None of `n_islands`) defined nunmber islands.
                In case a list with >1 optimizers is used, the corresponding number of islands will be created within the archipelago.
                The currently supported list of optimizer can be found at pyfoomb.generalized_islands.PygmoOptimizers.optimizers
                Default is `de1220`, which makes each island to use this algorithm.
            optimizers_kwargs : List[dict] or dict
                A list of optimizer_kwargs as dicts, corresponding to the list of optimizers. 
                In case more >1 optimizers are used, the 1-item list of optimizer_kwargs will be applied to all of the optimizers.
                Default is `{}`, i.e. no additional optimizer kwargs.
            rel_pop_size : float
                Determines the population size on each island, relative to the number of unknown to be estimated, 
                i.e. pop_size = rel_pop_size * len(unknowns), rounded to the next integer.
                Default is 10, which creates population sizes 10 times the number of unknowns.
            evolutions : int
                Defines how often the populations on the islands are evolved. 
                Migrations between the populations of the islands occur after each finished evolution. 
                Migration depends of the topology of the archipelago, as well as the defined migration polices, 
                which are parts of `archipelago_kwargs`.
                Default is 5, which triggers five rounds of evolution.
            archipelago_kwargs : dict
                The keyword arguments for instantiation of the archipelago.
                In case `archipelago_kwargs` has no key `t`, the `pygmo.fully_connected()` topology will be used 
                Default is {}, i.e. an empty dictionary, which implies the use of `pygmo.fully_connected()` topology.
            atol_islands : float
                Defines a stopping criterion that is checked after each evolution.
                If the std of the islands' losses < atol_islands + rtol_islands * abs(mean(islands' losses)), then the optimization is stopped.
                Default is None, which implies no effect for this argument.
            rtol_islands : float
                Defines a stopping criterion that is checked after each evolution.
                If the std of the islands' losses < atol_islands + rtol_islands * abs(mean(islands' losses)), then the optimization is stopped.
                Default is 1e-6.
            n_islands : int
                Specifies the number of parallel estimations per MC samples for all archipelagos in an estimation batch.
                In case a list of optimizers is provided, the number of islands is implicitly defined by its length. 
                Must use values > 1.
                Default is 4.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.
            loss_calculator : LossCalculator
                By subclassing `LossCalculator`, user-defined constraints can be implemented. The resulting subclass needs to be provided.
                Default is LossCalculator, which implements no additional constraints
            jobs_to_save : int
                Set to repeatedly run the specifified number of MC samples and to save the results from each repeat to file.
                Default is None, which causes no result storage ti file.
            max_memory_share : float
                Defines the allowed memory share in usage, for which no repeats are run anymore. Has only effect if `jobs_to_save` is not None.
                Default is 0.95, meaning that repeat are only run if used memory share is less than 95 %.

        Returns
        -------
            estimates : pandas.DataFrame
                The values from repeated estimation for the requested unknowns.
                Only converged estimations are included.

        Raises
        ------
            AttributeError
                Measurements have no errors.
            ValueError
                Degree of archipelago parallelization is < 2.
            TypeError
                A list containing not only Measurement objects is provided.
            KeyError:
                Non-unique unknowns detected.
            ValueError:
                Invalid parameters shall be estimated.
        """

        if jobs_to_save is None:
            _estimate = self._estimate_parallel_MC_sampling(
                unknowns=unknowns,
                measurements=measurements,
                bounds=bounds, 
                mc_samples=mc_samples, 
                reuse_errors_as_weights=reuse_errors_as_weights, 
                metric=metric, 
                report_level=report_level, 
                optimizers=optimizers, 
                optimizers_kwargs=optimizers_kwargs, 
                rel_pop_size=rel_pop_size, 
                evolutions=evolutions, 
                archipelago_kwargs=archipelago_kwargs, 
                atol_islands=atol_islands, 
                rtol_islands=rtol_islands, 
                n_islands=n_islands, 
                handle_CVodeError=handle_CVodeError, 
                loss_calculator=loss_calculator,
            )
            return pandas.DataFrame.from_dict(_estimate)

        _estimate_batches = []
        session_id = int(time.monotonic())
        for i in range(1, jobs_to_save+1):

            curr_memory_share = psutil.virtual_memory().percent/100
            if curr_memory_share > max_memory_share:
                print(f'Cannot run MC estimation job due to low memory: {(1-curr_memory_share)*100:.2f} % free memory left')
            else:
                _estimate_batch = self._estimate_parallel_MC_sampling(
                    unknowns=unknowns,
                    measurements=measurements,
                    bounds=bounds, 
                    mc_samples=mc_samples, 
                    reuse_errors_as_weights=reuse_errors_as_weights, 
                    metric=metric, 
                    report_level=report_level, 
                    optimizers=optimizers, 
                    optimizers_kwargs=optimizers_kwargs, 
                    rel_pop_size=rel_pop_size, 
                    evolutions=evolutions, 
                    archipelago_kwargs=archipelago_kwargs, 
                    atol_islands=atol_islands, 
                    rtol_islands=rtol_islands, 
                    n_islands=n_islands, 
                    handle_CVodeError=handle_CVodeError, 
                    loss_calculator=loss_calculator,
                )

                _filename = f'{self.model_name}_MC-sample-estimates_session-id-{session_id}_job-{i}.xlsx'
                _df = pandas.DataFrame.from_dict(_estimate_batch)
                _estimate_batches.append(_df)
                _df.to_excel(_filename)
                if report_level > 0:
                    print(f'Current memory usage is {psutil.virtual_memory().percent:.2f} %.\nSaved results of job #{i} to file: {_filename}\n')

        return pandas.concat(_estimate_batches, ignore_index=True)


    def _estimate_parallel_MC_sampling(self,
                                       unknowns:list, 
                                       measurements:List[Measurement], 
                                       bounds:List[Tuple],
                                       mc_samples:int=25, 
                                       reuse_errors_as_weights:bool=True,
                                       metric:str='negLL', 
                                       report_level:int=0,
                                       optimizers:List[str]='de1220', 
                                       optimizers_kwargs:List[dict]={},
                                       rel_pop_size:float=10.0, 
                                       evolutions:int=25, 
                                       archipelago_kwargs:dict={},
                                       atol_islands:float=None, 
                                       rtol_islands:float=1e-6, 
                                       n_islands:int=4,
                                       handle_CVodeError:bool=True,
                                       loss_calculator:LossCalculator=LossCalculator,
                                       ) -> dict:

        """
        Performs Monte-Carlo sampling from measurements to create new measurements, according to the statitical distribution of the respective Measurement objects.
        For each newly created measurement, the requested unknowns (parameters) are estimated, resulting in an empirical distribution of parameter values. 
        these empirical distributions for the parameters can be assessed for uncertainties and correlations.
        For each MC sample, a parallel estimation procedure is carried out, for details see methods `estimate_parallel` and `estimate_parallel_continued`.
        Depending on the available number of CPUs on your machine, these estimation procedure are run in parallel.
        The selection of suitable hyperparameters, e.g. which optimizers, etc., use method `estimate_parallel` and refer to corresponding Jupyter notebooks.

        NOTE: To increase the number of MC samples to an arbiratry high number, run this method several times and store intermediate results. 
              Afterwards, these can be merged. 

        NOTE: This method puts considerable computational load on your machine.

        Arguments
        ---------
            unknowns : dict or list
                The parameters to be estimated. Can be any of the model parameters, initial values or observation parameters. 
            measurements : List[Measurement]
                The measurements from which the parameters are to be estimated.
            bounds : List[tuple]
                List of tuples (lower, upper), one tuple for each parameter. Must be provided when using the global optimizer. 
                Default is None.

        Keyword arguments
        -----------------
            mc_samples : int
                The number of MC samples that shall be drawn from the measurement data. 
                Default is 25.
            reuse_errors_as_weights : bool
                Uses the measurement errors as weights for each set of measurement samples drawn. 
                Default is True.
            metric : str
                The metric according to which the loss to be minimized is calculated. 
                Can be one of, e.g. `negLL` (negative log-likelihood), 'SS' (sum of squares), or `WSS` (weighted sum of squares).
                Default is `negLL`, which implies that the corresponding Measurement objects are accordingly specified.
            report_level : int
                Enables informative output about the estimation process. 
                Default is 0, which is no output.
                1 = Prints a dot for each processing batch, the total runtime and the ratio of samples that reached convergence.
                2 = Prints likewise for 1, but with more information on each batch.
                3 = Prints additionally the runtime of each batch, as well as as summary for the obtained parameter distributions.
                4 = Prints additionally the reason for a MS sample to finish (i.e., 
                    whether convergence or the maximum number of evolutions was reached).
                5 = Prints additionally information on the creation of the archipelagos for each batch
                6 = Prints additionally the current evolution for each MC samples, and report any handled integration error.
            optimizers : List[str] or str
                A list of names for the pygmo optimization algorithms of choice. For a list of such to be conveniently used, 
                see `PygmoOptimizers` class of this module. 
                In case a list with one item is used, this optimizer is used for all explicitly 
                or implicitly (default None of `n_islands`) defined nunmber islands.
                In case a list with >1 optimizers is used, the corresponding number of islands will be created within the archipelago.
                The currently supported list of optimizer can be found at pyfoomb.generalized_islands.PygmoOptimizers.optimizers
                Default is `de1220`, which makes each island to use this algorithm.
            optimizers_kwargs : List[dict] or dict
                A list of optimizer_kwargs as dicts, corresponding to the list of optimizers. 
                In case more >1 optimizers are used, the 1-item list of optimizer_kwargs will be applied to all of the optimizers.
                Default is `{}`, i.e. no additional optimizer kwargs.
            rel_pop_size : float
                Determines the population size on each island, relative to the number of unknown to be estimated, 
                i.e. pop_size = rel_pop_size * len(unknowns), rounded to the next integer.
                Default is 10, which creates population sizes 10 times the number of unknowns.
            evolutions : int
                Defines how often the populations on the islands are evolved. 
                Migrations between the populations of the islands occur after each finished evolution. 
                Migration depends of the topology of the archipelago, as well as the defined migration polices, 
                which are parts of `archipelago_kwargs`.
                Default is 5, which triggers five rounds of evolution.
            archipelago_kwargs : dict
                The keyword arguments for instantiation of the archipelago.
                In case `archipelago_kwargs` has no key `t`, the `pygmo.fully_connected()` topology will be used 
                Default is {}, i.e. an empty dictionary, which implies the use of `pygmo.fully_connected()` topology.
            atol_islands : float
                Defines a stopping criterion that is checked after each evolution.
                If the std of the islands' losses < atol_islands + rtol_islands * abs(mean(islands' losses)), then the optimization is stopped.
                Default is None, which implies no effect for this argument.
            rtol_islands : float
                Defines a stopping criterion that is checked after each evolution.
                If the std of the islands' losses < atol_islands + rtol_islands * abs(mean(islands' losses)), then the optimization is stopped.
                Default is 1e-6.
            n_islands : int
                Specifies the number of parallel estimations per MC samples for all archipelagos in an estimation batch.
                In case a list of optimizers is provided, the number of islands is implicitly defined by its length. 
                Must use values > 1.
                Default is 4.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.
            loss_calculator : LossCalculator
                By subclassing `LossCalculator`, user-defined constraints can be implemented. The resulting subclass needs to be provided.
                Default is LossCalculator, which implements no additional constraints

        Returns
        -------
            estimates : dict
                The values from repeated estimation for the requested unknowns.
                Only converged estimations are included.
            estimates_info : dict
                Contains additional information for all finished jobs.

        Raises
        ------
            AttributeError
                Measurements have no errors.
            ValueError
                Degree of archipelago parallelization is < 2.
            TypeError
                A list containing not only Measurement objects is provided.
            KeyError:
                Non-unique unknowns detected.
            ValueError:
                Invalid parameters shall be estimated.
        """

        _start = time.time()

        # Some input error checkings
        if len(unknowns) != len(bounds):
            raise ValueError('Number of unkowns does not match number of pairs of upper and lower bounds.')

        for _item in measurements:
            if not isinstance(_item, Measurement):
                raise TypeError('Must provide a list of Measurement objects')

        if not Helpers.all_measurements_have_errors(measurements):
            raise AttributeError('Measurements must have errors')

        if not Helpers.has_unique_ids(unknowns):
            raise KeyError(Messages.bad_unknowns)

        _valid_unknowns = self._get_valid_parameter_names()
        for _unknown in unknowns:
            if _unknown not in _valid_unknowns:
                raise ValueError(f'Detected invalid unknown to be estimated: {_unknown} vs. {_valid_unknowns}')
        
        # Sort unknowns and corresponding bounds 
        _unknowns_names_sorted = sorted(unknowns, key=str.lower)
        _unknowns = [_unknown_name for _unknown_name in _unknowns_names_sorted]
        _bounds = [bounds[unknowns.index(_unknown_name)] for _unknown_name in _unknowns_names_sorted]

        if report_level >= 6:
            _verbosity_CVodeError = True
        else:
            _verbosity_CVodeError = False

        # Check whether to use the same optimizer for all islands or a list of optimizers
        if isinstance(optimizers, str):
            optimizers = [optimizers]*n_islands
        elif isinstance(optimizers, list):
            n_islands = len(optimizers)
        if isinstance(optimizers_kwargs, dict):
            optimizers_kwargs = [optimizers_kwargs]*n_islands

        if n_islands < 2:
            raise ValueError('Must use at least 2 islands per archipelago, either by specifying `n_islands` or using a list with more than 1 optimizers for kwargs `optimizers`.')

        if atol_islands is None:
            atol_islands = 0.0
        if rtol_islands is None:
            rtol_islands = 0.0

        # Calculate the number of archipelagos that can be run in parallel
        n_archis = max([int(numpy.floor(joblib.cpu_count()/n_islands)), 1])

        # Collects all finished estimation jobs
        _mc_estimates = []

        # Run parallel estimation jobs batch-wise
        for i in range(1, mc_samples+1, n_archis):

            _batch_no = int((i-1+n_archis)/n_archis)

            if report_level == 1:
                if _batch_no % 120 == 0:
                    end = '\n'
                else:
                    end = ''
                print('.', end=end)

            elif report_level >= 2:
                if report_level >= 3:
                    _insert = '\n'
                else:
                    _insert = ''
                _current_samples = [_sample for _sample in range(i, _batch_no*n_archis+1) if _sample <= mc_samples]
                _first = _current_samples[0]
                _last = f' to {_current_samples[-1]}' if _current_samples[-1] != _first else ''
                print(f'{_insert}------- Starting batch #{_batch_no} for MC sample {_first}{_last}.')

            # Initialize the current batch
            _batch_start = time.time()
            mc_count = i
            active_archis = []

            # Create a batch of achipelagos
            for j in range(n_archis):
                if mc_count+j > mc_samples:
                    break

                # Create the problem for a MC sample
                _pg_problem = pygmo.problem(
                    loss_calculator(
                        unknowns=_unknowns, 
                        bounds=_bounds, 
                        metric=metric, 
                        measurements=self._draw_measurement_samples(measurements, reuse_errors_as_weights), 
                        caretaker_loss_fun=self.loss_function, 
                        handle_CVodeError=handle_CVodeError, 
                        verbosity_CVodeError=_verbosity_CVodeError,
                    )
                )

                # Create the archipelago for the current problem
                _archi = ArchipelagoHelpers.create_archipelago(
                    unknowns=_unknowns, 
                    optimizers=optimizers, 
                    optimizers_kwargs=optimizers_kwargs, 
                    pg_problem=_pg_problem, 
                    rel_pop_size=rel_pop_size, 
                    archipelago_kwargs=archipelago_kwargs, 
                    log_each_nth_gen=None, 
                    report_level=0, 
                )
                _archi.mc_info = f'MC sample #{mc_count+j}'
                _archi.finished = False
                _archi.problem = loss_calculator
                _archi.wait_check()

                if report_level >= 5:
                    print(f'{_archi.mc_info}: created archipelago with {len(_archi)} islands')
                active_archis.append(_archi)

            # Evolve the archipelagos in the current batch
            for j in range(len(active_archis)):
                for evo in range(1, evolutions+1):

                    # Start an async evolution for all non-converged archis
                    for _archi in active_archis:
                        if not _archi.finished: 
                            if report_level >= 6:
                                print(f'\t{_archi.mc_info}: running evolution {evo}')
                            _archi.evolve()

                    # Wait for all archis to finish
                    for _archi in active_archis:
                        if not _archi.finished:
                            _archi.wait_check()

                    # Check the archis for results
                    for _archi in active_archis:

                        # Calculate convergence criterion
                        _losses = numpy.array(_archi.get_champions_f()).flatten()
                        _stop_criterion = atol_islands + rtol_islands * numpy.abs(numpy.mean(_losses))
                        _abs_std = numpy.std(_losses, ddof=1)

                        # Check for convergence for the non-finished archi
                        if _abs_std < _stop_criterion and not _archi.finished:
                            _best_estimates = ArchipelagoHelpers.estimates_from_archipelago(_archi)
                            _best_estimates['convergence'] = True
                            _best_estimates['max_evos'] = False
                            _best_estimates['archi'] = f'{_archi.mc_info}'
                            _best_estimates['losses'] = _losses
                            _archi.finished = True
                            _mc_estimates.append(_best_estimates)
                            if report_level >= 4:
                                print(f'{_archi.mc_info}: convergence')

                        # Check for max evolutions for the non-finished archi
                        elif evo == evolutions and not _archi.finished:
                            _best_estimates = ArchipelagoHelpers.estimates_from_archipelago(_archi)
                            _best_estimates['convergence'] = False
                            _best_estimates['max_evos'] = True
                            _best_estimates['archi'] = f'{_archi.mc_info}'
                            _best_estimates['losses'] = _losses
                            _archi.finished = True
                            _mc_estimates.append(_best_estimates)
                            if report_level >= 4:
                                print(f'{_archi.mc_info}: no convergence after max. evolutions ({evo}).')

                    if all([_archi.finished for _archi in active_archis]):
                        del active_archis
                        gc.collect()
                        active_archis = []
                        break

                if all([_archi.finished for _archi in active_archis]):
                    del active_archis
                    gc.collect()
                    active_archis = []
                    break

            if report_level >= 3:
                print(f'Runtime for batch {_batch_no} was {(time.time() - _batch_start)/60:.2f} min.')

            # All requested MC samples were run
            if mc_count > mc_samples:
                break

        # Comprehend results
        aug_unknowns = [*_unknowns, 'convergence', 'max_evos', 'archi', 'losses']
        estimates_info = {str(_p) : [] for _p in aug_unknowns}
        for _mc_estimate in _mc_estimates:
            for _p in _mc_estimate:
                estimates_info[_p].append(_mc_estimate[_p])
        estimates = {
            str(_p) : numpy.array(numpy.array(estimates_info[_p])[estimates_info['convergence']]) # include only samples that converged
            for _p in _unknowns
        }
        if report_level >= 1:
            _end = time.time()
            # Runtime was > 1 h
            if (_end-_start)/3600 > 1:
                print(f'\n-----------------------------------------------\nTotal runtime was {(_end-_start)/3600:.2f} h.')
            else:
                print(f'\n-----------------------------------------------\nTotal runtime was {(_end-_start)/60:.2f} min.')
            print(f'Convergence ratio was {numpy.sum(estimates_info["convergence"])/len(estimates_info["convergence"])*100:.1f} %.')

        if report_level >= 3:
            print('\nSummaries for empirical parameter distributions\n-----------------------------------------------')
            print(pandas.DataFrame(estimates).describe().T)

        return estimates


    def estimate_repeatedly(self, 
                            unknowns:list, measurements:List[Measurement], bounds:List[tuple], metric:str='negLL',
                            jobs:int=10, rel_jobs:float=None, report_level:int=0, reset_afterwards:bool=False, handle_CVodeError:bool=True,
                            ) -> Tuple[dict, list]:
        """
        Runs a several global estimations for the requested unknowns. Resulting distributions for the estimated parameters 
        can be inspected for measures of dispersion or correlations among parameters. In case a rather high number of estimation jobs is run, 
        the resulting distributions can be nicely investigated using the Visualization.show_parameter_distributions method.

        NOTE: This method puts considerable computational load on your machine.

        Arguments
        ---------
            unknowns : list or dict
                The parameters to be estimated. Can be any of the model parameters, initial values or observation parameters. 
                In case a dict is provided, the corresponding values are ignored.
            measurements : List[Measurement]
                The data, from which the repeated estimation is performed. 
                Can provide a Measurement object for any model state or observation.
            bounds : List[Tuple]
                Bounds for for each unknown to be estimated. 

        Keyword arguments
        -----------------
            metric : str
                The metric according to which the loss to be minimized is calculated. 
                Can be one of, e.g. `negLL` (negative log-likelihood), 'SS' (sum of squares), or `WSS` (weighted sum of squares).
                Default is `negLL`, which implies that the corresponding Measurement object are accordingly specified.
            jobs : int
                The number of estimation jobs that are requested. 
                Default is 10.
            rel_jobs : float
                Number of estimation jobs, relative to the number of unknowns: rel_jobs * number of unknowns. 
                Overrides jobs argument. Default is None, which implies use of `jobs`.
            report_level : int
                Enables informative output about the estimation process. 
                Default is 0, which is no output.
                1 = prints a summary of the empirical parameter distributions and basic information about the parallization.
                2 = reports additionally about each parallel estimation job.
            reset_afterwards : bool
                To reset the Caretaker object after the estimation has finished. 
                Default is False.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.

        Returns
        -------
            repeated_estimates : dict
                The values from repeated estimation for the requested unknowns.
            results : list
                Contains estimation_info dicts for each estimation job.
            TypeError
                A list containing not only Measurement objects is provided. 

        Warns
        -----
            UserWarning
                Property `optimizer_kwargs` of this Caretaker instance has key `disp`.
        """

        warnings.warn(
            'This method will be deprecated in future releases of pyFOOMB.', 
            PendingDeprecationWarning
        )

        for _item in measurements:
            if not isinstance(_item, Measurement):
                raise TypeError('Must provide a list of Measurement objects')

        if isinstance(unknowns, list):
            unknowns = {unknown : None for unknown in unknowns}

        # rel_jobs overrides jobs
        if rel_jobs is not None:
            jobs = int(numpy.ceil(rel_jobs * len(bounds)))

        if self.optimizer_kwargs is not None:
            if 'disp' in self.optimizer_kwargs.keys():
                warnings.warn(
                    'Reporting progress for each single optimization job is deactivated for parallel multi-estimation methods', 
                    UserWarning,
                )

        # collect arg instances for each parallel job
        arg_instances = [
            {
                'unknowns' : unknowns, 
                'measurements' : measurements, 
                'bounds' : bounds, 
                'metric' : metric,
                'use_global_optimizer' : True, 
                'handle_CVodeError' : handle_CVodeError, 
                'optimizer_kwargs' : {'disp' : False},
                }
           for job in range(jobs)
        ]

        parallel_verbosity = 0
        if report_level >= 1:
            parallel_verbosity = 1
        if report_level >= 2:
            parallel_verbosity = 11

        # do the jobs
        repeated_estimates, results = self._estimate_parallelized_helper(arg_instances, unknowns, parallel_verbosity)

        if report_level >= 1:
            print('\nSummaries for empirical parameter distributions\n-----------------------------------------------')
            print(pandas.DataFrame(repeated_estimates).describe().T)
        if report_level >= 2:
            _runtimes_min = [result[1]['runtime_min'] for result in results]
            print(f'\nAverage runtime per estimation job was {numpy.mean(_runtimes_min):.2f} +/- {numpy.std(_runtimes_min, ddof=1):.2f} min')

        return repeated_estimates, results


    def estimate_MC_sampling(self, 
                             unknowns:list, measurements:List[Measurement], bounds:List[tuple]=None, 
                             metric:str='negLL', reuse_errors_as_weights:bool=True,
                             mc_samples:int=100, rel_mc_samples:float=None, 
                             report_level:int=0, reset_afterwards:bool=True, use_global_optimizer:bool=True, 
                             handle_CVodeError:bool=True, 
                             ) -> Tuple[dict, dict]:
        """
        Performs Monte-Carlo sampling from measurements, and re-estimates parameters. Per default, global optimization is used.
        Resulting bootstrapped distributions for the parameters can be assessed for parameter uncertainties and correlations.

        NOTE: This method puts considerable computational load on your machine.

        Arguments
        ---------
            unknowns : dict or list
                The parameters to be estimated. Can be any of the model parameters, initial values or observation parameters. 
                Providing a list of valid unknowns causes the use of scipy's differential evolution optimizer. 
                To use the local minimizer, unknowns must be of type dict, with initial guesses as values.
            measurements : List[Measurement]
                The measurements from which the parameters are to be estimated.

        Keyword arguments
        -----------------
            bounds : List[tuple]
                List of tuples (lower, upper), one tuple for each parameter. Must be provided when using the global optimizer. 
                Default is None.
            metric : str
                The metric according to which the loss to be minimized is calculated. 
                Can be one of, e.g. `negLL` (negative log-likelihood), 'SS' (sum of squares), or `WSS` (weighted sum of squares).
                Default is `negLL`, which implies that the corresponding Measurement object are accordingly specified.
            reuse_errors_as_weights : bool
                Uses the measurement errors as weights for each set of measurement samples drawn. 
                Default is True.
            mc_samples : int
                The number of MC samples that shall be drawn from the measurement data. 
                Default is 100.
            rel_mc_samples : float
                Number of MC samples = rel_mc_samples * number of measurement points. Overrides mc_samples. Default is None.
            report_level : int
                Enables informative output about the estimation process. 
                Default is 0, which is no output.
                1 = prints a summary of the empirical parameter distributions and basic information about the parallization.
                2 = reports additionally about each parallel estimation job.
            reset_afterwards : bool
                Resets the Caretaker object after the estimation has finished.
                Default is False.
            use_global_optimizer : bool
                Use global optimizer instead of local minimizer for repeated parameter estimation. 
                Default is True.
            parallel_verbosity : int
                Control output level of the parallel job work, default is 0. 
                See joblib documentation for futher details.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.

        Returns
        -------
            repeated_estimates : dict
                The values from repeated estimation for the requested unknowns.
            results : list
                Contains estimation_info dicts for each estimation job.

        Raises
        ------
            TypeError
                No initial guesses are provided when using the local optimizer.
            TypeError
                No bounds provided when using the global optimizer.
            ValueError
                Measurements have no errors.
            TypeError
                A list containing not only Measurement objects is provided.

        Warns
        -----
            UserWarning
                Property `optimizer_kwargs` of this Caretaker instance has key `disp`.
        """

        warnings.warn(
            'This method will be deprecated in future releases of pyFOOMB. Use method `estimate_parallel_MC_sampling` instead', 
            PendingDeprecationWarning
        )

        for _item in measurements:
            if not isinstance(_item, Measurement):
                raise TypeError('Must provide a list of Measurement objects')

        # Error handling
        if not use_global_optimizer and not isinstance(unknowns, dict):
            raise ValueError('Must provide initial guesses when using the local optimizer')
        if use_global_optimizer and bounds is None:
            raise ValueError('Must provide bounds to use the global optimizer')
        if not Helpers.all_measurements_have_errors(measurements):
            raise AttributeError('Measurements cannot have no errors')

        if isinstance(unknowns, list):
            unknowns = {unknown : None for unknown in unknowns}

        if self.optimizer_kwargs is not None:
            if 'disp' in self.optimizer_kwargs.keys():
                warnings.warn(
                    'Reporting progress for each single optimization job is deactivated for parallel multi-estimation methods', 
                    UserWarning,
                )

        # calcuate number of estimation jobs
        if rel_mc_samples is not None:
            n_meas = numpy.sum([len(measurement.timepoints) for measurement in measurements])
            mc_samples = int(numpy.ceil(n_meas * rel_mc_samples))

        # collect arg instances for parallel jobs
        arg_instances = []
        for i in range(mc_samples):
            _rnd_measurements = self._draw_measurement_samples(measurements, reuse_errors_as_weights)
            arg_instances.append(
                {
                    'unknowns' : unknowns, 
                    'measurements' : _rnd_measurements, 
                    'bounds' : bounds, 
                    'metric' : metric,
                    'handle_CVodeError' : handle_CVodeError,
                    'use_global_optimizer' : use_global_optimizer,
                    'reset_afterwards' : True,
                    'optimizer_kwargs' : {'disp' : False},
                }
            )   

        parallel_verbosity = 0
        if report_level >= 1:
            parallel_verbosity = 1
        if report_level >= 2:
            parallel_verbosity = 11

        # do the jobs
        repeated_estimates, results = self._estimate_parallelized_helper(arg_instances, unknowns, parallel_verbosity)

        if report_level >= 1:
            print('\nSummaries for empirical parameter distributions\n-----------------------------------------------')
            print(pandas.DataFrame(repeated_estimates).describe().T)
        if report_level >= 2:
            _runtimes_min = [result[1]['runtime_min'] for result in results]
            print(f'\nAverage runtime per estimation job was {numpy.mean(_runtimes_min):.2f} +/- {numpy.std(_runtimes_min, ddof=1):.2f} min')

        return repeated_estimates, results


    def get_sensitivities(self,
                          measurements:List[Measurement]=None, responses:list='all', parameters:list=None,
                          tfinal:float=None, abs_h:float=None, rel_h:float=1e-3, 
                          handle_CVodeError:bool=True, verbosity_CVodeError:bool=False,
                          ) -> List[Sensitivity]:

        """
        Approximates sensitivities of model responses w.r.t. parameters using Central Difference Quotient: f'(x) = f(x+h) - f(x-h) / (2*h).
        Indicates how a model response (i.e., a state or observation) changes dynamically in time
        with a small change in a certain parameters (i.e., a model parameter, initial value, or observation parameter).

        Keyword arguments
        -----------------
            measurements : List[Measurement]
                Can provide a Measurement object for any model state or observation. 
                Default is None, which implies that `tfinal` cannot be None.
            responses : list
                Specific model responses (state or observable), for which the sensitivities are requested. 
                Default is `all´, which causes sensitivities for all model responses.
            parameters (list or dict, default=None) : The parameters for which the sensitivities are requested. 
                In case a dict is provided, the corresponding values will be set.
                In case a list is provided, the corresponding values for the current mapping will be used.
            tfinal : float
                The final integration time. 
                Default is None, which implies that `measurements` cannot be None.
            abs_h : float
                Absolute perturbation for central difference quotient. `rel_h` must be set to None for `abs_h` to take effect.
                Default is None.
            rel_h : float
                Relative pertubation for central difference quotient. Overrides use of abs_h. 
                Absolute perturbation for each parametric sensitivity is then calculated according to: abs_h = rel_h * max(1, |p|). 
                Default is 1e-3.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.
            verbosity_CVodeError : bool
                Enables informative output during handling CVodeErrors. Default is False.

        Returns
        -------
            sensitivities : List[Sensitivities]

        Raises
        ------
            TypeError
                Wrong type for kwarg `responses`.
            ValueError
                Non-unique (case-insensitive) respones given.
            ValueError
                Non-unique (case-insensitive) parameters given.
            ValueError
                Given parameters are not known according to the current parameter mapping.
            ValueError
                Neither measurements dict nor tfinal is provided.
            TypeError
                Wrong type for kwarg `parameters`.
            TypeError
                A list containing not only Measurement objects is provided.
        """

        if not isinstance(responses, list) and responses != 'all':
            raise TypeError('Responses must be either of type list or `all`')
        if responses != 'all':
            if not Helpers.has_unique_ids(responses):
                raise ValueError(Messages.non_unique_ids)

        if measurements is not None:
            for _item in measurements:
                if not isinstance(_item, Measurement):
                    raise TypeError(f'Must provide a list of Measurement objects: {_item} in {measurements}')

        # timepoints for integration
        t = numpy.array([])
        if measurements is not None:
            t = numpy.append(t, Helpers.get_unique_timepoints(measurements))
        if tfinal is not None:
            tfinal = numpy.array(tfinal)
            if t.size == 0:
                t = numpy.append(t, tfinal)
            elif tfinal > max(t):
                t = numpy.append(t, tfinal)
        if t.size == 0:
            raise ValueError('Must provide either measurements or tfinal')
        if t.size == 1:
            _simulations = self.simulate(t=t, verbosity=50)
            t = Helpers.get_unique_timepoints(_simulations)

        # set parameters if provided
        _parameter_names = self._get_valid_parameter_names()
        if parameters is not None:
            if not Helpers.has_unique_ids(parameters):
                raise ValueError(Messages.non_unique_ids)
            if not set(parameters).issubset(set(_parameter_names)):
                raise ValueError(f'Invalid parameters: {set(parameters).difference(set(_parameter_names))}. Valid parameters are: {_parameter_names}.')
            if isinstance(parameters, dict):
                self.set_parameters(parameters)
            elif isinstance(parameters, list):
                _parameters = self._get_all_parameters()
                parameters = {p : _parameters[p] for p in parameters}
        else:
            parameters = self._get_all_parameters()

        sensitivities = []
        for _id in self.replicate_ids:
            sensitivities.extend(self._get_sensitivities_parallel(_id, parameters, rel_h, abs_h, t, responses))
        return sensitivities


    def get_information_matrix(self, 
                               measurements:List[Measurement], estimates:dict, 
                               sensitivities:List[Sensitivity]=None, handle_CVodeError:bool=True, verbosity_CVodeError:bool=False,
                               ) -> numpy.ndarray:
        """
        Constructs Fisher information matrix (FIM) by calculating a FIM at each distinct timepoint where at least one measurement was made. 
        The time-varying FIMs are added up to the FIM. FIM(t) are build using sensitivities, which are approximated using the central difference quotient method.
        FIM is of shape (n_estimated_parameters, n_estimated_parameters), with parameters sorted alphabetically (case-insensitive).
        Non-intertible FIM indicates that parameter(s) cannot be identified from the given measurements.

        Arguments
        ---------
            measurements : List[Measurement]
                Can provide a Measurement object for any model state or observation.
            estimates : dict
                The parameters (model parameters, initial values, observation parameters) that have been estimated previously.

        Keyword arguments
        -----------------
            sensitivities : List[Sensitivity]
                These may have been calculated previously using the method `get_sensitivities`. 
                Default is None, which causes calculation of sensitivities.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.
            verbosity_CVodeError : bool
                Enables informative output during handling CVodeErrors. Default is False.

        Returns
        -------
            FIM : numpy.ndarray
                Fisher information matrix is of shape (n_estimated_parameters, n_estimated_parameters), 
                with values of rows and cols corresponding to the parameters (sorted alphabetically case-insensitive).
        
        Raises
        ------
            TypeError
                A list containing not only Measurement objects is provided.
            TypeError
                A list containing not only Sensitivity objects is provided.
        """
        
        for _item in measurements:
            if not isinstance(_item, Measurement):
                raise TypeError('Must provide a list of Measurement objects')

        if sensitivities is None:
            sensitivities = self.get_sensitivities(
                measurements=measurements, 
                parameters=estimates, 
                handle_CVodeError=handle_CVodeError,
            )
        else:
            for _item in sensitivities:
                if not isinstance(_item, Sensitivity):
                    raise TypeError('Must provide a list of Sensitivity objects')

        all_t = Helpers.get_unique_timepoints(measurements)

        FIMs = {}
        FIM = numpy.full(shape=(len(estimates), len(estimates)), fill_value=0)
        for _id in self.replicate_ids:
            FIMs[_id] = []
            for _t in all_t:
                _FIM_t = self._get_information_matrix_at_t(t=_t, measurements=measurements, estimates=estimates, sensitivities=sensitivities, replicate_id=_id)
                FIMs[_id].append(_FIM_t)
                FIM = FIM + _FIM_t
        return FIM


    def get_parameter_uncertainties(self, 
                                    estimates:dict, 
                                    measurements:List[Measurement], 
                                    sensitivities:List[Sensitivity]=None,
                                    report_level:int=0, 
                                    handle_CVodeError:bool=True, 
                                    verbosity_CVodeError:bool=True,
                                    ) -> dict:
        """
        Calculates uncertainties for estimated parameters, based on variance-covariance matrix derived from sensitivity-based Fisher information matrix.

        NOTE: The parameter variance-covariance matrix represents a symmetric, linear approximation to the parameter (co)-variances. 
              Other methods such a Monte-Carlo sampling can discover non-linear correlations, but require significant computational load.

        Arguments
        ---------
            estimates : dict
                Dictionary holding the previously estimated parameter values.
            measurements : List[Measurement]
                The measurements from which the parameters have been estimated.

        Keyword arguments
        -----------------
            sensitivities : List[Sensitivity]
                These may have been calculated previously using the method `get_sensitivities`. 
                Default is None, which causes calculation of sensitivities.
            report_level : int
                Controls depth of informative output, default is 0 which is no output.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.
            verbosity_CVodeError : bool
                Enables informative output during handling CVodeErrors. Default is False.

        Returns
        -------
            parameter_information : dict
                A dictionary summarizing the parameters, their values and standard errors.
        
        Raises
        ------
            TypeError
                A list containing not only Measurement objects is provided.
            TypeError
                A list containing not only Sensitivity objects is provided.
        """

        for _item in measurements:
            if not isinstance(_item, Measurement):
                raise TypeError('Must provide a list of Measurement objects')

        if sensitivities is None:
            sensitivities = self.get_sensitivities(
                measurements=measurements, 
                parameters=estimates, 
                handle_CVodeError=handle_CVodeError, 
                verbosity_CVodeError=verbosity_CVodeError,
            )
        else:
            for _item in sensitivities:
                if not isinstance(_item, Sensitivity):
                    raise TypeError('Must provide a list of Sensitivity objects')

        matrices = self.get_parameter_matrices(measurements=measurements, estimates=estimates, sensitivities=sensitivities)
        std_errs = numpy.sqrt(numpy.diag(matrices['Cov']))

        if report_level>=1:
            print('\nEstimated parameters:\n----------')
            for _p, _err in zip(sorted(estimates.keys(), key=str.lower), std_errs):
                print(f'{_p}: {estimates[_p]:.2e} +/- {_err:.2e} ({abs(_err/estimates[_p]*100):.2f} %)')

        parameter_information = {}
        parameter_information['Parameters'] = sorted(estimates.keys(), key=str.lower)
        parameter_information['Values'] = numpy.array([estimates[_p] for _p in sorted(estimates.keys(), key=str.lower)])
        parameter_information['StdErrs'] = std_errs
        return parameter_information


    def get_optimality_criteria(self, Cov:numpy.ndarray, report_level:int=0) -> dict:
        """
        Calculates single-value optimality criteria from a parameter variance-covariance matrix.

        Arguments
        ---------
            Cov : numpy.ndarray
                The parameter covariance matrix for the estimated parameters.

        Keyword arguments
        -----------------
            report_level : int
                Controls informative output on optimality criteria.
                Default is 0, which is no print output.

        Returns
        -------
            opt_criteria : dict
                The calculated optimality criteria.
        """

        criteria = ['A', 'D', 'E', 'E_mod']
        cov_evaluator = CovOptimality()
        opt_criteria = {_criterion : cov_evaluator.get_value(_criterion, Cov) for _criterion in criteria}

        if report_level >=1:
            print('\nOptimality criteria:\n----------')
            for _criterion in criteria:
                print(f'{_criterion}: {opt_criteria[_criterion]:.2e}')

        return opt_criteria


    def get_parameter_matrices(self, 
                               estimates:dict, 
                               measurements:List[Measurement], 
                               sensitivities:List[Sensitivity]=None, 
                               handle_CVodeError:bool=True,
                               ) -> Dict[str, numpy.ndarray]:
        """
        Calculate Fisher information matrix FIM, as well as corresponding variance-covariance matrix Cov and correlation matrix Corr.

        Arguments
        ---------
            estimates : dict
                Dictionary holding the previously estimated parameter values.
            measurements : List[Measurement]
                The measurements from which the parameters have been estimated.

        Keyword arguments
        -----------------
            sensitivities : List[Sensitivity]
                These may have been calculated previously using the method `get_sensitivities`. 
                Default is None, which causes calculation of sensitivities.
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
                Default is True.

        Returns
        -------
            matrices : Dict[str, numpy.ndarray]
                The different parameter matrices.

        Raises
        ------
            TypeError
                A list containing not only Measurement objects is provided.
            TypeError
                A list containing not only Sensitivity objects is provided.   
        """

        for _item in measurements:
            if not isinstance(_item, Measurement):
                raise TypeError('Must provide a list of Measurement objects')

        if sensitivities is not None:
            for _item in sensitivities:
                if not isinstance(_item, Sensitivity):
                    raise TypeError('Must provide a list of Sensitivity objects')

        FIM = self.get_information_matrix(measurements=measurements, estimates=estimates, sensitivities=sensitivities, handle_CVodeError=handle_CVodeError)
        try:
            Cov = numpy.linalg.inv(FIM)
        except LinAlgError:
            warnings.warn('Information matrix not invertible.', UserWarning)
            Cov = numpy.full(shape=FIM.shape, fill_value=numpy.inf)
        Corr = Calculations.cov_into_corr(Cov)
        matrices = {}
        matrices['FIM'] = FIM
        matrices['Cov'] = Cov
        matrices['Corr'] = Corr
        return matrices


    def set_integrator_kwargs(self, integrator_kwargs:dict):
        """
        Set some options for the used CVode integrator. 
        These are propagated to all internally handled ExtendedSimulator instances.
        Typical options are `atol` or `rtol`. For all options, see https://jmodelica.org/assimulo/ODE_CVode.html. 
        Note that not all options may have an effect due to the use of the integrator in this package.

        Arguments
        ---------
            integrator_kwargs : dict
                The CVode integrator options to be set. 
        """
        
        for _id in self.replicate_ids:
            self.simulators[_id].integrator_kwargs = integrator_kwargs


    def set_parameters(self, parameters:dict):
        """
        Assigns specfic values to parameters, according to the current parameter mapping.

        Arguments
        ---------
            values : dict
                Key-value pairs for parameters that are to be set.
        """

        self._parameter_manager.set_parameter_values(parameters)
        self._propagate_parameters_through_simulators()
    

    def reset(self):
        """
        Resets the Caretaker object to its state after instantiation, preserving the current replicates
        """

        self.__init__(
            bioprocess_model_class=self.__bioprocess_model_class, 
            model_parameters=self.__model_parameters, 
            states=self.__states, 
            initial_values=self.__initial_values, 
            replicate_ids=self.__replicate_ids, 
            initial_switches=self.__initial_switches, 
            model_name=self.__model_name, 
            observation_functions_parameters=self.__observation_functions_parameters,
        )


    def apply_mappings(self, mappings:list):
        """
        A list of mappings that are applied to the parameters among the replicates.
        An item of the mappings list must either be a tuple with the structure (replicate_id, global_name, local_name, value) or
        a ParameterMapper instance according to ParameterMapper(replicate_id=..., global_name=..., local_name=..., value=...).

        NOTE: replicate_id can also be a list, which applies the mapping to all replicate in this list.

        NOTE: replicate_id can also be `all`, which applies the mapping to all replicates. 

        Arguments
        ---------
            mappings : list of ParameterMapper and/or tuple
                A list of mappings, which can be a tupe or ParameterMapper objects, or a mix of them.
                Example: [ParameterMapper(replicate_id=..., global_name=..., local_name=..., value=...), ]
                Example: [(replicate_id, global_name, local_name, value), ]
                Example : [
                           ParameterMapper(replicate_id=..., global_name=..., local_name=..., value=...), 
                           (replicate_id, global_name, local_name, value), 
                          ]
                
        Raises
        ------
            TypeError
                Any mapping is not a tuple or ParameterMapper object.
            ValueError
                Any mapping has an invalid replicate id.
            ValueError
                Any mapping has a invalid global parameter name.

        Warns
        -----
            UserWarning
                This method is called from implicitly defined single replicate Caretaker objects.
        """

        if not (len(self.replicate_ids) == 1 and self.replicate_ids[0] == SINGLE_ID):
            self._parameter_manager.apply_mappings(mappings)
            self._propagate_parameters_through_simulators()
        else:
            warnings.warn(
                'Parameter mappings cannot be applied for single replicate Caretaker objects. Use `set_parameters()` method instead.', 
                UserWarning
            )


    #%% Private methods

    def _get_sensitivities_parallel(self, replicate_id:str, parameters:list, rel_h:float, abs_h:float, t:numpy.ndarray, responses:list) -> List[Sensitivity]:
        """
        Calculates sensitivities in parallel for a specific replicate_id.
        
        Arguments
        ---------
            replicate_id : str
                The replicate_id for which the sensitivities are requested.
            parameters : list
                The model parameters, initial values and/or observation parameters for which the sensitivites are requested
            rel_h : float
                The central difference perturbation values, relative to the parameters.
            abs_h = float
                The absolute central difference perturbation value.
            t : numpy.ndarray
                The timepoints at which the sensitivities are to be calculated.
            responses : list
                The model states and/or observations for which the sensitivities are to be calculated.

        Returns
        -------
            List[Sensitivity]
                
        Raises
        ------
            ValueError
                Sensitivities are to be calculated for unknown model responses.
        """

        # Calculate sensitivities for each replicate
        arg_instances = []
        if responses == 'all':
            _responses = self.simulators[replicate_id]._get_allowed_measurement_keys()
        elif not set(responses).issubset(set(self.simulators[replicate_id]._get_allowed_measurement_keys())):
            raise ValueError(
                f'Invalid model responses for sensitivity calculation. {set(responses).difference(set(self.simulators[replicate_id]._get_allowed_measurement_keys()))}'
            )
        else:
            _responses = responses
        # Collect arg instances
        for _response in _responses:
            for _parameter in parameters:
                arg_instance = (replicate_id, _response, _parameter, rel_h, abs_h, parameters, t)
                arg_instances.append(arg_instance)
        # Do the parallel work for this replicate_id
        n_cpus = joblib.cpu_count()
        if len(arg_instances) < n_cpus:
            n_jobs = len(arg_instances)
        else:
            n_jobs = n_cpus

        with joblib.parallel_backend('loky', n_jobs=n_jobs):
            sensitivities = joblib.Parallel(verbose=0)(map(joblib.delayed(self._d_response_i_wrt_d_parameter_j_central_difference), arg_instances))

        return list(sensitivities)


    def _d_response_i_wrt_d_parameter_j_central_difference(self, arg) -> Sensitivity:
        """
        Helper method for parallelization of sensitivity calculation.
        """

        replicate_id, response_i, parameter_j, rel_h, abs_h, parameters, t = arg
        
        # set parameter perturbation
        if rel_h is not None:
            abs_h = rel_h * max([1, abs(parameters[parameter_j])])

        if abs_h*1.1 <= EPS64:
            warnings.warn(f'Parameter perturbation for finite differences is in the same order of machine precision. {abs_h} vs. {EPS64}', UserWarning)


        # forward simulations with parameter forward perturbation
        _pars_plus = copy.deepcopy(parameters)
        _pars_plus[parameter_j] = _pars_plus[parameter_j] + abs_h
        self.set_parameters(_pars_plus)
        simulation_plus = self.simulators[replicate_id].simulate(t=t, verbosity=50)
        simulation_plus_resp_i = Helpers.extract_time_series(simulation_plus, name=response_i, replicate_id=replicate_id)
        y_plus = simulation_plus_resp_i.values

        # forward simulations with parameter backward perturbation
        _pars_minus = copy.deepcopy(parameters)
        _pars_minus[parameter_j] = _pars_minus[parameter_j] - abs_h
        self.set_parameters(_pars_minus)
        simulation_minus = self.simulators[replicate_id].simulate(t=t, verbosity=50)
        simulation_minus_resp_i = Helpers.extract_time_series(simulation_minus, name=response_i, replicate_id=replicate_id)
        y_minus = simulation_minus_resp_i.values

        # approx fprime
        dyi_dthetai = (y_plus - y_minus) / (2 * abs_h)
        timepoints = simulation_plus_resp_i.timepoints
        return Sensitivity(timepoints=timepoints, values=dyi_dthetai, response=response_i, parameter=parameter_j, h=abs_h, replicate_id=replicate_id)


    def _draw_measurement_samples(self, measurements:List[Measurement], reuse_errors_as_weights:bool=True) -> List[Measurement]:
        """
        Helper method for `estimate_MC_sampling` method.

        Arguments
        ---------
            measurements : List[Measurement]
                The measurements from which the parameters have been estimated. 
                Assumes that this argument was run through method `utils.Helpers.check_kinetic_data_dict()`

        Keyword arguments
        -----------------
            reuse_errors_as_weights : bool
                Uses the measurement errors as weights for each set of measurement samples drawn. 
                Default is True.

        Returns
        -------
            rnd_measurements : List[Measurement]
                A copy of the `measurements` argument, but with its `values` property replaced with random values, 
                according to its `distribution` property. Its `errors` property is replaced by a vector of ones, 
                if the keyword argument `reuse_errors_as_weights` was False.
        """
        
        rnd_measurements = copy.deepcopy(measurements)
        for i in range(len(measurements)):
            _rnd_values = measurements[i]._get_random_samples_values()
            rnd_measurements[i].values = _rnd_values
            if not reuse_errors_as_weights:
                rnd_measurements[i].errors = numpy.ones_like(_rnd_values)

        return rnd_measurements


    def _estimate_parallelized_helper(self, arg_instances, unknowns, parallel_verbosity):
        """
        Helper method for `estimate_MC_sampling` and `estimate_repeatedly` methods.
        """

        jobs = len(arg_instances)
        # do the jobs
        n_cpus = joblib.cpu_count()
        if jobs < n_cpus:
            n_jobs = jobs
        else:
            n_jobs = n_cpus
        with joblib.parallel_backend('loky', n_jobs=n_jobs):
            results = joblib.Parallel(verbose=parallel_verbosity)(map(joblib.delayed(self._parallel_estimate_wrapper), arg_instances))
        # collect returns
        repeated_estimates = {p : [] for p in unknowns.keys()}
        for result in results:
            for p in repeated_estimates.keys():
                repeated_estimates[p].append(result[0][p])
        return repeated_estimates, results


    def _get_all_parameters(self) -> dict:
        """
        Get all currently specified local parameters and their values.
        """

        parameters = {}
        for _p in self._parameter_manager._parameters:
            parameters[_p.local_name] = _p.value
        return {_p : parameters[_p] for _p in sorted(parameters.keys(), key=str.lower)}


    def _get_information_matrix_at_t(self, t:float, measurements:List[Measurement], estimates:dict, sensitivities:List[Sensitivity], replicate_id:str) -> numpy.ndarray:
        """
        Calculates Fisher information matrix a timepoint t.

        Arguments
        ---------
            t : float
                Timepoint at which the FIM is calculated
            measurements : List[Measurement]
                The measurements from which the parameters have been estimated. 
            estimates : dict
                The parameters which have been estimated from the measurements.
            sensitivities : List[Sensitivity]
                Sensitivities that have been calculated from the measurements and estimated parameters.

        Returns
        -------
            FIM_t : numpy.ndarray
                Fisher information matrix at timepoint t, has size n_parameters x n_parameters

        Raises
        ------
            AttributeError
                Measurement objects have no error property set.
        """

        if not Helpers.all_measurements_have_errors(measurements):
            raise AttributeError('Measurement errors property not set.')

        measured_responses = sorted(set([measurement.name for measurement in measurements]), key=str.lower)
        estimated_parameters = sorted(estimates.keys(), key=str.lower)
        S_t = numpy.full(shape=(len(measured_responses), len(estimated_parameters)), fill_value=numpy.nan)
        err = numpy.full(shape=len(measured_responses), fill_value=numpy.nan)
        for i, _measured_response in enumerate(measured_responses):
            for j, _parameter in enumerate(estimated_parameters):

                _sensitivity = Helpers.extract_time_series(
                    sensitivities, 
                    name=f'd({_measured_response})/d({_parameter})', 
                    replicate_id=replicate_id,
                )
                _measurement = Helpers.extract_time_series(
                    measurements, 
                    name=_measured_response, 
                    replicate_id=replicate_id,
                )

                # only account for this response if it has been measured
                if _measurement is not None and t in _measurement.timepoints:
                    S_t[i, j] = _sensitivity.values[numpy.argwhere(_sensitivity.timepoints==t)]
                    err[i] = _measurement.errors[numpy.argwhere(_measurement.timepoints==t)]
                else:
                    S_t[i, j] = 0
                    err[i] = numpy.inf

        Sigma_t = numpy.diag(numpy.square(err))
        FIM_t = S_t.T @ numpy.linalg.inv(Sigma_t) @ S_t
        return FIM_t


    def _get_parameters_for_replicate(self, replicate_id:str) -> OwnDict:
        """
        Converts the local parameter names into the parameter names for a specific replicate_id.

        Arguments
        ---------
            replicate_id : str
                The current replicate_id of interest.

        Returns
        -------
            OwnDict
        """

        return self._parameter_manager.get_parameters_for_replicate(replicate_id)


    def _get_valid_parameter_names(self) -> list:
        """
        Get a list of all currently valid parameter names, whose corresponding value is a float.
        Lists or vectors cannot be estimated.
        """

        valid_names = []
        for _parameter in self._parameter_manager._parameters:
            if _parameter.local_name not in valid_names:
                if isinstance(_parameter.value, (float, int)):
                    valid_names.append(_parameter.local_name)
        return valid_names



    def _loss_fun_scipy(self, 
                       guess:list, guess_dict:dict, metric:str, measurements:List[Measurement], 
                       handle_CVodeError:bool, verbosity_CVodeError:bool, 
                       ) -> float:
        """
        The objective function for parameter estimation called by the optimizer, 
        wrapping the Caretakers loss function for its use with the scipy optimizers.

        Arguments
        ---------
            guess : list or array: 
                The vector of parameter values, has the same order as in guess_dict.
            guess_dict : dict
                Maintains the meta-information on the guess values. Values correspond to `guess`.
            measurements : List[Measurement] 
                The data, from which the parameter estimation is performed. 
                Can provide a Measurement object for any model state or observation.

        Keyword arguments
        -----------------
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
            verbosity_CVodeError : bool
                Enables informative output during handling CVodeErrors.

        Returns
        -------
            loss : float
        """

        _guess_dict = {p : _guess for p, _guess in zip(guess_dict.keys(), guess)}

        return self.loss_function(
            _guess_dict, 
            metric, 
            measurements, 
            handle_CVodeError, 
            verbosity_CVodeError,
        )


    def loss_function(self, 
                 guess_dict:dict, metric:str, measurements:List[Measurement], 
                 handle_CVodeError:bool=True, verbosity_CVodeError:bool=False, 
                 ) -> float:
        """
        The objective function for parameter estimation called by the optimizer.

        Arguments
        ---------
            guess : list or array: 
                The vector of parameter values, has the same order as in guess_dict.
            guess_dict : dict
                The dictionary of parameters for which the loss shall be calculated.
            measurements : List[Measurement] 
                The data, from which the parameter estimation is performed. 
                Can provide a Measurement object for any model state or observation.

        Keyword arguments
        -----------------
            handle_CVodeError : bool
                Catches CVodeError raised by the solver, in order to not interrupt the estimations for toxic parameter values. 
            verbosity_CVodeError : bool
                Enables informative output during handling CVodeErrors.

        Returns
        -------
            loss : float

        Raises
        ------
            ValueError
                In case some parameter value are NaN or None, which are rare cases.
            CVodeError
                All of them, except for certain flags if these are handled.
        """

        if None in list(guess_dict.values()) or not any(numpy.isfinite(list(guess_dict.values()))):
            raise ValueError(f'Some unknowns have invalid values. {guess_dict}')

        self.set_parameters(guess_dict)
        losses = []
        for _id in self.replicate_ids:
            _simulator = self.simulators[_id]
            _parameters = self._get_parameters_for_replicate(_id)
            loss = _simulator._get_loss_for_minimzer(
                metric=metric,
                guess_dict=_parameters, 
                measurements=measurements, 
                handle_CVodeError=handle_CVodeError, 
                verbosity_CVodeError=verbosity_CVodeError, 
            )
            losses.append(loss)
        if not all(numpy.isnan(losses)):
            loss = numpy.nansum(losses)
        else:
            loss = numpy.nan
        return loss


    def _parallel_estimate_wrapper(self, kwargs):
        return self.estimate(**kwargs)


    def _propagate_parameters_through_simulators(self):
        """
        Sets parameter values for all simulators managing the different replicate_ids.
        """

        for _id in self.replicate_ids:
            _parameters = self._parameter_manager.get_parameters_for_replicate(_id)
            self.simulators[_id].set_parameters(_parameters)
