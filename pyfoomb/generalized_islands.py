import contextlib
import io
import joblib
from matplotlib import pyplot
import numpy
import psutil
from typing import Callable, Dict, List, Tuple

from assimulo.solvers.sundials import CVodeError
import pygmo

from .datatypes import Measurement


class LossCalculator():
    """
    Defines the objective that is used to create an pygmo problem instance. 
    See pygmo docu for further information (e.g., https://esa.github.io/pagmo2/docs/python/tutorials/coding_udp_simple.html).
    """

    def __init__(self, 
                 unknowns:list, bounds:list, metric:str, measurements:List[Measurement], caretaker_loss_fun:Callable, 
                 handle_CVodeError:bool=True, verbosity_CVodeError:bool=False,
                 ):
        """
        Arguments
        ---------
            unknowns : list
                The unknowns to be estimated.
            bounds : list
                Corresponding list of (upper, lower) bounds.
            metric : str
                The loss metric, which is minimized for model calibration.
            measurements : List[Measurement]
                The measurements for which  the model will be calibrated.
            caretaker_loss_fun : Callable
                The Caretaker's loss function.

        Keyword arguments
        -----------------
            handle_CVodeError : bool
                to handle arising CVodeErrors.
                Default is True, which returns an infinite loss.
            verbosity_CVodeError : bool
                To report about handled CVodeErrros.
                Default is False.
        """

        self.unknowns = unknowns
        self.metric = metric
        self.lower_bounds = [_bounds[0] for _bounds in bounds]
        self.upper_bounds = [_bounds[1] for _bounds in bounds]
        self.measurements = measurements
        self.caretaker_loss_fun = caretaker_loss_fun
        self.handle_CVodeError = handle_CVodeError
        self.verbosity_CVodeError = verbosity_CVodeError


    @property
    def current_parameters(self) -> dict:
        return self._current_parameters


    @current_parameters.setter
    def current_parameters(self, value):
        self._current_parameters = {unknown : _x for unknown, _x in zip(self.unknowns, value)}


    def check_constraints(self) -> List[bool]:
        return [True]


    def get_model_loss(self) -> float:
        """
        Calculates the loss for the current parameter values.

        Returns
        -------
            loss : float
        """
        try:
            loss = self.caretaker_loss_fun(
                self.current_parameters, 
                self.metric, 
                self.measurements, 
                self.handle_CVodeError, 
                self.verbosity_CVodeError, 
            )
            if numpy.isnan(loss):
                loss = numpy.inf
        except CVodeError:
            loss = numpy.inf
        return loss


    def fitness(self, x) -> List[float]:
        """
        Method for fitness calculation, as demanded by the pygmo package. 
        """

        # (1): Create the current parameter dictionary from the current guess vector
        self.current_parameters = x

        # (2) Check if any constraint is violated
        constraints_ok = self.check_constraints()
        if not all(constraints_ok):
            loss = numpy.inf

        # (3) Evaluate the Caretakers objective function only is no constraints have been violated
        else:
            loss = self.get_model_loss()

        # TODO: Regularization can be added to the loss here

        return [loss]


    def get_bounds(self) -> tuple:
        """
        Method for checking the parameter bounds, as demanded by the pygmo package. 
        """

        return (self.lower_bounds, self.upper_bounds)


    def gradient(self, x):
        """
        Method for gradient calculation, as demanded by the pygmo package for some pygmo optimizers.
        """

        return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)


class PyfoombArchipelago(pygmo.archipelago):
    """
    An archipelago subclass, extended with specific properties needed for the pyFOOMB package.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_info = None
        self.finished = None
        self.problem = LossCalculator

    @property
    def mc_info(self):
        return self._mc_info

    @mc_info.setter
    def mc_info(self, value):
        self._mc_info = value

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, value):
        self._finished = value

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value) -> LossCalculator:
        self._problem = value


class PygmoOptimizers():
    """
    Static class that conveniently handles pygmo optimizers, as well as default kwargs that are considered useful.
    """

    optimizers = {
        'bee_colony' : pygmo.bee_colony,
        #'cmaes' : pygmo.cmaes,
        'compass_search' : pygmo.compass_search,
        'de' : pygmo.de,
        'de1220' : pygmo.de1220,
        'gaco' : pygmo.gaco,
        'ihs' : pygmo.ihs,
        'maco' : pygmo.maco,
        'mbh' : pygmo.mbh,
        'moead' : pygmo.moead,
        'nlopt' : pygmo.nlopt,
        'nsga2' : pygmo.nsga2,
        'nspso' : pygmo.nspso,
        'pso' : pygmo.pso,
        'pso_gen' : pygmo.pso_gen,
        'sade' : pygmo.sade,
        'sea' : pygmo.sea,
        'sga' : pygmo.sga,
        'simulated_annealing' : pygmo.simulated_annealing,
        #'xnes' : pygmo.xnes,
    }


    default_kwargs = {
        'bee_colony' : {'limit' : 2, 'gen' : 10},
        #'cmaes' : {'gen' : 10, 'force_bounds' : False, 'ftol' : 1e-8, 'xtol' : 1e-8},
        'compass_search' : {'max_fevals' : 100, 'start_range' : 1, 'stop_range' : 1e-6},
        'de' : {'gen' : 10, 'ftol' : 1e-8, 'xtol' : 1e-8},
        'de1220' : {'gen' : 10, 'variant_adptv' : 2, 'ftol' : 1e-8, 'xtol' : 1e-8},
        'gaco' : {'gen' : 10},
        'ihs' : {'gen' : 10*4},
        'maco' : {'gen' : 10},
        'mbh' : {'algo' : 'compass_search', 'perturb' : 0.1, 'stop' : 2},
        'moead' : {'gen' : 10},
        'nlopt' : {'solver' : 'lbfgs'},
        'nsga2' : {'gen' : 10},
        'nspso' : {'gen' : 10},
        'pso' : {'gen' : 10},
        'pso_gen' : {'gen' : 10},
        'sade' : { 'gen' : 10, 'variant_adptv' : 2, 'ftol' : 1e-8, 'xtol' : 1e-8},
        'sea' : {'gen' : 10*4},
        'sga' : {'gen' : 10},
        'simulated_annealing' : {},
        #'xnes' : {'gen' : 10, 'ftol' : 1e-8, 'xtol' : 1e-8, 'eta_mu' : 0.05},
    }


    @staticmethod
    def get_optimizer_algo_instance(name:str, kwargs:dict=None) -> pygmo.algorithm:
        """
        Get an pygmo optimizer instance.
        In case 'mbh' is chosen, key for the inner algorithm corresponds the respective names prefixed with `inner_`.

        Arguments
        ---------
            name : str
                The name of the desired optimizer.

        Keyword arguments
        -----------------
            kwargs : dict
                Additional kwargs for creation of the optimizer instance, 
                beside the default kwargs (see corresponding class attribute).

        Returns
        -------
            pygmo.algorithm

        Raises
        ------
            ValueError
                Argument `name` specifies an unsupported optimizer.
        """

        if name not in PygmoOptimizers.optimizers.keys():
            raise ValueError(f'Unsupported optimizer {name} chosen. Valid choices are {PygmoOptimizers.optimizers.keys()}')

        _kwargs = {}
        if name in PygmoOptimizers.default_kwargs.keys():
            _kwargs.update(PygmoOptimizers.default_kwargs[name])
        if kwargs is not None:
            for _kwarg in kwargs:
                _kwargs[_kwarg] = kwargs[_kwarg]

        if name == 'mbh':
            # (1) Get inner algorithm
            _inner_algo = _kwargs['algo']
            _inner_kwargs = {}
            _outer_kwargs = {}
            for key in _kwargs:
                if key.startswith('inner_'):
                    _inner_kwargs[key[6:]] = _kwargs[key]
                else:
                    _outer_kwargs[key] = _kwargs[key]

            _algo_instance = PygmoOptimizers.get_optimizer_algo_instance(_inner_algo, _inner_kwargs)
            # (2) get new kwargs, cleaned from inner kwargs
            _kwargs = _outer_kwargs
            _kwargs['algo'] = _algo_instance

        return pygmo.algorithm(PygmoOptimizers.optimizers[name](**_kwargs))


class ParallelEstimationInfo():

    def __init__(self, archipelago:PyfoombArchipelago, evolutions_trail:dict=None):
        """
        Arguments
        ---------
            archipelago : PyfoombArchipelago
                The archipelago for which the evolutions have been run.

        Keyword arguments
        -----------------
            evolutions_trail : dict
                Information about previous evolutions run with the archipelago.
                Default is None, which causes creation of a new dictionary.
        """
        self.archipelago = archipelago
        if evolutions_trail is None:
            evolutions_trail = {}
            evolutions_trail['cum_runtime_min'] = []
            evolutions_trail['evo_time_min'] = []
            evolutions_trail['best_losses'] = []
            evolutions_trail['best_estimates'] = []
            evolutions_trail['estimates_info'] = []
        self.evolutions_trail = evolutions_trail

    @property
    def runtime_trail(self):
        return numpy.array(self.evolutions_trail['cum_runtime_min'])


    @property
    def evotime_trail(self):
        return numpy.cumsum(self.evolutions_trail['evo_time_min'])


    @property
    def losses_trail(self):
        return numpy.array([_info['losses'].flatten() for _info in self.evolutions_trail['estimates_info']])


    @property
    def best_loss_trail(self):
        return numpy.min(self.losses_trail, axis=1)


    @property
    def average_loss_trail(self) -> numpy.ndarray:
        return numpy.mean(self.losses_trail, axis=1)


    @property
    def std_loss_trail(self) -> numpy.ndarray:
        return numpy.std(self.losses_trail, axis=1, ddof=1)


    @property
    def estimates(self) -> dict:
        return ArchipelagoHelpers.estimates_from_archipelago(self.archipelago)


    def plot_loss_trail(self, x_log:bool=True):
        """
        Shows the progression of the loss during the estimation process, more specifically the development 
        of the best loss, the average loss  and the correopnding CV among the parallel optimizations.

        Keyword arguments
        -----------------
            x_log : bool
                To show the x-axis (the runtime) in log scale or not.
                Default is True.

        Returns
        -------
            fig : The figure object
            ax : The axis object
        """

        fig, ax = pyplot.subplots(nrows=2, ncols=1, dpi=100, figsize=(10, 5), sharex=True)
        ax[0].plot(self.evotime_trail, self.best_loss_trail, marker='.', linestyle='--', label='Best', zorder=2)
        ax[0].plot(self.evotime_trail, self.average_loss_trail, marker='.', linestyle='--', label='Average', zorder=1)
        ax[0].set_ylabel('Loss', size=14)
        ax[1].plot(
            self.evotime_trail, numpy.abs(self.std_loss_trail/self.average_loss_trail*100), 
            marker='.', linestyle='--', label='CV of losses',
            )
        ax[1].set_ylabel('CV in %', size=14)
        ax[1].set_xlabel('Cumulated evolution time in min', size=14)
        for _ax in ax.flat:
            _ax.legend(frameon=False)
            _ax.xaxis.set_tick_params(labelsize=12)
            _ax.yaxis.set_tick_params(labelsize=12)
            if x_log:
                _ax.set_xscale('log')
        fig.tight_layout()
        return fig, ax


class ArchipelagoHelpers():

    @staticmethod
    def estimates_from_archipelago(archipelago:PyfoombArchipelago) -> dict:
        """
        Extracts the current estimated values for the optimization probelm of an archipelago.

        Arguments
        ---------
            archipelago : PyfoombArchipelago
                The evolved archipelago.

        Returns
        -------
            dict : The current estimates.
        
        """

        unknowns = ArchipelagoHelpers.problem_from_archipelago(archipelago).unknowns
        best_idx = numpy.argmin(numpy.array(archipelago.get_champions_f()).flatten())
        estimates = {
            _unknown : _x 
            for _unknown, _x in zip(unknowns, archipelago[int(best_idx)].get_population().champion_x)
        }
        return estimates.copy() # maybe a deep copy needed?


    @staticmethod
    def problem_from_archipelago(archipelago:PyfoombArchipelago) -> LossCalculator:
        """
        Extracts the optimization problem from an archipelago, implemented as (subclass of) LossCalculator.

        Arguments
        ---------
            archipelago : PyfoombArchipelago
                The evolved archipelago.

        Returns
        -------
            LossCalculator
        """

        return archipelago[0].get_population().problem.extract(archipelago.problem)


    @staticmethod
    def create_population(pg_problem, pop_size, seed):
        return pygmo.population(pg_problem, pop_size, seed=seed)

    @staticmethod 
    def parallel_create_population(arg):
        pg_problem, pop_size, seed = arg
        return ArchipelagoHelpers.create_population(pg_problem, pop_size, seed)

    @staticmethod
    def create_archipelago(unknowns:list, 
                           optimizers:list, 
                           optimizers_kwargs:list, 
                           pg_problem:pygmo.problem, 
                           rel_pop_size:float,
                           archipelago_kwargs:dict, 
                           log_each_nth_gen:int, 
                           report_level:int,
                           ) -> PyfoombArchipelago:
        """
        Helper method for parallelized estimation using the generalized island model.
        Creates the archipelago object for running several rounds of evolutions.

        Arguments
        ---------
            unknowns : list
                The unknowns, sorted alphabetically and case-insensitive. 
            optimizers : list
                A list of optimizers to be used on individual islands. 
            optimizers_kwargs : list
                A list of corresponding kwargs.
            pg_problem : pygmo.problem
                An pygmo problem instance.
            archipelago_kwargs : dict
                Additional kwargs for archipelago creation.
            log_each_nth_gen : int
                Specifies at which each n-th generation the algorithm stores logs. 
            report_level : int
                Prints information on the archipelago creation for values >= 1.

        Returns
        -------
            archipelago : PyfoombArchipelago
        """

        _cpus = joblib.cpu_count()

        # There is one optimizer with a set of kwargs
        if len(optimizers) == 1 and len(optimizers_kwargs) == 1:
            optimizers = optimizers * _cpus
            optimizers_kwargs = optimizers_kwargs * _cpus
        # Several optimizers with the same kwargs
        elif len(optimizers) > 1 and len(optimizers_kwargs) == 1:
            optimizers_kwargs = optimizers_kwargs * len(optimizers)
        # Several kwargs for the same optimizer
        elif len(optimizers) == 1 and len(optimizers_kwargs) > 1:
            optimizers = optimizers * len(optimizers_kwargs)
        elif len(optimizers) != len(optimizers_kwargs):
            raise ValueError('Number of optimizers does not match number of corresponding kwarg dicts')

        # Get the optimizer intances
        algos = [
            PygmoOptimizers.get_optimizer_algo_instance(
                name=_optimizers, kwargs=_optimizers_kwargs
            ) 
            for _optimizers, _optimizers_kwargs in zip(optimizers, optimizers_kwargs)
        ]

        # Update number of islands
        n_islands = len(algos)      

        if report_level >= 1:
            print(f'Creating archipelago with {n_islands} islands. May take some time...')

        pop_size = int(numpy.ceil(rel_pop_size*len(unknowns))) 
        prop_create_args = (
            (pg_problem, pop_size, seed*numpy.random.randint(0, 1e4))
            for seed, pop_size in enumerate([pop_size] * n_islands)
        )
        try:
            parallel_verbose = 0 if report_level == 0 else 1
            with joblib.parallel_backend('loky', n_jobs=n_islands):
                pops = joblib.Parallel(verbose=parallel_verbose)(map(joblib.delayed(ArchipelagoHelpers.parallel_create_population), prop_create_args))
        except Exception as ex:
            print(f'Parallelized archipelago creation failed, falling back to sequential\n{ex}')
            pops = (ArchipelagoHelpers.parallel_create_population(prop_create_arg) for prop_create_arg in prop_create_args)

        # Now create the empyty archipelago
        if not 't' in archipelago_kwargs.keys():
            archipelago_kwargs['t'] = pygmo.fully_connected()
        archi = PyfoombArchipelago(**archipelago_kwargs)
        archi.set_migrant_handling(pygmo.migrant_handling.preserve)
        
        # Add the populations to the archipelago and wait for its construction
        with contextlib.redirect_stdout(io.StringIO()):
            for _pop, _algo in zip(pops, algos):
                if log_each_nth_gen is not None:
                    _algo.set_verbosity(int(log_each_nth_gen))
                _island = pygmo.island(algo=_algo, pop=_pop, udi=pygmo.mp_island())
                archi.push_back(_island)
        archi.wait_check()

        return archi


    @staticmethod
    def extract_archipelago_results(archipelago:PyfoombArchipelago) -> Tuple[dict, float, dict]:
        """
        Get the essential and further informative results from an archipelago object.

        Arguments
        ---------
            archipelago : PyfoombArchipelago
                The archipelago object after finished evolution.

        Returns
        -------
            Tuple[dict, float, dict]
                The best estimates as dict, according to the best (smallest) loss.
                The best loss.
                A dictionary with several informative results.
        """

        estimates_info = {}   
        best_estimates = ArchipelagoHelpers.estimates_from_archipelago(archipelago)
        unknowns = list(best_estimates.keys())

        best_idx = numpy.argmin(numpy.array(archipelago.get_champions_f()).flatten())
        best_loss = float(archipelago[int(best_idx)].get_population().champion_f)

        estimates = {
            _unknown : _x 
            for _unknown, _x in zip(unknowns, numpy.array([island.get_population().champion_x for island in archipelago], dtype=float).T)
        }
        losses = numpy.array([float(island.get_population().champion_f) for island in archipelago], dtype=float)
 
        estimates_info['best_estimates'] = best_estimates
        estimates_info['best_loss'] = best_loss
        estimates_info['estimates'] = estimates
        estimates_info['losses'] = losses

        return best_estimates, best_loss, estimates_info


    @staticmethod
    def check_evolution_stop(current_losses:numpy.ndarray, 
                             atol_islands:float, rtol_islands:float, 
                             current_runtime_min:float, max_runtime_min:float,
                             current_evotime_min:float, max_evotime_min:float,
                             max_memory_share:float,
                             ) -> dict:
        """
        Checks if losses between islands have been sufficiently converged.

        Arguments
        ---------
            current_losses : numpy.ndarray
                The best losses of all islands after an evolution.
            atol_islands : float
                stop_criterion = atol_islands + rtol_islands * numpy.abs(numpy.mean(current_losses))
            rtol_islands : float
                stop_criterion = atol_islands + rtol_islands * numpy.abs(numpy.mean(current_losses))
            current_runtime : float
                The current runtime in min of the estimation process after a completed evolution.
            max_runtime : float
                The maximal runtime in min the estimation process can take.
            max_memory_share : float
                The maximum relative memory occupation for which evolutions are run

        Returns
        -------
            stopping_criteria : dict
        """

        stopping_criteria = {
            'convergence' : False,
            'max_runtime' : False,
            'max_evotime' : False,
            'max_memory_share' : False,
        }

        # Check convergence
        if atol_islands is None:
            atol_islands = 0.0
        if rtol_islands is None:
            rtol_islands = 0.0
        
        _stop_criterion = atol_islands + rtol_islands * numpy.abs(numpy.mean(current_losses))
        _abs_std = numpy.std(current_losses, ddof=1)
        if _abs_std < _stop_criterion:
            stopping_criteria['convergence'] = True

        # Check runtime
        if (current_runtime_min is not None) and (max_runtime_min is not None) and (current_runtime_min > max_runtime_min):
            stopping_criteria['max_runtime'] = True

        # Check evolution time
        if (current_evotime_min is not None) and (max_evotime_min is not None) and (current_evotime_min > max_evotime_min):
            stopping_criteria['max_evotime'] = True

        # Check memory occupation
        curr_memory_share = psutil.virtual_memory().percent/100
        if curr_memory_share > max_memory_share:
            stopping_criteria['max_memory_share'] = True

        return stopping_criteria


    @staticmethod
    def report_evolution_result(evolutions_results:dict, report_level:int):
        """
        Helper method for parallel estimation method to report progress.

        Arguments
        ---------
            evolutions_result : dict
                Contains information on the result of an evolution.
            report_level : int
                Controls the output that is printed.
                2 = prints the best loss, as well as information about archipelago creation and evolution. 
                3 = prints additionally average loss among all islands, and the runtime of the evolution.
                4 = prints additionally the parameter values for the best loss, and the average parameter values 
                    among the champions of all islands in the `archipelago` after the evolutions. 
        """

        if report_level < 2:
            return

        if report_level >= 2:
            _evolution = len(evolutions_results['evo_time_min'])
            print(f'-------------Finished evolution {_evolution}-------------')
            _best_loss = evolutions_results['best_losses'][-1]
            print(f'Current best loss: {_best_loss}')

        if report_level >= 3:
            _estimates_info = evolutions_results['estimates_info'][-1]
            _mean = numpy.mean(_estimates_info['losses'])
            _std = numpy.std(_estimates_info['losses'], ddof=1)
            _cv = numpy.abs(_std/_mean*100)
            print(f'Average loss among the islands: {_mean:.6f} +/- {_std:.6f} ({_cv:.6f} %)')
        
        if report_level >= 4:
            _evo_time_min = evolutions_results['evo_time_min'][-1]
            print(f'Run time for this evolution was {_evo_time_min:.2f} min')
