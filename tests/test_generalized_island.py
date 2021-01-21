import numpy as np
from typing import List
import pytest

import matplotlib as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from pyfoomb.caretaker import Caretaker
from pyfoomb.datatypes import Measurement
from pyfoomb.generalized_islands import ArchipelagoHelpers
from pyfoomb.generalized_islands import LossCalculator
from pyfoomb.generalized_islands import PygmoOptimizers
from pyfoomb.generalized_islands import ParallelEstimationInfo

from modelling_library import ModelLibrary

class StaticHelpers():

    data_single = [Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30], errors=[0.1, 0.2, 0.3])]
    data_multi = [Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30], errors=[0.1, 0.2, 0.3], replicate_id='1st')]
    unknowns = ['y00', 'y10']
    bounds = [(-100, 100), (-100, 100)]

@pytest.fixture
def caretaker_single():
    name = 'model01'
    return Caretaker(
        bioprocess_model_class=ModelLibrary.modelclasses[name],
        model_parameters=ModelLibrary.model_parameters[name],
        initial_values=ModelLibrary.initial_values[name],
    )

@pytest.fixture
def caretaker_multi():
    name = 'model01'
    return Caretaker(
        bioprocess_model_class=ModelLibrary.modelclasses[name],
        model_parameters=ModelLibrary.model_parameters[name],
        initial_values=ModelLibrary.initial_values[name],
        replicate_ids=['1st', '2nd']
    )

@pytest.fixture
def constrainted_loss_calculator():
    class OwnLossCalculator(LossCalculator):
            def constraint_1(self):
                p1 = self.current_parameters[StaticHelpers.unknowns[0]]
                return p1 < 0
            def constraint_2(self):
                p2 = self.current_parameters[StaticHelpers.unknowns[1]]
                return p2 > 0
            def constraint_3(self):
                p1 = self.current_parameters[StaticHelpers.unknowns[0]]
                p2 = self.current_parameters[StaticHelpers.unknowns[1]]
                return (p1 + p2) <= 1
            def check_constraints(self) -> List[bool]:
                return [self.constraint_1(), self.constraint_2(), self.constraint_3()]
    return OwnLossCalculator

@pytest.fixture
def evolutions_trail():
    return {
        'cum_runtime_min' : [1, 2, 3],
        'evo_time_min' : [0.5, 0.5, 0.4],
        'best_losses' : [300, 200, 100],
        'estimates_info' : [
            {'losses': np.array([302, 301, 300])},
            {'losses': np.array([202, 201, 200])},
            {'losses': np.array([102, 101, 100])}
        ],
    }


class TestLossCalculator():

    @pytest.mark.parametrize('metric', ['negLL', 'WSS', 'SS'])
    @pytest.mark.parametrize(
        'fitness_vector', 
        [
            [0, 0],
            [100, 100],
        ]
    )
    def test_fitness(self, caretaker_single, fitness_vector, metric):
        pg_problem = LossCalculator(
            StaticHelpers.unknowns, 
            StaticHelpers.bounds, 
            metric, 
            StaticHelpers.data_single, 
            caretaker_single.loss_function,
        )
        pg_problem.fitness(fitness_vector)
        pg_problem.gradient(fitness_vector)

    @pytest.mark.parametrize(
        'fitness_vector, inf_loss', 
        [
            ([-1, 1], False),
            ([0, 1], True),
            ([-1, 0], True),
            ([0, 0], True),
            ([1, 1], True),
            ([-1, 100], True),
        ]
    )
    def test_contrained_loss_calculator(self, caretaker_single, constrainted_loss_calculator, fitness_vector, inf_loss):
        pg_constraint_problem = constrainted_loss_calculator(
            StaticHelpers.unknowns, 
            StaticHelpers.bounds, 
            'negLL', 
            StaticHelpers.data_single, 
            caretaker_single.loss_function,
        )
        loss = pg_constraint_problem.fitness(fitness_vector)
        assert np.isinf(loss) == inf_loss


class TestPygmoOptimizers():

    @pytest.mark.parametrize('algo_name', list(PygmoOptimizers.optimizers.keys()))
    def test_get_algo_instance_defaults(self, algo_name):
        PygmoOptimizers.get_optimizer_algo_instance(name=algo_name)

    @pytest.mark.parametrize(
        'algo_name, kwargs', 
        [
            ('mbh', {'perturb' : 0.05, 'inner_stop_range' : 1e-3}),
            ('de1220', {'gen' : 10})
        ]
    )
    def test_get_algo_instance(self, algo_name, kwargs):
        PygmoOptimizers.get_optimizer_algo_instance(name=algo_name, kwargs=kwargs)


class TestParallelEstimationInfo():

    def test_properties_methods(self, evolutions_trail):
        est_info = ParallelEstimationInfo(archipelago='archipelago_mock', evolutions_trail=evolutions_trail)
        est_info.average_loss_trail
        est_info.best_loss_trail
        est_info.losses_trail
        est_info.std_loss_trail
        est_info.runtime_trail
        est_info.plot_loss_trail()
        est_info.plot_loss_trail(x_log=True)


class TestArchipelagoHelpers():

    @pytest.mark.parametrize('atol', [None, 1e-1])
    @pytest.mark.parametrize('rtol', [None, 1e-1])
    @pytest.mark.parametrize('curr_runtime', [None, 10])
    @pytest.mark.parametrize('max_runtime', [None, 5])
    @pytest.mark.parametrize('curr_evotime', [None, 10])
    @pytest.mark.parametrize('max_evotime', [None, 5])
    @pytest.mark.parametrize('max_memory_share', [0, 0.95])
    def test_check_evolution_stop(self, atol, rtol, curr_runtime, max_runtime, curr_evotime, max_evotime, max_memory_share):
        ArchipelagoHelpers.check_evolution_stop(
            current_losses=np.array([10.1, 10.2, 9.9]), 
            atol_islands=atol, 
            rtol_islands=rtol, 
            current_runtime_min=curr_runtime, 
            max_runtime_min=max_runtime,
            current_evotime_min=curr_evotime,
            max_evotime_min=max_evotime,
            max_memory_share=max_memory_share,
        )

    @pytest.mark.parametrize('report_level', [0, 1, 2, 3, 4])
    def test_report_evolution_results(self, report_level):
        reps = 2
        mock_evolution_results = {
            'evo_time_min' : [1]*reps,
            'best_losses' : [[1111, 1111]]*reps,
            'best_estimates' : [
                {'p1' : 1, 'p2' : 10}, 
            ],
            'estimates_info' : [
                {
                    'losses' : [1000, 1100, 1110, 1111],
                }
            ]*reps,   
        }
        ArchipelagoHelpers.report_evolution_result(mock_evolution_results, report_level)
        