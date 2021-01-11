
import pytest

from pyfoomb.caretaker import Caretaker
from pyfoomb.datatypes import Measurement
from pyfoomb.visualization import Visualization
from pyfoomb.visualization import VisualizationHelpers

from modelling_library import ModelLibrary

class StaticHelpers():

    data_single = [Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30], errors=[0.1, 0.2, 0.3])]
    data_multi = [
        Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30], errors=[0.1, 0.2, 0.3], replicate_id='1st'),
        Measurement(name='y0', timepoints=[1, 5, 10], values=[10, 20 ,30], errors=[0.1, 0.2, 0.3], replicate_id='2nd'),
    ]
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

class TestVisualizationHelpers():

    @pytest.mark.parametrize('n', [10, 20, 21, 10.5, 10.6, 20.5, 20.6])
    def test_colors(self, n):
        VisualizationHelpers.get_n_colors(n)


class TestVisualization():

    @pytest.mark.parametrize('measurements', [StaticHelpers.data_single, StaticHelpers.data_multi])
    def test_show_kinetic_data(self, measurements):
        Visualization.show_kinetic_data(time_series=measurements)

    def test_show_kinetic_data_many(self):
        Visualization.show_kinetic_data_many(time_series=[StaticHelpers.data_multi]*2)

    @pytest.mark.parametrize('measurements', [StaticHelpers.data_single, StaticHelpers.data_multi])
    def test_compare_estimates(self, caretaker_single, measurements):
        Visualization.compare_estimates(
            parameters={_p : 10 for _p in StaticHelpers.unknowns},
            measurements=measurements,
            caretaker=caretaker_single,
        )

    @pytest.mark.parametrize('measurements', [StaticHelpers.data_single, StaticHelpers.data_multi])
    def test_compare_estimates_many(self, caretaker_single, measurements):
        Visualization.compare_estimates_many(
            parameters={_p : [10]*3 for _p in StaticHelpers.unknowns},
            measurements=measurements,
            caretaker=caretaker_single,
        )

    @pytest.mark.parametrize('estimates', [None, {'p1': 2.5, 'p2' : 5.5}])
    def test_show_parameter_distributions(self, estimates):
        Visualization.show_parameter_distributions(
            parameter_collections={
                'p1' : [1, 2, 3],
                'p2' : [4, 5, 6]
            },
            estimates=estimates,
        )

