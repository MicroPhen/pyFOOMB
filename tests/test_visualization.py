import matplotlib
matplotlib.use('agg')
 
import numpy as np
import pytest

from pyfoomb.caretaker import Caretaker
from pyfoomb.datatypes import Measurement
from pyfoomb.datatypes import TimeSeries
from pyfoomb.visualization import Visualization
from pyfoomb.visualization import VisualizationHelpers

from modelling_library import ModelLibrary

class StaticHelpers():

    data_single = [Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 10 ,10], errors=[0.1, 0.2, 0.3])]
    data_multi = [
        Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 10 ,10], errors=[0.1, 0.2, 0.3], replicate_id='1st'),
        Measurement(name='y0', timepoints=[1, 5, 10], values=[10, 10 ,10], errors=[0.1, 0.2, 0.3], replicate_id='2nd'),
    ]
    many_time_series_1 = [
        [TimeSeries(name='T1', timepoints=[1, 2, 3, 4], values=[10, 10, 10, 10], replicate_id='1st')],
        [TimeSeries(name='T1', timepoints=[1, 2, 3], values=[10, 10, 10], replicate_id='1st')],
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

    @pytest.mark.filterwarnings('ignore:All-NaN slice encountered')
    @pytest.mark.parametrize(
        'time_series' ,
        [
            [StaticHelpers.data_multi]*2,
            StaticHelpers.many_time_series_1,
        ]
    )
    def test_show_kinetic_data_many(self, time_series):
        figsax = Visualization.show_kinetic_data_many(time_series=time_series)
        for _key in figsax:
            for _line in figsax[_key][1][0].lines:
                for _value in _line._y:
                    assert _value == 10 or np.isnan(_value)

    @pytest.mark.parametrize('measurements', [StaticHelpers.data_single, StaticHelpers.data_multi])
    def test_compare_estimates(self, caretaker_single, measurements):
        Visualization.compare_estimates(
            parameters={_p : 10 for _p in StaticHelpers.unknowns},
            measurements=measurements,
            caretaker=caretaker_single,
        )

    @pytest.mark.parametrize(
        'caretaker, data', 
        [
            (caretaker_multi, StaticHelpers.data_multi),
            (caretaker_single, StaticHelpers.data_single),
        ]
    )

    def test_compare_estimates_many(self, caretaker, data, request):
        caretaker = request.getfixturevalue(caretaker)
        Visualization.compare_estimates_many(
            parameter_collections={_p : [10]*3 for _p in StaticHelpers.unknowns},
            measurements=StaticHelpers.data_multi,
            caretaker=caretaker,
            show_measurements_only=True,
        )

    def test_compare_estimates_many_single(self, caretaker_multi, data):
        Visualization.compare_estimates_many(
            parameter_collections={_p : [10]*3 for _p in StaticHelpers.unknowns},
            measurements=StaticHelpers.data_single,
            caretaker=caretaker_multi,
            show_measurements_only=True,
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
