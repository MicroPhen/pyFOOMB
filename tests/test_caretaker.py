
import numpy as np

import pytest

from pyfoomb import Measurement
from pyfoomb import ModelState
from pyfoomb import ParameterMapper

from pyfoomb.datatypes import Sensitivity
from pyfoomb.caretaker import Caretaker

import modelling_library
from modelling_library import ModelLibrary


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

class StaticHelpers():

    data_single = [Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30], errors=[0.1, 0.2, 0.3])]
    data_multi = [Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30], errors=[0.1, 0.2, 0.3], replicate_id='1st')]
    unknowns = ['y00']
    bounds = [(-100, 100)]


class TestBaseFunctionalities():

    @pytest.mark.parametrize('name', ModelLibrary.modelnames)
    def test_creation_manager_methods(self, name):

        model_class = ModelLibrary.modelclasses[name]
        model_parameters = ModelLibrary.model_parameters[name]
        initial_values = ModelLibrary.initial_values[name]
        mapping = ParameterMapper(replicate_id='3rd', global_name=list(model_parameters.keys())[0])

        # Single replicate caretaker, with no explictly named replicate_ids
        caretaker01 = Caretaker(
            bioprocess_model_class=model_class,
            model_parameters=model_parameters,
            initial_values=initial_values,
            model_name=name
        )
        # Cannot add another replicate_id to this Caretaker
        with pytest.raises(AttributeError):
            caretaker01.add_replicate('new_id')
        # Cannot apply a mapping to this Caretaker
        with pytest.warns(UserWarning):
            caretaker01.apply_mappings(mapping)

        # Caretaker with several replicate_ids
        caretaker02 = Caretaker(
            bioprocess_model_class=model_class,
            model_parameters=model_parameters,
            initial_values=initial_values,
            replicate_ids=['1st']
        )
        # Can add more replicate_ids
        caretaker02.add_replicate('2nd')
        # But cannot add an existing replicate_id (case-insensitive)
        with pytest.raises(ValueError):
            caretaker02.add_replicate('1ST')

        # Cannot use non-unique (case-insensitive) replicate_ids
        with pytest.raises(ValueError):
            Caretaker(
                bioprocess_model_class=model_class,
                model_parameters=model_parameters,
                initial_values=initial_values,
                replicate_ids=['1st', '1ST']
            )

        # When adding a new replicate_id, a corresponding mapping can be given
        caretaker02.add_replicate(replicate_id='3rd', mappings=[mapping])
        # But the replicate_ids must match
        with pytest.raises(KeyError):
            caretaker02.add_replicate(replicate_id='4th', mappings=[mapping])

    @pytest.mark.parametrize(
        'model_name, model_class, correct_initial_switches, autodetect_ok',
        [
            ('model03', modelling_library.Model03, [False], True),
            ('model06', modelling_library.Model06, [False]*2, True),
            ('model06', modelling_library.Model06_V02, [False]*3, True),
            ('model06', modelling_library.Model06_V03, [False]*3, False),
        ]
    )
    def test_autodetection_of_initial_switches(self, model_name, model_class, correct_initial_switches, autodetect_ok):
        model_parameters = ModelLibrary.model_parameters[model_name]
        initial_values = ModelLibrary.initial_values[model_name]
        initial_switches = ModelLibrary.initial_switches[model_name]
        rid = 'Test_id'

        if not autodetect_ok:
            with pytest.warns(UserWarning):
                Caretaker(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values, replicate_ids=[rid])
            caretaker = Caretaker(
                bioprocess_model_class=model_class, 
                model_parameters=model_parameters, 
                initial_values=initial_values, 
                initial_switches=correct_initial_switches,
                replicate_ids=[rid],
            )
        else:
            caretaker = Caretaker(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values, replicate_ids=[rid])
        for actual, expected in zip(caretaker.simulators[rid].bioprocess_model.initial_switches, correct_initial_switches):
                assert actual == expected

    def test_properties(self, caretaker_multi):
        # Get the current parameter mapping
        caretaker_multi.parameter_mapping
        # Setting optimizer kwargs must be a dict or None
        with pytest.raises(ValueError):
            caretaker_multi.optimizer_kwargs = ('strategy' , 'randtobest1exp')
        caretaker_multi.optimizer_kwargs = {'strategy' : 'randtobest1exp'}
        caretaker_multi.optimizer_kwargs = None
        assert caretaker_multi.optimizer_kwargs is None

    def test_simulate(self, caretaker_single, caretaker_multi):
        t = 24
        caretaker_single.simulate(t=t)
        caretaker_multi.simulate(t=t)
        caretaker_single.set_integrator_kwargs({'atol' : 1e-8, 'rtol' : 1e-8})
        caretaker_single.simulate(t=t, parameters={'rate0' : 100})

    def test_reset(self, caretaker_single, caretaker_multi):
        caretaker_single.reset()
        caretaker_multi.reset()

    @pytest.mark.parametrize(
        'guess_dict', 
        [
            {'rate0' : np.inf},
            {'rate0' : None},
            {'rate0' : np.inf}
        ]
    )
    def test_loss_function(self, caretaker_single, guess_dict):
        with pytest.raises(ValueError):
            caretaker_single.loss_function(
                guess_dict=guess_dict, 
                metric='negLL', 
                measurements=StaticHelpers.data_single,
            )


class TestEstimateMethods():

    data_single = StaticHelpers.data_single
    data_multi = StaticHelpers.data_multi
    unknowns = StaticHelpers.unknowns
    bounds = StaticHelpers.bounds

    def test_estimate(self, caretaker_single):
        # Now testing the different parameterizations for the estimate method
        caretaker_single.estimate(unknowns=self.unknowns, measurements=self.data_single, bounds=self.bounds)
        caretaker_single.estimate(unknowns=self.unknowns, measurements=self.data_single, bounds=self.bounds, report_level=4)
        # For the local minimizer, unknowns must be a dict containing the initial guesses
        caretaker_single.estimate(unknowns={self.unknowns[0] : 100}, measurements=self.data_single, use_global_optimizer=False, optimizer_kwargs={'disp' : True})
        # Measurements must be a list of type Measurement
        with pytest.raises(TypeError):
            caretaker_single.estimate(unknowns=self.unknowns, measurements=[ModelState(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30])], bounds=self.bounds)
        # As ususal, unknowns must be unique (case-insensitive)
        with pytest.raises(KeyError):
            caretaker_single.estimate(unknowns=['y0', 'Y0'], measurements=self.data_single, bounds=self.bounds*2)
        # Can estimate only parameters that are used with the model
        with pytest.raises(ValueError):
            caretaker_single.estimate(unknowns=['some_parameter'], measurements=self.data_single, bounds=self.bounds)
        # Must give initial guess for the local minimizer
        with pytest.raises(ValueError):
            caretaker_single.estimate(self.unknowns, measurements=self.data_single, use_global_optimizer=False)
        # The global optimizer requires bounds
        with pytest.raises(ValueError):
            caretaker_single.estimate(unknowns=self.unknowns, measurements=self.data_single)
        # Using optimizer_kwargs for the estimation, in case the Caretaker's property is not None, a warning is raises
        caretaker_single.optimizer_kwargs = {'strategy' : 'randtobest1exp'}
        with pytest.warns(UserWarning):
            caretaker_single.estimate(unknowns=self.unknowns, measurements=self.data_single, bounds=self.bounds, optimizer_kwargs={'strategy' : 'randtobest1exp'})

    def test_estimate_parallel(self, caretaker_multi):
        caretaker_multi.estimate_parallel(unknowns=self.unknowns, measurements=self.data_multi, bounds=self.bounds, evolutions=2, optimizers='compass_search')
        # Continue the estimation procedure
        _, est_info = caretaker_multi.estimate_parallel(unknowns=self.unknowns, measurements=self.data_multi, bounds=self.bounds, report_level=5, evolutions=2, optimizers='compass_search')
        caretaker_multi.estimate_parallel_continued(estimation_result=est_info, evolutions=121, report_level=1, rtol_islands=None)
        # As ususal, unknowns must be unique (case-insensitive)
        with pytest.raises(KeyError):
            caretaker_multi.estimate_parallel(unknowns=['y0', 'Y0'], measurements=self.data_multi, bounds=self.bounds*2)
        # Can estimate only parameters that are used with the model
        with pytest.raises(ValueError):
            caretaker_multi.estimate_parallel(unknowns=['some_parameter'], measurements=self.data_multi, bounds=self.bounds)
        # Length of bounds must match the length of unknowns
        with pytest.raises(ValueError):
            caretaker_multi.estimate_parallel(unknowns=self.unknowns, measurements=self.data_multi, bounds=self.bounds*2)

    @pytest.mark.parametrize(
        'arguments', 
        [
            {'evolutions' : 2},
            {'evolutions' : 2, 'reuse_errors_as_weights' : False},
            {'evolutions' : 2, 'jobs_to_save' : 1, 'report_level' : 6},
            {'evolutions' : 121, 'report_level' : 1, 'rtol_islands' : None},
            {'evolutions' : 121, 'report_level' : 2, 'rtol_islands' : None},
            {'evolutions' : 121, 'report_level' : 3, 'rtol_islands' : None},
        ]
    )    
    def test_estimate_parallel_MC_sampling_kwargs(self, caretaker_single, caretaker_multi, arguments):
        caretaker_multi.estimate_parallel_MC_sampling(
            unknowns=self.unknowns, 
            measurements=self.data_multi, 
            bounds=self.bounds, 
            mc_samples=2, 
            optimizers='compass_search', 
            **arguments,
        )

    def test_estimate_parallel_MC_sampling(self, caretaker_multi):
        # Length of bounds must match the length of unknowns
        with pytest.raises(ValueError):
            caretaker_multi.estimate_parallel_MC_sampling(unknowns=self.unknowns, measurements=self.data_multi, bounds=self.bounds*2, mc_samples=2)
        # Measurements must be a list of type Measurement
        with pytest.raises(TypeError):
            caretaker_multi.estimate_parallel_MC_sampling(unknowns=self.unknowns, measurements=[ModelState(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30])], bounds=self.bounds)

        # for MC sampling, measurements with errors must be used
        with pytest.raises(AttributeError):
            caretaker_multi.estimate_parallel_MC_sampling(
                unknowns=self.unknowns, 
                measurements=[Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30], replicate_id='1st')], 
                bounds=self.bounds, 
                mc_samples=2,
            )
        
        # As ususal, unknowns must be unique (case-insensitive)
        with pytest.raises(KeyError):
            caretaker_multi.estimate_parallel_MC_sampling(unknowns=['y0', 'Y0'], measurements=self.data_multi, bounds=self.bounds*2, mc_samples=2)
        
        # Can estimate only parameters that are used with the model
        with pytest.raises(ValueError):
            caretaker_multi.estimate_parallel_MC_sampling(unknowns=['some_parameter'], measurements=self.data_multi, bounds=self.bounds)
        
        # The number of islands must be >0
        with pytest.raises(ValueError):
            caretaker_multi.estimate_parallel_MC_sampling(unknowns=self.unknowns, measurements=self.data_multi, bounds=self.bounds, n_islands=1, mc_samples=2)
        with pytest.raises(ValueError):
            caretaker_multi.estimate_parallel_MC_sampling(unknowns=self.unknowns, measurements=self.data_multi, bounds=self.bounds, optimizers=['compass_search'], mc_samples=2)

        # Convergence should never be reached, meaning that there are no results
        empty_est = caretaker_multi.estimate_parallel_MC_sampling(
            unknowns=self.unknowns, 
            measurements=self.data_multi, 
            bounds=self.bounds, 
            evolutions=2,
            optimizers=['compass_search']*2,
            rtol_islands=None,
            report_level=4,
        )
        assert empty_est.empty

    def test_order_of_bounds(self, caretaker_multi):
        bounds = [(100, 100), (200, 200), (300, 300)] # By choosing the same lower and upper bounds, the estimation output is fixed
        unknowns = ['y10', 'y00', 'rate0']
        expected = {_u : _b[0] for _u, _b in zip(unknowns, bounds)}
        with pytest.warns(RuntimeWarning):
            estimates, _ = caretaker_multi.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_multi)
        assert estimates == expected


    def test_estimate_repeatedly(self, caretaker_single):
        """
        The estimate_repeatedly method will be deprecated in favor of the pygmo-based estimation methods
        """
        # Calling this method raises a warning
        with pytest.warns(PendingDeprecationWarning):
            caretaker_single.estimate_repeatedly(
                unknowns=self.unknowns, 
                measurements=self.data_single, 
                bounds=self.bounds, 
                rel_jobs=0.1,
                report_level=2
            )

        # Measurements must be a list of type Measurement
        with pytest.warns(PendingDeprecationWarning):
            with pytest.raises(TypeError):
                caretaker_single.estimate_repeatedly(
                    unknowns=self.unknowns, 
                    measurements=[ModelState(name='y0', timepoints=[1, 2, 3], 
                    values=[10, 20 ,30])], 
                    bounds=self.bounds,
                    rel_jobs=0.1,
                )

        # Optimizer output is suppressed for this method
        caretaker_single.optimizer_kwargs = {'disp' : True}
        with pytest.warns(PendingDeprecationWarning):
            caretaker_single.estimate_repeatedly(
                unknowns=self.unknowns, 
                measurements=self.data_single, 
                bounds=self.bounds, 
                rel_jobs=0.1,
            )

    def test_estimate_MC_sampling(self, caretaker_single):
        """
        The estimate_repeatedly method will be deprecated in favor of the pygmo-based estimation methods
        """
        # Calling this method raises a warning
        with pytest.warns(PendingDeprecationWarning):
            caretaker_single.estimate_MC_sampling(
                unknowns=self.unknowns, 
                measurements=self.data_single, 
                bounds=self.bounds, 
                rel_mc_samples=0.1,
                report_level=2
            )
        # Measurements must be a list of type Measurement
        with pytest.warns(PendingDeprecationWarning):
            with pytest.raises(TypeError):
                caretaker_single.estimate_MC_sampling(
                    unknowns=self.unknowns, 
                    measurements=[ModelState(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30])], 
                    bounds=self.bounds,
                    rel_mc_samples=0.1,
                )
        # As ususal, unknowns must be unique (case-insensitive)
        with pytest.warns(PendingDeprecationWarning):
            with pytest.raises(KeyError):
                caretaker_single.estimate_MC_sampling(unknowns=['y0', 'Y0'], measurements=self.data_single, bounds=self.bounds*2)
        # Can estimate only parameters that are used with the model
        with pytest.warns(PendingDeprecationWarning):
            with pytest.raises(ValueError):
                caretaker_single.estimate_MC_sampling(unknowns=['some_parameter'], measurements=self.data_single, bounds=self.bounds)
        # Must give initial guess for the local minimizer
        with pytest.warns(PendingDeprecationWarning):
            with pytest.raises(ValueError):
                caretaker_single.estimate_MC_sampling(self.unknowns, measurements=self.data_single, use_global_optimizer=False)
        # The global optimizer requires bounds
        with pytest.warns(PendingDeprecationWarning):
            with pytest.raises(ValueError):
                caretaker_single.estimate_MC_sampling(unknowns=self.unknowns, measurements=self.data_single)
        # Measurements must have errors to be able to sample
        with pytest.warns(PendingDeprecationWarning):
            with pytest.raises(AttributeError):
                caretaker_single.estimate_MC_sampling(
                    unknowns=self.unknowns, 
                    measurements=[Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30])], 
                    bounds=self.bounds,
                    rel_mc_samples=0.1,
                )
        # Optimizer output is suppressed for this method
        caretaker_single.optimizer_kwargs = {'disp' : True}
        with pytest.warns(PendingDeprecationWarning):
            caretaker_single.estimate_MC_sampling(unknowns=self.unknowns, measurements=self.data_single, bounds=self.bounds, rel_mc_samples=0.1)


class TestSensitivites():

    def test_get_sensitivies(self, caretaker_multi):
        caretaker_multi.get_sensitivities(measurements=StaticHelpers.data_multi)
        caretaker_multi.get_sensitivities(tfinal=24)
        caretaker_multi.get_sensitivities(tfinal=24, abs_h=1e-5)
        caretaker_multi.get_sensitivities(tfinal=24, responses=['y0'])
        caretaker_multi.get_sensitivities(tfinal=24, parameters=['y00'])
        caretaker_multi.get_sensitivities(tfinal=24, parameters={'y00' : 1000})
        # Include tfinal in the timepoints of the measurements
        caretaker_multi.get_sensitivities(measurements=StaticHelpers.data_multi, tfinal=24)
        # Must requrest known responses
        with pytest.raises(ValueError):
            caretaker_multi.get_sensitivities(tfinal=24, responses=['unknown_response'])
        # Must provide either measurements or tfinal
        with pytest.raises(ValueError):
            caretaker_multi.get_sensitivities()
        # Requested responses must be either a list or "all"
        with pytest.raises(TypeError):
            caretaker_multi.get_sensitivities(measurements=StaticHelpers.data_multi, responses=('y0'))
        # Requested responses must be unique (case-insensitive)
        with pytest.raises(ValueError):
            caretaker_multi.get_sensitivities(measurements=StaticHelpers.data_multi, responses=['y0', 'Y0'])
        # If used, must use a list of Measurement objects
        with pytest.raises(TypeError):
            caretaker_multi.get_sensitivities(measurements=[ModelState(name='y0', timepoints=[1, 2, 3], values=[10, 20 ,30], replicate_id='1st')])
        # If used, must request sensitivities for uniunique (case-insensitive) parameters
        with pytest.raises(ValueError):
            caretaker_multi.get_sensitivities(tfinal=24, parameters=['y00', 'Y00'])
        # If used, must request used parameters
        with pytest.raises(ValueError):
            caretaker_multi.get_sensitivities(tfinal=24, parameters=['some_parameter'])

    @pytest.mark.parametrize(
        'abs_h, rel_h, should_warn', 
        [
            (1e-3, None, False),
            (None, 1e-18, True),
            (1e-18, 1e-18, True),
        ],
    )
    def test_centr_diff(self, caretaker_single, abs_h, rel_h, should_warn):
        arg = (
            None, # replicated_id
            'y0', # response_i
            'y00', # parameter_j
            rel_h, # rel_h
            abs_h, # abs_h
            {'y00' : 100, 'y10' : 10}, # parameters
            [1, 2, 3], # t
        )
        if should_warn:
            with pytest.warns(UserWarning):
                caretaker_single._d_response_i_wrt_d_parameter_j_central_difference(arg)
        else:
            caretaker_single._d_response_i_wrt_d_parameter_j_central_difference(arg)

class TestMatrices():

    timepoints_a = [1, 2, 3]
    timepoints_b = [4, 5, 6]
    timepoints_c = [*timepoints_a, *timepoints_b]
    additional_timepoints = [10, 11, 12]
    estimates = {'p1' : 10, 'p2' : 100}

    def test_get_infomation_matrix_at_t(self, caretaker_single):
        
        # Create some mock measurements and sensitivities
        measurements = [
            Measurement(name='a', timepoints=self.timepoints_a, values=np.square(self.timepoints_a), errors=np.sqrt(self.timepoints_a)),
            Measurement(name='b', timepoints=self.timepoints_b, values=np.square(self.timepoints_b), errors=np.sqrt(self.timepoints_b)),
            Measurement(name='c', timepoints=self.timepoints_c, values=np.square(self.timepoints_c), errors=np.sqrt(self.timepoints_c)),
        ]
        sensitivities = [
            Sensitivity(timepoints=self.timepoints_a, values=np.square(self.timepoints_a), response='a', parameter='p1'),
            Sensitivity(timepoints=self.timepoints_b, values=np.square(self.timepoints_b), response='b', parameter='p1'),
            Sensitivity(timepoints=self.timepoints_c, values=np.square(self.timepoints_c), response='c', parameter='p1'),
            Sensitivity(timepoints=self.timepoints_a, values=np.square(self.timepoints_a), response='a', parameter='p2'),
            Sensitivity(timepoints=self.timepoints_b, values=np.square(self.timepoints_b), response='b', parameter='p2'),
            Sensitivity(timepoints=self.timepoints_c, values=np.square(self.timepoints_c), response='c', parameter='p2'),
        ]

        # The measurements are required to have errors
        with pytest.raises(AttributeError):
            caretaker_single._get_information_matrix_at_t(
                t=24, 
                measurements=[Measurement(name='a', timepoints=self.timepoints_a, values=np.square(self.timepoints_a))], 
                estimates=self.estimates, 
                sensitivities=sensitivities, 
                replicate_id=None,
            )

        # A FIM at any timepoint where no measurements have been collected should have no information
        FIM_add = np.sum(
            [
                caretaker_single._get_information_matrix_at_t(t=_t, measurements=measurements, estimates=self.estimates, sensitivities=sensitivities, replicate_id=None)
                for _t in self.additional_timepoints
            ]
        )
        assert np.sum(FIM_add.flatten()) == 0 # the overall sum must therefore be 0

        # Create some FIMs to compare
        FIM_a = np.sum(
            [
                caretaker_single._get_information_matrix_at_t(t=_t, measurements=measurements, estimates=self.estimates, sensitivities=sensitivities, replicate_id=None)
                for _t in self.timepoints_a
            ]
        )
        FIM_b = np.sum(
            [
                caretaker_single._get_information_matrix_at_t(t=_t, measurements=measurements, estimates=self.estimates, sensitivities=sensitivities, replicate_id=None)
                for _t in self.timepoints_b
            ]
        )
        FIM_c = np.sum(
            [
                caretaker_single._get_information_matrix_at_t(t=_t, measurements=measurements, estimates=self.estimates, sensitivities=sensitivities, replicate_id=None)
                for _t in self.timepoints_c
            ]
        )
        # FIM_a and FIM_b should have the same information content as FIM_c has
        assert np.allclose(FIM_a+FIM_b, FIM_c)
        # FIM_a, as well as FIM_b have less information content as FIM_c
        assert np.less_equal(FIM_a, FIM_c)
        assert np.less_equal(FIM_b, FIM_c)

    def test_get_matrices_and_related(self, caretaker_single):
        # Create some mock variables
        measurements = [
            Measurement(name='y0', timepoints=[1, 2, 3], values=[10, 20, 30], errors=[0.1, 0.2, 0.3])
        ]
        modelstates = [
            ModelState(name='y0', timepoints=[1, 2, 3], values=[10, 20, 30])
        ]
        estimates = {'y00' : 90}

        matrices = caretaker_single.get_parameter_matrices(estimates=estimates, measurements=measurements)
        caretaker_single.get_optimality_criteria(Cov=matrices['Cov'], report_level=1)
        caretaker_single.get_information_matrix(estimates=estimates, measurements=measurements)
        caretaker_single.get_parameter_uncertainties(estimates=estimates, measurements=measurements, report_level=1)
        # Only lists of measurements and sensitivites can be used for the respective arguments
        with pytest.raises(TypeError):
            caretaker_single.get_information_matrix(estimates=estimates, measurements=modelstates)
        with pytest.raises(TypeError):
            caretaker_single.get_information_matrix(estimates=estimates, measurements=measurements, sensitivities=modelstates)
        with pytest.raises(TypeError):
            caretaker_single.get_parameter_uncertainties(estimates=estimates, measurements=modelstates)
        with pytest.raises(TypeError):
            caretaker_single.get_parameter_uncertainties(estimates=estimates, measurements=measurements, sensitivities=modelstates)
        with pytest.raises(TypeError):
            caretaker_single.get_parameter_matrices(estimates=estimates, measurements=modelstates)
        with pytest.raises(TypeError):
            caretaker_single.get_parameter_matrices(estimates=estimates, measurements=measurements, sensitivities=modelstates)

        # This measurement carries no informatio w.r.t. the estimates
        with pytest.warns(UserWarning):
            uncerts = caretaker_single.get_parameter_uncertainties(
                estimates=estimates, 
                measurements=[Measurement(name='y1', timepoints=[1, 2, 3], values=[10, 20, 30], errors=[0.1, 0.2, 0.3])],
            )
        # Therefore, the derivced stdErr are infinite
        assert np.isinf(uncerts['StdErrs'])
