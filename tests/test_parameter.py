
import pytest

from pyfoomb.parameter import Parameter
from pyfoomb.parameter import ParameterManager
from pyfoomb.parameter import ParameterMapper


class TestParameter():

    @pytest.mark.parametrize(
        "local_name, value", 
        [
            (None, None),
            ('p_local', None),
            ('p_local', 1000)
        ],
    )
    def test_init(self, local_name, value):
        Parameter(global_name='p_global', replicate_id='1st', local_name=local_name, value=value)


class TestParameterMapper():

    def test_init(self):
        replicate_id = '1st'
        global_name = 'p_global'
        local_name = 'p_local'
        value = 1

        ParameterMapper(replicate_id, global_name, local_name, value)

        # Without specifying a local name, this is build from the global name and replicate_id
        pm = ParameterMapper(replicate_id, global_name)
        assert pm.local_name == f'{global_name}_{replicate_id}'
        # The value defaults to None
        assert pm.value is None

        # Must use a local name when mapping shall be applied to all or several replicate_ids
        with pytest.raises(ValueError):
            ParameterMapper(replicate_id='all', global_name=global_name)
        with pytest.raises(ValueError):
            ParameterMapper(replicate_id=['1st', '2nd'], global_name=global_name)


class TestParameterManager():

    replicate_ids = ['1st', '2nd']
    parameters = {'p1' : 1, 'p2' : 2}

    @pytest.fixture()
    def parameter_manager(self):
        return ParameterManager(replicate_ids=self.replicate_ids, parameters=self.parameters)

    @pytest.fixture()
    def mappings(self):
        return [
            ParameterMapper(_replicate_id, _global_parameter)
            for _replicate_id in self.replicate_ids
            for _global_parameter in self.parameters
        ]

    def test_init(self, parameter_manager):
        # Must provide case-senstitive unique replicate_ids
        with pytest.raises(ValueError):
            ParameterManager(replicate_ids=['1st', '1ST'], parameters=self.parameters)
        # Can set replicate_ids only during instantiation
        with pytest.raises(AttributeError):
            parameter_manager.replicate_ids = ['1st', '2nd']
        # Can set global parameters only during instantiation
        with pytest.raises(AttributeError):
            parameter_manager.global_parameters = self.parameters

    def test_apply_parameter_mappings(self, parameter_manager, mappings):
        # Apply a single mapping
        parameter_manager.apply_mappings(mappings[0])
        # Apply a list of mappings
        parameter_manager.apply_mappings(mappings[1:])
        # Now set some parameter values, there will be a Warning issued for the unknown parameter
        with pytest.warns(UserWarning):
            parameter_manager.set_parameter_values(
                {
                    'p1' : 1000, # a global parameter
                    'p1_1st' : 100, # a local parameter
                    'p_unknown' : 10, # unknown parameter
                }
            )
        # One can define a local parameter for multiple replicate ids
        parameter_manager.apply_mappings(
            ParameterMapper(replicate_id=['1st', '2nd'], global_name='p1', local_name='p1_local', value=10000),
        )
        # Can also be applied to all replicates
        parameter_manager.apply_mappings(
            ParameterMapper(replicate_id='all', global_name='p1', local_name='p1_local'),
        )
        # Must use only known replicate ids
        with pytest.raises(ValueError):
            parameter_manager.apply_mappings(
                ParameterMapper(replicate_id=['1st', '2nd', 'invalid'], global_name='p1', local_name='p1_local'),
            )
        with pytest.raises(ValueError):
            parameter_manager.apply_mappings(
                ParameterMapper(replicate_id='invalid', global_name='p1', local_name='p1_local'),
            )
        # Must use only known global parameters
        with pytest.raises(ValueError):
            parameter_manager.apply_mappings(
                ParameterMapper(replicate_id='1st', global_name='p_unknown'),
            )

    def test_parameter_other_mapping_related_methods(self, parameter_manager):
        # The managed mappings can be shown as DataFrame
        parameter_manager.parameter_mapping
        # Can get the current parameter mappings as list of ParameterMappers
        parameter_manager.get_parameter_mappers()
        # Get the current parameter values for a specific replicate_id
        parameter_manager.get_parameters_for_replicate(replicate_id='1st')
        # There is a private method to check the parameter mappings before applying them
        # Check for being ParameterMapper objects
        with pytest.raises(TypeError):
            parameter_manager._check_mappings(mappings=['I am a string'])
        # Each unique local parameter name must have the same value for the mapping
        with pytest.raises(ValueError):
            parameter_manager._check_mappings(
                [
                    ParameterMapper(replicate_id='1st', global_name='p1', local_name='p1_local', value=100),
                    ParameterMapper(replicate_id='2nd', global_name='p1', local_name='p1_local', value=1000),
                ]
            )
            