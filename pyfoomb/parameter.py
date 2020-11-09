import collections
import copy
from dataclasses import dataclass
import numpy
import pandas
from typing import List
import warnings

from .constants import Messages
from .utils import Helpers
from .utils import OwnDict


@dataclass
class ParameterMapper():
    """
    Maps global parameter names to local ones, specific for a certain replicate
    """

    replicate_id : str
    global_name : str
    local_name : str = None
    value : float = None
        
    def __post_init__(self):
        if self.local_name is None:
            if self.replicate_id == 'all' or isinstance(self.replicate_id, list):
                raise ValueError('Argument `local_name` cannot be None when `replicate_id` is "all" or a list of replicate_ids')
            else:
                self.local_name = f'{self.global_name}_{self.replicate_id}'


class Parameter():

    def __init__(self, global_name:str, replicate_id:str, local_name:str=None, value:float=None):
        """
        Arguments
        ---------
            global_name : str
                The global name of the parameter
            replicate_id : str
                The replicate_id for which the local_name may apply

        Keyword arguments
        -----------------
            local_name : str
                the local name for the parameter, specifically for the corresponding replicate_id.
                Default is None, which sets the global name as local name
            value : float
                Default is None
        """

        self.global_name = global_name 
        self.replicate_id = replicate_id
        if local_name is not None:
            self.local_name = local_name
        else:
            self.local_name = global_name
        self.value = value


class ParameterManager():
    """
    Manages a list of Parameter objects and their mappings between 
    a bioprocess model (including observation functions) and replicates of the model instances.
    """
    
    def __init__(self, replicate_ids:list, parameters:dict):
        """
        Arguments
        ---------
            replicate_ids : list
                A list of case-sensitive unique replicate ids.

            parameters : dict
                Parameters that are set. Keys are the global parameter names, which are used as local names. 
                Values are set correspondingly.
        """
        
        self._is_init = True
        self.global_parameters = parameters
        self.replicate_ids = replicate_ids
        self._parameters = [
            Parameter(
                global_name=p, 
                replicate_id=_id,
                value=self.global_parameters[p],
            )
            for p in self.global_parameters.keys()
            for _id in self.replicate_ids
        ]
        self._is_init = False


    #%% Properties

    @property
    def replicate_ids(self) -> list:
        return self._replicate_ids


    @replicate_ids.setter
    def replicate_ids(self, value:list):
        if self._is_init:
            if not Helpers.has_unique_ids(value):
                raise ValueError(Messages.non_unique_ids)
            self._replicate_ids = value
        else:
            raise AttributeError('Replicate ids can only be set during initialization of the ParameterManager')


    @property
    def global_parameters(self) -> list:
        return self._global_parameters


    @global_parameters.setter
    def global_parameters(self, value):
        if self._is_init:
            if not Helpers.has_unique_ids(value):
                raise ValueError(Messages.non_unique_ids)
            if isinstance(value, list):
                _value = {p : numpy.nan for p in sorted(value, key=str.lower)}
            elif isinstance(value, dict):
                _value = {p : value[p] for p in sorted(value.keys(), key=str.lower)}
            self._global_parameters = _value
        else:
            raise AttributeError('Global parameters can only be set during initialization of the ParameterManager')


    @property
    def parameter_mapping(self) -> pandas.DataFrame:
        _values = [p.value for p in self._parameters]
        _global_names = [p.global_name for p in self._parameters]
        _local_names = [p.local_name for p in self._parameters]
        _replicate_ids = [p.replicate_id for p in self._parameters]
        df = pandas.DataFrame([_global_names, _local_names, _replicate_ids, _values]).T
        df.columns = ['global_name', 'local_name', 'replicate_id', 'value']
        return df.set_index(['global_name', 'replicate_id'])


    #%% Public methods

    def set_parameter_values(self, parameters:dict):
        """
        Assigns values to some parameters.
        Valid keys are the global names or local names or model parameters, initial values, 
        or observation parameters, according to the current parameter mapping.

        Arguments
        ---------
            parameters : dict
                The parameter names and corresponding values to be set.

        Warns
        -----
            UserWarning
                Values for unknown parameters are set.
        """

        known_parameters = set([_parameter.local_name for _parameter in self._parameters])
        new_parameters = set(parameters.keys())
        unknown_parameters = new_parameters.difference(known_parameters)
        if len(unknown_parameters) > 0:
            warnings.warn(f'Detected unknown parameters, which are ignored: {unknown_parameters}', UserWarning)

        for p in parameters.keys():
            for _parameter in self._parameters:
                if _parameter.local_name == p:
                    _parameter.value = parameters[p]


    def apply_mappings(self, mappings:List[ParameterMapper]):
        """
        An item of the mappings list must be a ParameterMapper instance according to ParameterMapper(replicate_id=..., global_name=..., local_name=..., value=...).

        NOTE:
            replicate_id can also be a list, which applies the mapping to all replicate in this list.
            replicate_id can also be 'all', which applies the mapping to all replicates. 

        Arguments
        ---------
            mappings : list
                A list of mappings, which can be a tupe or ParameterMapper objects, or a mix of them.

        Raises
        ------
            TypeError
                Any mapping is not a ParameterMapper object.
            ValueError
                A mapping has an invalid replicate id.
            ValueError
                A mapping has a invalid global parameter name.
        """

        if isinstance(mappings, ParameterMapper):
            mappings = [mappings]

        self._check_mappings(mappings)
        # save parameters in case the mapping is not valid
        _backup_parameters = copy.deepcopy(self._parameters)
        for mapping in mappings:

            _replicate_id = mapping.replicate_id
            _global_name = mapping.global_name
            _local_name = mapping.local_name
            _value = mapping.value

            if isinstance(_replicate_id, list):
                for _id in list(_replicate_id):
                    if _id not in self.replicate_ids:
                        raise ValueError(f'Invalid replicate id: {_id}')
            elif _replicate_id not in self.replicate_ids and _replicate_id != 'all':
                raise ValueError(f'Invalid replicate id: {_replicate_id}')

            if _global_name not in self.global_parameters.keys():
                raise ValueError(f'Invalid global parameter name: {_global_name}')

            self._apply_single_mapping(_replicate_id, _global_name, _local_name, _value)

        self._check_joint_uniqueness_local_names_and_values(_backup_parameters)


    def get_parameter_mappers(self) -> List[ParameterMapper]:
        """
        Returns a list of ParameterMapper objects representing the current parameter mapping.0
        """
        
        mappings = []
        for p in self._parameters:
            mapping = ParameterMapper(replicate_id=p.replicate_id, global_name=p.global_name, local_name=p.local_name, value=p.value)
            mappings.append(mapping)
        return mappings


    def get_parameters_for_replicate(self, replicate_id:str) -> OwnDict:
        """
        Extracts the parameters for a specific replicate.

        Arguments
        ---------
            replicate_id : str
                The specific (unique) id of a replicate.

        Returns
        -------
            OwnDict with keys as global parameter names and corresponding values.
        """

        parameters_dict = {}
        for _parameter in self._parameters:
            if _parameter.replicate_id == replicate_id:
                parameters_dict[_parameter.global_name] = _parameter.value
        return OwnDict(parameters_dict)


    #%% Private methods

    def _apply_single_mapping(self, replicate_id:list, global_name:str, local_name:str, value:float=None):
        """
        Helper method that applies a single mapping.

        Arguments
        ---------
            replicate_id : list, or str, or 'all'
                Identifies the replicates for which the mapping is applied. 
                Can be a single id, a list of those, or 'all'.
            global_name : str
                Identifies the global parameter that is mapped.
            local_name : str
                The local, replicate-specific name of the global parameter.
            
        Keyword arguments
        -----------------
            value : float
                The parameters value for the mapping.
                Default is None, which uses the value of the corresponding global parameter.
        """

        if replicate_id == 'all':
            replicate_id = self.replicate_ids

        # make a single item list in case only one replicate_id is addressed
        if isinstance(replicate_id, str):
            replicate_id = [replicate_id]

        for _parameter in self._parameters:
            for _replicate_id in replicate_id:
                if _parameter.replicate_id == _replicate_id and _parameter.global_name == global_name:
                    _parameter.local_name = local_name
                    if value is not None:
                        _parameter.value = value
    

    def _check_mappings(self, mappings:List[ParameterMapper]):
        """
        Checks that the mappings have unique pairs for local_names and value,
        as well as each mappings item is a Parameter Mapping objects.

        Arguments
        ---------
            mappings : List[ParameterMapper]
                The list of parameter mappings to be applied.

        Raises
        ------
            TypeError
                An item of mappings is not a Parameter object.
            ValueError
                Same parameters in mappings have not unique values.
        """

        local_names = []
        values = []
        for mapping in mappings:

            if not isinstance(mapping, ParameterMapper):
                raise TypeError(f'Items of mappings must be of type ParameterMapper. Invalid mapping item: {mapping}')
            local_names.append(mapping.local_name)
            values.append(mapping.value)

        name_value_mapping = {p : None for p in sorted(set(local_names), key=str.lower)}
        for _name, _value in zip(local_names, values):
            if name_value_mapping[_name] is None:
                name_value_mapping[_name] = _value
            elif name_value_mapping[_name] != _value:
                raise ValueError(
                    f'Parameters of mappings to be applied must have unique values. Parameter with at least two different values detected. "{_name}" with {name_value_mapping[_name]} and {_value}.'
                )


    def _check_joint_uniqueness_local_names_and_values(self, backup_parameters:List[Parameter]):
        """
        Checks that the application of a valid mapping does not result in non-unique pairs of local_name and value

        Arguments
        ---------
            backup_parameters : List[Parameter]
                The backup parameters will be applied in case a ValueError is raised

        Raises
        ------
            ValueError
                A parameter among different replicates has different values.
        """

        local_names = [_parameter.local_name for _parameter in self._parameters]
        values = [ _parameter.value for _parameter in self._parameters]
        name_value_mapping = {p : None for p in sorted(set(local_names), key=str.lower)}
        for _name, _value in zip(local_names, values):
            if name_value_mapping[_name] is None:
                name_value_mapping[_name] = _value
            elif name_value_mapping[_name] != _value:
                self._parameters = backup_parameters
                raise ValueError(
                    f'Parameters must have unique values. Parameter with at least two different values detected. "{_name}" with {name_value_mapping[_name]} and {_value}.'
                )
