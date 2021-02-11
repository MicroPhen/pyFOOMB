
import inspect
import numpy

from typing import Callable, List
import warnings

from .constants import Messages
from .modelling import BioprocessModel
from .modelling import ObservationFunction
from .simulation import Simulator


class ModelChecker():
    """
    A helper class providing methods to assist users in consitent model implementation.
    """

    def check_model_consistency(self, simulator:Simulator, report:bool=True) -> bool:
        """
        Runs several consistency checks for the implemented bioprocess model and observations functions for a Simulator instance.

        Arguments
        ---------
            simulator : Simulator

        Keyword arguments
        -----------------
            report : bool
                Reports if a model and/or observer is not fully specified.
                Default is True

        Returns
        -------
            bool
                The status of the currently implemented model consistency checking routines.

        Warns
        -----
            UserWarning
                Some consistency checks failed.
        """

        checks_ok = []

        # Check for correct unpacking of states in rhs method
        _check_ok = self._check_state_unpacking(simulator.bioprocess_model.rhs, simulator.bioprocess_model.states)
        if not _check_ok:
            warnings.warn('A possible inconsistency for state vector unpacking in method `rhs` was detected.')
        checks_ok.append(_check_ok)

        # Check for correct order of state derivatives
        _check_ok = self._check_rhs_derivatives_order(simulator.bioprocess_model.rhs, simulator.bioprocess_model.states)
        if not _check_ok:
            warnings.warn('A possible inconsistency for returning order of state derivatives in method `rhs` was detected.')
        checks_ok.append(_check_ok)

        # Check for correct parameter unpacking in rhs method
        _check_ok = self._check_parameter_unpacking(simulator.bioprocess_model.rhs, simulator.bioprocess_model.model_parameters)
        if not _check_ok:
            warnings.warn('A possible inconsistency for parameter unpacking in method `rhs` was detected.')
        checks_ok.append(_check_ok)

        if simulator.bioprocess_model.initial_switches is not None:

            # Check for sw arg in rhs signature
            _check_ok = self._check_sw_arg(simulator.bioprocess_model.rhs)
            checks_ok.append(_check_ok)

            # Check for correct unpacking of states in state_events method
            _check_ok = self._check_state_unpacking(simulator.bioprocess_model.state_events, simulator.bioprocess_model.states)
            if not _check_ok:
                warnings.warn('A possible inconsistency for state vector unpacking in method `state_events` was detected.')
            checks_ok.append(_check_ok)

            # Check for correct parameter unpacking in state_events method
            _check_ok = self._check_parameter_unpacking(simulator.bioprocess_model.state_events, simulator.bioprocess_model.model_parameters)
            if not _check_ok:
                warnings.warn('A possible inconsistency for parameter unpacking in method `state_events` was detected.')
            checks_ok.append(_check_ok)

            # Check for correct unpacking of states in change_states method
            _check_ok = self._check_state_unpacking(simulator.bioprocess_model.change_states, simulator.bioprocess_model.states)
            if not _check_ok:
                warnings.warn('A possible inconsistency for state vector unpacking in method `change_states` was detected.')
            checks_ok.append(_check_ok)

            # Check for correct parameter unpacking in change_states method
            _check_ok = self._check_parameter_unpacking(simulator.bioprocess_model.change_states, simulator.bioprocess_model.model_parameters)
            if not _check_ok:
                warnings.warn('A possible inconsistency for parameter unpacking in method `state_events` was detected.')
            checks_ok.append(_check_ok)

        # Call methods to see if there are any issues
        checks_ok.append(
            self._call_checks_bioprocess_model_methods(simulator, True)
        )

        if simulator.observer is not None:

            for _obs_fun in simulator.observer.observation_functions:
                _observation_function = simulator.observer.observation_functions[_obs_fun]

                checks_ok.append(
                    self._check_observe_method(_observation_function.observe, _observation_function.observation_parameters)
                )

                self._call_check_observe_method(_observation_function)

        return all(checks_ok)


    def _check_observe_method(self, method:Callable, observation_parameters:dict) -> bool:
        """
        Checks the observe method of an Observationfunction subclass for inconsistencies.
        
        Arguments
        ---------
            method : Callable
                The `observe` method of an ObservationFunction subclass.

            observation_parameters : dict
                The corresponding observation parameter values.

        Returns
        -------
            check_ok : bool
                The status of the currently implemented consistency check

        Warns
        -----
            UserWarning
                Parameters are unpacked in the wrong order.
                Variable names do not match the keys of the `observation_parameters`.
        """

        check_ok = True

        _lines = inspect.getsourcelines(method)
        all_in_one = ''.join(_lines[0]).replace('\n', '').replace(' ', '').replace('\\', '').replace(',]', ']').split('#')[0]

        # check for correct parameter unpacking when model_parameters are unpacked at once
        if 'self.observation_parameters.to_numpy()' in all_in_one:
            search_str = str(list(observation_parameters.keys())).replace(' ', '').replace("'", "").replace('[','').replace(']','')+'=self.observation_parameters.to_numpy()'
            if not search_str in all_in_one:
                correct_str = search_str.replace(',', ', ').replace('=', ' = ')
                warnings.warn(
                    f'Detected wrong order of parameter unpacking at once. Correct order is {correct_str}', 
                    UserWarning
                )
                check_ok = False

        for _line in _lines[0]:
            curr_line = _line.replace(' ', '').replace('\n','').split('#')[0]
            # check correct variable naming for explicit parameter unpacking
            if 'self.observation_parameters[' in curr_line:
                ok_unpack = False
                for p in list(observation_parameters.keys()):
                    valid_par_var1 = f"{p}=self.observation_parameters['{p}']"
                    valid_par_var2 = f'{p}=self.observation_parameters["{p}"]'
                    if valid_par_var1 in curr_line or valid_par_var2 in curr_line:
                        ok_unpack = True
                        break
                if not ok_unpack:
                    _line_msg = _line.replace('\n','')
                    warnings.warn(
                        f'Variable names from explicit parameter unpacking must match those of the corresponding keys.\nThis line is bad: {_line_msg}', 
                        UserWarning
                    )
                    check_ok = False

        return check_ok


    def _call_check_observe_method(self, observation_function:ObservationFunction):
        """
        Calls the `observe` method of an Observationfunction object to check for errors.

        Arguments
        ---------
            observation_function : ObservationFunction
                An ObservationFunction object of the current Simulator instance under investigation.
        """

        state_values = [
            -1.0,
            0.0, 
            1.0,
            numpy.array([0, 1,]),
            numpy.array([-1, 0, 1,])
        ]

        for _state_values in state_values:
            observation_function.observe(_state_values)


    def _check_state_unpacking(self, method:Callable, states:list) -> bool:
        """
        Checks the order of state unpacking in a method.

        Arguments
        ---------
            method : Callable
                The bioprocess model method to be checked.

            states : list
                the states of the bioprocess model intance of the current Simulator object under investigation.

        Returns
        -------
            check_ok : bool
                The status of the currently implemented consistency check

        Warns
        -----
            UserWarning
                States are unpacked in the wrong order.
        """

        check_ok = True

        _lines = inspect.getsourcelines(method)
        _code_text = ''.join(_lines[0]).replace('\n', '').replace(' ', '').replace('\\', '').replace(',]', ']').split('#')[0]
        _doc = inspect.getdoc(method)
        _doc_text = _doc.replace('\n', '').replace(' ', '').replace('\\', '').replace(',]', ']')
        code_text = _code_text.replace(_doc_text, '')

        # Check for correct unpacking of states
        if '=y' in code_text:
            states_str = str(states).replace("'", '').replace('[','').replace(']','').replace(' ','')+'=y'
            if states_str not in code_text:
                correct_states_str = states_str.replace(',', ', ').replace('=', ' = ')
                warnings.warn(
                    f'{Messages.unpacking_state_vector}. Correct order would be {correct_states_str}', 
                    UserWarning,
                )
                check_ok = False

        return check_ok


    def _check_parameter_unpacking(self, method:Callable, model_parameters:dict) -> bool:
        """
        Checks a methhod for consitent parameter unpacking.

        Arguments
        ---------
            method : Callable
                the method to be checked.

            model_parameters : dict
                The corresponding model parameter values.

        Returns
        -------
            check_ok : bool
                The status of the currently implemented consistency check.

        Warns
        -----
            UserWarning
                Parameters are unpacked in the wrong order.
                Variable names do not match the keys of the `model_parameters`.
        """

        check_ok = True

        _lines = inspect.getsourcelines(method)
        _code_text = ''.join(_lines[0]).replace('\n', '').replace(' ', '').replace('\\', '').replace(',]', ']').split('#')[0]
        _doc = inspect.getdoc(method)
        _doc_text = _doc.replace('\n', '').replace(' ', '').replace('\\', '').replace(',]', ']')
        code_text = _code_text.replace(_doc_text, '')

        # Check for correct parameter unpacking when model_parameters are unpacked at once
        if 'self.model_parameters.to_numpy()' in code_text:
            search_str = str(list(model_parameters.keys())).replace(' ', '').replace("'", "").replace('[','').replace(']','')+'=self.model_parameters.to_numpy()'
            if not search_str in code_text:
                correct_str = search_str.replace(',', ', ').replace('=', ' = ')
                warnings.warn(
                        f'Detected wrong order of parameter unpacking at once. Correct order would be {correct_str}', 
                        UserWarning,
                    )
                check_ok = False

        # Check correct parameter unpacking using the model_parameters dict keys
        # First get the lines of the doc string and remove any whitespaces
        _doc_lines = _doc.replace(' ', '').split('\n')
        for _line in _lines[0]:
            # Make sure that not the lines of the docstring
            if _line.replace(' ', '').replace('\n','') not in _doc_lines:
                curr_line = _line.replace(' ', '').replace('\n','').split('#')[0]
                # Check correct variable naming for explicit parameter unpacking
                if 'self.model_parameters[' in curr_line:
                    ok_unpack = False
                    for p in list(model_parameters.keys()):
                        valid_par_var1 = f"{p}=self.model_parameters['{p}']"
                        valid_par_var2 = f'{p}=self.model_parameters["{p}"]'
                        if valid_par_var1 in curr_line or valid_par_var2 in curr_line:
                            ok_unpack = True
                            break
                    if not ok_unpack:
                        _line_msg = _line.replace('\n','')
                        warnings.warn(
                            f'Variable names from explicit parameter unpacking should match those of the corresponding keys.\nThis line seems bad: {_line_msg}.\nValid model parameters are {list(model_parameters.keys())}',
                            UserWarning,
                            )
                        check_ok = False

        return check_ok


    def _check_sw_arg(self, method:Callable) -> bool:
        """
        Checks for use of `sw` argument in corresponding bioprocess model methods.

        Arguments
        ---------
            method : Callable
                The method in whose signature the `sw` argument shall be used.

        Returns
        -------
            check_ok : bool
                The status of the currently implemented consistency check.

        Warns
        -----
            UserWarning
                The `sw` argument is missing.
        """

        check_ok = True

        _lines = inspect.getsourcelines(method)
        code_text = ''.join(_lines[0]).replace('\n', '').replace(' ', '').replace('\\', '').replace(',]', ']')
        
        first_line = _lines[0][0].replace(' ', '').replace('\n','')
        if not 'sw' in first_line:
            warnings.warn(Messages.missing_sw_arg, UserWarning)
            check_ok = False

        return check_ok


    def _check_rhs_derivatives_order(self, rhs_method:Callable, states:list) -> bool:
        """
        Check that the order of the returned state derivatives is correct.

        Arguments
        ---------
            rhs_method : Callable
                The right-hand-side method implemented by the user.
            states : list
                The states of the bioprocess model.
        Returns
        -------
            check_ok : bool
                The status of the currently implemented consistency check.

        Warns
        -----
            UserWarning
                The returned derivatives are not in the correct order.
        """

        check_ok = True

        _lines = inspect.getsourcelines(rhs_method)
        code_text = ''.join(_lines[0]).replace('\n', '').replace(' ', '').replace('\\', '').replace(',]', ']')

        # Check rhs return for correct order of derivatives
        ders = [f'd{_state}dt' for _state in states]
        ders_str = str(ders).replace("'", '').replace('[','').replace(']','').replace(' ','')
        if not ders_str in code_text:
            correct_ders_str = ders_str.replace(',',', ')
            warnings.warn(
                f'{Messages.wrong_return_order_state_derivatives}. Correct return is numpy.array([{correct_ders_str}])', 
                UserWarning,
            )
            check_ok = False

        return check_ok


    def _call_checks_bioprocess_model_methods(self, simulator:Simulator, report:bool=False) -> bool:
        """
        Runs several call checks for the implemented bioprocess model and observations functions for a Simulator instance.

        Arguments
        ---------
            simulator : Simulator

        Keyword arguments
        -----------------
            report : bool
                Reports if a model and/or observer is not fully specified.
                Default is True

        Returns
        -------
            bool
                The status of the currently implemented model consistency checking routines.

        Warns
        -----
            UserWarning
                The `state_events` methods returns a different number of events in certain situations.
                The number of initial switches does not match the number of events returned by the state_events method.
        """

        check_ok = True
            
        # Call method `rhs`
        if simulator.bioprocess_model.initial_switches is not None:
            try:
                simulator.bioprocess_model.rhs(
                    t=0, 
                    y=simulator.bioprocess_model.initial_values.to_numpy(), 
                    sw=simulator.bioprocess_model.initial_switches,
                )
            except Exception as e:
                check_ok = False
                warnings.warn(f'Set `initial_switches` argument. Autodetection for number of events failed: {e}', UserWarning)
                return check_ok
            try:
                # Invert the switches
                simulator.bioprocess_model.rhs(
                    t=0, 
                    y=simulator.bioprocess_model.initial_values.to_numpy(), 
                    sw=numpy.invert(simulator.bioprocess_model.initial_switches),
                )
            except Exception as e:
                check_ok = False
                warnings.warn(f'Set `initial_switches` argument. Autodetection for number of events failed: {e}', UserWarning)
                return check_ok
        else:
            simulator.bioprocess_model.rhs(
                t=0, 
                y=simulator.bioprocess_model.initial_values.to_numpy(),
            )

        # Call method 'state_events'
        state_events_list_01 = simulator.bioprocess_model.state_events(
            t=0, 
            y=simulator.bioprocess_model.initial_values.to_numpy(), 
            sw=simulator.bioprocess_model.initial_switches,
        )
        if simulator.bioprocess_model.initial_switches is not None:
            state_events_list_02 = simulator.bioprocess_model.state_events(
                t=0, 
                y=simulator.bioprocess_model.initial_values.to_numpy(), 
                sw=numpy.invert(simulator.bioprocess_model.initial_switches),
            )

        # Call method `change_states`
        simulator.bioprocess_model.change_states(
            t=0, 
            y=simulator.bioprocess_model.initial_values.to_numpy(),
            sw=simulator.bioprocess_model.initial_switches,
        )
        if simulator.bioprocess_model.initial_switches is not None:
            simulator.bioprocess_model.change_states(
                t=0, 
                y=simulator.bioprocess_model.initial_values.to_numpy(),
                sw=numpy.invert(simulator.bioprocess_model.initial_switches),
            )

        # Check length of returned event list with length of initial switches
        if simulator.bioprocess_model.initial_switches is not None:
            if len(state_events_list_01) != len(state_events_list_02):
                warnings.warn(
                    'The number of returned events seems to vary with the states of the switches',
                    UserWarning,
                )
                check_ok = False
            elif len(state_events_list_01) != len(simulator.bioprocess_model.initial_switches):
                warnings.warn(
                    f'Number of initial switches does not match with number of events: {len(simulator.bioprocess_model.initial_switches)} vs. {len(state_events_list_01)}', 
                    UserWarning,
                )
                check_ok = False

        return check_ok