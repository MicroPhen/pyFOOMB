import numpy

class Constants():
    eps_float64 = numpy.finfo(numpy.float64).eps
    single_id = None
    observed_state_key = 'observed_state'
    pretty_metrics = {
        'negLL' : 'negative-log-likelihood',
        'SS' : 'sum-of-squares',
        'WSS' : 'weighted-sum-of-squares',
    }
    handled_CVodeErrors = [-1, -4]


class Messages():
    bad_unknowns = 'Bad type of unknowns, must be either of type list or dict'
    cvode_boundzero = 'Detected CVodeError, probably due to some bounds being 0'
    invalid_measurements = 'Detected invalid measurement keys'
    invalid_unknowns = 'Detected invalid unknowns to be estimated'
    invalid_initial_values_type = 'Initial values must be provided as dictionary'
    missing_bounds = 'Must provide bounds for global parameter optimization'
    missing_values = 'Missing values'
    missing_sw_arg = 'Detected event handling in model. Provide "sw" argument in rhs signature: rhs(self, t, y, sw).'
    non_unique_ids = 'Detected non-unique (case-insensitive) keys/items/ids'
    unpacking_state_vector = 'Unpacking the state vector y is required in alphabetical case-insensitve order'
    wrong_return_order_state_derivatives = 'State derivatives are returned in the wrong order or do not match the pattern "d(state)dt"'
