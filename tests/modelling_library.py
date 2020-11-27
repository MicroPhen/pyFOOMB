
import numpy as np

from pyfoomb import BioprocessModel
from pyfoomb import ObservationFunction


#%% Models that work

class Model01(BioprocessModel):
    def rhs(self, t, y, sw=None):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return np.array([dy0dt, dy1dt])


class Model02(BioprocessModel):
    def rhs(self, t, y, sw=None):
        k = self.model_parameters['k']
        dydt = -k * y
        return dydt


class Model03(BioprocessModel):
    def rhs(self, t, y, sw):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return np.array([dy0dt, dy1dt])

    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y1 = y1
        return np.array([event_y1])


class Model04(BioprocessModel):
    def rhs(self, t, y, sw=None):
        y0, y1 = y
        rate0 = self.model_parameters['rate0']
        rate1 = self.model_parameters['rate1']
        dy0dt = rate0
        dy1dt = rate1
        return np.array([dy0dt, dy1dt])


class Model05(BioprocessModel):
    def rhs(self, t, y=None):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return np.array([dyAdt, dyBdt, dyCdt, dyDdt])   


class Model06(BioprocessModel):
    # This model cannot have rate0 and rate1 to be zero
    def rhs(self, t, y, sw):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        if sw[0]:
            dy0dt = 1/rate1
        else:
            dy0dt = rate0

        if sw[1]:
            dy1dt = 1/rate0
        else:
            dy1dt = rate1
        return np.array([dy0dt, dy1dt])

    def state_events(self, t, y, sw):
        y0, y1 = y
        event_t1 = t - 5
        event_t2 = t - 2
        event_t3 = t - 1
        return np.array([event_t1, event_t2, event_t3])   

    def change_states(self, t, y, sw):
        y0, y1 = y
        if sw[2]:
            y0 = self.initial_values['y00']
            y1 = self.initial_values['y10']
        return [y0, y1]


class ModelLibrary():

    modelnames = [
        'model01',
        'model02',
        'model03',
        'model04',
        'model05',
        'model06',
    ]

    modelclasses = {
        'model01' : Model01,
        'model02' : Model02,
        'model03' : Model03,
        'model04' : Model04,
        'model05' : Model05,
        'model06' : Model06,
    }

    states = {
        'model01' : ['y0' , 'y1'],
        'model02' : ['y'],
        'model03' : ['y0' , 'y1'],
        'model04' : ['y0' , 'y1'],
        'model05' : ['yA', 'yB', 'yC', 'yD'],
        'model06' : ['y0' , 'y1'],
    }

    model_parameters = {
        'model01' : {'rate0' : 0.0, 'rate1' : 1.0},
        'model02' : {'k' : 0.02},
        'model03' : {'rate0' : 2.0, 'rate1' : 3.0},
        'model04' : {'rate0' : 4.0, 'rate1' : 5.0},
        'model05' : {'rateA' : 10.0, 'rateB' : 11.0, 'rateC' : 12.0, 'rateD' : 13.0, 'rateE' : 14.0},
        'model06' : {'rate0' : -2.0, 'rate1' : -3.0},
    }

    initial_values = {
        'model01' : {'y00' : 0.0, 'y10' : 1.0},
        'model02' : {'y0' : 100.0},
        'model03' : {'y00' : 2.0, 'y10' : 3.0},
        'model04' : {'y00' : 4.0, 'y10' : 5.0},
        'model05' : {'yA0' : 100.0, 'yB0' : 200.0, 'yC0' : 300.0, 'yD0': 400.0},
        'model06' : {'y00' : 20.0, 'y10' : 30.0},
    }

    initial_switches = {
        'model01' : None,
        'model02' : None,
        'model03' : [False],
        'model04' : None,
        'model05' : None,
        'model06' : [False, False, False],
    }


class ObservationFunction01(ObservationFunction):

    def observe(self, state_values):
        slope_01 = self.observation_parameters['slope_01']
        offset_01 = self.observation_parameters['offset_01']
        return state_values * slope_01 + offset_01


class ObservationFunctionLibrary():

    names = [
        'obsfun01'
    ]

    observation_functions = {
        'obsfun01' : ObservationFunction01,
    }

    observation_function_parameters = {
        'obsfun01' : {'slope_01' : 2, 'offset_01' : 10},
    }

    observed_states = {
        'obsfun01' : 'y0',
    }

