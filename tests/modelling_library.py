
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
    def rhs(self, t, y):
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


class Model06_V02(Model06):
    # The auto-detection of events will work on the Simulator level, so the user need to provided the initial switches
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_t1 = t - 5
        event_t2 = t - 2
        event_t3 = t - 1
        events = [event_t1, event_t2, event_t3]
        return events

class Model06_V03(Model06):
    # The auto-detection of events will fail as the sw arg can't be None, so the user need to provided the initial switches
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_t1 = t - 5
        event_t2 = t - 2
        event_t3 = t - 1
        sw_1 = sw[0]
        sw_2 = sw[1]
        sw_3 = sw[2]
        events = [event_t1, event_t2, event_t3]
        return events


class Model07(Model02):

    def state_events(self, t, y, sw):
        event_t = t - 10
        return [event_t]

    def change_states(self, t, y, sw):
        if sw[0]:
            y = y + 10
        return y


# Variants of Model03
class Model03_V02(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y1 = y1
        return np.array([event_y1,])


class Model03_V03(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        return np.array([event_y0, event_y1])     
    

class Model03_V04(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        return np.array([event_y0, event_y1,])       


class Model03_V05(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([event_y0, event_y1, event_t])     
    

class Model03_V06(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([event_y0, event_y1, event_t,])       


# Autodetection of number of events from return of method `state_events`
class Model03_V07(BioprocessModel):
    def rhs(self, t, y, sw):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        if sw[0]:
            dy0dt = rate1
        else:
            dy0dt = rate0

        if sw[1]:
            dy1dt = rate0
        else:
            dy1dt = rate1
        return np.array([dy0dt, dy1dt])

    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([event_y0, event_y1, event_t,])   

    def change_states(self, t, y, sw):
        y0, y1 = y
        if sw[2]:
            y0 = self.initial_values['y00']
            y1 = self.initial_values['y10']
        return [y0, y1]


# Autodetection of number of events from return of method `state_events`
class Model03_V08(Model03_V07):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0 
        event_y1 = y1
        event_t = t - 5
        return np.array([
            event_y0, 
            event_y1, 
            event_t,
        ])   


# Autodetection of number of events from return of method `state_events`
class Model03_V09(Model03_V07):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([
            event_y0, 
            event_y1, 
            event_t
        ])   


# Autodetection of number of events from return of method `state_events`
class Model03_V10(Model03_V07):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([
            event_y0, event_y1, event_t
        ])   


# Autodetection of number of events from return of method `state_events`
class Model03_V11(Model03_V07):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([
            event_y0, event_y1, event_t,
        ])   


# Autodetection of number of events from return of method `state_events`
class Model03_V12(Model03):
    def state_events(self, t, y, sw):
        event_t = t - 5
        return np.array([event_t]) 


# Autodetection of number of events from return of method `state_events`
class Model03_V13(Model03):
    def state_events(self, t, y, sw):
        event_t = t - 5
        return np.array([event_t,])


# Bad variants of Model03

class Model03_BadV01(BioprocessModel):
    # state vector is unpacked in the wrong order
    def rhs(self, t, y):
        y1, y0 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return np.array([dy0dt, dy1dt])


class Model03_BadV02(BioprocessModel):
    # derivatives of state vector are return in the wrong order
    def rhs(self, t, y):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return np.array([dy1dt, dy0dt])


class Model03_BadV03(BioprocessModel):
    # state vector is unpacked in the wrong order
    # derivatives of state vector are return in the wrong order
    def rhs(self, t, y):
        y1, y0 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return np.array([dy1dt, dy0dt])


class Model03_BadV04(BioprocessModel):
    # name of parameter variable does not match the corresponding key
    def rhs(self, t, y):
        y0, y1 = y
        rate0 = self.model_parameters['rate0']
        any_parameter = self.model_parameters['rate1']
        dy0dt = rate0
        dy1dt = any_parameter
        return np.array([dy0dt, dy1dt])


class Model03_BadV05(BioprocessModel):
    # parameters are unpacked in wrong order
    def rhs(self, t, y):
        y0, y1 = y
        rate1, rate0 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return np.array([dy0dt, dy1dt])


class Model03_BadV06(Model03_V06):
    # state vector is unpacked in the wrong order
    def state_events(self, t, y, sw):
        y1, y0 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([event_y0, event_y1, event_t,])       


class Model03_BadV07(Model03_V06):
    # parameters are unpacked in wrong order
    def state_events(self, t, y, sw):
        y0, y1 = y
        rate1, rate0 = self.model_parameters.to_numpy()
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([event_y0, event_y1, event_t,])       


class Model03_BadV08(Model03_V06):
    # name of parameter variable does not match the corresponding key
    def state_events(self, t, y, sw):
        y0, y1 = y
        rate0 = self.model_parameters['rate0']
        any_parameter = self.model_parameters['rate1']
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([event_y0, event_y1, event_t,])   


class Model03_BadV09(Model03_V06):
    # name of parameter variable does not match the corresponding key
    def state_events(self, t, y, sw):
        y0, y1 = y
        any_parameter = self.model_parameters['rate1']
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return np.array([event_y0, event_y1, event_t,])


class Model06_Bad01(Model06):
    # Has an undefined variable
    def rhs(self, t, y, sw):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        if sw[0]:
            dy0dt = rate99
        else:
            dy0dt = rate0

        if sw[1]:
            dy1dt = rate0
        else:
            dy1dt = rate1
        return np.array([dy0dt, dy1dt])


class Model06_Bad02(Model06):
    # Has an undefined variable
    def rhs(self, t, y, sw):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        if sw[0]:
            dy0dt = rate1
        else:
            dy0dt = rate99

        if sw[1]:
            dy1dt = rate0
        else:
            dy1dt = rate1
        return np.array([dy0dt, dy1dt])


class Model06_Bad03(Model06):

    # Has an undefined variable
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_t1 = t - 5
        event_t3 = t - 1
        return np.array([event_t1, event_t2, event_t3])  


class Model06_Bad04(Model06):
    # Has an undefined variable
    def change_states(self, t, y, sw):
        y0, y1 = y
        y00 = self.initial_values['y00']
        y10 = self.initial_values['y10']

        if sw[1]:
            y0 = y000000000000000
            y1 = y10

        return [y0, y1]


class Model06_Bad05(Model06):
    # the number of events depends on the switches
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_t1 = t - 5
        event_t2 = t - 2
        event_t3 = t - 1
        if sw[1]:
            events = [event_t1, event_t3]
        else:
            events = [event_t1, event_t2, event_t3]
        return events 


class Model06_Bad06(Model06):
    # Inconsitent parameter unpacking
    def change_states(self, t, y, sw):
        y0, y1 = y
        rate00000 = self.model_parameters['rate0']
        return [y0, y1]


class Model06_Bad07(Model06):
    # Inconsitent parameter unpacking
    def change_states(self, t, y, sw):
        y0, y1 = y
        rate1, rate0 = self.model_parameters.to_numpy()
        return [y0, y1]


class Model06_Bad08(Model06):
    # Inconsitent state unpacking
    def change_states(self, t, y, sw):
        y1, y0 = y
        rate0 = self.model_parameters['rate0']
        return [y0, y1]



class ModelLibrary():

    modelnames = [
        'model01',
        'model02',
        'model03',
        'model04',
        'model05',
        'model06',
        'model07',
    ]

    modelclasses = {
        'model01' : Model01,
        'model02' : Model02,
        'model03' : Model03,
        'model04' : Model04,
        'model05' : Model05,
        'model06' : Model06,
        'model07' : Model07
    }

    states = {
        'model01' : ['y0' , 'y1'],
        'model02' : ['y'],
        'model03' : ['y0' , 'y1'],
        'model04' : ['y0' , 'y1'],
        'model05' : ['yA', 'yB', 'yC', 'yD'],
        'model06' : ['y0' , 'y1'],
        'model07' : ['y'],
    }

    model_parameters = {
        'model01' : {'rate0' : 0.0, 'rate1' : 1.0},
        'model02' : {'k' : 0.02},
        'model03' : {'rate0' : 2.0, 'rate1' : 3.0},
        'model04' : {'rate0' : 4.0, 'rate1' : 5.0},
        'model05' : {'rateA' : 10.0, 'rateB' : 11.0, 'rateC' : 12.0, 'rateD' : 13.0, 'rateE' : 14.0},
        'model06' : {'rate0' : -2.0, 'rate1' : -3.0},
        'model07' : {'k' : 0.02},
    }

    initial_values = {
        'model01' : {'y00' : 0.0, 'y10' : 1.0},
        'model02' : {'y0' : 100.0},
        'model03' : {'y00' : 2.0, 'y10' : 3.0},
        'model04' : {'y00' : 4.0, 'y10' : 5.0},
        'model05' : {'yA0' : 100.0, 'yB0' : 200.0, 'yC0' : 300.0, 'yD0': 400.0},
        'model06' : {'y00' : 20.0, 'y10' : 30.0},
        'model07' : {'y0' : 100.0},
    }

    initial_switches = {
        'model01' : None,
        'model02' : None,
        'model03' : [False],
        'model04' : None,
        'model05' : None,
        'model06' : [False, False, False],
        'model07' : None,
    }

    bad_variants_model03 = [
        Model03_BadV01,
        Model03_BadV01,
        Model03_BadV02,
        Model03_BadV03,
        Model03_BadV04,
        Model03_BadV05,
        Model03_BadV06,
        Model03_BadV07,
        Model03_BadV08,
        Model03_BadV09,
    ]

    bad_variants_model06 = [
        Model06_Bad01,
        Model06_Bad02,
        Model06_Bad03,
        Model06_Bad04,
        Model06_Bad05,
        Model06_Bad06,
        Model06_Bad07,
        Model06_Bad08,
    ]

    variants_model03 = [
        Model03_V02,
        Model03_V03,
        Model03_V04,
        Model03_V05,
        Model03_V06,
        Model03_V07,
        Model03_V08,
        Model03_V09,
        Model03_V10,
        Model03_V11,
        Model03_V12,
        Model03_V13
    ]


class ObservationFunction01(ObservationFunction):

    def observe(self, state_values):
        slope_01 = self.observation_parameters['slope_01']
        offset_01 = self.observation_parameters['offset_01']
        return state_values * slope_01 + offset_01


class ObservationFunction02(ObservationFunction):
    def observe(self, state_values):
        p1, p2, p3, p4, p5 = self.observation_parameters.to_numpy()
        return state_values + p1 + p2 + p3 + p4 + p5


class ObservationFunction02_V02(ObservationFunction):
    def observe(self, state_values):
        p1, \
            p2, p3, \
            p4, p5 \
            = self.observation_parameters.to_numpy()
        return state_values + p1 + p2 + p3 + p4 + p5


class ObservationFunction02_V03(ObservationFunction):
    def observe(self, state_values):
        p1, \
            p2, p3, \
            p4, p5 = self.observation_parameters.to_numpy()
        return state_values + p1 + p2 + p3 + p4 + p5


class ObservationFunction02_V04(ObservationFunction):
    def observe(self, state_values):
        p1 = self.observation_parameters['p1']
        p2 = self.observation_parameters['p2']
        p3 = self.observation_parameters['p3']
        p4 = self.observation_parameters['p4']
        p5 = self.observation_parameters['p5']
        return state_values + p1 + p2 + p3 + p4 + p5


class ObservationFunction01_Bad01(ObservationFunction):
    # observation parameters are unpacked in the wrong order
    def observe(self, model_values):
        slope_01, offset_01 = self.observation_parameters.to_numpy()
        return model_values * slope_01 + offset_01


class ObservationFunction01_Bad02(ObservationFunction):
    # observation parameter variable name does not match corresponding keys
    def observe(self, model_values):
        slope_01 = self.observation_parameters['slope_01']
        some_offset = self.observation_parameters['offset_01']
        return model_values * slope_01 + some_offset


class ObservationFunctionLibrary():

    names = [
        'obsfun01',
        'obsfun02',
    ]

    observation_functions = {
        'obsfun01' : ObservationFunction01,
        'obsfun02' : ObservationFunction02,
    }

    observation_function_parameters = {
        'obsfun01' : {'slope_01' : 2, 'offset_01' : 10},
        'obsfun02' : {'p1' : 1.0, 'p2' : 2.0, 'p3' : 3.0, 'p4' : 4.0, 'p5' : 5.0},
    }

    observed_states = {
        'obsfun01' : 'y0',
        'obsfun02' : 'y1',
    }

    variants_obsfun01 = [
        ObservationFunction01,
    ]

    variants_obsfun02 = [
        ObservationFunction02,
        ObservationFunction02_V02,
        ObservationFunction02_V03,
        ObservationFunction02_V04
    ]


    bad_variants_obsfun01 = [
        ObservationFunction01_Bad01,
        ObservationFunction01_Bad02,
    ]
