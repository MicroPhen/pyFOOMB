import copy
from IPython.display import display
from matplotlib import pyplot
import numpy
import pandas
import pathlib
import time
import re
import scipy
import sys
from typing import Dict, List, Tuple
import unittest

from assimulo.solvers.sundials import CVodeError

import pyfoomb
from pyfoomb.caretaker import Caretaker
from pyfoomb import utils

from pyfoomb.constants import Constants

from pyfoomb.datatypes import TimeSeries
from pyfoomb.datatypes import Measurement
from pyfoomb.datatypes import ModelState
from pyfoomb.datatypes import Observation
from pyfoomb.datatypes import Sensitivity

from pyfoomb.generalized_islands import LossCalculator
from pyfoomb.generalized_islands import PygmoOptimizers
from pyfoomb.generalized_islands import ArchipelagoHelpers

from pyfoomb.modelling import BioprocessModel
from pyfoomb.modelling import ObservationFunction

from pyfoomb.model_checking import ModelChecker

from pyfoomb.oed import CovOptimality

from pyfoomb.parameter import ParameterManager
from pyfoomb.parameter import ParameterMapper
from pyfoomb.parameter import Parameter

from pyfoomb.simulation import ModelObserver
from pyfoomb.simulation import Simulator
from pyfoomb.simulation import ExtendedSimulator

from pyfoomb.visualization import Visualization
from pyfoomb.visualization import VisualizationHelpers

OBSERVED_STATE_KEY = Constants.observed_state_key
SINGLE_ID = Constants.single_id

OBS_SLOPE_01 = 2
OBS_OFFSET_01 = 10
VERY_HIGH_NUMBER = numpy.inf


#%% Mock models

class Model01(BioprocessModel):
    def rhs(self, t, y):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return numpy.array([dy0dt, dy1dt])


class Model02(BioprocessModel):
    def rhs(self, t, y):
        k = self.model_parameters['k']
        dydt = -k * y
        return dydt


class Model03(BioprocessModel):
    def rhs(self, t, y, sw):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return numpy.array([dy0dt, dy1dt])

    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y1 = y1
        return numpy.array([event_y1])


class Model03_V02(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y1 = y1
        return numpy.array([event_y1,])


class Model03_V03(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        return numpy.array([event_y0, event_y1])     
    

class Model03_V04(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        return numpy.array([event_y0, event_y1,])       


class Model03_V05(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return numpy.array([event_y0, event_y1, event_t])     
    

class Model03_V06(Model03):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return numpy.array([event_y0, event_y1, event_t,])       


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
        return numpy.array([dy0dt, dy1dt])

    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return numpy.array([event_y0, event_y1, event_t,])   

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
        return numpy.array([
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
        return numpy.array([
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
        return numpy.array([
            event_y0, event_y1, event_t
        ])   


# Autodetection of number of events from return of method `state_events`
class Model03_V11(Model03_V07):
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return numpy.array([
            event_y0, event_y1, event_t,
        ])   


# Autodetection of number of events from return of method `state_events`
class Model03_V12(Model03):
    def state_events(self, t, y, sw):
        event_t = t - 5
        return numpy.array([event_t]) 


# Autodetection of number of events from return of method `state_events`
class Model03_V13(Model03):
    def state_events(self, t, y, sw):
        event_t = t - 5
        return numpy.array([event_t,]) 


class Model04(BioprocessModel):
    def rhs(self, t, y):
        y0, y1 = y
        rate0 = self.model_parameters['rate0']
        rate1 = self.model_parameters['rate1']
        dy0dt = rate0
        dy1dt = rate1
        return numpy.array([dy0dt, dy1dt])


class Model05(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([dyAdt, dyBdt, dyCdt, dyDdt])


class Model05_V01(BioprocessModel):
    def rhs(self, t, y):
        yA, \
            yB, \
            yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([dyAdt, dyBdt, dyCdt, dyDdt])


class Model05_V02(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, \
            rateC, \
            rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([dyAdt, dyBdt, dyCdt, dyDdt])


class Model05_V03(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, \
            rateC, \
            rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([dyAdt, dyBdt, dyCdt, dyDdt,])


class Model05_V04(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([dyAdt, 
                            dyBdt, 
                            dyCdt, 
                            dyDdt])


class Model05_V05(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([dyAdt, 
                            dyBdt, 
                            dyCdt, 
                            dyDdt,])


class Model05_V06(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([
            dyAdt, 
            dyBdt, 
            dyCdt, 
            dyDdt
        ])


class Model05_V07(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([
            dyAdt, 
            dyBdt, 
            dyCdt, 
            dyDdt
        ])


class Model05_V08(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([
            dyAdt, 
            dyBdt, 
            dyCdt, 
            dyDdt,
        ])


class Model05_V09(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([
            dyAdt, dyBdt, 
            dyCdt, dyDdt
        ])


class Model05_V10(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return numpy.array([
            dyAdt, dyBdt, 
            dyCdt, dyDdt,
        ])


class Model05_V11(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return [dyAdt, dyBdt, dyCdt, dyDdt]


class Model05_V12(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return [dyAdt, dyBdt, dyCdt, dyDdt]


class Model05_V13(BioprocessModel):
    def rhs(self, t, y):
        yA, yB, yC, yD = y
        rateA, rateB, rateC, rateD, rateE = self.model_parameters.to_numpy()
        dyAdt = rateA
        dyBdt = rateB
        dyCdt = rateC
        dyDdt = rateD - rateE
        return dyAdt, dyBdt, dyCdt, dyDdt


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
        return numpy.array([dy0dt, dy1dt])

    def state_events(self, t, y, sw):
        y0, y1 = y
        event_t1 = t - 5
        event_t2 = t - 2
        event_t3 = t - 1
        return numpy.array([event_t1, event_t2, event_t3])   

    def change_states(self, t, y, sw):
        y0, y1 = y
        if sw[2]:
            y0 = self.initial_values['y00']
            y1 = self.initial_values['y10']
        return [y0, y1]


class Model06_V02(Model06):
    # The auto-detection of event will fail, so the user need to provided the initial switches
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_t1 = t - 5
        event_t2 = t - 2
        event_t3 = t - 1
        events = [event_t1, event_t2, event_t3]
        return events


# Bad variants of Model03
class Model03_BadV01(BioprocessModel):
    # sw argument in rhs is missing
    def rhs(self, t, y, sw=None):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return numpy.array([dy0dt, dy1dt])

    def state_events(self, t, y, sw):
        y0, y1 = y
        event_y1 = y1
        return numpy.array([event_y1])


class Model03_BadV02(BioprocessModel):
    # state vector is unpacked in the wrong order
    def rhs(self, t, y):
        y1, y0 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return numpy.array([dy0dt, dy1dt])


class Model03_BadV03(BioprocessModel):
    # derivatives of state vector are return in the wrong order
    def rhs(self, t, y):
        y0, y1 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return numpy.array([dy1dt, dy0dt])


class Model03_BadV04(BioprocessModel):
    # state vector is unpacked in the wrong order
    # derivatives of state vector are return in the wrong order
    def rhs(self, t, y):
        y1, y0 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return numpy.array([dy1dt, dy0dt])


class Model03_BadV05(BioprocessModel):
    # name of parameter variable does not match the corresponding key
    def rhs(self, t, y):
        y0, y1 = y
        rate0 = self.model_parameters['rate0']
        any_parameter = self.model_parameters['rate1']
        dy0dt = rate0
        dy1dt = any_parameter
        return numpy.array([dy0dt, dy1dt])


class Model03_BadV06(BioprocessModel):
    # parameters are unpacked in wrong order
    def rhs(self, t, y):
        y0, y1 = y
        rate1, rate0 = self.model_parameters.to_numpy()
        dy0dt = rate0
        dy1dt = rate1
        return numpy.array([dy0dt, dy1dt])


class Model03_BadV07(Model03_V06):
    # state vector is unpacked in the wrong order
    def state_events(self, t, y, sw):
        y1, y0 = y
        rate0, rate1 = self.model_parameters.to_numpy()
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return numpy.array([event_y0, event_y1, event_t,])       


class Model03_BadV08(Model03_V06):
    # parameters are unpacked in wrong order
    def state_events(self, t, y, sw):
        y0, y1 = y
        rate1, rate0 = self.model_parameters.to_numpy()
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return numpy.array([event_y0, event_y1, event_t,])       


class Model03_BadV09(Model03_V06):
    # name of parameter variable does not match the corresponding key
    def state_events(self, t, y, sw):
        y0, y1 = y
        rate0 = self.model_parameters['rate0']
        any_parameter = self.model_parameters['rate1']
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return numpy.array([event_y0, event_y1, event_t,])   


class Model03_BadV10(Model03_V06):
    # name of parameter variable does not match the corresponding key
    def state_events(self, t, y, sw):
        y0, y1 = y
        any_parameter = self.model_parameters['rate1']
        event_y0 = y0
        event_y1 = y1
        event_t = t - 5
        return numpy.array([event_y0, event_y1, event_t,])   


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
        return numpy.array([dy0dt, dy1dt])


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
        return numpy.array([dy0dt, dy1dt])


class Model06_Bad03(Model06):

    # Has an undefined variable
    def state_events(self, t, y, sw):
        y0, y1 = y
        event_t1 = t - 5
        event_t3 = t - 1
        return numpy.array([event_t1, event_t2, event_t3])  


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


class ModelParametrizations():

    model_class  = {
        'model_01' : Model01,
        'model_02' : Model02,
        'model_03' : Model03,
        'model_03_v07' : Model03_V07,
        'model_04' : Model04,
        'model_05' : Model05,
        'model_06' : Model06,
    }

    model_names = {
        'model_01' : '1st model',
    }

    states = {
        'model_01' : ['y0' , 'y1'],
        'model_02' : ['y'],
        'model_03' : ['y0' , 'y1'],
        'model_03_v07' : ['y0' , 'y1'],
        'model_04' : ['y0' , 'y1'],
        'model_05' : ['yA', 'yB', 'yC', 'yD'],
        'model_06' : ['y0' , 'y1'],
    }

    model_parameters = {
        'model_01' : {'rate0' : 0.0, 'rate1' : 1.0},
        'model_02' : {'k' : 0.02},
        'model_03' : {'rate0' : 2.0, 'rate1' : 3.0},
        'model_03_v07' : {'rate0' : -2.0, 'rate1' : -3.0},
        'model_04' : {'rate0' : 4.0, 'rate1' : 5.0},
        'model_05' : {'rateA' : 10.0, 'rateB' : 11.0, 'rateC' : 12.0, 'rateD' : 13.0, 'rateE' : 14.0},
        'model_06' : {'rate0' : -2.0, 'rate1' : -3.0},
    }

    initial_values = {
        'model_01' : {'y00' : 0.0, 'y10' : 1.0},
        'model_02' : {'y0' : 100.0},
        'model_03' : {'y00' : 2.0, 'y10' : 3.0},
        'model_03_v07' : {'y00' : 20.0, 'y10' : 30.0},
        'model_04' : {'y00' : 4.0, 'y10' : 5.0},
        'model_05' : {'yA0' : 100.0, 'yB0' : 200.0, 'yC0' : 300.0, 'yD0': 400.0},
        'model_06' : {'y00' : 20.0, 'y10' : 30.0},
    }

    observation_functions = {
        'model_01' : ['obs_fun_01', 'obs_fun_02'],
        'model_02' : ['obs_fun_04'],
        'model_03' : ['obs_fun_01', 'obs_fun_02'],
        'model_03_v07' : ['obs_fun_01', 'obs_fun_02'],
        'model_04' : ['obs_fun_01', 'obs_fun_02', 'obs_fun_03'],
        'model_06' : ['obs_fun_01', 'obs_fun_02'],
    }


#%% Mock observations

class ObservationFunction01(ObservationFunction): 
    def observe(self, state_values):
        slope_01 = self.observation_parameters['slope_01']
        offset_01 = self.observation_parameters['offset_01']
        return state_values * slope_01 + offset_01


class ObservationFunction02(ObservationFunction):
    def observe(self, state_values):
        slope_02 = self.observation_parameters['slope_02']
        offset_02 = self.observation_parameters['offset_02']
        return state_values * slope_02 + offset_02


class ObservationFunction03(ObservationFunction):
    def observe(self, state_values):
        slope_03 = self.observation_parameters['slope_03']
        offset_03 = self.observation_parameters['offset_03']
        return state_values * slope_03 + offset_03


class ObservationFunction04(ObservationFunction):
    def observe(self, state_values):
        slope_04 = self.observation_parameters['slope_04']
        offset_04 = self.observation_parameters['offset_04']
        return state_values * slope_04 + offset_04


class ObservationFunction04_V01(ObservationFunction):
    def observe(self, state_values):
        offset_04, slope_04 = self.observation_parameters.to_numpy()
        return state_values * slope_04 + offset_04


class ObservationFunction05_V01(ObservationFunction):
    def observe(self, state_values):
        p1, p2, p3, p4, p5 = self.observation_parameters.to_numpy()
        return state_values + p1 + p2 + p3 + p4 + p5

    
class ObservationFunction05_V02(ObservationFunction):
    def observe(self, state_values):
        p1, \
            p2, p3, \
            p4, p5 \
            = self.observation_parameters.to_numpy()
        return state_values + p1 + p2 + p3 + p4 + p5


class ObservationFunction05_V03(ObservationFunction):
    def observe(self, state_values):
        p1, \
            p2, p3, \
            p4, p5 = self.observation_parameters.to_numpy()
        return state_values + p1 + p2 + p3 + p4 + p5


class ObservationFunction05_V04(ObservationFunction):
    def observe(self, state_values):
        p1 = self.observation_parameters['p1']
        p2 = self.observation_parameters['p2']
        p3 = self.observation_parameters['p3']
        p4 = self.observation_parameters['p4']
        p5 = self.observation_parameters['p5']
        return state_values + p1 + p2 + p3 + p4 + p5


# Bad variants of ObservationFunction04
class ObservationFunction04_Bad01(ObservationFunction):
    # observation parameters are unpacked in the wrong order
    def observe(self, model_values):
        slope_04, offset_04 = self.observation_parameters.to_numpy()
        return model_values * slope_04 + offset_04


class ObservationFunction04_Bad02(ObservationFunction):
    # observation parameter variable name does not match corresponding keys
    def observe(self, model_values):
        slope_04 = self.observation_parameters['slope_04']
        some_offset = self.observation_parameters['offset_04']
        return model_values * slope_04 + some_offset


class ObservationParametrizations():

    observed_functions = {
        'obs_fun_01' : ObservationFunction01,
        'obs_fun_02' : ObservationFunction02,
        'obs_fun_03' : ObservationFunction03,
        'obs_fun_04' : ObservationFunction04,
        'obs_fun_05' : ObservationFunction05_V01,
    }

    observed_states = {
        'obs_fun_01' : 'y0',
        'obs_fun_02' : 'y1',
        'obs_fun_03' : 'y0',
        'obs_fun_04' : 'y',
        'obs_fun_05' : 'y',
    }

    observation_parameters = {
        'obs_fun_01' : {'slope_01' : OBS_SLOPE_01, 'offset_01' : OBS_OFFSET_01},
        'obs_fun_02' : {'slope_02' : -2.0, 'offset_02' : -10.0},
        'obs_fun_03' : {'slope_03' : -1*OBS_SLOPE_01, 'offset_03' : -1*OBS_OFFSET_01},
        'obs_fun_04' : {'slope_04' : 2.0, 'offset_04' : 10.0},
        'obs_fun_05' : {'p1' : 10.0, 'p2' : 10.0, 'p3' : 10.0, 'p4' : 10.0, 'p5' : 10.0},
    }


#%% Helpers

class TestingHelpers():

    @staticmethod
    def get_model_building_blocks(curr_model:str):
        model_class = ModelParametrizations.model_class[curr_model]
        initial_values = ModelParametrizations.initial_values[curr_model]
        model_parameters = ModelParametrizations.model_parameters[curr_model]
        return model_class, initial_values, model_parameters

    @staticmethod
    def get_observation_building_blocks(curr_obs_fun:str):
        observation_function = ObservationParametrizations.observed_functions[curr_obs_fun]
        observation_parameters = ObservationParametrizations.observation_parameters[curr_obs_fun]
        observed_state = ObservationParametrizations.observed_states[curr_obs_fun]
        observation_parameters_with_state = dict(observation_parameters)
        observation_parameters_with_state.update({OBSERVED_STATE_KEY : observed_state})
        return (observation_function, observation_parameters_with_state)

    @staticmethod
    def get_observation_functions_parameters(observed_model:str):
        observation_functions_parameters = []
        for curr_obs_fun in ModelParametrizations.observation_functions[observed_model]:
            observation_functions_parameters.append(TestingHelpers.get_observation_building_blocks(curr_obs_fun))
        return observation_functions_parameters

    @staticmethod
    def get_observation_function_instance(curr_obs_fun:str):
        observation_function = ObservationParametrizations.observed_functions[curr_obs_fun]
        observation_parameters = ObservationParametrizations.observation_parameters[curr_obs_fun]
        observed_state = ObservationParametrizations.observed_states[curr_obs_fun]
        obs_fun = observation_function(observed_state=observed_state, observation_parameters=list(observation_parameters.keys()))
        obs_fun.set_parameters(observation_parameters)
        return obs_fun

    @staticmethod
    def get_observer_instance(observed_model:str):
        observation_functions_parameters = []
        for curr_obs_fun in ModelParametrizations.observation_functions[observed_model]:
            observation_functions_parameters.append(TestingHelpers.get_observation_building_blocks(curr_obs_fun))
        return ModelObserver(observation_functions_parameters)

    @staticmethod
    def get_simulator_instance(curr_model:str):
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        observation_functions_parameters = TestingHelpers.get_observation_functions_parameters(curr_model)
        return Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values, observation_functions_parameters=observation_functions_parameters)

    @staticmethod
    def get_extended_simulator_instance(curr_model:str):
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        observation_functions_parameters = TestingHelpers.get_observation_functions_parameters(curr_model)
        return ExtendedSimulator(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values, observation_functions_parameters=observation_functions_parameters)
        
    @staticmethod
    def noisy_samples(simulations:List[TimeSeries], samples=15, rel_err=0.05, abs_err=0.05, repetitions=3, with_errors=True):

        artifical_data = []

        for simulation in simulations:
            _name = simulation.name
            _replicate_id = simulation.replicate_id
            _timepoints = simulation.timepoints
            _values = simulation.values
            
            _rnd_values = [
                numpy.random.normal(loc=_values, scale=abs(_values*rel_err+abs_err)) 
                for _ in range(repetitions)
            ]

            _values = numpy.mean(_rnd_values, axis=0)
            _errors = numpy.std(_rnd_values, ddof=1, axis=0)
            selector = numpy.sort(
                numpy.random.choice(range(len(_timepoints)), size=samples, replace=False),
            )

            if with_errors:
                artifical_data.append(
                    Measurement(name=_name, timepoints=_timepoints[selector], values=_values[selector], errors=_errors[selector], replicate_id=_replicate_id),
                )
            else:
                artifical_data.append(
                    Measurement(name=_name, timepoints=_timepoints[selector], values=_values[selector], replicate_id=_replicate_id),
                )

        return artifical_data

    @staticmethod
    def get_caretaker(curr_model:str, replicate_ids:list=None) -> Caretaker:
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        observation_functions_parameters = TestingHelpers.get_observation_functions_parameters(curr_model)
        caretaker = Caretaker(
            bioprocess_model_class=model_class, 
            model_parameters=model_parameters, 
            initial_values=initial_values, 
            observation_functions_parameters=observation_functions_parameters,
            replicate_ids=replicate_ids
        )
        return caretaker


#%% Tests
 

class TestBioprocessModel(unittest.TestCase):

    def test_init_model(self):
        curr_model = 'model_01'
        model_class = ModelParametrizations.model_class[curr_model]
        model_parameters_list = list(ModelParametrizations.model_parameters[curr_model].keys())
        states = ModelParametrizations.states[curr_model]
        model_name = ModelParametrizations.model_names[curr_model]
        bad_states_01 = ['y0' , 'y1', 'y1']
        bad_states_02 = ['y0' , 'y1', 'Y1']

        model_class(states=states, model_parameters=model_parameters_list, model_name=model_name)
        model = model_class(states=states, model_parameters=model_parameters_list)

        print(model)

        with self.assertRaises(AttributeError):
            model.states = states
        with self.assertRaises(TypeError):
            model_class(states={state : numpy.nan for state in states}, model_parameters=model_parameters_list)
        with self.assertRaises(KeyError):
            model_class(states=bad_states_01, model_parameters=model_parameters_list)
        with self.assertRaises(KeyError):
            model_class(states=bad_states_02, model_parameters=model_parameters_list)


    def test_init_variants_Model05(self):

        curr_model = 'model_05'
        model_parameters = list(ModelParametrizations.model_parameters[curr_model].keys())
        states = ModelParametrizations.states[curr_model]
        models = [
            Model05, Model05_V01, Model05_V02, Model05_V03, Model05_V04, 
            Model05_V05, Model05_V06, Model05_V07, Model05_V08, Model05_V09, 
            Model05_V10, Model05_V11, Model05_V12, Model05_V13,
        ]

        for model in models:
            model(model_parameters=model_parameters, states=states)


    def test_init_model_with_events(self):
        curr_model = 'model_03'
        model_parameters_list = list(ModelParametrizations.model_parameters[curr_model].keys())
        states = ModelParametrizations.states[curr_model]

        model_03_variants = [Model03, Model03_V02, Model03_V03, Model03_V04, Model03_V05, Model03_V06, Model03_V07, Model03_V08, Model03_V09, Model03_V10, Model03_V11, Model03_V12, Model03_V13]
        sw0s = [
            [False], 
            [False], 
            [False, False], 
            [False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False],
            [False],
        ]

        for model_03_variant, sw0 in zip(model_03_variants, sw0s):
            model_03_variant(states=states, model_parameters=model_parameters_list, initial_switches=sw0)


    def test_initial_switch_detection(self):
        curr_model = 'model_03'
        model_parameters_list = list(ModelParametrizations.model_parameters[curr_model].keys())
        states = ModelParametrizations.states[curr_model]

        model_03_variants = [Model03, Model03_V02, Model03_V03, Model03_V04, Model03_V05, Model03_V06, Model03_V07, Model03_V08, Model03_V09, Model03_V10, Model03_V11, Model03_V12, Model03_V13]
        expected_sw0s = [
            [False], 
            [False], 
            [False, False], 
            [False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False],
            [False],
        ]

        for model_03_variant, expected in zip(model_03_variants, expected_sw0s):
            _model_object = model_03_variant(states=states, model_parameters=model_parameters_list)
            actual = _model_object.sw0
            self.assertListEqual(list(actual), expected)


    def test_property_setters(self):
        curr_model = 'model_03'
        model_class = ModelParametrizations.model_class[curr_model]
        model_parameters_list = list(ModelParametrizations.model_parameters[curr_model].keys())
        states = ModelParametrizations.states[curr_model]
        model_parameters = ModelParametrizations.model_parameters[curr_model]
        initial_values = ModelParametrizations.initial_values[curr_model]
        initial_switches = [True]

        bad_parameters_01 = ['rate0', 'rate1']
        bad_parameters_02 = {'rate0' : -999, 'rate_bad' : -999}
        bad_parameters_03 = {'rate0' : -999, 'rate0' : -999}
        bad_parameters_04 = {'rate0' : -999, 'Rate0' : -999}
        bad_initial_values_01 = ['y0' , 'y1']
        bad_initial_values_02 = {'y0_bad' : -999, 'y10' : -999}
        bad_initial_values_03 = {'y10' : -999, 'y10' : -999}
        bad_initial_values_04 = {'y10' : -999, 'Y10' : -999}
        bad_initial_switches_01 = [True, False]
        bad_initial_switches_02 = ['True']

        model = model_class(model_parameters=model_parameters_list, states=states)
        auto_sw0 = model.initial_switches

        # Setting the properties
        with self.assertRaises(TypeError):
            model.initial_values = bad_initial_values_01
        with self.assertRaises(TypeError):
            model.model_parameters = bad_parameters_01
        model.model_parameters = model_parameters
        model.initial_values = initial_values
        model.initial_switches = initial_switches
        manual_sw0 = model.initial_switches
        self.assertNotEqual(auto_sw0, manual_sw0)
        with self.assertRaises(ValueError):
            model.initial_switches = bad_initial_switches_01
        with self.assertRaises(ValueError):
            model.initial_switches = bad_initial_switches_02
        with self.assertRaises(KeyError):
            model.model_parameters = bad_parameters_02
        with self.assertRaises(KeyError):
            model.model_parameters = bad_parameters_03
        with self.assertRaises(KeyError):
            model.model_parameters = bad_parameters_04
        with self.assertRaises(KeyError):
            model.initial_values = bad_initial_values_02
        with self.assertRaises(KeyError):
            model.initial_values = bad_initial_values_03
        with self.assertRaises(KeyError):
            model.initial_values = bad_initial_values_04


    def test_set_parameters(self):
        curr_model = 'model_03'
        model_class = ModelParametrizations.model_class[curr_model]
        model_parameters_list = list(ModelParametrizations.model_parameters[curr_model].keys())
        states = ModelParametrizations.states[curr_model]
        model_parameters = ModelParametrizations.model_parameters[curr_model]
        initial_values = ModelParametrizations.initial_values[curr_model]
        model = model_class(model_parameters=model_parameters_list, states=states)

        bad_parameters = {'y0' : -999, 'Y0' : -999}
        with self.assertRaises(KeyError):
            model.set_parameters(bad_parameters)
        model.set_parameters(model_parameters)
        for p in model.model_parameters.keys():
            self.assertIsNotNone(model.model_parameters[p])
        model.set_parameters(initial_values)
        for iv in model.initial_values.keys():
            self.assertIsNotNone(model.initial_values[iv])
        model.set_parameters({'unregarded_parameter' : -9999})


class TestObservationFunction(unittest.TestCase):

    def test_init(self):
        curr_obs_fun = 'obs_fun_01'
        obs_fun = ObservationParametrizations.observed_functions[curr_obs_fun]
        observed_state = ObservationParametrizations.observed_states[curr_obs_fun]
        observation_parameters = ObservationParametrizations.observation_parameters[curr_obs_fun]
        observation_parameters_list = list(observation_parameters.keys())

        obs_fun_copy01 = obs_fun(observed_state=observed_state, observation_parameters=observation_parameters_list)
        for obs_par in obs_fun_copy01.observation_parameters:
            self.assertIsNone(obs_fun_copy01.observation_parameters[obs_par])

        obs_fun_copy02 = obs_fun(observed_state=observed_state, observation_parameters=observation_parameters)
        for obs_par in obs_fun_copy02.observation_parameters:
            self.assertIsNone(obs_fun_copy02.observation_parameters[obs_par])

        invalid_observed_states = [1, 1.0, True, {'a':1, 'b':2}, ['a', 'b']]
        for invalid_observed_state in invalid_observed_states:
            with self.assertRaises(ValueError):
                 obs_fun(observed_state=invalid_observed_state, observation_parameters=observation_parameters_list)


    def test_model_state(self):
        curr_obs_fun = 'obs_fun_01'
        obs_fun = ObservationParametrizations.observed_functions[curr_obs_fun]
        observed_state = ObservationParametrizations.observed_states[curr_obs_fun]
        observation_parameters_list = list(ObservationParametrizations.observation_parameters[curr_obs_fun].keys())

        obs_fun = obs_fun(observed_state=observed_state, observation_parameters=observation_parameters_list)

        with self.assertRaises(AttributeError):
            obs_fun.observed_state = 'OtherState'


    def test_setting_parameters(self):
        curr_obs_fun = 'obs_fun_01'
        obs_fun = ObservationParametrizations.observed_functions[curr_obs_fun]
        observed_state = ObservationParametrizations.observed_states[curr_obs_fun]
        observation_parameters = ObservationParametrizations.observation_parameters[curr_obs_fun]
        observation_parameters_list = list(observation_parameters.keys())
        observation_parameters_with_state = dict(observation_parameters)
        observation_parameters_with_state.update({OBSERVED_STATE_KEY : 'y0'})

        bad_parameters_01 = ['slope_01', 'offset_01']
        bad_parameters_02 = {'slope_01' : -999, 'offset_bad' : -999}
        bad_parameters_03 = dict(observation_parameters)
        bad_parameters_03.update({OBSERVED_STATE_KEY : 'ZYZ'})
        bad_parameters_04 = {'slope_01' : -999, 'Slope_01' : -999}

        obs_fun = obs_fun(observed_state=observed_state, observation_parameters=observation_parameters_list)

        obs_fun.observation_parameters = observation_parameters
        obs_fun.observation_parameters = observation_parameters_with_state

        with self.assertRaises(ValueError):
            obs_fun.observation_parameters = bad_parameters_01
        with self.assertRaises(ValueError):
            obs_fun.observation_parameters = bad_parameters_03
        with self.assertRaises(KeyError):
            obs_fun.observation_parameters = bad_parameters_02
        with self.assertRaises(KeyError):
            obs_fun.observation_parameters = bad_parameters_04
        with self.assertRaises(AttributeError):
            obs_fun.observed_state = 'XYZ'

        obs_fun.set_parameters({'slope_01' : -999})
        obs_fun.set_parameters(observation_parameters)
        obs_fun.set_parameters({'unregarded_parameter' : -9999})


    def test_get_observation(self):

        class BlankObsFun(ObservationFunction):
            def observe(self, state_values):
                return super().observe(state_values)
        blank_obs_fun = BlankObsFun(observed_state='A', observation_parameters=['p1'])
        with self.assertRaises(NotImplementedError):
            blank_obs_fun.observe(numpy.array([0, 1, 2, 3]))
        
        print(blank_obs_fun)

        curr_obs_fun = 'obs_fun_01'
        obs_fun = TestingHelpers.get_observation_function_instance(curr_obs_fun)
        t = numpy.linspace(0, 10)
        y0 = numpy.sqrt(t)

        model_state_copy01 = ModelState(name='y0', timepoints=t, values=y0)
        observation = obs_fun.get_observation(model_state_copy01)
        expected = y0 * OBS_SLOPE_01 + OBS_OFFSET_01
        actual = observation.values
        for _expected, _actual in zip(expected, actual):
            self.assertEqual(_expected, _actual)

        model_state_copy02 = ModelState(name='y_bad', timepoints=t, values=y0)
        with self.assertRaises(KeyError):
            observation = obs_fun.get_observation(model_state_copy02)


class TestModelChecker(unittest.TestCase):

    def setUp(self):
        self.model_checker = ModelChecker()


    def test_bioprocess_model_checking(self):

        curr_model = 'model_03'
        model_parameters = ModelParametrizations.model_parameters[curr_model]
        initial_values = ModelParametrizations.initial_values[curr_model]
        bad_models = [Model03_BadV01, Model03_BadV02, Model03_BadV03, Model03_BadV04, Model03_BadV05, Model03_BadV06, Model03_BadV07, Model03_BadV08, Model03_BadV09, Model03_BadV10]
        for bad_model in bad_models:
            simulator = Simulator(bioprocess_model_class=bad_model, model_parameters=model_parameters, initial_values=initial_values)
            self.model_checker.check_model_consistency(simulator)


    def test_observation_function_checking(self):

        curr_model = 'model_02'
        model_class = ModelParametrizations.model_class[curr_model]
        model_parameters = ModelParametrizations.model_parameters[curr_model]
        initial_values = ModelParametrizations.initial_values[curr_model]

        curr_obs_fun = 'obs_fun_04'
        observed_state = ObservationParametrizations.observed_states[curr_obs_fun]
        obs_pars = ObservationParametrizations.observation_parameters[curr_obs_fun]
        observes_state = ObservationParametrizations.observed_states[curr_obs_fun]
        obs_pars[OBSERVED_STATE_KEY] = observes_state

        _obs_funs = [
            ObservationFunction04_V01,
            ObservationFunction04_Bad01,
            ObservationFunction04_Bad02,
        ]

        for _obs_fun in _obs_funs:
            simulator = Simulator(
                bioprocess_model_class=model_class, 
                model_parameters=model_parameters, 
                initial_values=initial_values, 
                observation_functions_parameters=[(_obs_fun, obs_pars)],
            )
            self.model_checker.check_model_consistency(simulator)


    def test_observation_function_checking_multi_line(self):

        curr_model = 'model_02'
        model_class = ModelParametrizations.model_class[curr_model]
        model_parameters = ModelParametrizations.model_parameters[curr_model]
        initial_values = ModelParametrizations.initial_values[curr_model]

        curr_obs_fun = 'obs_fun_05'
        observed_state = ObservationParametrizations.observed_states[curr_obs_fun]
        obs_pars = ObservationParametrizations.observation_parameters[curr_obs_fun]
        observes_state = ObservationParametrizations.observed_states[curr_obs_fun]
        obs_pars[OBSERVED_STATE_KEY] = observes_state

        _obs_funs = [
            ObservationFunction05_V01,
            ObservationFunction05_V02,
            ObservationFunction05_V03,
            ObservationFunction05_V04,
        ]

        for _obs_fun in _obs_funs:
            simulator = Simulator(
                bioprocess_model_class=model_class, 
                model_parameters=model_parameters, 
                initial_values=initial_values, 
                observation_functions_parameters=[(_obs_fun, obs_pars)],
            )
            self.model_checker.check_model_consistency(simulator)



    def test_call_checks_model_methods(self):

        curr_model = 'model_06'
        model_parameters = ModelParametrizations.model_parameters[curr_model]
        initial_values = ModelParametrizations.initial_values[curr_model]

        # The number of explicity provided switches does not match the number of events
        simulator = Simulator(
            bioprocess_model_class=Model06, 
            model_parameters=model_parameters, 
            initial_values=initial_values,
            initial_switches=[False]*4,
        )
        self.model_checker.check_model_consistency(simulator)

        with self.assertRaises(NameError):
            simulator = Simulator(
                bioprocess_model_class=Model06_Bad01, 
                model_parameters=model_parameters, 
                initial_values=initial_values,
            )
            self.model_checker.check_model_consistency(simulator)

        with self.assertRaises(NameError):
            simulator = Simulator(
                bioprocess_model_class=Model06_Bad02, 
                model_parameters=model_parameters, 
                initial_values=initial_values,
            )
            self.model_checker.check_model_consistency(simulator)

        with self.assertRaises(NameError):
            simulator = Simulator(
                bioprocess_model_class=Model06_Bad03, 
                model_parameters=model_parameters, 
                initial_values=initial_values,
            )
            self.model_checker.check_model_consistency(simulator)

        with self.assertRaises(NameError):
            simulator = Simulator(
                bioprocess_model_class=Model06_Bad04, 
                model_parameters=model_parameters, 
                initial_values=initial_values,
            )
            self.model_checker.check_model_consistency(simulator)

        # For this model, the number of events depends on the switches
        simulator = Simulator(
            bioprocess_model_class=Model06_Bad05, 
            model_parameters=model_parameters, 
            initial_values=initial_values,
            initial_switches=[False]*3,
        )
        self.model_checker.check_model_consistency(simulator)

        # For this model, the Simulator must be instantiated with the correct switches
        simulator = Simulator(
            bioprocess_model_class=Model06_V02, 
            model_parameters=model_parameters, 
            initial_values=initial_values,
            initial_switches=[False]*3
        )
        self.model_checker.check_model_consistency(simulator)

        # For these models, parameter unpacking in change_states method is inconsistent
        simulator = Simulator(
            bioprocess_model_class=Model06_Bad06, 
            model_parameters=model_parameters, 
            initial_values=initial_values,
        )
        self.model_checker.check_model_consistency(simulator)

        simulator = Simulator(
            bioprocess_model_class=Model06_Bad07, 
            model_parameters=model_parameters, 
            initial_values=initial_values,
        )
        self.model_checker.check_model_consistency(simulator)

        # For this model, the state vector unpacking in change_states method is inconsistent
        simulator = Simulator(
            bioprocess_model_class=Model06_Bad08, 
            model_parameters=model_parameters, 
            initial_values=initial_values,
        )
        self.model_checker.check_model_consistency(simulator)


class TestModelObserver(unittest.TestCase):

    def test_init_observer(self):
        observed_models = ['model_01', 'model_03']
        for observed_model in observed_models:
            observation_functions_parameters = []
            for curr_obs_fun in ModelParametrizations.observation_functions[observed_model]:
                observation_functions_parameters.append(TestingHelpers.get_observation_building_blocks(curr_obs_fun))
            model_observer = ModelObserver(observation_functions_parameters)
            print(model_observer)

        # Parameter names must be unique among all observations functions handled by a ModelObserver object
        bad_observation_functions_parameters_01 = [
            (ObservationFunction01, {'slope_01' : 2, 'offset_01' : 10, OBSERVED_STATE_KEY : 'y0'}),
            (ObservationFunction01, {'slope_01' : 2, 'offset_01' : 10, OBSERVED_STATE_KEY : 'y0'}),
        ]
        with self.assertRaises(KeyError):
            ModelObserver(bad_observation_functions_parameters_01)

        bad_observation_functions_parameters_02 = [
            [ObservationFunction01, {'slope_01' : 2, 'offset_01' : 10, OBSERVED_STATE_KEY : 'y0'}],
            (ObservationFunction02, {'slope_02' : 2, 'offset_02' : 10, OBSERVED_STATE_KEY : 'y0'}),
        ]
        with self.assertRaises(ValueError):
            ModelObserver(bad_observation_functions_parameters_02)

        bad_observation_functions_parameters_03 = [
            (ObservationFunction01, {'slope_01' : 2, 'offset_01' : 10, OBSERVED_STATE_KEY : 'y0'}),
            [ObservationFunction02, {'slope_02' : 2, 'offset_02' : 10, OBSERVED_STATE_KEY : 'y0'}],
        ]
        with self.assertRaises(ValueError):
            ModelObserver(bad_observation_functions_parameters_03)

        bad_observation_functions_parameters_04 = [
            (ObservationFunction01, {'slope_01' : 2, 'offset_01' : 10, OBSERVED_STATE_KEY : 'y0'}),
            ('ObservationFunction02', {'slope_02' : 2, 'offset_02' : 10, OBSERVED_STATE_KEY : 'y0'}),
        ]
        with self.assertRaises(Exception):
            ModelObserver(bad_observation_functions_parameters_04)
        
        bad_observation_functions_parameters_05 = [
            (ObservationFunction01, {'slope_01' : 2, 'offset_01' : 10, OBSERVED_STATE_KEY : 'y0'}),
            (ObservationFunction02, ['slope_02', 'offset_02', 'y0']),
        ]
        with self.assertRaises(TypeError) as cm:
            ModelObserver(bad_observation_functions_parameters_05)
        print(cm.exception)

        bad_observation_functions_parameters_06 = [
            (ObservationFunction01, {'slope_01' : 2, 'offset_01' : 10, OBSERVED_STATE_KEY : 'y0'}),
            (ObservationFunction02, {'slope_02' : 2, 'offset_02' : 10}),
        ]
        with self.assertRaises(KeyError):
            ModelObserver(bad_observation_functions_parameters_06)

        # The same state can be observed by multiple observation functions
        obs_state = 'y0'
        observation_functions_parameters = [
            (ObservationFunction01, {'slope_01' : 2, 'offset_01' : 10, OBSERVED_STATE_KEY : obs_state}),
            (ObservationFunction02, {'slope_02' : 2, 'offset_02' : 10, OBSERVED_STATE_KEY : obs_state}),
        ]
        observer = ModelObserver(observation_functions_parameters)
        self.assertGreater(len(observer._lookup[obs_state]), 1)


    def test_observe(self):
        t_y0 = numpy.linspace(0, 10)
        y0 = numpy.sqrt(t_y0)
        t_y1 = numpy.logspace(0, 10)
        y1 = numpy.square(t_y1)
        model_state_y0 = ModelState(name='y0', timepoints=t_y0, values=y0)
        model_state_y1 = ModelState(name='y1', timepoints=t_y1, values=y1)
        model_states = [model_state_y0, model_state_y1]

        observed_model = 'model_01'
        observation_functions_parameters_01 = []
        for curr_obs_fun in ModelParametrizations.observation_functions[observed_model]:
            observation_functions_parameters_01.append(TestingHelpers.get_observation_building_blocks(curr_obs_fun))
        observer_01 = ModelObserver(observation_functions_parameters_01)
        observations_01 = observer_01.get_observations(model_states)

        # Can only observe unique model states
        #with self.assertRaises(KeyError):
        #    observer_01.get_observations({'y0' : model_state_y0, 'y1' : model_state_y0})

        # Can have more than one observation for a certain modelstate
        observed_model = 'model_04'
        observation_functions_parameters_04 = []
        for curr_obs_fun in ModelParametrizations.observation_functions[observed_model]:
            observation_functions_parameters_04.append(TestingHelpers.get_observation_building_blocks(curr_obs_fun))
        observer_04 = ModelObserver(observation_functions_parameters_04)
        observations_04 = observer_04.get_observations(model_states)
        observed_states = []
        for obs in observations_04:
            observed_states.append(obs.observed_model_state)
        self.assertEqual(observed_states.count('y0'), 2)
        self.assertEqual(observed_states.count('y1'), 1)


    def test_set_parameters(self):
        observed_model = 'model_01'
        observer_before = TestingHelpers.get_observer_instance(observed_model)
        observer_after = TestingHelpers.get_observer_instance(observed_model)
        new_obs_pars = {p : VERY_HIGH_NUMBER for p in observer_after._obs_pars_names}
        
        observer_after.set_parameters(new_obs_pars)
        for obs_fun_name in observer_before.observation_functions.keys():
            obs_fun_before = observer_before.observation_functions[obs_fun_name]
            for new_p_before in obs_fun_before.observation_parameters.keys():
                self.assertNotEqual(obs_fun_before.observation_parameters[new_p_before], VERY_HIGH_NUMBER)
            obs_fun_after = observer_after.observation_functions[obs_fun_name]
            for new_p_after in obs_fun_after.observation_parameters.keys():
                self.assertEqual(obs_fun_after.observation_parameters[new_p_after], VERY_HIGH_NUMBER)


class TestSimulator(unittest.TestCase):
    
    def test_init(self):
        curr_model = 'model_03'
        model_class = ModelParametrizations.model_class[curr_model]
        model_parameters = ModelParametrizations.model_parameters[curr_model]
        model_parameters_list = list(model_parameters.keys())
        initial_values = ModelParametrizations.initial_values[curr_model]
        states = ModelParametrizations.states[curr_model]
        initial_switches = [False]

        Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters_list, states=states, initial_values=initial_values, initial_switches=initial_switches)
        Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, states=states, initial_values=initial_values, initial_switches=initial_switches)
        Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, states=states)
        Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values)

        with self.assertRaises(ValueError):
            Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters_list)
        with self.assertRaises(TypeError):
            Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, states={'a':1, 'b':2})
        with self.assertRaises(ValueError):
            Simulator(bioprocess_model_class=model_class, model_parameters=True, states=states)


    def test_set_parameters(self):
        curr_model = 'model_03'
        model_class = ModelParametrizations.model_class[curr_model]
        model_parameters = ModelParametrizations.model_parameters[curr_model]
        model_parameters_list = list(model_parameters.keys())
        initial_values = ModelParametrizations.initial_values[curr_model]
        states = ModelParametrizations.states[curr_model]

        simulator_copy01 = Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values)
        simulator_copy01.set_parameters(initial_values)

        for iv in simulator_copy01.bioprocess_model.initial_values.keys():
            self.assertIsNotNone(simulator_copy01.bioprocess_model.initial_values[iv])

        observation_functions_parameters = TestingHelpers.get_observation_functions_parameters(curr_model)
        simulator_copy02 = Simulator(
            bioprocess_model_class=model_class, 
            model_parameters=model_parameters, 
            initial_values=initial_values, 
            observation_functions_parameters=observation_functions_parameters,
        )


    def test_reset(self):
        pass


    def test_simulation_states(self):

        curr_model = 'model_03'
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        t_01 = 24
        t_02 = numpy.array(t_01)
        t_03 = numpy.array([t_01])
        t_04 = numpy.arange(t_01)
        t_05 = numpy.linspace(0, t_01)

        simulator_copy01 = Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values)
        simulator_copy01.simulate(t=t_01, verbosity=50)
        simulator_copy01.simulate(t=t_02, verbosity=50)
        simulator_copy01.simulate(t=t_03, verbosity=50)
        simulator_copy01.simulate(t=t_04, verbosity=50)
        simulator_copy01.simulate(t=t_05, verbosity=50)

        simulator_copy02 = Simulator(bioprocess_model_class=model_class, model_parameters=list(model_parameters.keys()), initial_values=initial_values)
        # The resulting TypeError is suppresed by assimulo
        simulator_copy02.simulate(t=24, verbosity=50)
        simulator_copy02.simulate(t=24, verbosity=50, parameters=model_parameters)

        # run a model where events happen
        curr_model = 'model_03_v07'
        simulator_02 = TestingHelpers.get_simulator_instance(curr_model)
        simulator_02.simulate(t=24, verbosity=50)
    

    def test_integrator_kwargs(self):

        curr_model = 'model_02'
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        simulator = Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values)
        t = 24
        sims_01 = simulator.simulate(t=t, verbosity=50)

        with self.assertRaises(ValueError):
            simulator.integrator_kwargs = ('atol', 1e-8, 'rtol' , 1e-8)

        simulator.integrator_kwargs = {'atol' : 1e-12, 'rtol' : 1e-12}
        sims_02 = simulator.simulate(t=t, verbosity=50)

        # lowering the integrator tolerances increases the number of integration timepoints
        for _sim_01, _sim_02 in zip(sims_01, sims_02):
            self.assertTrue(len(_sim_01.timepoints) < len(_sim_02.timepoints))

        simulator.integrator_kwargs = None
        sims_03 = simulator.simulate(t=t, verbosity=50)
        for _sim_01, _sim_03 in zip(sims_01, sims_03):
            self.assertEqual(len(_sim_01.timepoints), len(_sim_03.timepoints))
            for _v1, _v3 in zip(_sim_01.values, _sim_03.values):
                self.assertAlmostEqual(_v1, _v3)


    def test_compare_with_analytical_solution(self):
        curr_model = 'model_02'
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        simulator = Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values)
        t = numpy.linspace(0, 10, 5)
        
        y_analytical = initial_values['y0'] * numpy.exp(-model_parameters['k'] * t)
        simulation = simulator.simulate(t=t, verbosity=50)
        _simulation = utils.Helpers.extract_time_series(simulation, name='y', replicate_id=simulator.replicate_id)
        for _y_analytical, _y_simulation in zip(y_analytical, _simulation.values):
            self.assertAlmostEqual(_y_analytical, _y_simulation, 3)


    def test_simulation_states_and_observations(self):
        curr_model = 'model_04'
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        observation_functions_parameters = TestingHelpers.get_observation_functions_parameters(curr_model)

        simulator = Simulator(bioprocess_model_class=model_class, model_parameters=model_parameters, initial_values=initial_values, observation_functions_parameters=observation_functions_parameters)
        simulation = simulator.simulate(t=24, verbosity=50)
        expected = sorted(['y0', 'y1', 'ObservationFunction01', 'ObservationFunction02', 'ObservationFunction03'], key=str.lower)
        actual = sorted([_simulation.name for _simulation in simulation], key=str.lower)
        self.assertListEqual(expected, actual)

        # all observations must be zero with these parameters
        different_obs_pars = {
            'slope_01' : 0, 
            'slope_02' : 0, 
            'slope_03' : 0, 
            'offset_01' : 0,
            'offset_02' : 0,
            'offset_03' : 0,
        }
        simulation_02 = simulator.simulate(t=24, verbosity=50, parameters=different_obs_pars)
        for _sim in simulation_02:
            if isinstance(_sim, Observation):
                for value in _sim.values:
                    self.assertEqual(value, 0)

        # states do not change with these parameters
        different_model_pars = {
            'y00' : 100, 
            'y10' : 200,
            'rate0' : 0,
            'rate1' : 0,
        }
        simulation_03 = simulator.simulate(t=24, verbosity=50, parameters=different_model_pars)
        for _sim in simulation_03:
            if _sim.name == 'y0':
                for value in _sim.values:
                    self.assertTrue(value, different_model_pars['y00'])
            elif _sim.name == 'y1':
                for value in _sim.values:
                    self.assertTrue(value, different_model_pars['y10'])
        
                    
    def test_handle_CVodeError(self):
        curr_model = 'model_06'
        timepoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        simulator = TestingHelpers.get_simulator_instance(curr_model)
        simulator.simulate(t=timepoints, verbosity=50)

        different_parameter = {'rate0' : numpy.nan}
        with self.assertRaises(CVodeError):
            simulator.simulate(t=timepoints, parameters=different_parameter, verbosity=50)

        toxic_parameters = {'rate0' : 0, 'rate1' : 0}
        with self.assertRaises(CVodeError):
            simulator.simulate(t=timepoints, parameters=toxic_parameters, verbosity=50)


    def test_simulation_all_mock_models(self):
        models = [
            Model01, 
            Model02, 
            Model03, Model03_V02, Model03_V03, Model03_V04, Model03_V05, 
            Model03_V06, Model03_V07, Model03_V08, Model03_V09, Model03_V10, 
            Model03_V11, Model03_V12, Model03_V13, 
            Model04, 
            Model05, Model05_V01, Model05_V02, Model05_V03, Model05_V04, 
            Model05_V05, Model05_V06, Model05_V07, Model05_V08, Model05_V09, 
            Model05_V10, Model05_V11, Model05_V12, Model05_V13, Model06
            ]
        model_names = [
            'model_01', 
            'model_02', 
            'model_03', 'model_03', 'model_03', 'model_03', 'model_03', 
            'model_03', 'model_03', 'model_03', 'model_03', 'model_03', 
            'model_03', 'model_03', 'model_03', 
            'model_04', 
            'model_05', 'model_05', 'model_05', 'model_05', 'model_05', 
            'model_05', 'model_05', 'model_05', 'model_05', 'model_05', 
            'model_05', 'model_05', 'model_05', 'model_05', 'model_06'
            ]

        for model, curr_model in zip(models, model_names):
            initial_values = ModelParametrizations.initial_values[curr_model]
            model_parameters = ModelParametrizations.model_parameters[curr_model]
            simulator = Simulator(bioprocess_model_class=model, model_parameters=model_parameters, initial_values=initial_values)
            simulator.simulate(t=24, verbosity=50)



class TestExtendedSimulator(unittest.TestCase):

    def setUp(self):
        self.metrics = ['SS', 'WSS', 'negLL']


    def test_methods(self):
        curr_model = 'model_03'
        ext_simulator = TestingHelpers.get_extended_simulator_instance(curr_model)
        ext_simulator.get_all_parameters()
        ext_simulator._get_allowed_measurement_keys()


    def test_get_loss_related_methods(self):
        curr_model = 'model_04'
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        observation_functions_parameters = TestingHelpers.get_observation_functions_parameters(curr_model)
        ext_simulator = ExtendedSimulator(
            bioprocess_model_class=model_class, 
            model_parameters=model_parameters, 
            initial_values=initial_values, 
            observation_functions_parameters=observation_functions_parameters,
            replicate_id=SINGLE_ID,
        )
        simulation = ext_simulator.simulate(t=24, verbosity=50)
        data = TestingHelpers.noisy_samples(simulation, samples=5)
        data_wo_errs = TestingHelpers.noisy_samples(simulation, samples=5, with_errors=False)

        for metric in self.metrics:
            ext_simulator._get_loss(metric=metric, measurements=data)
            ext_simulator._get_loss(metric=metric, measurements=data, parameters=model_parameters)

        ext_simulator._get_loss(metric='SS', measurements=data_wo_errs)
        with self.assertRaises(AttributeError):
            ext_simulator._get_loss(metric='WSS', measurements=data_wo_errs)
        with self.assertRaises(AttributeError):
            ext_simulator._get_loss(metric='negLL', measurements=data_wo_errs)

        for metric in self.metrics:
            ext_simulator._get_loss_for_minimzer(metric, model_parameters, data, True, True)
            ext_simulator._get_loss_for_minimzer(metric, model_parameters, data, False, True)
            ext_simulator._get_loss_for_minimzer(metric, model_parameters, data, False, False)
            # The resulting TypeError is suppressed by assimulo 
            ext_simulator._get_loss_for_minimzer(metric, {p : None for p in model_parameters}, data, True, True)


class TestParameterMapper(unittest.TestCase):

    def test_init(self):

        replicate_id = '1st'
        global_name = 'p'
        local_name = 'p_loc'
        value = 1234

        ParameterMapper(replicate_id, global_name, local_name, value)

        _pm = ParameterMapper(replicate_id, global_name)
        self.assertEqual(_pm.local_name, f'{global_name}_{replicate_id}')

        _pm = ParameterMapper(replicate_id, global_name, value=value)
        self.assertEqual(_pm.local_name, f'{global_name}_{replicate_id}')

        with self.assertRaises(ValueError):
            ParameterMapper('all', global_name)

        with self.assertRaises(ValueError):
            ParameterMapper(['1st', '2nd'], global_name)


class TestParameter(unittest.TestCase):

    def test_init(self):
        Parameter(global_name='p_global', replicate_id='1st')
        Parameter(global_name='p_global', replicate_id='1st', local_name='p_local')
        Parameter(global_name='p_global', replicate_id='1st', local_name='p_local', value=1000)


class TestParameterManager(unittest.TestCase):

    def setUp(self):
        self.replicate_ids = ['1st', '2nd', '3rd']
        self.curr_model = 'model_03'
        self.parameters = {}
        self.parameters.update(ModelParametrizations.model_parameters[self.curr_model])
        self.parameters.update(ModelParametrizations.initial_values[self.curr_model])
        for _obs_fun in ModelParametrizations.observation_functions[self.curr_model]:
            self.parameters.update(ObservationParametrizations.observation_parameters[_obs_fun])


    def test_init(self):

        parameter_manager = ParameterManager(self.replicate_ids, self.parameters)

        bad_ids = [
            ['1st', '1St', '2nd', '3rd'], 
            ['1st', '1st', '2nd', '3rd'], 
            ]
        for _bad_ids in bad_ids:
            with self.assertRaises(ValueError):
                ParameterManager(_bad_ids, self.parameters)

        with self.assertRaises(AttributeError):
            parameter_manager.replicate_ids = self.replicate_ids

        with self.assertRaises(AttributeError):
            parameter_manager.global_parameters = self.parameters
        
        with self.assertRaises(TypeError):
            ParameterManager(self.replicate_ids, tuple(self.parameters.keys()))

        non_unique_parameters = list(self.parameters.keys())
        non_unique_parameters.extend(list(self.parameters.keys()))
        with self.assertRaises(ValueError):
            ParameterManager(self.replicate_ids, non_unique_parameters)

        parameter_manager_02 = ParameterManager(self.replicate_ids, list(self.parameters.keys()))
        for p in parameter_manager_02.global_parameters.keys():
            self.assertTrue(numpy.isnan(parameter_manager_02.global_parameters[p]))


    def test_parameter_setting_and_mapping(self):
        parameter_manager_01 = ParameterManager(self.replicate_ids, self.parameters)
        parameter_manager_02 = ParameterManager(self.replicate_ids, self.parameters)
        parameter_manager_03 = ParameterManager(self.replicate_ids, self.parameters)

        mappings_01 = []
        mappings_02 = []
        for i, _id in enumerate(self.replicate_ids):
            for p in self.parameters:
                _mapping_01 = ParameterMapper(replicate_id=_id, global_name=p, local_name=f'{p}_{_id}', value=self.parameters[p]*(i+1)*1.1)
                mappings_01.append(_mapping_01)
                _mapping_02 = ParameterMapper(replicate_id=_id, global_name=p, local_name=f'{p}_{_id}', value=self.parameters[p]*(i+1)*1.1)
                mappings_02.append(_mapping_02)
        parameter_manager_01.apply_mappings(mappings_01)
        parameter_manager_02.apply_mappings(mappings_02)

        self.assertTrue(pandas.DataFrame.equals(parameter_manager_01.parameter_mapping, parameter_manager_02.parameter_mapping))

        # now set some of the mapped parameters
        new_parameters = {f'{p}_{self.replicate_ids[2]}' : 1111 for p in self.parameters}
        parameter_manager_01.set_parameter_values(new_parameters)
        # save the curent mapping for comparison
        _pm_before = copy.deepcopy(parameter_manager_01.parameter_mapping)
        # this has no effect since the local parameter names have all been changed
        parameter_manager_01.set_parameter_values(self.parameters)
        self.assertTrue(pandas.DataFrame.equals(_pm_before, parameter_manager_01.parameter_mapping))

        mappings_03 = [
            ParameterMapper(replicate_id=self.replicate_ids, global_name=list(self.parameters.keys())[3], local_name=list(self.parameters.keys())[3], value=1234),
            ParameterMapper(['1st', '2nd'], list(self.parameters.keys())[4], f'{list(self.parameters.keys())[4]}_1', 4321),
            ParameterMapper('all', list(self.parameters.keys())[5], f'{list(self.parameters.keys())[5]}', 9999),
            ParameterMapper(replicate_id=['1st', '3rd'], global_name=list(self.parameters.keys())[2], local_name=f'{list(self.parameters.keys())[2]}_1', value=-666666),
            ]
        parameter_manager_03.apply_mappings(mappings_03)
        display(parameter_manager_03.parameter_mapping)

        for _id in self.replicate_ids:
            display(parameter_manager_03.get_parameters_for_replicate(_id))

        # items of mappings must be either tuple of ParameterMapper
        invalid_mappings_01 = [
            'Im no tuple',
            ]
        with self.assertRaises(TypeError):
            parameter_manager_03.apply_mappings(invalid_mappings_01)

        # has an invalid replicate id
        invalid_mappings_02 = [
            ParameterMapper(replicate_id=self.replicate_ids, global_name=list(self.parameters.keys())[3], local_name=list(self.parameters.keys())[3], value=1234),
            ParameterMapper('Invalid_id', list(self.parameters.keys())[4], f'{list(self.parameters.keys())[4]}_1', 4321),
            ]
        with self.assertRaises(ValueError):
            parameter_manager_03.apply_mappings(invalid_mappings_02)
        
        # has a list of invalid replicate ids
        invalid_mappings_03 = [
            ParameterMapper(replicate_id=self.replicate_ids, global_name=list(self.parameters.keys())[3], local_name=list(self.parameters.keys())[3], value=1234),
            ParameterMapper(['Invalid_id1', 'Invalid_id2'], list(self.parameters.keys())[4], f'{list(self.parameters.keys())[4]}_1', 4321),
            ]
        with self.assertRaises(ValueError):
            parameter_manager_03.apply_mappings(invalid_mappings_03)

        # has an invalid global parameter name
        invalid_mappings_04 = [
            ParameterMapper('all', list(self.parameters.keys())[5], f'{list(self.parameters.keys())[5]}', 9999),
            ParameterMapper('all', 'invalid_global_parameter', 'some_local_name', 9999),
            ]
        with self.assertRaises(ValueError):
            parameter_manager_03.apply_mappings(invalid_mappings_04)

        unknown_parameters = {f'{p}_unknown' : 1010101 for p in self.parameters}
        with self.assertWarns(UserWarning):
            parameter_manager_01.set_parameter_values(unknown_parameters)

        # A Parameter object with no values of local_name and value can also be used
        parameter_manager_03.apply_mappings(
            [
                ParameterMapper(replicate_id='2nd', global_name='offset_01'),
                ParameterMapper(replicate_id='3rd', global_name='offset_01'),
            ]
        )
        for rid in ['2nd', '3rd']:
            _pm = parameter_manager_03.parameter_mapping
            self.assertEqual(_pm.loc[(f'offset_01', rid), 'local_name'], f'offset_01_{rid}')
            self.assertEqual(_pm.loc[(f'offset_01', rid), 'value'], _pm.loc[(f'offset_01', '1st'), 'value'])

    def test_check_joint_uniqueness_local_names_and_values(self):
        parameter_manager = ParameterManager(self.replicate_ids, self.parameters)

        invalid_mappings_01 = [
            ParameterMapper(replicate_id='all', global_name='rate0', local_name='rate0_1', value=1000), 
            ParameterMapper(replicate_id='3rd', global_name='rate0', local_name='rate0_1', value=2000),
        ]
        invalid_mappings_02 = [
            ParameterMapper(replicate_id='1st', global_name='rate0', local_name='rate0_1', value=1000), 
            ParameterMapper(replicate_id='2nd', global_name='rate0', local_name='rate0_1', value=2000),
        ]
        invalid_mappings_03 = [
            ParameterMapper(replicate_id=self.replicate_ids, global_name='rate0', local_name='rate0_1', value=2000),
            ParameterMapper(replicate_id='3rd', global_name='rate0', local_name='rate0_1', value=1000), 
        ]

        # checks that the mappings have unique pairs for local_names and value
        invalid_mappings = [invalid_mappings_01, invalid_mappings_02, invalid_mappings_03]
        for i, _invalid_mappings in enumerate(invalid_mappings):
            with self.assertRaises(ValueError):
                parameter_manager.apply_mappings(_invalid_mappings)

        # checks that the application of a valid mapping does not result in non-unique pairs of local_name and value
        parameter_manager.apply_mappings([ParameterMapper(replicate_id='all', global_name='rate0', local_name='rate0_1', value=1000)])
        with self.assertRaises(ValueError):
            # It should not be possible to set a different value for a local_name in some replicate
            parameter_manager.apply_mappings([ParameterMapper(replicate_id='3rd', global_name='rate0', local_name='rate0_1', value=2000)])
        with self.assertRaises(ValueError):
            parameter_manager.apply_mappings(invalid_mappings_02)
        with self.assertRaises(ValueError):
            parameter_manager.apply_mappings(invalid_mappings_03)


class TestCaretaker(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(1234)
        curr_model = 'model_02'
        self.caretaker_single = TestingHelpers.get_caretaker(curr_model)
        self.mp = 'k'
        self.iv = 'y0'
        sim_single = self.caretaker_single.simulate(t=24)
        self.data_single = TestingHelpers.noisy_samples(sim_single, samples=5)
        self.data_single_wo_errs = TestingHelpers.noisy_samples(sim_single, samples=5, with_errors=False)

        self.replicate_ids = ['1st', '2nd', '3rd']
        self.caretaker_multi = TestingHelpers.get_caretaker(curr_model, replicate_ids=self.replicate_ids)
        self.caretaker_multi_with_mappings = copy.deepcopy(self.caretaker_multi)
        mappings = [
            ParameterMapper(replicate_id=['1st', '3rd'], 
                            global_name=self.mp, 
                            local_name=f'{self.mp}_1', 
                            value=self.caretaker_multi_with_mappings._get_all_parameters()[self.mp]*1.2,
                            ),
            ParameterMapper(replicate_id='3rd', 
                            global_name=self.iv, 
                            local_name=f'{self.iv}_1', 
                            value=self.caretaker_multi_with_mappings._get_all_parameters()[self.iv]*1.1,
                            ),
        ]
        self.caretaker_multi_with_mappings.apply_mappings(mappings)
        sim_multi_with_mappings = self.caretaker_multi_with_mappings.simulate(t=24)
        self.data_multi_with_mappings = TestingHelpers.noisy_samples(sim_multi_with_mappings, samples=5)
        self.data_multi_with_mappings_wo_errs = TestingHelpers.noisy_samples(sim_multi_with_mappings, samples=5, with_errors=False)


    def test_reset(self):
        _caretaker_multi_with_mappings = copy.deepcopy(self.caretaker_multi_with_mappings)
        self.assertFalse(_caretaker_multi_with_mappings.parameter_mapping.equals(self.caretaker_multi.parameter_mapping))
        _caretaker_multi_with_mappings.set_parameters({'k_1' : 100000, 'y0' : -9999999})
        self.assertFalse(_caretaker_multi_with_mappings.parameter_mapping.equals(self.caretaker_multi_with_mappings.parameter_mapping))
        _caretaker_multi_with_mappings.reset()
        self.assertTrue(_caretaker_multi_with_mappings.parameter_mapping.equals(self.caretaker_multi.parameter_mapping))


    def test_init(self):

        curr_model = 'model_02'

        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        observation_functions_parameters = TestingHelpers.get_observation_functions_parameters(curr_model)


        caretaker_single = Caretaker(
            bioprocess_model_class=model_class, 
            model_parameters=model_parameters, 
            initial_values=initial_values, 
            observation_functions_parameters=observation_functions_parameters,
        )
        caretaker_multi = Caretaker(
            bioprocess_model_class=model_class, 
            model_parameters=model_parameters, 
            initial_values=initial_values, 
            observation_functions_parameters=observation_functions_parameters,
            replicate_ids=self.replicate_ids,
        )

        # Cannot create a Caretaker with case-insensitive non-unique replicate_ids
        bad_ids = [
            ['1st', '1St', '2nd', '3rd'], 
            ['1st', '1st', '2nd', '3rd'], 
            ]
        for _bad_ids in bad_ids:
            with self.assertRaises(ValueError):
                Caretaker(
                    bioprocess_model_class=model_class, 
                    model_parameters=model_parameters, 
                    initial_values=initial_values, 
                    observation_functions_parameters=observation_functions_parameters,
                    replicate_ids=_bad_ids,
                )

        # Cannot create a Caretaker with an observation functions observing an unknown model state
        observation_functions_parameters[-1][1][OBSERVED_STATE_KEY] = 'NotKnown'
        with self.assertRaises(ValueError):
            Caretaker(
                bioprocess_model_class=model_class, 
                model_parameters=model_parameters, 
                initial_values=initial_values, 
                observation_functions_parameters=observation_functions_parameters,
                replicate_ids=self.replicate_ids,
            )


    def test_add_replicate(self):
        caretaker = copy.deepcopy(self.caretaker_multi_with_mappings)
        new_id_01 = '4th'
        new_id_02 = '5th'
        mp = 'k'
        iv = 'y0'

        est1, _ = caretaker.estimate(
            unknowns=[mp, iv], 
            bounds=[(0.01, 0.03), (50, 150)], 
            measurements=self.data_multi_with_mappings,
            report_level=2,
            )

        # Store current ParameterMappers
        old_mappers = caretaker._parameter_manager.get_parameter_mappers()
        caretaker.add_replicate(replicate_id=new_id_01)
        # Compare with new mappers
        new_mapping = caretaker.parameter_mapping
        for _old_mapper in old_mappers:
            self.assertEqual(new_mapping.loc[_old_mapper.global_name, _old_mapper.replicate_id]['local_name'], _old_mapper.local_name)
            self.assertEqual(new_mapping.loc[_old_mapper.global_name, _old_mapper.replicate_id]['value'], _old_mapper.value)

        with self.assertRaises(KeyError):
            caretaker.add_replicate(replicate_id=new_id_02, mappings=[ParameterMapper(replicate_id=['1st', new_id_02], global_name=mp, local_name=f'{mp}_2', value=caretaker._get_all_parameters()[mp]*1.2)])
        with self.assertRaises(KeyError):
            caretaker.add_replicate(replicate_id=new_id_02, mappings=[ParameterMapper(replicate_id='1st', global_name=mp, local_name=f'{mp}_2', value=caretaker._get_all_parameters()[mp]*1.2)])
        caretaker.add_replicate(replicate_id=new_id_02, mappings=[ParameterMapper(replicate_id=new_id_02, global_name=mp, local_name=f'{mp}_2', value=caretaker._get_all_parameters()[mp]*1.2)])

        with self.assertRaises(ValueError):
            caretaker.add_replicate(replicate_id=new_id_01)

        with self.assertRaises(AttributeError):
            self.caretaker_single.add_replicate(replicate_id=new_id_01)

        est2, _ = caretaker.estimate(
            unknowns=[mp, iv], 
            bounds=[(0.01, 0.03), (50, 150)], 
            measurements=self.data_multi_with_mappings,
            report_level=2,
            )

        # To this caretaker, a mapping can be applied for the explicitly created single replicate id
        caretaker_single_explicit = TestingHelpers.get_caretaker('model_02', ['explicit_replicate'])
        caretaker_single_explicit.apply_mappings([ParameterMapper('explicit_replicate', self.iv, f'{self.iv}_new', 1000.0)])


    def test_parameter_setting_and_mappings(self):

        caretaker_single_copy = copy.deepcopy(self.caretaker_single)


        caretaker_single_before = copy.deepcopy(self.caretaker_single)
        caretaker_multi_before = copy.deepcopy(self.caretaker_multi)

        mp = 'k'
        iv = 'y0'
        mappings = [
            ParameterMapper(
                replicate_id=['1st', '3rd'], 
                global_name=mp, 
                local_name=f'{mp}_1', 
                value=self.caretaker_multi._get_all_parameters()[mp]*1.2
            ),
            ParameterMapper('3rd', iv, f'{iv}_1', self.caretaker_multi._get_all_parameters()[iv]*0.3)
        ]
        self.caretaker_multi.apply_mappings(mappings)

        # applying each single mapping should have the same effect like apply the list of mappings at once
        caretaker_multi_copy = copy.deepcopy(caretaker_multi_before)
        for mapping in mappings:
            caretaker_multi_copy.apply_mappings(mapping)
        self.assertTrue(pandas.DataFrame.equals(self.caretaker_multi.parameter_mapping, caretaker_multi_copy.parameter_mapping))
        
        # applying parameter mappings to a single replicate caretaker has no effect
        self.caretaker_single.apply_mappings(mappings)
        self.assertTrue(pandas.DataFrame.equals(caretaker_single_before.parameter_mapping, self.caretaker_single.parameter_mapping))

        new_parameters_multi = {f'{mp}_1' : 9999, f'{iv}_1':1234}
        new_parameters_single = {mp : 9999, iv : 1234}
        
        # setting parameter names that are not known has no effect
        self.caretaker_single.set_parameters(new_parameters_multi)
        self.assertTrue(pandas.DataFrame.equals(caretaker_single_before.parameter_mapping, self.caretaker_single.parameter_mapping))
        # setting parameter names that are  known has effect
        self.caretaker_single.set_parameters(new_parameters_single)
        self.assertFalse(pandas.DataFrame.equals(caretaker_single_before.parameter_mapping, self.caretaker_single.parameter_mapping))
        
        self.caretaker_multi.set_parameters(new_parameters_multi)
        self.assertFalse(pandas.DataFrame.equals(caretaker_multi_before.parameter_mapping, self.caretaker_multi.parameter_mapping))
        # save parameter mapping
        _caretaker_multi_between = copy.deepcopy(self.caretaker_multi)
        self.caretaker_multi.set_parameters(new_parameters_single)
        self.assertFalse(pandas.DataFrame.equals(_caretaker_multi_between.parameter_mapping, self.caretaker_multi.parameter_mapping))


    def test_simulate(self):
        mp = 'k'
        iv = 'y0'

        # single replicate caretaker
        simulation_single_01 = self.caretaker_single.simulate(t=24)
        new_parameters_single = {mp : 9999, iv : 1234}
        simulation_single_02 = self.caretaker_single.simulate(t=24, parameters=new_parameters_single)

        # multi replicate caretaker
        # apply a mapping first
        mappings = [
            ParameterMapper(replicate_id=['1st', '3rd'], global_name=mp, local_name=f'{mp}_1', value=self.caretaker_multi._get_all_parameters()[mp]*1.2),
            ParameterMapper('3rd', iv, f'{iv}_1', self.caretaker_multi._get_all_parameters()[iv]*0.3)
        ]
        self.caretaker_multi.apply_mappings(mappings)
        
        simulation_multi_01 = self.caretaker_multi.simulate(t=24)

        _p_before = self.caretaker_multi._get_all_parameters()
        simulation_multi_02 = self.caretaker_multi.simulate(t=24, parameters=new_parameters_single)
        _p_after = self.caretaker_multi._get_all_parameters()
        self.assertDictEqual(_p_before, _p_after)

        new_parameters_multi = {f'{mp}_1' : 9999, f'{iv}_1':1234}
        simulation_multi_03 = self.caretaker_multi.simulate(t=24, parameters=new_parameters_multi)

        self.caretaker_multi.set_integrator_kwargs({'atol' : 1e-8, 'rtol' : 1e-8})
        self.caretaker_multi.simulate(t=24)
        self.caretaker_multi.set_integrator_kwargs(None)


    def test_order_of_bounds(self):
        # check order of bounds for estimation
        bounds = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
        unknowns = [f'{self.iv}', f'{self.iv}_1', f'{self.mp}_1']
        expected = {_u : _b[0] for _u, _b in zip(unknowns, bounds)}
        estimates, _ = self.caretaker_multi_with_mappings.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings)
        self.assertDictEqual(expected, estimates)


    def test_optimizer_kwargs(self):
        unknowns = [f'{self.mp}_1', self.iv, f'{self.iv}_1']
        bounds = [(0, 1), (50, 150), (50, 150)]
        _, _ = self.caretaker_multi_with_mappings.estimate(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
        )
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.optimizer_kwargs = ('strategy' , 'randtobest1exp')

        self.caretaker_multi_with_mappings.optimizer_kwargs = {'strategy' : 'randtobest1exp'}
        _, _ = self.caretaker_multi_with_mappings.estimate(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
        )
        self.caretaker_multi_with_mappings.optimizer_kwargs = None

        self.caretaker_multi_with_mappings.optimizer_kwargs = {'disp' : True}
        _, _ = self.caretaker_multi_with_mappings.estimate(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
        )

        _, _ = self.caretaker_multi_with_mappings.estimate(
            unknowns={_unknown : _guess for _unknown, _guess in zip(unknowns, [0.5, 100, 100])}, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
            use_global_optimizer=False,
        )

        self.caretaker_multi_with_mappings.optimizer_kwargs = {'disp' : False}
        _, _ = self.caretaker_multi_with_mappings.estimate(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
            report_level=4,
        )

        self.caretaker_multi_with_mappings.optimizer_kwargs = {'strategy' : 'randtobest1exp'}
        _, _ = self.caretaker_multi_with_mappings.estimate(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
            optimizer_kwargs={'disp' : True},
        )
        _, _ = self.caretaker_multi_with_mappings.estimate(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
            optimizer_kwargs={'strategy' : 'best1exp'},
        )
        _, _ = self.caretaker_multi_with_mappings.estimate(
           unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
            optimizer_kwargs={'popsize' : 3},
        )


    def test_get_sensitivities_multi_replicate(self):
        parameters = {f'{self.mp}_1' : 0.02, self.iv : 100, f'{self.iv}_1' : 100}
        sensis_01 =  self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24)
        sensis_02 =  self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=parameters)
        
        self.assertTrue(len(set([s_01.name for s_01 in sensis_01])) > len(set([s_02.name for s_02 in sensis_02])))

        self.caretaker_multi_with_mappings.get_sensitivities(measurements=self.data_multi_with_mappings)
        self.caretaker_multi_with_mappings.get_sensitivities(measurements=self.data_multi_with_mappings, tfinal=24)
        self.caretaker_multi_with_mappings.get_sensitivities(measurements=self.data_multi_with_mappings, tfinal=48)

        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.get_sensitivities()
        sensis_03 = self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters={self.iv : 100}, responses=['y'])
        self.assertEqual(len(set([s_03.name for s_03 in sensis_03])), 1)
        self.caretaker_multi_with_mappings.set_parameters({self.iv : 100})
        sensis_04 = self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[str(self.iv)], responses=['y'])

        _mixed = [*self.data_multi_with_mappings, *sensis_04]
        with self.assertRaises(TypeError):
            self.caretaker_multi_with_mappings.get_sensitivities(measurements=_mixed)


        # kwargs testing
        self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[str(self.iv)], responses=['y'], rel_h=1e-2)
        self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[str(self.iv)], responses=['y'], abs_h=1e-17)
        self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[str(self.iv)], responses=['y'], rel_h=1e-17)
        self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[str(self.iv)], responses=['y'], handle_CVodeError=False)
        self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[str(self.iv)], responses=['y'], verbosity_CVodeError=True)
        self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[str(self.iv)], responses=['y'], handle_CVodeError=False, verbosity_CVodeError=True)

        with self.assertRaises(TypeError):
            self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, responses='y')
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, responses=['y', 'y'])
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, responses=['y', 'Y'])
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[str(self.iv), str(self.iv)])
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[str(self.iv), str(self.iv).upper()])
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, parameters=[f'{self.mp}_1', self.iv, f'{self.iv}_1', 'bad_parameter'])
        with self.assertRaises(ValueError):
            # z is an invalid reponse
            self.caretaker_multi_with_mappings.get_sensitivities(tfinal=24, responses=['y', 'z'])


    def test_get_information_matrix_at_t(self):
        timepoints_a = [1, 2, 3]
        timepoints_b = [4, 5, 6]
        timepoints_c = [1, 2, 3, 4, 5, 6]
        additional_timepoints = [10, 11, 12]

        estimates = {'p1' : 10, 'p2' : 100}
        measurements = [
            Measurement(name='a', timepoints=timepoints_a, values=numpy.square(timepoints_a), errors=numpy.sqrt(timepoints_a), replicate_id=Constants.single_id),
            Measurement(name='b', timepoints=timepoints_b, values=numpy.square(timepoints_b), errors=numpy.sqrt(timepoints_b), replicate_id=Constants.single_id),
            Measurement(name='c', timepoints=timepoints_c, values=numpy.square(timepoints_c), errors=numpy.sqrt(timepoints_c), replicate_id=Constants.single_id),
            ]
        measurements_wo_errs = {
            Measurement(name='a', timepoints=timepoints_a, values=numpy.square(timepoints_a), replicate_id=Constants.single_id),
            Measurement(name='b', timepoints=timepoints_b, values=numpy.square(timepoints_b), replicate_id=Constants.single_id),
            Measurement(name='c', timepoints=timepoints_c, values=numpy.square(timepoints_c), replicate_id=Constants.single_id),
            }
        sensitivities = [
            Sensitivity(timepoints=timepoints_a, values=numpy.square(timepoints_a), response='a', parameter='p1', replicate_id=Constants.single_id),
            Sensitivity(timepoints=timepoints_b, values=numpy.square(timepoints_b), response='b', parameter='p1', replicate_id=Constants.single_id),
            Sensitivity(timepoints=timepoints_c, values=numpy.square(timepoints_c), response='c', parameter='p1', replicate_id=Constants.single_id),
            Sensitivity(timepoints=timepoints_a, values=numpy.square(timepoints_a), response='a', parameter='p2', replicate_id=Constants.single_id),
            Sensitivity(timepoints=timepoints_b, values=numpy.square(timepoints_b), response='b', parameter='p2', replicate_id=Constants.single_id),
            Sensitivity(timepoints=timepoints_c, values=numpy.square(timepoints_c), response='c', parameter='p2', replicate_id=Constants.single_id),
            ]

        for _t in numpy.unique(numpy.concatenate([timepoints_a, timepoints_b, timepoints_c, additional_timepoints])):
            FIM_t = self.caretaker_single._get_information_matrix_at_t(t=_t, measurements=measurements, estimates=estimates, sensitivities=sensitivities, replicate_id=Constants.single_id)
            self.assertEqual(FIM_t.shape, (len(estimates), len(estimates)))
            if _t in additional_timepoints:
                # these FIMs should have no information
                for _value in FIM_t.flatten():
                    self.assertEqual(_value, 0)
        
        FIM_a = numpy.full((len(estimates), len(estimates)), fill_value=0.0, dtype=float)
        for _t_a in timepoints_a:
            _FIM_t_a = self.caretaker_single._get_information_matrix_at_t(t=_t_a, measurements=measurements, estimates=estimates, sensitivities=sensitivities, replicate_id=Constants.single_id)
            FIM_a += _FIM_t_a
        
        FIM_b = numpy.full((len(estimates), len(estimates)), fill_value=0.0, dtype=float)
        for _t_b in timepoints_b:
            _FIM_t_b = self.caretaker_single._get_information_matrix_at_t(t=_t_b, measurements=measurements, estimates=estimates, sensitivities=sensitivities, replicate_id=Constants.single_id)
            FIM_b += _FIM_t_b

        FIM_c = numpy.full((len(estimates), len(estimates)), fill_value=0.0, dtype=float)
        for _t_c in timepoints_c:
            _FIM_t_c = self.caretaker_single._get_information_matrix_at_t(t=_t_c, measurements=measurements, estimates=estimates, sensitivities=sensitivities, replicate_id=Constants.single_id)
            FIM_c += _FIM_t_c

        # FIM_a should contain less information than FIM_c
        for _v1, _v2 in zip(FIM_a.flatten(), FIM_c.flatten()):
            self.assertLessEqual(_v1, _v2)

        # FIM_b should contain less information than FIM_c
        for _v1, _v2 in zip(FIM_b.flatten(), FIM_c.flatten()):
            self.assertLessEqual(_v1, _v2)

        # FIM_a and FIM_b should contain the same information as FIM_c
        FIM_ab = FIM_a + FIM_b
        for _v1, _v2 in zip(FIM_ab.flatten(), FIM_c.flatten()):
            self.assertAlmostEqual(_v1, _v2)

        for _t in numpy.unique(numpy.concatenate([timepoints_a, timepoints_b, timepoints_c, additional_timepoints])):
            with self.assertRaises(AttributeError):
                self.caretaker_single._get_information_matrix_at_t(t=_t, measurements=measurements_wo_errs, estimates=estimates, sensitivities=sensitivities, replicate_id=Constants.single_id)

        for _t in numpy.unique(numpy.concatenate([timepoints_a, timepoints_b, timepoints_c, additional_timepoints])):
            self.caretaker_single._get_information_matrix_at_t(t=_t, measurements=measurements, estimates=estimates, sensitivities=sensitivities, replicate_id=Constants.single_id)


    def test_parameter_matrices(self):
        estimates_single = {self.mp : 0.02, self.iv : 100}
        sensitivities_single = self.caretaker_single.get_sensitivities(measurements=self.data_single)
        self.caretaker_single.get_information_matrix(measurements=self.data_single, estimates=estimates_single)
        self.caretaker_single.get_information_matrix(measurements=self.data_single, estimates=estimates_single, sensitivities=sensitivities_single)
        self.caretaker_single.get_parameter_matrices(measurements=self.data_single, estimates=estimates_single)
        self.caretaker_single.get_parameter_matrices(measurements=self.data_single, estimates=estimates_single, sensitivities=sensitivities_single)

        estimates_multi = {f'{self.mp}_1' : 0.02, self.iv : 100, f'{self.iv}_1' : 100}
        sensitivities_multi = self.caretaker_multi_with_mappings.get_sensitivities(measurements=self.data_multi_with_mappings)
        self.caretaker_multi_with_mappings.get_information_matrix(measurements=self.data_multi_with_mappings, estimates=estimates_multi)
        self.caretaker_multi_with_mappings.get_information_matrix(measurements=self.data_multi_with_mappings, estimates=estimates_multi, sensitivities=sensitivities_multi)
        self.caretaker_multi_with_mappings.get_parameter_matrices(measurements=self.data_multi_with_mappings, estimates=estimates_multi)
        self.caretaker_multi_with_mappings.get_parameter_matrices(measurements=self.data_multi_with_mappings, estimates=estimates_multi, sensitivities=sensitivities_multi)

        # This will cause a non-invertible information matrix
        self.caretaker_multi_with_mappings.get_parameter_matrices(measurements=self.data_single, estimates=estimates_multi)

        _mixed = [*self.data_single, *sensitivities_single]
        with self.assertRaises(TypeError):      
            self.caretaker_single.get_information_matrix(measurements=_mixed, estimates=estimates_single, sensitivities=sensitivities_single)
        with self.assertRaises(TypeError):      
            self.caretaker_single.get_information_matrix(measurements=self.data_single, estimates=estimates_single, sensitivities=_mixed)
        with self.assertRaises(TypeError):      
            self.caretaker_single.get_parameter_matrices(measurements=_mixed, estimates=estimates_single, sensitivities=sensitivities_single)
        with self.assertRaises(TypeError):      
            self.caretaker_single.get_parameter_matrices(measurements=self.data_single, estimates=estimates_single, sensitivities=_mixed)


    def test_get_parameter_uncertainties(self):
        estimates_single = {self.mp : 0.02, self.iv : 100}
        sensitivities_single = self.caretaker_single.get_sensitivities(measurements=self.data_single)
        self.caretaker_single.get_parameter_uncertainties(estimates=estimates_single, measurements=self.data_single)
        self.caretaker_single.get_parameter_uncertainties(estimates=estimates_single, measurements=self.data_single, sensitivities=sensitivities_single)
        self.caretaker_single.get_parameter_uncertainties(estimates=estimates_single, measurements=self.data_single, sensitivities=sensitivities_single, report_level=1)
        self.caretaker_single.get_parameter_uncertainties(estimates=estimates_single, measurements=self.data_single, sensitivities=sensitivities_single, handle_CVodeError=False)
        self.caretaker_single.get_parameter_uncertainties(estimates=estimates_single, measurements=self.data_single, sensitivities=sensitivities_single, verbosity_CVodeError=False)
        self.caretaker_single.get_parameter_uncertainties(estimates=estimates_single, measurements=self.data_single, sensitivities=sensitivities_single, handle_CVodeError=False, verbosity_CVodeError=False)

        estimates_multi = {f'{self.mp}_1' : 0.02, self.iv : 100, f'{self.iv}_1' : 100}
        sensitivities_multi = self.caretaker_multi_with_mappings.get_sensitivities(measurements=self.data_multi_with_mappings)
        self.caretaker_multi_with_mappings.get_parameter_uncertainties(estimates=estimates_multi, measurements=self.data_multi_with_mappings)
        self.caretaker_multi_with_mappings.get_parameter_uncertainties(estimates=estimates_multi, measurements=self.data_multi_with_mappings, sensitivities=sensitivities_multi)
        self.caretaker_multi_with_mappings.get_parameter_uncertainties(estimates=estimates_multi, measurements=self.data_multi_with_mappings, sensitivities=sensitivities_multi, report_level=1)

        _mixed = [*self.data_single, *sensitivities_single]
        with self.assertRaises(TypeError):
            self.caretaker_single.get_parameter_uncertainties(estimates=estimates_single, measurements=_mixed, sensitivities=sensitivities_single)
        with self.assertRaises(TypeError):
            self.caretaker_single.get_parameter_uncertainties(estimates=estimates_single, measurements=self.data_single, sensitivities=_mixed)


    def test_get_optimality_criteria(self):
        estimates_single = {self.mp : 0.02, self.iv : 100}
        sensitivities_single =  self.caretaker_single.get_sensitivities(measurements=self.data_single)
        parameter_matrices_single = self.caretaker_single.get_parameter_matrices(measurements=self.data_single, estimates=estimates_single)
        self.caretaker_single.get_optimality_criteria(parameter_matrices_single['Cov'])
        self.caretaker_single.get_optimality_criteria(parameter_matrices_single['Cov'], report_level=1)

        estimates_multi = {f'{self.mp}_1' : 0.02, self.iv : 100, f'{self.iv}_1' : 100}
        sensitivities_multi =  self.caretaker_multi_with_mappings.get_sensitivities(measurements=self.data_multi_with_mappings)
        parameter_matrices_multi = self.caretaker_multi_with_mappings.get_parameter_matrices(measurements=self.data_multi_with_mappings, estimates=estimates_multi)
        self.caretaker_multi_with_mappings.get_optimality_criteria(parameter_matrices_multi['Cov'])
        self.caretaker_multi_with_mappings.get_optimality_criteria(parameter_matrices_multi['Cov'], report_level=1)


    def test_draw_measurement_samples(self):

        new_measurements = self.caretaker_multi_with_mappings._draw_measurement_samples(measurements=self.data_multi_with_mappings)
        for _new_measurement, _data in zip(new_measurements, self.data_multi_with_mappings):
            for _v1, _v2 in zip(_data.values, _new_measurement.values):
                self.assertNotAlmostEqual(_v1, _v2)


class TestCaretakerEstimateMethods(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(1234)
        curr_model = 'model_02'
        self.caretaker_single = TestingHelpers.get_caretaker(curr_model)
        self.mp = 'k'
        self.iv = 'y0'
        self.sim_single = self.caretaker_single.simulate(t=24)
        self.data_single = TestingHelpers.noisy_samples(self.sim_single, samples=5)
        self.data_single_wo_errs = TestingHelpers.noisy_samples(self.sim_single, samples=5, with_errors=False)

        self.caretaker_multi = TestingHelpers.get_caretaker(curr_model, replicate_ids=['1st', '2nd', '3rd'])
        self.caretaker_multi_with_mappings = copy.deepcopy(self.caretaker_multi)
        mappings = [
            ParameterMapper(replicate_id=['1st', '3rd'], 
                            global_name=self.mp, 
                            local_name=f'{self.mp}_1', 
                            value=self.caretaker_multi_with_mappings._get_all_parameters()[self.mp]*1.2
                            ),
            ParameterMapper('3rd', 
                            self.iv, 
                            f'{self.iv}_1', 
                            self.caretaker_multi_with_mappings._get_all_parameters()[self.iv]*1.1
                            )
        ]
        self.caretaker_multi_with_mappings.apply_mappings(mappings)
        self.sim_multi_with_mappings = self.caretaker_multi_with_mappings.simulate(t=24)
        self.data_multi_with_mappings = TestingHelpers.noisy_samples(self.sim_multi_with_mappings, samples=5)
        self.data_multi_with_mappings_wo_errs = TestingHelpers.noisy_samples(self.sim_multi_with_mappings, samples=5, with_errors=False)

        # Use an contraint OwnLossCalculator
        class ConstrainedLossCalculator(LossCalculator):
            def constraint(self):
                # We don't care if this constraint makes sense
                return numpy.sum(list(self.current_parameters.values())) > 100
            def check_constraints(self):
                return [self.constraint()]
        self.constrained_loss_calculator = ConstrainedLossCalculator


    def test_estimate_parallel_MC_sampling(self):

        unknowns = [self.mp, self.iv]
        bounds = [(0, 1), (90, 120)]

        # Lengths of bounds and unknowns must match
        with self.assertRaises(ValueError):
            self.caretaker_single.estimate_parallel_MC_sampling(
                unknowns=unknowns, 
                bounds=[(0, 1)]*(len(unknowns)-1), 
                measurements=self.data_single,
                mc_samples=2,
                evolutions=2,
                optimizers='compass_search',
            )

        # kwarg testing report_level
        for _report_level in range(0, 7):
            self.caretaker_single.estimate_parallel_MC_sampling(
                unknowns=unknowns, 
                bounds=bounds, 
                measurements=self.data_single,
                report_level=_report_level,
                mc_samples=2,
                evolutions=2,
                optimizers='compass_search',
            )

        # kwarg testing metric
        for _metric in ['negLL', 'WSS', 'SS']:
            self.caretaker_single.estimate_parallel_MC_sampling(
                unknowns=unknowns, 
                bounds=bounds, 
                measurements=self.data_single,
                metric=_metric,
                mc_samples=2,
                evolutions=2,
                optimizers='compass_search',
            )

        # test number of parallelization per archipelago for all archipelago in a batch
        self.caretaker_single.estimate_parallel_MC_sampling(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            rtol_islands=0.01,
            mc_samples=2,
            n_islands=2,
            optimizers='compass_search',
        )
        self.caretaker_single.estimate_parallel_MC_sampling(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            rtol_islands=0.01,
            mc_samples=2,
            n_islands=3,
            optimizers='compass_search',
        )
        self.caretaker_single.estimate_parallel_MC_sampling(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            rtol_islands=0.01,
            mc_samples=2,
            n_islands=4,
            optimizers='compass_search',
        )

        # Explicitly define not enough parallel islands per archipelago
        with self.assertRaises(ValueError):
            self.caretaker_single.estimate_parallel_MC_sampling(
                unknowns=unknowns, 
                bounds=bounds, 
                measurements=self.data_single,
                n_islands=1,
            )

        # Implicitly define not enough parallel islands per archipelago
        with self.assertRaises(ValueError):
            self.caretaker_single.estimate_parallel_MC_sampling(
                unknowns=unknowns, 
                bounds=bounds, 
                measurements=self.data_single,
                optimizers=['compass_search'],
            )

        # Convergence should never be reached, meaning that there are no results
        est_mc = self.caretaker_single.estimate_parallel_MC_sampling(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            optimizers='compass_search',
            mc_samples=2,
            rtol_islands=None,
        )
        self.assertTrue(est_mc.empty)

        # invalid unknowns
        with self.assertRaises(ValueError):
            self.caretaker_single.estimate_parallel_MC_sampling(
                unknowns=['invalid_unknown'], 
                bounds=[(0, 1)], 
                measurements=self.data_single,
            )

        # Non-unique unknowns
        with self.assertRaises(KeyError):
            self.caretaker_single.estimate_parallel_MC_sampling(
                unknowns=unknowns*2, 
                bounds=bounds*2, 
                measurements=self.data_single,
            )

        # not only Measurement objects as data
        with self.assertRaises(TypeError):
            self.caretaker_single.estimate_parallel_MC_sampling(
                unknowns=unknowns, 
                bounds=bounds, 
                measurements=[*self.data_single, *self.sim_single],
            )

        # Measurements must have errors
        with self.assertRaises(AttributeError):
            self.caretaker_single.estimate_parallel_MC_sampling(
                unknowns=unknowns, 
                bounds=bounds, 
                measurements=self.data_single_wo_errs,
            )

        self.caretaker_single.estimate_parallel_MC_sampling(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            report_level=_report_level,
            evolutions=5,
            mc_samples=5,
            optimizers='compass_search',
            loss_calculator = self.constrained_loss_calculator,
        )


    def test_estimate_single_replicate_caretaker(self):

        unknowns = [self.mp, self.iv]
        bounds = [(0, 1), (90, 120)]

        # testing kwargs
        self.caretaker_single.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_single, use_global_optimizer=True)
        self.caretaker_single.estimate(unknowns={p : 1 for p in unknowns}, bounds=bounds, measurements=self.data_single, use_global_optimizer=True)
        self.caretaker_single.estimate(unknowns={p : 1 for p in unknowns}, measurements=self.data_single)
        self.caretaker_single.estimate(unknowns={p : 1 for p in unknowns}, measurements=self.data_single, use_global_optimizer=False)
        self.caretaker_single.estimate(unknowns={p : 1 for p in unknowns}, measurements=self.data_single, use_global_optimizer=False, report_level=1)
        self.caretaker_single.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_single, use_global_optimizer=True, report_level=2)

        with self.assertRaises(ValueError):
            self.caretaker_single.estimate(unknowns=unknowns, measurements=self.data_single)
        with self.assertRaises(ValueError):
            self.caretaker_single.estimate(unknowns=unknowns, measurements=self.data_single, use_global_optimizer=True)
        with self.assertRaises(ValueError):
            self.caretaker_single.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_single, use_global_optimizer=False)

        _mixed = [*self.data_single, *self.sim_single]
        with self.assertRaises(TypeError):
            self.caretaker_single.estimate(unknowns=unknowns, measurements=_mixed)

        # testing invalid unknowns
        invalid_unknowns_01 = ['y0', 'y0']
        invalid_unknowns_02 = ['y0', 'Y0']
        invalid_unknowns_03 = {'y0' : 1, 'Y0' : 1}
        for _invalid_unknowns in [invalid_unknowns_01, invalid_unknowns_02, invalid_unknowns_03]:
            with self.assertRaises(KeyError):
                self.caretaker_single.estimate(unknowns=_invalid_unknowns, bounds=bounds, measurements=self.data_single)

        invalid_unknowns_04 = ['y0', 'bad_key']
        with self.assertRaises(ValueError):
            self.caretaker_single.estimate(unknowns=invalid_unknowns_04, bounds=bounds, measurements=self.data_single)


    def test_estimate_multi_replicate_caretaker(self):

        unknowns = [f'{self.mp}_1', self.iv, f'{self.iv}_1']
        bounds = [(0, 1), (90, 120), (90, 120)]

        invalid_unknowns = ['y0', 'bad_key']
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.estimate(unknowns=invalid_unknowns, bounds=bounds, measurements=self.data_multi_with_mappings)

        # kwargs testing
        self.caretaker_multi_with_mappings.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, handle_CVodeError=False)
        self.caretaker_multi_with_mappings.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, handle_CVodeError=True)
        for _metric in Constants.pretty_metrics.keys():
            self.caretaker_multi_with_mappings.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, metric=_metric)

        # These should fail due to invalid initial guess values
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.estimate(unknowns={'y0' : numpy.nan}, bounds=[(90, 120)], measurements=self.data_multi_with_mappings, handle_CVodeError=False, use_global_optimizer=False)
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.estimate(unknowns={'y0' : numpy.nan}, bounds=[(90, 120)], measurements=self.data_multi_with_mappings, handle_CVodeError=False)
        # This should not fail because the global optimizer ignore the initial guess values
        self.caretaker_multi_with_mappings.estimate(unknowns={'y0' : numpy.nan}, bounds=[(90, 120)], measurements=self.data_multi_with_mappings, handle_CVodeError=False, use_global_optimizer=True)

        # test data with measurements keys that are not known to the model. 

        data_multi_plus_irrelevant = []
        data_multi_plus_irrelevant.extend(self.data_multi_with_mappings)
        for _replicate_id in self.caretaker_multi_with_mappings._replicate_ids:
            data_multi_plus_irrelevant.append(
                Measurement(name='irrelevant', timepoints=[0, 1], values=[10, 20], errors=[100, 200], replicate_id=_replicate_id)
            )

        est_01, _ = self.caretaker_multi_with_mappings.estimate(unknowns={_u : _g for _u, _g in zip(unknowns, [0.02, 100, 100])}, bounds=bounds, measurements=self.data_multi_with_mappings, use_global_optimizer=False)
        est_02, _ = self.caretaker_multi_with_mappings.estimate(unknowns={_u : _g for _u, _g in zip(unknowns, [0.02, 100, 100])}, bounds=bounds, measurements=data_multi_plus_irrelevant, use_global_optimizer=False)
        self.assertDictEqual(est_01, est_02)

        self.caretaker_multi_with_mappings.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, report_level=1)
        self.caretaker_multi_with_mappings.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, report_level=2)
        self.caretaker_multi_with_mappings.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, report_level=3)
        self.caretaker_multi_with_mappings.estimate(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, report_level=4)


    def test_estimate_with_CVodeError_handling(self):
        numpy.random.seed(1234)
        curr_model = 'model_06'
        mp = 'rate0'
        iv = 'y10'
        unknowns = [iv]
        toxic_parameters = {mp : 0}

        # single model entity caretaker
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        observation_functions_parameters = TestingHelpers.get_observation_functions_parameters(curr_model)
        caretaker_single = Caretaker(
            bioprocess_model_class=model_class, 
            model_parameters=model_parameters, 
            initial_values=initial_values, 
            observation_functions_parameters=observation_functions_parameters,
        )
        sim_single = caretaker_single.simulate(t=24)
        data_single = TestingHelpers.noisy_samples(sim_single, samples=5)

        caretaker_single.set_parameters(toxic_parameters)
        caretaker_single.optimizer_kwargs = {'maxiter' : 1}
        with self.assertRaises(ValueError):
            # when handling CVodeError, the DE optimizer will not find an solution, because the obj fun returns always Nan
            # afterwards, the DE optimizer runs a local optimization, which in this case uses NaN as guess values
            caretaker_single.estimate(unknowns=unknowns, measurements=data_single, bounds=[(50, 150)])
        caretaker_single.optimizer_kwargs = {'polish' : False, 'maxiter' : 1}
        _est, _est_info = caretaker_single.estimate(unknowns=unknowns, measurements=data_single, bounds=[(50, 150)])
        _, _ = caretaker_single.estimate(unknowns=unknowns, measurements=data_single, bounds=[(50, 150)])
        caretaker_single.optimizer_kwargs = None
        self.assertEqual(_est_info['loss'], numpy.inf)
        with self.assertRaises(CVodeError):
            caretaker_single.estimate(unknowns=unknowns, measurements=data_single, bounds=[(50, 150)], handle_CVodeError=False)

        # multi model entity caretaker
        replicate_ids = ['1st', '2nd', '3rd']
        caretaker_multi = Caretaker(
            bioprocess_model_class=model_class, 
            model_parameters=model_parameters, 
            initial_values=initial_values, 
            observation_functions_parameters=observation_functions_parameters,
            replicate_ids=replicate_ids,
        )
        mappings = [
            ParameterMapper(replicate_id=['1st', '3rd'], 
                            global_name=mp, 
                            local_name=f'{mp}_1', 
                            value=model_parameters[mp]*1.2
                            ),
            ParameterMapper('3rd', 
                            iv, 
                            f'{iv}_1', 
                            initial_values[iv]*1.1
                            )
        ]
        caretaker_multi.apply_mappings(mappings)
        sim_multi = caretaker_multi.simulate(t=24)
        data_multi = TestingHelpers.noisy_samples(sim_multi, samples=5)

        caretaker_multi.set_parameters(toxic_parameters)
        caretaker_multi.optimizer_kwargs = {'maxiter' : 5}
        with self.assertRaises(ValueError):
            # when handling CVodeError, the DE optimizer will not find an solution, because the obj fun returns always Nan
            # afterwards, the DE optimizer runs a local optimization, which in this case uses NaN as guess values
            caretaker_multi.estimate(unknowns=unknowns, measurements=data_multi, bounds=[(50, 150)])
        caretaker_multi.optimizer_kwargs = {'polish' : False, 'maxiter' : 5}
        _est, _est_info = caretaker_multi.estimate(unknowns=unknowns, measurements=data_multi, bounds=[(50, 150)])
        caretaker_multi.optimizer_kwargs = None
        self.assertEqual(_est_info['loss'], numpy.inf)
        with self.assertRaises(CVodeError):
            caretaker_multi.estimate(unknowns=unknowns, measurements=data_multi, bounds=[(50, 150)], handle_CVodeError=False)


    def test_estimate_repeatedly_single_replicate_caretaker(self):
        unknowns = [self.mp, self.iv]
        bounds = [(0, 1), (90, 120)]

        # kwargs testing
        for _metric in Constants.pretty_metrics.keys():
            self.caretaker_single.estimate_repeatedly(
                unknowns=unknowns, 
                bounds=bounds, 
                metric=_metric,
                measurements=self.data_single,
                rel_jobs=0.1,
            )


    def test_estimate_repeatedly_multi_replicate_caretaker(self):

        unknowns = [f'{self.mp}_1', self.iv, f'{self.iv}_1']
        bounds = [(0, 1), (90, 120), (90, 120)]

        # kwargs testing
        for _metric in Constants.pretty_metrics.keys():
            self.caretaker_multi_with_mappings.estimate_repeatedly(
                unknowns=unknowns, 
                bounds=bounds, 
                metric=_metric,
                measurements=self.data_multi_with_mappings,
                rel_jobs=0.1,
            )

        _, _ = self.caretaker_multi_with_mappings.estimate_repeatedly(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
            report_level=1,
            jobs=2,
        )

        _, _ = self.caretaker_multi_with_mappings.estimate_repeatedly(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
            report_level=2,
            jobs=2,
        )

        self.caretaker_multi_with_mappings.optimizer_kwargs = {'disp' : True}
        _, _ = self.caretaker_multi_with_mappings.estimate_repeatedly(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_multi_with_mappings,
            jobs=2,
        )

        _mixed = [*self.data_multi_with_mappings, *self.sim_multi_with_mappings]
        with self.assertRaises(TypeError):
            self.caretaker_multi_with_mappings.estimate_repeatedly(unknowns=unknowns, bounds=bounds, measurements=_mixed, jobs=2)


    def test_estimate_MC_sampling_multi_replicate(self):
        unknowns = [f'{self.mp}_1', self.iv, f'{self.iv}_1']
        unknowns_dict = {f'{self.mp}_1' : 0.02, self.iv : 100, f'{self.iv}_1' : 100}
        bounds = [(0, 1), (90, 120), (90, 120)]
        _, _ = self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, mc_samples=2)
        _, _ = self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, mc_samples=2, report_level=1)
        _, _ = self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, mc_samples=2, report_level=2)
        _, _ = self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns_dict, bounds=bounds, measurements=self.data_multi_with_mappings, mc_samples=2, use_global_optimizer=False)
        _, _ = self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns_dict, bounds=bounds, measurements=self.data_multi_with_mappings, mc_samples=2, rel_mc_samples=0.1, use_global_optimizer=False)
        _, _ = self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns_dict, bounds=bounds, measurements=self.data_multi_with_mappings, mc_samples=2, rel_mc_samples=0.1, use_global_optimizer=False, reuse_errors_as_weights=False)
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, mc_samples=2, use_global_optimizer=False)
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns, measurements=self.data_multi_with_mappings, mc_samples=5)
        with self.assertRaises(ValueError):
            self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns_dict, measurements=self.data_multi_with_mappings, mc_samples=2, use_global_optimizer=True)
        with self.assertRaises(AttributeError):
            self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns_dict, bounds=bounds, measurements=self.data_multi_with_mappings_wo_errs, mc_samples=5)

        self.caretaker_multi_with_mappings.optimizer_kwargs = {'disp' : True}
        _, _ = self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns, bounds=bounds, measurements=self.data_multi_with_mappings, mc_samples=2)

        _mixed = [*self.data_multi_with_mappings, *self.sim_multi_with_mappings]
        with self.assertRaises(TypeError):
            self.caretaker_multi_with_mappings.estimate_MC_sampling(unknowns=unknowns, bounds=bounds, measurements=_mixed, mc_samples=2)


    def test_estimate_parallel(self):

        unknowns = [self.mp, self.iv]
        bounds = [(0, 1), (90, 120)]

        # Lengths of bounds and unknowns must match
        with self.assertRaises(ValueError):
            self.caretaker_single.estimate_parallel(
                unknowns=unknowns, 
                bounds=[(0, 1)]*(len(unknowns)+1), 
                measurements=self.data_single,
            )

        with self.assertRaises(KeyError):
            self.caretaker_single.estimate_parallel(
                unknowns=[unknowns[0] for _ in unknowns], 
                bounds=bounds, 
                measurements=self.data_single,
            )

        with self.assertRaises(ValueError):
            self.caretaker_single.estimate_parallel(
                unknowns=['bad_unknown'], 
                bounds=bounds, 
                measurements=self.data_single,
            )

        est_01, est_info_01 = self.caretaker_single.estimate_parallel(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            evolutions=1,
        )

        est_02, est_info_02 = self.caretaker_single.estimate_parallel(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            evolutions=1,
        )

        est_03, est_info03 = self.caretaker_single.estimate_parallel_continued(
            estimation_result=est_info_01,
        )

        for _report_level in [1, 2, 3, 4, 5]:
            _, _ = self.caretaker_multi_with_mappings.estimate_parallel(
                unknowns=unknowns, 
                bounds=bounds, 
                measurements=self.data_single,
                evolutions=1,
                report_level=_report_level,
            )

            _, _ = self.caretaker_multi_with_mappings.estimate_parallel_continued(
                estimation_result=est_info_02,
                report_level=_report_level,
            )

        _, _ = self.caretaker_single.estimate_parallel(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            optimizers=['compass_search'],
            evolutions=121,
        )

        # testing stopping criteria
        _, _ = self.caretaker_single.estimate_parallel(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            optimizers=['compass_search', 'compass_search'],
            evolutions=5,
            atol_islands=1000,
        )

        _, _ = self.caretaker_single.estimate_parallel(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            optimizers=['compass_search', 'compass_search'],
            evolutions=5,
            rtol_islands=1000,
        )

        _, _ = self.caretaker_single.estimate_parallel(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            optimizers=['compass_search', 'compass_search'],
            evolutions=5,
            max_runtime_min=0.01,
        )

        _, _ = self.caretaker_single.estimate_parallel(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            optimizers=['compass_search', 'compass_search'],
            evolutions=5,
            max_evotime_min=0.01,
        )

        # Use a contrained LossCalculator
        est_constr, est_constr_info = self.caretaker_single.estimate_parallel(
            unknowns=unknowns, 
            bounds=bounds, 
            measurements=self.data_single,
            evolutions=1,
            loss_calculator=self.constrained_loss_calculator,
        )

        est_constr, est_constr_info = self.caretaker_single.estimate_parallel_continued(
            estimation_result=est_constr_info,
        )


class TestDatatypes(unittest.TestCase):

    def constant_error_model(self, values, parameters):
            offset = parameters['offset']
            return numpy.ones_like(values)*offset

    def linear_error_model(self, values, parameters):
            offset = parameters['offset']
            slope = parameters['slope']
            return values * slope + offset

    def squared_error_model(self, values, parameters):
            w0 = parameters['w0']
            w1 = parameters['w1']
            w2 = parameters['w2']
            return w0 + values*w1 + numpy.square(values)*w2

    def setUp(self):
        self.timepoints = [1, 2, 3, 4, 5]
        self.values = [100, 200, 300, 400, 500]
        self.errors = [1/1, 1/2, 1/3, 1/4, 1/5]
        self.state = 'y1'
        self.info = 'Info'
        self.name = 'Name'
        self.replicate_id = 'ID'
        self.parameter = 'p1'
        self.metrics = ['SS', 'WSS', 'negLL']

        self.linear_error_model_parameters = {
            'offset' : 0,
            'slope' : 1,
        }

        self.squared_error_model_parameters = {
            'w0' : 1,
            'w1' : 0.1,
            'w2' : 0.02,
        }


    def test_inits(self):
        with self.assertRaises(ValueError):
            ModelState(name=self.name, timepoints=self.timepoints, values=None)
        with self.assertRaises(ValueError):
            ModelState(name=self.name, timepoints=None, values=self.values)

        ModelState(name=self.name, timepoints=self.timepoints, values=self.values)
        ModelState(name=self.name, timepoints=self.timepoints, values=self.values, info=self.info)
        ModelState(name=self.name, timepoints=numpy.array(self.timepoints), values=self.values)
        ModelState(name=self.name, timepoints=self.timepoints, values=numpy.array(self.values))
        ModelState(name=self.name, timepoints=numpy.array(self.timepoints), values=numpy.array(self.values))

        Observation(name=self.name, timepoints=self.timepoints, values=self.values, observed_state=self.state)

        Sensitivity(timepoints=self.timepoints, values=self.values, response=self.state, parameter=self.parameter)
        Sensitivity(timepoints=self.timepoints, values=self.values, response=self.state, parameter=self.parameter, h=1e-8)

        TimeSeries(name=self.name, timepoints=self.timepoints, values=self.values)
        
        Measurement(name=self.name, timepoints=self.timepoints, values=self.values)
        Measurement(name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors)

        # Errors must be > 0
        with self.assertRaises(ValueError):
            Measurement(name=self.name, timepoints=self.timepoints, values=self.values, errors=numpy.zeros_like(self.errors))

        with self.assertRaises(ValueError):
            Measurement(name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors, error_distribution=scipy.stats.bernoulli)

        # Can set new vectors with mathcing length, but the getter will mask the position of non-numeric values in all vectors
        values_to_clean_01 = [10, 20, None, 40, 50]
        values_to_clean_02 = [10, 20, numpy.nan, 40, 50]
        values_to_clean_03 = [10, 20, numpy.inf, 40, 50]
        values_to_clean_04 = [10, 20, -numpy.inf, 40, 50]
        values_to_clean = [values_to_clean_01, values_to_clean_02, values_to_clean_03, values_to_clean_04]

        for _values_to_clean in values_to_clean:
            _measurement = Measurement(name=self.name, timepoints=self.timepoints, values=_values_to_clean, errors=self.errors)
            for _v1, _v2 in zip(_measurement.timepoints, numpy.array([1, 2, 4, 5])):
                self.assertEqual(_v1, _v2)
            for _v1, _v2 in zip(_measurement.values, numpy.array([10, 20, 40, 50])):
                self.assertEqual(_v1, _v2)
            for _v1, _v2 in zip(_measurement.errors, numpy.array([1/1, 1/2, 1/4, 1/5])):
                self.assertEqual(_v1, _v2)


    def test_str(self):
        _pattern_1 = f' | Name: {self.name} | Replicate ID: {self.replicate_id}'
        _pattern_2 = f' | Name: {self.name} | Replicate ID: {self.replicate_id} | Info: {self.info}'
        
        time_series_1 = TimeSeries(name=self.name, timepoints=self.timepoints, values=self.values, replicate_id=self.replicate_id)
        self.assertEqual(f'TimeSeries{_pattern_1}', str(time_series_1))
        time_series_2 = TimeSeries(name=self.name, timepoints=self.timepoints, values=self.values, replicate_id=self.replicate_id, info=self.info)
        self.assertEqual(f'TimeSeries{_pattern_2}', str(time_series_2))

        model_state_1 = ModelState(name=self.name, timepoints=self.timepoints, values=self.values, replicate_id=self.replicate_id)
        self.assertEqual(f'ModelState{_pattern_1}', str(model_state_1))
        model_state_2 = ModelState(name=self.name, timepoints=self.timepoints, values=self.values, replicate_id=self.replicate_id, info=self.info)
        self.assertEqual(f'ModelState{_pattern_2}', str(model_state_2))

        measurement_1 = Measurement(name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors, replicate_id=self.replicate_id)
        self.assertEqual(f'Measurement{_pattern_1}', str(measurement_1))
        measurement_2 = Measurement(name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors, replicate_id=self.replicate_id, info=self.info)
        self.assertEqual(f'Measurement{_pattern_2}', str(measurement_2))

        observation = Observation(name=self.name, timepoints=self.timepoints, values=self.values, observed_state=self.state, replicate_id=self.replicate_id)
        self.assertEqual(f'Observation{_pattern_1} | Info: {self.name}, observes {self.state}', str(observation))

        sensitivity = Sensitivity(timepoints=self.timepoints, values=self.values, replicate_id=self.replicate_id, response=self.state, parameter=self.parameter, h=1e-8, info=self.info)
        self.assertEqual(f'Sensitivity | Name: d({self.state})/d({self.parameter}) | Replicate ID: {self.replicate_id} | Info: {self.info}', str(sensitivity))


    def test_measurement_methods(self):

        measurement_wo_errs = Measurement(name=self.state, timepoints=self.timepoints[1::2], values=self.values[1::2])
        measurement = Measurement(name=self.state, timepoints=self.timepoints[1::2], values=self.values[1::2], errors=self.errors[1::2])

        predictions_bad = [ModelState(name=f'{self.state}_{i}', timepoints=self.timepoints, values=self.values) for i in range(3)]
        predictions = copy.deepcopy(predictions_bad)
        predictions.append(ModelState(name=self.state, timepoints=self.timepoints, values=self.values))

        for metric in self.metrics:
            measurement.get_loss(metric=metric, predictions=predictions)

        for metric in self.metrics:
            nan_loss_1 = measurement.get_loss(metric=metric, predictions=predictions_bad)
            self.assertTrue(numpy.isnan(nan_loss_1))

        predictions_fewer_timepoints = [ModelState(name=self.state, timepoints=self.timepoints[:-2], values=self.values[:-2])]
        for metric in self.metrics:
            loss = measurement.get_loss(metric=metric, predictions=predictions_fewer_timepoints)
            self.assertFalse(numpy.isnan(loss))

        predictions_for_nan_losses = [ModelState(name=self.state, timepoints=self.timepoints, values=numpy.zeros_like(self.values)*numpy.nan)]
        for metric in self.metrics:
            nan_loss_2 = measurement.get_loss(metric=metric, predictions=predictions_for_nan_losses)
            self.assertTrue(numpy.isnan(nan_loss_2))

        measurement_wo_errs.get_loss(metric='SS', predictions=predictions)
        with self.assertRaises(AttributeError):
            measurement_wo_errs.get_loss(metric='WSS', predictions=predictions)
        with self.assertRaises(AttributeError):
            measurement_wo_errs.get_loss(metric='negLL', predictions=predictions)
        with self.assertRaises(NotImplementedError):
            measurement_wo_errs.get_loss(metric='bad_metric', predictions=predictions)

        with self.assertRaises(ValueError):
            non_unique_identifable_predictions = predictions*2
            measurement.get_loss(metric='negLL', predictions=non_unique_identifable_predictions)

        _random_samples = measurement._get_random_samples_values()
        for _v1, _v2 in zip(_random_samples, measurement.values):
            self.assertNotAlmostEqual(_v1, _v2)
        with self.assertRaises(AttributeError):
            measurement_wo_errs._get_random_samples_values()

        measurement_t_distri = Measurement(name=self.state, timepoints=self.timepoints[1::2], values=self.values[1::2], errors=self.errors[1::2], error_distribution=scipy.stats.t)
        with self.assertRaises(Exception):
            measurement_t_distri.get_loss(metric='negLL', predictions=predictions)
        measurement_t_distri.get_loss(metric='negLL', predictions=predictions, distribution_kwargs={'df' : 4})


    def test_setters(self):
        values_01 = numpy.array([10, 20, 30, 40, 50])
        values_02 = numpy.array([[10], [20], [30], [40], [50]])
        
        bad_values_01 = numpy.array([10, 20, 30, 40])
        bad_values_02 = numpy.array([[10, 20, 30, 40, 50], [10, 20, 30, 40, 50]])
        bad_values_03 = numpy.array([[], [10, 20, 30, 40, 50]])
        bad_values_04 = numpy.array([[10, 20, 30, 40, 50], []])
        bad_values_05 = numpy.array([[10, 20, 30, 40, 50]])

        model_state = ModelState(name=self.name, timepoints=self.timepoints, values=self.values)
        model_state.values = values_01
        model_state.values = values_02
        model_state.timepoints = numpy.array(self.timepoints)

        with self.assertRaises(ValueError):
            model_state.values = bad_values_01
        with self.assertRaises(ValueError):
            model_state.values = bad_values_02
        with self.assertRaises(ValueError):
            model_state.values = bad_values_03
        with self.assertRaises(ValueError):
            model_state.values = bad_values_04
        with self.assertRaises(ValueError):
            model_state.values = bad_values_05

        measurement = Measurement(name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors)
        with self.assertRaises(ValueError):
            measurement.errors = bad_values_01
        with self.assertRaises(ValueError):
            measurement.errors = bad_values_02
        with self.assertRaises(ValueError):
            measurement.errors = bad_values_03


    def test_setters_omitting_non_numeric_values(self):

        vector_01 = [0, 1, 2, numpy.nan]
        vector_02 = [0, 1, None, 3]
        vector_03 = ['some_string', 1, 2, 3]
        vector_04 = [0, numpy.inf, 2, numpy.nan]
        vector_05 = [numpy.nan, 1, None, 3]
        vector_06 = ['some_string', None, 2, numpy.nan]

        timepoints = numpy.array([1, 2, 3, 4])
        values = numpy.square(timepoints)
        errors = numpy.sqrt(timepoints)

        time_series_ref = TimeSeries(name='time_series_ref', timepoints=timepoints, values=values)
        measurement_ref = Measurement(name='measurement_ref', timepoints=timepoints, values=values, errors=errors)

        vectors = [vector_01, vector_02, vector_03, vector_04, vector_05, vector_06]
        for vector in vectors:
            time_series = TimeSeries(name='time_series', timepoints=timepoints, values=vector)
            self.assertLess(numpy.sum(~numpy.isnan(time_series.timepoints)), len(time_series_ref.timepoints))
            self.assertLess(numpy.sum(~numpy.isnan(time_series.values)), len(time_series_ref.values))

        vectors_with_zeros = [vector_01, vector_02, vector_04]
        for vector in vectors_with_zeros:
            with self.assertRaises(ValueError):
                Measurement(name='measurement', timepoints=timepoints, values=values, errors=vector)

        vectors_without_zeros = [vector_03, vector_05, vector_06]
        for vector in vectors_without_zeros:
            measurement = Measurement(name='measurement', timepoints=timepoints, values=values, errors=vector)
            self.assertLess(numpy.sum(~numpy.isnan(measurement.timepoints)), len(measurement_ref.timepoints))
            self.assertLess(numpy.sum(~numpy.isnan(measurement.values)), len(measurement_ref.values))
            self.assertLess(numpy.sum(~numpy.isnan(measurement.errors)), len(measurement_ref.errors))


    def test_plots(self):

        model_state = ModelState(name=self.name, timepoints=self.timepoints, values=self.values)
        model_state.plot()
        pyplot.close()
        model_state.plot(title='testing_title')

        print(model_state)

        observation = Observation(name=self.name, timepoints=self.timepoints, values=self.values, observed_state=self.state, replicate_id='test_id')
        observation.plot()
        pyplot.close()

        measurement_01 = Measurement(name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors)
        measurement_01.plot()
        pyplot.close()
        measurement_01.plot(title='testing_title')
        pyplot.close()

        measurement_02 = Measurement(name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors, replicate_id='test_id')
        measurement_02.plot()
        pyplot.close()

        measurement_03 = Measurement(name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors, replicate_id='test_id', info='measurement_info')
        measurement_03.plot()
        pyplot.close()

        sensitivity_01 = Sensitivity(timepoints=self.timepoints, values=self.values, response=self.state, parameter=self.parameter)
        sensitivity_01.plot()
        pyplot.close()
        sensitivity_02 = Sensitivity(timepoints=self.timepoints, values=self.values, response=self.state, parameter=self.parameter, h=0.001)
        sensitivity_02.plot()
        pyplot.close()


    def test_measurement_error_models(self):

        measurement_01 = Measurement(
            name=self.name, timepoints=self.timepoints, values=self.values,
            error_model=self.linear_error_model, 
            error_model_parameters=self.linear_error_model_parameters,
        )

        measurement_02 = Measurement(
            name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors, 
            error_model=self.linear_error_model, 
            error_model_parameters=self.linear_error_model_parameters,
        )

        # After application error model parameters, the errors of both Measurement objects are different
        measurement_02.error_model_parameters = {'offset' : 10, 'slope' : 1}
        for _e01, _e02 in zip(measurement_01.errors, measurement_02.errors):
            self.assertNotEqual(_e01, _e02)

        # After application error model parameters, the errors of both Measurement objects are different
        measurement_02.error_model = self.constant_error_model
        for _e01, _e02 in zip(measurement_01.errors, measurement_02.errors):
            self.assertNotEqual(_e01, _e02)

        # Setting parameters that are unknown to the current error model fails 
        _measurement = Measurement(
            name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors, 
            error_model=self.linear_error_model, 
            error_model_parameters=self.linear_error_model_parameters,
        )
        with self.assertRaises(KeyError):
            _measurement.error_model_parameters = self.squared_error_model_parameters

        # Setting an error model that uses other parameter than the currently known will fails
        _measurement = Measurement(
            name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors, 
            error_model=self.linear_error_model, 
            error_model_parameters=self.linear_error_model_parameters,
        )
        with self.assertRaises(KeyError):
            _measurement.error_model = self.squared_error_model

        # A new error model with correspnoding parametrization can be set using a specific method
        measurement = Measurement(
                name=self.name, timepoints=self.timepoints, values=self.values, errors=self.errors, 
                error_model=self.linear_error_model, 
                error_model_parameters=self.linear_error_model_parameters,
            )
        measurement.update_error_model(
            error_model=self.squared_error_model, 
            error_model_parameters=self.squared_error_model_parameters,
        )
        measurement.apply_error_model(report_level=1)


    def test_measurement_error_models_nan_values(self):

        timepoints_w_nan = [1, 2, numpy.nan, 4, 5]
        values_w_nan = [100, 200, 300, numpy.nan, 500]
        errors_w_nan = [1/1, 1/2, 1/3, 1/4, numpy.nan]

        # instantiation should work
        measurement_01 = Measurement(
            name=self.name, timepoints=timepoints_w_nan, values=self.values,
            error_model=self.linear_error_model, 
            error_model_parameters=self.linear_error_model_parameters,
        )
        self.assertEqual(measurement_01.length, 4)

        measurement_02 = Measurement(
            name=self.name, timepoints=self.timepoints, values=values_w_nan,
            error_model=self.linear_error_model, 
            error_model_parameters=self.linear_error_model_parameters,
        )
        self.assertEqual(measurement_02.length, 4)

        measurement_03 = Measurement(
            name=self.name, timepoints=timepoints_w_nan, values=values_w_nan,
            error_model=self.linear_error_model, 
            error_model_parameters=self.linear_error_model_parameters,
        )
        self.assertEqual(measurement_03.length, 3)

        # updating the error model should preserve the error masks
        _mask_error_01 = measurement_01.mask_errors
        measurement_01.update_error_model(error_model=self.squared_error_model, error_model_parameters=self.squared_error_model_parameters)
        for _v1, _v2 in zip(_mask_error_01, measurement_01.mask_errors):
            self.assertEqual(_v1, _v2)

        _mask_error_02 = measurement_02.mask_errors
        measurement_02.update_error_model(error_model=self.squared_error_model, error_model_parameters=self.squared_error_model_parameters)
        for _v1, _v2 in zip(_mask_error_02, measurement_02.mask_errors):
            self.assertEqual(_v1, _v2)

        _mask_error_03 = measurement_03.mask_errors
        measurement_03.update_error_model(error_model=self.squared_error_model, error_model_parameters=self.squared_error_model_parameters)
        for _v1, _v2 in zip(_mask_error_03, measurement_03.mask_errors):
            self.assertEqual(_v1, _v2)

        # error can still be set explicitly
        measurement_01.errors = self.errors
        self.assertEqual(measurement_01.length, 4)

        measurement_01.errors = errors_w_nan
        self.assertEqual(measurement_01.length, 3)


class TextCovOptimality(unittest.TestCase):

    def test_cov_evaluation(self):

        Cov_01 = numpy.random.rand(3, 3) + 0.001
        Cov_02 = numpy.full(shape=(3, 3), fill_value=numpy.inf)
        Cov_03 = numpy.random.rand(3, 4) + 0.001
        criteria = ['A', 'D', 'E', 'E_mod']

        cov_evaluator = CovOptimality()

        for criterion in criteria:
           cov_evaluator.get_value(criterion, Cov_01)

        opt_value = cov_evaluator.get_value(criterion[0], Cov_02)
        self.assertTrue(opt_value, numpy.nan)

        with self.assertRaises(ValueError):
            cov_evaluator.get_value(criterion[0], Cov_03)

        with self.assertRaises(KeyError):
            cov_evaluator.get_value('bad_criterion', Cov_01)


class TestUtils(unittest.TestCase):

    maxDiff = None

    def test_OwnDict(self):

        own_dict = utils.OwnDict(
            {
                'a' : 1, 
                'b' : 2, 
                'c' : 3,
            }
        )

        expected = numpy.array([1, 2, 3])
        actual = own_dict.to_numpy()

        for _e, _a in zip(expected, actual):
            self.assertEqual(_e, _a)


    def test_bounds_to_floats(self):

        int_bounds = [(0, 10), (100, 1000), (10000, 100000)]
        float_bounds = utils.Helpers.bounds_to_floats(int_bounds)

        for _float_bounds in float_bounds:
            self.assertTrue(isinstance(_float_bounds[0], float) and isinstance(_float_bounds[1], float))


    def test_measurements_with_errors(self):
        timepoints = range(1, 11)
        measurements_01 = [
            Measurement(name='A', timepoints=timepoints, values=timepoints),
            Measurement(name='B', timepoints=timepoints, values=timepoints, errors=numpy.sqrt(timepoints)),
        ]
        measurements_02 = [
            Measurement(name='A', timepoints=timepoints, values=timepoints, errors=numpy.square(timepoints)),
            Measurement(name='B', timepoints=timepoints, values=timepoints, errors=numpy.sqrt(timepoints)),
        ]

        self.assertFalse(utils.Helpers.all_measurements_have_errors(measurements_01))
        self.assertTrue(utils.Helpers.all_measurements_have_errors(measurements_02))


    def test_has_unique_ids(self):

        ok_dict_01 = {'a01' : 1, 'b01' : 2}
        ok_dict_02 = {'b01' : 1, 'a01' : 2}
        ok_dict_03 = utils.OwnDict({'a01' : 1, 'b01' : 2})
        ok_list_01 = ['a01', 'b01']
        ok_list_02 = ['c01']

        bad_dict = {'a01' : 1, 'A01' : 2}
        bad_list_01 = ['a01', 'b01', 'b01']
        bad_list_02 = ['a01', 'B01', 'b01']

        bad_type = 'some_type'

        self.assertTrue(utils.Helpers.has_unique_ids(ok_dict_01))
        self.assertTrue(utils.Helpers.has_unique_ids(ok_dict_02))
        self.assertTrue(utils.Helpers.has_unique_ids(ok_dict_03))
        self.assertTrue(utils.Helpers.has_unique_ids(ok_list_01))
        self.assertTrue(utils.Helpers.has_unique_ids(ok_list_02))

        self.assertFalse(utils.Helpers.has_unique_ids(bad_dict, report=True))
        self.assertFalse(utils.Helpers.has_unique_ids(bad_list_01, report=True))
        self.assertFalse(utils.Helpers.has_unique_ids(bad_list_02, report=True))

        with self.assertRaises(TypeError):
            utils.Helpers.has_unique_ids(bad_type)


    def test_corr_matrix(self):

        Cov =  numpy.array([[504.0, 360.0, 180.0],
                            [360.0, 360.0, 0.0],
                            [180.0, 0.0, 720.0]])
        Bad_Cov = numpy.array([[504.0, 360.0, 180.0],
                               [180.0, 0.0, 720.0]])

        _ = utils.Calculations.cov_into_corr(Cov)
        _ = utils.Calculations.cov_into_corr(numpy.full(shape=Cov.shape, fill_value=numpy.inf))
        with self.assertRaises(ValueError):
            utils.Calculations.cov_into_corr(Bad_Cov)


    def test_get_unique_timepoints(self):

        _t = numpy.array([0, 1, 2, 3])
        _values = numpy.square(_t)

        single_01 = [
            Measurement(name='test', timepoints=_t, values=_values)
        ]
        single_02 = [
            Measurement(name='test', timepoints=_t, values=_values, replicate_id=SINGLE_ID), 
        ]
        t1 = utils.Helpers.get_unique_timepoints(single_01)
        t2 = utils.Helpers.get_unique_timepoints(single_02)
        for v1, v2 in zip(t1, t2):
            self.assertEqual(v1, v2)

        multi_01 = [
            Measurement(name='test', timepoints=_t, values=_values, replicate_id='1st'), 
            Measurement(name='test', timepoints=_t, values=_values, replicate_id='2nd'),
            Measurement(name='test', timepoints=_t, values=_values, replicate_id='3rd'),
        ]
        t3 = utils.Helpers.get_unique_timepoints(multi_01)
        for v1, v2 in zip(_t, t3):
            self.assertEqual(v1, v2)

        multi_02 = [
            Measurement(name='test1', timepoints=_t, values=_values, replicate_id='1st'), 
            Measurement(name='test2', timepoints=_t*10, values=_values, replicate_id='1st'), 
            Measurement(name='test', timepoints=_t*100, values=_values, replicate_id='2nd'),
            Measurement(name='test1', timepoints=_t, values=_values, replicate_id='3rd'), 
            Measurement(name='test2', timepoints=numpy.array([99, 999, 9999]), values=[1, 2, 3], replicate_id='3rd'),
        ]
        expected = numpy.unique(numpy.concatenate([_t, _t*10, _t*100, 99, 999, 9999], axis=None))
        t4 = utils.Helpers.get_unique_timepoints(multi_02)
        for v1, v2 in zip(_t, expected):
            self.assertEqual(v1, v2)


    def test_extract_time_series(self):
        timepoints = [1, 2, 3, 4, 5]
        values = numpy.square(numpy.array(timepoints))
        errors = numpy.sqrt(numpy.array(timepoints))

        time_series_objects_1 = [
            TimeSeries(name=_name, replicate_id=_rid, timepoints=timepoints, values=values)
            for _rid in ('1st', '2nd', '3rd')
            for _name in 'ABCD'
        ]

        time_series_objects_2 = [
            Measurement(name=_name, replicate_id=_rid, timepoints=timepoints, values=values, errors=errors)
            for _rid in ('1st', '2nd', '3rd')
            for _name in 'ABCD'
        ]

        time_series_objects_3 = [
            ModelState(name=_name, replicate_id=_rid, timepoints=timepoints, values=values)
            for _rid in ('1st', '2nd', '3rd')
            for _name in 'ABCD'
        ]

        time_series_objects_1_2 = time_series_objects_1 + time_series_objects_2
        time_series_objects_1_3 = time_series_objects_1 + time_series_objects_3

        _name = 'B'
        _replicate_id = '3rd'
        _extracted = utils.Helpers.extract_time_series(time_series_objects_1, name=_name, replicate_id=_replicate_id)
        self.assertEqual(_extracted.name, _name)
        self.assertEqual(_extracted.replicate_id, _replicate_id)

        _extracted = utils.Helpers.extract_time_series(time_series_objects_2, name=_name, replicate_id=_replicate_id)
        self.assertEqual(_extracted.name, _name)
        self.assertEqual(_extracted.replicate_id, _replicate_id)

        _extracted = utils.Helpers.extract_time_series(time_series_objects_3, name=_name, replicate_id=_replicate_id)
        self.assertEqual(_extracted.name, _name)
        self.assertEqual(_extracted.replicate_id, _replicate_id)

        with self.assertRaises(ValueError):
            utils.Helpers.extract_time_series(time_series_objects_1_2, name=_name, replicate_id=_replicate_id)

        with self.assertWarns(UserWarning):
            utils.Helpers.extract_time_series(time_series_objects_1_3, name=_name, replicate_id='unknown_id', no_extraction_warning=True)
        with self.assertWarns(UserWarning):
            utils.Helpers.extract_time_series(time_series_objects_1_3, name='unknown_name', replicate_id=_replicate_id, no_extraction_warning=True)


class TestGeneralizedIslands(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(1234)
        curr_model = 'model_03'
        self.mp = 'rate0'
        self.iv = 'y00'
        self.model_class, self.initial_values, self.model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        self.observation_functions_parameters = TestingHelpers.get_observation_functions_parameters(curr_model)
        self.obs_pars = self.observation_functions_parameters[0][1]
        self.replicate_ids = ['1st', '2nd', '3rd']
        self.caretaker = Caretaker(
            bioprocess_model_class=self.model_class, 
            model_parameters=self.model_parameters, 
            initial_values=self.initial_values, 
            observation_functions_parameters=self.observation_functions_parameters,
            replicate_ids=self.replicate_ids,
        )
        mappings = [
            ParameterMapper(replicate_id=['1st', '3rd'], 
                            global_name=self.mp, 
                            local_name=f'{self.mp}_1', 
                            value=self.model_parameters[self.mp]*1.2
                            ),
            ParameterMapper('3rd', 
                            self.iv, 
                            f'{self.iv}_1', 
                            self.initial_values[self.iv]*1.1
                            )
        ]
        self.caretaker.apply_mappings(mappings)
        simulation = self.caretaker.simulate(t=24)
        self.data = TestingHelpers.noisy_samples(simulation, samples=5)
        self.data_wo_errs = TestingHelpers.noisy_samples(simulation, samples=5, with_errors=False)


    def test_LossCalculator(self):
        unknowns = [self.mp, self.iv]
        bounds = [(0, 1000)]*len(unknowns)
        pg_problem_01 = LossCalculator(unknowns, bounds, 'negLL', self.data, self.caretaker.loss_function)
        pg_problem_01.fitness([0, 0])
        pg_problem_01.fitness([1000, 1000])
        pg_problem_01.gradient([0, 0])
        pg_problem_01.gradient([1000, 1000])

        pg_problem_02 = LossCalculator(unknowns, bounds, 'SS', self.data_wo_errs, self.caretaker.loss_function)
        pg_problem_02.fitness([0, 0])
        pg_problem_02.fitness([1000, 1000])
        pg_problem_02.gradient([0, 0])
        pg_problem_02.gradient([1000, 1000])

        pg_problem_03 = LossCalculator(unknowns, bounds, 'WSS', self.data_wo_errs, self.caretaker.loss_function)
        with self.assertRaises(AttributeError):
            pg_problem_03.fitness([0, 0])

        # An own LossCalculator with some constraints
        class OwnLossCalculator(LossCalculator):
            def constraint_1(self):
                p1 = self.current_parameters[unknowns[0]]
                return p1 < 0
            def constraint_2(self):
                p2 = self.current_parameters[unknowns[1]]
                return p2 > 0
            def constraint_3(self):
                p1 = self.current_parameters[unknowns[0]]
                p2 = self.current_parameters[unknowns[1]]
                return (p1 + p2) <= 1
            def check_constraints(self) -> List[bool]:
                return [self.constraint_1(), self.constraint_2(), self.constraint_3()]

        pg_constraint_problem = OwnLossCalculator(unknowns, bounds, 'negLL', self.data, self.caretaker.loss_function)

        # violates no constraints
        _loss = pg_constraint_problem.fitness([-1, 1])
        self.assertFalse(numpy.isinf(_loss[0]))
        # violates constraint_1
        _loss = pg_constraint_problem.fitness([0, 1])
        self.assertTrue(numpy.isinf(_loss[0]))
        # violates constraint_2
        _loss = pg_constraint_problem.fitness([-1, 0])
        self.assertTrue(numpy.isinf(_loss[0]))
        # violates constraint_1 and constraint_2
        _loss = pg_constraint_problem.fitness([0, 0])
        self.assertTrue(numpy.isinf(_loss[0]))
        # violates constraint_1 and constraint_3
        _loss = pg_constraint_problem.fitness([1, 1])
        self.assertTrue(numpy.isinf(_loss[0]))
        # violates constraint_3
        _loss = pg_constraint_problem.fitness([-1, 100])
        self.assertTrue(numpy.isinf(_loss[0]))

        # run estimations using the OwnLossCalculator



    def test_create_archipelago(self):

        curr_model = 'model_03'
        model_class, initial_values, model_parameters = TestingHelpers.get_model_building_blocks(curr_model)
        caretaker = Caretaker(
            bioprocess_model_class=self.model_class, 
            model_parameters=self.model_parameters, 
            initial_values=self.initial_values, 
        )
        simulation = caretaker.simulate(t=24)
        data = TestingHelpers.noisy_samples(simulation, samples=3)

        # Can give a list with different algorithms for archipelago creation
        optimizers = (
            ['ihs', 'de1220', 'pso', 'simulated_annealing'], 
            ['ihs'], 
            ['de1220'], 
            ['pso'], 
            ['simulated_annealing'],
        )
        
        optimizers_kwargs = (
            [{}]*4,
            [{}],
        )

        _unknowns = list(model_parameters.keys())
        _bounds = [
            (model_parameters[p]*0.1, model_parameters[p]*10) 
            for p in model_parameters
        ]

        problem = LossCalculator(
            unknowns=_unknowns, 
            bounds=_bounds, 
            metric='SS', 
            measurements=data, 
            caretaker_loss_fun=caretaker.loss_function, 
        )

        for _optimizers in optimizers:
            for _optimizers_kwargs in optimizers_kwargs:
                _archi = ArchipelagoHelpers.create_archipelago(
                    unknowns=_unknowns, 
                    optimizers=_optimizers, 
                    optimizers_kwargs=_optimizers_kwargs,
                    pg_problem=problem, 
                    rel_pop_size=5.0, 
                    archipelago_kwargs={},
                    report_level=1,
                    log_each_nth_gen=None,
                )
                time.sleep(0.2)
                _archi = None

        with self.assertRaises(ValueError):
            ArchipelagoHelpers.create_archipelago(
                    unknowns=_unknowns, 
                    optimizers=list(PygmoOptimizers.optimizers.keys()), 
                    optimizers_kwargs=[{}, {}],
                    pg_problem=problem, 
                    rel_pop_size=5.0, 
                    archipelago_kwargs={},
                    report_level=1,
                    log_each_nth_gen=None,
                )

        with self.assertRaises(ValueError):
            ArchipelagoHelpers.create_archipelago(
                    unknowns=_unknowns, 
                    optimizers='unknown_optimizer', 
                    optimizers_kwargs=[{}],
                    pg_problem=problem, 
                    rel_pop_size=5.0, 
                    archipelago_kwargs={},
                    report_level=1,
                    log_each_nth_gen=None,
                )

        # The problem class assigned to the corresponding property must subclass LossCalculator
        archi = ArchipelagoHelpers.create_archipelago(
                unknowns=_unknowns, 
                optimizers=['de1220'], 
                optimizers_kwargs=[{}],
                pg_problem=problem, 
                rel_pop_size=5.0, 
                archipelago_kwargs={},
                report_level=1,
                log_each_nth_gen=100,
            )

        archi.problem = LossCalculator

        class OwnLossCalculator(LossCalculator):
            pass
        archi.problem = OwnLossCalculator


    def test_parallel_estimation_result(self):

        est, est_info = self.caretaker.estimate_parallel(
            unknowns=[self.mp], 
            bounds=[(0, 100)], 
            measurements=self.data,
            optimizers_kwargs={'gen' : 5},
            rel_pop_size=10,
            )
        est_info.plot_loss_trail()
        est_info.plot_loss_trail(x_log=False)
        self.assertDictEqual(est, est_info.estimates)


    def test_check_evolution_stop(self):

        current_losses_01 = numpy.array([10, 20, 30])
        current_losses_02 = numpy.array([10.1, 10.2, 9.9])

        atols = [None, 1e-4, 1e-3, 1e-2, 1e-1]
        rtols = [None, 1e-4, 1e-3, 1e-2, 1e-1]
        current_runtime_mins = [None, 5, 10]
        max_runtime_mins = [None, 5, 10]
        current_evotime_mins = [None, 5, 10]
        max_evotime_mins = [None, 5, 10]
        max_memory_shares = [0, 0.1, 0.95]

        for _atol in atols:
            for _rtol in rtols:
                for _curr_runtime in current_runtime_mins:
                    for _max_runtime in max_runtime_mins:
                        for _curr_evotime in current_evotime_mins:
                            for _max_evotime in max_evotime_mins:
                                for _max_memory_share in max_memory_shares:
                                    ArchipelagoHelpers.check_evolution_stop(
                                        current_losses=current_losses_01, 
                                        atol_islands=_atol, 
                                        rtol_islands=_rtol, 
                                        current_runtime_min=_curr_runtime, 
                                        max_runtime_min=_max_runtime,
                                        current_evotime_min=_curr_evotime,
                                        max_evotime_min=_max_evotime,
                                        max_memory_share=_max_memory_share,
                                    )
                                    ArchipelagoHelpers.check_evolution_stop(
                                        current_losses=current_losses_02, 
                                        atol_islands=_atol, 
                                        rtol_islands=_rtol, 
                                        current_runtime_min=_curr_runtime, 
                                        max_runtime_min=_max_runtime,
                                        current_evotime_min=_curr_evotime,
                                        max_evotime_min=_max_evotime,
                                        max_memory_share=_max_memory_share,
                                    )


    def test_report_evolution_results(self):

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

        for _report_level in [0, 1, 2, 3, 4]:
            ArchipelagoHelpers.report_evolution_result(mock_evolution_results, _report_level)


class TestVisualizationHelpers(unittest.TestCase):

    def test_colors(self):
        ns = [10, 20, 21, 10.5, 10.6, 20.5, 20.6]
        for n in ns:
            VisualizationHelpers.get_n_colors(n)


class TestVisualization(unittest.TestCase):

    def setUp(self):
        t = [0, 1.5, 2, 5.5, 10]

        numpy.random.seed(1234)
        curr_model = 'model_02'
        self.caretaker_single = TestingHelpers.get_caretaker(curr_model)
        self.mp = 'k'
        self.iv = 'y0'
        self.sim_single = self.caretaker_single.simulate(t=24)
        self.data_single = TestingHelpers.noisy_samples(self.sim_single, samples=5)
        self.data_single_wo_errs = TestingHelpers.noisy_samples(self.sim_single, samples=5, with_errors=False)

        self.caretaker_multi = TestingHelpers.get_caretaker(curr_model, replicate_ids=['1st', '2nd', '3rd'])
        self.caretaker_multi_with_mappings = copy.deepcopy(self.caretaker_multi)
        mappings = [
            ParameterMapper(replicate_id=['1st', '3rd'], 
                            global_name=self.mp, 
                            local_name=f'{self.mp}_1', 
                            value=self.caretaker_multi_with_mappings._get_all_parameters()[self.mp]*1.2
                            ),
            ParameterMapper('3rd', 
                            self.iv,
                            f'{self.iv}_1', 
                            self.caretaker_multi_with_mappings._get_all_parameters()[self.iv]*1.1
                            )
        ]
        self.caretaker_multi_with_mappings.apply_mappings(mappings)
        self.sim_multi_with_mappings = self.caretaker_multi_with_mappings.simulate(t=24)
        self.data_multi_with_mappings = TestingHelpers.noisy_samples(self.sim_multi_with_mappings, samples=5)
        self.data_multi_with_mappings_wo_errs = TestingHelpers.noisy_samples(self.sim_multi_with_mappings, samples=5, with_errors=False)

        self.est_single, _ = self.caretaker_single.estimate(
            unknowns=[self.mp, self.iv], 
            bounds=[(0, 1), (90, 120)], 
            measurements = self.data_single,
        )
        self.est_rep_single, _ = self.caretaker_single.estimate_repeatedly(
            unknowns=[self.mp, self.iv], 
            bounds=[(0, 1), (90, 120)], 
            measurements = self.data_single,
            jobs=2,
        )

        self.est_multi, _ = self.caretaker_multi_with_mappings.estimate(
            unknowns=[f'{self.mp}_1', self.iv, f'{self.iv}_1'], 
            bounds=[(0, 1), (90, 120), (90, 120)], 
            measurements= self.data_multi_with_mappings,
        )
        self.est_rep_multi, _ = self.caretaker_multi_with_mappings.estimate_repeatedly(
            unknowns=[f'{self.mp}_1', self.iv, f'{self.iv}_1'], 
            bounds=[(0, 1), (90, 120), (90, 120)], 
            measurements= self.data_multi_with_mappings,
            jobs=2,
        )
    

    def test_result_plotting(self):
        Visualization.show_kinetic_data(self.sim_single)
        pyplot.close()
        Visualization.show_kinetic_data(self.data_single)
        pyplot.close()
        Visualization.show_kinetic_data(self.data_single, ncols=2)
        pyplot.close()
        Visualization.show_kinetic_data(self.sim_multi_with_mappings)
        pyplot.close()
        Visualization.show_kinetic_data(self.data_multi_with_mappings)
        pyplot.close()
        Visualization.show_kinetic_data(self.data_multi_with_mappings, ncols=2)
        pyplot.close()
        _bad_time_series = [TimeSeries(name='A', timepoints=[1, 2], values=[3, 4]), 'ABCD']
        with self.assertRaises(TypeError):
            Visualization.show_kinetic_data(_bad_time_series)
            pyplot.close()

        Visualization.show_parameter_distributions(self.est_rep_single)
        pyplot.close()
        Visualization.show_parameter_distributions(self.est_rep_multi)
        pyplot.close()
        Visualization.show_parameter_distributions(self.est_rep_multi, estimates=self.est_multi)
        pyplot.close()
        Visualization.show_parameter_distributions(self.est_rep_single, show_corr_coeffs=True)
        pyplot.close()
        Visualization.show_parameter_distributions(self.est_rep_multi, show_corr_coeffs=True)
        pyplot.close()
        Visualization.show_parameter_distributions(self.est_rep_multi, estimates=self.est_multi, show_corr_coeffs=True)
        pyplot.close()
        Visualization.compare_estimates(parameters=self.est_single, measurements=self.data_single, caretaker=self.caretaker_single)
        pyplot.close()
        Visualization.compare_estimates(parameters=self.est_multi, measurements=self.data_multi_with_mappings, caretaker=self.caretaker_multi_with_mappings, truth=self.sim_multi_with_mappings)
        pyplot.close()
        with self.assertRaises(TypeError):
            Visualization.compare_estimates(parameters=self.est_single, measurements=self.sim_single, caretaker=self.caretaker_single)
            pyplot.close()

        savenames = ['testplot.pdf']
        for savename in savenames:
            _savename = pathlib.Path(savename)
            Visualization.show_kinetic_data(self.sim_multi_with_mappings, savename=_savename)
            pyplot.close()
            Visualization.show_parameter_distributions(self.est_rep_multi, savename=_savename)
            pyplot.close()
            Visualization.compare_estimates(parameters=self.est_multi, measurements=self.data_multi_with_mappings, caretaker=self.caretaker_multi_with_mappings, truth=self.sim_multi_with_mappings, savename=_savename)
            pyplot.close()
        # remove generated files 
        current_directory = pathlib.Path('.')
        for current_file in current_directory.iterdir():
            if re.match(r'^.*testplot..*$', str(current_file)):
                pathlib.Path.unlink(current_file)

        # Test column numbers
        for i in range(1, 5):
            Visualization.show_kinetic_data(self.sim_multi_with_mappings, ncols=i)


if __name__ == '__main__':

    suite = unittest.TestSuite()

    # comment the test classes to exclude them from the test
    test_classes = (
        TestBioprocessModel,
        TestObservationFunction,
        TestModelObserver,
        TestSimulator,
        TestExtendedSimulator,
        TestParameterMapper,
        TestParameter,
        TestParameterManager,
        TestCaretaker,
        TestCaretakerEstimateMethods,
        TestDatatypes,
        TextCovOptimality,
        TestUtils,
        TestGeneralizedIslands,
        TestVisualizationHelpers,
        TestVisualization,
        TestModelChecker,
    )
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))

    runner = unittest.TextTestRunner()
    runner.run(suite)
